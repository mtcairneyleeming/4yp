from flax.training import train_state  # Useful dataclass to keep train state
import jax
import jax.numpy as jnp
import numpy as onp
from functools import partial

import time
import signal
from typing import Callable, Any, Dict, Optional


class SimpleTrainState(train_state.TrainState):
    key: jax.random.KeyArray


@partial(jax.jit, static_argnames=["loss_fn"])
def compute_batch_losses(state: SimpleTrainState, loss_fn, batch):
    output = state.apply_fn({"params": state.params}, batch, training=False)
    return loss_fn(*output)


@partial(jax.jit, static_argnames=["loss_fn"])
def training_step(state: SimpleTrainState, batch, loss_fn):
    # note train_key is fixed for all iterations, we vary it here:
    current_train_key = jax.random.fold_in(key=state.key, data=2 * state.step)

    def loss_fn_for_diff(params):
        """A function, only of the parameters, that provides the loss on this given batch"""

        # will call the module with the batch
        outputs = state.apply_fn(
            {"params": params}, batch, training=True, rngs={"train_latent_dist": current_train_key}
        )  # batch will contain just the function values
        loss = loss_fn(*outputs)

        return loss

    grad_fn = jax.grad(loss_fn_for_diff)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)  # increments state.step, so rng key fold_in does change
    return state


def run_training(
    loss_fn,
    compute_epoch_metrics: Optional[Callable[[SimpleTrainState, Any, Any], Dict]],
    num_epochs: int,
    train_draws: jax.Array,
    test_draws: jax.Array,
    initial_state: SimpleTrainState,
):
    """
    Run `num_epochs` training steps, starting from `initial_state`, using `loss_fn`.

    `train_draws` is batched, whilst `test_draws` is a single batch for use for incremental metric updates.

    `compute_epoch_metrics` is passed the current state, as well as the outputs of the final state at the end of a epoch, on the last training batch, and the test batch
    """
    return run_training_datastream(
        loss_fn,
        compute_epoch_metrics,
        num_epochs,
        train_draws.shape[0],
        lambda i: train_draws,
        lambda i: test_draws[-1],
        initial_state,
    )


def run_training_shuffle(
    loss_fn,
    compute_epoch_metrics: Optional[Callable[[SimpleTrainState, Any, Any], Dict]],
    num_epochs: int,
    train_draws: jax.Array,
    test_draws: jax.Array,
    initial_state: SimpleTrainState,
    shuffle_key: jax.random.KeyArray,
):
    """
    Same as run_training, except we shuffle the train_draws on each epoch
    """

    def shuffler(i: int):
        key = jax.random.fold_in(shuffle_key, i)
        shuffled = jax.random.permutation(key, train_draws, 0, independent=False)
        return shuffled

    return run_training_datastream(
        loss_fn,
        compute_epoch_metrics,
        num_epochs,
        train_draws.shape[0],
        shuffler,
        lambda i: test_draws[-1],
        initial_state,
    )


def run_training_infinite(
    loss_fn,
    compute_epoch_metrics: Optional[Callable[[SimpleTrainState, Any, Any], Dict]],
    all_train_draws: jax.Array,
    test_draws: jax.Array,
    initial_state: SimpleTrainState,
):
    """
    Same as run_training, except that fresh training data is used at each epoch
    """
    return run_training_datastream(
        loss_fn,
        compute_epoch_metrics,
        all_train_draws.shape[0],
        all_train_draws.shape[1],
        lambda i: all_train_draws[i],
        lambda i: test_draws[-1],
        initial_state,
    )


def run_training_datastream(
    loss_fn,
    compute_epoch_metrics: Optional[Callable[[SimpleTrainState, Any, Any], Dict]],
    num_epochs: int,
    num_train_batches: int,
    get_epoch_train_data: Callable[[int], jax.Array],
    get_epoch_test_data: Callable[[int], jax.Array],
    initial_state: SimpleTrainState,
):
    """
    Run `num_epochs` training steps, starting from `initial_state`, using `loss_fn`. Perform `num_train_batches` batch steps in each epoch.

    get_epoch_train/test_data will be called with the index of the current batch

    `compute_epoch_metrics` is passed the current state, as well as the outputs of the final state at the end of a epoch, on the last training batch, and the test batch

    If it recieves a KeyboardInterrupt, it will end early, and return the previous complete epoch.
    """
    loss_fn = jax.jit(loss_fn)

    state = initial_state
    start = time.time()
    metrics_history = {
        "train_loss": onp.zeros((num_epochs)),
        "test_loss": onp.zeros((num_epochs)),
        "epoch_times": onp.zeros((num_epochs)),
    }
    prev_state = None
    try:
        for i in range(num_epochs):
            # note this is a different indexing scheme to the Flax tutorial

            curr_training_data = get_epoch_train_data(i)
            for j in range(num_train_batches):
                # Run optimization steps over training batches and compute batch metrics

                state = training_step(
                    state, curr_training_data[j], loss_fn=loss_fn
                )  # get updated train state (which contains the updated parameters)

            prev_state = state  # so that if we fail, we can return a consistent (state, training_data) pair
            test_state = state
            

            metrics_history["train_loss"][i] = compute_batch_losses(test_state, loss_fn, curr_training_data[-1])
            metrics_history["test_loss"][i] = compute_batch_losses(test_state, loss_fn, get_epoch_test_data(i))
            metrics_history["epoch_times"][i] = time.time() - start

            if compute_epoch_metrics is not None:
                train_output = test_state.apply_fn(
                    {"params": test_state.params}, curr_training_data[-1], training=False
                )
                test_output = test_state.apply_fn({"params": test_state.params}, get_epoch_test_data(i), training=False)

                metrics = compute_epoch_metrics(test_state, train_output, test_output)

                for metric, value in metrics.items():
                    if i == 0 and not metric in metrics_history:
                        metrics_history[metric] = onp.zeros((num_epochs))
                    metrics_history[metric][i] = value

                del train_output, test_output

            if i % 5 == 0:
                print(
                    f"epoch: {(i) }, test_loss: {metrics_history['test_loss'][i]}, train_loss: {metrics_history['train_loss'][i]}",
                    flush=True,
                )
        print(f"Done, in {time.time()-start}s ", flush=True)
        return state, metrics_history
    except KeyboardInterrupt as e:
        metrics_history["interrupted"] = f"Interrupted at {i},{j}"
        metrics_history["final_epoch"] = i - 1
        return prev_state, metrics_history
