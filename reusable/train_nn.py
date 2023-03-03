from flax.training import train_state  # Useful dataclass to keep train state
import jax
import jax.numpy as jnp
from functools import partial

import time
import signal
from typing import Callable, Any, Dict


class SimpleTrainState(train_state.TrainState):
    key: jax.random.KeyArray


@partial(jax.jit, static_argnames=["loss_fn", "training"])
def compute_batch_loss(*, state: SimpleTrainState, batch, loss_fn, training=True):
    output = state.apply_fn({"params": state.params}, batch, training=training)
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
    compute_epoch_metrics: Callable[[SimpleTrainState, Any, Any], Dict],
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
    compute_epoch_metrics: Callable[[SimpleTrainState, Any, Any], Dict],
    num_epochs: int,
    train_draws: jax.Array,
    test_draws: jax.Array,
    initial_state: SimpleTrainState,
):
    """
    Same as run_training, except we shuffle the train_draws on each epoch
    """
    raise NotImplementedError()
    return run_training_datastream(
        loss_fn,
        compute_epoch_metrics,
        num_epochs,
        train_draws.shape[0],
        lambda i: train_draws,
        lambda i: test_draws[-1],
        initial_state,
    )



def run_training_infinite(
    loss_fn,
    compute_epoch_metrics: Callable[[SimpleTrainState, Any, Any], Dict],
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
    compute_epoch_metrics: Callable[[SimpleTrainState, Any, Any], Dict],
    num_epochs: int,
    num_train_batches: int,
    get_epoch_train_data: Callable[[int], jax.Array],
    get_epoch_test_data: Callable[[int], jax.Array],
    initial_state: SimpleTrainState
):
    """
    Run `num_epochs` training steps, starting from `initial_state`, using `loss_fn`. Perform `num_train_batches` batch steps in each epoch.

    get_epoch_train/test_data will be called with the index of the current batch

    `compute_epoch_metrics` is passed the current state, as well as the outputs of the final state at the end of a epoch, on the last training batch, and the test batch
    
    If it recieves a KeyboardInterrupt, it will end early, and return the previous complete epoch.
    """

 
    state = initial_state
    start = time.time()
    metrics_history = {
        "train_loss": jnp.zeros((num_epochs)),
        "test_loss": jnp.zeros((num_epochs)),
        "epoch_times": jnp.zeros((num_epochs)),
        "batch_times": jnp.zeros((num_epochs, num_train_batches))
    }
    prev_state = None
    try:
        for i in range(num_epochs):
            # note this is a different indexing scheme to the Flax tutorial

            batch_losses = []
            batch_times = []
            curr_training_data = get_epoch_train_data(i)
            for j in range(num_train_batches):
                # Run optimization steps over training batches and compute batch metrics
                
                    state = training_step(
                        state, curr_training_data[j], loss_fn=loss_fn
                    )  # get updated train state (which contains the updated parameters)
                    batch_losses.append(
                        compute_batch_loss(state=state, batch=curr_training_data[j], loss_fn=loss_fn, training=False)
                    )
                    batch_times.append(time.time() - start)
                

            prev_state = state # so that if we fail, we can return a consistent (state, training_data) pair
            test_state = state
            train_output = test_state.apply_fn({"params": test_state.params}, curr_training_data[-1], training=False)
            test_output = test_state.apply_fn({"params": test_state.params}, get_epoch_test_data(i), training=False)

            metrics = compute_epoch_metrics(test_state, train_output, test_output)

            metrics["train_loss"] = batch_losses[-1]
            metrics["train_avg_loss"] = jnp.mean(jnp.array(batch_losses))
            metrics["test_loss"] = loss_fn(*test_output)
            metrics["batch_times"] = jnp.array(batch_times)
            metrics["epoch_times"] = time.time() - start

            for metric, value in metrics.items():
                if i == 0 and not metric in metrics_history:
                    metrics_history[metric] = jnp.zeros((num_epochs))
                metrics_history[metric] = metrics_history[metric].at[i].set(value)

            if i % 5 == 0:
                print(f"epoch: {(i+1) }, {metrics}", flush=True)
        print(f"Done, in {time.time()-start}s ", flush=True)
        return state, metrics_history
    except KeyboardInterrupt as e:
                    metrics_history["interrupted"] = f"Interrupted at {i},{j}"
                    metrics_history["final_epoch"] = i-1
                    return prev_state, metrics_history
