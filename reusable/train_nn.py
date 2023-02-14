from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct
import jax
import jax.numpy as jnp
from functools import partial

from numpyro.infer import Predictive
import time
from typing import Callable, Any, Dict


def gen_gp_batches(x, gp_model, gp_kernel, num_batches, batch_size, rng_key, draw_access="y"):
    pred = Predictive(gp_model, num_samples=num_batches * batch_size)
    draws = pred(rng_key, x=x, gp_kernel=gp_kernel, jitter=1e-5)[draw_access]

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_batches, batch_size, -1))  # note the last shape size will be args["n"]
    return draws

def gen_labelled_gp_batches(x, gp_model, gp_kernel, num_batches, batch_size, rng_key, draw_access="y", label_access="c"):
    pred = Predictive(gp_model, num_samples=num_batches * batch_size)
    full_draws = pred(rng_key, x=x, gp_kernel=gp_kernel, jitter=1e-5)
    
    function_vals = full_draws[draw_access]
    labels = full_draws[label_access]
    labels = labels[:, None] # expands array size

    
    draws = jnp.concatenate((function_vals,labels), axis=-1)

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_batches, batch_size, -1))  # note the last shape size will be args["n"] + 1, for the labels
    return draws


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
    compute_epoch_metrics: Callable[[SimpleTrainState, jax.Array, jax.Array, Any, Any], Dict],
    num_epochs: int,
    train_draws: jax.Array,
    test_draws: jax.Array,
    initial_state: SimpleTrainState,
):
    """
        Run `num_epochs` training steps, starting from `initial_state`, using `loss_fn`. `train_draws` is batched, 
        whilst `test_draws` is a single batch for use for incremental metric updates.

        For `compute_epoch_metrics`, note that train_output will always be the output of the final state at the end of a epoch, on the last batch.
    """
    state = initial_state
    start = time.time()
    metrics_history = {
        "train_loss": jnp.zeros((num_epochs)),
        "test_loss": jnp.zeros((num_epochs)),
    }

    for i in range(num_epochs):
        # note this is a different indexing scheme to the Flax tutorial

        batch_losses = []

        for j in range(train_draws.shape[0]):
            # Run optimization steps over training batches and compute batch metrics
            state = training_step(
                state, train_draws[j], loss_fn=loss_fn
            )  # get updated train state (which contains the updated parameters)
            batch_losses.append(compute_batch_loss(state=state, batch=train_draws[j], loss_fn=loss_fn, training=False))

        test_state = state
        test_output = test_state.apply_fn({"params": test_state.params}, test_draws[-1], training=False)
        train_output = test_state.apply_fn({"params": test_state.params}, train_draws[-1], training=False)

        metrics = compute_epoch_metrics(test_state, test_draws, train_draws, train_output, test_output)

        metrics["train_loss"] = batch_losses[-1]
        metrics["train_avg_loss"] = jnp.mean(jnp.array(batch_losses))
        metrics["test_loss"] = compute_batch_loss(state=test_state, batch=test_draws[-1], loss_fn=loss_fn, training=False)

        for metric, value in metrics.items():
            if i == 0 and not metric in metrics_history:
                metrics_history[metric] = jnp.zeros((num_epochs))
            metrics_history[metric] = metrics_history[metric].at[i].set(value)

        if i % 5 == 0:
            print(f"epoch: {(i+1) }, {metrics}")
    print(f"Done, in {time.time()-start}s ")
    return state, metrics_history


# Kept for notebook 02, but not terribly useful in general


@struct.dataclass
class Metrics(metrics.Collection):

    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics
    key: jax.random.KeyArray
