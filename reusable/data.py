import jax.numpy as jnp
import jax.random as random

from numpyro.infer import Predictive
import numpyro
import numpyro.distributions as dist
import time

def __gen_batch(x, gp_model, gp_kernel, batch_size, rng_key, draw_access, jitter):
    pred = Predictive(gp_model, num_samples=batch_size)
    draws = pred(rng_key, x=x, gp_kernel=gp_kernel, jitter=jitter)[draw_access]


    return draws

def gen_one_batch(x, gp_model, gp_kernel, batch_size, rng_key, draw_access="y", jitter=1e-6):
    all_generated = 0 # total number of samples generated, incl NaNs
    num_successes = 0 # total number of non-NaN samples
    draws = None
    key = rng_key
    start = time.time()
    prev = start
    print(f"Starting, need {batch_size}")
    while all_generated <= 5 * batch_size:
        to_generate = min(500, batch_size - num_successes)
        
        key, new_key = random.split(key, 2) # otherwise will just generate the same data!
        new_draws = __gen_batch(x, gp_model, gp_kernel, to_generate, new_key, draw_access, jitter)

        nan_locs= jnp.any(jnp.isnan(new_draws), axis=-1) # so we get 1 
        all_generated += to_generate
        num_successes += to_generate -jnp.count_nonzero(nan_locs)
        
        filtered = new_draws[~nan_locs]
 
        if draws == None:
            draws = filtered
        else:
            draws = jnp.concatenate((draws, filtered), axis=0)

        curr = time.time()
        print(f"Looped: gen {to_generate}/{batch_size - num_successes} , elapsed: {curr-start}, last batch in {curr-prev}")
        prev = curr
        if num_successes == batch_size:
            print(f"Used {all_generated}, total time = {curr - start}")
            return draws
    raise Exception(f"failed to generate enough non-Nans {num_successes}/{batch_size}, attempts: {all_generated}")

def get_batches_generator(x, gp_model, gp_kernel, num_batches, batch_size, draw_access="y", jitter=1e-6):
    pred = Predictive(gp_model, num_samples=num_batches * batch_size)

    def func(rng_key):
        draws =  pred(rng_key, x=x, gp_kernel=gp_kernel, jitter=jitter)[draw_access]
        return jnp.reshape(draws, (num_batches, batch_size, -1))

    return func

def gen_all_gp_batches(x, gp_model, gp_kernel, num_epochs, num_batches, batch_size, rng_key, draw_access="y", jitter=1e-6):

    draws = gen_one_batch(x, gp_model, gp_kernel, num_epochs* num_batches * batch_size, rng_key, draw_access, jitter)

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_epochs, num_batches, batch_size, -1))  # note the last shape size will be args["n"]
    return draws

def gen_gp_batches(x, gp_model, gp_kernel, num_batches, batch_size, rng_key, draw_access="y", jitter=1e-6):

    draws = gen_one_batch(x, gp_model, gp_kernel, num_batches * batch_size, rng_key, draw_access, jitter)

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_batches, batch_size, -1))  # note the last shape size will be args["n"]
    return draws


def gen_latent_batches(latent_dim, num_batches, batch_size, rng_key):
    def model():
        z = numpyro.sample(
            "z", dist.MultivariateNormal(loc=jnp.zeros(latent_dim), covariance_matrix=jnp.identity(latent_dim))
        )
        return z

    pred = Predictive(model, num_samples=num_batches * batch_size)
    draws = pred(rng_key)["z"]

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_batches, batch_size, -1))  # note the last shape size will be args["n"]
    return draws


def pair_batched_data(xss, yss):
    """Given 2 sets of datapoints in batches (same num. of batches and same size of batches),
    concatenate the datapoints, preserving batch structure"""
    # 0 = batches, 1 = datapoints,
    return jnp.concatenate((xss, yss), axis=2)
