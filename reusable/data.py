import jax.numpy as jnp


from numpyro.infer import Predictive
import numpyro
import numpyro.distributions as dist


def gen_one_batch(x, gp_model, gp_kernel, batch_size, rng_key, draw_access="y"):
    pred = Predictive(gp_model, num_samples=batch_size)
    draws = pred(rng_key, x=x, gp_kernel=gp_kernel, jitter=1e-5)[draw_access]

    return draws

def get_batches_generator(x, gp_model, gp_kernel, num_batches, batch_size, draw_access="y"):
    pred = Predictive(gp_model, num_samples=num_batches * batch_size)

    def func(rng_key):
        draws =  pred(rng_key, x=x, gp_kernel=gp_kernel, jitter=1e-5)[draw_access]
        return jnp.reshape(draws, (num_batches, batch_size, -1))

    return func

def gen_all_gp_batches(x, gp_model, gp_kernel, num_epochs, num_batches, batch_size, rng_key, draw_access="y"):

    draws = gen_one_batch(x, gp_model, gp_kernel, num_epochs* num_batches * batch_size, rng_key, draw_access)

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_epochs, num_batches, batch_size, -1))  # note the last shape size will be args["n"]
    return draws

def gen_gp_batches(x, gp_model, gp_kernel, num_batches, batch_size, rng_key, draw_access="y"):

    draws = gen_one_batch(x, gp_model, gp_kernel, num_batches * batch_size, rng_key, draw_access)

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
