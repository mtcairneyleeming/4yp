

import jax.numpy as jnp


from numpyro.infer import Predictive
import numpyro
import numpyro.distributions as dist



def gen_gp_batches(x, gp_model, gp_kernel, num_batches, batch_size, rng_key, draw_access="y"):
    pred = Predictive(gp_model, num_samples=num_batches * batch_size)
    draws = pred(rng_key, x=x, gp_kernel=gp_kernel, jitter=1e-5)[draw_access]

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_batches, batch_size, -1))  # note the last shape size will be args["n"]
    return draws

def gen_latent_batches(latent_dim, num_batches, batch_size, rng_key):

    def model():
        z = numpyro.sample("z", dist.MultivariateNormal(loc=jnp.zeros(latent_dim), covariance_matrix=jnp.identity(latent_dim)))
        return z


    pred = Predictive(model, num_samples=num_batches * batch_size)
    draws = pred(rng_key)["z"]

    # batch these [note above ensures there are the right no. of elements], so the reshape works perfectly
    draws = jnp.reshape(draws, (num_batches, batch_size, -1))  # note the last shape size will be args["n"]
    return draws
