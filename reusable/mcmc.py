
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from .vae import VAE_Decoder


def vae_mcmc(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params, y=None, obs_idx=None):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
    decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)  

    f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z))
    sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

    if y is None: # durinig prediction
        numpyro.sample("y_pred", dist.Normal(f, sigma))
    else: # during inference
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)

def cvae_mcmc(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params, y=None, obs_idx=None, c=None, binary_prior=True):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
    if c is None:
        if binary_prior:
            c = numpyro.sample("c", dist.Bernoulli(0.5)).reshape(1) 
        else:
            c = numpyro.sample("c", dist.Beta(1e-4,1e-4)).reshape(1) 
    else:
        c = numpyro.deterministic("c", jnp.array([c]))
    z_c = numpyro.deterministic("z_c", jnp.concatenate([z, c], axis=0))

    decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)  

    f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z_c))
    sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

    if y is None: # durinig prediction
        numpyro.sample("y_pred", dist.Normal(f, sigma))
    else: # during inference
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)



def cvae_length_mcmc(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params,  obs_idx=None, length_prior_choice = "uniform"):

    if length_prior_choice == "uniform":
        prior = dist.Uniform(0.01, 0.5)
    elif length_prior_choice == "invgamma":
        prior = dist.InverseGamma(4,1)

    def func(y=None,  length=None, var=None,):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
        if length is None:
            length = numpyro.sample("c", prior).reshape(1) 
        else:
            length = numpyro.deterministic("c", jnp.array([length]))

        z_c = numpyro.deterministic("z_c", jnp.concatenate([z, length], axis=0))

        decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)  

        f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z_c))
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

        if y is None: # durinig prediction
            numpyro.sample("y_pred", dist.Normal(f, sigma))
        else: # during inference
            numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)
    return func



def gp_length_mcmc(all_x, gp_kernel,obs_idx=None, jitter=1e-5, length_prior_choice = "uniform"):

    if length_prior_choice == "uniform":
        prior = dist.Uniform(0.01, 0.5)
    elif length_prior_choice == "invgamma":
        prior = dist.InverseGamma(4,1)

    def func(y=None,  length=None, var=None,):

        if length==None:
            length = numpyro.sample("kernel_length", prior)
            
        if var==None:
            var = numpyro.sample("kernel_var", dist.LogNormal(0.,0.1))
            
        k = gp_kernel(all_x, var, length, jitter)

        length = jnp.array(length).reshape(1) 

        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(all_x.shape[:-1]), covariance_matrix=k))
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

        if y is None: # durinig prediction
            numpyro.sample("y_pred", dist.Normal(f, sigma))
        else: # during inference
            numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)
    return func
    