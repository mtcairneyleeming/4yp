""" VAE machinery for PriorVAE, implemented in Flax:
a combination of Liza's code and https://github.com/google/flax/blob/main/examples/vae/train.py 
with some of https://num.pyro.ai/en/0.7.1/examples/prodlda.html too

The VAE itself, plus suitable wrappings of it for use in SVI and MCMC
"""


import flax.linen as nn
import jax.numpy as jnp

from numpyro.contrib.module import flax_module
import numpyro
import numpyro.distributions as dist


class VAE_Encoder(nn.Module):
    hidden_dim1: int
    hidden_dim2: int
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim1, kernel_init=nn.initializers.normal(), name="ENC Hidden1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim2, kernel_init=nn.initializers.normal(), name="ENC Hidden2")(x)
        x = nn.relu(x)
        mean_z = nn.Dense(self.latent_dim, kernel_init=nn.initializers.normal(), name="ENC Mean")(x)
        c = nn.Dense(self.latent_dim, kernel_init=nn.initializers.normal(), name="ENC Cov")(x)
        diag_cov = jnp.exp(c)
        return mean_z, diag_cov


class VAE_Decoder(nn.Module):
    """
    VAE De
    """
    hidden_dim1: int
    hidden_dim2: int
    out_dim: int


    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim1, kernel_init=nn.initializers.normal(), name="DEC Hidden1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim2, kernel_init=nn.initializers.normal(), name="DEC Hidden2")(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.normal(), name="DEC Recons")(x)
        return x


# SVI model using the Decoder above
def vae_model(batch, hidden_dim1, hidden_dim2, latent_dim, vae_var):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)

    decoder = flax_module(
            "decoder",
            VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim),
            input_shape=(batch_dim, latent_dim)
        )

    with numpyro.plate("batch", batch_dim):
        #z = numpyro.sample("z", dist.Normal(jnp.zeros((latent_dim,)), jnp.ones((latent_dim,))).to_event(1))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([latent_dim]).to_event(1))
        gen_loc = decoder(z)

        return numpyro.sample("obs", dist.Normal(gen_loc, vae_var).to_event(1), obs=batch)      
        

# SVI guide using the encoder above
def vae_guide(batch, hidden_dim1, hidden_dim2, latent_dim, vae_var):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)

    encoder = flax_module(
        "encoder",
        VAE_Encoder(hidden_dim1, hidden_dim2, latent_dim),
        input_shape=(batch_dim, out_dim),
    )
    z_loc, z_std = encoder(batch)
    with numpyro.plate("batch", batch_dim):
        return numpyro.sample("z", dist.Normal(z_loc, z_std).to_event(1))
        



def vae_sample(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
    decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)  
    f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z))
    return f


def vae_mcmc(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params, y=None, obs_idx=None):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
    decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)  

    f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z))
    sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

    if y is None: # durinig prediction
        numpyro.sample("y_pred", dist.Normal(f, sigma))
    else: # during inference
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)