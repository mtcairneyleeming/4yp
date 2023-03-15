""" VAE machinery for PriorVAE, implemented in Flax:
a combination of Liza's code and https://github.com/google/flax/blob/main/examples/vae/train.py 
with some of https://num.pyro.ai/en/0.7.1/examples/prodlda.html too

The VAE itself, plus suitable wrappings of it for use in SVI and MCMC
"""


import flax.linen as nn
import jax.numpy as jnp
import jax.random as random

from numpyro.contrib.module import flax_module
import numpyro
import numpyro.distributions as dist


class VAE_Encoder(nn.Module):

    hidden_dim1 :int
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




class VAE(nn.Module):
    ''' A complete JAX module for the VAE, for use in training
    '''
    hidden_dim1 :int
    hidden_dim2: int
    latent_dim: int
    out_dim: int
    conditional: bool


    @nn.compact
    def __call__(self, x, training=False):
        z_mu, z_sd = VAE_Encoder(hidden_dim1=self.hidden_dim1, hidden_dim2=self.hidden_dim2, latent_dim=self.latent_dim) (x) 
        '''During training random sample from the learned ZDIMS-dimensional
           normal distribution; during inference its mean.
        '''
        if training:
            rng_key = self.make_rng("train_latent_dist")
            std = jnp.exp(z_sd / 2)
            eps = random.normal(rng_key, std.shape)
            x_sample = jnp.add( jnp.multiply(eps, std), z_mu)
        else:
            x_sample =  z_mu

        if self.conditional:
            x_sample = jnp.concatenate((x_sample, x[:, -1:]), axis=-1) # x[-1] will be the label, c

        generated_x = VAE_Decoder(hidden_dim1=self.hidden_dim1, hidden_dim2=self.hidden_dim2, out_dim=self.out_dim)(x_sample) 
        return x, generated_x, z_mu,z_sd


class Single_Decoder(nn.Module):
    """
    Decoder set up for use on its own.
    The input parameter is made up of vectors x,z, where x is the datapoint to be reconstructed (which is not used at all by the model), and z an IID 
    standard normal multivariate Gaussian in the latent space.

    x is provided to make training easier (as we can just return x and z), and for use, just apply some dummy vector.
    the training 
    """
    
    hidden_dim1: int
    hidden_dim2: int
    out_dim: int


    @nn.compact
    def __call__(self, input, **_):
        x,z = jnp.split(input, [self.out_dim], axis=-1)
        z = nn.Dense(self.hidden_dim1, kernel_init=nn.initializers.normal(), name="DEC Hidden1")(z)
        z = nn.relu(z)
        z = nn.Dense(self.hidden_dim2, kernel_init=nn.initializers.normal(), name="DEC Hidden2")(z)
        z = nn.relu(z)
        z = nn.Dense(self.out_dim, kernel_init=nn.initializers.normal(), name="DEC Recons")(z)
        return x,z


# SVI model using the Decoder above
def vae_model(batch, hidden_dim1, hidden_dim2, latent_dim, vae_var):
    # not rewritten for conditional, as that would require sampling u/l or something
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)

    decoder = flax_module(
            "decoder",
            VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim, conditional=False),
            input_shape=(batch_dim, latent_dim)
        )

    with numpyro.plate("batch", batch_dim):
        #z = numpyro.sample("z", dist.Normal(jnp.zeros((latent_dim,)), jnp.ones((latent_dim,))).to_event(1))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([latent_dim]).to_event(1))
        gen_loc = decoder(z)

        return numpyro.sample("obs", dist.Normal(gen_loc, vae_var).to_event(1), obs=batch)      
        

# SVI guide using the encoder above
def vae_guide(batch, hidden_dim1, hidden_dim2, latent_dim, vae_var):
    # not rewritten for conditional, as that would require sampling u/l or something
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)

    encoder = flax_module(
        "encoder",
        VAE_Encoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim= latent_dim, conditional=False),
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

def cvae_sample(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params, condition=None):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))

    if condition is None:
        c = numpyro.sample("c", dist.Bernoulli(0.5)).reshape(1) 
    else:
        c = numpyro.deterministic("c", jnp.array([condition]))
    z_c = numpyro.deterministic("z_c", jnp.concatenate([z, c], axis=0))
    decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)  
    f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z_c))
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



def cvae_length_mcmc(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params, y=None, obs_idx=None, length=None):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
    if length is None:
        length = numpyro.sample("c", dist.Uniform(0.01, 0.5)).reshape(1) 
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



def decoder_sample(hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
    
    x_z = numpyro.deterministic("x_z", jnp.concatenate([jnp.ones(out_dim), z], axis=0))
    decoder_nn = Single_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)  
    f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, x_z)[1])
    return f
