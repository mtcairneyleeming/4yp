import jax.numpy as jnp



import numpyro
import numpyro.distributions as dist


def OneDGP(gp_kernel, x, jitter=2e-5, var=None, length=None, y=None, noise=False):
    """The original, basic GP, with the length sampled from a fixed prior"""
    if length==None:
        length = numpyro.sample("kernel_length", dist.InverseGamma(4,1))
        
    if var==None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.,0.1))
        
    k = gp_kernel(x, var, length, jitter)
    
    if noise==False:
        numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    return y



def OneDGP_BinaryCond(gp_kernel, x, jitter=1e-6, var=None, length=None, y=None, noise=False, u=None):
    """A GP, with a binary condition on the lengthscale, built from the standard GP above."""
    if u==None:
        u = numpyro.sample("u", dist.Bernoulli(0.5)).reshape(1) 
    else:
       u = numpyro.deterministic("u", jnp.array([u]))

    if length==None:  
        length = 0.1* u + 0.4 * (1-u)
            
    if var == None:
        var = 1.0

        
    k = gp_kernel(x, var, length, jitter)
    
    if noise==False:
        y = numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        y = numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    
    y_c = numpyro.deterministic("y_c", jnp.concatenate([y, u], axis=0))


def OneDGP_UnifLS(gp_kernel, x, jitter=2e-5, var=None, length=None, y=None, noise=False):
    """The original, basic GP, with the length sampled from a fixed prior"""
    if length==None:
        length = numpyro.sample("kernel_length", dist.Uniform(0.01, 0.5))
        
    if var==None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.,0.1))
        
    k = gp_kernel(x, var, length, jitter)

    length = jnp.array(length).reshape(1) 
    
    if noise==False:
        y= numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[:-1]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[:-1]), covariance_matrix=k))
        y= numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    y_c = numpyro.deterministic("y_c", jnp.concatenate([y, length], axis=0))
    return y
