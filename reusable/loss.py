"""
Standard loss functions"""

import jax
import jax.numpy as jnp
import optax

@jax.jit
def RCL(y, reconstructed_y, mean, log_sd):
    """reconstruction loss, averaged over the datapoints (not summed)"""
    return jnp.mean(optax.l2_loss(reconstructed_y, y))  # 1/y.shape[0] *


@jax.jit
def KLD(y, reconstructed_y, mean, log_sd):
    """KL divergence between the distribution N(mean, log_sd) and a standard normal.
    e.g. see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions"""
    return -0.5 * jnp.mean(1 + log_sd - jnp.power(mean, 2) - jnp.exp(log_sd))

from reusable.mmd import mmd_matrix_impl
from reusable.kernels import rbf_kernel, rq_kernel

def MMD_rbf(ls):
    return MMD_rbf_sum([ls])

def MMD_rbf_sum(lss):

    @jax.jit
    def func(y, reconstructed_y, *args):
        return mmd_matrix_impl(y, reconstructed_y, lambda x, z: sum([rbf_kernel(x, z, ls) for ls in lss]) , normalise=True)
    
    func.__name__ =  f"mmd_rbf_sum:"  + ";".join([str(l) for l in lss])
    return func

def MMD_rqk(ls, scale):
    return MMD_rqk_sum([ls], [scale])

def MMD_rqk_sum(lss, scales):

    @jax.jit
    def func(y, reconstructed_y, *args):
        return mmd_matrix_impl(y, reconstructed_y, lambda x, z: sum([rq_kernel(x, z, ls, scale) for ls, scale in zip(lss, scales)]), normalise=True)
    
    func.__name__ =  f"mmd_rqk_sum:"  + ";".join([f"{str(l)},{str(s)}" for l,s in zip(lss, scales)])
    return func


@jax.jit
def MMD_rbf_ls_01_025_05_1_2_4_16_32(y, reconstructed_y):
    return mmd_matrix_impl(
        y,
        reconstructed_y,
        lambda x, z: rbf_kernel(x, z, 0.1)
        + rbf_kernel(x, z, 0.25)
        + rbf_kernel(x, z, 0.5)
        + rbf_kernel(x, z, 1.0)
        + rbf_kernel(x, z, 2.0)
        + rbf_kernel(x, z, 4.0)
        + rbf_kernel(x, z, 16.0)
        + rbf_kernel(x, z, 32.0),
    )


def combo_loss(f, g, f_scale=1, g_scale=1):
    @jax.jit 
    def func(*args):
        return f_scale *f(*args) + g_scale * g(*args)
    
    func.__name__ = f"{f_scale if f_scale != 1 else ''}{f.__name__}+{g_scale if g_scale != 1 else ''}{g.__name__}"

    return func

def combo3_loss(f, g, h, f_scale=1, g_scale=1, h_scale=1):
    @jax.jit 
    def func(*args):
        return f_scale *f(*args) + g_scale * g(*args) + h_scale * h(*args)
    
    func.__name__ = f"{f_scale if f_scale != 1 else ''}{f.__name__}+{g_scale if g_scale != 1 else ''}{g.__name__}+{h_scale if h_scale != 1 else ''}{h.__name__}"

    return func