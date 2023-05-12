"""
Standard loss functions"""

import jax
import jax.numpy as jnp
import optax


@jax.jit
def RCL(y, reconstructed_y, *args):
    """reconstruction loss, averaged over the datapoints (not summed)"""
    return jnp.mean(optax.l2_loss(reconstructed_y, y))  # 1/y.shape[0] *


@jax.jit
def KLD(y, reconstructed_y, mean, log_sd):
    """KL divergence between the distribution N(mean, exp(log_sd)) and a standard normal.
    e.g. see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions"""
    return -0.5 * jnp.mean(1 + log_sd - jnp.power(mean, 2) - jnp.exp(log_sd))


from reusable.mmd import mmd_matrix_impl, mmd_mem_efficient
from reusable.kernels import rbf_kernel, rq_kernel, rbf_kernel_multi, rq_kernel_multi


def MMD_rbf(ls, mem_efficient=False):
    k = rbf_kernel(ls)
    func = mmd_mem_efficient(k) if mem_efficient else mmd_matrix_impl(k)

    func.__name__ = f"mmd_rbf_sum-{ls}"
    return func


def MMD_rbf_sum(lss, mem_efficient=False):
    k = rbf_kernel_multi(lss)
    func = mmd_mem_efficient(k) if mem_efficient else mmd_matrix_impl(k)

    func.__name__ = f"mmd_rbf_sum-" + ";".join([f"{str(l)}" for l in lss])
    return func


def MMD_rqk(ls, scale, mem_efficient=False):
    k = rq_kernel(ls, scale)
    func = mmd_mem_efficient(k) if mem_efficient else mmd_matrix_impl(k)

    func.__name__ = f"mmd_rqk_sum-{ls},{scale}"
    return func


def MMD_rqk_sum(lss, scales, mem_efficient=False):
    k = rq_kernel_multi(lss, scales)
    func = mmd_mem_efficient(k) if mem_efficient else mmd_matrix_impl(k)

    func.__name__ = f"mmd_rqk_sum-" + ";".join([f"{str(l)},{str(s)}" for l, s in zip(lss, scales)])
    return func


def combo_loss(f, g, f_scale=1, g_scale=1):
    @jax.jit
    def func(*args):
        return f_scale * f(*args) + g_scale * g(*args)

    func.__name__ = f"{f_scale if f_scale != 1 else ''}{f.__name__}+{g_scale if g_scale != 1 else ''}{g.__name__}"

    return func


def combo3_loss(f, g, h, f_scale=1, g_scale=1, h_scale=1):
    @jax.jit
    def func(*args):
        return f_scale * f(*args) + g_scale * g(*args) + h_scale * h(*args)

    func.__name__ = f"{f_scale if f_scale != 1 else ''}{f.__name__}+{g_scale if g_scale != 1 else ''}{g.__name__}+{h_scale if h_scale != 1 else ''}{h.__name__}"

    return func


def combo_multi_loss(fs, scales):
    @jax.jit
    def func(*args):
        s = 0
        for (f, scale) in zip(fs, scales):
            s += scale * f(*args)
        return s

    name = ""
    for (f, scale) in zip(fs, scales):
        if name != "":
            name += "+"
        name += f"{scale if scale != 1 else ''}{f.__name__}"

    func.__name__ = name

    return func


def conditional_loss_wrapper(f):
    """Remove the condition c from the original samples, so reconstrunction loss works"""

    @jax.jit
    def func(*args):
        new_args = (args[0][:, :-1], *args[1:])

        return f(*new_args)

    func.__name__ = f"c_wrapper:{f.__name__}"

    return func
