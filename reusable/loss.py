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
    """KL divergence between the distribution N(mean, log_sd) and a standard normal.
    e.g. see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions"""
    return -0.5 * jnp.mean(1 + log_sd - jnp.power(mean, 2) - jnp.exp(log_sd))


from reusable.mmd import mmd_matrix_impl, mmd_mem_efficient
from reusable.kernels import rbf_kernel, rq_kernel


def MMD_rbf(ls, mem_efficient=False):
    return MMD_rbf_sum([ls], mem_efficient=mem_efficient)


def MMD_rbf_sum(lss, mem_efficient=False):
    if mem_efficient:

        @jax.jit
        def func(y, reconstructed_y, *args):
            return mmd_mem_efficient(
                y, reconstructed_y, lambda x, z: sum([rbf_kernel(x, z, ls) for ls in lss]), normalise=True
            )

    else:

        # @jax.jit
        def func(y, reconstructed_y, *args):
            return mmd_matrix_impl(
                y, reconstructed_y, lambda x, z: sum([rbf_kernel(x, z, ls) for ls in lss]), normalise=True
            )

    func.__name__ = f"mmd_rbf_sum-" + ";".join([str(l) for l in lss])
    return func


def MMD_rqk(ls, scale, mem_efficient=False):
    return MMD_rqk_sum([ls], [scale], mem_efficient=mem_efficient)


def MMD_rqk_sum(lss, scales, mem_efficient=False):

    if mem_efficient:

        @jax.jit
        def func(y, reconstructed_y, *args):
            return mmd_mem_efficient(
                y,
                reconstructed_y,
                lambda x, z: sum([rq_kernel(x, z, ls, scale) for ls, scale in zip(lss, scales)]),
                normalise=True,
            )

    else:

        @jax.jit
        def func(y, reconstructed_y, *args):
            return mmd_matrix_impl(
                y,
                reconstructed_y,
                lambda x, z: sum([rq_kernel(x, z, ls, scale) for ls, scale in zip(lss, scales)]),
                normalise=True,
            )

    func.__name__ = f"mmd_rqk_sum-" + ";".join([f"{str(l)},{str(s)}" for l, s in zip(lss, scales)])
    return func


@jax.jit
def MMD_rbf_ls_01_025_05_1_2_4_16_32(y, reconstructed_y, mem_efficient=False):
    return MMD_rbf_sum([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 32.0], mem_efficient=mem_efficient)


def combo_loss(f, g, f_scale=1, g_scale=1):
    @jax.jit
    def func(*args):
        return f_scale * f(*args) + g_scale * g(*args)

    func.__name__ = f"{f_scale if f_scale != 1 else ''}{f.__name__}+{g_scale if g_scale != 1 else ''}{g.__name__}"

    return func


def combo3_loss(f, g, h, f_scale=1, g_scale=1, h_scale=1):
    # @jax.jit
    def func(*args):
        return f_scale * f(*args) + g_scale * g(*args) + h_scale * h(*args)

    func.__name__ = f"{f_scale if f_scale != 1 else ''}{f.__name__}+{g_scale if g_scale != 1 else ''}{g.__name__}+{h_scale if h_scale != 1 else ''}{h.__name__}"

    return func


def conditional_loss_wrapper(f):
    """Remove the condition c from the original samples, so reconstrunction loss works"""

    @jax.jit
    def func(*args):
        new_args = (args[0][:, :-1], *args[1:])

        return f(*new_args)

    func.__name__ = f"c_wrapper:{f.__name__}"

    return func
