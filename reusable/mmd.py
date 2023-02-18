"""Implementations of Empirical MMD - e.g. see lemma 6 of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf"""


from functools import partial
import jax.numpy as jnp
import jax


@partial(jax.jit, static_argnames=["kernel_f"])
def mmd_mem_efficient(xs, ys, kernel_f):
    """Memory-efficient version, though very slow differentiation"""

    n, _ = xs.shape  # so n is the number of vectors, and d the dimension of each vector
    m, _ = ys.shape


    
    Kx_term = jax.lax.fori_loop(
        0, n, lambda i, acc: acc + jax.lax.fori_loop(0, n, lambda j, acc2: acc2 + kernel_f(xs[i], xs[j]), 0.0), 0.0
    ) - jax.lax.fori_loop(0, n, lambda i, acc: acc + kernel_f(xs[i], xs[i]), 0.0)

    Ky_term = jax.lax.fori_loop(
        0, m, lambda i, acc: acc + jax.lax.fori_loop(0, m, lambda j, acc2: acc2 + kernel_f(ys[i], ys[j]), 0.0), 0.0
    )  - jax.lax.fori_loop(0, m, lambda i, acc: acc + kernel_f(ys[i], ys[i]), 0.0)

    Kxy_term = jax.lax.fori_loop(
        0, n, lambda i, acc: acc + jax.lax.fori_loop(0, m, lambda j, acc2: acc2 + kernel_f(xs[i], ys[j]), 0.0), 0.0
    )

    return  Kx_term + Ky_term- 2 * Kxy_term


@partial(jax.jit, static_argnames=["kernel_f"])
def mmd_matrix_impl(xs, ys, kernel_f):
    """Matrix implementation: uses lots of memory, suitable for differentiation"""

    # Generate a kernel matrix by looping over each entry in x, y (both gm1, gm are functions!)
    gm1 = jax.vmap(lambda x, y: kernel_f(x, y), (0, None), 0)
    gm = jax.vmap(lambda x, y: gm1(x, y), (None, 0), 1)

    # step one - generate
    Kx = gm(xs, xs)
    Kx_term = jnp.sum(Kx) - jnp.sum(jnp.diagonal(Kx))
    del Kx
    Ky = gm(ys, ys)
    Ky_term = jnp.sum(Ky) - jnp.sum(jnp.diagonal(Ky))
    del Ky
    Kxy = gm(xs, ys)
    Kxy_term = jnp.sum(Kxy)
    del Kxy

    return Kx_term + Ky_term - 2* Kxy_term


# Liza's code, for comparison
# Note it actually sums several kernels.


# def MMD(x, y, kernel="multiscale"):
#     """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))

#     dxx = rx.t() + rx - 2. * xx # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz # Used for C in (1)

#     XX, YY, XY = (torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device))

#     if kernel == "multiscale":

#         bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#         for a in bandwidth_range:
#             XX += a**2 * (a**2 + dxx)**-1
#             YY += a**2 * (a**2 + dyy)**-1
#             XY += a**2 * (a**2 + dxy)**-1

#     if kernel == "rbf":

#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             XX += torch.exp(-0.5*dxx/a)
#             YY += torch.exp(-0.5*dyy/a)
#             XY += torch.exp(-0.5*dxy/a)

#     return torch.mean(XX + YY - 2. * XY)


