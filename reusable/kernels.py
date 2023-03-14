import jax.numpy as jnp
import jax.debug 

def sq_euclidian_dist(x, z):
    """Euclidean distance between each point in x,z, suitable for JAX differentiation/etc. Liza's code"""
    if len(x.shape) == 1:
        x = x.reshape(x.shape[0], 1)
    if len(z.shape) == 1:
        z = z.reshape(x.shape[0], 1)  # fixed a bug.
    n_x, m = x.shape
    n_z, m_z = z.shape
    assert m == m_z
    delta = jnp.zeros((n_x, n_z))
    for d in jnp.arange(m):
        x_d = x[:, d]
        z_d = z[:, d]
        delta += (x_d[:, jnp.newaxis] - z_d) ** 2
    return delta


def esq_kernel(x,  var, length, jitter=2e-5):
    """For GPs only!!! as it returns a matrix"""
    dist = sq_euclidian_dist(x, x)

    deltaXsq = dist / (length**2)
    k = var * jnp.exp(-0.5 * deltaXsq)
    #jax.debug.print("{c}", c=-jnp.linalg.eigh(k)[0][..., 0])
    correction = 0 # the code below works, but runs very very slowly
    # # due to numerical instability, the smallest eigenvalue may sometimes be <0 - 
    # # thus we calculate the smallest eigenvalue, and if negative,
    # # we subtract it from the diagonal. 
    # correction = 2 * jnp.maximum(-jnp.linalg.eigh(k)[0][..., 0], 0)
    k += (jitter +correction) * jnp.eye(x.shape[0])
    return k


def rbf_kernel(x, z, length):
    return jnp.exp(-1 / length**2 * jnp.dot(x - z, x - z))


def rq_kernel(x, z, length, scale):
    return jnp.power(1 + jnp.dot(x - z, x - z) / (2 * scale * length**2), -scale)
