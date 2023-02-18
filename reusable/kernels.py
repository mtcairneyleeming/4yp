import jax.numpy as jnp


def euclidian_dist(x, z):
    """Euclidean distance between each point in x,z, suitable for JAX differentiation/etc. Liza's code"""
    x = jnp.array(x)
    z = jnp.array(z)
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
    return jnp.sqrt(delta)


def esq_kernel(x, z, var, length, noise=0, jitter=1.0e-6):
    """For GPs only!!! as it returns a matrix"""
    dist = euclidian_dist(x, z)
    deltaXsq = jnp.power(dist / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k


def rbf_kernel(x, z, length):
    return jnp.exp(-1 / length**2 * jnp.dot(x - z, x - z))


def rq_kernel(x, z, length, scale):
    return jnp.power(1 + jnp.dot(x - z, x - z) / (2 * scale * length**2), -scale)
