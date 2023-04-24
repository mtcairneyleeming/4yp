import jax.numpy as jnp
import jax

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


def esq_kernel(x, length, jitter=2e-5):
    """For GPs only!!! as it returns a matrix"""
    dist = sq_euclidian_dist(x, x)

    deltaXsq = dist / (length**2)
    k = jnp.exp(-0.5 * deltaXsq)
    k += jitter * jnp.eye(x.shape[0])
    return k

def rq_matrix_kernel(scale):
    """Scale is separated, because the current GP only has prior on kernel length"""
    
    def func (x, length, jitter=2e-5):
        """For GPs only!!! as it returns a matrix"""
        dist = sq_euclidian_dist(x, x)

        deltaXsq = dist / (2* scale * length**2)
        k = jnp.power(1 + deltaXsq, -scale)
        k += jitter * jnp.eye(x.shape[0])
        return k
    
    return func


def rbf_kernel(length):
    @jax.jit
    def func(x, z):
        diff = x - z
        dot = jnp.dot(diff, diff)
        return jnp.exp(-1 / length**2 * dot)

    return func


def rbf_kernel_multi(lengths):
    @jax.jit
    def func(x, z):
        
        diff = x - z
        dot = jnp.dot(diff, diff)
        sum = 0
        for length in lengths:
            sum += jnp.exp(-1 / length**2 * dot)
        return sum

    return func


def rq_kernel(length, scale):
    @jax.jit
    def func(x, z):
        return jnp.power(1 + jnp.dot(x - z, x - z) / (2 * scale * length**2), -scale)

    return func

def rq_kernel_multi(lengths, scales):
    @jax.jit
    def func(x, z):
        diff = x - z
        dot = jnp.dot(diff, diff)
        sum = 0
        for length, scale in zip(lengths, scales):
            sum += jnp.power(1 + dot / (2 * scale * length**2), -scale)
        return sum 

    return func

