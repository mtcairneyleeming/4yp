import jax.numpy as jnp
import jax.scipy

import numpyro
import numpyro.distributions as dist


def OneDGP(gp_kernel, x, jitter=2e-5, var=None, length=None, y=None, noise=False):
    """The original, basic GP, with the length sampled from a fixed prior"""
    if length == None:
        length = numpyro.sample("kernel_length", dist.InverseGamma(4, 1))

    if var == None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 0.1))

    k = gp_kernel(x, length, jitter)

    length = jnp.array(length).reshape(1)

    if noise == False:
        y = numpyro.sample(
            "y",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
            obs=y,
        )
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample(
            "f",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
        )
        y = numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    y_c = numpyro.deterministic("y_c", jnp.concatenate([y, length], axis=0))

    return y


def setup_prior(prior_choice, prior_args):
    if prior_choice == "invgamma":
        conc = prior_args.get("concentration", 4)
        rate = prior_args.get("rate", 1)
        return dist.InverseGamma(conc, rate)

    elif prior_choice == "lognormal":
        loc = prior_args.get("location", 0.0)
        scale = prior_args.get("scale", 0.1)
        return dist.LogNormal(loc, scale)

    elif prior_choice == "normal":
        loc = prior_args.get("location", 0.0)
        scale = prior_args.get("scale", 15.0)
        return dist.Normal(loc, scale)

    elif prior_choice == "halfnormal":
        scale = prior_args.get("scale", 15.0)
        return dist.HalfNormal(scale)

    elif prior_choice == "gamma":
        conc = prior_args.get("concentration", 4)
        rate = prior_args.get("rate", 1)
        return dist.Gamma(conc, rate)

    elif prior_choice == "uniform":
        lower = prior_args.get("lower", 0.01)
        upper = prior_args.get("upper", 0.5)
        return dist.Uniform(lower, upper)

    raise NotImplementedError(f"Unknown prior choice {prior_choice}")


def BuildGP(
    gp_kernel,
    jitter=2e-5,
    obs_idx=None,
    noise=False,
    length_prior_choice="invgamma",
    length_prior_args={},
    variance_prior_choice="lognormal",
    variance_prior_args={},
):
    length_prior = setup_prior(length_prior_choice, length_prior_args)
    variance_prior = setup_prior(variance_prior_choice, variance_prior_args)

    print("Mean", variance_prior.mean, "Variance", variance_prior.variance)

    def func(x, var=None, length=None, y=None, **kwargs):
        """The original, basic GP, with the length sampled from a fixed prior"""
        if length == None:
            length = numpyro.sample("kernel_length", length_prior)

        if var == None:
            var = numpyro.sample("kernel_var", variance_prior)

        k = gp_kernel(x, length, jitter)

        length = jnp.array(length).reshape(1)

        if noise == False:
            y = numpyro.sample(
                "y",
                dist.TransformedDistribution(
                    dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                    dist.transforms.AffineTransform(0, var),
                ),
                obs=y,
            )
        else:
            pass
            f = numpyro.sample(
                "f",
                dist.TransformedDistribution(
                    dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                    dist.transforms.AffineTransform(0, var),
                ),
            )
            sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

            if obs_idx is not None:  # in this case y will be lower dimension, and observed - unsurprisingly!
                y = numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)
            else:
                y = numpyro.sample("y", dist.Normal(f, sigma), obs=y)

        y_c = numpyro.deterministic("y_c", jnp.concatenate([y, length], axis=0))

        return y

    return func


def OneDGP_BinaryCond(gp_kernel, x, jitter=1e-6, var=None, length=None, y=None, noise=False, u=None):
    """A GP, with a binary condition on the lengthscale, built from the standard GP above."""
    if u == None:
        u = numpyro.sample("u", dist.Bernoulli(0.5)).reshape(1)
    else:
        u = numpyro.deterministic("u", jnp.array([u]))

    if length == None:
        length = 0.1 * u + 0.4 * (1 - u)

    if var == None:
        var = 1.0

    k = gp_kernel(x, length, jitter)

    if noise == False:
        y = numpyro.sample(
            "y",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
            obs=y,
        )
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample(
            "f",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
        )
        y = numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    y_c = numpyro.deterministic("y_c", jnp.concatenate([y, u], axis=0))


def OneDGP_UnifLS(gp_kernel, x, jitter=2e-5, var=None, length=None, y=None, noise=False):
    """The original, basic GP, with the length sampled from a fixed prior"""
    if length == None:
        length = numpyro.sample("kernel_length", dist.Uniform(0.01, 0.5))

    if var == None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 0.1))

    k = gp_kernel(x, var, length, jitter)

    length = jnp.array(length).reshape(1)

    if noise == False:
        y = numpyro.sample(
            "y",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[:-1]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
            obs=y,
        )
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample(
            "f",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[:-1]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
        )
        y = numpyro.sample("y", dist.Normal(f, sigma), obs=y)

    y_c = numpyro.deterministic("y_c", jnp.concatenate([y, length], axis=0))
    return y


def BuildGP_Binomial(
    N,
    gp_kernel,
    jitter=2e-5,
    obs_idx=None,
    noise=False,
    length_prior_choice="invgamma",
    length_prior_args={},
    variance_prior_choice="lognormal",
    variance_prior_args={},
):
    length_prior = setup_prior(length_prior_choice, length_prior_args)
    variance_prior = setup_prior(variance_prior_choice, variance_prior_args)
    # -1.3418452 0.21973312
    def func(x, var=None, length=None, y=None, **kwargs):
        """The original, basic GP, with the length sampled from a fixed prior"""
        if length == None:
            length = numpyro.sample("kernel_length", length_prior)

        if var == None:
            var = numpyro.sample("kernel_var", variance_prior)

        k = gp_kernel(x, length, jitter)

        length = jnp.array(length).reshape(1)

        # the gp
        f = numpyro.sample(
            "f",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
        )

        probs = numpyro.deterministic("p", jnp.exp(f) / (1 + jnp.exp(f)))

        if noise == False:
            y = numpyro.sample("y", dist.Binomial(N, probs=probs), obs=y)
        else:
            sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
            g = numpyro.sample("g", dist.Binomial(N, probs=probs))

            if obs_idx is not None:  # in this case y will be lower dimension, and observed - unsurprisingly!
                y = numpyro.sample("y", dist.Normal(g[obs_idx], sigma), obs=y)
            else:
                y = numpyro.sample("y", dist.Normal(g, sigma), obs=y)

        # y_c = numpyro.deterministic("y_c", jnp.concatenate([y, length], axis=0))

        return y

    return func


def BuildGP_Weibull(
    scale, gp_kernel, jitter=2e-5, obs_idx=None, noise=False, length_prior_choice="invgamma", prior_args={}
):
    prior = setup_prior(length_prior_choice, prior_args)

    # -1.3418452 0.21973312
    def func(x, var=None, length=None, y=None, **kwargs):
        """The original, basic GP, with the length sampled from a fixed prior"""
        if length == None:
            length = numpyro.sample("kernel_length", prior)

        if var == None:
            var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 0.1))

        k = gp_kernel(x, length, jitter)

        length = jnp.array(length).reshape(1)

        # the gp
        f = numpyro.sample(
            "f",
            dist.TransformedDistribution(
                dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k),
                dist.transforms.AffineTransform(0, var),
            ),
        )

        link = jax.scipy.special.expit(f)

        if noise == False:
            y = numpyro.sample("y", dist.Weibull(jnp.repeat(scale, f.shape[0]), link), obs=y)
        else:
            sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
            g = numpyro.sample("g", dist.Weibull(1, link))

            if obs_idx is not None:  # in this case y will be lower dimension, and observed - unsurprisingly!
                y = numpyro.sample("y", dist.Normal(g[obs_idx], sigma), obs=y)
            else:
                y = numpyro.sample("y", dist.Normal(g, sigma), obs=y)

        # y_c = numpyro.deterministic("y_c", jnp.concatenate([y, length], axis=0))

        return y

    return func
