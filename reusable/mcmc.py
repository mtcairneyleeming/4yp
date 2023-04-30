import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from .vae import VAE_Decoder
from .gp import setup_prior
from numpyro.infer import NUTS, init_to_median, MCMC, HMC, MixedHMC
import time


def vae_mcmc(
    hidden_dim1,
    hidden_dim2,
    latent_dim,
    decoder_params,
    obs_idx=None,
    noise=False,
):
    def func(x, var=None, y=None, **kwargs):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))

        decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=x.shape[0])

        f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z))
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        if y is None:  # during prediction
            numpyro.sample("y_pred", dist.Normal(f, sigma))
        else:  # during inference
            numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)

    return func


def cvae_mcmc(
    hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params, y=None, obs_idx=None, c=None, binary_prior=True
):
    z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
    if c is None:
        if binary_prior:
            c = numpyro.sample("c", dist.Bernoulli(0.5)).reshape(1)
        else:
            c = numpyro.sample("c", dist.Beta(1e-4, 1e-4)).reshape(1)
    else:
        c = numpyro.deterministic("c", jnp.array([c]))
    z_c = numpyro.deterministic("z_c", jnp.concatenate([z, c], axis=0))

    decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=out_dim)

    f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z_c))
    sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

    if y is None:  # durinig prediction
        numpyro.sample("y_pred", dist.Normal(f, sigma))
    else:  # during inference
        numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)


def cvae_length_mcmc(
    hidden_dim1,
    hidden_dim2,
    latent_dim,
    decoder_params,
    obs_idx=None,
    noise=False,
    length_prior_choice="uniform",
    prior_args={},
):

    prior = setup_prior(length_prior_choice, prior_args)

    def func(x, var=None, length=None, y=None, **kwargs):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim)))
        if length is None:
            length = numpyro.sample("c", prior).reshape(1)
        else:
            length = numpyro.deterministic("c", jnp.array([length]))

        z_c = numpyro.deterministic("z_c", jnp.concatenate([z, length], axis=0))
        decoder_nn = VAE_Decoder(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, out_dim=x.shape[0])

        f = numpyro.deterministic("f", decoder_nn.apply(decoder_params, z_c))
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))

        if y is None:  # durinig prediction
            numpyro.sample("y", dist.Normal(f, sigma))
        else:  # during inference
            numpyro.sample("y", dist.Normal(f[obs_idx], sigma), obs=y)

    return func


def run_mcmc(
    num_warmup,
    num_samples,
    num_chains,
    rng_key,
    model_mcmc,
    mcmc_arguments,
    verbose=False,
    thinning=1,
    group_by_chain=False,
    max_run_length=200,
    increment_save_fun=None,
    use_mixed_hmc=False,
):
    start = time.time()

    init_strategy = init_to_median(num_samples=10)
    if use_mixed_hmc:
        kernel = MixedHMC(HMC(model_mcmc, init_strategy=init_strategy))
    else:
        kernel = NUTS(model_mcmc, init_strategy=init_strategy)

    if max_run_length == None:
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            thinning=thinning,
            jit_model_args=True,
            progress_bar=True,
        )
        mcmc.run(rng_key, **mcmc_arguments)
        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

    else:
        assert (
            num_warmup % max_run_length == 0 and num_samples % max_run_length == 0
        ), "Must be able to evenly divide the number of warmup and true samples by the run length"
        warmup_per = max_run_length
        samples_per = max_run_length
        warmup_runs = num_warmup // max_run_length
        sample_runs = num_samples // max_run_length

        mcmc = MCMC(
            kernel,
            num_warmup=warmup_per,
            num_samples=samples_per,
            num_chains=num_chains,
            thinning=thinning,
            jit_model_args=True,
            progress_bar=True,
        )

        for i in range(warmup_runs):
            mcmc.warmup(rng_key, **mcmc_arguments)
            print(f"Done MCMC warmup run {i+1}/{warmup_runs}", flush=True)

        for i in range(sample_runs):
            mcmc.run(rng_key, **mcmc_arguments)
            print(f"Done MCMC run {i+1}/{sample_runs}", flush=True)
            if increment_save_fun is not None:
                increment_save_fun(i, mcmc.get_samples(group_by_chain=group_by_chain))
            if verbose:  # and i> 4 and  i % 5 == 0:
                mcmc.print_summary(exclude_deterministic=False)

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)

    print("\nMCMC elapsed time:", time.time() - start)

    return mcmc.get_samples(group_by_chain=group_by_chain)
