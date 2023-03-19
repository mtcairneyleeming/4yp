"""
Do PriorCVAE on a 50x50 grid

train on GP prior, with lengthscale uniform in 0.01, 0.5

test on GP with true ls 0.05

"""

import time
import sys
import os
import jax.numpy as jnp

# Numpyro
import numpyro
import optax
from jax import random
from numpyro.infer import Predictive

from reusable.data import gen_gp_batches
from reusable.gp import BuildGP
from reusable.kernels import esq_kernel
from reusable.loss import KLD, RCL, combo_loss, conditional_loss_wrapper
from reusable.train_nn import SimpleTrainState, run_training_shuffle
from reusable.util import (
    save_samples,
    save_datasets,
    save_training,
    gen_file_name,
    save_args,
    load_datasets,
    get_decoder_params,
)
from reusable.vae import VAE
from reusable.mcmc import cvae_length_mcmc, run_mcmc

numpyro.set_host_device_count(4)

args = {
    "expcode": 14,
    # GP prior configuration
    "n": 50,
    "dim": 2,
    "gp_kernel": esq_kernel,
    "rng_key": random.PRNGKey(2),
}
args.update(
    {
        "total_n": args["n"] ** args["dim"],
        "axis_range": jnp.arange(0, 1, 1 / args["n"]),
    }
)
args.update(
    {
        "grid_x": jnp.array(jnp.meshgrid(*([args["axis_range"]] * args["dim"]))).T.reshape(
            *([args["n"]] * args["dim"]), args["dim"]
        )
    }
)
args.update(
    {  # so we can use the definition of n to define x
        "x": jnp.reshape(args["grid_x"], (-1, args["dim"])),
        "conditional": True,
        # VAE configuration
        "hidden_dim1": 70,
        "hidden_dim2": 70,
        "latent_dim": 50,
        "vae_var": 0.1,
        # learning
        "num_epochs": 200,
        "learning_rate": 1.0e-3,
        "batch_size": 400,
        "train_num_batches": 400,
        "test_num_batches": 20,
        # MCMC parameters
        "num_warmup": 4000,
        "num_samples": 4000,
        "thinning": 1,
        "num_chains": 3,
        "num_samples_to_save": 4000,
        "rng_key_ground_truth": random.PRNGKey(4),
        "length_prior_choice": "uniform",
        "length_prior_arguments": {"lower": 0.01, "upper": 0.5},
        "ground_truth_ls": 0.2,
    }
)


pre_generated_data = len(sys.argv) >= 2 and sys.argv[1] == "load_generated"

use_gp = len(sys.argv) >= 2 and sys.argv[1] == "use_gp"

on_arc = "SLURM_JOBID" in os.environ

gp = BuildGP(args["gp_kernel"], 5e-5, None, True, args["length_prior_choice"], args["length_prior_arguments"])


rng_key, _ = random.split(random.PRNGKey(4))
rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)


if not pre_generated_data and not use_gp:

    # NOTE changed draw_access - y_c is [y,u] for this
    train_draws = gen_gp_batches(
        args["x"],
        gp,
        args["gp_kernel"],
        args["train_num_batches"],
        args["batch_size"],
        rng_key_train,
        draw_access="y_c",
        jitter=5e-5,
    )
    test_draws = gen_gp_batches(
        args["x"],
        gp,
        args["gp_kernel"],
        1,
        args["test_num_batches"] * args["batch_size"],
        rng_key_test,
        draw_access="y_c",
        jitter=5e-5,
    )
    save_datasets(
        args["expcode"],
        gen_file_name(
            args["expcode"],
            args,
            "raw_gp",
            False,
            ["num_epochs", "hidden_dim1", "hidden_dim2", "latent_dim", "vae_var", "learning_rate"],
        ),
        train_draws,
        test_draws,
    )

elif not use_gp:
    train_draws, test_draws = load_datasets(
        args["expcode"],
        gen_file_name(
            args["expcode"],
            args,
            "raw_gp",
            False,
            ["num_epochs", "hidden_dim1", "hidden_dim2", "latent_dim", "vae_var", "learning_rate"],
        ),
        on_arc=on_arc,
    )


rng_key, rng_key_init, rng_key_init_state, rng_key_train, rng_key_shuffle = random.split(rng_key, 5)


if not use_gp:

    module = VAE(
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        out_dim=args["total_n"],
        conditional=True,
    )
    params = module.init(rng_key_init, jnp.ones((args["batch_size"], args["total_n"] + 1,)))[
        "params"
    ]  # initialize parameters by passing a template image
    tx = optax.adam(args["learning_rate"])
    state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init_state)

    state, metrics_history = run_training_shuffle(
        conditional_loss_wrapper(combo_loss(RCL, KLD)),
        lambda *_: {},
        args["num_epochs"],
        train_draws,
        test_draws,
        state,
        rng_key_shuffle,
    )

    save_training(args["expcode"], gen_file_name(args["expcode"], args), state, metrics_history)


# fixed to generate a "ground truth" GP we will try and infer
rng_key_ground_truth = random.PRNGKey(4)
rng_key_ground_truth, rng_key_ground_truth_obs_mask = random.split(rng_key_ground_truth, 2)

ground_truth_predictive = Predictive(gp, num_samples=1)
gt_draws = ground_truth_predictive(
    rng_key_ground_truth,
    x=args["x"],
    gp_kernel=args["gp_kernel"],
    jitter=1e-5,
    noise=True,
    length=args["ground_truth_ls"],
)
args["ground_truth"] = gt_draws["f"][0]
args["ground_truth_y_draw"] = gt_draws["y"][0]


num_obs = int(args["total_n"] * 0.7)
obs_mask = jnp.concatenate((jnp.full((num_obs), True), jnp.full((args["total_n"] - num_obs), False)))
args["obs_mask"] = random.permutation(rng_key_ground_truth_obs_mask, obs_mask)
args["obs_idx"] = jnp.array([x for x in range(args["total_n"]) if args["obs_mask"][x] == True])
args["ground_truth_y_obs"] = args["ground_truth_y_draw"][args["obs_idx"]]

save_args(args["expcode"], "gp" if use_gp else "v6", args)


rng_key, rng_key_all_mcmc, rng_key_true_mcmc = random.split(rng_key, 3)

f = (
    BuildGP(args["gp_kernel"], 5e-5, args["obs_idx"], True, args["length_prior_choice"], args["length_prior_arguments"])
    if use_gp
    else cvae_length_mcmc(
        args["hidden_dim1"],
        args["hidden_dim2"],
        args["latent_dim"],
        get_decoder_params(state),
        args["obs_idx"],
        True,
        args["length_prior_choice"],
        args["length_prior_arguments"],
    )
)


label = "gp" if use_gp else "cvae"


mcmc_samples = run_mcmc(
    args["num_warmup"],
    args["num_samples"],
    args["num_chains"],
    rng_key_true_mcmc,
    f,
    args["x"],
    args["ground_truth_y_obs"],
    condition=args["ground_truth_ls"],
    verbose=True,
)

save_samples(
    args["expcode"],
    gen_file_name(args["expcode"], args, f"inference_true_ls_mcmc_{label}", include_mcmc=True),
    mcmc_samples,
)


mcmc_samples = run_mcmc(
    args["num_warmup"],
    args["num_samples"],
    args["num_chains"],
    rng_key_true_mcmc,
    f,
    args["x"],
    args["ground_truth_y_obs"],
    args["obs_idx"],
    condition=None,
    verbose=True,
)

save_samples(
    args["expcode"],
    gen_file_name(args["expcode"], args, f"inference_all_ls_mcmc_{label}", include_mcmc=True),
    mcmc_samples,
)
