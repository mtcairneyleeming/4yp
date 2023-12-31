"""
Try the synthetic data example
"""
import os
import sys
import jax.numpy as jnp
import jax.random as random
import optax
from jax import random
from numpyro.infer import Predictive
import numpyro

numpyro.set_host_device_count(4)

from reusable.data import gen_gp_batches
from reusable.gp import BuildGP_Poisson
from reusable.kernels import esq_kernel
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD
from reusable.train_nn import SimpleTrainState, run_training_shuffle
from reusable.util import (
    save_args,
    save_training,
    setup_signals,
    gen_file_name,
    save_scores,
    get_decoder_params,
    save_datasets,
    load_datasets,
    load_training_state,
    save_samples,
)
from reusable.vae import VAE, vae_sample
from reusable.mcmc import vae_mcmc, run_mcmc
from reusable.scoring import calc_correlation_mats, calc_frob_norms, calc_mmd_scores, calc_moments

pre_generated_data = len(sys.argv) > 2 and sys.argv[2] == "load_generated"

pre_trained = len(sys.argv) > 2 and sys.argv[2] == "pre_trained"

on_arc = "SLURM_JOBID" in os.environ

index = int(sys.argv[1])

print(f"Starting 21, index={index}, pre_gen: {pre_generated_data}, pre_trained={pre_trained}", flush=True)
setup_signals()

args = {
    # GP prior configuration
    "n": 100,
    "gp_kernel": esq_kernel,
    "rng_key": random.PRNGKey(2),
}
args.update(
    {  # so we can use the definition of n to define x
        "x": jnp.arange(0, 1, 1 / args["n"]),
        # VAE configuration
        "hidden_dim1": 100,
        "hidden_dim2": 100,
        "latent_dim": 100,
        # learning
        "num_epochs": 50,
        "learning_rate": 1.0e-3,
        "batch_size": 800,
        "train_num_batches": 100,
        "test_num_batches": 2,
        "length_prior_choice": "invgamma",
        "length_prior_arguments": {"concentration": 4.0, "rate": 1.0},
        "variance_prior_choice": "lognormal",
        "variance_prior_arguments": {"location": 0.0, "scale": 0.1},
        "scoring_num_draws": 5000,
        "expcode": "21",
        "loss_fns": [None, combo_loss(RCL, KLD), combo3_loss(RCL, KLD, MMD_rbf(4.0), 0.01, 1, 10)],
        # MCMC parameters
        "num_warmup": 1000,
        "num_samples": 4000,
        "thinning": 1,
        "num_chains": 4,
    }
)

gt_gp = BuildGP_Poisson(
    args["gp_kernel"],
    noise=True,
    length_prior_choice=args["length_prior_choice"],
    length_prior_args=args["length_prior_arguments"],
    variance_prior_choice=args["variance_prior_choice"],
    variance_prior_args=args["variance_prior_arguments"],
)

rng_key_ground_truth = random.PRNGKey(1234)  # fixed to generate a "ground truth" GP we will try and infer

ground_truth_predictive = Predictive(gt_gp, num_samples=1)
gt_draws = ground_truth_predictive(rng_key_ground_truth, x=args["x"], gp_kernel=args["gp_kernel"])
ground_truth = gt_draws["f"].T
ground_truth_y_draw = gt_draws["y"].T

args["ground_truth"] = ground_truth_y_draw

args["obs_idx_lst"] = [[22, 50], [16, 33, 57, 96], [8, 24, 45, 61, 77, 84]]


args["loss_fn_names"] = ["gp" if x is None else x.__name__ for x in args["loss_fns"]]

save_args(args["expcode"], "3", args)


print(f" index {index}/{len(args['loss_fns']) -1} (0-indexed!)")

loss_fn = args["loss_fns"][index]

using_gp = loss_fn is None

rng_key = args["rng_key"]


rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)

gp = BuildGP_Binomial(
    args["binomial_N"],
    args["gp_kernel"],
    noise=False,
    length_prior_choice=args["length_prior_choice"],
    length_prior_args=args["length_prior_arguments"],
    variance_prior_choice=args["variance_prior_choice"],
    variance_prior_args=args["variance_prior_arguments"],
)


if not using_gp and not pre_trained:
    if not pre_generated_data:
        train_draws = gen_gp_batches(
            args["x"],
            gp,
            args["gp_kernel"],
            args["train_num_batches"],
            args["batch_size"],
            rng_key_train,
        )

        test_draws = gen_gp_batches(
            args["x"],
            gp,
            args["gp_kernel"],
            1,
            args["test_num_batches"] * args["batch_size"],
            rng_key_test,
        )
        save_datasets(
            args["expcode"],
            gen_file_name(args["expcode"], args, "raw_gp", data_only=True),
            train_draws,
            test_draws,
        )

    else:
        train_draws, test_draws = load_datasets(
            args["expcode"],
            gen_file_name(
                args["expcode"],
                args,
                "raw_gp",
                data_only=True,
            ),
            on_arc=on_arc,
        )

    print(f"Generated data, train size: {train_draws.nbytes}, test size: {test_draws.nbytes}", flush=True)

file_name = gen_file_name(args["expcode"], args, "gp" if loss_fn is None else loss_fn.__name__)


rng_key, rng_key_init, rng_key_train, rng_key_shuffle = random.split(rng_key, 4)

if not using_gp:

    module = VAE(
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        out_dim=args["n"],
        conditional=False,
    )
    params = module.init(rng_key, jnp.ones((args["n"],)))["params"]
    tx = optax.adam(args["learning_rate"])
    state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

    if not pre_trained:
        print("Starting training", flush=True)

        final_state, metrics_history = run_training_shuffle(
            loss_fn, None, args["num_epochs"], train_draws, test_draws, state, rng_key_shuffle
        )

        save_training(args["expcode"], file_name, final_state, metrics_history)

        del train_draws
        del test_draws

    else:
        final_state = load_training_state(args["expcode"], file_name, state, arc_learnt_models_dir=on_arc)

    args["decoder_params"] = get_decoder_params(final_state)


rng_key, rng_key_gp, rng_key_vae = random.split(rng_key, 3)


if not using_gp:
    print("Drawing from GP", flush=True)

    gp_predictive = Predictive(gp, num_samples=args["scoring_num_draws"])
    gp_draws = gp_predictive(rng_key_gp, x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5)["y"]

    print("Drawing from VAE", flush=True)

    plot_vae_predictive = Predictive(vae_sample, num_samples=args["scoring_num_draws"])
    vae_draws = plot_vae_predictive(
        rng_key_vae,
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        out_dim=args["n"],
        decoder_params=get_decoder_params(final_state),
    )["f"]

    print("Calculating Frobenius norms", flush=True)
    frob_norms = calc_frob_norms(calc_correlation_mats(vae_draws), calc_correlation_mats(gp_draws))

    print("Calculating moments", flush=True)
    vae_moments = calc_moments(vae_draws)

    gp_moments = calc_moments(gp_draws)

    print("Calculating MMD", flush=True)
    mmd_scores = calc_mmd_scores(gp_draws, vae_draws)

    save_scores(
        args["expcode"],
        file_name,
        {"frobenius": frob_norms, "vae_moments": vae_moments, "mmd": mmd_scores, "gp_moments": gp_moments},
    )


for obs_idx in args["obs_idx_lst"]:
    obs_idx = jnp.array(obs_idx)
    f = (
        BuildGP_Binomial(
            args["binomial_N"],
            args["gp_kernel"],
            noise=True,
            length_prior_choice=args["length_prior_choice"],
            length_prior_args=args["length_prior_arguments"],
            variance_prior_choice=args["variance_prior_choice"],
            variance_prior_args=args["variance_prior_arguments"],
            obs_idx=obs_idx,
        )
        if using_gp
        else vae_mcmc(
            args["hidden_dim1"],
            args["hidden_dim2"],
            args["latent_dim"],
            args["decoder_params"],
            obs_idx=obs_idx,
            noise=True,
        )
    )

    label = "gp" if using_gp else f"{loss_fn.__name__}"

    rng_key, rng_key_mcmc = random.split(rng_key, 2)



    mcmc_samples = run_mcmc(
        args["num_warmup"],
        args["num_samples"],
        args["num_chains"],
        rng_key_mcmc,
        f,
        {"x": args["x"], "y": args["ground_truth"][obs_idx, 0]},
        verbose=True,
        max_run_length=None,
        use_mixed_hmc=False
    )
    save_samples(
        args["expcode"],
        gen_file_name(
            args["expcode"], args, f"inference_{label}_{'-'.join([str(x) for x in obs_idx])}_mcmc", include_mcmc=True
        ),
        mcmc_samples,
    )
