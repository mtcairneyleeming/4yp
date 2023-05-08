"""
Run PriorVAE with a couple of loss functions on the NY data
"""
import os
import sys
import jax.numpy as jnp
import jax.random as random
import optax
from jax import random
import spacv
import numpy as onp
from numpyro.infer import Predictive
import numpyro

numpyro.set_host_device_count(4)

from reusable.data import gen_gp_batches, gen_latent_batches, pair_batched_data
from reusable.gp import BuildGP
from reusable.kernels import esq_kernel
from reusable.loss import MMD_rbf, MMD_rqk
from reusable.train_nn import SimpleTrainState, run_training_shuffle
from reusable.util import (
    save_args,
    save_training,
    setup_signals,
    gen_file_name,
    save_scores,
    get_model_params,
    save_datasets,
    load_datasets,
    load_training_state,
    save_samples,
)
from reusable.vae import Single_Decoder, decoder_sample
from reusable.geo import load_state_centroids, centroids_to_coords, get_processed_temp_data
from reusable.mcmc import vae_mcmc, run_mcmc
from reusable.scoring import calc_correlation_mats, calc_frob_norms, calc_mmd_scores, calc_moments
from reusable.split import calculate_obs_fracs, calculate_spatial_cv


load_gp_data = len(sys.argv) > 2 and sys.argv[2] == "load_gp_data"

pre_generated_data = len(sys.argv) > 2 and sys.argv[2] == "load_generated"

pre_trained = len(sys.argv) > 2 and sys.argv[2] == "pre_trained"

skip_scores = len(sys.argv) > 3 and sys.argv[3] == "skip_scores"

on_arc = "SLURM_JOBID" in os.environ

index = int(sys.argv[1])

print(
    f"Starting 29, index={index}, pre_gen: {pre_generated_data}, pre_trained={pre_trained}, skip_scores={skip_scores}",
    flush=True,
)
setup_signals()


args = {
    # geographic data
    "state": 36,  # New York
    # ground truth
    "year": 2010,
    "coord_scaling_factor": 1e5,
    "aggr_method": "mean",
    # GP prior configuration
    "gp_kernel": esq_kernel,
    "rng_key": random.PRNGKey(2),
}

state_centroids = load_state_centroids(args["state"])
coords, coord_means = centroids_to_coords(state_centroids, args["coord_scaling_factor"])

args.update(
    {  # so we can use the definition of n to define x
        "x": coords,
        "n": coords.shape[0],
        # VAE configuration
        "hidden_dim1": 630,  # 0.35 * n
        "hidden_dim2": 575,
        "latent_dim": 540,
        # learning
        "num_epochs": 50,
        "learning_rate": 1.0e-3,
        "batch_size": 400,
        "train_num_batches": 200,
        "test_num_batches": 2,
        "length_prior_choice": "invgamma",
        "length_prior_arguments": {"concentration": 4.0, "rate": 1.0},
        "variance_prior_choice": "gamma",
        "variance_prior_arguments": {"concentration": 5.25, "rate": 0.5},
        "scoring_num_draws": 2000,
        "expcode": "29",
        "loss_fns": [MMD_rbf(x) for x in [1, 4, 8, 12]]
        + [MMD_rqk(2, 0.1), MMD_rqk(4, 1), MMD_rqk(6, 10), MMD_rqk(8, 100)],
        # MCMC parameters
        "num_warmup": 1000,
        "num_samples": 4000,
        "thinning": 1,
        "num_chains": 4,
        "jitter_scaling": 1 / 300 * 6e-6,  # n times this gives the jitter,
        "num_cv_splits": 5,
        "obs_fracs": [0.01, 0.05, 0.10, 0.2, 0.3],
        "observations_rng_key": random.PRNGKey(123456789),
    }
)


args["ground_truth"], args["temp_mean_offset"] = get_processed_temp_data(
    args["state"], args["year"], args["aggr_method"]
)

rng_key_observations, rng_key_spatial_cv = random.split(args["observations_rng_key"], 2)

obs_idx_lst = calculate_obs_fracs(args["obs_fracs"], args["n"], rng_key_observations) + calculate_spatial_cv(
    args["num_cv_splits"], state_centroids["geometry"], args["n"], rng_key_spatial_cv
)


args["loss_fn_names"] = ["gp" if x is None else x.__name__ for x in args["loss_fns"]]

save_args(args["expcode"], "1", args)

print(f" index {index}/{len(args['loss_fns']) -1} (0-indexed!)")

loss_fn = args["loss_fns"][index]

using_gp = loss_fn is None

rng_key, _ = random.split(random.PRNGKey(4))


rng_key, rng_key_train, rng_key_test, rng_key_latent_train, rng_key_latent_test = random.split(rng_key, 5)

gp = BuildGP(
    args["gp_kernel"],
    noise=False,
    length_prior_choice=args["length_prior_choice"],
    length_prior_args=args["length_prior_arguments"],
    variance_prior_choice=args["variance_prior_choice"],
    variance_prior_args=args["variance_prior_arguments"],
)


if not using_gp and not pre_trained:
    if pre_generated_data:
        train_draws, test_draws = load_datasets(
            args["expcode"],
            gen_file_name(args["expcode"], args, "raw_batched", data_only=True),
            on_arc=on_arc,
        )
    else:
        train_latent_draws = gen_latent_batches(
            args["latent_dim"], args["train_num_batches"], args["batch_size"], rng_key_latent_train
        )
        test_latent_draws = gen_latent_batches(
            args["latent_dim"], 1, args["test_num_batches"] * args["batch_size"], rng_key_latent_test
        )
        if load_gp_data:
            train_gp_draws, test_gp_draws = load_datasets(
                19,  # args["expcode"],
                gen_file_name(
                    19,  # args["expcode"],
                    args,
                    "raw_gp",
                    data_only=True,
                ),
                on_arc=on_arc,
            )
        else:
            train_gp_draws = gen_gp_batches(
                args["x"],
                gp,
                args["gp_kernel"],
                args["train_num_batches"],
                args["batch_size"],
                rng_key_train,
                jitter=args["n"] * args["jitter_scaling"],
            )

            test_gp_draws = gen_gp_batches(
                args["x"],
                gp,
                args["gp_kernel"],
                1,
                args["test_num_batches"] * args["batch_size"],
                rng_key_test,
                jitter=args["n"] * args["jitter_scaling"],
            )

        train_draws = pair_batched_data(train_gp_draws, train_latent_draws)
        test_draws = pair_batched_data(test_gp_draws, test_latent_draws)
        save_datasets(
            args["expcode"],
            gen_file_name(args["expcode"], args, "raw_batched", data_only=True),
            train_draws,
            test_draws,
        )

file_name = gen_file_name(args["expcode"], args, "gp" if loss_fn is None else loss_fn.__name__)


rng_key, rng_key_init, rng_key_train, rng_key_shuffle = random.split(rng_key, 4)

if not using_gp:
    module = Single_Decoder(
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        out_dim=args["n"],
    )
    params = module.init(rng_key, jnp.ones((args["n"] + args["latent_dim"],)))["params"]
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

    args["decoder_params"] = get_model_params(final_state)


rng_key, rng_key_gp, rng_key_vae = random.split(rng_key, 3)


if not using_gp and not skip_scores:
    print("Drawing from GP", flush=True)

    gp_predictive = Predictive(gp, num_samples=args["scoring_num_draws"])
    gp_draws = gp_predictive(rng_key_gp, x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5)["y"]

    print("Drawing from VAE", flush=True)

    plot_vae_predictive = Predictive(decoder_sample, num_samples=args["scoring_num_draws"])
    vae_draws = plot_vae_predictive(
        rng_key_vae,
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        out_dim=args["n"],
        decoder_params=get_model_params(final_state),
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


label = "gp" if using_gp else f"{loss_fn.__name__}"

label += f"{args['state']}_{args['year']}_{args['aggr_method']}"


rng_key, rng_key_mcmc = random.split(rng_key, 2)

for i, obs_idx in enumerate(obs_idx_lst):
    f = (
        BuildGP(
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
    mcmc_samples = run_mcmc(
        args["num_warmup"],
        args["num_samples"],
        args["num_chains"],
        random.fold_in(rng_key_mcmc, i),
        f,
        {"x": args["x"], "y": args["ground_truth"][obs_idx]},
        verbose=True,
        max_run_length=100 if using_gp else None,
        increment_save_fun=(
            lambda j, x: save_samples(
                args["expcode"],
                gen_file_name(
                    args["expcode"], args, f"inference_{label}_split{i}_intermed_{j}_mcmc", include_mcmc=True
                ),
                x,
            )
        )
        if using_gp
        else None,
    )
    save_samples(
        args["expcode"],
        gen_file_name(args["expcode"], args, f"inference_{label}_split{i}_mcmc", include_mcmc=True),
        mcmc_samples,
    )
