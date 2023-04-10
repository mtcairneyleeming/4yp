"""
Experiments to determine scalability, of PriorVAE, with 2 loss fns: standard RCL+KLD, and combo of RCL, KLD and MMD

Experiment 1: n vs train_num_batches, fix batch size
- scalability to larger sets

Experiment 2: train_num_batches vs batch_size, fix n['s]
- incl both fixed amount of data and less-data regimes

Experiment 3: n vs num_epochs, fixed amount of data

Experiment 4: n vs VAE dim scaling

Note selecting between these is done with the job name.
"""
import os
import sys

index = int(sys.argv[1])
experiment = sys.argv[2]

pre_trained = len(sys.argv) > 3 and sys.argv[3] == "pre_trained"

on_arc = "SLURM_JOBID" in os.environ

print(f"Starting experiment {experiment}, index {index}", flush=True)


import jax.numpy as jnp
import jax.random as random
import optax
from jax import random
from numpyro.infer import Predictive

from reusable.data import gen_gp_batches
from reusable.gp import BuildGP
from reusable.kernels import esq_kernel
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD
from reusable.train_nn import SimpleTrainState, run_training_shuffle
from reusable.util import (
    save_args,
    save_training,
    setup_signals,
    gen_file_name,
    update_args_11,
    save_scores,
    get_decoder_params,
    load_training_state,
)
from reusable.vae import VAE, vae_sample

from reusable.scoring import calc_correlation_mats, calc_frob_norms, calc_mmd_scores, calc_moments

setup_signals()
print("Starting work", flush=True)


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
        "hidden_dim1": 35,
        "hidden_dim2": 32,
        "latent_dim": 30,
        "vae_var": 0.1,
        # learning
        "num_epochs": 100,
        "learning_rate": 1.0e-3,
        "batch_size": 400,
        "train_num_batches": 500,
        "test_num_batches": 20,
        "mmd_rbf_ls": 4.0,

        "length_prior_choice": "invgamma",
        "length_prior_arguments": {"concentration": 4.0, "rate": 1.0},

        "scoring_num_draws": 5000,

        "jitter_scaling": 1 / 300 * 4e-6,  # n times this gives the jitter

        "exp5": {
            "Arange": [25, 50, 100, 150, 200, 225, 250],
            "Brange": [50, 75, 100, 125, 150, 175, 200],
            "Adesc": "n",
            "Bdesc": "train_num_batches",
        },
        "exp6": {
            "Arange": [
                50,
                100,
                200,
                400,
                500,
                600,
                800,
            ],  # idea is diagonal is the same amount of data as current: 200,000
            "Brange": [250, 333, 400, 500, 1000, 2000, 4000],
            "Adesc": "batch_size",
            "Bdesc": "train_num_batches",
        },
        "exp7": {
            "Arange": [25, 50, 100, 150, 200, 250, 300],
            "Brange": [1, 5, 10, 25, 50, 100],
            "Adesc": "n",
            "Bdesc": "num_epochs",
        },
        "exp8": {
            "Arange": [25, 50, 100, 150, 200, 250, 300],
            "Brange": [0.25, 0.5, 1, 2, 4, 8],
            "Adesc": "n",
            "Bdesc": "vae_scale_factor",
        },
        "exp9": {
            "Arange": [100, 200, 400, 800, 1600, 3200, 6400],
            "Brange": [100, 200, 400, 800],
            "Adesc": "n",
            "Bdesc": "train_num_batches",
            "loss_fns": [combo_loss(RCL, KLD)]
        },
        "experiment": experiment,
    }
)

loss_fns = [combo_loss(RCL, KLD), combo3_loss(RCL, KLD, MMD_rbf(args["mmd_rbf_ls"]), 0.01, 1, 10)]
if "loss_fns" in args[experiment]:
    loss_fns = args[experiment]["loss_fns"]

args["loss_fns"] = [l.__name__ for l in loss_fns]

save_args("11", experiment, args)

Arange = args[experiment]["Arange"]
Brange = args[experiment]["Brange"]
ar = len(Arange)
br = len(Brange)
b = index // ar
a = index % ar

print(f"Exp {experiment}, a={a}/{ar-1}, b={b}/{br-1}, index={index}/{ar*br-1} [indices 0-(n-1)]")


update_args_11(args, args[experiment], a, b)


rng_key, rng_key_train, rng_key_test = random.split(args["rng_key"], 3)


gp = BuildGP(
    args["gp_kernel"],
    noise=False,
    length_prior_choice=args["length_prior_choice"],
    prior_args=args["length_prior_arguments"],
)


if not pre_trained:

    train_draws = gen_gp_batches(
        args["x"],
        gp,
        args["gp_kernel"],
        args["train_num_batches"],
        args["batch_size"],
        rng_key_train,
        jitter=args["n"] * args["jitter_scaling"],
    )
    test_draws = gen_gp_batches(
        args["x"],
        gp,
        args["gp_kernel"],
        1,
        args["test_num_batches"] * args["batch_size"],
        rng_key_test,
        jitter=args["n"] * args["jitter_scaling"],
    )


print("Generated data", flush=True)

rng_key, rng_key_init, rng_key_train, rng_key_train_shuffle, rng_key_scores = random.split(rng_key, 5)

module = VAE(
    hidden_dim1=args["hidden_dim1"],
    hidden_dim2=args["hidden_dim2"],
    latent_dim=args["latent_dim"],
    out_dim=args["n"],
    conditional=False,
)

params = module.init(rng_key, jnp.ones((args["n"],)))["params"]  # initialize parameters by passing a template image
tx = optax.adam(args["learning_rate"])
state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)


print("Starting training", flush=True)

def calc_scores(final_state, file_name, rng_key):

    rng_key, rng_key_gp, rng_key_vae = random.split(rng_key, 3)

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
        "11",
        file_name,
        {"frobenius": frob_norms, "vae_moments": vae_moments, "mmd": mmd_scores, "gp_moments": gp_moments},
    )


for loss_fn in loss_fns:
    file_name = gen_file_name("11", args, f"11_{experiment}_{index}_{loss_fn.__name__}")
 
    if args[experiment]["Bdesc"] != "num_epochs" and args[experiment]["Adesc"] != "num_epochs":
        # timing is now automatic in run_training

        if not pre_trained:
            final_state, metrics_history = run_training_shuffle(
                loss_fn, lambda *_: {}, args["num_epochs"], train_draws, test_draws, state, rng_key_train_shuffle
            )
            save_training("11", file_name, final_state, metrics_history)
        else: 
            final_state = load_training_state("11", file_name, state, arc_learnt_models_dir=on_arc)

        calc_scores(final_state, file_name, rng_key_scores)

    else:
        prev_history = {}
        iterate_list = Brange if args[experiment]["Bdesc"] == "num_epochs" else Arange
        final_state = state

        for j, _ in enumerate(iterate_list):
            new_index = j * ar + a            
            args = update_args_11(args, args[experiment], a, b)  # set num_epochs correctly now!
            file_name = gen_file_name("11", args, f"11_{experiment}_{new_index}_{loss_fn.__name__}")
            
            if not pre_trained:

                next_range = iterate_list[j] - iterate_list[j - 1] if j > 0 else iterate_list[0]

                final_state, h = run_training_shuffle(loss_fn, lambda *_: {}, next_range, train_draws, test_draws, final_state, random.fold_in(rng_key_train_shuffle, j))

                if j > 0:
                    for metric, value in h.items():
                        if metric in ["interrupted", "final_epoch"]:
                            prev_history[metric] = value
                        elif metric in [
                            "epoch_times",
                            "batch_times",
                        ]:  # correct fact that we don't pass times back in to run_training
                            prev_history[metric] = jnp.append(
                                prev_history[metric], value + prev_history["epoch_times"][-1], axis=0
                            )
                        else:
                            prev_history[metric] = jnp.append(prev_history[metric], value, axis=0)

                else:
                    prev_history = h

                save_training("11", file_name, final_state, metrics_history)
            else:
                final_state = load_training_state("11", file_name, state, arc_learnt_models_dir=on_arc)

            calc_scores(final_state, file_name, random.fold_in(rng_key_scores, j))

            if "interrupted" in h:
                print("SIGTERM sent, not iterating")
                sys.exit(0)
