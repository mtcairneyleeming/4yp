"""
Rewritten decoder-only
"""
import os
import sys
import jax.numpy as jnp
import jax.random as random
import optax
from jax import random
from numpyro.infer import Predictive

from reusable.data import gen_gp_batches, gen_latent_batches, pair_batched_data
from reusable.gp import BuildGP
from reusable.kernels import esq_kernel
from reusable.loss import combo_loss, MMD_rbf, RCL, MMD_rqk
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
)
from reusable.vae import Single_Decoder, decoder_sample

from reusable.scoring import calc_correlation_mats, calc_frob_norms, calc_mmd_scores, calc_moments

pre_generated_data = len(sys.argv) >= 4 and sys.argv[3] == "load_generated"

pre_trained = len(sys.argv) >= 4 and sys.argv[3] == "pre_trained"

on_arc = "SLURM_JOBID" in os.environ

experiment = str(sys.argv[2])
index = int(sys.argv[1])

print(f"Starting 20, experiment {experiment}, index={index}", flush=True)
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
        "hidden_dim1": 35,
        "hidden_dim2": 32,
        "latent_dim": 30,
        "vae_var": 0.1,
        # learning
        "num_epochs": 150,
        "learning_rate": 1.0e-3,
        "batch_size": 400,
        "train_num_batches": 500,
        "test_num_batches": 20,
        "length_prior_choice": "invgamma",
        "length_prior_arguments": {"concentration": 4.0, "rate": 1.0},
        "scoring_num_draws": 5000,
        "expcode": "20",
    }
)
if experiment == "exp1":  # basic introduction to different loss functions
    args["loss_fns"] = [
        combo_loss(RCL, MMD_rbf(1.0), 1, 10),
        combo_loss(RCL, MMD_rbf(4.0), 1, 10),
        combo_loss(RCL, MMD_rbf(7.0), 1, 10),
        combo_loss(RCL, MMD_rbf(10.0), 1, 10),
        MMD_rbf(1.0),
        MMD_rbf(4.0),
        MMD_rbf(7.0),
        MMD_rbf(10.0),
    ]


if experiment == "exp2":  # expand upon the MMD_rqk kerne;
    args["Arange"] = [1, 2, 4, 6, 8]
    args["Brange"] = [0.1, 1, 10, 100]

    temp_loss_fns = [[MMD_rqk(a, b) for a in args["Arange"]] for b in args["Brange"]]
    args["loss_fns"] = [x for xs in temp_loss_fns for x in xs]


if experiment == "exp3":  # vary the scaling term
    args["Arange"] = [10, 1, 0.1, 0.01]
    args["Brange"] = [1, 5, 10, 15, 20]

    temp_loss_fns = [[combo_loss(RCL, MMD_rbf(4.0), a, b) for a in args["Arange"]] for b in args["Brange"]]
    args["loss_fns"] = [x for xs in temp_loss_fns for x in xs]


args["experiment"] = experiment

args["loss_fn_names"] = [x.__name__ for x in args["loss_fns"]]

save_args(args["expcode"], args["experiment"], args)


print(f" index {index}/{len(args['loss_fns']) -1} (0-indexed!)")

loss_fn = args["loss_fns"][index]

rng_key = args["rng_key"]


rng_key, rng_key_train, rng_key_test, rng_key_latent_train, rng_key_latent_test = random.split(rng_key, 5)


gp = BuildGP(
    args["gp_kernel"],
    noise=False,
    length_prior_choice=args["length_prior_choice"],
    prior_args=args["length_prior_arguments"],
)


if not pre_trained:
    if not pre_generated_data:

        train_gp_draws = gen_gp_batches(
            args["x"], gp, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train
        )
        train_latent_draws = gen_latent_batches(
            args["latent_dim"], args["train_num_batches"], args["batch_size"], rng_key_latent_train
        )
        test_gp_draws = gen_gp_batches(
            args["x"], gp, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test
        )
        test_latent_draws = gen_latent_batches(
            args["latent_dim"], 1, args["test_num_batches"] * args["batch_size"], rng_key_latent_test
        )

        train_draws = pair_batched_data(train_gp_draws, train_latent_draws)
        test_draws = pair_batched_data(test_gp_draws, test_latent_draws)
        save_datasets(
            args["expcode"],
            gen_file_name(args["expcode"], args, "raw_batched", data_only=True),
            train_draws,
            test_draws,
        )

    else:
        train_draws, test_draws = load_datasets(
            args["expcode"],
            gen_file_name(
                args["expcode"],
                args,
                "raw_batched",
                data_only=True,
            ),
            on_arc=on_arc,
        )

    print("Generated data", flush=True)

file_name = gen_file_name(args["expcode"], args, args["experiment"] + loss_fn.__name__)


rng_key, rng_key_init, rng_key_train, rng_key_shuffle = random.split(rng_key, 4)

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


rng_key, rng_key_gp, rng_key_vae = random.split(rng_key, 3)

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
