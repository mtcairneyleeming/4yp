"""
Rewritten loss function comparisons 


Compare (various sets of ) loss functions, save training history & decoder/etc.
+ results of MMD/Frobenius norm comparisons
"""
import os
import sys
import jax.numpy as jnp
import jax.random as random
import optax
from jax import random
from numpyro.infer import Predictive

from reusable.data import gen_gp_batches
from reusable.gp import BuildGP
from reusable.kernels import esq_kernel, rq_matrix_kernel
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD, MMD_rqk
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
)
from reusable.vae import VAE, vae_sample

from reusable.scoring import calc_correlation_mats, calc_frob_norms, calc_mmd_scores, calc_moments

pre_generated_data = len(sys.argv) >= 3 and sys.argv[2] == "load_generated"

pre_trained = len(sys.argv) >= 3 and sys.argv[2] == "pre_trained"

on_arc = "SLURM_JOBID" in os.environ

index = int(sys.argv[1])

print(f"Starting 27, index={index}", flush=True)
setup_signals()


args = {
    # GP prior configuration
    "n": 100,
    "rng_key": random.PRNGKey(2),
}
args.update(
    {  # so we can use the definition of n to define x
        "x": jnp.arange(0, 1, 1 / args["n"]),
        # VAE configuration
        "hidden_dim1": 35,
        "hidden_dim2": 32,
        "latent_dim": 30,
        # learning
        "num_epochs": 100,
        "learning_rate": 1.0e-3,
        "batch_size": 400,
        "train_num_batches": 500,
        "test_num_batches": 20,
        "scoring_num_draws": 5000,
        "expcode": "27",
    }
)

args["length_priors"] = [
    {
        "length_prior_choice": "invgamma",
        "length_prior_arguments": {"concentration": 1.0, "rate": 1.0},
    },
    {
        "length_prior_choice": "lognormal",
        "length_prior_arguments": {"location": 0.0, "scale": 1.0},
    },
    {
        "length_prior_choice": "lognormal",
        "length_prior_arguments": {"location": 0.0, "scale": 0.25},
    },
]

args["variance_priors"] = [
    {
        "variance_prior_choice": "gamma",
        "variance_prior_arguments": {"concentration": 5, "rate": 0.25},
    },
    {
        "variance_prior_choice": "halfnormal",
        "variance_prior_arguments": {"scale": 1},
    },
    {
        "variance_prior_choice": "uniform",
        "variance_prior_arguments": {"lower": 2, "upper": 12},
    },
]

args["gp_kernels"] = [esq_kernel, rq_matrix_kernel(0.5)]

args["loss_fns"] = [
    combo_loss(RCL, KLD),
    combo3_loss(RCL, KLD, MMD_rbf(4.0), 0.01, 1, 10),
    combo3_loss(RCL, KLD, MMD_rqk(4, 10), 0.1, 1, 10),
]


total = len(args["length_priors"]) * len(args["variance_priors"]) * len(args["gp_kernels"]) * len(args["loss_fns"])

args_index = 1

args["loss_fn_names"] = [x.__name__ for x in args["loss_fns"]]

save_args(args["expcode"], 2, args)


print(f" index {index}/{total -1} (0-indexed!)")

length_index, variance_index, gp_kernel_index, loss_fn_index = jnp.unravel_index(
    index, (len(args["length_priors"]), len(args["variance_priors"]), len(args["gp_kernels"]), len(args["loss_fns"]))
)


print(f"Split indices: {length_index}, {variance_index}, {gp_kernel_index }, {loss_fn_index}")

args.update(args["length_priors"][length_index])

args.update(args["variance_priors"][variance_index])

args["gp_kernel"] = args["gp_kernels"][gp_kernel_index]

loss_fn = args["loss_fns"][loss_fn_index]

rng_key, _ = random.split(random.PRNGKey(4))


rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)

gp = BuildGP(
    args["gp_kernel"],
    noise=False,
    length_prior_choice=args["length_prior_choice"],
    length_prior_args=args["length_prior_arguments"],
    variance_prior_choice=args["variance_prior_choice"],
    variance_prior_args=args["variance_prior_arguments"],
)


if not pre_trained:
    if not pre_generated_data:
        train_draws = gen_gp_batches(
            args["x"], gp, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train
        )

        test_draws = gen_gp_batches(
            args["x"], gp, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test
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

    print("Generated data", flush=True)

file_name = gen_file_name(args["expcode"], args, loss_fn.__name__)


rng_key, rng_key_init, rng_key_train, rng_key_shuffle = random.split(rng_key, 4)

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
    args["expcode"],
    file_name,
    {"frobenius": frob_norms, "vae_moments": vae_moments, "mmd": mmd_scores, "gp_moments": gp_moments},
)
