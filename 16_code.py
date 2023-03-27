"""
Rewritten loss function comparisons 


Compare (various sets of ) loss functions, save training history & decoder/etc.
+ results of MMD/Frobenius norm comparisons
"""
import time
import sys
import jax.numpy as jnp
import jax.random as random
import optax
from jax import random
from numpyro.infer import Predictive

from reusable.data import gen_gp_batches
from reusable.gp import BuildGP
from reusable.kernels import esq_kernel
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD, MMD_rqk
from reusable.train_nn import SimpleTrainState, run_training_shuffle
from reusable.util import save_args, save_training, setup_signals, gen_file_name, save_scores, get_decoder_params
from reusable.vae import VAE, vae_sample

from reusable.scoring import calc_correlation_mats, calc_frob_norms, calc_mmd_scores, calc_moments

experiment = str(sys.argv[2])
index = int(sys.argv[1])

print(f"Starting 16, experiment {experiment}, index={index}", flush=True)
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
        "learning_rate": 1.0e-4,
        "batch_size": 400,
        "train_num_batches": 500,
        "test_num_batches": 20,
        "length_prior_choice": "invgamma",
        "length_prior_arguments": {"concentration": 4.0, "rate": 1.0},
        "scoring_num_draws": 20000,
    }
)
if experiment == "exp1":
    args["experiment"] = "exp1"

    args["loss_fns"] = [
        combo_loss(RCL, KLD),
        combo_loss(KLD, MMD_rbf(4.0)),
        combo_loss(KLD, MMD_rbf(10.0)),
        combo_loss(KLD, MMD_rbf(20.0)),
        combo3_loss(RCL, KLD, MMD_rbf(4.0)),
        combo3_loss(RCL, KLD, MMD_rbf(10.0)),
        combo3_loss(RCL, KLD, MMD_rbf(20.0)),
    ]


if experiment == "exp2":
    args["experiment"] = "exp2"
    args["Arange"] = [4, 8, 12, 20]
    args["Brange"] = [0.25, 1, 5, 10, 20]

    temp_loss_fns = [[combo_loss(MMD_rqk(a, b), KLD) for a in args["Arange"]] for b in args["Brange"]]
    args["loss_fns"] = [x for xs in temp_loss_fns for x in xs]

if experiment == "exp2_rcl":
    args["experiment"] = "exp2_rcl"
    args["Arange"] = [4, 8, 12, 20]
    args["Brange"] = [0.25, 1, 5, 10, 20]

    temp_loss_fns = [
        [
            combo3_loss(
                RCL,
                MMD_rqk(a, b),
                KLD,
            )
            for a in args["Arange"]
        ]
        for b in args["Brange"]
    ]
    args["loss_fns"] = [x for xs in temp_loss_fns for x in xs]

if experiment == "exp3":
    args["experiment"] = "exp3"
    args["Arange"] = [10, 1, 0.1, 0.01]
    args["Brange"] = [1, 5, 10, 15, 20]

    temp_loss_fns = [[combo3_loss(RCL, MMD_rbf(4.0), KLD, a, b, 1) for a in args["Arange"]] for b in args["Brange"]]
    args["loss_fns"] = [x for xs in temp_loss_fns for x in xs]

save_args("16", args["experiment"], args)


print(f" index {index}/{len(args['loss_fns']) -1} (0-indexed!)")

loss_fn = args["loss_fns"][index]

rng_key, _ = random.split(random.PRNGKey(4))


rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)

gp = BuildGP(
    args["gp_kernel"],
    noise=False,
    length_prior_choice=args["length_prior_choice"],
    prior_args=args["length_prior_arguments"],
)

train_draws = gen_gp_batches(
    args["x"], gp, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train
)

test_draws = gen_gp_batches(
    args["x"], gp, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test
)


print("Generated data", flush=True)


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


print("Starting training", flush=True)


final_state, metrics_history = run_training_shuffle(
    loss_fn, lambda *_: {}, args["num_epochs"], train_draws, test_draws, state, rng_key_shuffle
)

save_training("16", gen_file_name("16", args, args["experiment"] + loss_fn.__name__), final_state, metrics_history)

del train_draws
del test_draws

rng_key, rng_key_gp, rng_key_vae = random.split(rng_key, 3)


gp_predictive = Predictive(gp, num_samples=args["scoring_num_draws"])
gp_draws = gp_predictive(rng_key_gp, x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5)["y"]

plot_vae_predictive = Predictive(vae_sample, num_samples=args["scoring_num_draws"])
vae_draws = plot_vae_predictive(
    rng_key_vae,
    hidden_dim1=args["hidden_dim1"],
    hidden_dim2=args["hidden_dim2"],
    latent_dim=args["latent_dim"],
    out_dim=args["n"],
    decoder_params=get_decoder_params(final_state),
)["f"]


frob_norms = calc_frob_norms(calc_correlation_mats(vae_draws), calc_correlation_mats(gp_draws))

vae_moments = calc_moments(vae_draws)

mmd_scores = calc_mmd_scores(gp_draws, vae_draws)

save_scores(
    "16",
    gen_file_name("16", args, str(index)),
    {"frobenius": frob_norms, "vae_moments": vae_moments, "mmd": mmd_scores},
)
