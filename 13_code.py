"""
Experiments to determine scalability of just PriorVAE, to high 
"""
import time
import sys

index = int(sys.argv[1])

print(f"Starting 13, index {index}", flush=True)


import jax.numpy as jnp
import jax.random as random
import optax
from jax import random

from reusable.data import gen_gp_batches
from reusable.gp import OneDGP
from reusable.kernels import esq_kernel
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD
from reusable.train_nn import SimpleTrainState, run_training
from reusable.util import save_args, save_training, setup_signals, update_args_11, gen_file_name
from reusable.vae import VAE

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
        "num_epochs": 200,
        "learning_rate": 1.0e-4,
        "batch_size": 400,
        "train_num_batches": 300,
        "test_num_batches": 20,
        "mmd_rbf_ls": 4.0,


        "Arange": [100, 200, 400, 800, 1600, 3200, 6400],
        "Brange": [100, 200, 400, 800],
        "Adesc": "n",
        "Bdesc": "train_num_batches",
        "jitter_scaling": 1/300 * 4e-6  # n times this gives the jitter
    }
)

save_args("13", "13", args)

print("Saved args", flush=True)

Arange = args["Arange"]
Brange = args["Brange"]

ar = len(Arange)
br = len(Brange)

b = index // ar
a = index % ar

print(f"Exp  13 a={a}/{ar-1}, b={b}/{br-1}, index={index}/{ar*br-1} [indices 0-(n-1)]")

update_args_11(args, args,  a, b)

args["x"] = jnp.arange(0, 1, 1 / args["n"])  # if we have changed it!

loss_fn = combo_loss(RCL, KLD)

rng_key, rng_key_train, rng_key_test = random.split(args["rng_key"], 3)

start_time = time.time()

train_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train, jitter=args["n"] * args["jitter_scaling"]
)
test_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test, jitter=args["n"] * args["jitter_scaling"]
)


print("Generated data", flush=True)

rng_key, rng_key_init, rng_key_train = random.split(rng_key, 3)

module = VAE(
    hidden_dim1=args["hidden_dim1"],
    hidden_dim2=args["hidden_dim2"],
    latent_dim=args["latent_dim"],
    out_dim=args["n"],
    conditional=False,
)
print("Init", args["n"], train_draws.shape)
params = module.init(rng_key, jnp.ones((args["n"],)))["params"]  # initialize parameters by passing a template image
tx = optax.adam(args["learning_rate"])
state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

print("Starting training", flush=True)



final_state, metrics_history = run_training(
    loss_fn, lambda *_: {}, args["num_epochs"], train_draws, test_draws, state
)

save_training("13", gen_file_name("13", args, f"{loss_fn.__name__}_{index}"), final_state, metrics_history)
