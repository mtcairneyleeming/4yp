"""
Quick comparisons to check if shuffling the training data is important 



"""
import time
import sys

index = int(sys.argv[1])

print(f"Starting experiment  12, index {index}", flush=True)


import jax.numpy as jnp
import jax.random as random
import optax
from jax import random

from reusable.data import gen_gp_batches
from reusable.gp import OneDGP
from reusable.kernels import esq_kernel
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD, MMD_rqk
from reusable.train_nn import SimpleTrainState, run_training, run_training_shuffle
from reusable.util import save_args, save_training, setup_signals, gen_file_name
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
        "test_num_batches": 20
        
    }
)

loss_fns = [combo_loss(RCL, KLD), 
            combo3_loss(RCL, KLD, MMD_rbf(4.0), 0.01, 1, 10),
            combo3_loss(RCL, KLD, MMD_rbf(1.0), 0.01, 1, 10),
            combo3_loss(RCL, KLD, MMD_rqk(1.0, 0.1), 0.01, 1, 10),
            combo3_loss(RCL, KLD, MMD_rqk(1.0, 1.0), 0.01, 1, 10),
            combo3_loss(RCL, KLD, MMD_rqk(1.0, 10.0), 0.01, 1, 10),
            ]

args["loss_fns"] = [l.__name__ for l in loss_fns]

save_args("12", "12", args)

print("Saved args", flush=True)


rng_key, rng_key_train, rng_key_test = random.split(args["rng_key"], 3)

start_time = time.time()

train_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train
)
test_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test
)


data_time = time.time()

print("Generated data", flush=True)

rng_key, rng_key_init, rng_key_train, rng_key_shuffle = random.split(rng_key, 4)

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

init_time = time.time()

print("Starting training", flush=True)

name = loss_fns[index].__name__

final_state, metrics_history = run_training(
    loss_fns[index], None, args["num_epochs"], train_draws, test_draws, state
)

save_training("12", gen_file_name("12", args, "standard"), final_state, metrics_history)
#### shuffled




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

final_state, metrics_history = run_training_shuffle(
    loss_fns[index], None, args["num_epochs"], train_draws, test_draws, state, rng_key_shuffle
)

save_training("12", gen_file_name("12", args, "shuffle"), final_state, metrics_history)