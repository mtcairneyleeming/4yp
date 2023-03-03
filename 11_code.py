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
import time
import sys

index = int(sys.argv[1])
experiment = sys.argv[2]

print(f"Starting experiment {experiment}, index {index}", flush=True)


import dill
import jax.numpy as jnp
import jax.random as random
import optax
from flax import serialization
from flax.core.frozen_dict import freeze
from jax import random

from reusable.data import gen_gp_batches
from reusable.gp import OneDGP
from reusable.kernels import esq_kernel
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD
from reusable.train_nn import SimpleTrainState, run_training
from reusable.util import decoder_filename, get_savepath, save_args, setup_signals
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
        "num_epochs": 150,
        "learning_rate": 1.0e-4,
        "batch_size": 400,
        "train_num_batches": 100,
        "test_num_batches": 1,
        "mmd_rbf_ls": 4.0,
        "11_exp1": {
            "Arange": [25, 50, 100, 150, 200, 225, 250],
            "Brange": [50, 75, 100, 125, 150, 175, 200],
            "Adesc": "n",
            "Bdesc": "train_num_batches",
        },
        "11_exp2": {
            "Arange": [
                50,
                100,
                200,
                400,
                500,
                600,
                800,
            ],  # idea is diagonal is the same amount of data as current: 40,000
            "Brange": [50, 67, 80, 100, 200, 400, 800],
            "Adesc": "batch_size",
            "Bdesc": "train_num_batches",
        },
        "11_exp3": {
            "Arange": [25, 50, 100, 150, 200, 225, 250],
            "Brange": [50, 100, 150, 200, 250, 300, 350, 400],
            "Adesc": "n",
            "Bdesc": "num_epochs",
        },
        "11_exp4": {
            "Arange": [25, 50, 100, 150, 200, 225, 250],
            "Brange": [0.25, 0.5, 1, 1.5, 2, 3, 4],
            "Adesc": "n",
            "Bdesc": "vae_scale_factor",
        },
        "experiment": experiment,
    }
)

save_args(experiment, args)

print("Saved args", flush=True)

Arange = args[experiment]["Arange"]
Brange = args[experiment]["Brange"]

ar = len(Arange)
br = len(Arange)

b = index // ar
a = index % ar

print(f"Exp {experiment}, a={a}/{ar-1}, b={b}/{br-1}, index={index}/{ar*br-1} [indices 0-(n-1)]")

match experiment:
    case "11_exp1":
        args["n"] = Arange[a]
        args["train_num_batches"] = Brange[b]
    case "11_exp2":
        args["batch_size"] = Arange[a]
        args["train_num_batches"] = Brange[b]
    case "11_exp3":
        args["n"] = Arange[a]
        # will save at intermediate stops
    case "11_exp4":
        args["n"] = Arange[a]
        args["hidden_dim1"] = int(Brange[b] * 35)
        args["hidden_dim2"] = int(Brange[b] * 32)
        args["latent_dim"] = int(Brange[b] * 30)
    case _:
        raise NotImplementedError(f"Help! {experiment}")

args["x"] = jnp.arange(0, 1, 1 / args["n"])  # if we have changed it!

loss_fns = [combo_loss(RCL, KLD), combo3_loss(RCL, KLD, MMD_rbf(args["mmd_rbf_ls"]), 0.01, 1, 10)]

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

init_time = time.time()

print("Starting training", flush=True)



for loss_fn in loss_fns:
    name = f"{loss_fn.__name__}_{experiment}_{index}"

    print(name, flush=True)
    if experiment != "11_exp3":
        # timing is now automatic in run_training

        final_state, metrics_history = run_training(
            loss_fn, lambda *_: {}, args["num_epochs"], train_draws, test_draws, state
        )

        with open(f'{get_savepath()}/{decoder_filename("11", args, suffix=name)}', "wb") as file:
            file.write(serialization.to_bytes(freeze({"params": final_state.params})))

        with open(f'{get_savepath()}/{decoder_filename("11", args, suffix=name+"_metrics_hist")}', "wb") as file:
            dill.dump(metrics_history, file)
        print(f"Saved {name}")
    else:
        prev_history = {}
        for i, _ in enumerate(Brange):
            name = f"{loss_fn.__name__}_{experiment}_{a}_{i}"
            next_range = Brange[i] - Brange[i - 1] if i > 0 else Brange[0]

            state, h = run_training(loss_fn, lambda *_: {}, next_range, train_draws, test_draws, state)

            with open(f'{get_savepath()}/{decoder_filename("11", args, suffix=name)}', "wb") as file:
                file.write(serialization.to_bytes(freeze({"params": state.params})))

            if i > 0:
                for metric, value in h.items():
                    prev_history[metric] = jnp.append(prev_history[metric], value, axis=0)
            else:
                prev_history = h

            with open(f'{get_savepath()}/{decoder_filename("11", args, suffix=name+"_metrics_hist")}', "wb") as file:
                dill.dump(prev_history, file)

            print(f"Saved {name}")

            if "interrupted" in metrics_history:
                print("SIGTERM sent, not iterating")
                break


