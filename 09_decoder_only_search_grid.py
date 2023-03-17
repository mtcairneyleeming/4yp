"""
Decoder only network - search grid"""

print("Starting", flush=True)
from jax import random

import jax.numpy as jnp
import dill
from flax import serialization
import numpyro
from numpyro.infer import Predictive


numpyro.set_host_device_count(3)

from reusable.kernels import esq_kernel, rbf_kernel

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
    
    }
)


rng_key = args["rng_key"]


from reusable.gp import OneDGP
from reusable.data import gen_gp_batches, gen_latent_batches, pair_batched_data

rng_key, rng_key_train, rng_key_test, rng_key_latent_train, rng_key_latent_test = random.split(rng_key, 5)

train_gp_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train
)
train_latent_draws = gen_latent_batches(
    args["latent_dim"], args["train_num_batches"], args["batch_size"], rng_key_latent_train
)
test_gp_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test
)
test_latent_draws = gen_latent_batches(
    args["latent_dim"], 1, args["test_num_batches"] * args["batch_size"], rng_key_latent_test
)


train_draws = pair_batched_data(train_gp_draws, train_latent_draws)
test_draws = pair_batched_data(test_gp_draws, test_latent_draws)

print("Generated data", flush=True)

from reusable.vae import Single_Decoder
from reusable.train_nn import SimpleTrainState
import optax

rng_key, rng_key_init, rng_key_train = random.split(rng_key, 3)

module = Single_Decoder(hidden_dim1=args["hidden_dim1"], hidden_dim2=args["hidden_dim2"], out_dim=args["n"])
params = module.init(rng_key, jnp.ones((args["n"] + args["latent_dim"],)))[
    "params"
]  # initialize parameters by passing a template vector
tx = optax.adam(args["learning_rate"])
state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)


from flax.core.frozen_dict import freeze



from reusable.train_nn import run_training


import jax.random as random
from reusable.util import decoder_filename, __get_savepath


from reusable.loss import MMD_rqk

print("Starting training", flush=True)

args["description"] = "MMD_rq:"
args["l_range"] = [0.5, 1, 5, 10, 25, 50]
args["a_range"] = [0.01, 0.1, 0.5, 1, 2, 10, 25, 100]

ar = len(args["a_range"])
lr = len(args["l_range"])

print(ar, lr, ", total=", ar * lr)

import sys

index = int(sys.argv[1])

l = index // ar
a = index % ar

print(l, a)



loss_fn = MMD_rqk(args["l_range"][l], args["a_range"][a])


print(loss_fn.__name__, flush=True)
final_state, metrics_history = run_training(
    loss_fn, lambda *_: {}, args["num_epochs"], train_draws, test_draws, state
)

from reusable.util import gen_file_name, save_training, save_args

save_training("09", gen_file_name("09", args, loss_fn.__name__))

save_args(f"09", "09", args)

