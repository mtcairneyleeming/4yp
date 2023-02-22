"""
Decoder only network"""

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
        # MCMC parameters
        "num_warmup": 1000,
        "num_samples": 1000,
        "thinning": 1,
        "num_chains": 3,
        "pretrained_vae": False,
        "mmd_rbf_ls": 4.0,
        "mmd_rq_ls": 4.0,
        "mmd_rq_scale": 1,  # TODO: JUSTIFY?
    }
)


rng_key, _ = random.split(random.PRNGKey(4))


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


import optax
import jax

from reusable.vae import vae_sample
from flax.core.frozen_dict import freeze

from reusable.mmd import mmd_matrix_impl
from reusable.kernels import rbf_kernel, rq_kernel



from reusable.train_nn import run_training


import jax.random as random
from reusable.util import decoder_filename, get_savepath


from reusable.loss import MMD_rbf, MMD_rqk, MMD_rbf_sum, MMD_rqk_sum

print("Starting training", flush=True)

# Run 1
# loss_fns = (
#     [MMD_rbf(l) for l in [0.5, 1, 2, 8, 16]]
#     + [MMD_rbf_sum([1,2,4,16,32])]
#     + [MMD_rqk(1, l ) for l in [0.25, 0.5, 1, 8, 16]]
#     + [MMD_rqk(4, l) for l in [0.25, 0.5, 1, 8, 16]]
# )

#Run 2
loss_fns = (
    [MMD_rbf(l) for l in [4, 6, 8, 10, 12]]
    + [MMD_rbf_sum([0.1, 0.25, 0.5, 1,2,4,16,32])]
    + [MMD_rqk(l, 0.25 ) for l in [0.25, 0.5, 1, 4, 8, 16]]
    + [MMD_rqk(l, 0.1) for l in [0.25, 0.5, 1, 4, 8, 16]]
)
args["loss_functions"] = [x.__name__ for x in loss_fns]


import sys

index = int(sys.argv[1])
loss_fn = loss_fns[index]
print(loss_fn.__name__, flush=True)
final_state, metrics_history = run_training(
    loss_fn, lambda *_: {}, args["num_epochs"], train_draws, test_draws, state
)

with open(f'{get_savepath()}/{decoder_filename("05", args, suffix=loss_fn.__name__)}', "wb") as file:
    file.write(serialization.to_bytes(freeze({"params": final_state.params})))

with open(
    f'{get_savepath()}/{decoder_filename("05", args, suffix=loss_fn.__name__+"_metrics_hist")}', "wb"
) as file:
    dill.dump(metrics_history, file)


from reusable.util import save_args

# might need to depend on job in the future!
save_args(f"05", args)

print("Saved args", flush=True)
