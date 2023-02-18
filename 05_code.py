# exported from the 04 notebook on 15/02/2023 at 15:01

# # PriorVAE: testing MMD (and in general different loss functions)
#
# For a 1D GP

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

# Loss functions


def MMD_rbf_customls(ls):
    @jax.jit
    def MMD_rbf(y, reconstructed_y):
        return mmd_matrix_impl(y, reconstructed_y, lambda x, z: rbf_kernel(x, z, ls))

    MMD_rbf.__name__ = "MMD_rbf_" + str(ls)

    return MMD_rbf


def MMD_rql_custom(ls, scale):
    @jax.jit
    def MMD_rqk(y, reconstructed_y):
        return mmd_matrix_impl(y, reconstructed_y, lambda x, z: rq_kernel(x, z, ls, scale))

    MMD_rqk.__name__ = f"MMD_rqk_{ls}_{scale}"

    return MMD_rqk


@jax.jit
def MMD_rbf_ls_1_2_4_16_32(y, reconstructed_y):
    return mmd_matrix_impl(
        y,
        reconstructed_y,
        lambda x, z: rbf_kernel(x, z, 1.0)
        + rbf_kernel(x, z, 2.0)
        + rbf_kernel(x, z, 4.0)
        + rbf_kernel(x, z, 16.0)
        + rbf_kernel(x, z, 32.0),
    )


from reusable.train_nn import run_training


import jax.random as random
from numpyro.infer import Predictive
from reusable.gp import OneDGP
from reusable.util import decoder_filename, get_savepath


print("Starting training", flush=True)


loss_fns = (
    [MMD_rbf_customls(l) for l in [0.5, 1, 2, 8, 16]]
    + [MMD_rbf_ls_1_2_4_16_32]
    + [MMD_rql_custom(1, l ) for l in [0.25, 0.5, 1, 8, 16]]
    + [MMD_rql_custom(4, l) for l in [0.25, 0.25, 1, 8, 16]]
)
args["loss_functions"] = [x.__name__ for x in loss_fns]


for loss_fn in loss_fns:
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

save_args("05", args)

print("Saved args", flush=True)

# %%
