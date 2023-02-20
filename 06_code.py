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
        "batch_size": 200,
        "train_num_batches": 500,
        "test_num_batches": 1,
        # MCMC parameters
        "num_warmup": 1000,
        "num_samples": 1000,
        "thinning": 1,
        "num_chains": 3,
        "pretrained_vae": False,

        "mmd_rbf_ls": 4.0,
        "mmd_rq_ls": 4.0,
        "mmd_rq_scale": 1 # TODO: JUSTIFY?
    }
)


rng_key, _ = random.split(random.PRNGKey(4))


from reusable.gp import OneDGP
from reusable.data import gen_gp_batches

rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)

train_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train
)

test_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test
)


print("Generated data", flush=True)

# %%
from reusable.vae import VAE
from reusable.train_nn import SimpleTrainState
import optax

rng_key, rng_key_init, rng_key_train = random.split(rng_key, 3)

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


import optax
import jax

from reusable.vae import vae_sample
from flax.core.frozen_dict import freeze

from reusable.mmd import mmd_matrix_impl
from reusable.kernels import rbf_kernel, rq_kernel


@jax.jit
def RCL(y, reconstructed_y, mean, log_sd):
    """reconstruction loss, averaged over the datapoints (not summed)"""
    return jnp.mean(optax.l2_loss(reconstructed_y, y))  # 1/y.shape[0] *


@jax.jit
def KLD(y, reconstructed_y, mean, log_sd):
    """KL divergence between the distribution N(mean, log_sd) and a standard normal.
    e.g. see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions"""
    return -0.5 * jnp.mean(1 + log_sd - jnp.power(mean, 2) - jnp.exp(log_sd))


@jax.jit
def MMD_rbf(y, reconstructed_y, mean, log_sd):
    return mmd_matrix_impl(y, reconstructed_y, lambda x, z: rbf_kernel(x, z, args["mmd_rbf_ls"]), normalise=True)

def MMD_rqk(y, reconstructed_y, mean, log_sd):
    return mmd_matrix_impl(y, reconstructed_y, lambda x, z: rq_kernel(x, z, args["mmd_rq_ls"], args["mmd_rq_scale"]), normalise=True)

def gridsearch_rcl_mmd_rbf_w_kld(rcl_factor, mmd_factor):
    @jax.jit
    def func(*args):
        return KLD(*args) + mmd_factor * MMD_rbf(*args) + rcl_factor * RCL(*args)
    func.__name__ = f"kld_{mmd_factor}mmd_rbf_{rcl_factor}rcl"

    return func

def gridsearch_rcl_mmd_rq_w_kld(rcl_factor, mmd_factor):
    @jax.jit
    def func(*args):
        return KLD(*args) + mmd_factor * MMD_rqk(*args) + rcl_factor * RCL(*args)
    func.__name__ = f"kld_{mmd_factor}mmd_rq_{rcl_factor}rcl"

    return func


def compute_epoch_metrics(final_state: SimpleTrainState, test_samples, train_samples, train_output, test_output):
    print("epoch done", flush=True)
    current_metric_key = jax.random.fold_in(key=final_state.key, data=2 * final_state.step + 1)

    vae_draws = Predictive(vae_sample, num_samples=args["batch_size"])(
        current_metric_key,
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        out_dim=args["n"],
        decoder_params=freeze({"params": final_state.params["VAE_Decoder_0"]}),
    )["f"]

    metrics = {
        "train_mmd_rbf": MMD_rbf(*train_output),
        "test_mmd_rbf": MMD_rbf(*test_output),
        "train_mmd_rqk": MMD_rqk(*train_output),
        "test_mmd_rqk": MMD_rqk(*test_output),
        "train_mmd_rbf_new_draws": MMD_rbf(vae_draws, train_samples[-1], 0, 0), # ignore 0s, just there to satisfy extra arguments
        "test_mmd_rbf_new_draws": MMD_rbf(vae_draws, test_samples[-1], 0,0),
        "train_kld": KLD(*train_output),
        "test_kld": KLD(*test_output),
        "train_rcl": RCL(*train_output),
        "test_rcl": RCL(*test_output),
    }

    return metrics


from reusable.train_nn import run_training


from reusable.vae import vae_sample
import jax.random as random
from numpyro.infer import Predictive
from reusable.gp import OneDGP
from reusable.util import decoder_filename, get_savepath


_, rng_key_predict = random.split(random.PRNGKey(2))

plot_gp_predictive = Predictive(OneDGP, num_samples=1000)
gp_draws = plot_gp_predictive(rng_key_predict, x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5)["y"]

print("Starting training", flush=True)

args["description"] = "KLD+ s * MMD_rbf + t * RCL"
args["s_range"] = [1, 10, 25, 50]
args["t_range"] = [0.01, 0.1, 0.5, 1]

sr = len(args["s_range"])
tr = len(args["t_range"])

print(sr, tr, ", total=", sr * tr)

import sys

index = int(sys.argv[1])

s = index % tr
t = index // tr

loss_fn = gridsearch_rcl_mmd_rbf_w_kld(s, t)


print(loss_fn.__name__, flush=True)
final_state, metrics_history = run_training(
    loss_fn, compute_epoch_metrics, args["num_epochs"], train_draws, test_draws, state
)

with open(f'{get_savepath()}/{decoder_filename("06", args, suffix=loss_fn.__name__)}', "wb") as file:
    file.write(serialization.to_bytes(freeze({"params": final_state.params["VAE_Decoder_0"]})))

with open(
    f'{get_savepath()}/{decoder_filename("06", args, suffix=loss_fn.__name__+"_metrics_hist")}', "wb"
) as file:
    dill.dump(metrics_history, file)


from reusable.util import save_args

save_args("06", args)

print("Saved args", flush=True)
