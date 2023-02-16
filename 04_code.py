# exported from the 04 notebook on 15/02/2023 at 15:01 
# %% [markdown]
# # PriorVAE: testing MMD (and in general different loss functions)
# 
# For a 1D GP

# %%
print("Starting", flush=True)
from jax import random
print("Starting jax", flush=True)

import jax.numpy as jnp

print("Starting jnp", flush=True)
import time
print("Starting time", flush=True)
import dill
print("Starting dill", flush=True)
from flax import serialization

# Numpyro
print("Starting lfa", flush=True)
import numpyro
print("Starting numpyro", flush=True)
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive


# %%
numpyro.set_host_device_count(3)

# %%
print("Starting numpyro infer", flush=True)
from reusable.kernels import esq_kernel, rbf_kernel

print("Loaded", flush=True)

args = {
    # GP prior configuration
    "n": 100,
    "gp_kernel": esq_kernel,
    "rng_key": random.PRNGKey(2),
}
args.update({ # so we can use the definition of n to define x
    
    "x": jnp.arange(0, 1, 1/args["n"]),

    # VAE configuration
    "hidden_dim1": 35,
    "hidden_dim2": 32,
    "latent_dim": 30,
    "vae_var": 0.1,

    "mmd_kernel": lambda x,z: rbf_kernel(x,z, 4.0),

    # learning
    "num_epochs": 100,
    "learning_rate": 1.0e-4,
    "batch_size": 400,
    "train_num_batches": 500,
    "test_num_batches": 1,

    # MCMC parameters
    "num_warmup": 1000,
    "num_samples": 1000,
    "thinning": 1,
    "num_chains": 3,

    "pretrained_vae": False


})

from reusable.util import save_args

save_args("04", args)

print("Saved args", flush=True)

rng_key, _ = random.split(random.PRNGKey(4))

# %% [markdown]
# ## SVI to learn VAE parameters

# %%
if not args["pretrained_vae"]:
    from reusable.gp import OneDGP
    from reusable.train_nn import gen_gp_batches
    print(" IMports for Gen data", flush=True)

    rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)

    train_draws = gen_gp_batches(args["x"], OneDGP, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train)
    print("Done train gen data", flush=True)
    test_draws = gen_gp_batches(args["x"], OneDGP, args["gp_kernel"], 1, args["test_num_batches"]* args["batch_size"], rng_key_test)


print("Gen data", flush=True)

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


# %%
import optax
import jax

from reusable.vae import vae_sample
from flax.core.frozen_dict import freeze
from functools import partial

from reusable.mmd import orig_mmd

@jax.jit
def RCL(y, reconstructed_y, mean, log_sd):
    """reconstruction loss, averaged over the datapoints (not summed)"""
    return  jnp.sum(optax.l2_loss(reconstructed_y, y)) # 1/y.shape[0] *

@jax.jit
def KLD(y, reconstructed_y, mean, log_sd):
    """KL divergence between the distribution N(mean, log_sd) and a standard normal.
    e.g. see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions"""
    return -0.5 * jnp.sum(1 + log_sd - jnp.power(mean, 2) - jnp.exp(log_sd))

@jax.jit
def MMD(y, reconstructed_y, mean, log_sd):
    return orig_mmd(y, reconstructed_y, args["mmd_kernel"])

@jax.jit
def rcl_kld(*args):
    return RCL(*args) + KLD(*args)

@jax.jit
def rcl_kld_50mmd(*args):
    return RCL(*args) + KLD(*args) + 50* MMD(*args)

@jax.jit
def kld_50mmd(*args):
    return 50* MMD(*args) + KLD(*args)


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
        "train_mmd": orig_mmd(vae_draws, train_samples[-1], args["mmd_kernel"]),
        "test_mmd": orig_mmd(vae_draws, test_samples[-1], args["mmd_kernel"]),
        "train_kld": KLD(*train_output),
        "test_kld": KLD(*test_output),
        "train_rcl": RCL(*train_output),
        "test_rcl": RCL(*test_output)
    }

    return metrics


# %%
from reusable.train_nn import run_training
import matplotlib.pyplot as plt
from plotting.plots import plot_training

from reusable.vae import vae_sample
from plotting.plots import compare_draws
import jax.random as random
from numpyro.infer import Predictive
from reusable.gp import OneDGP
from reusable.util import decoder_filename, get_savepath



_, rng_key_predict = random.split(random.PRNGKey(2))

plot_gp_predictive = Predictive(OneDGP, num_samples=1000)
gp_draws = plot_gp_predictive(rng_key_predict, x=args["x"], gp_kernel = args["gp_kernel"], jitter=1e-5)['y']

print("Starting training", flush=True)

saved_states  = {}
if not args["pretrained_vae"]:
    loss_fns = [rcl_kld_50mmd, kld_50mmd, rcl_kld]
    for loss_fn in loss_fns:
        print( loss_fn.__name__, flush=True)
        final_state, metrics_history = run_training(
            loss_fn, compute_epoch_metrics, args["num_epochs"], train_draws, test_draws, state
        )
        saved_states[loss_fn.__name__] = final_state
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(24, 4))
        for i,t in enumerate(["loss", "mmd", "kld", "rcl"]):

            plot_training(
                jnp.array(metrics_history["test_"+t]).flatten(),
                jnp.array(metrics_history["train_"+t]).flatten(),
                f"Test/train {t} for " + loss_fn.__name__,
                t,
                axs[i],
            )
        plt.show()
        print(metrics_history)
        plot_vae_predictive = Predictive(vae_sample, num_samples=1000)
        vae_draws = plot_vae_predictive(
            rng_key_predict,
            hidden_dim1=args["hidden_dim1"],
            hidden_dim2=args["hidden_dim2"],
            latent_dim=args["latent_dim"],
            out_dim=args["n"],
            decoder_params=freeze({"params": final_state.params["VAE_Decoder_0"]}),
        )["f"]
            


        compare_draws(args["x"], gp_draws, vae_draws, "GP priors we want to encode", "Priors learnt by VAE w/ loss" + loss_fn.__name__, '$y=f_{GP}(x)$', '$y=f_{VAE}(x)$')
        plt.show()

        file_path = f'{get_savepath()}/{decoder_filename("04", args, suffix=loss_fn.__name__)}'

        if not args["pretrained_vae"]:
            decoder_params = state.params["VAE_Decoder_0"]

            decoder_params = freeze({"params": decoder_params})
            args["decoder_params"] = decoder_params
            with open(file_path, 'wb') as file:
                file.write(serialization.to_bytes(decoder_params))

        file_path = f'{get_savepath()}/{decoder_filename("04", args, suffix=loss_fn.__name__+"_metrics_hist")}'

        with open(file_path, 'wb') as file:
            dill.dump(metrics_history, file)


