"""
PriorCVAE on the CAR dataset

"""

import time
import sys

import jax.numpy as jnp

import geopandas as gpd

# Numpyro
import numpyro
import optax
from jax import random
from numpyro.infer import Predictive

from reusable.data import gen_gp_batches
from reusable.gp import BuildGP
from reusable.kernels import esq_kernel
from reusable.loss import KLD, RCL, combo_loss, conditional_loss_wrapper
from reusable.train_nn import SimpleTrainState, run_training_shuffle
from reusable.util import (
    save_samples,
    save_datasets,
    save_training,
    gen_file_name,
    save_args,
    load_datasets,
    get_decoder_params,
)
from reusable.vae import VAE
from reusable.mcmc import cvae_length_mcmc, run_mcmc

numpyro.set_host_device_count(4)


s = gpd.read_file("data/zwe2016phia.geojson")

s = s[["area_id", "geometry", "y", "n_obs", "estimate"]]
s["y"] = round(s["y"]).astype(int)
s["n_obs"] = round(s["n_obs"]).astype(int)

temp_centroids = s["geometry"].to_crs("EPSG:32735").centroid
centroids = gpd.GeoDataFrame()
centroids["x"] = temp_centroids.geometry.apply(lambda x: x.x)
centroids["y"] = temp_centroids.geometry.apply(lambda x: x.y)
x_coords = jnp.array(centroids["x"])
y_coords = jnp.array(centroids["y"])
x_coords = x_coords - jnp.mean(x_coords)
y_coords = y_coords - jnp.mean(y_coords)
coords = jnp.dstack((x_coords, y_coords))[0]

del x_coords, y_coords

args = {
    "expcode": 15,
    "dim": 2,
    "gp_kernel": esq_kernel,
    "rng_key": random.PRNGKey(2),
    "x": coords / 1e6,
    "conditional": True,
    # VAE configuration
    "hidden_dim1": 40,
    "hidden_dim2": 40,
    "latent_dim": 20,
    "vae_var": 0.1,
    # learning
    "num_epochs": 200,
    "learning_rate": 1.0e-3,
    "batch_size": 400,
    "train_num_batches": 800,
    "test_num_batches": 20,
    # MCMC parameters
    "num_warmup": 16000,
    "num_samples": 20000,
    "thinning": 1,
    "num_chains": 3,
    "num_samples_to_save": 4000,
    "rng_key_ground_truth": random.PRNGKey(4),
    "length_prior_choice": "lognormal",
    "length_prior_arguments": {"location": -1.3558, "scale": 0.5719},
}


pre_generated_data = len(sys.argv) >= 2 and sys.argv[1] == "load_generated"

use_gp = len(sys.argv) >= 2 and sys.argv[1] == "use_gp"

save_args(args["expcode"], "gp" if use_gp else "v6", args)


rng_key = random.PRNGKey(4)
rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)


gp = BuildGP(args["gp_kernel"], 8e-6, args["length_prior_choice"], args["length_prior_arguments"])

if not pre_generated_data and not use_gp:

    # NOTE changed draw_access - y_c is [y,u] for this
    train_draws = gen_gp_batches(
        args["x"],
        gp,
        args["gp_kernel"],
        args["train_num_batches"],
        args["batch_size"],
        rng_key_train,
        draw_access="y_c",
        jitter=5e-5,
    )
    test_draws = gen_gp_batches(
        args["x"],
        gp,
        args["gp_kernel"],
        1,
        args["test_num_batches"] * args["batch_size"],
        rng_key_test,
        draw_access="y_c",
        jitter=5e-5,
    )
    save_datasets(
        args["expcode"],
        gen_file_name(
            args["expcode"],
            args,
            "raw_gp",
            False,
            ["num_epochs", "hidden_dim1", "hidden_dim2", "latent_dim", "vae_var", "learning_rate"],
        ),
        train_draws,
        test_draws,
    )

elif not use_gp:
    train_draws, test_draws = load_datasets(
        args["expcode"],
        gen_file_name(
            args["expcode"],
            args,
            "raw_gp",
            False,
            ["num_epochs", "hidden_dim1", "hidden_dim2", "latent_dim", "vae_var", "learning_rate"],
        ),
    )


rng_key, rng_key_init, rng_key_init_state, rng_key_train, rng_key_shuffle = random.split(rng_key, 5)

if not use_gp:

    module = VAE(
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        out_dim=args["x"].shape[0],
        conditional=True,
    )
    params = module.init(rng_key_init, jnp.ones((args["batch_size"], args["x"].shape[0] + 1,)))[
        "params"
    ]  # initialize parameters by passing a template image
    tx = optax.adam(args["learning_rate"])
    state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init_state)

    state, metrics_history = run_training_shuffle(
        conditional_loss_wrapper(combo_loss(RCL, KLD)),
        lambda *_: {},
        args["num_epochs"],
        train_draws,
        test_draws,
        state,
        rng_key_shuffle,
    )

    args["decoder_params"] = get_decoder_params(state)

    save_training(args["expcode"], gen_file_name(args["expcode"], args), state, metrics_history)


ground_truth = jnp.array(s["estimate"])


rng_key, rng_key_mcmc = random.split(rng_key, 2)

# hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params,  obs_idx=None, length_prior_choice

f = (
    gp
    if use_gp
    else cvae_length_mcmc(
        args["hidden_dim1"],
        args["hidden_dim2"],
        args["latent_dim"],
        args["decoder_params"],
        args["length_prior_choice"],
        args["length_prior_arguments"],
    )
)

label = "gp" if use_gp else "cvae"

mcmc_samples = run_mcmc(
    args["num_warmup"],
    args["num_samples"],
    args["num_chains"],
    rng_key_mcmc,
    f,
    args["x"],
    ground_truth,
    jnp.arange(args["x"].shape[0]),
    condition=None,
    verbose=True,
)
save_samples(
    args["expcode"], gen_file_name(args["expcode"], args, f"inference_{label}_mcmc", include_mcmc=True), mcmc_samples
)
