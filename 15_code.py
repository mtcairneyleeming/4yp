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
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from reusable.data import gen_gp_batches
from reusable.gp import OneDGP_UnifLS
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
from reusable.vae import VAE, cvae_length_mcmc, cvae_sample

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
coords = jnp.dstack((x_coords, y_coords))

del x_coords, y_coords

args = {
    "expcode": 15,
    "dim": 2,
    "gp_kernel": esq_kernel,
    "rng_key": random.PRNGKey(2),
    "x": coords/ 1e6,
    "conditional": True,
    # VAE configuration
    "hidden_dim1": 35,
    "hidden_dim2": 32,
    "latent_dim": 50,
    "vae_var": 0.1,
    # learning
    "num_epochs": 1000,
    "learning_rate": 1.0e-3,
    "batch_size": 400,
    "train_num_batches": 400,
    "test_num_batches": 5,
    # MCMC parameters
    "num_warmup": 4000,
    "num_samples": 4000,
    "thinning": 1,
    "num_chains": 3,
    "num_samples_to_save": 4000,
    "rng_key_ground_truth": random.PRNGKey(4),
}


save_args(args["expcode"], "args", args)


pre_generated_data = sys.argv[1] = "load_generated"


rng_key, _ = random.split(random.PRNGKey(4))


rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)
# generate a complete set of training and test data


if not pre_generated_data:

    # NOTE changed draw_access - y_c is [y,u] for this
    train_draws = gen_gp_batches(
        args["x"],
        OneDGP_UnifLS,
        args["gp_kernel"],
        args["train_num_batches"],
        args["batch_size"],
        rng_key_train,
        draw_access="y_c",
        jitter=5e-5,
    )
    test_draws = gen_gp_batches(
        args["x"],
        OneDGP_UnifLS,
        args["gp_kernel"],
        1,
        args["test_num_batches"] * args["batch_size"],
        rng_key_test,
        draw_access="y_c",
        jitter=5e-5,
    )
    save_datasets(
        args["expcode"], gen_file_name(args["expcode"], args, "raw_gp", False, ["num_epochs"]), train_draws, test_draws
    )

else:
    train_draws, test_draws = load_datasets(
        args["expcode"], gen_file_name(args["expcode"], args, "raw_gp", False, ["num_epochs"])
    )


rng_key, rng_key_init, rng_key_init_state, rng_key_train, rng_key_shuffle = random.split(rng_key, 5)

module = VAE(
    hidden_dim1=args["hidden_dim1"],
    hidden_dim2=args["hidden_dim2"],
    latent_dim=args["latent_dim"],
    out_dim=args["n"] ** args["dim"],
    conditional=True,
)
params = module.init(rng_key_init, jnp.ones((args["batch_size"], args["n"] ** args["dim"] + 1,)))[
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


def run_mcmc_cvae(rng_key, model_mcmc, y_obs, c=None, verbose=False):
    start = time.time()

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model_mcmc, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False,
    )
    mcmc.run(
        rng_key,
        y=y_obs,
        length=c,
    )
    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    print("\nMCMC elapsed time:", time.time() - start)

    return mcmc.get_samples()


# fixed to generate a "ground truth" GP we will try and infer

ground_truth_predictive = Predictive(OneDGP_UnifLS, num_samples=1)
gt_draws = ground_truth_predictive(
    args["rng_key_ground_truth"], x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5, noise=True, length=0.05
)
ground_truth = gt_draws["f"][0]
ground_truth_y_draw = gt_draws["y"][0]

obs_idx = jnp.arange(0, args["x"].shape[0])  # i.e. all of them

obs_mask = jnp.isin(jnp.arange(0, args["n"] ** args["dim"]), obs_idx, assume_unique=True)


ground_truth_y_obs = ground_truth_y_draw[obs_idx]
x_obs = jnp.arange(0, args["n"] ** args["dim"])[obs_idx]


rng_key, rng_key_all_mcmc, rng_key_true_mcmc = random.split(rng_key, 3)

# hidden_dim1, hidden_dim2, latent_dim, out_dim, decoder_params,  obs_idx=None, length_prior_choice

mcmc_samples = run_mcmc_cvae(
    rng_key_true_mcmc,
    cvae_length_mcmc(
        args["hidden_dim1"],
        args["hidden_dim2"],
        args["latent_dim"],
        args["n"] ** args["dim"],
        args["decoder_params"],
        obs_idx,
        "invgamma",
    ),
    ground_truth_y_obs,
    c=1,
)
save_samples(args["expcode"], gen_file_name(args["expcode"], args, "inference_true_ls_mcmc"), mcmc_samples)

mcmc_samples = run_mcmc_cvae(
    rng_key_all_mcmc,
    cvae_length_mcmc(
        args["hidden_dim1"],
        args["hidden_dim2"],
        args["latent_dim"],
        args["n"] ** args["dim"],
        args["decoder_params"],
        obs_idx,
        "invgamma",
    ),
    ground_truth_y_obs,
    c=None,
)
save_samples(args["expcode"], gen_file_name(args["expcode"], args, "inference_all_ls_mcmc"), mcmc_samples)
