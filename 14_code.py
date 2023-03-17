"""
Do PriorCVAE on a 50x50 grid

train on GP prior, with lengthscale uniform in 0.01, 0.5

test on GP with true ls 0.05

"""

import time
import sys

import jax.numpy as jnp

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

args = {
    "expcode": 14,
    # GP prior configuration
    "n": 50,
    "dim": 2,
    "gp_kernel": esq_kernel,
    "rng_key": random.PRNGKey(2),
}
args.update(
    {
        "axis_range": jnp.arange(0, 1, 1 / args["n"]),
    }
)
args.update(
    {
        "grid_x": jnp.array(jnp.meshgrid(*([args["axis_range"]] * args["dim"]))).T.reshape(
            *([args["n"]] * args["dim"]), args["dim"]
        )
    }
)
args.update(
    {  # so we can use the definition of n to define x
        "x": jnp.reshape(args["grid_x"], (-1, args["dim"])),
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
)

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


def run_mcmc_cvae(rng_key, model_mcmc, y_obs, obs_idx, c=None, verbose=False):
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
        args["hidden_dim1"],
        args["hidden_dim2"],
        args["latent_dim"],
        args["n"] ** args["dim"],
        args["decoder_params"],
        y=y_obs,
        obs_idx=obs_idx,
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

obs_idx = jnp.array(
    [
        2,
        7,
        24,
        28,
        33,
        34,
        37,
        48,
        49,
        55,
        61,
        69,
        71,
        75,
        84,
        102,
        103,
        115,
        116,
        123,
        141,
        145,
        157,
        185,
        187,
        188,
        193,
        197,
        208,
        227,
        228,
        229,
        233,
        235,
        260,
        271,
        280,
        290,
        304,
        316,
        317,
        321,
        331,
        343,
        346,
        349,
        352,
        358,
        368,
        369,
        372,
        378,
        380,
        390,
        394,
        408,
        424,
        442,
        450,
        452,
        454,
        462,
        483,
        489,
        511,
        518,
        522,
        524,
        548,
        551,
        555,
        560,
        565,
        566,
        567,
        580,
        582,
        593,
        601,
        611,
        613,
        616,
        618,
        619,
        645,
        651,
        654,
        655,
        656,
        665,
        666,
        676,
        690,
        695,
        709,
        729,
        732,
        744,
        749,
        767,
        794,
        796,
        813,
        815,
        841,
        843,
        844,
        845,
        852,
        863,
        866,
        870,
        872,
        877,
        878,
        883,
        901,
        902,
        904,
        909,
        923,
        932,
        944,
        947,
        948,
        949,
        966,
        967,
        974,
        982,
        993,
        994,
        996,
        998,
        1009,
        1018,
        1039,
        1052,
        1055,
        1058,
        1060,
        1062,
        1064,
        1067,
        1069,
        1070,
        1076,
        1109,
        1110,
        1112,
        1119,
        1120,
        1147,
        1156,
        1166,
        1182,
        1187,
        1194,
        1197,
        1198,
        1199,
        1202,
        1210,
        1223,
        1239,
        1265,
        1266,
        1280,
        1285,
        1289,
        1295,
        1301,
        1304,
        1305,
        1309,
        1325,
        1335,
        1342,
        1363,
        1370,
        1374,
        1378,
        1382,
        1402,
        1407,
        1410,
        1414,
        1453,
        1493,
        1504,
        1507,
        1516,
        1525,
        1539,
        1540,
        1541,
        1542,
        1550,
        1556,
        1565,
        1572,
        1576,
        1577,
        1582,
        1597,
        1599,
        1601,
        1607,
        1631,
        1632,
        1635,
        1640,
        1656,
        1667,
        1670,
        1680,
        1685,
        1697,
        1709,
        1715,
        1718,
        1741,
        1748,
        1757,
        1758,
        1764,
        1768,
        1770,
        1781,
        1790,
        1794,
        1816,
        1817,
        1825,
        1826,
        1841,
        1849,
        1869,
        1873,
        1884,
        1886,
        1894,
        1896,
        1905,
        1930,
        1936,
        1943,
        1948,
        1965,
        1973,
        1979,
        1984,
        1987,
        1989,
        2022,
        2025,
        2028,
        2052,
        2058,
        2079,
        2102,
        2103,
        2104,
        2129,
        2133,
        2135,
        2136,
        2141,
        2151,
        2152,
        2159,
        2161,
        2166,
        2179,
        2208,
        2235,
        2238,
        2239,
        2243,
        2270,
        2290,
        2293,
        2301,
        2304,
        2318,
        2322,
        2357,
        2363,
        2364,
        2370,
        2380,
        2389,
        2392,
        2413,
        2414,
        2425,
        2442,
        2443,
        2482,
        2499,
    ]
)  # indexing into flattened array


obs_mask = jnp.isin(jnp.arange(0, args["n"] ** args["dim"]), obs_idx, assume_unique=True)


ground_truth_y_obs = ground_truth_y_draw[obs_idx]
x_obs = jnp.arange(0, args["n"] ** args["dim"])[obs_idx]


rng_key, rng_key_all_mcmc, rng_key_true_mcmc = random.split(rng_key, 3)

mcmc_samples = run_mcmc_cvae(rng_key_true_mcmc, cvae_length_mcmc, ground_truth_y_obs, obs_idx, c=1)
save_samples(args["expcode"], gen_file_name(args["expcode"], args, "inference_true_ls_mcmc"), mcmc_samples)

mcmc_samples = run_mcmc_cvae(rng_key_all_mcmc, cvae_length_mcmc, ground_truth_y_obs, obs_idx, c=None)
save_samples(args["expcode"], gen_file_name(args["expcode"], args, "inference_all_ls_mcmc"), mcmc_samples)
