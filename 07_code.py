# PriorVAE - comparing "infinite" to the previous setting of iterating over the same dataset

print("Starting", flush=True)
import dill
import jax.numpy as jnp
import jax.profiler
import numpyro
from flax import serialization
from jax import random
from numpyro.infer import Predictive

numpyro.set_host_device_count(3)

from reusable.kernels import esq_kernel

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
        "batch_size": 500,
        "train_num_batches": 1000,
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
print(args["x"].nbytes)

rng_key, _ = random.split(random.PRNGKey(4))


from reusable.loss import KLD, RCL, MMD_rbf, combo3_loss

loss_fns = [combo3_loss(RCL, KLD, MMD_rbf(args["mmd_rbf_ls"]), 0.01, 1, s) for s in [1, 10, 25, 50]]


args["loss_functions"] = [x.__name__ for x in loss_fns]
print(len(loss_fns))

import sys

index = int(sys.argv[1])


loss_fn = loss_fns[index // 2]

infinite = index % 2 == 0

print(loss_fn.__name__, flush=True)

from reusable.data import (gen_all_gp_batches, gen_gp_batches,
                           get_batches_generator)
from reusable.gp import OneDGP

rng_key, rng_key_train, rng_key_test = random.split(rng_key, 3)


if not infinite:

    repeated_train_draws = gen_gp_batches(
        args["x"], OneDGP, args["gp_kernel"], args["train_num_batches"], args["batch_size"], rng_key_train
    )

test_draws = gen_gp_batches(
    args["x"], OneDGP, args["gp_kernel"], 1, args["test_num_batches"] * args["batch_size"], rng_key_test
)

del rng_key_train
del rng_key_test

print("Generated data", flush=True)

import optax

from reusable.train_nn import SimpleTrainState
from reusable.vae import VAE

rng_key, rng_key_init, rng_key_train, rng_key_test = random.split(rng_key, 4)

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


import jax
import optax
from flax.core.frozen_dict import freeze

from reusable.loss import KLD, RCL, MMD_rbf, MMD_rqk
from reusable.vae import vae_sample

mmd_rbf = MMD_rbf(args["mmd_rbf_ls"])
mmd_rqk = MMD_rqk(args["mmd_rq_ls"], args["mmd_rq_scale"])


def compute_epoch_metrics(final_state: SimpleTrainState, train_output, test_output):
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
        "train_mmd_rbf": mmd_rbf(*train_output),
        "test_mmd_rbf": mmd_rbf(*test_output),
        "train_mmd_rqk": mmd_rqk(*train_output),
        "test_mmd_rqk": mmd_rqk(*test_output),
        "train_mmd_rbf_new_draws": mmd_rbf(
            vae_draws, repeated_train_draws[-1], 0, 0
        ),  # ignore 0s, just there to satisfy extra arguments
        "test_mmd_rbf_new_draws": mmd_rbf(vae_draws, test_draws[-1], 0, 0),
        "train_kld": KLD(*train_output),
        "test_kld": KLD(*test_output),
        "train_rcl": RCL(*train_output),
        "test_rcl": RCL(*test_output),
    }

    return metrics


import jax.random as random
from numpyro.infer import Predictive

from reusable.gp import OneDGP
from reusable.train_nn import run_training, run_training_datastream
from reusable.util import decoder_filename, get_savepath
from reusable.vae import vae_sample

print("Starting training", flush=True)


if not infinite:
    final_state, metrics_history = run_training(
        loss_fn, compute_epoch_metrics, args["num_epochs"], repeated_train_draws, test_draws, state
    )
else:
    train_batch_gen = get_batches_generator(
        args["x"], OneDGP, args["gp_kernel"], args["train_num_batches"], args["batch_size"]
    )

    final_state, metrics_history = run_training_datastream(
        loss_fn,
        compute_epoch_metrics,
        args["num_epochs"],
        args["train_num_batches"],
        lambda i: train_batch_gen(random.fold_in(rng_key_train, i)),
        lambda i: test_draws,
        state,
    )

with open(
    f'{get_savepath()}/{decoder_filename("07", args, suffix=loss_fn.__name__+ "_inf" if infinite else "")}', "wb"
) as file:
    file.write(serialization.to_bytes(freeze({"params": final_state.params["VAE_Decoder_0"]})))

with open(
    f'{get_savepath()}/{decoder_filename("07", args, suffix=loss_fn.__name__+"_metrics_hist"+ "_inf" if infinite else "")}',
    "wb",
) as file:
    dill.dump(metrics_history, file)

jax.profiler.save_device_memory_profile(f'{get_savepath()}/{decoder_filename("07", args, suffix=f"{index}.prof")}')

from reusable.util import save_args

save_args("04", args)

print("Saved args", flush=True)
