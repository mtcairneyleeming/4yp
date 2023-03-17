import time

import jax.numpy as jnp
from jax import random
from numpyro import optim
from numpyro.infer import SVI, Predictive, Trace_ELBO

from reusable.util import save_args, save_training, gen_file_name
from reusable.kernels import esq_kernel

from flax.core.frozen_dict import freeze

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
        "num_epochs": 5,
        "learning_rate": 1.0e-3,
        "batch_size": 100,
        "epoch_train_size": 1000,
        "epoch_test_size": 1000,
    }
)

save_args("01", "args", args)


from reusable.gp import OneDGP

from reusable.vae import vae_guide, vae_model

adam = optim.Adam(step_size=args["learning_rate"])

svi = SVI(
    vae_model,
    vae_guide,
    adam,
    Trace_ELBO(),
    hidden_dim1=args["hidden_dim1"],
    hidden_dim2=args["hidden_dim2"],
    latent_dim=args["latent_dim"],
    vae_var=args["vae_var"],
)

rng_key, rng_key_predictive, rng_key_init = random.split(args["rng_key"], 3)

gp_predictive = Predictive(OneDGP, num_samples=args["batch_size"])


def gp_batch_gen(rng_key_predictive):
    return gp_predictive(rng_key_predictive, x=args["x"], gp_kernel=args["gp_kernel"])


init_batch = gp_batch_gen(rng_key_init)["y"]

svi_state = svi.init(rng_key_init, init_batch)


from reusable.svi import svi_test_eval, svi_training_epoch

test_losses = []

print("Starting SVI")

for i in range(args["num_epochs"]):
    rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)

    t_start = time.time()

    _, svi_state = svi_training_epoch(
        svi, rng_key_train, svi_state, args["epoch_train_size"], gp_batch_gen, args["batch_size"]
    )

    test_loss = svi_test_eval(svi, rng_key_test, svi_state, args["epoch_train_size"], gp_batch_gen, args["batch_size"])
    test_losses += [test_loss]

    print("Epoch {}: loss = {} ({:.2f} s.)".format(i, test_loss, time.time() - t_start))

    if jnp.isnan(test_loss):
        break

decoder_params = svi.get_params(svi_state)["decoder$params"]

decoder_params = freeze({"params": decoder_params})
save_training("01", gen_file_name("01", args), decoder_params, test_losses)
