from jax import random
import jax.numpy as jnp

import time
import dill
from flax import serialization

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive
from numpyro.diagnostics import hpdi

from reusable.kernels import esq_kernel

import util

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

    # learning
    "num_epochs": 50,
    "learning_rate": 1.0e-3,
    "batch_size": 100,
    "epoch_train_size": 1000,
    "epoch_test_size": 1000,

    # MCMC parameters
    "num_warmup": 1000,
    "num_samples": 1000,
    "thinning": 1,
    "num_chains": 3,

    "pretrained_vae": False


})

util.save_args("01", args)



from reusable.gp import OneDGP

if not args["pretrained_vae"]:

    from reusable.vae import vae_model, vae_guide

    adam = optim.Adam(step_size=args["learning_rate"])

    svi = SVI(
        vae_model,
        vae_guide,
        adam,
        Trace_ELBO(),
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        vae_var=args["vae_var"] 
    )

    rng_key, rng_key_predictive, rng_key_init = random.split(args["rng_key"], 3)

    gp_predictive = Predictive(OneDGP, num_samples=args["batch_size"])

    def gp_batch_gen(rng_key_predictive):
        return gp_predictive(rng_key_predictive, x=args["x"], gp_kernel=args["gp_kernel"])

    init_batch = gp_batch_gen(rng_key_init)["y"]

    svi_state = svi.init(rng_key_init, init_batch)




if not args["pretrained_vae"]:
    from reusable.svi import svi_test_eval, svi_training_epoch
    test_losses = []

    for i in range(args["num_epochs"]):
        rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)
        
        t_start = time.time()

        _, svi_state = svi_training_epoch(svi, rng_key_train, svi_state, args["epoch_train_size"], gp_batch_gen, args["batch_size"] )
        
        test_loss = svi_test_eval(svi, rng_key_test, svi_state, args["epoch_train_size"], gp_batch_gen, args["batch_size"])
        test_losses += [test_loss]

        print(
            "Epoch {}: loss = {} ({:.2f} s.)".format(
                i, test_loss, time.time() - t_start
            )
        )
        
        if jnp.isnan(test_loss): break


from flax.core.frozen_dict import freeze

if not args["pretrained_vae"]:
    decoder_params = svi.get_params(svi_state)["decoder$params"]
    #print(decoder_params)
    decoder_params = freeze({"params": decoder_params})
    args["decoder_params"] = decoder_params
    with open(f'{util.get_savepath()}/01_decoder_1d_n{args["n"]}', 'wb') as file:
       file.write(serialization.to_bytes(decoder_params))