import os
import dill
import signal
from flax import serialization
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
from reusable.util import save_args, load_args
from reusable.loss import combo3_loss, combo_loss, MMD_rbf, RCL, KLD

experiment  = "exp9"

args = load_args(11, 1, experiment)


loss_fns = [combo_loss(RCL, KLD), combo3_loss(RCL, KLD, MMD_rbf(args["mmd_rbf_ls"]), 0.01, 1, 10)]
if "loss_fns" in args[experiment]:
    loss_fns = args[experiment]["loss_fns"]

args["loss_fns"] = [combo_loss(RCL, KLD).__name__ ]

save_args(11, experiment, args)
