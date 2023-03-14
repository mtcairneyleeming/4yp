import os
import dill
import signal
from flax import serialization
from flax.core.frozen_dict import freeze
import jax.numpy as jnp


def get_savepath():
    # work out where to save outputs:
    if os.path.isdir("output"):  # running in an ARC job (using my submission script)
        save_path = "output"
    elif os.path.isdir("learnt_models"):  # running from the root directory of the git repo, not in a job
        save_path = "learnt_models"
    else:  # somewhere else?
        save_path = "."
    return save_path


def save_args(name, args):

    with open(f"{get_savepath()}/{name}_args.dill", "wb+") as f:
        dill.dump(args, f)


def decoder_filename(file_code, args, suffix="_decoder"):
    """Return a file name that reflects the params used to generate the saved weights. If the structure of args changes, this will gracefully fail,
    as it uses a default value if any of the params change."""

    param_names = (
        "hidden_dim1",
        "hidden_dim2",
        "latent_dim",
        "vae_var",
        "num_epochs",
        "learning_rate",
        "batch_size",
        "train_num_batches",
    )
    vals = []
    for p in param_names:
        vals.append(str(args.get(p, ".")))

    return f"{file_code}_" + "_".join(vals) + suffix


def setup_signals():
    def print_signal(sig, frame):
        print("Script recieved signal:", sig)
        if sig == 15:
            print("SIGTERM recieved, raising SIGINT")
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, print_signal)
    signal.signal(signal.SIGCONT, print_signal)


def save_training(path, final_state, metrics_history):
    if final_state is not None:
        with open(path, "wb") as file:
            file.write(serialization.to_bytes(freeze({"params": final_state.params})))

    with open(path + "_metrics_hist", "wb") as file:
        dill.dump(metrics_history, file)
    print(f"Saved {path}")


def update_args_once(args, key, value):
    if key in ["n", "train_num_batches", "batch_size", "num_epochs"]:
        args[key] = value
    elif key == "vae_scale_factor":
        args["hidden_dim1"] = int(value * 35)
        args["hidden_dim2"] = int(value * 32)
        args["latent_dim"] = int(value * 30)
    else:
        raise NotImplementedError(f"Unknown key: {key} (value: {value})")
    return args


def update_args_11(args, exp_details, i, j):

    Arange = exp_details["Arange"]
    Brange = exp_details["Brange"]
    Adesc = exp_details["Adesc"]
    Bdesc = exp_details["Bdesc"]
    args = update_args_once(args, Adesc, Arange[i])
    args = update_args_once(args, Bdesc, Brange[j])


    args["x"] = jnp.arange(0, 1, 1 / args["n"])  # if we have changed it!

    return args
