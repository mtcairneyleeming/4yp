import os
import dill
import signal
from flax import serialization
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
import glob


def __get_savepath():
    # work out where to save outputs:
    if os.path.isdir("output"):  # running in an ARC job (using my submission script)
        save_path = "output"
    elif os.path.isdir("learnt_models"):  # running from the root directory of the git repo, not in a job
        save_path = "learnt_models"
    else:  # somewhere else?
        save_path = "."
    return save_path


def count_saved_args(exp_code, args_file_ext):
    base_path = f"{__get_savepath()}/{str(exp_code)}"
    return len(glob.glob1(base_path, f"*{args_file_ext}"))


def save_args(exp_code, file_name, args, args_file_ext=".args"):
    count = count_saved_args(exp_code, args_file_ext) + 1
    path = __get_savepath() + f"/{exp_code}/{exp_code}_{count}_{file_name}{args_file_ext}"
    with open(path, "wb+") as f:
        dill.dump(args, f)
    print(f"Saved args to {path}")


def load_args(exp_code, count, file_name, args_file_ext=".args"):
    insert = f"{count}_{file_name}" if count is not None else file_name
    path = __get_savepath() + f"/{exp_code}/{exp_code}_{insert}{args_file_ext}"
    with open(path, "rb+") as f:
        return dill.load(f)


def save_training(exp_code, file_name, final_state, metrics_history, state_file_ext=".state", hist_file_ext=".hist"):
    base_path = __get_savepath() + f"/{exp_code}/{file_name}"
    if final_state is not None:
        with open(base_path + state_file_ext, "wb") as file:
            file.write(serialization.to_bytes(final_state))

    with open(base_path + hist_file_ext, "wb") as file:
        dill.dump(metrics_history, file)
    print(f"Saved {base_path+state_file_ext} and {base_path+hist_file_ext}")


def load_training_state(exp_code, file_name, dummy_state, state_file_ext=".state"):
    base_path = __get_savepath() + f"/{exp_code}/{file_name}"

    with open(base_path + state_file_ext, "rb") as file:
        bytes = file.read()
        return serialization.from_bytes(dummy_state, bytes)


def get_decoder_params(state):
    return freeze({"params": state.params["VAE_Decoder_0"]})


def load_training_history(exp_code, file_name, hist_file_ext=".hist"):
    base_path = __get_savepath() + f"/{exp_code}/{file_name}"

    with open(base_path + hist_file_ext, "rb") as file:
        return dill.load(file)


def save_samples(exp_code, file_name, samples, samples_file_ext=".samples"):
    base_path = __get_savepath() + f"/{exp_code}/{file_name}"
    with open(base_path + samples_file_ext, "wb") as file:
        dill.dump(samples, file)
    print(f"Saved {base_path+samples_file_ext}")


def load_samples(exp_code, file_name, samples_file_ext=".samples"):
    base_path = __get_savepath() + f"/{exp_code}/{file_name}"
    with open(base_path + samples_file_ext, "rb") as file:
        return dill.load(file)


def save_scores(exp_code, file_name, scores, scores_file_ext=".scores"):
    base_path = __get_savepath() + f"/{exp_code}/{file_name}"
    with open(base_path + scores_file_ext, "wb") as file:
        dill.dump(scores, file)
    print(f"Saved {base_path+scores_file_ext}")


def load_scoress(exp_code, file_name, scores_file_ext=".scores"):
    base_path = __get_savepath() + f"/{exp_code}/{file_name}"
    with open(base_path + scores_file_ext, "rb") as file:
        return dill.load(file)


def save_datasets(exp_code, file_name, train_data, test_data, data_file_ext=".npz"):
    path = f"{__get_savepath()}/{exp_code}/{file_name}{data_file_ext}"
    jnp.savez(path, train=train_data, test=test_data)


def load_datasets(exp_code, file_name, data_file_ext=".npz", on_arc=False):
    if on_arc:
        path = f"data/{file_name}{data_file_ext}"
    else:
        path = f"{__get_savepath()}/{exp_code}/{file_name}{data_file_ext}"
    data = jnp.load(path)
    train_draws = data["train"]
    test_draws = data["test"]
    return train_draws, test_draws


def gen_file_name(exp_prefix, naming_args, desc_suffix="", include_mcmc=False, args_leave_out=[]):
    """Return a file name that reflects the params used to generate the saved weights. If the structure of args changes, this will gracefully fail,
    as it uses a default value if any of the params change."""

    param_names = [
        x
        for x in [
            "n",
            "hidden_dim1",
            "hidden_dim2",
            "latent_dim",
            "vae_var",
            "num_epochs",
            "learning_rate",
            "batch_size",
            "train_num_batches",
            "scoring_num_draws",
        ]
        if x not in args_leave_out
    ]
    if include_mcmc:
        param_names = param_names + [
            x
            for x in [
                "num_warmup",
                "num_samples",
                "thinning",
                "num_chains",
                "num_samples_to_save",
            ]
            if x not in args_leave_out
        ]

    vals = []
    for p in param_names:
        if p in naming_args:
            vals.append(str(naming_args[p]))

    return f"{exp_prefix}__" + "_".join(vals) + "__" + desc_suffix


def setup_signals():
    def print_signal(sig, frame):
        print("Script recieved signal:", sig)
        if sig == 15:
            print("SIGTERM recieved, raising SIGINT")
            raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, print_signal)
    signal.signal(signal.SIGCONT, print_signal)


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
