import os
import dill
import signal
from flax import serialization
from flax.core.frozen_dict import freeze
import jax.numpy as jnp
import glob


def __get_savepath(exp_code, arc_data_dir=False, arc_learnt_models_dir=False):
    # work out where to save outputs:
    if arc_data_dir:
        return "data"
    elif arc_learnt_models_dir:  # e.g. for accessing pre-trained
        dir = os.getenv("WORKING_DIR")
        save_path = f"{dir}/learnt_models"
    elif os.path.isdir("output"):  # running in an ARC job (using my submission script)
        save_path = "output"
    elif os.path.isdir("learnt_models"):  # running from the root directory of the git repo, not in a job
        save_path = "learnt_models"
    else:  # somewhere else?
        save_path = "."
    return f"{save_path}/{exp_code}"


def count_saved_args(exp_code, args_file_ext):
    return len(glob.glob1(__get_savepath(exp_code), f"*{args_file_ext}"))


def save_args(exp_code, file_name, args, args_file_ext=".args"):
    count = count_saved_args(exp_code, args_file_ext) + 1
    path = __get_savepath(exp_code) + f"/{exp_code}_{count}_{file_name}{args_file_ext}"
    with open(path, "wb+") as f:
        dill.dump(args, f)
    print(f"Saved {path}")


def load_args(exp_code, count, file_name, args_file_ext=".args"):
    insert = f"{count}_{file_name}" if count is not None else file_name
    path = __get_savepath(exp_code) + f"/{exp_code}_{insert}{args_file_ext}"
    with open(path, "rb+") as f:
        return dill.load(f)


def save_training(exp_code, file_name, final_state, metrics_history, state_file_ext=".state", hist_file_ext=".hist"):
    base_path = __get_savepath(exp_code) + f"/{file_name}"
    if final_state is not None:
        with open(base_path + state_file_ext, "wb") as file:
            file.write(serialization.to_bytes(final_state))

    with open(base_path + hist_file_ext, "wb") as file:
        dill.dump(metrics_history, file)
    print(f"Saved {base_path+state_file_ext}")
    print(f"Saved {base_path+hist_file_ext}")


def load_training_state(exp_code, file_name, dummy_state, arc_learnt_models_dir=False, state_file_ext=".state"):
    with open(
        __get_savepath(exp_code, arc_learnt_models_dir=arc_learnt_models_dir) + f"/{file_name}{state_file_ext}", "rb"
    ) as file:
        bytes = file.read()
        return serialization.from_bytes(dummy_state, bytes)


def get_decoder_params(state, decoder_name=None):
    if decoder_name == None:
        decoder_name = "VAE_Decoder_0"
    try:
        p = state["params"][decoder_name]

    except TypeError:
        p = state.params[decoder_name]

    return freeze({"params": p})


def get_model_params(state):
    try:
        p = state["params"]
    except TypeError:
        p = state.params
    return freeze({"params": p})


def load_training_history(exp_code, file_name, hist_file_ext=".hist"):
    with open(__get_savepath(exp_code) + f"/{file_name}{hist_file_ext}", "rb") as file:
        return dill.load(file)


def save_samples(exp_code, file_name, samples, samples_file_ext=".samples"):
    path = __get_savepath(exp_code) + f"/{file_name}{samples_file_ext}"
    with open(path, "wb") as file:
        dill.dump(samples, file)
    print(f"Saved {path}")


def load_samples(exp_code, file_name, samples_file_ext=".samples"):
    with open(__get_savepath(exp_code) + f"/{file_name}{samples_file_ext}", "rb") as file:
        return dill.load(file)


def save_scores(exp_code, file_name, scores, scores_file_ext=".scores"):
    path = __get_savepath(exp_code) + f"/{file_name}{scores_file_ext}"
    with open(path, "wb") as file:
        dill.dump(scores, file)
    print(f"Saved {path}")


def load_scores(exp_code, file_name, scores_file_ext=".scores"):
    with open(__get_savepath(exp_code) + f"/{file_name}{scores_file_ext}", "rb") as file:
        return dill.load(file)


def save_datasets(exp_code, file_name, train_data, test_data, data_file_ext=".npz"):
    path = f"{__get_savepath(exp_code)}/{file_name}{data_file_ext}"
    jnp.savez(path, train=train_data, test=test_data)
    print(f"Saved {path}")


def load_datasets(exp_code, file_name, data_file_ext=".npz", on_arc=False):
    data = jnp.load(f"{__get_savepath(exp_code, arc_data_dir=on_arc)}/{file_name}{data_file_ext}")
    train_draws = data["train"]
    test_draws = data["test"]
    return train_draws, test_draws


def gen_file_name(exp_prefix, naming_args, desc_suffix="", backcompat=True, data_only=False, include_mcmc=False, args_leave_out=[]):
    """Return a file name that reflects the params used to generate the saved weights. If the structure of args changes, this will gracefully fail,
    as it uses a default value if any of the params change."""

    STANDARD_PARAMS = [
        "n",
        "hidden_dim1",
        "hidden_dim2",
        "latent_dim",
        "vae_var",
        "leaky_relu",
        "num_epochs",
        "learning_rate",
        "batch_size",
        "train_num_batches",
        "scoring_num_draws",
    ]
    MCMC_PARAMS = [
        "num_warmup",
        "num_samples",
        "thinning",
        "num_chains",
        "num_samples_to_save",
    ]

    DATA_ONLY_PARAMS = [
        "n",
        "batch_size",
        "train_num_batches",
        "test_num_batches",
        "length_prior_choice",
        "length_prior_arguments",
        "variance_prior_choice",
        "variance_prior_arguments"
    ]

    param_names = []
    if backcompat or data_only:
        param_names = param_names + [x for x in DATA_ONLY_PARAMS if x not in args_leave_out]
    
    if not data_only:
        param_names = param_names +  [x for x in STANDARD_PARAMS if x not in args_leave_out]
        if include_mcmc:
            param_names = param_names + [x for x in MCMC_PARAMS if x not in args_leave_out]

    vals = []
    for p in param_names:
        if p in naming_args:
            if isinstance(naming_args[p], dict):
                vals.append("~".join([str(x) for x in naming_args[p].values()]))
            else:
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
