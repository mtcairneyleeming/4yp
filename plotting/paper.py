import jax.numpy as jnp
import numpy as onp
from reusable.util import (
    load_args,
    load_training_history,
    gen_file_name,
    get_decoder_params,
    load_training_state,
    get_model_params,
)
import matplotlib.pyplot as plt
from plotting.plots import plot_training, plot_draws_hpdi
import jax.random as random
from reusable.vae import VAE, vae_sample, Single_Decoder, decoder_sample
from reusable.train_nn import SimpleTrainState
import optax
from numpyro.infer import Predictive
from reusable.gp import BuildGP
import pandas
from plotting.helpers import (
    align_left_backfill,
    align_right_backfill,
    calc_plot_dimensions,
    clear_unused_axs,
    align_right_backfill_with_gp,
    pretty_loss_fn_name,
)


def plot_individual_training_history(code, exp_name, args_count, index):
    args = load_args(str(code), args_count, exp_name)
    loss_fn = args["loss_fn_names"][index]
    hist = load_training_history(
        code, gen_file_name(code, args, (args["experiment"] if "experiment" in args else "") + loss_fn)
    )
    plot_training(
        hist["test_loss"],
        hist["train_loss"],
        pretty_loss_fn_name(loss_fn),
        save_path=f"./gen_plots/{code}/{code}_{exp_name}_{args_count}_{index}_training.pdf",
    )


def get_training_history(code, exp_name, args_count, index):
    args = load_args(str(code), args_count, exp_name)
    loss_fn = args["loss_fn_names"][index]
    hist = load_training_history(
        code, gen_file_name(code, args, (args["experiment"] if "experiment" in args else "") + loss_fn)
    )
    return hist


def plot_individual_trained_draws(
    code,
    exp_name,
    args_count,
    index,
    include_gp=False,
    include_standard_vae=False,
    single_decoder=False,
    leaky_relu=True,
):
    rng_key = random.PRNGKey(3)
    rng_key, rng_key_gp = random.split(rng_key, 2)

    args = load_args(str(code), str(args_count), exp_name)

    args["loss_fns"] = [args["loss_fns"][index]]
    args["loss_fn_names"] = [args["loss_fn_names"][index]]

    if include_standard_vae:
        args["loss_fn_names"] = ["RCL+KLD"] + args["loss_fn_names"]
        args["loss_fns"] = [None] + args["loss_fns"]
    num_cols = len(args["loss_fn_names"]) + (1 if include_gp else 0)

    fig, axs = plt.subplots(nrows=1, ncols=num_cols, figsize=(num_cols * 6, 5))

    if include_gp:
        gp = BuildGP(
            args["gp_kernel"],
            noise=False,
            length_prior_choice=args["length_prior_choice"],
            length_prior_args=args["length_prior_arguments"],
            variance_prior_choice=args["variance_prior_choice"],
            variance_prior_args=args["variance_prior_arguments"],
        )

        plot_gp_predictive = Predictive(gp, num_samples=5000)

        gp_draws = plot_gp_predictive(rng_key_gp, x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5)["y"]
        plot_draws_hpdi(
            gp_draws,
            args["x"],
            f"GP draws",
            "$y=f_{GP}(x)$",
            "GP",
            ax=axs[0],
        )

    for i, loss_fn in enumerate(args["loss_fn_names"]):
        rng_key, rng_key_init, rng_key_predict = random.split(rng_key, 3)

        module = VAE(
            hidden_dim1=args["hidden_dim1"],
            hidden_dim2=args["hidden_dim2"],
            latent_dim=args["latent_dim"],
            out_dim=args["n"],
            conditional=False,
            leaky=leaky_relu,
        )

        if single_decoder:
            single_decoder = Single_Decoder(
                hidden_dim1=args["hidden_dim1"], hidden_dim2=args["hidden_dim2"], out_dim=args["n"], leaky=leaky_relu
            )

        params = module.init(rng_key, jnp.ones((args["n"],)))["params"]
        tx = optax.adam(args["learning_rate"])
        state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

        if single_decoder:
            params = single_decoder.init(rng_key, jnp.ones((args["n"] + args["latent_dim"],)))["params"]

            dec_state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

        if include_standard_vae and loss_fn == "RCL+KLD":
            standard_args = load_args(16, 1, "exp1")
            decoder_params = get_decoder_params(
                load_training_state("16", gen_file_name("16", standard_args, "exp1" + loss_fn), state)
            )
        else:
            if single_decoder:

                decoder_params = get_model_params(
                    load_training_state(
                        code,
                        gen_file_name(code, args, (args["experiment"] if "experiment" in args else "") + loss_fn),
                        dec_state,
                    )
                )
            else:
                decoder_params = get_decoder_params(
                    load_training_state(
                        code,
                        gen_file_name(code, args, (args["experiment"] if "experiment" in args else "") + loss_fn),
                        state,
                    )
                )

        vae_predictive = Predictive(decoder_sample if single_decoder else vae_sample, num_samples=5000)
        vae_draws = vae_predictive(
            rng_key_predict,
            hidden_dim1=args["hidden_dim1"],
            hidden_dim2=args["hidden_dim2"],
            latent_dim=args["latent_dim"],
            out_dim=args["n"],
            decoder_params=decoder_params,
        )["f"]
        plot_draws_hpdi(
            vae_draws,
            args["x"],
            pretty_loss_fn_name(loss_fn),
            "$y=f_{DEC}(x)$" if single_decoder else "$y=f_{VAE}(x)$",
            "PriorDec" if single_decoder else "PriorVAE",
            ax=axs[i + 1 if include_gp else i],
        )

    fig.tight_layout()

    fig.savefig(f"./gen_plots/{code}/{code}_{exp_name}_{args_count}_{index}_draws.pdf")


def plot_training_histories(code, exp_name, args_count, num_cols=None, num_rows=None, backfill=None):
    args = load_args(str(code), args_count, exp_name)

    twoD, num_rows, num_cols = calc_plot_dimensions(args, num_cols, num_rows, False, False)

    match backfill:
        case None:
            mapping = lambda i: i
        case "align_left":
            mapping = align_left_backfill(len(args["loss_fns"]), num_rows, num_cols)
        case "align_right":
            mapping = align_right_backfill(len(args["loss_fns"]), num_rows, num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 5))

    clear_unused_axs(axs, mapping, twoD, len(args["loss_fn_names"]))

    i = 0
    for loss_fn in args["loss_fn_names"]:
        if loss_fn == "gp":
            continue

        try:
            hist = load_training_history(
                code, gen_file_name(code, args, (args["experiment"] if "experiment" in args else "") + loss_fn)
            )
            plot_training(
                hist["test_loss"],
                hist["train_loss"],
                pretty_loss_fn_name(loss_fn),
                ax=axs[onp.unravel_index(mapping(i), (num_rows, num_cols)) if twoD else i],
            )
        except FileNotFoundError as e:
            print(e)

        i += 1

    fig.tight_layout()

    fig.savefig(f"./gen_plots/{code}/{code}_{exp_name}_{args_count}_training.pdf")


def plot_trained_draws(
    code,
    exp_name,
    args_count,
    num_cols=None,
    num_rows=None,
    backfill=None,
    separate_gp=False,
    include_standard_vae=False,
    single_decoder=False,
    leaky_relu=True,
    filter_loss_fns=None,
    gp_builder=None,
    plot_range=None,
):
    rng_key = random.PRNGKey(3)
    rng_key, rng_key_gp = random.split(rng_key, 2)

    args = load_args(str(code), str(args_count), exp_name)
    print(args["loss_fn_names"])

    if filter_loss_fns is not None:
        new_ln = []
        new_l = []
        for i, l in enumerate(args["loss_fn_names"]):
            if l in filter_loss_fns:
                new_l.append(args["loss_fns"][i])
                new_ln.append(args["loss_fn_names"][i])

        args["loss_fns"] = new_l
        args["loss_fn_names"] = new_ln

    if include_standard_vae:
        args["loss_fn_names"] = ["RCL+KLD"] + args["loss_fn_names"]
        args["loss_fns"] = [None] + args["loss_fns"]

    twoD, num_rows, num_cols = calc_plot_dimensions(args, num_cols, num_rows, True, separate_gp, include_standard_vae)
    print(len(args["loss_fn_names"]), twoD, num_rows, num_cols)

    match backfill:
        case None:
            if separate_gp:
                mapping = lambda i: 0 if i == 0 else i - 1 + num_cols
            else:
                mapping = lambda i: i
        case "align_left":
            mapping = align_left_backfill(len(args["loss_fn_names"]) + 1, num_rows, num_cols)
        case "align_right":
            mapping = align_right_backfill_with_gp(len(args["loss_fn_names"]) + 1, num_rows, num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 5))

    clear_unused_axs(axs, mapping, twoD, len(args["loss_fn_names"]) + 1)

    if gp_builder is None:
        gp = BuildGP(
            args["gp_kernel"],
            noise=False,
            length_prior_choice=args["length_prior_choice"],
            length_prior_args=args["length_prior_arguments"],
            variance_prior_choice=args["variance_prior_choice"],
            variance_prior_args=args["variance_prior_arguments"],
        )
    else:
        gp = gp_builder(args)

    plot_gp_predictive = Predictive(gp, num_samples=5000)

    gp_draws = plot_gp_predictive(rng_key_gp, x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5)["y"]
    plot_draws_hpdi(
        gp_draws,
        args["x"],
        f"GP draws",
        "$y=f_{GP}(x)$",
        "GP",
        ax=axs[onp.unravel_index(mapping(0), (num_rows, num_cols)) if twoD else 0],
        _min=plot_range[0] if plot_range is not None else None,
        _max=plot_range[1] if plot_range is not None else None,
    )
    i = 0
    for loss_fn in args["loss_fn_names"]:
        if loss_fn == "gp":
            continue
        rng_key, rng_key_init, rng_key_predict = random.split(rng_key, 3)

        module = VAE(
            hidden_dim1=args["hidden_dim1"],
            hidden_dim2=args["hidden_dim2"],
            latent_dim=args["latent_dim"],
            out_dim=args["n"],
            conditional=False,
            leaky=leaky_relu,
        )

        if single_decoder:
            single_decoder = Single_Decoder(
                hidden_dim1=args["hidden_dim1"], hidden_dim2=args["hidden_dim2"], out_dim=args["n"], leaky=leaky_relu
            )

        params = module.init(rng_key, jnp.ones((args["n"],)))["params"]
        tx = optax.adam(args["learning_rate"])
        state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

        if single_decoder:
            params = single_decoder.init(rng_key, jnp.ones((args["n"] + args["latent_dim"],)))["params"]

            dec_state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

        if include_standard_vae and loss_fn == "RCL+KLD":
            standard_args = load_args(16, 1, "exp1")
            decoder_params = get_decoder_params(
                load_training_state("16", gen_file_name("16", standard_args, "exp1" + loss_fn), state)
            )
        else:
            if single_decoder:

                decoder_params = get_model_params(
                    load_training_state(
                        code,
                        gen_file_name(code, args, (args["experiment"] if "experiment" in args else "") + loss_fn),
                        dec_state,
                    )
                )
            else:
                decoder_params = get_decoder_params(
                    load_training_state(
                        code,
                        gen_file_name(code, args, (args["experiment"] if "experiment" in args else "") + loss_fn),
                        state,
                    )
                )

        vae_predictive = Predictive(decoder_sample if single_decoder else vae_sample, num_samples=5000)
        vae_draws = vae_predictive(
            rng_key_predict,
            hidden_dim1=args["hidden_dim1"],
            hidden_dim2=args["hidden_dim2"],
            latent_dim=args["latent_dim"],
            out_dim=args["n"],
            decoder_params=decoder_params,
        )["f"]
        
        plot_draws_hpdi(
            vae_draws,
            args["x"],
            pretty_loss_fn_name(loss_fn),
            "$y=f_{DEC}(x)$" if single_decoder else "$y=f_{VAE}(x)$",
            "PriorDec" if single_decoder else "PriorVAE",
            ax=axs[onp.unravel_index(mapping(i + 1), (num_rows, num_cols)) if twoD else i + 1],
            _min=plot_range[0] if plot_range is not None else None,
            _max=plot_range[1] if plot_range is not None else None,
        )

        i += 1

    fig.tight_layout()

    fig.savefig(f"./gen_plots/{code}/{code}_{exp_name}_{args_count}_draws.pdf")
