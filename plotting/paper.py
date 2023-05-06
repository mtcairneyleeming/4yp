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
from mpl_toolkits.axes_grid1 import AxesGrid
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

# ======================================================================================
# Get training histories/etc.


def get_training_histories(code, exp_name, args_count, file_back_compat=None):
    args = load_args(str(code), args_count, exp_name)
    return get_training_histories_from_args(args, file_back_compat)


def get_training_histories_from_args(args, file_back_compat=None):

    hists = []

    for loss_fn in args["loss_fn_names"]:
        if loss_fn == "gp":
            continue

        try:
            hists.append(
                (
                    loss_fn,
                    load_training_history(
                        args["expcode"],
                        gen_file_name(
                            args["expcode"],
                            args,
                            (args["experiment"] if "experiment" in args else "") + loss_fn,
                            back_compat_version=file_back_compat,
                        ),
                    ),
                )
            )
        except FileNotFoundError as e:
            print(e)
    return hists, args


def get_training_history(code, exp_name, args_count, index, file_back_compat=None):
    return get_training_histories(code, exp_name, args_count, file_back_compat)[index][0]


def get_trained_draws(
    code,
    exp_name,
    args_count,
    file_back_compat=None,
    include_standard_vae=False,
    include_gp=True,
    use_single_decoder=False,
    leaky_relu=True,
    filter_loss_fns=None,
    gp_builder=None,
):
    return get_trained_draws_from_args(
        load_args(str(code), str(args_count), exp_name),
        file_back_compat=file_back_compat,
        include_standard_vae=include_standard_vae,
        include_gp=include_gp,
        use_single_decoder=use_single_decoder,
        leaky_relu=leaky_relu,
        filter_loss_fns=filter_loss_fns,
        gp_builder=gp_builder,
    )


def get_trained_draws_from_args(
    args,
    file_back_compat=None,
    include_standard_vae=False,
    include_gp=True,
    use_single_decoder=False,
    leaky_relu=True,
    filter_loss_fns=None,
    gp_builder=None,
):
    rng_key = random.PRNGKey(3)
    rng_key, rng_key_gp = random.split(rng_key, 2)

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

    if include_gp:
        if gp_builder is None:
            gp = BuildGP(
                args["gp_kernel"],
                noise=False,
                length_prior_choice=args["length_prior_choice"],
                length_prior_args=args["length_prior_arguments"],
                variance_prior_choice=args.get("variance_prior_choice", "lognormal"),
                variance_prior_args=args.get("variance_prior_arguments", {"location": 0.0, "scale": 0.1}),
            )
        else:
            gp = gp_builder(args)

        plot_gp_predictive = Predictive(gp, num_samples=5000)

    draws = []

    if include_gp:
        gp_draws = plot_gp_predictive(rng_key_gp, x=args["x"], gp_kernel=args["gp_kernel"], jitter=1e-5)["y"]

        draws.append((gp_draws, "GP"))

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

        if use_single_decoder:
            single_decoder = Single_Decoder(
                hidden_dim1=args["hidden_dim1"], hidden_dim2=args["hidden_dim2"], out_dim=args["n"], leaky=leaky_relu
            )

        params = module.init(rng_key, jnp.ones((args["n"],)))["params"]
        tx = optax.adam(args["learning_rate"])
        state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

        if use_single_decoder:
            params = single_decoder.init(rng_key, jnp.ones((args["n"] + args["latent_dim"],)))["params"]

            dec_state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)

        if include_standard_vae and loss_fn == "RCL+KLD":
            standard_args = load_args(16, 1, "exp1")
            decoder_params = get_decoder_params(
                load_training_state("16", gen_file_name("16", standard_args, "exp1" + loss_fn, "A"), state)
            )
        else:
            if use_single_decoder:

                decoder_params = get_model_params(
                    load_training_state(
                        args["expcode"],
                        gen_file_name(
                            args["expcode"],
                            args,
                            (args["experiment"] if "experiment" in args else "") + loss_fn,
                            file_back_compat,
                        ),
                        dec_state,
                    )
                )
            else:
                decoder_params = get_decoder_params(
                    load_training_state(
                        args["expcode"],
                        gen_file_name(
                            args["expcode"],
                            args,
                            (args["experiment"] if "experiment" in args else "") + loss_fn,
                            file_back_compat,
                        ),
                        state,
                    )
                )

        vae_predictive = Predictive(decoder_sample if use_single_decoder else vae_sample, num_samples=5000)
        vae_draws = vae_predictive(
            rng_key_predict,
            hidden_dim1=args["hidden_dim1"],
            hidden_dim2=args["hidden_dim2"],
            latent_dim=args["latent_dim"],
            out_dim=args["n"],
            decoder_params=decoder_params,
        )["f"]

        draws.append((vae_draws, loss_fn))

    return draws, args


# ======================================================================================
# Plot loaded data


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


def plot_training_histories(histories, save_file_name, num_rows, num_cols, backfill=None):
    match backfill:
        case None:
            mapping = lambda i: i
        case "align_left":
            mapping = align_left_backfill(len(histories), num_rows, num_cols)
        case "align_right":
            mapping = align_right_backfill(len(histories), num_rows, num_cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 5))

    clear_unused_axs(axs, mapping, len(histories))

    i = 0
    for (loss_fn, hist) in histories:
        if loss_fn == "gp":
            continue

        try:
            plot_training(
                hist["test_loss"],
                hist["train_loss"],
                pretty_loss_fn_name(loss_fn),
                ax=axs.flat[mapping(i)],
            )
        except FileNotFoundError as e:
            print(e)

        i += 1

    fig.tight_layout()

    fig.savefig(f"./gen_plots/{save_file_name}_training.pdf")


def plot_trained_draws(
    draws,
    x,
    num_cols,
    num_rows,
    save_file_name,
    backfill=None,
    separate_gp=False,
    plot_range=None,
    y_axis_label="$y=f_{VAE}(x)$",
    legend_label="PriorVAE",
    page_max_rows=3,
):
    assert num_cols * num_rows >= len(draws)

    # add an extra row if asked, or if it won't fit in the grid
    if separate_gp:
        num_rows += 1

    match backfill:
        case None:
            if separate_gp:
                mapping = lambda i: 0 if i == 0 else i - 1 + num_cols
            else:
                mapping = lambda i: i
        case "align_left":
            mapping = align_left_backfill(len(draws), num_rows, num_cols)
        case "align_right":
            mapping = align_right_backfill_with_gp(len(draws), num_rows, num_cols)

    figs = []
    axes = onp.empty((0,))
    print("sillt", page_max_rows)
    if page_max_rows is not None:

        iter_total = (num_rows // page_max_rows) * page_max_rows
        print(num_rows, iter_total, num_rows // page_max_rows + 1)
        for i in range(num_rows // page_max_rows + 1):
            if i == num_rows // page_max_rows and iter_total == num_rows:
                continue
            nrows = num_rows - iter_total if i == num_rows // page_max_rows else page_max_rows
            fig, a = plt.subplots(nrows=nrows, ncols=num_cols, figsize=(9, 6 * nrows / page_max_rows))
            figs.append(fig)
            axes = onp.concatenate((axes, a.flat))
    else:
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 5))
        figs.append(fig)
        axes = axs.flat

    clear_unused_axs(axes, mapping, len(draws) + 1)

    for i, (draw, title) in enumerate(draws):
        if plot_range is None:
            plot_range_i = [None, None]
        elif len(plot_range) > 2:
            plot_range_i = plot_range[i]
        else:
            plot_range_i = plot_range
        plot_draws_hpdi(
            draw,
            x,
            pretty_loss_fn_name(title),
            "$y=f_{GP}(x)$" if title == "gp" else y_axis_label,
            "GP" if title == "gp" else legend_label,
            ax=axes[mapping(i)],
            _min=plot_range_i[0],
            _max=plot_range_i[1],
        )
    for i, fig in enumerate(figs):
        fig.tight_layout()

        fig.savefig(f"./gen_plots/{save_file_name}_draws_{i}.pdf")


def plot_trained_draws_compact(
    draws,
    x,
    num_cols,
    num_rows,
    save_file_name,
    backfill=None,
    separate_gp=False,
    plot_range=None,
    y_axis_label="$y=f_{VAE}(x)$",
    legend_label="PriorVAE",
    page_max_rows=3,
):
    assert num_cols * num_rows >= len(draws)

    # add an extra row if asked, or if it won't fit in the grid
    if separate_gp:
        num_rows += 1

    match backfill:
        case None:
            if separate_gp:
                mapping = lambda i: 0 if i == 0 else i - 1 + num_cols
            else:
                mapping = lambda i: i
        case "align_left":
            mapping = align_left_backfill(len(draws), num_rows, num_cols)
        case "align_right":
            mapping = align_right_backfill_with_gp(len(draws), num_rows, num_cols)

    figs = []
    axes = onp.empty((0,))
    if page_max_rows is not None:

        iter_total = (num_rows // page_max_rows) * page_max_rows
        print(num_rows, iter_total, num_rows // page_max_rows + 1)
        for i in range(num_rows // page_max_rows + 1):
            if i == num_rows // page_max_rows and iter_total == num_rows:
                continue
            nrows = num_rows - iter_total if i == num_rows // page_max_rows else page_max_rows
            # fig = plt.figure()
            # fig.set_size_inches(9, 6 * nrows / page_max_rows)
            # grid = AxesGrid(fig, (1, 1, 1), nrows_ncols=(nrows, num_cols), label_mode="L", share_all=True, axes_pad=0.12, aspect=False)

            # figs.append(fig)
            # axes = onp.concatenate((axes, grid.axes_all))
            fig, a = plt.subplots(nrows=nrows, ncols=num_cols, figsize=(9, 6 * nrows / page_max_rows))
            figs.append(fig)
            axes = onp.concatenate((axes, a.flat))
    else:
        fig, axs = plt.subplots(
            nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 5), sharex="row", sharey="row"
        )
        figs.append(fig)
        axes = axs.flat

    clear_unused_axs(axes, mapping, len(draws) + 1)

    for i, (draw, title) in enumerate(draws):
        if plot_range is None:
            plot_range_i = [None, None]
        elif len(plot_range) > 2:
            plot_range_i = plot_range[i]
        else:
            plot_range_i = plot_range
        plot_draws_hpdi(
            draw,
            x,
            pretty_loss_fn_name(title),
            (y_axis_label[i // num_cols] if isinstance(y_axis_label, list) else y_axis_label) if mapping(i) % num_cols == 0 else "",
            None,
            ax=axes[mapping(i)],
            _min=plot_range_i[0],
            _max=plot_range_i[1],
            show_legend=mapping(i) % num_cols == 0,
            show_x_label=False
        )
        # if mapping(i) % num_cols != 0:
        #    axes[mapping(i)].set
    for i, fig in enumerate(figs):
        fig.tight_layout(pad=0.5)
        fig.savefig(f"./gen_plots/{save_file_name}_draws_{i}.pdf")


# ======================================================================================
# Get and plot helper methods


def plot_simple_hists(code, exp_name, args_disambig, file_compat):
    hists, args = get_training_histories(code, exp_name, args_disambig, file_compat)
    plot_training_histories(hists, f"/{code}/{code}_{exp_name}_{args_disambig}", *calc_plot_dimensions(args, False))


def plot_simple_draws(
    code,
    exp_name,
    args_disambig,
    file_compat,
    include_standard_vae=False,
    use_single_decoder=False,
    leaky_relu=True,
    filter=None,
    gp_builder=None,
    plot_range=None,
):
    draws, args = get_trained_draws(
        code,
        exp_name,
        args_disambig,
        file_back_compat=file_compat,
        include_standard_vae=include_standard_vae,
        use_single_decoder=use_single_decoder,
        leaky_relu=leaky_relu,
        filter_loss_fns=filter,
        gp_builder=gp_builder,
    )
    plot_trained_draws(
        draws,
        args["x"],
        *calc_plot_dimensions(args, True),
        f"/{code}/{code}_{exp_name}_{args_disambig}",
        backfill="align_right",
        plot_range=plot_range,
    )
