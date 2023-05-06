import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp
import numpyro.diagnostics
import math
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.lines
import matplotlib.ticker
from matplotlib.colors import Normalize, SymLogNorm

# matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "9"


def plot_draws(draws, x_locs, title, ylabel, ax=None, save_path=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i in range(draws.shape[0]):
        ax.plot(x_locs, draws[i, :])
        if i > 30:
            break

    ax.set_xlabel("$x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_draws_hpdi(
    draws, x, title, ylabel, legend_label, ax=None, save_path=None, _min=-2, _max=2, show_legend=True, show_x_label=True
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if _min is None:
        _min = -2
    if _max is None:
        _max = 2
    ax.set_xlim([0, 1])
    ax.set_ylim([_min, _max])
    if show_x_label:
        ax.set_xlabel("$x$", size=8)
    ax.set_ylabel(ylabel, size=8)
    ax.set_title(title, size=9)
    ax.locator_params("x", nbins=6)

    lines_alpha = 0.1
    N_lines = 15

    # -----------------------
    draws = draws[~jnp.isnan(draws).any(axis=1), :]

    if draws.shape[0] == 0:
        print(f"WARNING! all draws were NaN for title {title}, ylabel {ylabel}")

        return ax

    mean = jnp.nanmean(draws, axis=0)
    hpdi = numpyro.diagnostics.hpdi(draws, 0.9)

    for j in range(0, N_lines):
        ax.plot(x, draws[j, :], alpha=lines_alpha, color="darkgreen")

    hpdi_handle = ax.fill_between(x, hpdi[0], hpdi[1], alpha=0.1, interpolate=True, label="95% HPDI")
    mean_handle = ax.plot(x, mean, label="mean")

    if show_legend:
        ax.legend(
            loc=4,
            handles=[
                matplotlib.lines.Line2D([], [], color="darkgreen", alpha=0.35, label="draws"),
                hpdi_handle,
                mean_handle[0],
            ],
            prop={"size": 9},
        )
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches=None)


def compare_draws(
    x, draws1, draws2, title1, title2, ylabel1, ylabel2, legend_label1, legend_label2, save_path=None, _min=-2, _max=2
):

    # plot results
    fig = plt.figure()
    fig.set_size_inches(6, 3)
    axs = AxesGrid(fig, (1, 1, 1), nrows_ncols=(1, 2), label_mode="L", share_all=True, axes_pad=0.12, aspect=False)

    plot_draws_hpdi(draws1, x, title1, ylabel1, legend_label1, axs[0], _min=_min, _max=_max, show_legend=False)
    plot_draws_hpdi(draws2, x, title2, ylabel2, "", axs[1], _min=_min, _max=_max)

    print(fig.get_size_inches())

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_cov_mat(draws, title, ax=None, save_path=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    mat = jnp.cov(jnp.transpose(draws))

    cmap_choice = "plasma"

    ax.imshow(mat, cmap=cmap_choice)
    ax.axis("off")
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_one_inference(
    inferred_priors,
    x,
    ground_truth,
    x_obs,
    y_obs,
    title,
    ylabel,
    legend_label,
    ax=None,
    save_path=None,
    _min=-2,
    _max=2,
    legend=True,
):
    if ax is None:
        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_subplot(111)

    N_lines = 15

    mean = jnp.mean(inferred_priors, axis=0)
    hpdi = numpyro.diagnostics.hpdi(inferred_priors, 0.9)

    ax.fill_between(
        x,
        hpdi[0],
        hpdi[1],
        alpha=0.1,
        interpolate=True,
        label=f"95% HPDI",
    )
    for j in range(N_lines):
        ax.plot(
            x,
            inferred_priors[j, :],
            alpha=0.1,
            color="darkgreen",
            label=f"posterior draws" if j == 0 else "",
        )

    ax.plot(x, mean, label="predicted mean")
    ax.plot(x, ground_truth, label="ground truth", color="orange")
    if x_obs is not None and y_obs is not None:
        ax.scatter(x_obs, y_obs, color="red", label="observed data", s=20)
    ax.set_title(title)
    if legend:
        ax.legend(loc="upper right", prop={"size": 9})
    ax.set_ylim([_min, _max])
    ax.set_xlabel("$x$")
    ax.set_ylabel(ylabel)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def compare_inference_steps(
    x, ground_truth, x_obss, y_obss, plain_prior_samples, inferred_priors_list, title="VAE", save_path=None
):
    fig = plt.figure()
    fig.set_size_inches(6, 3)
    axs = AxesGrid(
        fig,
        (1, 1, 1),
        nrows_ncols=(1, len(inferred_priors_list)),
        label_mode="L",
        share_all=True,
        axes_pad=0.12,
        aspect=False,
    )

    # plot_draws_hpdi(plain_prior_samples, x, f"{title} draws", f"$f_{{{title}}}(x)", title, ax=axs[0])

    for i in range(len(inferred_priors_list)):
        plot_one_inference(
            inferred_priors_list[i],
            x,
            ground_truth,
            x_obss[i],
            y_obss[i],
            f"{len(x_obss[i])} observations",
            "$f(x)$",
            "VAE",
            ax=axs[i],
            legend=(i == len(inferred_priors_list) - 1),
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_lengthscales(lss, title, ax=None, save_path=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.hist(lss)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("$l$")
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_training(test, train, title, note="", ax=None, save_path=None, ylims=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    assert len(test) == len(train)

    l = jnp.arange(len(test))

    ax.plot(l, test, label="test " + note)
    ax.plot(l, train, label="train " + note)
    ax.set_xlabel("epochs")
    ax.set_ylabel(note)
    ax.set_title(title)
    ax.legend()

    if ylims is not None:
        ax.set_ylim(*ylims)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_training_pair(testA, trainA, testB, trainB, titleA, titleB, note, fig=None, save_path=None, ylims=None):
    if fig is None:
        fig = plt.figure()

    axs = fig.subplots(1, 2)
    all = jnp.concatenate([testA, trainA, testB, trainB])
    ylims = (jnp.min(all), jnp.max(all))

    plot_training(testA, trainA, titleA, note, axs[0], None, ylims)
    plot_training(testB, trainB, titleB, note, axs[1], None, ylims)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_moments(
    moments,
    moment_indices,
    x_locs,
    title,
    correct_moments=None,
    scale="linear",
    use_legend=False,
    ax=None,
    save_path=None,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    colours = [plt.cm.tab10(i) for i in range(len(moment_indices))]

    for i, m in enumerate(moments):
        ax.plot(x_locs, m, color=colours[i], label=f"{moment_indices[i]}th moment")
        if not use_legend:
            ax.text(x_locs[-1] + 0.01 * i, m[-1], s=f"{moment_indices[i]}", va="center", fontsize=14, color=colours[i])

    if correct_moments:
        for i, m in enumerate(correct_moments):
            ax.plot(x_locs, m, color=colours[i], linestyle=":")

    ax.set_xlabel("$x$")
    # ax.set_ylabel("$i$th moment")
    if use_legend:
        ax.legend()
    ax.set_yscale(scale)
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_matrix(mat, title=None, ylabel=None, colour_norm=None, cmap=None, show_colorbar=True, ax=None, save_path=None):
    createdAx = False
    if ax is None:
        createdAx = True
        print("Created fig")
        fig = plt.figure()

        ax = fig.subplots(1, 1)

    if cmap is None:
        cmap = plt.get_cmap("plasma")
        cmap.set_bad(color="red")

    if colour_norm is None:
        colour_norm = Normalize()

    plotted = ax.matshow(
        mat,
        cmap=cmap,
        norm=colour_norm,
        interpolation="bilinear",
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize="large")
    if createdAx and show_colorbar:
        fig.colorbar(plotted, ax=ax)

    if createdAx and save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return plotted


def plot_correlation_grid(gp_draws, vae_draws, matrix_orders=[1, 2, 3, 4, 5]):
    from plotting.plots import plot_matrix
    from reusable.moments import correlation

    gp_mats = []
    vae_mats = []

    for order in matrix_orders:
        gp_mats.append(correlation(gp_draws, order))# jnp.log()
        vae_mats.append(correlation(vae_draws, order)) # jnp.log()
    vmin = 0.01#  min(jnp.nanmin(jnp.array(gp_mats)), jnp.nanmin(jnp.array(vae_mats)))
    vmax = max(jnp.nanmax(jnp.array(gp_mats)), jnp.nanmax(jnp.array(vae_mats)))
    print(vmin, vmax)
    fig = plt.figure()
    fig.set_size_inches(6, 3)
    axs = AxesGrid(
        fig,
        (1, 1, 1),
        nrows_ncols=(2, len(matrix_orders)),
        label_mode="L",
        share_all=True,
        axes_pad=0.12,
        aspect=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad="2%",
    )

    cmap = plt.get_cmap("plasma")
    cmap.set_bad(color="red")

    norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.001) # Normalize(vmin=vmin, vmax=vmax) #

    for k, order in enumerate(matrix_orders):
        plot_matrix(
            gp_mats[k],
            title=f"$ f^{order}$",
            ylabel="GP" if k == 0 else None,
            colour_norm=norm,
            cmap=cmap,
            ax=axs[k],
        )

        out = plot_matrix(
            vae_mats[k],
            ylabel="VAE" if k == 0 else None,
            colour_norm=norm,
            cmap=cmap,
            ax=axs[len(matrix_orders) + k],
        )

    #out.set_clim(vmin=0)

    axs.cbar_axes[0].colorbar(out) #, ticks=matplotlib.ticker.LinearLocator(6))
    #axs.cbar_axes[0].set_yticklabels([f"{x:.3f}" for x in onp.exp(axs.cbar_axes[0].get_yticks())])

    return fig


def plot_times_graph(times, x, curve_labels, x_label, legend_title, title, is_relative=False, ax=None, save_path=None):
    x = onp.array(x)
    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

    for i, label in enumerate(curve_labels):
        data = times[i]
        nans = onp.isnan(data)
        ax.plot(x[~nans], (times[i])[~nans], label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("time difference, minutes" if is_relative else "time, minutes")
    ax.set_title(title)
    ax.legend(title=legend_title, loc="upper left")

    if is_relative:
        ax.yaxis.set_major_formatter("{x:+.0f}")
    else:
        ax.yaxis.set_major_formatter("{x:.0f}")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_scores_graph(
    times,
    x,
    curve_labels,
    x_label,
    y_label,
    legend_title,
    title,
    is_relative=False,
    ax=None,
    save_path=None,
    num_decimals=0,
    plot_range=None,
):
    x = onp.array(x)
    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

    for i, label in enumerate(curve_labels):
        data = times[i]
        nans = onp.isnan(data)
        ax.plot(x[~nans], (times[i])[~nans], label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if plot_range is None:
        quant = onp.nanpercentile(times, 90, axis=None)
        m = onp.nanmax(times, axis=None)
        if quant < 0.4 * m:
            top = quant
        else:
            top = m
        plot_range = [0, 1.2 * top]
    print(title, x_label, plot_range)

    ax.set_ylim(plot_range)
    ax.set_title(title)
    ax.legend(title=legend_title, loc="upper right")

    if is_relative:
        ax.yaxis.set_major_formatter(f"{{x:+.{num_decimals}f}}")
    else:
        ax.yaxis.set_major_formatter(f"{{x:.{num_decimals}f}}")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_times_matrix(mat, yticks, xticks, ylabel, xlabel, title, upper_limit=None, fig=None, save_path=None):
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot()

    current_cmap = plt.get_cmap()
    current_cmap.set_bad(color="red")
    if upper_limit is not None:
        current_cmap.set_over("orange")
    if upper_limit is not None:
        vmax = min(1.1 * onp.nanmax(mat), upper_limit)
    else:
        vmax = None
    plotted = ax.matshow(mat, cmap=current_cmap, vmax=vmax)

    ax.xaxis.set_label_position("top")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(onp.arange(0, len(xticks)), xticks)
    ax.set_yticks(onp.arange(0, len(yticks)), yticks)

    fig.colorbar(plotted, ax=ax, extend="max")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_score_contours(mat, yticks, xticks, ylabel, xlabel, title, upper_limit=None, fig=None, save_path=None):
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot()

    current_cmap = plt.get_cmap()
    current_cmap.set_bad(color="red")
    if upper_limit is not None:
        current_cmap.set_over("orange")
    if upper_limit is not None:
        vmax = min(1.1 * onp.nanmax(mat), upper_limit)
    else:
        vmax = None

    plotted = ax.contourf(mat, corner_mask=True)
    ax.contour(plotted, cmap=current_cmap, vmax=vmax)

    ax.xaxis.set_label_position("bottom")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(onp.arange(0, len(xticks)), xticks)
    ax.set_yticks(onp.arange(0, len(yticks)), yticks)

    fig.colorbar(plotted, ax=ax, extend="max")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_mmd_matrix(mat, mask, yticks, xticks, ylabel, xlabel, title, fig=None, save_path=None):
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot()

    mat = onp.array(mat)
    mask = onp.array(mask)

    masked_times = onp.ma.array(mat, mask=mask)

    current_cmap = plt.get_cmap("gist_yarg")
    current_cmap.set_bad(color="red")

    plotted = ax.matshow(masked_times, cmap=current_cmap)

    ax.xaxis.set_label_position("top")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(onp.arange(0, len(xticks)), xticks, rotation=45, ha="left")
    ax.set_yticks(onp.arange(0, len(yticks)), yticks)

    fig.colorbar(plotted, ax=ax)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_2d_one_draw(draw, title, mask=None, fig=None, save_path=None, vmin=None, vmax=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(title)

    ax = fig.subplots(1)

    current_cmap = plt.get_cmap()
    current_cmap.set_bad(color="red")

    if mask is not None:
        mat = onp.array(draw)
        mask = onp.array(mask)

        plotted = ax.imshow(onp.ma.array(mat, mask=mask), cmap=current_cmap)
    else:
        plotted = ax.imshow(draw, cmap=current_cmap)

    ax.axis("off")
    fig.colorbar(plotted, ax=ax)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return plotted.get_clim()


def plot_2d_draws(draws, num_to_plot, num_per_row, title, fig=None, save_path=None):
    if fig is None:
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(title)

    axs = fig.subplots(math.ceil(num_to_plot / num_per_row), num_per_row)

    current_cmap = plt.get_cmap()
    current_cmap.set_bad(color="red")

    vmin = jnp.min(draws[:num_to_plot], None)
    vmax = jnp.max(draws[:num_to_plot], None)

    for i, ax in enumerate(axs.flat):
        plotted = ax.imshow(draws[i], cmap=current_cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

    fig.colorbar(plotted, ax=axs.ravel().tolist())

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
