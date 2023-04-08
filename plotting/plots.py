import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp
import numpyro.diagnostics
import math


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


def plot_draws_hpdi(draws, x, title, ylabel, legend_label, ax=None, save_path=None, _min=-2, _max=2):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)


    lines_alpha = 0.1
    N_lines = 15

    # -----------------------
    draws = draws[~jnp.isnan(draws).any(axis=1), :]
    mean = jnp.nanmean(draws, axis=0)
    hpdi = numpyro.diagnostics.hpdi(draws, 0.9)

    for j in range(1, N_lines):
        ax.plot(x, draws[j, :], alpha=lines_alpha, color="darkgreen", label="")
    # separate from other GP draws to label it
    ax.plot(x, draws[0, :], alpha=lines_alpha, color="darkgreen", label=f"{legend_label} draws")

    ax.fill_between(x, hpdi[0], hpdi[1], alpha=0.1, interpolate=True, label="95% HPDI")
    ax.plot(x, mean, label="mean")
    ax.legend(loc=4)
    ax.set_ylim([_min, _max])
    ax.set_xlabel("$x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def quick_compare_draws(
    x,
    draws1,
    draws2,
    title1="Examples of priors we want to encode",
    title2="Priors learnt by VAE",
    ylabel1="$y=f_{GP}(x)$",
    ylabel2="$y=f_{VAE}(x)$",
    save_path=None,
):

    # plot results
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    plot_draws(draws1[:10], x, title1, ylabel1, ax=axs[0])
    plot_draws(draws2, x, title2, ylabel2, ax=axs[1])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def compare_draws(x, draws1, draws2, title1, title2, ylabel1, ylabel2, legend_label1, legend_label2, save_path=None):

    # plot results
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

    plot_draws_hpdi(draws1, x, title1, ylabel1, legend_label1, axs[0])
    plot_draws_hpdi(draws2, x, title2, ylabel2, legend_label2, axs[1])

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


def plot_one_inference(inferred_priors, x, ground_truth, x_obs, y_obs, title, ylabel, legend_label, ax=None, save_path=None, _min=-2, _max=2, legend=True):
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
        label=f"posterior: 95% BCI",
    )
    for j in range(N_lines):
        ax.plot(
            x,
            inferred_priors[j, :],
            alpha=0.1,
            color="darkgreen",
            label=f"{legend_label} posterior draws" if j == 0 else "",
        )

    ax.plot(x, mean, label="predicted mean")
    ax.plot(x, ground_truth, label="ground truth", color="orange")
    ax.scatter(x_obs, y_obs, color="red", label="observed data", s=60)
    ax.set_title(title)
    if legend:
        ax.legend(loc=4)
    ax.set_ylim([_min, _max])
    ax.set_xlabel("$x$")
    ax.set_ylabel(ylabel)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")




def compare_inference_steps(
    x, ground_truth, x_obss, y_obss, plain_prior_samples, inferred_priors_list, title="VAE", fig=None, save_path=None
):
    if fig is None:
        fig = plt.figure(figsize=(15, 4)) # should be 

    # plot results
    axs = fig.subplots(nrows=1, ncols=len(inferred_priors_list) + 1)

    plot_draws_hpdi(plain_prior_samples, x, f"{title} draws", f"$f_{{{title}}}(x)", title, ax=axs[0])

    for i in range(len(inferred_priors_list)):
        plot_one_inference(
            inferred_priors_list[i],
            x,
            ground_truth,
            x_obss[i],
            y_obss[i],
            f"{len(x_obss[i])} observations",
            "$y=f_{VAE}(x)$",
            "VAE",
            ax=axs[i + 1],
            legend = (i == 0)
        )
    fig.tight_layout()

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


def plot_moments(moments, moment_indices, x_locs, title, correct_moments=None, scale="linear", use_legend=False, ax=None, save_path=None):
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
    #ax.set_ylabel("$i$th moment")
    if use_legend:
        ax.legend()
    ax.set_yscale(scale)
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_matrix(mat, title=None, ylabel = None, vmin=None, vmax=None, show_colorbar=True, ax=None, save_path=None):
    createdAx = False
    if ax is None:
        createdAx = True
        print("Created fig")
        fig = plt.figure()

        ax = fig.subplots(1, 1)  # add_axes([.1, .1, 0.8, 0.8])

    current_cmap = plt.get_cmap("plasma")
    current_cmap.set_bad(color="red")

    plotted = ax.matshow(mat, cmap=current_cmap, vmin=vmin, vmax=vmax, norm="log")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize="large")
    if createdAx and show_colorbar:
        fig.colorbar(plotted, ax=ax)

    if createdAx and save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return plotted


def plot_correlation_grid(gp_draws, vae_draws, matrix_orders = [1,2,3,4,5]):
    from plotting.plots import plot_matrix
    from reusable.moments import correlation
    from matplotlib.ticker import LogFormatter

    gp_mats = []
    vae_mats = [] 

    for order in matrix_orders:
        gp_mats.append(correlation(gp_draws, order))
        vae_mats.append(correlation(vae_draws, order))

    vmin = max(0.001, min([jnp.min(m, axis=None) for m in gp_mats] + [jnp.min(m, axis=None) for m in vae_mats]))
    vmax =  max([jnp.max(m, axis=None) for m in gp_mats]+[jnp.max(m, axis=None) for m in vae_mats])

    fig = plt.figure(figsize=(2 * len(matrix_orders), 4))
    axs = fig.subplots(2, len(matrix_orders))

    out = None
    for k, order in enumerate(matrix_orders):
        plot_matrix(gp_mats[k], title= f"$ f^{order}$", ylabel="GP" if k ==0 else None, vmin=vmin, vmax=vmax, ax=axs[0,k])
        
        out = plot_matrix(vae_mats[k], ylabel ="VAE" if k == 0 else None, vmin=vmin, vmax=vmax,  ax=axs[1,k])

    fig.subplots_adjust(right=0.925, left=0)
    cbar_ax = fig.add_axes([0.95, 0.125, 0.025, 0.75])
    formatter = plt.LogFormatter(10, labelOnlyBase=False) 
    #cb = plt.colorbar(ticks=[1,5,10,20,50], format=formatter)
    cbar = fig.colorbar(out, cax=cbar_ax, ticks=[0,1,10,100,1000], format=formatter)
    #cbar.ax.locator_params([1,2,3,4])

    return fig

def plot_times_graph(times, x, curve_labels, x_label, legend_title, title, is_relative=False, ax=None, save_path=None):
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)

    for i, label in enumerate(curve_labels):
        ax.plot(x, times[i], label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("time difference, minutes" if is_relative else "time, minutes")
    ax.set_title(title)
    ax.legend(title=legend_title, loc="upper left")

    if is_relative:
        ax.yaxis.set_major_formatter('{x:+.0f}')
        #ax.yaxis.set_major_formatter(lambda x, pos: ("+" if x > 0 else "") + str(x) + "s")

    else:
        ax.yaxis.set_major_formatter('{x:.0f}')
        #ax.yaxis.set_major_formatter(lambda x, pos: str(x) + "s")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_times_matrix(mat, mask, yticks, xticks, ylabel, xlabel, title, fig=None, save_path=None):
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot()

    mat = onp.array(mat)
    mask = onp.array(mask)

    masked_times = onp.ma.array(mat, mask=mask)

    current_cmap = plt.get_cmap()
    current_cmap.set_bad(color="red")

    plotted = ax.matshow(masked_times, cmap=current_cmap)

    ax.xaxis.set_label_position("top")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(onp.arange(0, len(xticks)), xticks)
    ax.set_yticks(onp.arange(0, len(yticks)), yticks)

    fig.colorbar(plotted, ax=ax)

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
