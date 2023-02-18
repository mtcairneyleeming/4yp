import matplotlib.pyplot as plt
import jax.numpy as jnp
from numpyro.diagnostics import hpdi


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


def plot_draws_hpdi(draws, x, title, ylabel, ax=None, save_path=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    _min, _max = -2, 2
    lines_alpha = 0.1
    N_lines = 15

    # -----------------------
    draws = draws[~jnp.isnan(draws).any(axis=1), :]
    mean1 = jnp.nanmean(draws, axis=0)
    hpdi1 = hpdi(draws, 0.9)

    for j in range(1, N_lines):
        ax.plot(x, draws[j, :], alpha=lines_alpha, color="darkgreen", label="")
    # separate from other GP draws to label it
    ax.plot(x, draws[0, :], alpha=lines_alpha, color="darkgreen", label="GP draws")

    ax.fill_between(x, hpdi1[0], hpdi1[1], alpha=0.1, interpolate=True, label="95% HPDI")
    ax.plot(x, mean1, label="mean")
    ax.legend(loc=4)
    ax.set_ylim([_min, _max])
    ax.set_xlabel("$x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


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


def compare_draws(x, draws1, draws2, title1, title2, ylabel1, ylabel2, save_path=None):

    # plot results
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

    plot_draws_hpdi(draws1, x, title1, ylabel1, axs[0])
    plot_draws_hpdi(draws2, x, title2, ylabel2, axs[1])

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


def plot_one_inference(x, ground_truth, x_obs, y_obs, inferred_priors, title, ax=None, save_path=None):
    if ax is None:
        fig = plt.figure(figsize=(7, 5))

        ax = fig.add_subplot(111)

    N_lines = 15

    mean_post_pred = jnp.mean(inferred_priors, axis=0)
    hpdi_post_pred = hpdi(inferred_priors, 0.9)

    ax.fill_between(
        x,
        hpdi_post_pred[0],
        hpdi_post_pred[1],
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
            label=f"posterior draws" if j == 0 else "",
        )

    ax.plot(x, mean_post_pred, label="predicted mean")
    ax.plot(x, ground_truth, label="ground truth", color="orange")
    ax.scatter(x_obs, y_obs, color="red", label="observed data", s=60)
    ax.set_title(title)
    ax.legend(loc=4)
    ax.set_ylim([-2.5, 2.5])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_ground_truth_with_priors(x, ground_truth, plain_prior_samples, title, ax=None, save_path=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 5))

        ax = fig.add_subplot(111)

    N_lines = 30

    mean_plain = jnp.mean(plain_prior_samples, axis=0)
    hpdi_plain = hpdi(plain_prior_samples, 0.9)

    N_lines = 30
    for j in range(N_lines):
        ax.plot(
            x,
            plain_prior_samples[j, :],
            alpha=0.1,
            color="darkgreen",
            label="prior draws" if j == 0 else "",
        )

    ax.fill_between(
        x,
        hpdi_plain[0],
        hpdi_plain[1],
        alpha=0.1,
        interpolate=True,
        label=f"{title} prior: 95% BCI",
    )
    ax.plot(x, mean_plain, label="mean")
    ax.legend(loc=4)
    ax.set_ylim([-2.5, 2.5])
    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def compare_inference_steps(
    x, ground_truth, x_obss, y_obss, plain_prior_samples, inferred_priors_list, title="VAE", fig=None, save_path=None
):
    if fig is None:
        fig = plt.figure(figsize=(20, 5))

    # plot results
    axs = fig.subplots(nrows=1, ncols=len(inferred_priors_list) + 1)

    plot_ground_truth_with_priors(x, ground_truth, plain_prior_samples, title=f"{title} prior", ax=axs[0])

    for i in range(len(inferred_priors_list)):
        plot_one_inference(
            x,
            ground_truth,
            x_obss[i],
            y_obss[i],
            inferred_priors_list[i],
            title="n datapoints =" + str(len(x_obss[i])),
            ax=axs[i + 1],
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
