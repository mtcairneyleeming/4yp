"""
A collection of scoring methods, e.g. MMD, Frobenius norms


"""

from .moments import correlation, sample_central_moment
from .loss import MMD_rbf, MMD_rqk
import jax.numpy as jnp


def calc_moments(draws, orders=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    return [sample_central_moment(i, draws) for i in orders]


def calc_correlation_mats(draws, orders=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    return [correlation(draws, o) for o in orders]


def calc_frob_norms(vae_mats, gp_mats):

    assert len(gp_mats) == len(vae_mats)

    return [jnp.linalg.norm(v - g, ord="fro") for v, g in zip(vae_mats, gp_mats)]


MMD_SCORING_KERNELS = [
    MMD_rbf(1.0),
    MMD_rbf(4.0),
    MMD_rbf(10.0),
    MMD_rqk(4.0, 0.25),
    MMD_rqk(8.0, 0.25),
    MMD_rqk(4.0, 1),
    MMD_rqk(8.0, 1),
    MMD_rqk(4.0, 10),
    MMD_rqk(8.0, 10),
    MMD_rqk(4.0, 100),
    MMD_rqk(8.0, 100),
]


def calc_mmd_scores(gp_draws, vae_draws, kernels=MMD_SCORING_KERNELS):
    return [(mmd.__name__, mmd(gp_draws, vae_draws)) for mmd in MMD_SCORING_KERNELS]
