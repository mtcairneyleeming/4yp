import geopandas as gpd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reusable.geo import load_state_boundaries
from mpl_toolkits.axes_grid1 import AxesGrid
from shapely.geometry import Point
from plotting.helpers import (
    align_left_backfill,
    align_right_backfill,
    calc_plot_dimensions,
    clear_unused_axs,
    align_right_backfill_with_gp,
    pretty_loss_fn_name,
)
from .consts import *

class StateLoader(object):
    states = {}

    def load_state_boundaries(self, state: int):
        if state not in self.states:
            self.states[state] = load_state_boundaries(state)
        return self.states[state]


loader = StateLoader()


def plot_on_state(
    data,
    state,
    legend_title,
    title,
    ax=None,
    vmin=None,
    vmax=None,
    cmap=None,
    show_colorbar=True,
    edge_highlight_indices=None,
    points_to_plot=None,
    save_file_name=None, 
):

    ax_was_none = ax is None
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.subplots(1, 1)

    
    ax.set_aspect('equal')

    if title is not None:
        ax.set_title(title, size=15)

    geom = loader.load_state_boundaries(state)["geometry"]
    newframe = gpd.GeoDataFrame({"d": data}, geometry=geom)
    if cmap is None:
        cmap = plt.get_cmap()
        cmap.set_bad(color="red")

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.001)
    else:
        cax = None

    if points_to_plot is not None:
        # assuming in the same units as existing data
        points_to_plot = [Point(x) for x in points_to_plot]

        point_frame = gpd.GeoDataFrame(geometry=points_to_plot)


        point_frame.plot(ax=ax, marker='o', color='orange', markersize=1, zorder=2)


    newframe.plot(
        "d", ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cax=cax, legend=show_colorbar, legend_kwds={"label": legend_title}
    )

    if edge_highlight_indices is not None:
        newframe.loc[edge_highlight_indices].plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.5)

    ax.set_axis_off()

    if ax_was_none and  save_file_name is not None:
        fig.savefig("gen_plots/" + save_file_name + ".png", dpi=300)


def plot_multi_on_state(
    datas, state,  legend_title, suptitle=None, titles=None, fig=None, num_in_row=4, edge_highlight_indices=None,
    points=None,
    save_file_name=None, 
    backfill=None,
    show_cbar=True
):
    vmin = jnp.nanmin(datas, None)
    vmax = jnp.nanmax(datas, None)

    if datas.shape[0] <= num_in_row:
        num_in_row = datas.shape[0]

    num_rows = (datas.shape[0] + num_in_row - 1) // num_in_row

    match backfill:
        case None:
            mapping = lambda i: i
        case "align_left":
            mapping = align_left_backfill(len(datas), num_rows, num_in_row)
        case "align_right":
            mapping = align_right_backfill(len(datas), num_rows, num_in_row)

    print(num_rows, num_in_row)

    if fig is None:
        fig = plt.figure(figsize=(1.5 * PAGE_WIDTH, 1.5 * num_rows * PAGE_WIDTH / num_in_row))

    if suptitle is not None:
        fig.suptitle(suptitle)

    

    axs = AxesGrid(
        fig,
        111,
        nrows_ncols=(num_rows, num_in_row),
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="single" if show_cbar else None,
        cbar_size="5%",
        cbar_pad="2%",
        axes_pad=(0.02, 0.08)
    )
    

    cmap = plt.get_cmap()
    cmap.set_bad(color="red")

    if points is not None:
        assert len(points) >= datas.shape[0]
    else:
        points = [None] * datas.shape[0]

    for i in range(datas.shape[0]):
        title = titles[i] if titles is not None else None

        plot_on_state(
            datas[i],
            state,
            None,
            title,
            ax=axs[mapping(i)],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show_colorbar=False,
            edge_highlight_indices=edge_highlight_indices,
            points_to_plot = points[i]
        )

    if show_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

        # fake up the array of the scalar mappable. Urgh...
        sm._A = []


        c = axs.cbar_axes[0].colorbar(sm, label=legend_title)
        c.set_label(legend_title, size=15)
 

    clear_unused_axs(axs, mapping, len(datas))

    if save_file_name is not None:
        fig.savefig("gen_plots/" + save_file_name + ".png", dpi=300, bbox_inches=0)


def mask_for_plotting(data, visible_indices):
    out = onp.full_like(data, onp.nan)
    out[visible_indices] = data[visible_indices]
    return out
