import geopandas as gpd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reusable.geo import load_state_boundaries
from mpl_toolkits.axes_grid1 import AxesGrid
from shapely.geometry import Point

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
    title,
    legend_title,
    ax=None,
    vmin=None,
    vmax=None,
    cmap=None,
    show_colorbar=True,
    edge_highlight_indices=None,
    points_to_plot=None
):

    ax_was_none = ax is None
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.subplots(1, 1)

    
    ax.set_aspect('equal')

    if title is not None:
        ax.set_title(title)

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


    print("PLotiting")
    newframe.plot(
        "d", ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cax=cax, legend=show_colorbar, legend_kwds={"label": legend_title}
    )

    if edge_highlight_indices is not None:
        newframe.loc[edge_highlight_indices].plot(ax=ax, facecolor="none", edgecolor="red", linewidth=0.5)


    
  

    #ax.set_axis_off()


def plot_multi_on_state(
    datas, state, suptitle, legend_title, titles=None, fig=None, num_in_row=4, edge_highlight_indices=None,
    points=None
):
    vmin = jnp.nanmin(datas, None)
    vmax = jnp.nanmax(datas, None)

    if datas.shape[0] <= num_in_row:
        num_in_row = datas.shape[0]

    num_rows = (datas.shape[0] + num_in_row - 1) // num_in_row

    if fig is None:
        fig = plt.figure(figsize=(num_in_row * 6, num_rows * 6))

    axs = AxesGrid(
        fig,
        111,
        nrows_ncols=(num_rows, num_in_row),
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad="2%",
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
            title,
            None,
            ax=axs[i],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show_colorbar=False,
            edge_highlight_indices=edge_highlight_indices,
            points_to_plot = points[i]
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # fake up the array of the scalar mappable. Urgh...
    sm._A = []

    # fig.colorbar(sm, ax=cax, label=legend_title)

    axs.cbar_axes[0].colorbar(sm, label=legend_title)
    # axs = fig.subplots(num_rows, num_in_row, squeeze=False)

    # divider = make_axes_locatable(axs.flat)
    # cax = divider.append_axes("right", size="5%", pad=0.001)

    # cmap = plt.get_cmap()
    # cmap.set_bad(color="red")

    # for i in range(datas.shape[0]):
    #     title = titles[i] if titles is not None else None

    #     plot_on_state(datas[i], state, title, None, ax=axs.flat[i], vmin=vmin, vmax=vmax, cmap=cmap, show_colorbar=False, edge_highlight_indices=edge_highlight_indices)
    # print(axs.shape)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # # fake up the array of the scalar mappable. Urgh...
    # sm._A = []

    # fig.colorbar(sm, ax=cax, label=legend_title)
    # # fig.colorbar(sm, ax=axs.ravel().tolist(), label=legend_title)
    # fig.suptitle(suptitle, fontsize=20)


def mask_for_plotting(data, visible_indices):
    out = onp.full_like(data, onp.nan)
    out[visible_indices] = data[visible_indices]
    return out
