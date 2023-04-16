import geopandas as gpd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp

from reusable.geo import load_state_boundaries


class StateLoader(object):
    states = {}

    def load_state_boundaries(self, state: int):
        if state not in self.states:
            self.states[state] = load_state_boundaries(state)
        return self.states[state]
    

loader = StateLoader()


def plot_on_state(data, state, title, legend_title, ax=None, vmin=None, vmax=None, show_colorbar=True):

    ax_was_none = ax is None
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.subplots(1,1)
        
    if title is not None:
        ax.set_title(title)
    
    geom = loader.load_state_boundaries(state)["geometry"]
    newframe = gpd.GeoDataFrame({"d": data}, geometry=geom)
    cmap = plt.get_cmap()
    cmap.set_bad(color="red")
    newframe.plot("d", ax=ax , vmin=vmin, vmax=vmax, cmap=cmap)

    if ax_was_none and show_colorbar:
        fig.colorbar(fig.gca().get_children()[0], ax=ax, label=legend_title)

        


def plot_multi_on_state(datas, state, suptitle, legend_title, titles=None, fig=None, num_in_row=4):
    vmin = jnp.min(datas, None)
    vmax = jnp.max(datas, None)

    num_rows = (datas.shape[0] + num_in_row -1 )// num_in_row

    if fig is None:
        fig = plt.figure(figsize=(num_in_row * 6, num_rows * 6))

    axs = fig.subplots(num_rows, num_in_row)

    for i in range(datas.shape[0]):
        title = titles[i] if titles is not None else None

        plot_on_state(datas[i], state, title, None, ax=axs[i // num_in_row, i % num_in_row], vmin=vmin, vmax=vmax, show_colorbar=False)

    fig.colorbar(fig.gca().get_children()[0], ax=axs[0,0], label=legend_title)
    fig.suptitle(suptitle, fontsize=20)

    
def mask_for_plotting(data, visible_indices):
    out = onp.full_like(data, onp.nan)
    out[visible_indices] = data[visible_indices]
    return out