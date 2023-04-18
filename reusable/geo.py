
import geopandas as gpd
import matplotlib.pyplot as plt 
import jax.numpy as jnp
import numpy as onp
import pyreadr
import pandas as pd

def state_name(st):
    if st == 36:
        return "New York", "NY"
    
    return "Unknown", "UN"


def load_state_boundaries(st):
    return gpd.read_file(f"data/state_{st}_boundaries.geojson")


# note this is sorted by ZCTA 
def load_state_centroids(st):
    return gpd.read_file(f"data/state_{st}_centroids.geojson")
    
def get_all_temp_data(st, year):
    return pd.read_pickle(f"data/state_{st}_{year}_temps.pkl")


def get_temp_data(st, year, aggr="year_max"):
    return pd.read_pickle(f"data/state_{st}_{year}_{aggr}_temps.pkl")

def centroids_to_coords(centroids):
    x_coords = jnp.array(centroids.geometry.apply(lambda x: x.x))
    y_coords = jnp.array(centroids.geometry.apply(lambda x: x.y))
    x_coords = x_coords - jnp.mean(x_coords)
    y_coords = y_coords - jnp.mean(y_coords)
    return jnp.dstack((x_coords, y_coords))[0]

