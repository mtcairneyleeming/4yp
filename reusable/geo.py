
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

def load_state_centroids(st):
    return gpd.read_file(f"data/state_{st}_centroids.geojson")
    

def get_temp_data(st, year, aggr="year_max"):
    return pd.read_pickle(f"data/state_{st}_{year}_{aggr}.pkl")