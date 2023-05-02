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


def get_all_temp_data(st, year) -> pd.DataFrame:
    return pd.read_pickle(f"data/state_{st}_{year}_temps.pkl")


def get_temp_data(st, year, aggr="year_max") -> pd.DataFrame:
    return pd.read_pickle(f"data/state_{st}_{year}_{aggr}_temps.pkl")


# def save_year_temps(state, year): # creates 1794 entries
#     file = pyreadr.read_r(f"data/PRISM/ny_tmean/weighted_area_raster_zip_{state}_tmean_daily_{year}.rds")
#     df = pd.DataFrame(file[None])
#     df.to_pickle(f"data/state_{state}_{year}_temps.pkl")
# def max_yearly_mean_temp(state, year): # creates 1794 entries
#     file = pyreadr.read_r(f"data/PRISM/ny_tmean/weighted_area_raster_zip_{state}_tmean_daily_{year}.rds")
#     df = pd.DataFrame(file[None])
#     df= df[df["tmean"] == df.groupby('zcta')["tmean"].transform("max")]
#     df = df.sort_values("zcta")
#     df.to_pickle(f"data/state_{state}_{year}_year_max_temps.pkl")
#     return df

# def mean_mean_temp(state, year): # creates 1794 entries
#     df = get_all_temp_data(state, year)
#     df = df.groupby('zcta').mean("tmean")
#     df = df.sort_values("zcta")
#     df.to_pickle(f"data/state_{state}_{year}_mean_temps.pkl")
#     return df


def get_processed_temp_data(st, year, aggr):
    """"""
    df = get_all_temp_data(st, year)
    means = df["tmean"].mean()
    if aggr == "mean":
        df = df.groupby("zcta").mean("tmean")

    elif aggr == "year_max":
        df = df[df["tmean"] == df.groupby("zcta")["tmean"].transform("max")]

    else:

        day_of_year = int(aggr)
        print(day_of_year)
        date = pd.to_datetime(day_of_year - 1, unit="D", origin=str(year))
        df = df[(df["day"] == f"{date.day:02}")]
        df = df[(df["month"] == f"{date.month:02}")]
        print(df)

    df = df.sort_values("zcta")

    return df["tmean"].to_numpy() - means, means


def centroids_to_coords(centroids, scaling_factor):
    x_coords = jnp.array(centroids.geometry.apply(lambda x: x.x))
    y_coords = jnp.array(centroids.geometry.apply(lambda x: x.y))
    mean_x = jnp.mean(x_coords)
    mean_y = jnp.mean(y_coords)
    x_coords = x_coords - mean_x
    y_coords = y_coords - mean_y
    return jnp.dstack((x_coords, y_coords))[0] / scaling_factor, (mean_x, mean_y)
