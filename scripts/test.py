# global package
import polars as pl
import numpy as np
import h5py
from pyproj import Transformer 
import datetime

import matplotlib.pyplot as plt

# read the total.parquet file
df = pl.read_parquet("/home/adrienbufort/data/total_float32.parquet")
print(df.head())

# also remove raws which have datetime < 2025-01-01 (in datetime format)
df = df.filter(pl.col("datetime") >= datetime.datetime(2025, 1, 1))

EPSG = "32630"

# now we want to apply the transformer to the positions of the stations
transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}")

# Extract LAT and LON as numpy arrays for the transformer
lat_array = df["LAT"].to_numpy()
lon_array = df["LON"].to_numpy()
positions_stations_transformed = transformer.transform(lat_array, lon_array)

# create two new columns in the dataframe
df = df.with_columns(
    pl.Series(name="LAT_transformed", values=positions_stations_transformed[0]),
    pl.Series(name="LON_transformed", values=positions_stations_transformed[1])
)

# get the coordinates of the center of the map
center_map_x, center_map_y = transformer.transform(45.9, 3.2)

# now put the coordinates of the stations in the dataframe
df = df.with_columns(
    ((pl.col("LAT_transformed") - center_map_x) // 500 + 3472//2).alias("position_x"),
    (-(pl.col("LON_transformed") - center_map_y) // 500 + 3472//2).alias("position_y")
)

print(df.select(["position_x", "position_y", "LAT", "LON"]).head())

# filter element not in [0, 3472]
df = df.filter(
    (pl.col("position_x") >= 0) & 
    (pl.col("position_x") <= 3472) & 
    (pl.col("position_y") >= 0) & 
    (pl.col("position_y") <= 3472)
)


# replace NaN and null by -100
df = df.fill_null(-100)

# save it somewhere
df.write_parquet("/home/adrienbufort/data/total_transformed.parquet")
