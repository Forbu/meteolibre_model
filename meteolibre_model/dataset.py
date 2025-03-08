import os
import datetime
import time

# import Image from PIL
from PIL import Image

import torch
import torch.utils.data as Dataset
import pandas as pd
import numpy as np

import h5py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

columns_measurements = [
    "RR1",
    "FF",
    "DD",
    "FXY",
    "DXY",
    "HXY",
    "QHXY",
    "FXI",
    "DXI",
    "HXI",
    "FXI3S",
    "HFXI3S",
    "T",
    "TN",
    "HTN",
    "TX",
    "HTX",
    "DG",
    "QDG",
]

columns_positions = ["position_x", "position_y"]


def transform_groundstation_data_into_image(df):
    lat = torch.tensor(df["position_x"].values, dtype=torch.int64, device=DEVICE)
    lon = torch.tensor(df["position_y"].values, dtype=torch.int64, device=DEVICE)

    nb_channels = len(columns_measurements)
    image_result = (
        torch.ones((3472, 3472, nb_channels), dtype=torch.float32, device=DEVICE) * -100
    )
    measurements = df[columns_measurements].values

    image_result[lon, lat, :] = torch.tensor(measurements)
    mask = image_result != -100

    return mask, image_result


class MeteoLibreDataset(Dataset.Dataset):
    def __init__(
        self,
        index_file,
        dir_index,
        groundstations_info,
        ground_height_image,
        nb_back_steps=3,
        nb_future_steps=1,
    ):
        self.index = pd.read_parquet(index_file)

        self.dir_index = dir_index

        self.groundstations_info = groundstations_info
        self.groundstations_info_df = pd.read_parquet(groundstations_info)

        self.groundstations_info_df = self.groundstations_info_df[
            columns_measurements + columns_positions + ["datetime"]
        ]

        # set index to datetime
        self.groundstations_info_df = self.groundstations_info_df.set_index("datetime")

        print(self.groundstations_info_df.head())

        self.groundstations_info_df = self.groundstations_info_df.fillna(-100)

        self.nb_back_steps = nb_back_steps
        self.nb_future_steps = nb_future_steps

        # manage gorund height image (read the .npy file)
        self.ground_height_image = np.load(ground_height_image)

        print(self.ground_height_image.shape)

    def __len__(self):
        return len(self.index) - self.nb_back_steps - self.nb_future_steps

    def __getitem__(self, index):
        index = int(index + self.nb_back_steps)
        dict_return = {}

        for back in range(self.nb_back_steps):
            path_file = os.path.join(
                self.dir_index, str(self.index["file_path_h5"].iloc[index - back - 1])
            )

            array = np.array(h5py.File(path_file, "r")["dataset1"]["data1"]["data"])
            array[array == array.max()] = 0

            dict_return["back_" + str(back)] = array

        for future in range(self.nb_future_steps):
            path_file = os.path.join(
                self.dir_index, str(self.index["file_path_h5"].iloc[index + future])
            )

            array = np.array(h5py.File(path_file, "r")["dataset1"]["data1"]["data"])
            array[array == array.max()] = 0

            dict_return["future_" + str(future)] = array

        current_date = self.index["datetime"].iloc[index]

        if current_date.minute != 0:
            round_date_previous = datetime.datetime(
                current_date.year,
                current_date.month,
                current_date.day,
                current_date.hour,
                0,
                0,
            )
        else:
            round_date_previous = current_date - datetime.timedelta(hours=1)

        round_date_next = round_date_previous + datetime.timedelta(hours=1)

        df_ground_station_previous = self.groundstations_info_df.loc[
            round_date_previous
        ]
        df_ground_station_next = self.groundstations_info_df.loc[round_date_next]

        mask_previous, ground_station_image_previous = (
            transform_groundstation_data_into_image(df_ground_station_previous)
        )
        mask_next, ground_station_image_next = transform_groundstation_data_into_image(
            df_ground_station_next
        )

        dict_return["ground_station_image_previous"] = ground_station_image_previous
        dict_return["ground_station_image_next"] = ground_station_image_next
        dict_return["mask_previous"] = mask_previous
        dict_return["mask_next"] = mask_next
        dict_return["hour"] = current_date.hour

        # dd ground height image
        dict_return["ground_height_image"] = self.ground_height_image

        return dict_return
