import os
import datetime
import time

# import Image from PIL
from PIL import Image

import torch
import torch.utils.data as Dataset
import pandas as pd
import numpy as np

import random

import h5py

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

RADAR_NORMALIZATION = 60.0
DEFAULT_VALUE = -1

columns_measurements = [
    "RR1",
    # "FF",
    # "DD",
    # "FXY",
    # "DXY",
    # "HXY",
    # "QHXY",
    # "FXI",
    # "DXI",
    # "HXI",
    # "FXI3S",
    # "HFXI3S",
    "T",
    # "TN",
    # "HTN",
    # "TX",
    # "HTX",
    # "DG",
    # "QDG",
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

    image_result[lon, lat, :] = torch.tensor(measurements, device=DEVICE)
    mask = image_result != -100

    return mask, image_result


class MeteoLibreDatasetPartialGrid(Dataset.Dataset):
    def __init__(
        self,
        index_file,
        dir_index,
        ground_height_image,
        nb_back_steps=3,
        nb_future_steps=1,
        shape_image=3472,
    ):
        self.index = pd.read_parquet(index_file)

        # sort by datetime
        self.index = self.index.sort_values(by="datetime")

        # set datetime as index
        self.index = self.index.set_index("datetime")

        self.dir_index = dir_index

        print(self.index.head())

        self.nb_back_steps = nb_back_steps
        self.nb_future_steps = nb_future_steps
        self.shape_image = shape_image

        # manage gorund height image (read the .npy file)
        self.ground_height_image = np.load(ground_height_image)
        self.ground_height_image = (
            self.ground_height_image - np.mean(self.ground_height_image)
        ) / np.std(self.ground_height_image)

        print(self.ground_height_image.shape)

    def __len__(self):
        return len(self.index) - self.nb_back_steps - self.nb_future_steps

    def __getitem__(self, index):
        index = int(index + self.nb_back_steps)

        current_date = self.index.index[index]

        dict_return = {}

        # now for every image, we select only a random 512x512 patch
        x = random.randint(0, self.shape_image // 2 - 512)
        y = random.randint(0, self.shape_image // 2 - 512)

        for future in range(self.nb_future_steps):
            path_file = os.path.join(
                self.dir_index, str(self.index["radar_file_path"].iloc[index + future])
            )

            # take
            array = np.array(
                h5py.File(path_file, "r")["dataset1"]["data1"]["data"][
                    x : (x + 1024) : 2, y : (y + 1024) : 2
                ]
            )

            array = array.astype(np.int32)

            array[array == 65535] = -DEFAULT_VALUE

            # if there is nothing > 0, we go on the next item
            if np.sum(array > 0.1) <= 10:
                # print("not enaught good point")
                return self[random.randint(0, len(self) - 1)]

            array = np.float32(array) / RADAR_NORMALIZATION  # normalization

            dict_return["future_" + str(future)] = array
            dict_return["mask_future_" + str(future)] = array != (
                -DEFAULT_VALUE / RADAR_NORMALIZATION
            )

        for back in range(self.nb_back_steps):
            path_file = os.path.join(
                self.dir_index,
                str(self.index["radar_file_path"].iloc[index - back - 1]),
            )
            # check if the delta with the current time is not too high
            time_back = self.index.index[index - back - 1]
            delta_time = current_date - time_back

            # convert delta time in minutes
            delta_time_minutes = delta_time.total_seconds() / 60

            if delta_time <= datetime.timedelta(hours=2):
                array = np.array(
                    h5py.File(path_file, "r")["dataset1"]["data1"]["data"][
                        x : (x + 1024) : 2, y : (y + 1024) : 2
                    ]
                )
                array = array.astype(np.int32)
                array[array == 65535] = -DEFAULT_VALUE

            else:
                # print("bad delta time", delta_time)
                array = np.ones((512, 512), dtype=np.float32) * -DEFAULT_VALUE

            array = np.float32(array) / RADAR_NORMALIZATION  # normalization

            dict_return["back_" + str(back)] = array
            # dict_return["mask_back_" + str(back)] = array != (-DEFAULT_VALUE / RADAR_NORMALIZATION)
            dict_return["time_back_" + str(back)] = delta_time_minutes / 60.0

        dict_return["hour"] = np.int32(current_date.hour) / 24.0
        dict_return["minute"] = np.int32(current_date.minute) / 30.0

        # dd ground height image
        dict_return["ground_height_image"] = self.ground_height_image[
            x : (x + 1024) : 2, y : (y + 1024) : 2
        ]

        return dict_return
