"""
In this part the idea is the create a dataloader from an index dataframe
"""

import os
import datetime

import torch
import torch.utils.data as Dataset
import pandas as pd

# import package for h5 file
import h5py

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

columns_positions = ["LAT_transformed", "LON_transformed"]

def transform_groundstation_data_into_image(df):
    """
    Transform the groundstation data into an image to be compute
    df have columns :
    postions_info 'LAT_transformed', 'LON_transformed'

    measurements_info : 'RR1', 'FF', 'DD', 'FXY', 'DXY', 'HXY', 'QHXY', 'FXI',
              'DXI', 'HXI', 'FXI3S', 'HFXI3S',
              'T', 'TN', 'HTN', 'TX', 'HTX',
              'DG', 'QDG'
    """
    # get the coordinates of the groundstation
    lat = df["LAT_transformed"].values
    lon = df["LON_transformed"].values
    
    nb_channels = len(columns_measurements)

    # init the multiple channel image to 0
    image_result = torch.ones((3472, 3472, nb_channels)) * -100

    # for each ground station, we add the measurements to the image
    image_result[lat, lon, :] = df[columns_measurements].values
    
    mask = image_result != -100

    return mask, image_result

class MeteoLibreDataset(Dataset.Dataset):
    def __init__(
        self,
        index_file,
        dir_index,
        groundstations_info,
        nb_back_steps=3,
        nb_future_steps=1,
    ):
        self.index = pd.read_parquet(index_file)
        self.dir_index = dir_index
        self.groundstations_info = groundstations_info
        self.nb_back_steps = nb_back_steps
        self.nb_future_steps = nb_future_steps

    def __len__(self):
        return len(self.index) - self.nb_back_steps - self.nb_future_steps

    def __getitem__(self, index):
        index = index + self.nb_back_steps

        dict_return = {}

        for back in range(self.nb_back_steps):
            # we add information for the back
            # we read the h5 file corresponding to the date in the index - back
            path_file = os.path.join(
                self.dir_index, str(self.index.iloc[index - back - 1]["file_path_h5"])
            )

            dict_return["back_" + str(back)] = h5py.File(path_file, "r")["data"]

        for future in range(self.nb_future_steps):
            # we add information for the future
            # we read the h5 file corresponding to the date in the index + future
            path_file = os.path.join(
                self.dir_index, str(self.index.iloc[index + future]["file_path_h5"])
            )

            dict_return["future_" + str(future)] = h5py.File(path_file, "r")["data"]

        current_date = self.index.iloc[index]["datetime"]

        # retrieve the closest round hour (if current_date is not a round hour, we take the previous one)
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

        round_date_next = current_date + datetime.timedelta(hours=1)

        # now we can select the ground station data for the two dates
        df_ground_station_previous = self.groundstations_info.loc[round_date_previous]
        df_ground_station_next = self.groundstations_info.loc[round_date_next]

        # 3. todo : preprocess the two ground station information to obtain a image of france
        mask_previous, ground_station_image_previous = transform_groundstation_data_into_image(
            df_ground_station_previous
        )

        mask_next, ground_station_image_next = transform_groundstation_data_into_image(
            df_ground_station_next
        )

        dict_return["ground_station_image_previous"] = ground_station_image_previous
        dict_return["ground_station_image_next"] = ground_station_image_next

        dict_return["mask_previous"] = mask_previous
        dict_return["mask_next"] = mask_next

        return dict_return
