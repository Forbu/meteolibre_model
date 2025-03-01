"""
In this part the idea is the create a dataloader from an index dataframe
"""

import os
import datetime

import torch
import torch.utils.data as Dataset
import polars as pl
import numpy as np

# import package for h5 file
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
    lat = torch.tensor(df["position_x"], dtype=torch.int64, device=DEVICE)
    lon = torch.tensor(df["position_y"], dtype=torch.int64, device=DEVICE)
    
    nb_channels = len(columns_measurements)

    end_time = datetime.datetime.now()
    
    # init the multiple channel image to 0
    image_result = torch.ones((3472, 3472, nb_channels), dtype=torch.float32, device=DEVICE) * -100

    end_time = datetime.datetime.now()

    # get the measurements for each ground station
    measurements = df.select(columns_measurements).to_numpy()

    # for each ground station, we add the measurements to the image
    image_result[lat, lon, :] = torch.tensor(measurements)
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
        self.index = pl.read_parquet(index_file)
        print(self.index.head())
        
        self.index = self.index.sort("datetime")

        # 
        
        self.dir_index = dir_index
        self.groundstations_info = groundstations_info

        # sort groundstations_info by datetime
        self.groundstations_info_df = pl.read_parquet(groundstations_info)
        
        # we select only the columns we need
        self.groundstations_info_df = self.groundstations_info_df.select(columns_measurements + columns_positions + ["datetime"])
        
        self.groundstations_info_df = self.groundstations_info_df.sort("datetime")
        
        # replace NaN by -100
        self.groundstations_info_df = self.groundstations_info_df.fill_null(-100)

        print(self.groundstations_info_df.head())
        

        self.nb_back_steps = nb_back_steps
        self.nb_future_steps = nb_future_steps

    def __len__(self):
        return self.index.height - self.nb_back_steps - self.nb_future_steps

    def __getitem__(self, index):
        index = int(index + self.nb_back_steps)

        dict_return = {}
        
        import time
        
        start_time = time.time()


        for back in range(self.nb_back_steps):
            # we add information for the back
            # we read the h5 file corresponding to the date in the index - back

            path_file = os.path.join(
                self.dir_index, str(self.index["file_path_h5"][index - back - 1])
            )

            dict_return["back_" + str(back)] = np.array(h5py.File(path_file, "r")["dataset1"]["data1"]["data"])

        for future in range(self.nb_future_steps):
            # we add information for the future
            # we read the h5 file corresponding to the date in the index + future
            path_file = os.path.join(
                self.dir_index, str(self.index["file_path_h5"][index + future])
            )

            dict_return["future_" + str(future)] = np.array(h5py.File(path_file, "r")["dataset1"]["data1"]["data"])

        current_date = self.index["datetime"][index]

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
        
        # time spent
        end_time = time.time()
        print(f"Time to get one item: {end_time - start_time}")

        # now we can select the ground station data for the two dates
        # Assuming groundstations_info is still a pandas DataFrame with MultiIndex
        # If groundstations_info is also converted to polars, this will need to be adapted
        # based on how the data is structured
        df_ground_station_previous = self.groundstations_info_df.filter(
            pl.col("datetime") == round_date_previous
        )
        df_ground_station_next = self.groundstations_info_df.filter(
            pl.col("datetime") == round_date_next
        )
        
        print("time spent to get the ground station data: ", time.time() - end_time)

        end_time = time.time()

        # 3. todo : preprocess the two ground station information to obtain a image of france
        mask_previous, ground_station_image_previous = transform_groundstation_data_into_image(
            df_ground_station_previous
        )

        mask_next, ground_station_image_next = transform_groundstation_data_into_image(
            df_ground_station_next
        )
        
        print("time spent to get the ground station image: ", time.time() - end_time)

        dict_return["ground_station_image_previous"] = ground_station_image_previous
        dict_return["ground_station_image_next"] = ground_station_image_next

        dict_return["mask_previous"] = mask_previous
        dict_return["mask_next"] = mask_next
        dict_return["hour"] = current_date.hour
        

        return dict_return
