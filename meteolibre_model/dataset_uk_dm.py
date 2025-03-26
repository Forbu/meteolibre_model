import datetime

import torch
from datasets import load_dataset
import numpy as np
from numpy.random import default_rng

NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[
        -NUM_TARGET_FRAMES - NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES
    ]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


class TFDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split):
        super().__init__()
        self.reader = load_dataset(
            "openclimatefix/nimrod-uk-1km",
            "sample",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        self.iter_reader = self.reader

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        try:
            row = next(self.iter_reader)
        except Exception:
            rng = default_rng()
            self.iter_reader = iter(
                self.reader.shuffle(
                    seed=rng.integers(low=0, high=100000), buffer_size=1000
                )
            )
            row = next(self.iter_reader)

        print(row["end_time_timestamp"])

        input_frames, target_frames = extract_input_and_target_frames(
            row["radar_frames"]
        )

        # get timestamp
        date_timestamp = row["end_time_timestamp"]

        # convert to datetime object
        date_object = datetime.fromtimestamp(date_timestamp)

        dict_return = {
            "hour": date_object.hour/24.,
            "minute": date_object.minute/60.,
            "input_radar_frames": input_frames,
            "target_radar_frames": target_frames,
        }

        return dict_return
