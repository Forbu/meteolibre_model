from datetime import datetime

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


def max_pool_2x2(frames):
    """
    Downsamples frames by a factor of 2 using max pooling.
    Assumes input frames have shape (H, W, T).
    """
    H, W, T = frames.shape
    # Ensure dimensions are even for simple 2x2 pooling
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"Frame dimensions ({H}, {W}) must be even for 2x2 max pooling.")

    # Reshape to group into 2x2 blocks: (H/2, 2, W/2, 2, T)
    # Then take the maximum over the 2x2 block dimensions (axes 1 and 3)
    pooled_frames = frames.reshape(H // 2, 2, W // 2, 2, T).max(axis=(1, 3))
    return pooled_frames


class TFDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split, nb_frame_futur=8):
        super().__init__()
        self.reader = load_dataset(
            "openclimatefix/nimrod-uk-1km",
            "sample",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        self.iter_reader = self.reader
        self.nb_frame_futur = nb_frame_futur

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

        input_frames, target_frames = extract_input_and_target_frames(
            row["radar_frames"]
        )

        # get timestamp
        date_timestamp = row["end_time_timestamp"]

        input_frames, target_frames = (
            np.squeeze(input_frames, axis=-1),
            np.squeeze(target_frames, axis=-1),
        )

        input_frames, target_frames = (
            np.moveaxis(input_frames, [0, 1, 2], [2, 0, 1]),
            np.moveaxis(target_frames, [0, 1, 2], [2, 0, 1]),
        )

        # check is there is something in the first frame (if not switch to another sample)
        if np.sum(input_frames[:, :, 0]) <= 20:
            return self.__getitem__(item + 1)


        # --- Max Pooling Step ---
        try:
            input_frames_pooled = input_frames #max_pool_2x2(input_frames)
            target_frames_pooled = target_frames #max_pool_2x2(target_frames)
        except ValueError as e:
            print(f"Skipping item due to incompatible shape for pooling: {e}")
            print(f"Original input shape: {input_frames.shape}")
            return self.__getitem__(item + 1)

        # convert to datetime object
        date_object = datetime.fromtimestamp(date_timestamp)

        dict_return = {
            "hour": date_object.hour / 24.0,
            "minute": date_object.minute / 60.0,
            "input_radar_frames": input_frames_pooled / 12.0,
            "target_radar_frames": target_frames_pooled[:, :, :self.nb_frame_futur] / 12.0,
        }

        return dict_return
