import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader  # noqa


class EgoviewsDataset(Dataset):
    def __init__(self, data_path):
        self.observed_egoviews = np.load(os.path.join(data_path, 'observed_egoviews.npy'))
        self.expected_egoviews = np.load(os.path.join(data_path, 'expected_egoviews.npy'))
        self.pose_errors = np.load(os.path.join(data_path, 'pose_errors.npy'))

    def __getitem__(self, idx):
        observed_egoview = self.observed_egoviews[idx]
        expected_egoview = self.expected_egoviews[idx]
        pose_error = self.pose_errors[idx].reshape(3, 1, 1)

        observed_egoview = torch.from_numpy(observed_egoview)
        expected_egoview = torch.from_numpy(expected_egoview)
        pose_error = torch.from_numpy(pose_error)

        return (observed_egoview, expected_egoview), pose_error

    def __len__(self):
        return self.observed_egoviews.shape[0]