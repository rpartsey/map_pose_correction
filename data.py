import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EgoviewsDataset(Dataset):
    def __init__(self, observed_egoviews_path, expected_egoviews_path, pose_errors_path):
        self.observed_egoviews = np.load(observed_egoviews_path)
        self.expected_egoviews = np.load(expected_egoviews_path)
        self.pose_errors = np.load(pose_errors_path)

    def __getitem__(self, idx):
        observed_egoview = self.observed_egoviews[idx]
        expected_egoview = self.expected_egoviews[idx]
        pose_error = self.pose_errors[idx].reshape(3, 1, 1)

        observed_egoview = torch.from_numpy(observed_egoview)
        expected_egoview = torch.from_numpy(expected_egoview)
        pose_error = torch.from_numpy(pose_error)

        return observed_egoview, expected_egoview, pose_error

    def __len__(self):
        return self.observed_egoviews.shape[0]