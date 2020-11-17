import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader  # noqa


class EgoviewsDataset(Dataset):
    def __init__(self, data_root, csv_path):
        self.data_root = data_root
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, idx):
        scene_name, episode_name, t = self.df.iloc[idx]
        episode_dir = os.path.join(self.data_root, scene_name, episode_name)

        poses_gt = np.load(os.path.join(episode_dir, 'poses_gt.npy'))
        poses_est = np.load(os.path.join(episode_dir, 'poses_est.npy'))
        pose_error = (poses_gt[t] - poses_est[t]).reshape(3, 1, 1)

        observed_projection = np.load(os.path.join(episode_dir, 'egoview_projections', f'{str(t).zfill(3)}.npy'))
        expected_projection = np.load(os.path.join(episode_dir, 'expected_egomaps', f'{str(t).zfill(3)}.npy'))

        observed_projection = torch.from_numpy(observed_projection).type(torch.float32)
        expected_projection = torch.from_numpy(expected_projection).type(torch.float32)
        pose_error = torch.from_numpy(pose_error).type(torch.float32)

        return (observed_projection, expected_projection), pose_error

    def __len__(self):
        return self.df.shape[0]


class EgoviewsDatasetV2(EgoviewsDataset):
    def __getitem__(self, idx):
        scene_name, episode_name, t = self.df.iloc[idx]
        episode_dir = os.path.join(self.data_root, scene_name, episode_name)

        poses_gt = np.load(os.path.join(episode_dir, 'poses_gt.npy'))
        poses_est = np.load(os.path.join(episode_dir, 'poses_est.npy'))
        pose_error = (self.step_delta(poses_gt, t) - self.step_delta(poses_est, t)).reshape(3, 1, 1)

        observed_projection = np.load(os.path.join(episode_dir, 'egoview_projections', f'{str(t).zfill(3)}.npy'))
        expected_projection = np.load(os.path.join(episode_dir, 'egoview_projections_prev', f'{str(t).zfill(3)}.npy'))

        observed_projection = torch.from_numpy(observed_projection).type(torch.float32)
        expected_projection = torch.from_numpy(expected_projection).type(torch.float32)
        pose_error = torch.from_numpy(pose_error).type(torch.float32)

        return (observed_projection, expected_projection), pose_error

    @staticmethod
    def step_delta(trajectory, t):
        return trajectory[t] - trajectory[t - 1] if t > 0 else trajectory[t]