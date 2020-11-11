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

        placeholder = '-.-'
        episode_dir = os.path.join(self.data_root, scene_name, episode_name, placeholder)
        file_name = f'{str(t).zfill(3)}.npy'

        poses_gt = np.load(episode_dir.replace(placeholder, 'poses_gt.npy'))
        poses_est = np.load(episode_dir.replace(placeholder, 'poses_est.npy'))
        pose_error = poses_gt[t].reshape(3, 1, 1) - poses_est[t].reshape(3, 1, 1)

        observed_egoview_projection = np.load(os.path.join(episode_dir.replace(placeholder, 'egoview_projections'), file_name))
        expected_egoview_projection = np.load(os.path.join(episode_dir.replace(placeholder, 'expected_egomaps'), file_name))

        observed_egoview = torch.from_numpy(observed_egoview_projection).type(torch.float32)
        expected_egoview = torch.from_numpy(expected_egoview_projection).type(torch.float32)
        pose_error = torch.from_numpy(pose_error).type(torch.float32)

        return (observed_egoview, expected_egoview), pose_error

    def __len__(self):
        return self.df.shape[0]
