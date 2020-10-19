import os
import torch

from data import EgoviewsDataset, DataLoader
from model import PoseCorrectionNet


preprocessed_data_dir = '/datasets/extra_space2/rpartsey/3d-navigation/habitat/data/map_pose_correction/preprocessed_data'
train_data_dir = os.path.join(preprocessed_data_dir, 'train')
device = 'cuda:2'
batch_size = 16


dataset = EgoviewsDataset(
    observed_egoviews_path=os.path.join(train_data_dir, 'observed_egoviews.npy'),
    expected_egoviews_path=os.path.join(train_data_dir, 'expected_egoviews.npy'),
    pose_errors_path=os.path.join(train_data_dir, 'pose_errors.npy')
)

loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size
)

for batch in loader:
    observed_egoviews, expected_egoviews, pose_errors = batch

    observed_egoviews = observed_egoviews.to(device)
    expected_egoviews = expected_egoviews.to(device)
    pose_errors = pose_errors.to(device)
    break

model = PoseCorrectionNet()
model.to(device)

prediction = model(observed_egoviews, expected_egoviews)

diff = prediction - pose_errors
print(diff.shape)
print(diff)
