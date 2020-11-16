import torch
import torch.nn as nn


class PoseLoss(nn.Module):
    def __init__(self, alpha=1., beta=1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred_pose, true_pose):
        B, C, H, W = true_pose.shape
        pred_pose = pred_pose.reshape(B, 3)
        true_pose = true_pose.reshape(B, 3)

        true_location = true_pose[:, :2]
        true_orientation = true_pose[:, 2]
        pred_location = pred_pose[:, :2]
        pred_orientation = pred_pose[:, 2]

        pose_loss = self.mse(pred_location, true_location) / B
        orientation_loss = (1. - torch.cos(pred_orientation - true_orientation)).mean()

        return self.alpha * pose_loss + self.beta * orientation_loss, pose_loss, orientation_loss
