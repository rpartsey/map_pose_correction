import torch
import torch.nn as nn


class PoseCorrectionNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.pose_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.location_head = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(128, 2, 1, 1)
        )
        self.orientation_head = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(128, 1, 1, 1)
        )

    def forward(self, observed_projection, expected_projection):
        x = torch.cat((observed_projection, expected_projection), 1)
        x = self.pose_conv(x)

        location = torch.tanh(self.location_head(x))
        orientation = self.orientation_head(x)

        x = torch.cat((location, orientation), dim=1)

        return x


class PoseCorrectionNetV2(PoseCorrectionNet):
    def __init__(self, in_channels=16):
        super().__init__(in_channels=in_channels+2)
        self.downsample_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

    def forward(self, observed_projection, expected_projection):
        x = super().forward(observed_projection, self.downsample_conv(expected_projection))

        return x


class NeuralSLAMPoseEstimator(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout

        # Pose Estimator
        self.pose_conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(4, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3, 3), stride=(1, 1)),
            nn.ReLU()
        ]))

        pose_conv_output = self.pose_conv(torch.randn(1, 4, self.vision_range, self.vision_range))
        self.pose_conv_output_size = pose_conv_output.view(-1).size(0)

        # projection layer
        self.pose_proj1 = nn.Linear(self.pose_conv_output_size, 1024)
        self.pose_proj2_x = nn.Linear(1024, 128)
        self.pose_proj2_y = nn.Linear(1024, 128)
        self.pose_proj2_o = nn.Linear(1024, 128)
        self.pose_proj3_x = nn.Linear(128, 1)
        self.pose_proj3_y = nn.Linear(128, 1)
        self.pose_proj3_o = nn.Linear(128, 1)

        if self.dropout > 0:
            self.pose_dropout1 = nn.Dropout(self.dropout)

    def forward(self, pose_est_input):
        pose_conv_output = self.pose_conv(pose_est_input)
        pose_conv_output = pose_conv_output.view(-1, self.pose_conv_output_size)

        proj1 = nn.ReLU()(self.pose_proj1(pose_conv_output))

        if self.dropout > 0:
            proj1 = self.pose_dropout1(proj1)

        proj2_x = nn.ReLU()(self.pose_proj2_x(proj1))
        pred_dx = self.pose_proj3_x(proj2_x)

        proj2_y = nn.ReLU()(self.pose_proj2_y(proj1))
        pred_dy = self.pose_proj3_y(proj2_y)

        proj2_o = nn.ReLU()(self.pose_proj2_o(proj1))
        pred_do = self.pose_proj3_o(proj2_o)

        pose_pred = torch.cat((pred_dx, pred_dy, pred_do), dim=1)

        return pose_pred
