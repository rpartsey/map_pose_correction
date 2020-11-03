import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseCorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(32 + 2 * 2, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.max_pool3 = nn.MaxPool2d(3, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.conv5 = nn.Conv2d(128, 3, 1, 1)

    def forward(self, egoview, egomap):
        x = self.conv1(egomap)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.max_pool1(x)

        x = torch.cat([x, egoview, F.interpolate(egomap, egoview.shape[2:])], 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.max_pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.adaptive_max_pool(x)

        x = self.conv5(x)

        location = x[:, :2, :, :]
        orientation = x[:, 2:, :, :]

        location = torch.sigmoid(location)
        origin = torch.full_like(location, fill_value=0.5)
        location = location - origin

        x = torch.cat((location, orientation), dim=1)

        return x
