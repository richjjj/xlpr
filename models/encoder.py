import torch.nn as nn


class HCEncoder_2(nn.Module):
    def __init__(self, nc=1):
        super().__init__()
        layers_config = [
            # in_channels, out_channels, kernel_size, stride, padding, pool
            (nc, 32, 3, 1, 1, False),  # conv1
            (32, 32, 3, 1, 1, False),  # conv2
            (32, 32, 3, 1, 1, True),  # conv3 + pool
            (32, 64, 3, 1, 1, False),  # conv4
            (64, 64, 3, 1, 1, False),  # conv5
            (64, 64, 3, 1, 1, True),  # conv6 + pool
            (64, 128, 3, 1, 1, False),  # conv7
            (128, 128, 3, 1, 1, False),  # conv8
            (128, 128, 3, 1, 1, False),  # conv9
        ]

        layers = []
        for in_c, out_c, k, s, p, pool in layers_config:
            layers.extend(
                [nn.Conv2d(in_c, out_c, k, s, p), nn.BatchNorm2d(out_c), nn.ReLU(True)]
            )
            if pool:
                layers.append(nn.MaxPool2d(2, 2))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)
