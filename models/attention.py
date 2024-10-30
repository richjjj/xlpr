import torch.nn as nn
import torch


class Attention_module_FC_2(nn.Module):
    def __init__(self, nc, K=8, downsample=4, w=96, h=32):
        super().__init__()
        self.K = K
        self.w = w
        self.h = h

        channels = [512, 256, 128]

        self.atten_0 = nn.Sequential(
            nn.Conv2d(nc, channels[1], 3, 1, 1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        self.atten_1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 1, 1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        # FC层的维度计算
        fc_dim = int(self.w * self.h / downsample / downsample / 16)
        self.atten_fc1 = nn.Linear(fc_dim, fc_dim)
        self.atten_fc2 = nn.Linear(fc_dim, fc_dim)

        self.cnn_1_1 = nn.Conv2d(channels[1], 64, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 反卷积层
        self.deconv1 = nn.ConvTranspose2d(
            channels[2], 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(
            64, self.K, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.K)

    def forward(self, input):
        x0 = self.atten_0(input)
        x1 = self.atten_1(x0)
        batch_size, channel, height, width = x1.size()

        fc_x = x1.view(batch_size, channel, -1)
        fc_atten = self.atten_fc2(self.atten_fc1(fc_x))
        fc_atten = fc_atten.reshape(batch_size, channel, height, width)

        # 生成注意力分数
        score = self.relu(self.deconv1(fc_atten))
        score = self.bn1(score + self.cnn_1_1(x0))
        atten = self.sigmoid(self.deconv2(score))

        # 注意力应用
        atten = atten.reshape(batch_size, self.K, -1)
        input_reshaped = input.reshape(batch_size, input.size(1), -1)
        input_reshaped = input_reshaped.permute(0, 2, 1)

        atten_out = torch.bmm(atten, input_reshaped)
        atten_out = atten_out.view(batch_size, self.K, -1)

        return atten_out
