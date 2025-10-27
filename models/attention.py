import torch.nn as nn
import torch


class Attention_module_FC_2(nn.Module):
    def __init__(self, nc, K=8, downsample=4, w=96, h=32):
        super().__init__()
        self.K = K
        self.downsample = downsample

        channels = [512, 256, 128]
        pooled_w, pooled_h = self._compute_spatial_size(w, h, downsample)
        fc_dim = pooled_w * pooled_h
        if fc_dim <= 0:
            raise ValueError(
                f"Invalid feature size {pooled_w}x{pooled_h} for input {w}x{h}. "
                "Please double-check the provided image dimensions."
            )

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

        self.atten_fc1 = nn.Linear(fc_dim, fc_dim)
        self.atten_fc2 = nn.Linear(fc_dim, fc_dim)

        self.cnn_1_1 = nn.Conv2d(channels[1], 64, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 反卷积层
        # 改进方案：使用kernel_size=4避免output_padding, rknn等框架不支持
        self.deconv1 = nn.ConvTranspose2d(
            channels[2], 64, kernel_size=4, stride=2, padding=1  # 去掉output_padding
            )
        self.deconv2 = nn.ConvTranspose2d(
            64, self.K, kernel_size=4, stride=2, padding=1
            )
        self.bn1 = nn.BatchNorm2d(64)
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

    def _compute_spatial_size(self, width: int, height: int, downsample: int):
        encoder_pools = self._pool_count(downsample)
        total_pools = encoder_pools + 2  # 额外两个池化来自注意力模块
        pooled_w = self._apply_pooling(width, total_pools)
        pooled_h = self._apply_pooling(height, total_pools)
        return pooled_w, pooled_h

    @staticmethod
    def _apply_pooling(size: int, count: int) -> int:
        for _ in range(count):
            size = (size + 1) // 2
        return size

    @staticmethod
    def _pool_count(downsample: int) -> int:
        count = 0
        value = max(downsample, 1)
        while value > 1:
            if value % 2 != 0:
                raise ValueError("downsample must be a power of two.")
            value //= 2
            count += 1
        return count
