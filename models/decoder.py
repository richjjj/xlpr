import torch.nn as nn


class FCDecoder(nn.Module):
    def __init__(self, nclass, input_dim=512):
        super(FCDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)

    def forward(self, input):
        return self.fc(input)
