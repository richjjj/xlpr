import torch.nn as nn


class FCDecoder(nn.Module):
    def __init__(self, nclass, input_dim=512):
        super(FCDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.softmax = nn.Softmax(dim=-1)
        self._export_onnx = False

    def enable_onnx_export(self, export: bool = True):
        self._export_onnx = export

    def forward(self, input):
        logits = self.fc(input)
        if self._export_onnx:
            return self.softmax(logits)
        return logits
