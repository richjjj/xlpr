import torch.nn as nn
from models.encoder import HCEncoder_2
from models.attention import Attention_module_FC_2
from models.decoder import FCDecoder


class LPR_model(nn.Module):
    def __init__(self, nc, nclass, imgW=96, imgH=32, K=8):
        super(LPR_model, self).__init__()

        self.encoder = HCEncoder_2(nc)

        self.attention = Attention_module_FC_2(
            nc=128, K=K, downsample=4, w=imgW, h=imgH
        )

        self.decoder = FCDecoder(nclass, input_dim=128)

    def forward(self, input):
        conv_out = self.encoder(input)
        atten_out = self.attention(conv_out)

        preds = self.decoder(atten_out)
        return preds
