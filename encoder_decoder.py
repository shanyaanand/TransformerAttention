from Layers import EncoderLayer, DecoderLayer
from multiheaded_attention import Norm, get_clones
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src):
        x = (src)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)
    
class Decoder(nn.Module):

    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.norm(x)
