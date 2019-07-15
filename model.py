from encoder_decoder import Encoder, Decoder
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, out_layer, d_model, N, heads):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.out = nn.Linear(d_model, out_layer)
    def forward(self, src, trg):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs)
        output = self.out(d_output)
        return output

# we don't perform softmax on the output as this will be handled 
# automatically by our loss function
