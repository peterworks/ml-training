from torch import nn, add
from torch.nn.functional import leaky_relu as r
from torch import tanh as t

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convTrans1 = nn.ConvTranspose2d(1, 256, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.convTrans2 = nn.ConvTranspose2d(256, 3, 5, stride=2)

    def forward(self, value):

        value = self.convTrans1(value)
        value = r(value)
        value = self.bn1(value)
        value = self.convTrans2(value)
        value = t(value)
        
        return value