from torch import nn, add
from torch.nn.functional import relu as r, logsigmoid as logsig
from torch import tanh as t

class Generator(nn.Module):
    def __init__(self, lookup):
        super().__init__()

        self.lookup = lookup

        self.convTransBlock = nn.Sequential(

            nn.ConvTranspose2d(3, 256, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 256, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 128, 5, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, 7, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 32, 5, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 16, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 16, 7, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 3, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3)

            # nn.ConvTranspose2d(128, 3, 7, stride=1),
        )

    def forward(self, value):

        value = self.convTransBlock(value)
        
        value = t(value)

        if(self.lookup):
            print("Generator size: ", value.size())
        
        return value