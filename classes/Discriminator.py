from torch import nn, sigmoid as s
from torch.nn.functional import leaky_relu as r

class Discriminator(nn.Module):
    def __init__(self, lookup=False, batch_size=1):
        super().__init__()

        self.dataset = None
        self.lookup = lookup
        self.batch_size = batch_size

        self.convBlock = nn.Sequential(

            nn.BatchNorm2d(3),

            nn.Conv2d(3, 16, 1, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 1, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 5, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 1, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 7, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 1, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 11, stride=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 1, stride=1)

        )

        self.out = nn.Linear(4608, 1)


    def forward(self, value):

        input_value = value

        value = self.convBlock(value)

        value = value.view(self.batch_size, -1)

        value = self.out(value)

        value = value.view(self.batch_size, -1)
        
        if(self.lookup):
            print("Discriminator size: ", value.size())
            value = s(value)
            value = value.view(self.batch_size, -1)
            return [value, input_value]

        if(not self.lookup):
            value = s(value)

        return value
        