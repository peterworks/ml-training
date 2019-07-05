from torch import nn, sigmoid as s
from torch.nn.functional import leaky_relu as r

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 128, 5, stride=6)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 5, stride=6)
        self.bn2 = nn.BatchNorm2d(256)

        self.ln1 = nn.Linear(1024, 512)
        self.norm1 = nn.LayerNorm(512)
        self.ln2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.ln3 = nn.Linear(256, 1)

    def forward(self, value):
        value = self.conv1(value)
        value = r(value)
        value = self.bn1(value)

        value = self.conv2(value)
        value = r(value)
        value = self.bn2(value)
        
        value = value.view(-1)

        value = self.ln1(value)
        value = r(value)
        value = self.norm1(value)
        value = self.ln2(value)
        value = r(value)
        value = self.norm2(value)
        value = self.ln3(value)                
        value = s(value)
        return value
        