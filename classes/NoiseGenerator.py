from numpy.random import normal as randomNpArray
from torch import from_numpy as numpyToTensor

class NoiseGenerator:

    def generate(self, batch_size: int, channels: int, dims: tuple, loc: float = 0.5, scale: float = 0.5):
        npArray = randomNpArray(loc=loc, scale=scale, size=(batch_size, channels, dims[0], dims[1]))
        randomTensor = numpyToTensor(npArray).cuda().float()
        return randomTensor