from matplotlib import pyplot as plt
from math import floor, sqrt
from torchvision import transforms as t

class Plotter:

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def plotImage(self, tensor, fig_dims: tuple = (8, 8), cmap: str ='gray'):
        fig, axs = plt.subplots(floor(sqrt(self.batch_size)), floor(sqrt(self.batch_size)), sharex=False, sharey=False)
        fig.set_figwidth(fig_dims[0])
        fig.set_figheight(fig_dims[1])    
        for genImageIndex, genImage in enumerate(tensor):
            plotImage = t.ToPILImage()(genImage.cpu().detach())
            axs[floor(genImageIndex / floor(sqrt(self.batch_size))), genImageIndex % floor(sqrt(self.batch_size))].imshow(plotImage, cmap=cmap)
        plt.pause(0.001)

    def multiGraph(self, plotsArray: list, colorsArray: list, epochsLimit: int = 1, fig_dims: tuple = (3, 3)):
        fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
        fig.set_figwidth(fig_dims[0])
        fig.set_figheight(fig_dims[1])
        for plotIndex, plot in enumerate(plotsArray):      
            axs.plot(plot[-(epochsLimit):], color=colorsArray[plotIndex])
        plt.pause(0.01)
        plt.close(fig=fig)
