from os.path import join as joinPath
from os import scandir
from torch.utils.data import Dataset
from skimage.io import imread as readImage
from PIL import Image
from torchvision.transforms.functional import rotate, to_tensor as t

class ImageDataset(Dataset):

    def __init__(self, root='./Data', transform_in=None):

        self.root = root
        self.transform_in = transform_in
        self.images = []

        path = joinPath(self.root)
        self.preload(path)

    def preload(self, path):
        for entry in scandir(path):
            if ( entry.is_file() and (not entry.name.startswith('.')) ):
                temp = t(Image.open(entry.path))
                if(temp.size()[0] == 3):
                    self.images.append(self.transform_in(Image.open(entry.path)).cuda())
            if ( (not entry.is_file()) and (not entry.name.startswith('.')) ):
                self.preload(entry.path)        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):  

        item = self.images[index]

        return item