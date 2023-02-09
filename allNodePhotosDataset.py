
from PIL import Image
from glob import glob
from torch.utils.data import Dataset

class AllNodePhotosData(Dataset):
    def __init__(self, node_photo_folder):
        self.photo_folder = node_photo_folder
        self.paths = glob(self.photo_folder+'*/*.png')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = Image.
