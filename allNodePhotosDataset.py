import cv2
import random
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset

class AllNodePhotosData(Dataset):
    def __init__(self, node_photo_folder, mode='train'):
        self.photo_folder = node_photo_folder
        self.paths = glob(self.photo_folder+'*/*.png')
        self.paths.sort()
        self.inds = list(range(len(self.paths)))

        if mode=='train':
            self.inds = [i for i in self.inds if i%7!=0 and i%8!=0 and i%9!=0]
        elif mode=='val':
            self.inds = [i for i in self.inds if i % 7 == 0 and i%8!=0 and i%9!=0]
        elif mode=='test':
            self.inds = [i for i in self.inds if i % 8 == 0 or i % 9 == 0]
        else:
            pass
            # print("Unsupported dataset split:",mode)
            # breakpoint()

        self.paths = np.array(self.paths)[self.inds]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = cv2.imread(self.paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image #TODO: also return the node number and the image number?

    def getRandom(self, item=None):
        if item is None:
            item = random.choice(self.paths)
        else:
            item = self.paths[item]

        image = cv2.imread(item)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

if __name__=='__main__':
    print('total files:',len(glob('nodePhotosSmall/*/*.png')))
    trainData = AllNodePhotosData('nodePhotosSmall/')
    print(len(trainData))
    valData = AllNodePhotosData('nodePhotosSmall/', 'val')
    print(len(valData))
    testData = AllNodePhotosData('nodePhotosSmall/', 'test')
    print(len(testData))
    print('train data contains none from other data:',sum([x for x in trainData.inds if x in valData.inds])==0 and sum([x for x in trainData.inds if x in testData.inds])==0)
    print('val data contains none from test data:', sum([x for x in valData.inds if x in testData.inds])==0)

