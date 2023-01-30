import torch
import cv2
import json

from torch.utils.data import Dataset
from glob import glob

class NodeDataset(Dataset):
    def __init__(self, node_num, mode='train', video_extension='.mp4'):
        self.mode = mode
        self.node_num = node_num
        self.videoPath = 'nodeVideos/node'+str(node_num)+video_extension
        self.photoPath = 'nodePhotos/node'+str(node_num)+'/'

        # labels of the form [x y z theta] since phi is all the same
        with open(self.photoPath + 'labels.json') as f:
            self.labels = json.load(f)
        self.samples = self.labels[self.mode]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        image = torch.FloatTensor(cv2.imread(self.photoPath+'node'+str(self.node_num)+'_'+self.samples[item]+'.png'))
        # x, y, z, theta
        data = torch.FloatTensor(self.labels[self.samples[item]])

        return image, data

if __name__ == '__main__':
    nodeData = NodeDataset(1,'val')
    print(nodeData.__len__())
    im, data = nodeData.__getitem__(20)
    print(im.shape)
    print(data)