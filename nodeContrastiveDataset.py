import torch
import cv2
import json
import random

from torch.utils.data import Dataset
from glob import glob

class NodeContrastiveDataset(Dataset):
    def __init__(self, node_num, mode='train', video_extension='.mp4'):
        self.mode = mode
        self.node_num = node_num
        self.videoPath = 'nodeVideos/node'+str(node_num)+video_extension
        self.photoPath = 'nodePhotos/node'+str(node_num)+'/'
        self.nodePaths = glob('nodePhotos/*')

        # labels of the form [x y z theta] since phi is all the same
        with open(self.photoPath + 'labels.json') as f:
            self.labels = json.load(f)
        self.samples = self.labels[self.mode]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        inNodeImage = torch.FloatTensor(cv2.imread(self.photoPath+'node'+str(self.node_num)+'_'+self.samples[item]+'.png'))
        # x, y, z, theta
        # inNodeData = torch.FloatTensor(self.labels[self.samples[item]])

        new_node = random.choice(range(len(self.nodePaths)))
        new_im_paths = glob(self.nodePaths[new_node]+'/*')
        path = random.choice(new_im_paths)
        if path is None:
            breakpoint()
        outNodeImage = torch.FloatTensor(cv2.imread(path))
        # with open(self.nodePaths[new_node]+'/labels.json') as f:
        #     labels = json.load(f)
        # # x, y, z, theta
        # outNodeData = torch.FloatTensor(labels[path.split('_')[-1].split('.')[0]])

        images = torch.stack([inNodeImage, outNodeImage])
        data = torch.FloatTensor([1,0]) #torch.stack([inNodeData, outNodeData])

        return images, data

if __name__ == '__main__':
    nodeData = NodeContrastiveDataset(1,'val')
    print(nodeData.__len__())
    im, data = nodeData.__getitem__(20)
    print(im.shape)
    print(data)