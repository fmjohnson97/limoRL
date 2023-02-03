import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision.models import feature_extraction, resnet18, ResNet18_Weights
from torch.nn import Linear, ReLU, Sigmoid

class ResnetNodeClassifier(nn.Module):
    def __init__(self, args, device):
        super(ResnetNodeClassifier, self).__init__()
        self.backbone=ResnetBackbone(args, device)

        self.fc1 = Linear(args.in_feats, args.hidden_feats)
        self.fc2 = Linear(args.hidden_feats, args.hidden_feats)
        self.fc3 = Linear(args.hidden_feats, args.hidden_feats)
        self.out_layer = Linear(args.hidden_feats, 1)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.backbone.extractFeatures(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.out_layer(x))
        return x


class ResnetBackbone():
    def __init__(self, args, device):
        self.return_node = args.return_node
        self.createResnetBackbone(device)
        self.device=device

    def createResnetBackbone(self,device):
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        model = resnet18(weights=weights).to(device)
        model.eval()
        self.feat_ext = feature_extraction.create_feature_extractor(model, return_nodes=[self.return_node])
        self.feat_ext = self.feat_ext.to(device)
        for param in self.feat_ext.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extractFeatures(self, img):
        if len(img.shape)>4:
            temp = img
            img = []
            for t in temp:
                img.extend(t)
            img = torch.stack(img)
        # breakpoint()
        img_ext=[]
        if len(img.shape)==4 and type(img) != Image:
            for im in img:
                if type(im)==torch.Tensor:
                    im = Image.fromarray(im.cpu().numpy().astype(np.uint8))
                else:
                    im = Image.fromarray(im)
                img_ext.append(torch.FloatTensor(self.preprocess(im)))
            img_ext=torch.stack(img_ext).to(self.device)
        else:
            # breakpoint()
            img_ext=torch.FloatTensor(img).unsqueeze(0).to(self.device)

        features = self.feat_ext(img_ext)
        # breakpoint()
        return features[self.return_node].reshape(len(features[self.return_node]), -1)