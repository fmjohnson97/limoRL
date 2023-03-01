import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from models import ResnetBackbone
from graph import Graph

def getArgs():
    parser = argparse.ArgumentParser()
    # resnet/backbone args
    parser.add_argument('--return_node', type=str, default='layer4', choices=['layer1', 'layer2', 'layer3', 'layer4'],
                        help='resnet layer to return features from')
    return parser.parse_args()


args = getArgs()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResnetBackbone(args, device)
graph = Graph('labGraphConfig5.json')

images = []
colors = []
ang_colors = []
for i in range(5):
    for ang in [0,15,30,45,50,75,90,105,120,135,150,165,180,270,360]:
        images.append(graph.getVertexFeats(i+1, ang))
        colors.append(i+1)
        ang_colors.append(ang)

feats = []
for im in tqdm(images):
    feats.extend(model.extractFeatures(np.stack([im])))

tsne=TSNE()
tsne_embedding = tsne.fit_transform(torch.flatten(torch.stack(feats),1).numpy())
plt.figure()
plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=colors)
plt.title('Cardinal Embeddings for Lab5 Color=node')
plt.figure()
plt.scatter(tsne_embedding[:,0],tsne_embedding[:,1],c=ang_colors)
plt.title('Cardinal Embeddings for Lab5 Color=angle')

breakpoint()

pca_embedding = torch.stack(feats).transpose(-1,-2).numpy()
plt.figure()
plt.imshow(pca_embedding[0][:,:,:3])
plt.title('Node1 angle 0')
plt.figure()
plt.imshow(pca_embedding[4][:,:,:3])
plt.title('Node2 angle 0')
plt.figure()
plt.imshow(pca_embedding[8][:,:,:3])
plt.title('Node3 angle 0')
plt.figure()
plt.imshow(pca_embedding[12][:,:,:3])
plt.title('Node4 angle 0')
plt.show()