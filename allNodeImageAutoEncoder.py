import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms as tvtf
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from matplotlib import pyplot as plt

from models import Encoder, Decoder
from allNodePhotosDataset import AllNodePhotosData

def getArgs():
    parser=argparse.ArgumentParser()

    parser.add_argument('--node_folder', type=str, default='nodePhotosSmall/', help='path to the node folder')

    # training hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='size of the latent vector of the AE')
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--save_name', default='allNodePhotoAE', help='prefix name for saving the SAC networks')


    return parser.parse_args()

def train(args, device, transforms):
    encoder = Encoder(args.hidden_dim).to(device)
    decoder = Decoder(args.hidden_dim).to(device)

    encOpt = optim.Adam(encoder.parameters(), lr=args.lr)
    decOpt = optim.Adam(decoder.parameters(), lr=args.lr)
    lossFunc = nn.MSELoss()

    trainData = AllNodePhotosData(args.node_folder)
    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)
    valData = AllNodePhotosData(args.node_folder, 'val')
    valLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=True)

    best_val = np.inf

    for e in range(args.epochs):
        epoch_loss = 0
        for batch in trainLoader:
            batch = batch.transpose(-1,1)#.float()
            # breakpoint()
            # to shape [batch, 3, 224, 224]
            batch = transforms(batch).to(device)
            output = decoder(encoder(batch))
            encOpt.zero_grad()
            decOpt.zero_grad()
            loss = lossFunc(batch, output)
            loss.backward()
            encOpt.step()
            decOpt.step()
            epoch_loss+=loss.item()
        print('Epoch',e,'Train loss:',epoch_loss,'Avg train loss:',epoch_loss/len(trainData))
        val_loss = test(encoder, decoder, device, transforms, valLoader, len(valData),'val')
        if val_loss < best_val:
            print('Latest Checkpoint is Epoch',e)
            torch.save(encoder.state_dict(), args.save_name+'_encoder.pt')
            torch.save(decoder.state_dict(), args.save_name+'_decoder.pt')
            best_val=val_loss

    return encoder, decoder

@torch.no_grad()
def test(enc, dec, device, transforms, dataLoader, dataLen, name='Test'):
    lossFunc = nn.MSELoss()
    total_loss = 0
    for batch in dataLoader:
        batch = batch.transpose(-1, 1)
        batch = transforms(batch).to(device)
        try:
            output = dec(enc(batch))
        except Exception as e:
            print(e)
            breakpoint()
        loss = lossFunc(batch, output)
        total_loss += loss.item()
    print('\t',name,'loss:', total_loss, 'Avg',name,'loss:', total_loss / dataLen)
    print()
    return total_loss

@torch.no_grad()
def showPredictions(encoder, decoder, transforms, n=5):
    testData = AllNodePhotosData(args.node_folder, 'test')
    for i in range(n):
        batch = testData.getRandom()
        batch = transforms(Image.fromarray(batch))
        output = decoder(encoder(batch.unsqueeze(0)))

        batch = ((np.transpose(batch.squeeze().numpy())*np.array([0.229, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255).astype(np.uint8)
        output = ((np.transpose(output.squeeze().numpy())*np.array([0.229, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255).astype(np.uint8)

        plt.figure()
        ax = plt.subplot(1, 2, 1)
        ax.imshow(batch)
        plt.title('Original Image')
        ax = plt.subplot(1, 2, 2)
        ax.imshow(output)
        plt.title('Predicted Image')
        plt.show()


if __name__ == '__main__':
    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = ResNet18_Weights.DEFAULT.transforms()
    encoder, decoder = train(args, device, transforms)
    testData = AllNodePhotosData(args.node_folder,'test')
    testLoader = DataLoader(testData, batch_size=args.batch_size, shuffle=True)
    test(encoder, decoder, device, transforms, testLoader, len(testData))

    # encoder = Encoder(args.hidden_dim)
    # decoder = Decoder(args.hidden_dim)

    encoder.load_state_dict(torch.load(args.save_name+'_encoder.pt', map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(args.save_name + '_decoder.pt',map_location=torch.device('cpu')))

    encoder.eval()
    decoder.eval()

    showPredictions(encoder, decoder, transforms)
