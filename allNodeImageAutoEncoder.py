import torch
import argparse
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from models import Encoder, Decoder
from allNodePhotosDataset import AllNodePhotosData

def getArgs():
    parser=argparse.ArgumentParser()

    parser.add_argument('--node_folder', type=str, default='nodePhotosSmall/', help='path to the node folder')

    # training hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=128, help='size of the latent vector of the AE')
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--save_name', default='allNodePhotoAE', help='prefix name for saving the SAC networks')


    return parser.parse_args()

def train(args, device):
    encoder = Encoder(args.hidden_dim).to(device)
    decoder = Decoder(args.hidden_dim).to(device)

    encOpt = optim.Adam(encoder.parameters(), lr=args.lr)
    decOpt = optim.Adam(decoder.parameters(), lr=args.lr)
    lossFunc = nn.MSELoss()

    trainData = AllNodePhotosData(args.node_folder)
    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)
    valData = AllNodePhotosData(args.node_folder, 'val')
    valLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=True)

    transforms = ResNet18_Weights.DEFAULT.transforms()
    best_val = np.inf

    for e in range(args.epochs):
        epoch_loss = 0
        for batch in trainLoader:
            batch = batch.transpose(-1,1)
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
        val_loss = test(encoder, decoder, device, valLoader, len(valData),'val')
        if val_loss < best_val:
            torch.save(encoder.state_dict(), args.save_name+'_encoder.pt')
            torch.save(decoder.state_dict(), args.save_name+'_decoder.pt')


    return encoder, decoder

@torch.no_grad()
def test(enc, dec, device, dataLoader, dataLen, name='Test'):
    lossFunc = nn.MSELoss()
    transforms = ResNet18_Weights.DEFAULT.transforms()
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
    return total_loss


if __name__ == '__main__':
    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = train(args, device)
    testData = AllNodePhotosData(args.node_folder,'test')
    testLoader = DataLoader(testData, batch_size=args.batch_size, shuffle=True)
    test(encoder, decoder, device, testLoader, len(testData))
