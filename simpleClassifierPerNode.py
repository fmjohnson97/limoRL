import argparse
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

from models import ResnetNodeClassifier
from nodeContrastiveDataset import NodeContrastiveDataset

def getArgs():
    parser=argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')

    # node classifier parameters
    parser.add_argument('--node_num', type=int, default=1, help='the node to learn on')
    parser.add_argument('--hidden_feats', type=int, default=256, help='hidden feat dimension')
    parser.add_argument('--in_feats', type=int, default=25088, help='input feat dimension')
    parser.add_argument('--validate', action='store_true', help='validate data during training')

    # resnet backbone args
    parser.add_argument('--return_node', type=str, default='layer4', choices=['layer1','layer2','layer3','layer4'], help='resnet layer to return features from')

    return parser.parse_args()


def train(args, device):
    model = ResnetNodeClassifier(args, device).to(device)

    opt = optim.Adam(lr=args.lr, params=model.parameters())
    loss_func = nn.BCELoss()

    train_data = NodeContrastiveDataset(args.node_num)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_data = NodeContrastiveDataset(args.node_num, mode='val')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    print('Training on',len(train_data),'images')
    if args.validate:
        print('Validating on',len(val_data),'images')

    best_train_loss = np.inf
    best_val_loss = np.inf

    for e in range(args.epochs):
        epoch_loss = 0
        for batch in train_loader:
            images, labels = batch
            output = model(images.to(device))
            opt.zero_grad()
            if labels.shape != output.shape:
                labels = labels.reshape(output.shape)
            loss = loss_func(output, labels.to(device))
            loss.backward()
            opt.step()

            epoch_loss+=loss.item()

        print("Episode",e,"train loss:",epoch_loss)
        print("\t avg train loss:", epoch_loss/len(train_data))
        if args.validate:
            val_loss = test(model, val_loader, device)
            avg_val_loss = val_loss / len(val_data)
            print("\t val loss:", val_loss)
            print("\t avg val loss:", avg_val_loss)
            if epoch_loss < best_train_loss and val_loss<best_val_loss:
                best_train_loss = epoch_loss
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'simpleClassNode'+str(args.node_num)+'.pt')
        else:
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                torch.save(model.state_dict(), 'simpleClassNode' + str(args.node_num) + '.pt')

    return model


@torch.no_grad()
def test(model, dataloader, device):
    loss_func = nn.BCELoss()
    test_loss = 0
    for batch in dataloader:
        images, labels = batch

        output = model(images.to(device))
        if labels.shape != output.shape:
            labels = labels.reshape(output.shape)
        loss = loss_func(output, labels.to(device))
        test_loss += loss.item()

    return test_loss



if __name__=='__main__':
    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = train(args, device)

    test_data = NodeContrastiveDataset(args.node_num, mode='val')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loss = test(model, test_data, device)
    avg_test_loss = test_loss / len(test_data)
    print("Test loss:", test_loss)
    print("Avg val loss:", avg_test_loss)
