import torch
import argparse

from models import ResnetNodeClassifier,

def getArgs():
    parser=argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default = 1000, help='max number of steps for each epoch')
    parser.add_argument('--use_policy_step', type=int, default=5000, help='number of steps before using the learned policy')
    parser.add_argument('--update_frequency', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for training')
    parser.add_argument('--save_name', default='sacDiscOnGraph', help='prefix name for saving the SAC networks')

    # buffer hyperparameters
    parser.add_argument('--buffer_limit', type=int, default=40000, help='max number of samples in the replay buffer')
    parser.add_argument('--buffer_init_steps', type=int, default=500, help='number of random actions to take before train loop')

    # sac hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for SAC RL')
    parser.add_argument('--alpha', type=float, default=0.2, help='discount factor for entropy for sac')
    parser.add_argument('--polyak', type=float, default=0.95, help='polyak averaging parameter')#.995???

    # policy hyperparameters
    parser.add_argument('--log_std_max', type=int, default=2, help='max bound for log_std clipping') #not used in discrete
    parser.add_argument('--log_std_min', type=int, default=-20, help='min bound for log_std clipping') #not used in discrete

    # resnet backbone args
    parser.add_argument('--return_node', type=str, default='layer4', choices=['layer1','layer2','layer3','layer4'], help='resnet layer to return features from')

    return parser.parse_args()

