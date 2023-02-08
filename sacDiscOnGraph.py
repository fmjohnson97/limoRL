import random
import torch
import argparse
import numpy as np

from models import SACDiscreteBaseline, ResnetBackbone, GoalReplayBuffer
from graph import GraphTraverser, Graph

def getArgs():
    parser=argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=200000, help='number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default = 50, help='max number of steps for each epoch')
    parser.add_argument('--use_policy_step', type=int, default=100, help='number of steps before using the learned policy')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for training')
    parser.add_argument('--save_name', default='sacDiscOnGraph', help='prefix name for saving the SAC networks')

    # buffer hyperparameters
    parser.add_argument('--buffer_limit', type=int, default=40000, help='max number of samples in the replay buffer')
    parser.add_argument('--buffer_init_steps', type=int, default=1000, help='number of random actions to take before train loop')

    # sac hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for SAC RL')
    parser.add_argument('--alpha', type=float, default=0.2, help='discount factor for entropy for sac')
    parser.add_argument('--polyak', type=float, default=0.95, help='polyak averaging parameter')#.995???

    # resnet backbone args
    parser.add_argument('--return_node', type=str, default='layer4', choices=['layer1','layer2','layer3','layer4'], help='resnet layer to return features from')

    return parser.parse_args()


def train(args, device):
    # initialize the environment and get the first observation
    env = GraphTraverser(Graph(config_path='labGraphConfig.json'))
    obs = env.getImg()

    #TODO: need to make SAC goal conditioned when getting actions!

    # initialize and populate the replay buffer
    replay_buffer = GoalReplayBuffer(args)
    for step in range(args.buffer_init_steps):
        action = random.choice(range(env.action_space))
        obs_new, reward, goal_info, done = env.step(action)
        # transforming action to store it as a vector
        action_ind = action
        action = np.zeros(env.action_space)
        action[action_ind] = 1
        # goal info is currently (start node,  star dir, goal node, goal dir)
        # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
        goal_img = env.getImg(goal_info[-2], goal_info[-1])
        # sample of the shape (s, a, r, g, s', done)
        replay_buffer.addSample([obs, action, reward, goal_img, obs_new, done])
        if reward == 1 or step % args.steps_per_epoch==0:
            env.randomInit()
            obs = env.getImg()
        else:
            obs = obs_new

    # initialize the model
    img_backbone = ResnetBackbone(args, device)
    model = SACDiscreteBaseline(args, img_backbone, env.action_space, device)
    model.train()

    # start training
    ep_count = 0
    ep_reward = 0
    ep_len = 0
    total_qloss = 0
    total_policyloss = 0
    env.randomInit()
    obs = env.getImg()
    for step in range(args.epochs*args.steps_per_epoch):
        # initial random action or use the learned policy
        if step < args.use_policy_step:
            action = random.choice(range(env.action_space))
        else:
            goal_img = env.getImg()
            action = model.get_action(np.stack([obs]), np.stack([goal_img]))

        # take action in the environment and save to replay/update trackers
        obs_new, reward, goal_info, done = env.step(action)
        # storing the action as a vector
        action_ind = action
        action = np.zeros(env.action_space)
        action[action_ind] = 1
        # goal info is currently (start node,  star dir, goal node, goal dir)
        # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
        goal_img = env.getImg(goal_info[-2], goal_info[-1])
        # sample of the shape (s, a, r, g, s', done)
        replay_buffer.addSample([obs, action, reward, goal_img, obs_new, done])
        ep_reward+=reward
        ep_len+=1

        # if done or goal reached, reset the environment and trackers, save model
        # else switch over the obs
        if done or ep_len>=args.steps_per_epoch:
            env.randomInit()
            obs = env.getImg()
            print('Epoch',ep_count,'completed in',ep_len,'steps with reward =',ep_reward)
            print('\t Total Q Loss:',total_qloss,' Avg Q Loss:',total_qloss/ep_len)
            print('\t Total Policy Loss:',total_policyloss,' Avg Policy Loss:',total_policyloss/ep_len)
            print()
            ep_len=0
            ep_reward=0
            total_qloss = 0
            total_policyloss = 0
            ep_count+=1
            # TODO: should we relabel the buffer?
            model.save_models(args.save_name)
        else:
            obs = obs_new

        # update model
        # TODO: should we use this as a positive and negative example like Hindsight Experience Replay???

        # sample the buffer, do the q and policy updates, update trackers
        sample = replay_buffer.sample()  # (s, a, r, s', done)
        qloss, policyloss = model.update(sample)
        total_qloss += qloss
        total_policyloss += policyloss

    return model

if __name__ == '__main__':
    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = train(args, device)

