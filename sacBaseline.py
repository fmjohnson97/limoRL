''' Implementation of a SAC baseline for the car racing openai gym environment with a resnet50 image processing backbone.
 Based HEAVILY on this:https://github.com/openai/spinningup/tree/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac '''

import torch
import random
import argparse
import itertools
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

from PIL import Image
from copy import deepcopy
from torchvision.models import feature_extraction, resnet18, ResNet18_Weights
from torch.nn import Linear, PReLU, Sigmoid, Tanh
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt


def getArgs():
    parser=argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default = 4000, help='max number of steps for each epoch')
    parser.add_argument('--use_policy_step', type=int, default=10000, help='number of steps before using the learned policy')
    parser.add_argument('--update_frequency', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_name', default=None, help='prefix name for saving the SAC networks')

    # buffer hyperparameters
    parser.add_argument('--buffer_limit', type=int, default=5000, help='max number of samples in the replay buffer')
    parser.add_argument('--buffer_init_steps', type=int, default=500, help='number of random actions to take before train loop')

    # sac hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for SAC RL')
    parser.add_argument('--alpha', type=float, default=0.2, help='discount factor for entropy for sac')
    parser.add_argument('--polyak', type=float, default=0.995, help='polyak averaging parameter')

    # policy hyperparameters
    parser.add_argument('--log_std_max', type=int, default=2, help='max bound for log_std clipping')
    parser.add_argument('--log_std_min', type=int, default=-20, help='min bound for log_std clipping')

    # resnet backbone args
    parser.add_argument('--return_node', type=str, default='layer4', choices=['layer1','layer2','layer3','layer4'], help='resnet layer to return features from')

    return parser.parse_args()


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SACBaseline(nn.Module):
    def __init__(self, args, backbone, action_dim, action_limit):
        super(SACBaseline, self).__init__()
        #TODO: learning schedule
        #TODO: alpha schedule

        # initialize networks
        # note: all using a shared feature extractor which isn't getting any loss backprop-ed
        feat_dim= 512 * 7 * 7
        self.q1network = QNetwork(backbone, feat_dim, action_dim)
        self.targetQ1 = deepcopy(self.q1network)
        self.q2network = QNetwork(backbone, feat_dim, action_dim)
        self.targetQ2 = deepcopy(self.q2network)
        self.policyNetwork = PolicyNetwork(args, backbone, feat_dim, action_dim, action_limit)

        # gathering hyperparameters
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.polyak = args.polyak

        #freeze target weights bc only updating with polyak averaging
        for param in self.targetQ1.parameters():
            param.requires_grad = False
        for param in self.targetQ2.parameters():
            param.requires_grad = False

        #initialized optimizers
        self.policy_opt = optim.Adam(self.policyNetwork.parameters(), lr=args.lr)
        self.q_opt = optim.Adam(itertools.chain(self.q1network.parameters(), self.q2network.parameters()), lr=args.lr)

        #init loss function
        self.loss_func=nn.MSELoss()

        #print stats
        self.total_params = sum(p.numel() for p in self.q1network.parameters() if p.requires_grad)
        self.total_params += sum(p.numel() for p in self.q2network.parameters() if p.requires_grad)
        self.total_params += sum(p.numel() for p in self.policyNetwork.parameters() if p.requires_grad)
        print('Initialized SAC Baseline with',self.total_params,'parameters!\n')

    @torch.no_grad()
    def get_action(self, observation, deterministic=False):
        action, _ = self.policyNetwork(observation, deterministic, False)
        return action.numpy()

    @torch.no_grad()
    def computeQTargets(self, sample):
        # (s, a, r, s', done)
        observation, action, reward, observation_new, done = sample
        #get the target action from the current policy
        action_new, log_pi_new = self.policyNetwork(observation_new)

        #get target q values
        targetq1_out = self.targetQ1(observation_new, torch.FloatTensor(action_new))
        targetq2_out = self.targetQ2(observation_new, torch.FloatTensor(action_new))
        q_targ_out = torch.min(targetq1_out, targetq2_out)

        return torch.FloatTensor(reward.reshape(len(reward), 1) + self.gamma * (1-done).reshape(len(done),1) * (q_targ_out.numpy() - self.alpha * log_pi_new.numpy()))

    def computePolicyLoss(self, observation):
        pi_action, log_pi_action = self.policyNetwork(observation)
        q1_policy = self.q1network(observation, torch.FloatTensor(pi_action))
        q2_policy = self.q2network(observation, torch.FloatTensor(pi_action))
        q_value = torch.min(q1_policy, q2_policy)
        return self.loss_func(q_value, self.alpha * log_pi_action)  # (alpha * logp_pi - q_pi).mean(); they use L1 loss for some reason???
        # TODO: change ^^^ if this doesn't converge

    def update(self, sample):
        observation, action, reward, observation_new, done = sample

        # compute q networks loss and backprop it
        self.q_opt.zero_grad()
        q_target = self.computeQTargets(sample)
        q1_out = self.q1network(observation, torch.FloatTensor(action))
        q2_out = self.q2network(observation, torch.FloatTensor(action))
        q1_loss = self.loss_func(q1_out, q_target)
        q2_loss = self.loss_func(q2_out, q_target)
        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.q_opt.step()

        # freeze q weights to ease policy network backprop computation
        for param in self.q1network.parameters():
            param.requires_grad = False
        for param in self.q2network.parameters():
            param.requires_grad = False

        # compute policy network loss and backprop it
        self.policy_opt.zero_grad()
        policy_loss = self.computePolicyLoss(observation)
        policy_loss.backward()
        self.policy_opt.step()

        #update target q networks; done before grad turned back on so no loss props to the target networks
        with torch.no_grad():
            for param, targ_param in zip(self.q1network.parameters(), self.targetQ1.parameters()):
                targ_param.data.mul_(self.polyak)
                targ_param.data.add_((1 - self.polyak) * param.data)
            for param, targ_param in zip(self.q2network.parameters(), self.targetQ2.parameters()):
                targ_param.data.mul_(self.polyak)
                targ_param.data.add_((1 - self.polyak) * param.data)

        # turn grad back on for the q networks
        for param in self.q1network.parameters():
            param.requires_grad = True
        for param in self.q2network.parameters():
            param.requires_grad = True

        return q_loss.item(), policy_loss.item()

    def save_models(self,name):
        if name is None:
            torch.save(self.q1network.state_dict(), 'sacBaseline_q1.pt')
            torch.save(self.q2network.state_dict(), 'sacBaseline_q2.pt')
            torch.save(self.policyNetwork.state_dict(), 'sacBaseline_policy.pt')
        else:
            torch.save(self.q1network.state_dict(), name+'_q1.pt')
            torch.save(self.q2network.state_dict(), name+'_q2.pt')
            torch.save(self.policyNetwork.state_dict(), name+'_policy.pt')


class QNetwork(nn.Module):
    def __init__(self, backbone, feat_dim, action_dim):
        super(QNetwork, self).__init__()
        #pretrained feature extractor
        self.backbone = backbone
        self.fc1=Linear(feat_dim + action_dim,feat_dim // 8)
        self.fc2 = Linear(feat_dim // 8, feat_dim // 64)
        self.fc3 = Linear(feat_dim // 64, 1)
        self.prelu1 = PReLU()
        self.prelu2 = PReLU()

        weights_init_(self.fc1)
        weights_init_(self.fc2)
        weights_init_(self.fc3)

    def forward(self, img, action):
        # takes in the state and the corresponding action given that state and random noise?
        feats=self.backbone.extractFeatures(img)
        feats = self.prelu1(self.fc1(torch.cat((feats, action), axis=-1)))
        feats = self.prelu2(self.fc2(feats))
        return self.fc3(feats)


class PolicyNetwork(nn.Module):
    def __init__(self, args, backbone, feat_dim, action_dim, action_limit):
        super(PolicyNetwork, self).__init__()
        # pretrained feature extractor
        self.backbone = backbone
        self.fc1 = Linear(feat_dim, feat_dim // 8)
        self.fc2 = Linear(feat_dim // 8, feat_dim // 64)
        self.prelu1 = PReLU()
        self.prelu2 = PReLU()
        self.mean = Linear(feat_dim // 64, action_dim)
        self.log_std = Linear(feat_dim // 64, action_dim)
        self.log_std_max = args.log_std_max
        self.log_std_min = args.log_std_min
        self.action_limit=action_limit[1]
        self.action_limit_low=action_limit[0]

        weights_init_(self.mean)
        weights_init_(self.log_std)
        weights_init_(self.fc1)
        weights_init_(self.fc2)

    def forward(self, img, deterministic=False, withLogProb=True):
        #extracts features from the image observation
        feats = self.backbone.extractFeatures(img)
        feats = self.prelu1(self.fc1(feats))
        feats = self.prelu2(self.fc2(feats))
        #Needs to output (1) the steering angle (between [-1,1]) and (2) whether to push the gas and (3) whether to push the break
        # steering angle is continuous (so need mean and std output for probabilistic modeling)
        # gas and break are disrecete (so sigmoid them to a boolean)

        mu = self.mean(feats)
        log_std = torch.clamp(self.log_std(feats), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if withLogProb:
            log_pi = pi_distribution.log_prob(pi_action).sum(axis=-1) - (2 * (np.log(2) - pi_action - nn.functional.softplus(-2 * pi_action))).sum(axis=1)
            log_pi = log_pi.reshape(len(log_pi),1)
        else:
            log_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.action_limit * pi_action #Note: Assumes that the action space is symmetric about 0!!!!!

        return pi_action, log_pi


class ReplayBuffer():
    def __init__(self, args):
        self.buffer=[]
        self.limit=args.buffer_limit
        self.batch_size=args.batch_size

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size=self.batch_size
        s=[]
        a=[]
        r=[]
        sprime=[]
        done=[]
        for i in range(batch_size):
            ind=random.randint(0,len(self.buffer)-1)
            temp=self.buffer.pop(ind)
            s.append(temp[0])
            a.append(temp[1])
            r.append(temp[2])
            sprime.append(temp[3])
            done.append(temp[4])
        return [np.stack(s), np.stack(a), np.stack(r), np.stack(sprime), np.stack(done)]

    def addSample(self, sample):
        #sample of the shape (s, a, r, s', done)
        self.buffer.append(sample)
        while len(self.buffer)>self.limit:
            self.buffer.pop(0)


class ResnetBackbone():
    def __init__(self, device, args):
        self.return_node = args.return_node
        self.createResnetBackbone(device)

    def createResnetBackbone(self,device):
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        model = resnet18(weights=weights).to(device)
        model.eval()
        self.feat_ext = feature_extraction.create_feature_extractor(model, return_nodes=[self.return_node]).to(device)
        for param in self.feat_ext.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extractFeatures(self, img):
        img_ext=[]
        if type(img) != Image:
            for im in img:
                im = Image.fromarray(im)
                img_ext.append(torch.FloatTensor(self.preprocess(im)))
        features = self.feat_ext(torch.stack(img_ext))
        # breakpoint()
        return features[self.return_node].reshape(len(features[self.return_node]), -1)


def train(args):
    #TODO: put all the network inputs on the same device
    #TODO: add win condition ending for training
    #TODO: test the model performance

    # make the resnet backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_backbone = ResnetBackbone(device, args)

    # empty replay buffer
    replay_buffer=ReplayBuffer(args)

    # initialize enironment
    env=gym.make("CarRacing-v2",max_episode_steps=args.steps_per_epoch, render_mode = "rgb_array")#"human")#, domain_randomize=True
    # obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0] #there's 3: steering[-1,1], break, gas
    act_limit_high = env.action_space.high[0]
    act_limit_low = env.action_space.low[0]

    #initialize networks/sac
    sac = SACBaseline(args, img_backbone, act_dim, [act_limit_low, act_limit_high])

    # initial data collection
    observation, info = env.reset(seed=42)
    ep_len = 0
    ep_reward = 0
    ep_count=0
    total_qloss = 0
    total_policyloss = 0

    ## train loop
    for step in range(args.steps_per_epoch * args.epochs): #TODO: determine how many updates you actually want to do
        if step < args.use_policy_step:
            # take random actions and add to the buffer (s, a, r, s', done)
            action = env.action_space.sample()
        else:
            action = sac.get_action(observation)
            frame = env.render()
            plt.imshow(frame)
            plt.pause(0.1)

        observation_new, reward, terminated, truncated, info = env.step(action)
        ep_len += 1
        ep_reward += reward

        # add to buffer
        d = terminated or truncated
        d = False if ep_len == args.steps_per_epoch else d #so that timing out isn't penalized
        replay_buffer.addSample([observation, action, reward, observation_new, d])

        #check if done else switch the observations
        if terminated or truncated or ep_len == args.steps_per_epoch:
            observation, info = env.reset()
            ep_count+=1
            print('Episode',ep_count,'completed in',ep_len,'steps with',ep_reward,'reward!')
            print('\t Total Q loss:',total_qloss,'Total Policy loss:',total_policyloss)
            print('\t (per step) Avg Q loss:',total_qloss/ep_len,'Policy loss:',total_policyloss/ep_len,'\n')
            ep_len = 0
            ep_reward = 0
            total_qloss = 0
            total_policyloss = 0
            sac.save_models(args.save_name)
            # Test the performance of the deterministic version of the agent.
            #     test_agent()
            # print test reward and episode length
        else:
            observation=observation_new

        if len(replay_buffer.buffer) >= args.buffer_init_steps and step % args.update_frequency==0:
            #sample from the replay buffer
            sample = replay_buffer.sample() #(s, a, r, s', done)

            # do the q and policy updates
            qloss, policyloss = sac.update(sample)
            total_qloss+=qloss
            total_policyloss+=policyloss

        #     logger.log_tabular('Q1Vals', with_min_and_max=True)
        #     logger.log_tabular('Q2Vals', with_min_and_max=True)
        #     logger.log_tabular('LogPi', with_min_and_max=True)

    env.close()
    return sac


    #TODO: potential test function for the agent
    # def test_agent():
    #     for j in range(num_test_episodes):
    #         o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    #         while not (d or (ep_len == max_ep_len)):
    #             # Take deterministic actions at test time
    #             o, r, d, _ = test_env.step(get_action(o, True))
    #             ep_ret += r
    #             ep_len += 1
    #         logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


if __name__=='__main__':
    args=getArgs()
    sac = train(args)