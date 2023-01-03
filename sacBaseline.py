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

from copy import deepcopy
from torchvision.models import resnet50, ResNet50_Weights, feature_extraction
from torch.nn import Linear, PReLU, Sigmoid, Tanh
from torch.distributions.normal import Normal


def getArgs():
    parser=argparse.ArgumentParser()

    parser.add_argument('--buffer_limit', type=int, default=5000, help='max number of samples in the replay buffer')
    parser.add_argument('--steps', type=int, default=100000, help='number of steps to train the network for')
    parser.add_argument('--buffer_init_steps', type=int, default=500, help='number of random actions to take before train loop')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default = 4000, help='max number of steps for each epoch')
    parser.add_argument('--use_policy_step', type=int, default=10000, help='number of steps before using the learned policy')
    parser.add_argument('--update_frequency', type=int, default=50)
    return parser.parse_args()

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class SACBaseline(nn.Module):
    def __init__(self, backbone, observation_dim, action_dim, action_limit, discount=.99, alpha=0.2, lr=1e-3, polyak=0.995):
        super(SACBaseline, self).__init__()
        #TODO: add constants to arg parser
        #TODO: learning schedule
        #TODO: alpha schedule

        # initialize networks
        # note: all using a shared feature extractor which isn't getting any loss backprop-ed
        for param in backbone.parameters():
            param.requires_grad = False
        self.q1network = QNetwork(backbone)
        self.targetQ1 = deepcopy(self.q1network)
        self.q2network = QNetwork(backbone)
        self.targetQ2 = deepcopy(self.q2network)
        self.policyNetwork = PolicyNetwork(backbone, action_dim, action_limit)

        # gathering hyperparameters
        self.gamma = discount
        self.alpha = alpha
        self.polyak = polyak

        #freeze target weights bc only updating with polyak averaging
        for param in self.targetQ1.parameters():
            param.requires_grad = False
        for param in self.targetQ2.parameters():
            param.requires_grad = False

        #initialized optimizers
        self.policy_opt = optim.Adam(self.policyNetwork, lr=lr)
        self.q_opt = optim.Adam(itertools.chain(self.q1network.parameters(), self.q2network.parameters()), lr=lr)

        #init loss function
        self.loss_func=nn.MSELoss()

        #print stats
        self.total_params=sum(p.numel() for p in self.q1network.parameters() if p.requires_grad)
        self.total_params += sum(p.numel() for p in self.q2network.parameters() if p.requires_grad)
        self.total_params += sum(p.numel() for p in self.policyNetwork.parameters() if p.requires_grad)
        print('Initialized SAC Baseline with',self.total_params,'parameters!\n')

    @torch.nograd()
    def get_action(self, observation, deterministic=False):
        action, _ = self.policyNetwork(observation, deterministic, False)
        return action.numpy()

    @torch.no_grad()
    def computeQTargets(self, sample):
        # (s, a, r, s', done)
        observation, action, reward, observation_new, done = sample
        if done:
            return reward
        else:
            #get the target action from the current policy
            action_new, log_pi_new = self.policyNetwork(observation_new)

            #get target q values
            targetq1_out = self.targetQ1(torch.cat((observation_new, action_new), axis=-1))
            targetq2_out = self.targetQ2(torch.cat((observation_new, action_new), axis=-1))
            q_targ_out = torch.min(targetq1_out, targetq2_out)

        return reward + self.gamma * (q_targ_out - self.alpha*log_pi_new)

    def computePolicyLoss(self, observation):
        pi_action, log_pi_action = self.policyNetwork(observation)
        q1_policy = self.q1network(torch.cat((observation, pi_action), axis=-1))
        q2_policy = self.q2network(torch.cat((observation, pi_action), axis=-1))
        q_value = torch.min(q1_policy, q2_policy)
        return self.loss_func(q_value,self.alpha * log_pi_action)  # (alpha * logp_pi - q_pi).mean(); they use L1 loss for some reason???
        # TODO: change ^^^ if this doesn't converge

    def update(self, sample):
        #TODO: make sure this works with batches >1
        observation, action, reward, observation_new, done = sample

        # compute q networks loss
        q_target = self.computeQTargets(sample)
        q1_out = self.q1network(torch.cat((observation, action), axis=-1))
        q2_out = self.q2network(torch.cat((observation, action), axis=-1))
        q1_loss = self.loss_func(q1_out, q_target)
        q2_loss = self.loss_func(q2_out,q_target)
        q_loss = q1_loss+q2_loss

        #get policy network loss
        policy_loss = self.computePolicyLoss(observation)

        # backprop losses
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        for param in self.q1network:
            param.requires_grad = False
        for param in self.q2network:
            param.requires_grad = False

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        for param in self.q1network:
            param.requires_grad = True
        for param in self.q2network:
            param.requires_grad = True

        #update target q networks
        with torch.no_grad():
            for param, targ_param in zip(self.q1network.parameters(), self.targetQ1.parameters()):
                targ_param.data.mul_(self.polyak)
                targ_param.data.add_((1 - self.polyak) * param.data)

class QNetwork(nn.Module):
    def __init__(self, backbone):
        super(QNetwork, self).__init__()
        #pretrained feature extractor
        self.backbone = backbone
        self.fc1=Linear(12,1)

        weights_init_(self.fc1)

    def forward(self, img):
        # takes in the state and the corresponding action given that state and random noise?
        feats=self.backbone.extractFeatures(img)
        breakpoint()
        return self.fc1(feats)


class PolicyNetwork(nn.Module):
    def __init__(self, backbone, action_dim, action_limit, log_std_max =2, log_std_min = -20):
        super(PolicyNetwork, self).__init__()
        # pretrained feature extractor
        self.backbone = backbone
        self.mean = Linear(12, action_dim)
        self.log_std = Linear(12, action_dim)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.action_limit=action_limit[1]
        self.action_limit_low=action_limit[0]

        weights_init_(self.mean)
        weights_init_(self.log_std)

    def forward(self, img, deterministic=False, withLogProb=True):
        #extracts features from the image observation
        feats = self.backbone.extractFeatures(img)
        breakpoint()
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
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - nn.F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            log_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, log_pi


class ReplayBuffer():
    def __init__(self, batch_size=1, limit=5000):
        self.buffer=[]
        self.limit=limit
        self.batch_size=batch_size

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size=self.batch_size
        buff_output=[]
        for i in range(batch_size):
            ind=random.randint(0,len(self.buffer)-1)
            buff_output.append(self.buffer.pop(ind))
        return np.array(buff_output)

    def addSample(self, sample):
        #sample of the shape (s, a, r, s', done)
        self.buffer.append(sample)
        while len(self.buffer)>self.limit:
            self.buffer.pop(0)


class ResnetBackbone():
    def __init__(self, device, return_node='layer4'):
        self.return_node = return_node
        self.createResnetBackbone(device)

    def createResnetBackbone(self,device):
        weights = ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        model = resnet50(weights=weights).to(device)
        model.eval()
        self.feat_ext = feature_extraction.create_feature_extractor(model, return_nodes=[self.return_node]).to(device)

    @torch.no_grad()
    def extractFeatures(self, img):
        img_ext = self.preprocess(img).unsqueeze(0)
        features = self.feat_ext(img_ext)
        breakpoint()
        return features[self.return_node].flatten()


def train(args, img_backbone):
    #TODO: add progress tracking for epoch reward and steps to termination

    # empty replay buffer
    replay_buffer=ReplayBuffer(args.buffer_limit)

    # initialize enironment
    env=gym.make("CarRacing-v2",max_episode_steps=args.steps_per_epoch)#, domain_randomize=True
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0] #there's 3: steering[-1,1], break, gas
    act_limit_high = env.action_space.high[0]
    act_limit_low = env.action_space.low[0]

    #initialize networks/sac
    sac = SACBaseline(img_backbone,obs_dim, act_dim, [act_limit_low, act_limit_high])

    # initial data collection
    observation, info = env.reset(seed=42)
    ep_len = 0

    ## train loop
    for step in range(args.steps_per_epoch * args.epochs): #TODO: determine how many updates you actually want to do
        if step < args.use_policy_step:
            # take random actions and add to the buffer (s, a, r, s', done)
            action = env.action_space.sample()
            breakpoint() #to see what the possible actions look like
        else:
            action = sac.get_action(observation)

        observation_new, reward, terminated, truncated, info = env.step(action) #TODO: this may not work with the brake and gas actions???
        ep_len += 1

        # add to buffer
        d = terminated or truncated
        d = False if ep_len == args.steps_per_epoch else d #so that timing out isn't penalized
        replay_buffer.addSample([observation, action, reward, observation_new, d])

        #check if done else switch the observations
        if terminated or truncated or ep_len == args.steps_per_epoch:
            observation, info = env.reset()
            ep_len=0
        else:
            observation=observation_new

        if len(replay_buffer) >= args.buffer_init_steps and step % args.update_frequency==0:
            #sample from the replay buffer
            sample = replay_buffer.sample() #(s, a, r, s', done)

            # do the q and policy updates
            sac.update(sample)

            #end of epoch logging???
            # # End of epoch handling
            # if (t + 1) % steps_per_epoch == 0:
            #     epoch = (t + 1) // steps_per_epoch
            #
            #     # Save model
            #     if (epoch % save_freq == 0) or (epoch == epochs):
            #         logger.save_state({'env': env}, None)
            #
            #     # Test the performance of the deterministic version of the agent.
            #     test_agent()
            #
            #     # Log info about epoch
            #     logger.log_tabular('Epoch', epoch)
            #     logger.log_tabular('EpRet', with_min_and_max=True)
            #     logger.log_tabular('TestEpRet', with_min_and_max=True)
            #     logger.log_tabular('EpLen', average_only=True)
            #     logger.log_tabular('TestEpLen', average_only=True)
            #     logger.log_tabular('TotalEnvInteracts', t)
            #     logger.log_tabular('Q1Vals', with_min_and_max=True)
            #     logger.log_tabular('Q2Vals', with_min_and_max=True)
            #     logger.log_tabular('LogPi', with_min_and_max=True)
            #     logger.log_tabular('LossPi', average_only=True)
            #     logger.log_tabular('LossQ', average_only=True)
            #     logger.log_tabular('Time', time.time() - start_time)

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
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_backbone=ResnetBackbone(device)
    sac = train(args, img_backbone)