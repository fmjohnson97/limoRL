import torch
import random
import itertools
import torch.nn as nn
import numpy as np
import torch.optim as optim

from PIL import Image
from copy import deepcopy
from torchvision.models import feature_extraction, resnet18, ResNet18_Weights
from torch.nn import Linear, ReLU, Sigmoid
from torch.distributions.categorical import Categorical

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

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
            try:
                img_ext=torch.FloatTensor(img).unsqueeze(0).to(self.device)
            except Exception as e:
                print(e)
                breakpoint()

        features = self.feat_ext(img_ext)
        # breakpoint()
        return features[self.return_node].reshape(len(features[self.return_node]), -1)

class SACDiscreteBaseline(nn.Module):
    def __init__(self, args, backbone, action_dim, device):
        super(SACDiscreteBaseline, self).__init__()
        #TODO: try value based method instead?
        #TODO: learning schedule
        #TODO: alpha schedule
        #TODO: add save flag for networks

        #TODO: add learned alpha

        self.device = device
        self.backbone=backbone
        # initialize networks
        # note: all using a shared feature extractor which isn't getting any loss backprop-ed
        feat_dim= 512 * 7 * 7 * 2 #*2 is for goal # cnn 4096# fc 2048 # resnet 512 * 7 * 7
        # self.img_fc = Linear(96*96*3,feat_dim).to(device)
        # self.img_fc = nn.Sequential(Conv2d(3, 32, kernel_size=8, stride=4, padding=0), ReLU(),
        #                             Conv2d(32, 64, kernel_size=4, stride=2, padding=0), ReLU(),
        #                             Conv2d(64, 64, kernel_size=3, stride=1, padding=0), Flatten()).to(device)
        # self.prelu = PReLU().to(device)
        self.q1network = QNetwork(feat_dim, action_dim).to(device)
        self.targetQ1 = deepcopy(self.q1network)
        self.targetQ1 = self.targetQ1.to(device)
        self.q2network = QNetwork(feat_dim, action_dim).to(device)
        self.targetQ2 = deepcopy(self.q2network)
        self.targetQ2 = self.targetQ2.to(device)
        self.policyNetwork = PolicyNetwork(args, feat_dim, action_dim).to(device)

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
        self.q1_opt = optim.Adam(self.q1network.parameters(), lr=args.lr)
        self.q2_opt = optim.Adam(self.q2network.parameters(), lr=args.lr)
        # self.q_opt = optim.Adam(itertools.chain(self.q1network.parameters(), self.q2network.parameters()), lr=args.lr)

        #init loss function
        self.q_loss_func=nn.MSELoss()
        self.pi_loss_func = nn.L1Loss()

        #print stats
        self.total_params = sum(p.numel() for p in self.q1network.parameters() if p.requires_grad)
        self.total_params += sum(p.numel() for p in self.q2network.parameters() if p.requires_grad)
        self.total_params += sum(p.numel() for p in self.policyNetwork.parameters() if p.requires_grad)
        print('Initialized SAC Baseline with',self.total_params,'parameters!\n')

    @torch.no_grad()
    def get_action(self, observation, goal_obs):
        # breakpoint()
        img_feats = self.backbone.extractFeatures(observation)
        goal_feats = self.backbone.extractFeatures(goal_obs)
        img_feats = torch.cat((img_feats,goal_feats), axis=-1)
        action_dist, action, log_action = self.policyNetwork(img_feats)
        # breakpoint()
        action = action.argmax(-1)
        return action.squeeze().cpu().numpy()

    @torch.no_grad()
    def computeQTargets(self, sample):
        # breakpoint()
        # (s, a, r, g, s', done)
        observation, action, reward, goal_imgs, observation_new, done = sample
        img_new_feats = self.backbone.extractFeatures(observation_new) #self.prelu(self.img_fc(observation_new)) #
        goal_feats = self.backbone.extractFeatures(goal_imgs)
        img_new_feats = torch.cat((img_new_feats, goal_feats), axis=-1)
        # print('imgnew',img_new_feats.mean())
        #get the target action from the current policy
        action_dist_new, action_new, log_action_new = self.policyNetwork(img_new_feats)

        #get target q values
        targetq1_out = self.targetQ1(img_new_feats, action_new)
        targetq2_out = self.targetQ2(img_new_feats, action_new)
        q_targ_out = torch.min(targetq1_out, targetq2_out)

        q_target = torch.FloatTensor(reward + self.gamma * (1-done) * np.sum((q_targ_out.cpu().numpy() - self.alpha * log_action_new.cpu().numpy())*action_new.cpu().numpy(), axis=-1))#TODO: ???
        q_target = q_target.reshape(len(q_target),-1).to(self.device)
        return q_target

        # this was for computing the policy loss
        #self.pi_loss_func(q_value, self.alpha * log_action)  #(q_value - self.alpha * log_pi_action).mean() # they use L1 loss for some reason???

    def update(self, sample):
        # breakpoint()
        observation, action, reward, goal_imgs, observation_new, done = sample
        # print(goal_info)
        img_feats = self.backbone.extractFeatures(observation) #self.prelu(self.img_fc(observation)) #
        goal_feats = self.backbone.extractFeatures(goal_imgs)
        img_feats = torch.cat((img_feats,goal_feats), axis=-1)
        action = torch.FloatTensor(action)#.reshape(-1,1)
        action = action.to(self.device)

        # compute q networks loss and backprop it
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_target = self.computeQTargets(sample)
        q1_out = self.q1network(img_feats, action)[torch.where(action==1)].reshape(-1,1) #
        q2_out = self.q2network(img_feats, action)[torch.where(action==1)].reshape(-1,1) #
        q1_loss = self.q_loss_func(q1_out, q_target)
        q2_loss = self.q_loss_func(q2_out, q_target)
        # q1_loss = torch.clamp(q1_loss, -1, 1)
        # q2_loss = torch.clamp(q2_loss, -1, 1)
        # q_loss = q1_loss + q2_loss #torch.clamp(q1_loss + q2_loss,-10,10) #TODO: determine how to fix besides reward clipping?
        # q_loss = torch.min(q1_loss, q2_loss)
        # q_loss.backward()#retain_graph=True)
        q1_loss.backward()
        q2_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        # freeze q weights to ease policy network backprop computation
        # for param in self.q1network.parameters():
        #     param.requires_grad = False
        # for param in self.q2network.parameters():
        #     param.requires_grad = False

        # breakpoint()
        # compute policy network loss and backprop it
        self.policy_opt.zero_grad()
        action_dist, action, log_action = self.policyNetwork(img_feats)
        with torch.no_grad():
            q1_policy = self.q1network(img_feats, action)
            q2_policy = self.q2network(img_feats, action)
            q_value = torch.min(q1_policy, q2_policy)
        policy_loss = torch.sum((self.alpha * log_action - q_value) * action, axis=-1).mean()
        # policy_loss = torch.clamp(policy_loss, -1, 1)
        # print('losses',q1_loss, q2_loss, q1_loss+q2_loss, policy_loss)
        policy_loss.backward()
        self.policy_opt.step()

        # print('q vals + targ', q1_out.mean(),q2_out.mean(), q_target.mean())

        #update target q networks; done before grad turned back on so no loss props to the target networks
        with torch.no_grad():
            for param, targ_param in zip(self.q1network.parameters(), self.targetQ1.parameters()):
                targ_param.data.mul_(self.polyak)
                targ_param.data.add_((1 - self.polyak) * param.data)
            for param, targ_param in zip(self.q2network.parameters(), self.targetQ2.parameters()):
                targ_param.data.mul_(self.polyak)
                targ_param.data.add_((1 - self.polyak) * param.data)

        # turn grad back on for the q networks
        # for param in self.q1network.parameters():
        #     param.requires_grad = True
        # for param in self.q2network.parameters():
        #     param.requires_grad = True

        # return q_loss.item(), policy_loss.item()
        return (q1_loss+q2_loss).item(), policy_loss.item()

    def save_models(self,name):
        if name is None:
            torch.save(self.q1network.state_dict(), 'sacBaselineDisc_q1.pt')
            torch.save(self.q2network.state_dict(), 'sacBaselineDisc_q2.pt')
            torch.save(self.policyNetwork.state_dict(), 'sacBaselineDisc_policy.pt')
        else:
            torch.save(self.q1network.state_dict(), name+'_q1.pt')
            torch.save(self.q2network.state_dict(), name+'_q2.pt')
            torch.save(self.policyNetwork.state_dict(), name+'_policy.pt')


class QNetwork(nn.Module):
    def __init__(self, feat_dim, action_dim):
        super(QNetwork, self).__init__()
        #pretrained feature extractor
        # self.fc1=Linear(feat_dim + action_dim,feat_dim // 8)
        # self.fc2 = Linear(feat_dim // 8, feat_dim // 64)
        # self.fc3 = Linear(feat_dim // 64, action_dim)

        self.fc1 = Linear(feat_dim + action_dim, feat_dim // 256)
        self.fc2 = Linear(feat_dim // 256, feat_dim // 256)
        self.fc3 = Linear(feat_dim // 256, action_dim)

        # self.fc1=Linear(feat_dim + action_dim,feat_dim)
        # self.fc2 = Linear(feat_dim, action_dim)
        self.prelu1 = ReLU()
        self.prelu2 = ReLU()

        weights_init_(self.fc1)
        weights_init_(self.fc2)
        weights_init_(self.fc3)

    def forward(self, img_feats, action):
        feats = self.prelu1(self.fc1(torch.cat((img_feats, action), axis=-1)))
        feats = self.prelu2(self.fc2(feats))
        return self.fc3(feats)

class PolicyNetwork(nn.Module):
    def __init__(self, args, feat_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # self.fc1 = Linear(feat_dim, feat_dim // 8)
        # self.fc2 = Linear(feat_dim // 8, feat_dim // 64)
        # self.logits = Linear(feat_dim // 64,action_dim)

        self.fc1 = Linear(feat_dim, feat_dim // 256)
        self.fc2 = Linear(feat_dim // 256, feat_dim // 256)
        self.logits = Linear(feat_dim // 256, action_dim)

        # self.fc1 = Linear(feat_dim , feat_dim )
        # self.fc2 = Linear(feat_dim , action_dim)
        self.prelu1 = ReLU()
        self.prelu2 = ReLU()
        self.sigmoid = Sigmoid()

        weights_init_(self.fc1)
        weights_init_(self.fc2)
        weights_init_(self.logits)

    def forward(self, img_feats):
        #extracts features from the image observation
        feats = self.prelu1(self.fc1(img_feats))
        # logits = self.sigmoid(self.fc2(feats))
        feats = self.prelu2(self.fc2(feats))
        #Needs to output [0: do nothing, 1: steer left, 2: steer right, 3: gas, 4: brake]
        logits = self.sigmoid(self.logits(feats))

        # z = logits == 0.0
        z = logits < 1e-8
        z = z.float() * 1e-8

        # print('logits',logits.mean(), (logits+z).mean(), torch.log(logits+z).mean())
        # breakpoint()

        return Categorical(logits+z), logits + z, torch.log(logits + z)

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
            temp=self.buffer[ind]
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

class GoalReplayBuffer():
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
        g=[]
        sprime=[]
        done=[]
        for i in range(batch_size):
            ind=random.randint(0,len(self.buffer)-1)
            temp=self.buffer[ind]
            s.append(temp[0])
            a.append(temp[1])
            r.append(temp[2])
            g.append(temp[3])
            sprime.append(temp[4])
            done.append(temp[5])
        try:
            return [np.stack(s), np.stack(a), np.stack(r), np.stack(g), np.stack(sprime), np.stack(done)]
        except Exception as e:
            print(e)
            breakpoint()

    def addSample(self, sample):
        # sample of the shape (s, a, r, g, s', done)
        self.buffer.append(sample)
        while len(self.buffer)>self.limit:
            self.buffer.pop(0)

    # Hindsight Experience Replay
    def addHERSample(self, sample, new_reward):
        # sample of the form s, a, r, g, s', done
        sample[3] = sample[4]  # replace old goal with next state (as if trying to get there the whole time)
        sample[2] = new_reward  # replace old reward with success reward (bc achieved the goal)
        sample[5] = True  # changing done to True because goal reached
        self.addSample(sample)
