# dueling dqn, qnetwork, and memory from here: https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn.py

import cv2
import torch
import random
import torch.nn as nn
import numpy as np

from collections import deque

class QNetwork(nn.Module):
    def __init__(self,feat_dim=4):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(feat_dim, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 3)

    def forward(self, state):
        # breakpoint()
        state = state.flatten(1)
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

class DuelingDQN(nn.Module):
    def __init__(self, args=None, device=None, img_backbone=None):
        super(DuelingDQN,self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.img_backbone = img_backbone
        if args is None:
            self.gamma = 0.99
            self.explore = 20000
            self.init_epsilon = 0.1
            self.final_epsilon = 0.0001
            self.update_steps = 4
            self.lr = 1e-4
        else:
            self.gamma = args.gamma
            self.explore = args.explore
            self.init_epsilon = args.init_epsilon
            self.final_epsilon = args.final_epsilon
            self.update_steps = args.target_update_freq
            self.lr = args.lr

        self.epsilon = self.init_epsilon

        self.onlineQNetwork = QNetwork(args.feat_dim).to(device)
        self.targetQNetwork = QNetwork(args.feat_dim).to(device)
        self.targetQNetwork.load_state_dict(self.onlineQNetwork.state_dict())

        self.optimizer = torch.optim.Adam(self.onlineQNetwork.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def forward(self, x, goal):
        breakpoint()
        x = self.img_backbone.extractFeatures(x)
        goal = self.img_backbone.extractFeatures(goal)
        return self.onlineQNetwork(torch.cat((x,goal), axis=-1))

    def update(self, batch, step):
        # breakpoint()
        state, action, reward, goal, next_state, goal_new, done = batch
        # state = torch.FloatTensor(state).to(self.device)
        goal, goal_vec = np.stack(goal[:, 0]), np.stack(goal[:, 1])
        # goal = torch.FloatTensor(goal_imgs).to(self.device)
        goal_new, goal_vec = np.stack(goal_new[:, 0]), np.stack(goal_new[:, 1])
        # goal_new = torch.FloatTensor(goal_new_imgs).to(self.device)
        # next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        if self.img_backbone is not None:
            state_feats = self.img_backbone.extractFeatures(state)
            goal_feats = self.img_backbone.extractFeatures(goal)
            next_state_feats = self.img_backbone.extractFeatures(next_state)
            goal_new_feats = self.img_backbone.extractFeatures(goal_new)

        with torch.no_grad():
            onlineQ_next = self.onlineQNetwork(torch.cat((next_state_feats,goal_new_feats), axis=-1))
            targetQ_next = self.targetQNetwork(torch.cat((next_state_feats,goal_new_feats), axis=-1))
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            y = reward + (1 - done) * self.gamma * targetQ_next.gather(1, online_max_action.long())

        # breakpoint()
        loss = self.loss_func(self.onlineQNetwork(torch.cat((state_feats,goal_feats), axis=-1)).gather(1, action.long()), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore

        if step % self.update_steps == 0:
            self.targetQNetwork.load_state_dict(self.onlineQNetwork.state_dict())

        return loss.item()

    def select_action(self, x, goal):
        # breakpoint()
        x = np.stack([cv2.cvtColor(cv2.imread(x[0]), cv2.COLOR_BGR2RGB)])
        goal = np.stack([cv2.cvtColor(cv2.imread(goal[0]), cv2.COLOR_BGR2RGB)])
        x = self.img_backbone.extractFeatures(x)
        goal = self.img_backbone.extractFeatures(goal)
        return self.onlineQNetwork.select_action(torch.cat((x,goal), axis=-1))

    def save_models(self, save_name, goalNode, goalDirection):
        torch.save(self.onlineQNetwork.state_dict(), save_name+'.pt')#+str(goalNode)+'_'+str(goalDirection)+'.pt')
