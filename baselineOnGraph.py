import random
import torch
import math
import argparse
import numpy as np
import time

from models import SimpleSACDisc, ResnetBackbone, GoalReplayBuffer
from graph import GraphTraverser, Graph

def getArgs():
    parser=argparse.ArgumentParser()

    #file parameters
    parser.add_argument('--config_file', type=str, default='labGraphConfig5.json', help='path to the graph config file')

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default = 20, help='max number of steps for each epoch')
    parser.add_argument('--use_policy_step', type=int, default=200, help='number of steps before using the learned policy')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')
    parser.add_argument('--save_name', default='sacDiscOnGraph', help='prefix name for saving the SAC networks')
    parser.add_argument('--target_update_freq', type=int, default=20, help='max number of samples in the replay buffer')
    parser.add_argument('--dist_reward', action='store_true', help='use the distance reward instead')
    parser.add_argument('--test', action='store_true', help='skip training and just test')
    parser.add_argument('--test_freq', type=int, default=10, help='number of epochs between testing')
    parser.add_argument('--epsilon', type=float, default=0.3, help='learning rate for training')
    parser.add_argument('--feat_dim', type=int, default=12, help='network input dimensions')
    parser.add_argument('--factor', type=int, default=1, help='hidden size of Q and Policy networks = feat_dim//factor')



    # buffer hyperparameters
    parser.add_argument('--buffer_limit', type=int, default=50000, help='max number of samples in the replay buffer')
    parser.add_argument('--buffer_init_steps', type=int, default=4000, help='number of random actions to take before train loop')
    parser.add_argument('--max_reward', type=float, default=1, help='max reward achievable by agent when reaches goal')

    # sac hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for SAC RL')
    parser.add_argument('--alpha', type=float, default=0.2, help='discount factor for entropy for sac')
    parser.add_argument('--polyak', type=float, default=0.995, help='polyak averaging parameter')

    # resnet/backbone args
    parser.add_argument('--return_node', type=str, default='layer4', choices=['layer1','layer2','layer3','layer4'], help='resnet layer to return features from')
    parser.add_argument('--hidden_dim', type=int, default=128, help='size of the latent vector of the AE')

    return parser.parse_args()


def train(args, device):
    # breakpoint()
    # initialize the environment and get the first observation
    env = GraphTraverser(Graph(config_path=args.config_file), distance_reward=args.dist_reward)
    env.randomInit()
    obs = env.getLandmarkVector()

    # initialize and populate the replay buffer
    replay_buffer = GoalReplayBuffer(args)
    for step in range(args.buffer_init_steps):
        # breakpoint()
        action = random.choice(range(env.action_space))
        # goal info is currently (start node,  star dir, goal node, goal dir)
        # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
        goal_img = []
        goal_vec = env.getLandmarkVector('goal')
        obs_new, reward, goal_info, done = env.step(action)
        obs_new = env.getLandmarkVector()
        # transforming action to store it as a vector
        action_ind = action
        action = np.zeros(env.action_space)
        action[action_ind] = 1
        goal_img_new = []
        goal_vec_new = env.getLandmarkVector('goal')
        # sample of the shape (s, a, r, g, s', g', done)
        replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new, goal_vec_new), done])
        # implementing hindsight experience replay
        # replay_buffer.addHERSample([obs, action, reward, goal_img, obs_new, done], args.max_reward)
        if done or reward == 1 or (step>0 and step % args.steps_per_epoch==0):
            env.randomInit()
            obs = env.getLandmarkVector()
        else:
            obs = obs_new

    # initialize the model
    model = SimpleSACDisc(args, env.action_space, device)
    model.train()

    # start training
    ep_count = 0
    ep_reward = 0
    ep_len = 0
    total_qloss = 0
    total_policyloss = 0
    env.randomInit()
    obs = env.getLandmarkVector()
    for step in range(args.epochs*args.steps_per_epoch):
        # initial random action or use the learned policy
        rand_num = random.random()
        # last part is epsilon greedy
        if step < args.use_policy_step or rand_num < math.exp(-1. * step / 1000):
            action = random.choice(range(env.action_space))
            # goal info is currently (start node,  star dir, goal node, goal dir)
            # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
            goal_img = []
            goal_vec = env.getLandmarkVector('goal')
        else:
            goal_img = []
            goal_vec = env.getLandmarkVector('goal')
            action = model.get_action(np.stack([obs]), np.stack([goal_vec]))

        # take action in the environment and save to replay/update trackers
        # if step > args.use_policy_step:
        #     print('pre state+goal', (env.current_node, env.current_direction), (env.goalNode, env.goalDirection))
        obs_new, reward, goal_info, done = env.step(action)
        obs_new = env.getLandmarkVector()
        # storing the action as a vector
        action_ind = action
        action = np.zeros(env.action_space)
        action[action_ind] = 1
        goal_img_new = []
        goal_vec_new = env.getLandmarkVector('goal')
        # sample of the shape (s, a, r, g, s', done)
        replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new,goal_vec_new), done])
        # implementing hindsight experience replay
        # replay_buffer.addHERSample([obs, action, reward, goal_img, obs_new, done], args.max_reward)
        # if step>args.use_policy_step:
        #     print('action+reward+state+goal', action, reward, (env.current_node, env.current_direction), (env.goalNode, env.goalDirection))
        ep_reward+=reward
        ep_len+=1

        # if done or goal reached, reset the environment and trackers, save model
        # else switch over the obs
        if done or ep_len>=args.steps_per_epoch:
            env.randomInit()
            obs = env.getLandmarkVector()
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

        # update model: sample the buffer, do the q and policy updates, update trackers
        sample = replay_buffer.sample()  # (s, a, r, g, s', g', done)
        qloss, policyloss = model.update(sample, step % args.target_update_freq == 0)
        total_qloss += qloss
        total_policyloss += policyloss

        if step>0 and step % args.steps_per_epoch == 0 and (step // args.steps_per_epoch) % args.test_freq == 0:
            test_reward = test(args, device, model)

    return model

def test(args, device, model=None):
    print('testing!')
    env = GraphTraverser(Graph(config_path=args.config_file))

    if model is None:
        img_backbone = ResnetBackbone(args, device)
        model = SimpleSACDisc(args, env.action_space, device)
        model.load_states(args)

    env.randomInit()
    obs = env.getLandmarkVector()
    done = False
    step = 0
    total_reward = 0
    actions = []
    locations = []
    start = [env.current_node, env.current_direction]
    goal = [env.goalNode, env.goalDirection]

    while not done and step < args.steps_per_epoch:
        action = model.get_action(np.stack([obs]), np.stack([env.getLandmarkVector('goal')]))
        obs_new, reward, goal_info, done = env.step(action)
        obs_new = env.getLandmarkVector()

        actions.append(action)
        locations.append([env.current_node, env.current_direction])
        total_reward+=reward
        step+=1

        obs = obs_new

    print('Start: Node', start[0], ',', start[1], 'degrees')
    print('Goal: Node', goal[0], ',', goal[1], 'degrees')
    print('Total Reward',total_reward)
    # print('Actions: 0=straight, 1=backward, 2=left, 3=right')
    # the actions are bad bc don't actually take action so how know how to update the angles
    optimal_solution = env.graph.findPath(start_node=start[0], end_node=goal[0], start_direction=start[1], end_direction=goal[1], base_angle=env.base_turn_angle)
    print('Optimal Solution:')
    print('path', optimal_solution[0])
    print('actions', optimal_solution[1])
    human_actions = []
    action_key = {0:'forward', 1:'left', 2:'right'}#3:'backward'
    for act in actions:
        human_actions.append(action_key[int(act)])
    print()
    print('Agent Actions:', list(zip(human_actions, ['to [node,dir]='] * len(locations), locations)))
    print()
    # action_diffs = []
    # i=0
    # for act in optimal_solution[1]:
    #     if i<len(human_actions):
    #         if human_actions[i] not in act:
    #             while i<len(human_actions) and human_actions[i] not in act:
    #                 action_diffs.append(human_actions[i])
    #                 i+=1
    #         # elif action_diffs[-1]!='same':
    #         else:
    #             action_diffs.append('same')
    # for act in human_actions[i:]:
    #     action_diffs.append(act)
    #     print()
    # print('Extraneous Actions:', action_diffs)
    time.sleep(10)
    return total_reward


if __name__ == '__main__':
    torch.manual_seed(525)
    np.random.seed(525)

    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.test:
        model = train(args, device)
    test(args, device)

