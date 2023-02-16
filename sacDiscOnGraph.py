import random
import torch
import math
import argparse
import numpy as np

from torchvision.models import ResNet18_Weights

from models import SACDiscreteBaseline, ResnetBackbone, GoalReplayBuffer, Encoder32
from graph import GraphTraverser, Graph

def getArgs():
    parser=argparse.ArgumentParser()

    #file parameters
    parser.add_argument('--config_file', type=str, default='labGraphConfig.json', help='path to the graph config file')

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=50000, help='number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default = 20, help='max number of steps for each epoch')
    parser.add_argument('--use_policy_step', type=int, default=200, help='number of steps before using the learned policy')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')
    parser.add_argument('--save_name', default='sacDiscOnGraph', help='prefix name for saving the SAC networks')
    parser.add_argument('--target_update_freq', type=int, default=20, help='max number of samples in the replay buffer')
    parser.add_argument('--dist_reward', action='store_true', help='use the distance reward instead')
    parser.add_argument('--test', action='store_true', help='skip training and just test')
    parser.add_argument('--epsilon', type=float, default=0.3, help='learning rate for training')



    # buffer hyperparameters
    parser.add_argument('--buffer_limit', type=int, default=40000, help='max number of samples in the replay buffer')
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
    # initialize the environment and get the first observation
    env = GraphTraverser(Graph(config_path=args.config_file), distance_reward=args.dist_reward)
    env.randomInit()
    obs = env.getImg()

    #TODO: need to make SAC goal conditioned when getting actions!

    # initialize and populate the replay buffer
    replay_buffer = GoalReplayBuffer(args)
    for step in range(args.buffer_init_steps):
        action = random.choice(range(env.action_space))
        start = env.current_node
        obs_new, reward, goal_info, done = env.step(action)
        # transforming action to store it as a vector
        action_ind = action
        action = np.zeros(env.action_space)
        action[action_ind] = 1
        end = env.current_node
        # goal info is currently (start node,  star dir, goal node, goal dir)
        # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
        goal_img = env.getGoalImg()
        # sample of the shape (s, a, r, g, s', done)
        replay_buffer.addSample([obs, action, reward, goal_img, obs_new, done])
        # artificially populating the buffer with more samples of the actions taken less often (forward/backward)
        if action_ind in [0,1] and end!=start:
            # putting more copies of the same action into the buffer
            for _ in range(3):
                replay_buffer.addSample([obs, action, reward, goal_img, obs_new, done])
            # if action is forward, reverse the obs and add as a backwards sample (and vice versa)
            # but not if dist reward=True (bc how compute reward in both cases)
            # but can do if dist reward = False
            if not args.dist_reward:
                # breakpoint()
                if action_ind==0:
                    action[0]=0
                    action[1]=1
                    if done:
                        done = False
                        reward = -1
                elif action_ind==1:
                    action[0] = 1
                    action[1] = 0
                    if done:
                        done = False
                        reward = -1
                for _ in range(4):
                    replay_buffer.addSample([obs_new, action, reward, goal_img, obs, done])
        # implementing hindsight experience replay
        # replay_buffer.addHERSample([obs, action, reward, goal_img, obs_new, done], args.max_reward)
        if reward == 1 or step % args.steps_per_epoch==0:
            env.randomInit()
            obs = env.getImg()
        else:
            obs = obs_new

    # initialize the model
    img_backbone = ResnetBackbone(args, device)
    # transforms = ResNet18_Weights.DEFAULT.transforms()
    # img_backbone = Encoder32(args.hidden_dim, transforms, device)
    # img_backbone.load_state_dict(torch.load('allNodePhotoAE_'+str(args.hidden_dim)+'hid_encoder.pt', map_location=torch.device('cpu')))
    # img_backbone = img_backbone.to(device)
    # img_backbone.eval()
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
        rand_num = random.random()
        # last part is epsilon greedy
        if step < args.use_policy_step or rand_num < math.exp(-1. * step / 900):
            action = random.choice(range(env.action_space))
        else:
            goal_img = env.getGoalImg()
            action = model.get_action(np.stack([obs]), np.stack([goal_img]))

        # take action in the environment and save to replay/update trackers
        # if step > args.use_policy_step:
        #     print('pre state+goal', (env.current_node, env.current_direction), (env.goalNode, env.goalDirection))
        obs_new, reward, goal_info, done = env.step(action)
        # storing the action as a vector
        action_ind = action
        action = np.zeros(env.action_space)
        action[action_ind] = 1
        # goal info is currently (start node,  star dir, goal node, goal dir)
        # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
        goal_img = env.getGoalImg()
        # sample of the shape (s, a, r, g, s', done)
        replay_buffer.addSample([obs, action, reward, goal_img, obs_new, done])
        if action_ind in [0,1] and end!=start:
            # putting more copies of the same action into the buffer
            for _ in range(9):
                replay_buffer.addSample([obs, action, reward, goal_img, obs_new, done])
            # if action is forward, reverse the obs and add as a backwards sample (and vice versa)
            # but not if dist reward=True (bc how compute reward in both cases)
            # but can do if dist reward = False
            if not args.dist_reward:
                # breakpoint()
                if action_ind==0:
                    action[0]=0
                    action[1]=1
                    if done:
                        done = False
                        reward = -1
                elif action_ind==1:
                    action[0] = 1
                    action[1] = 0
                    if done:
                        done = False
                        reward = -1
                for _ in range(10):
                    replay_buffer.addSample([obs_new, action, reward, goal_img, obs, done])
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

        # update model: sample the buffer, do the q and policy updates, update trackers
        sample = replay_buffer.sample()  # (s, a, r, g, s', done)
        qloss, policyloss = model.update(sample, step % args.target_update_freq == 0)
        total_qloss += qloss
        total_policyloss += policyloss

    return model

def test(args, device, model=None):
    print('testing!')
    env = GraphTraverser(Graph(config_path=args.config_file))

    if model is None:
        img_backbone = ResnetBackbone(args, device)
        model = SACDiscreteBaseline(args, img_backbone, env.action_space, device)
        model.load_states(args)

    env.randomInit()
    obs = env.getImg()
    done = False
    step = 0
    total_reward = 0
    actions = []
    locations = []
    start = [env.current_node, env.current_direction]
    goal = [env.goalNode, env.goalDirection]

    goal_img = env.getImg()
    while not done and step < args.steps_per_epoch:
        action = model.get_action(np.stack([obs]), np.stack([goal_img]))
        obs_new, reward, goal_info, done = env.step(action)

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
    action_key = {0:'forward', 1:'backward', 2:'left', 3:'right'}
    for act in actions:
        human_actions.append(action_key[int(act)])
    print()
    print('Agent Actions:', list(zip(human_actions, ['to [node,dir]='] * len(locations), locations)))
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



if __name__ == '__main__':
    torch.manual_seed(525)
    np.random.seed(525)

    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.test:
        model = train(args, device)
    test(args, device)#, model)

