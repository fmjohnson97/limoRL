import random
import torch
import math
import argparse
import numpy as np
import time

import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights

from dqn_models import Memory, DuelingDQN
from models import GoalReadImagesBuffer, ResnetBackbone
from graph import GraphTraverser, Graph

def getArgs():
    parser=argparse.ArgumentParser()

    #file parameters
    parser.add_argument('--config_file', type=str, default='labGraphConfig5.json', help='path to the graph config file')

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples used to update the networks at once')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs for training')
    parser.add_argument('--steps_per_epoch', type=int, default = 50, help='max number of steps for each epoch')
    parser.add_argument('--use_policy_step', type=int, default=200, help='number of steps before using the learned policy')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--save_name', default='ddqnDiscOnGraph_', help='prefix name for saving the SAC networks')
    parser.add_argument('--target_update_freq', type=int, default=4, help='max number of samples in the replay buffer')
    parser.add_argument('--dist_reward', action='store_true', help='use the distance reward instead')
    parser.add_argument('--test', action='store_true', help='skip training and just test')
    parser.add_argument('--test_freq', type=int, default=10, help='number of epochs between testing')
    parser.add_argument('--init_epsilon', type=float, default=0.3, help='initial epsilon value')
    parser.add_argument('--final_epsilon', type=float, default=0.0001, help='final epsilon value')
    parser.add_argument('--explore', type=int, default=20000, help='number of epochs for training')
    parser.add_argument('--feat_dim', type=int, default=512 * 7 * 7*2, help='number of epochs for training')
    parser.add_argument('--goal_node', type=int, default=None, help='number of epochs for training')
    parser.add_argument('--goal_dir', type=int, default=None, help='number of epochs for training')
    parser.add_argument('--img_paths', action='store_true', help='skip training and just test')


    # buffer hyperparameters
    parser.add_argument('--buffer_limit', type=int, default=50000, help='max number of samples in the replay buffer')
    parser.add_argument('--buffer_init_steps', type=int, default=4000, help='number of random actions to take before train loop')
    parser.add_argument('--max_reward', type=float, default=1, help='max reward achievable by agent when reaches goal')

    # sac hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for SAC RL')

    # resnet/backbone args
    parser.add_argument('--return_node', type=str, default='layer4', choices=['layer1','layer2','layer3','layer4'], help='resnet layer to return features from')

    return parser.parse_args()


def train(args, device):
    # initialize the environment and get the first observation
    env = GraphTraverser(Graph(config_path=args.config_file, img_paths=args.img_paths), distance_reward=args.dist_reward)
    env.randomInit()
    if args.goal_node is not None and args.goal_dir is not None:
        env.goalNode=args.goal_node
        env.goalDirection=args.goal_dir
    # print('Goal is', env.goalNode, env.goalDirection)
    obs = env.getImg()

    # initialize and populate the replay buffer
    replay_buffer = GoalReadImagesBuffer(args)
    step = 0
    while step < args.buffer_init_steps:
        # breakpoint()
        action = random.choice(range(env.action_space))
        # goal info is currently (start node,  star dir, goal node, goal dir)
        # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
        goal_img = env.getGoalImg()
        goal_vec = None #env.getGoalVector()
        obs_new, reward, goal_info, done = env.step(action)
        # transforming action to store it as a vector
        # action_ind = action
        # action = np.zeros(env.action_space)
        # action[action_ind] = 1
        goal_img_new = env.getGoalImg()
        goal_vec_new = None#env.getGoalVector()
        # sample of the shape (s, a, r, g, s', g', done)
        # if action_ind == 0:
        #     if (obs == obs_new).all():
        #         rand_int = random.random()
        #         # if rand_int > .8:
        #         #     step += 4
        #         #     for i in range(4):
        #         #         replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new, goal_vec_new), done])
        #     else:
        #         step += 2
        #         for i in range(2):
        #             replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new, goal_vec_new), done])
        # else:
        step += 1
        replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new, goal_vec_new), done])
        # implementing hindsight experience replay
        # replay_buffer.addHERSample([obs, action, reward, goal_img, obs_new, done], args.max_reward)
        if done or reward >= 1 or (step>0 and step % args.steps_per_epoch==0):
            env.randomInit()
            # print('Goal is', env.goalNode, env.goalDirection)
            obs = env.getImg()
            # print('Goal is', env.goalNode, env.goalDirection)
        else:
            obs = obs_new

    # breakpoint()
    # initialize the model
    img_backbone = ResnetBackbone(args, device)
    # transforms = ResNet18_Weights.DEFAULT.transforms()
    # img_backbone = Encoder32(args.hidden_dim, transforms, device)
    # img_backbone.load_state_dict(torch.load('allNodePhotoAE_'+str(args.hidden_dim)+'hid_encoder.pt', map_location=torch.device('cpu')))
    # img_backbone = img_backbone.to(device)
    # img_backbone.eval()
    model = DuelingDQN(args, device, img_backbone)
    model.train()

    # start training
    ep_count = 0
    ep_reward = 0
    ep_len = 0
    total_qloss = 0
    total_policyloss = 0
    all_total_rewards = []
    env.randomInit()
    # print('Goal is', env.goalNode, env.goalDirection)
    obs = env.getImg()
    for step in range(args.epochs*args.steps_per_epoch):
        # initial random action or use the learned policy
        rand_num = random.random()
        # last part is epsilon greedy
        if step < args.use_policy_step or rand_num < math.exp(-1. * step / 800):
            action = random.choice(range(env.action_space))
            # goal info is currently (start node,  star dir, goal node, goal dir)
            # getting the goal images to save instead of the goal info (since that's really for debug purposes tbh)
            goal_img = env.getGoalImg()
            goal_vec = None#env.getGoalVector()
        else:
            goal_img = env.getGoalImg()
            goal_vec = None#env.getGoalVector()
            action = model.select_action(np.stack([obs]), np.stack([goal_img]))#, np.stack([goal_vec]))

        # take action in the environment and save to replay/update trackers
        # if step > args.use_policy_step:
        #     print('pre state+goal', (env.current_node, env.current_direction), (env.goalNode, env.goalDirection))
        obs_new, reward, goal_info, done = env.step(action)
        # storing the action as a vector
        # action_ind = action
        # action = np.zeros(env.action_space)
        # action[action_ind] = 1
        goal_img_new = env.getGoalImg()
        goal_vec_new = None#env.getGoalVector()
        # sample of the shape (s, a, r, g, s', g', done)
        # if action_ind == 0:
        #     if (obs == obs_new).all():
        #         rand_int = random.random()
        #         # if rand_int > .9:
        #         #     replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new, goal_vec_new), done])
        #     else:
        #         for i in range(1):
        #             replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new, goal_vec_new), done])
        # else:
        replay_buffer.addSample([obs, action, reward, (goal_img, goal_vec), obs_new, (goal_img_new, goal_vec_new), done])
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
            # print('Goal is', env.goalNode, env.goalDirection)
            obs = env.getImg()
            # print('Goal is', env.goalNode, env.goalDirection)
            print('Epoch',ep_count,'completed in',ep_len,'steps with reward =',ep_reward)
            print('\t Total Q Loss:',total_qloss,' Avg Q Loss:',total_qloss/ep_len)
            print('\t Total Policy Loss:',total_policyloss,' Avg Policy Loss:',total_policyloss/ep_len)
            print()
            all_total_rewards.append(ep_reward)
            if len(all_total_rewards)>5 and step>args.use_policy_step and np.mean(all_total_rewards[-5:]) > -11/10:
                print('Step:',step)
                temp = [a[1] for a in replay_buffer.buffer]
                print(len([a for a in temp if a == 0]) / len(temp), 'is the forward percentage')
                print(len([a for a in temp if a == 1]) / len(temp), 'is the left percentage')
                count = 0
                temp=0
                while count<5 and temp>-2:
                    temp = test(args, device, model, env.goalNode, env.goalDirection)
                    count+=1
                if count>=5:
                    print('Training Done!')
                    exit(0)

            ep_len=0
            ep_reward=0
            total_qloss = 0
            total_policyloss = 0
            ep_count+=1
            # TODO: should we relabel the buffer?
            model.save_models(args.save_name, env.goalNode, env.goalDirection)
        else:
            obs = obs_new

        # update model: sample the buffer, do the q and policy updates, update trackers
        sample = replay_buffer.sample()  # (s, a, r, g, s', g', done)
        qloss= model.update(sample, step)
        total_qloss += qloss
        total_policyloss += 0

        if step>0 and step % args.steps_per_epoch == 0 and (step // args.steps_per_epoch) % args.test_freq == 0:
            print('Step:', step)
            temp = [a[1] for a in replay_buffer.buffer]
            print(len([a for a in temp if a == 0]) / len(temp), 'is the forward percentage')
            print(len([a for a in temp if a == 1]) / len(temp), 'is the left percentage')
            test_reward = test(args, device, model, env.goalNode, env.goalDirection)

    return model

def test(args, device, model=None, goalNode=None, goalDirection=None):
    print('testing!')
    env = GraphTraverser(Graph(config_path=args.config_file, img_paths=args.img_paths), distance_reward=args.dist_reward)
    if goalNode is not None:
        env.goalNode = goalNode
    if goalDirection is not None:
        env.goalDirection = goalDirection
    if model is None:
        img_backbone = ResnetBackbone(args, device)
        model = DuelingDQN(args, device, img_backbone)
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
        action = model.select_action(np.stack([obs]), np.stack([goal_img]))
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
    try:
        optimal_solution = env.graph.findPath(start_node=start[0], end_node=goal[0], start_direction=start[1], end_direction=goal[1], base_angle=env.base_turn_angle)
    except:
        print('Optimal solution timed out')
        optimal_solution=[None,None]
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
    time.sleep(5)
    return total_reward


if __name__ == '__main__':
    torch.manual_seed(525)
    np.random.seed(525)

    args = getArgs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.test:
        model = train(args, device)
    test(args, device)#, model)

