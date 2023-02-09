import random
import time
import json
import numpy as np

from glob import glob
from pynput import keyboard
from pynput.keyboard import Key
from PIL import Image
from matplotlib import pyplot as plt

class Graph():
    def __init__(self, config_path=None, config=None):
        if config is None or type(config)!=dict:
            if config_path is None:
                print('Graph object needs either a config json/dict object or the path to one')
                breakpoint()
            else:
                with open(config_path) as f:
                    self.config = json.load(f)
        else:
            self.config = config

        self.vertices = [None]*int(self.config['num_vertices'])
        self.populateVertices(self.config['num_vertices'], self.config['photoPath'])
        self.edges = np.zeros((self.config['num_vertices'],self.config['num_vertices']))
        self.makeEdgeTable(self.config['transitions'])
        # for [X,Y,theta,d], self.config['transitions'] has the vertices that are connected by edges [X,Y]
        # the directions (*approx* theta) you need to turn to get from X to Y assuming the wall opposite the door is
        # theta = 0/360. all angles are listed in degrees (not radians) and assuming the robot is facing theta = 0
        # and turning clockwise to get to Y
        # and the distance d (*approximate* but close enough) between the vertices in feet
        self.weights = [None]*int(self.config['num_vertices']) #TODO: is this just going to be the thing in the vertices list? or will this help with navigation/path planning?

    def populateVertices(self, num_vertices, assetPath):
        assets = glob(assetPath+'*')
        for i in range(num_vertices):
            if assetPath+'node'+str(i+1) in assets:
                self.vertices[i]=assetPath+'node'+str(i+1)+'/'

    def makeEdgeTable(self, transition_list):
        # TODO: should we put edges between the nodes and themselves? and choosing to stay means you're identifying you're in
        # the goal node?
        # assumes starts counting at Node 1 (no node 0)
        for transitions in transition_list:
            for transition in transitions:
                self.edges[transition[0]-1,transition[1]-1] = 1

    def getReachableVertices(self, current_node, current_direction, angle_threshold=15):
        return [x for x in self.config['transitions'][current_node-1] if abs(x[2]-current_direction)<angle_threshold]

    def getAllNeighbors(self, current_node):
        return [x for x in self.config['transitions'][current_node - 1]]

    def getVertexFeats(self, vertex_num, heading=None, heading_threshold=1):
        if heading is None:
            #TODO: lol this could fail so hard pls fix??? this function should return an image!?
            return self.vertices[vertex_num-1]
        else:
            #TODO: this is slow; maybe make a dictionary inside the json file with angle:image name
            with open(self.vertices[vertex_num-1]+'labels.json') as f:
                labels = json.load(f)
            labels.pop('train')
            labels.pop('val')
            labels.pop('test')
            for k,v in labels.items():
                if abs(v[-1]-heading)<heading_threshold:
                    return Image.open(self.vertices[vertex_num-1]+'node'+str(vertex_num)+'_'+str(k)+self.config['photoExtension'])

    def addNode(self, nodeFeatures=None, nodeFeatPath=None):
        if nodeFeatures is None:
            if nodeFeatPath is None:
                print('addNode function in Graph object needs either nodeFeatures or nodeFeatPath defined')
                breakpoint()
            else:
                # TODO: load the node features? assuming it's a video? or sequence of frames?
                pass
        else:
            # TODO: add to vertices list (the end for simplicity)
            # TODO: add to edges table (the end for simplicity)
            edgesToAdd= None
            for edges in edgesToAdd:
                self.addEdge(edges[0], edges[1])

    def addEdge(self, node1Num, node2Num):
        # making the executive decision that this is going to be a two-way graph
        # assumes starts counting at Node 1 (no node 0)
        self.edges[node1Num-1, node2Num-1] = 1
        self.edges[node2Num-1, node1Num-1] = 1

    def removeEdge(self, node1Num, node2Num):
        # assumes starts counting at Node 1 (no node 0)
        self.edges[node1Num-1, node2Num-1] = 0
        self.edges[node2Num-1, node1Num-1] = 0

    def removeNode(self, node_num):
        print("remove node function not implemented!!!")
        pass

    def findPath(self, start_node, end_node):
        print('find path function not implemented!!!')
        pass

class GraphTraverser():
    def __init__(self, graph: Graph, start_node=1, base_turn_angle=15, plotImgs=False, recordActions=False, human=False):
        self.graph = graph
        self.start_node = start_node
        self.base_turn_angle = base_turn_angle
        self.plotImgs = plotImgs
        self.recordActions = recordActions
        self.action_space = 4
        self.human = human #whether the traversal is by a human (true) or an RL agent (false)
        self.reset()

    def reset(self):
        self.current_node = self.start_node
        self.current_direction = 180
        self.setGoalNode()
        if self.recordActions:
            self.path = [self.current_node]
            self.actions = []

    def randomInit(self):
        self.current_node = random.choice(range(1,len(self.graph.vertices)+1))
        self.current_direction = random.choice(range(0,360,5))
        self.setGoalNode()
        if self.recordActions:
            self.path = [self.current_node]
            self.actions = []

    def setGoalNode(self, goal=None, direction=None):
        if goal is None:
            first_level_reachable = self.graph.getAllNeighbors(self.current_node)
            second_level_reachable = []
            for node_info in first_level_reachable:
                second_level_reachable.extend(self.graph.getAllNeighbors(node_info[1]))
            choice = random.choice(second_level_reachable)
            self.goalNode = choice[1]
            #TODO: broaden this range
            self.goalDirection = choice[2] #random.choice(range(360))
        else:
            self.goalNode = goal
            self.goalDirection=direction

    def step(self, action):
        # returns reward and observation
        # action is [straight, backward, left, right]
        goal = self.goalNode
        goal_dir = self.goalDirection
        start = self.current_node
        start_dir = self.current_direction
        if action == 0:
            reward = self.moveForward()
            reward = self.checkGoal(reward)
        elif action == 1:
            reward = self.moveBack()
            reward = self.checkGoal(reward)
        elif action == 2:
            reward = self.turnLeft()
            reward = self.checkGoal(reward)
        elif action == 3:
            reward = self.turnRight()
            reward = self.checkGoal(reward)
        else:
            print(action,"is not a valid action!")
            breakpoint()

        image = self.getImg()
        return image, reward, (start, start_dir, goal, goal_dir), reward==1

    def checkGoal(self, reward):
        if self.current_node == self.goalNode and abs(self.current_direction-self.goalDirection)<self.base_turn_angle:
            return 1
        return reward/10

    def on_key_release(self, key):
        #https://pynput.readthedocs.io/en/latest/keyboard.html#monitoring-the-keyboard
        if key == Key.right:
            print('turning right')
            self.turnRight()
        elif key == Key.left:
            print('turning left')
            self.turnLeft()
        elif key == Key.up:
            print('moving forward')
            self.moveForward()
        elif key == Key.down:
            print('moving backward')
            self.moveBack()
        elif key.char == 'q':
            print("Exiting Graph Traversal")
            return False

    def turnRight(self, angle=None):
        if angle is None:
            turn_angle = self.base_turn_angle
        else:
            turn_angle = angle
        self.current_direction += turn_angle
        self.current_direction = self.current_direction % 360
        if self.recordActions:
            # update rule is start node, end node, amount moved forward, angle turned
            self.actions.append([self.current_node, self.current_node, 'r', 0, turn_angle])
        if self.plotImgs:
            self.render()
        return -1

    def turnLeft(self, angle=None):
        if angle is None:
            turn_angle = self.base_turn_angle
        else:
            turn_angle = angle
        self.current_direction -= turn_angle
        self.current_direction = (self.current_direction + 360) % 360
        if self.recordActions:
            # update rule is start node, end node, action, amount moved forward, angle turned
            self.actions.append([self.current_node, self.current_node, 'l', 0, -turn_angle])
        if self.plotImgs:
            self.render()
        return -1

    def moveForward(self):
        nodeOptions = self.graph.getReachableVertices(self.current_node, self.current_direction)
        if len(nodeOptions)==0:
            if self.recordActions:
                # update rule is start node, end node, amount moved forward, angle turned
                self.actions.append([self.current_node, self.current_node, 'f', 0, 0])
            if self.human:
                print('Cant move forward!')
            # return -10
        else:
            if len(nodeOptions)==1:
                newNode = nodeOptions[0]
            else:
                angles = [abs(x[2]-self.current_direction) for x in nodeOptions]
                ang_ind = np.argmin(angles)
                newNode = nodeOptions[ang_ind]

            if self.recordActions:
                # update rule is start node, end node, amount moved forward, angle turned
                self.actions.append([self.current_node, newNode[1], 'f', newNode[3], 0])
                self.path.append(newNode[1])
            self.current_node = newNode[1]
        if self.plotImgs:
            self.render()
        return -1

    def moveBack(self):
        backwardsDirection = (self.current_direction+180) % 360
        nodeOptions = self.graph.getReachableVertices(self.current_node, backwardsDirection)
        if len(nodeOptions) == 0:
            if self.recordActions:
                # update rule is start node, end node, amount moved forward, angle turned
                self.actions.append([self.current_node, self.current_node, 'b', 0, 0])
            if self.human:
                print('Cant move backward!')
            # return -10
        else:
            if len(nodeOptions) == 1:
                newNode = nodeOptions[0]
            else:
                angles = [abs(x[2] - backwardsDirection) for x in nodeOptions]
                ang_ind = np.argmin(angles)
                newNode = nodeOptions[ang_ind]

            if self.recordActions:
                # update rule is start node, end node, amount moved forward, angle turned
                self.actions.append([self.current_node, newNode[1], 'b', -newNode[3], 0])
                self.path.append(newNode[1])
            self.current_node = newNode[1]
        if self.plotImgs:
            self.render()
        return -1

    def getImg(self, node=None, direction=None):
        if node is None:
            node = self.current_node
        if direction is None:
            direction = self.current_direction

        image = self.graph.getVertexFeats(node, direction)
        while image is None:
            print("Warning! no image for node",node,"in direction",direction)
            direction+=1
            print("Using direction", direction, "instead!")
            image = self.graph.getVertexFeats(node, direction)
            # breakpoint()
        return image

    def render(self):
        #TODO: keep track of the figure so the slowdown stops happening
        image = self.getImg()
        plt.imshow(image)
        plt.title('Node: '+str(self.current_node)+', Heading: '+str(self.current_direction))
        plt.pause(0.01)

    def traverse(self):
        prompt_text = 'Use the arrow keys to look/turn left/right and move up/down\n' \
                      'Enter the "q" key to quit\n'
        print(prompt_text)
        if self.plotImgs:
            self.render()
        with keyboard.Listener(on_release=self.on_key_release) as listener:
            listener.join()
        if self.plotImgs:
            plt.close(1)

    def save(self):
        with open(self.graph.config['envName']+'Trials.json','a') as f:
            try:
                trials = json.load(f)
                keys = list(trials.keys())
                new_key = int(keys[-1]) + 1
            except:
                trials = {}
                new_key = 0

            trials[new_key] = {"actions":self.actions,
                               "path":self.path}
            json.dump(trials, f)


if __name__ == '__main__':
    graph = Graph(config_path='labGraphConfig.json')
    # breakpoint()
    traverser = GraphTraverser(graph, plotImgs=True, recordActions=True)
    traverser.setGoalNode()
    traverser.traverse()
    time.sleep(2)
    traverser.traverse()
    traverser.save()
    traverser.reset()