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
    def __init__(self, graph: Graph, start_node=1, base_turn_angle=15, plotImgs=False):
        self.graph = graph
        self.start_node = start_node
        self.base_turn_angle = base_turn_angle
        self.plotImgs = plotImgs
        self.reset()

    def reset(self):
        self.current_node = self.start_node
        self.current_direction = 180
        self.path = [self.current_node]
        self.actions = []

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
        # update rule is start node, end node, amount moved forward, angle turned
        self.actions.append([self.current_node, self.current_node, 'r', 0, turn_angle])
        if self.plotImgs:
            self.render()

    def turnLeft(self, angle=None):
        if angle is None:
            turn_angle = self.base_turn_angle
        else:
            turn_angle = angle
        self.current_direction -= turn_angle
        self.current_direction = (self.current_direction + 360) % 360
        # update rule is start node, end node, action, amount moved forward, angle turned
        self.actions.append([self.current_node, self.current_node, 'l', 0, -turn_angle])
        if self.plotImgs:
            self.render()

    def moveForward(self):
        nodeOptions = graph.getReachableVertices(self.current_node, self.current_direction)
        if len(nodeOptions)==0:
            # update rule is start node, end node, amount moved forward, angle turned
            self.actions.append([self.current_node, self.current_node, 'f', 0, 0])
            print('Cant move forward!')
        else:
            if len(nodeOptions)==1:
                newNode = nodeOptions[0]
            else:
                angles = [abs(x[2]-self.current_direction) for x in nodeOptions]
                ang_ind = np.argmin(angles)
                newNode = nodeOptions[ang_ind]

            # update rule is start node, end node, amount moved forward, angle turned
            self.actions.append([self.current_node, newNode[1], 'f', newNode[3], 0])
            self.path.append(newNode[1])
            self.current_node = newNode[1]
        if self.plotImgs:
            self.render()

    def moveBack(self):
        backwardsDirection = (self.current_direction+180) % 360
        nodeOptions = graph.getReachableVertices(self.current_node, backwardsDirection)
        if len(nodeOptions) == 0:
            # update rule is start node, end node, amount moved forward, angle turned
            self.actions.append([self.current_node, self.current_node, 'b', 0, 0])
            print('Cant move backward!')
        else:
            if len(nodeOptions) == 1:
                newNode = nodeOptions[0]
            else:
                angles = [abs(x[2] - backwardsDirection) for x in nodeOptions]
                ang_ind = np.argmin(angles)
                newNode = nodeOptions[ang_ind]

            # update rule is start node, end node, amount moved forward, angle turned
            self.actions.append([self.current_node, newNode[1], 'b', -newNode[3], 0])
            self.path.append(newNode[1])
            self.current_node = newNode[1]
        if self.plotImgs:
            self.render()

    def getImg(self,node=None, direction=None):
        if node is None:
            node = self.current_node
        if direction is None:
            direction = self.current_direction

        return self.graph.getVertexFeats(node, direction)

    def render(self):
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
        breakpoint()
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
    traverser = GraphTraverser(graph, plotImgs=True)
    traverser.traverse()
    time.sleep(2)
    traverser.traverse()
    traverser.save()
    traverser.reset()