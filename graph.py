import numpy as np
import json

from glob import glob

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
        self.populateVertices(self.config['num_vertices'], self.config['videoPath'], self.config['videoExtension'])
        self.edges = np.zeros((self.config['num_vertices'],self.config['num_vertices']))
        self.makeEdgeTable(self.config['transitions'])
        # for [X,Y,theta,d], self.config['transitions'] has the vertices that are connected by edges [X,Y]
        # the directions (*approx* theta) you need to turn to get from X to Y assuming the wall opposite the door is
        # theta = 0/360. all angles are listed in degrees (not radians) and assuming the robot is facing theta = 0
        # and turning clockwise to get to Y
        # and the distance d (*approximate* but close enough) between the vertices in feet
        self.weights = [None]*int(self.config['num_vertices']) #TODO: is this just going to be the thing in the vertices list? or will this help with navigation/path planning?

    def populateVertices(self, num_vertices, videoPath, videoExtension):
        videos = glob(videoPath+'*')
        for i in range(num_vertices):
            if videoPath+'node'+str(i+1)+videoExtension in videos:
                self.vertices[i]=videoPath+'node'+str(i+1)+videoExtension

    def makeEdgeTable(self, transition_list):
        # TODO: should we put edges between the nodes and themselves? and choosing to stay means you're identifying you're in
        # the goal node?
        # assumes starts counting at Node 1 (no node 0)
        for transitions in transition_list:
            for transition in transitions:
                self.edges[transition[0]-1,transition[1]-1] = 1

    def getReachableVertices(self, current_node):
        return [x[1] for x in self.config['transitions'][current_node-1]]

    def addNode(self, nodeFeatures=None, nodeFeatPath=None):
        if nodeFeatures is None:
            if nodeFeatPath is None:
                print('addNode function in Graph object needs either nodeFeatures or nodeFeatPath defined')
                breakpoint()
            else:
                # TODO: load the node features? assuming it's a video? or sequence of frames?
                pass
        else:
            # TODO: add to vertices list
            # TODO: add to edges table
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

if __name__ == '__main__':
    graph = Graph(config_path='config.json')
    breakpoint()