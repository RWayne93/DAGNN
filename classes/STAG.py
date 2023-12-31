
from random import random as R, SystemRandom as SystemRandom
from random import seed
import json
import os
from math import tanh, log
import matplotlib.pyplot as plt
import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout


# Sorted Topological Acyclical Graph
class STAG:
    def __init__(self, input_size, hidden_size, output_size, seed_value=None):
        self.A = tuple(['A' + str(i).zfill(len(str(input_size - 1))) for i in range(input_size)]) # input node ids
        self.B = tuple(['B' + str(i).zfill(len(str(hidden_size - 1))) for i in range(hidden_size)]) # hidden node ids
        self.C = tuple(['C' + str(i).zfill(len(str(output_size - 1))) for i in range(output_size)]) # output node ids
        self.E = 0 # error
        self.F = 0 # fitness / score on unit-tests
        self.G = 9999 # max generations
        self.I = [] # unit tests / input values
        self.L = [] # list of links
        self.N = {} # node data / network
        self.O = {} # output nodes
        self.P = [] # unit tests / predicted values of output nodes
        self.Q = {} # links selected for random walk backward
        self.R = 0.005 # learning rate
        self.S = 0 # size / length of self.L
        self.T = 1.5 # threshold of error
        self.U = {} # unit tests
        self.W = (-1, 1) # initial domain of random weights
        self.X = None # previous links
        self.Y = None # previous network
        self.Z = True # prune while True
        self.seed_value = seed_value

        if seed_value is not None:
            seed(seed_value)
        
        if seed_value is None:
            self.seed_value = SystemRandom().randint(0, 2**32 - 1)
        
        seed(self.seed_value)

        D = self.B + self.C
        for node_1 in self.A + self.B:
            # assign input and hidden node ids to node data
            self.N |= {node_1: {'value': 0, 'links': {}}}

            # create link with random weight between input and hidden nodes to other hidden nodes and output nodes
            j = 0
            while j < len(D):
                while D[j] <= node_1: 
                    j += 1
                node_2 = D[j]
                self.L += [(node_1, node_2)]
                # return random weight between values of self.W
                self.N[node_1]['links'][node_2] = R() * (self.W[1] - self.W[0]) + self.W[0]
                j += 1

        # assign output node ids to node data
        for node in self.C: 
            self.N |= {node: {'value': 0}}
        self.L = tuple(self.L)
        self.S = len(self.L)


    # update links and network size
    def UpdateLinks(self):
        self.L = []
        nodes = [nodes for nodes in self.N if nodes[0] != 'C']
        for node_1 in nodes:
            for node_2 in self.N[node_1]['links']:
                self.L += [(node_1, node_2)]
        (self.L).sort(key = lambda x: x[1])
        (self.L).sort(key = lambda x: x[0])
        self.L = tuple(self.L)
        self.S = len(self.L)

    # propogate input node values through the network to compute output node values
    def Forward(self):

        # node activation function
        def Activate(x):
            return x * tanh(x) / 2

        # set input nodes to values assigned by unit test
        for node in self.A: 
            self.N[node]['value'] = self.I[node]
        # set hidden and output nodes to zero
        for node in self.B + self.C: 
            self.N[node]['value'] = 0

        last_activated = None
        for (node_1, node_2) in self.L:
            # activate each hidden node once
            if node_1[0] != 'A' and node_1 != last_activated:
                self.N[node_1]['value'] = Activate(self.N[node_1]['value'])
                last_activated = node_1
            # sum product of node values and weights
            self.N[node_2]['value'] += self.N[node_1]['value'] * self.N[node_1]['links'][node_2]

        # activate output nodes
        for node in self.C: 
            self.N[node]['value'] = Activate(self.N[node]['value'])

        # pass output nodes to self.O
        for node in self.C: 
            self.O |= {node: self.N[node]['value']}

        for node in self.P:
            # compound error between output nodes and expected outputs from unit test
            z = abs(self.O[node] - self.P[node])
            self.E += z * log(z + 1)

            # compound fitness
            self.F += z < 0.5

    # compute error from passing test through network
    def Test(self):
        self.E, self.F = 0, 0
        for inputs, outputs in self.U:
            self.I = inputs
            self.P = outputs
            self.Forward()

    # machine learning for network
    def Learn(self, minimize_error=False):

        initial_R = self.R + 0

        gen = 0
        self.E, self.F = 0, 0
        self.Test()

        again = self.F < len(self.U)
        if minimize_error:
            print('\n :: minimizing error ::\n')
            again = self.F == len(self.U) and self.E > self.T
        else:
            print('\n :: learning ::\n')

        # stop if max generations exceeded or fitness is 100% (if minimizing error: if max generations exceeded or fitness is not 100%)
        while gen < self.G and again:

            # get current error
            current_error = self.E + 0

            # adjust learning rate for select random links
            self.Q = {}
            while len(self.Q) < (self.S ** .4) // 1:
                self.Q |= {self.L[int(len(self.L) * R())]: 0}
            for Q in self.Q:
                node_1, node_2 = Q

                # temporarily adjust Q-link by learning rate
                self.N[node_1]['links'][node_2] += self.R

                #if self.N[node_1]['links'][node_2] > 0:
                    # if testing weight is positive, set Q to 0
                    #self.Q[Q] = 0
                #else:
                    # compute error difference after applying the learning rate
                self.Test()
                self.Q[Q] = (current_error - self.E) / self.R

                # remove learning rate from Q-link
                self.N[node_1]['links'][node_2] -= self.R

            # Sort Q links by magnitude of error difference
            self.Q = [(key, val) for (key, val) in self.Q.items()]
            (self.Q).sort(key = lambda x: abs(x[1]), reverse=True)

            # update link in network with largest error gradient by learning rate
            (node_1, node_2), err = self.Q[0]
            self.N[node_1]['links'][node_2] += self.R * [1, -1][err < 0]

            # print every 100th generation
            if gen % 100 == 0: 
                print(f'gen: {gen}    error: {self.E:0.4f}    score: {self.F} / {len(self.U)}')
                self.Visualize()
            gen += 1

            # test for next generation
            self.Test()

            if self.E < current_error:
                self.R *= 1.1
            else:
                if self.R > initial_R:
                    self.R = initial_R + 0
                else:
                    self.R /= 1.1

            again = self.F < len(self.U)
            if minimize_error:
                again = self.F == len(self.U)
                if again:
                    again &= self.E > self.T
                    # save links and network
                    self.X = str(self.L)
                    self.Y = str(self.N)

        self.R = initial_R + 0

        # stop pruning if unsuccessful
        if gen == self.G:
            print(f'gen: {gen}    error: {self.E:0.4f}    score: {self.F} / {len(self.U)}')
            print('\n :: max generations reached ::\n')
            #again = input('Continue Learning (<Enter> for Yes)?: ')
            if again == '':
                self.Learn(minimize_error)
            else:
                self.Z = False
                if minimize_error:
                    print('\n :: minimizing complete ::\n')
                else:
                    print('\n :: current failed end-state ::\n')
                    print(self.N)

        else:
            print(f'gen: {gen}    error: {self.E:0.4f}    score: {self.F} / {len(self.U)}')
            if not minimize_error:
                # save links and network
                self.X = str(self.L)
                self.Y = str(self.N)
                print('\n :: learning complete ::\n')
            print(self.N)



    # identifies and eliminates redundant nodes
    def Prune(self):

        self.G //= 10
        while self.Z:

            print('\n :: pruning ::')

            # save current links and network
            self.X = str(self.L)
            self.Y = str(self.N)

            # select smallest link
            j = 0
            link = self.L[j]
            node_1, node_2 = link
            weight = abs(self.N[node_1]['links'][node_2])
            smallest_link = [link, weight]
            for i in range(1, len(self.L)):
                link = self.L[i]
                node_1, node_2 = link
                weight = abs(self.N[node_1]['links'][node_2])
                if weight < smallest_link[1]:
                    j = i + 0
                    smallest_link = [link, weight]

            # delete smallest link
            node_1, node_2 = smallest_link[0]
            print(f'\nremoving link: ({node_1},{node_2})')
            self.N[node_1]['links'].pop(node_2)
            self.L = self.L[:j] + self.L[j+1:]

            # re-evaluate
            if self.Z:
                self.Learn()

        if self.X is not None:
            # restore previous links and network
            self.L = eval(self.X)
            self.N = eval(self.Y)
            self.G *= 10

            # minimize error and confirm unit testing
            self.Learn(minimize_error=True)

            # restore last successful links and network
            self.L = eval(self.X)
            self.N = eval(self.Y)
            self.Test()
            self.Test()
            print(self.N)
            print(f'\n :: pruning complete. pruned {self.S - len(self.L)} links. current size = {len(self.L)} ::')
            print(f'\n :: final unit testing.    error: {self.E:0.4f}    score: {self.F} / {len(self.U)} ::\n')


    def Visualize(self):
        # Create a new graph
        G = nx.DiGraph()

        # Add nodes with the node type as the node attribute
        for node in self.A:
            G.add_node(node, type='input')
        for node in self.B:
            G.add_node(node, type='hidden')
        for node in self.C:
            G.add_node(node, type='output')

        # Add edges with weights as the edge attribute
        for (node_1, node_2) in self.L:
            weight = self.N[node_1]['links'].get(node_2, 0)
            G.add_edge(node_1, node_2, weight=weight)

        # Define custom positions for the nodes
        pos = {**{node: (0, -i) for i, node in enumerate(self.A)},
               **{node: (1, -i + len(self.A)/2 - len(self.B)/2) for i, node in enumerate(self.B)},
               **{node: (2, -i + len(self.A)/2 - len(self.C)/2) for i, node in enumerate(self.C)}}

        # Draw the network nodes
        nx.draw_networkx_nodes(G, pos, nodelist=self.A, node_color='skyblue', node_size=700)
        nx.draw_networkx_nodes(G, pos, nodelist=self.B, node_color='lightgreen', node_size=700)
        nx.draw_networkx_nodes(G, pos, nodelist=self.C, node_color='salmon', node_size=700)

        # Draw the network node labels
        nx.draw_networkx_labels(G, pos, font_weight='bold')

        # Draw the network edges with colors and widths based on weight
        for edge in G.edges(data=True):
            color = 'green' if edge[2]['weight'] > 0 else 'red'
            width = abs(edge[2]['weight']) * 2  # Adjust width by weight magnitude
            nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], edge_color=color, width=width)

        # Update the plot with a title and without axis
        plt.title('Neural Network Visualization')
        plt.axis('off')  # Hide the axis
        plt.pause(0.1)  # Pause to update the plot


    # create html/js file with graph representation of network
    # def Visualize(self, node_size=20):

    #     file_string = '<html><body>'

    #     # links
    #     def Link(node_1, node_2, weight, x1, y1, x2, y2):
    #         return f'\n <line node_1="{node_1}" node_2="{node_2}" x1={x1+5+node_size/2} y1={y1+5+node_size/2} x2={x2+5+node_size/2} y2={y2+5+node_size/2} style="stroke-width:{abs(weight)}; stroke:{["#0f0b", "#f00b"][weight < 0]}"></line>'

    #     file_string += '\n\n<svg id="links">'
    #     for link in self.L:
    #         node_1, node_2 = link
    #         file_string += Link(node_1, node_2, self.N[node_1]['links'][node_2], {'A':100, 'B':200, 'C':300}[node_1[:1]], int(node_1[1:]) * 50 + 50, {'A':100, 'B':200, 'C':300}[node_2[:1]], int(node_2[1:]) * 50 + 50)
    #     file_string += '\n</svg>'

    #     # nodes
    #     def Node(id, value, x_pos, y_pos):
    #         rgb = min(255, int(abs(value)*255))
    #         rgb = [f'{rgb},0', f'0,{rgb}'][value > 0]
    #         return f'\n <span id="{id}" class="node" value={value} style="left:{str(x_pos)}; top:{str(y_pos)}; background-color:rgb({rgb},0)"></span>'

    #     file_string += '\n\n<div id="nodes">'
    #     for node in self.N:
    #         file_string += Node(node, self.N[node]['value'], {'A':100, 'B':200, 'C':300}[node[:1]], int(node[1:]) * 50 + 50)
    #     file_string += '\n</div>'

    #     # javascript
    #     file_string += f'\n\n<script>'

    #     # javascript move node functionality()
    #     # document.querySelectorAll('[my_attribute="attribute_value"]')

    #     file_string += '\n</script>'

    #     # css
    #     file_string += '\n\n<style>'
    #     file_string += '\n body {margin:0; background-color:#222}'
    #     file_string += '\n svg {height:100%; width:100%}'
    #     file_string += '\n .node {display:inline-block; position:absolute; height:20px; width:' + str(node_size) +'px; border-radius:50%; border:5px solid #000}'
    #     file_string += '\n</style>\n\n</body></html>'

    #     # write file
    #     file = open("graph.html", "w")
    #     file.write(file_string)
    #     file.close()

    def save_network_state(self, filepath='network_states.json'):
        # Check if the file exists and read the existing data
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                data = json.load(file)
        else:
            data = {}

        # Calculate the next run number based on the data
        run_number = str(len(data) + 1)  # We use len(data) + 1 to increment run numbers

        # Save the current network state and score
        data[run_number] = {
            'state': self.N,  # or however you access the network state to be stored
            'score': self.F , 
            'size': len(self.L) # or however you access the score to be stored
        }

        # Write the data back to the JSON file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

#######################################################################
#######################################################################
#######################################################################
#######################################################################

# UNIT TESTING / LEARNING / PRUNING

unit_tests = [
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1})
]


custom_weights = {
    # Connections from input nodes to hidden nodes
    ('A0', 'B0'): 0.1, ('A0', 'B1'): -0.2, ('A0', 'B2'): 0.3,
    ('A1', 'B0'): 0.4, ('A1', 'B1'): -0.5, ('A1', 'B2'): 0.6,
    ('A2', 'B0'): 0.7, ('A2', 'B1'): -0.8, ('A2', 'B2'): 0.9,
    ('A3', 'B0'): -0.1, ('A3', 'B1'): 0.2, ('A3', 'B2'): -0.3,
    ('A4', 'B0'): 0.4, ('A4', 'B1'): -0.5, ('A4', 'B2'): 0.6,
    ('A5', 'B0'): -0.7, ('A5', 'B1'): 0.8, ('A5', 'B2'): -0.9,
    ('A6', 'B0'): 0.1, ('A6', 'B1'): -0.2, ('A6', 'B2'): 0.3,
    
    # Connections from hidden nodes to the output node
    ('B0', 'C0'): -0.4, ('B1', 'C0'): 0.5, ('B2', 'C0'): -0.6,
}




#total_error = []

runs = 1000
# initialize
A, B, C = 7, 3, 1 # initial size is 34 (3 * (7 + 3) + 7 - 3)
# NN = STAG(A, B, C)
# 100 is a good seed
NN = STAG(A, B, C, seed_value=100)
print(NN.seed_value)
#print(NN.N)
NN.U = unit_tests

# Run
NN.Learn()
NN.Prune()
# filename = 'network_states.json'
# for _ in range(runs):
#     NN = STAG(A, B, C)
#     NN.U = unit_tests
#     NN.Learn()
#     NN.Prune()
#     NN.save_network_state(filename)
    #total_error.append(NN.E)

# average_error = sum(total_error) / len(total_error)
# print(f"Average total error after {runs} runs: {average_error}")

# NN.Test()
#NN.Visualize()
