from random import random as R

# Sorted Topological Acyclical Graph
class STAG:
    def __init__(self, input_size, hidden_size, output_size):
        self.A = tuple(['A' + str(i).zfill(len(str(input_size - 1))) for i in range(input_size)]) # input node ids
        self.B = tuple(['B' + str(i).zfill(len(str(hidden_size - 1))) for i in range(hidden_size)]) # hidden node ids
        self.C = tuple(['C' + str(i).zfill(len(str(output_size - 1))) for i in range(output_size)]) # output node ids
        self.E = 1 # error / fitness
        self.I = [] # unit tests / input values
        self.L = [] # list of links
        self.N = {} # node data
        self.O = {} # output nodes
        self.P = [] # unit tests / predicted values of output nodes
        self.Q = {} # links selected for random walk backward
        self.R = 0.001 # learning rate
        self.S = 0 # size / length of self.L
        self.T = 0.01 # threshold of error
        self.W = (-1, 1) # initial domain of random weights

        D = self.B + self.C
        for node_1 in self.A + self.B:
            # assign input and hidden node ids to node data
            self.N |= {node_1: {'value': 0, 'links': {}}}

            # create link with random weight between input and hidden nodes to other hidden nodes and output nodes
            j = 0
            while j < len(D):
                while D[j] <= node_1: j += 1
                node_2 = D[j]
                self.L += [(node_1, node_2)]
                self.N[node_1]['links'][node_2] = self.RandomWeight()
                j += 1

        # assign output node ids to node data
        for node in self.C: self.N |= {node: {'value': 0}}
        self.L = tuple(self.L)
        self.S = len(self.L)

    # return random weight between values of self.W
    def RandomWeight(self):
        return R() * (self.W[1] - self.W[0]) + self.W[0]

    # propogate input node values through the network to compute output node values
    def Forward(self):
        # set input nodes to values assigned by unit test
        for node in self.A: self.N[node]['value'] = self.I[node]
        # set hidden and output nodes to zero
        for node in self.B + self.C: self.N[node]['value'] = 0

        last_activated = None
        for (node_1, node_2) in self.L:
            # activate each hidden node once
            if node_1[0] != 'A' and node_1 != last_activated:
                self.N[node_1]['value'] = Activate(self.N[node_1]['value'])
                last_activated = node_1
            # sum product of node values and weights
            self.N[node_2]['value'] += self.N[node_1]['value'] * self.N[node_1]['links'][node_2]

        # activate output nodes
        for node in self.C: self.N[node]['value'] = Activate(self.N[node]['value'])

        # pass output nodes to self.O
        for node in self.C: self.O |= {node: self.N[node]['value']}

        # compute error / fitness as sum of differences squared between output nodes and expected outputs from unit test
        for node in self.P: self.E += Error(self.O[node], self.P[node])

    # select random links for later adjustment with learning rate
    def PickQ(self):
        self.Q = {}
        while len(self.Q) < (self.S ** .4) // 1:
            self.Q |= {self.L[int(len(self.L) * R())]: 0}

    # compute error from passing test through network
    def Test(self, tests):
        self.E = 0
        for inputs, outputs in tests:
            self.I = inputs
            self.P = outputs
            self.Forward()

    # machine learning for network
    def Learn(self, tests, max_gens=1000):
        gen = 0
        # stop if max generations exceeded or error passes threshold
        while gen < max_gens and self.E > self.T:
            # get current error
            self.Test(tests)
            current_error = self.E + 0
            print(f'gen: {gen}    error: {current_error:0.3f}')

            # adjust learning rate for each selected Q-link
            self.PickQ()
            for Q in self.Q:
                node_1, node_2 = Q

                # temporarily adjust Q-link by learning rate
                self.N[node_1]['links'][node_2] += self.R

                # compute error difference after applying the learning rate
                self.Test(tests)
                self.Q[Q] = (current_error - self.E) / self.R

                # remove learning rate from Q-link
                self.N[node_1]['links'][node_2] -= self.R

            # Sort Q links by magnitude of error difference
            self.Q = [(key, val) for (key, val) in self.Q.items()]
            (self.Q).sort(key = lambda x: abs(x[1]), reverse=True)

            # update link in network with largest error gradient by learning rate
            (node_1, node_2), err = self.Q[0]
            self.N[node_1]['links'][node_2] += self.R * [1, -1][err < 0]

            gen += 1
        print(self.N)

# node activation function: f(x) = x - tanh(x)
def Activate(x):
    e = 2.7182818284590452
    a, b = e ** x, e ** (-x)
    return x - (a - b) / (a + b)

# error function: difference squared
def Error(x, y):
    return (x - y) ** 2
"""
    a = (x - y) ** 2
    b = (2 * x - 1) * (2 * y - 1) < 0
    return 1 - (a + b)
"""

#######################################################################
#######################################################################
#######################################################################
#######################################################################

# UNIT TESTING

"""
A, B, C = 3, 1, 1
NN = STAG(A, B, C)

unit_tests = [
    ({'A0':1, 'A1':0, 'A2':0}, {'C0':1}),
    ({'A0':1, 'A1':0, 'A2':1}, {'C0':0}),
    ({'A0':1, 'A1':1, 'A2':0}, {'C0':0}),
    ({'A0':1, 'A1':1, 'A2':1}, {'C0':1})
]
"""

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
A, B, C = 7, 3, 1
NN = STAG(A, B, C)

# adjust learning rate, weights, and threshold
NN.R = 0.001
NN.W = [-1, 1]
NN.T = 0.25

NN.Learn(unit_tests, max_gens=99999)
