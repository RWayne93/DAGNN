
from random import random as R




# Sorted Topological Acyclical Graph
class STAG:
    def __init__(self, input_size, hidden_size, output_size):
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
        self.R = 0.001 # learning rate
        self.S = 0 # size / length of self.L
        self.T = 0.01 # threshold of error
        self.U = {} # unit tests
        self.W = (-1, 1) # initial domain of random weights
        self.X = None # previous links
        self.Y = None # previous network
        self.Z = True # prune while True

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

        for node in self.P:
            # compound error between output nodes and expected outputs from unit test
            self.E += Error(self.O[node], self.P[node])

            # compound fitness
            self.F += abs(self.O[node] - self.P[node]) < 0.5




    # select random links for later adjustment with learning rate
    def PickQ(self):
        self.Q = {}
        while len(self.Q) < (self.S ** .4) // 1:
            self.Q |= {self.L[int(len(self.L) * R())]: 0}




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

            # adjust learning rate for each selected Q-link
            self.PickQ()
            for Q in self.Q:
                node_1, node_2 = Q

                # temporarily adjust Q-link by learning rate
                self.N[node_1]['links'][node_2] += self.R

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
            if gen % 100 == 0: print(f'gen: {gen}    error: {self.E:0.4f}    score: {self.F} / {len(self.U)}')
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
            again = input('Continue Learning (<Enter> for Yes)?: ')
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

        if self.X != None:
            # restore previous links and network
            self.L = eval(self.X)
            self.N = eval(self.Y)
            self.G *= 10

            # minimize error and confirm unit testing
            NN.Learn(minimize_error=True)

            # restore last successful links and network
            self.L = eval(self.X)
            self.N = eval(self.Y)
            self.Test()
            self.Test()
            print(self.N)
            print(f'\n :: pruning complete. pruned {self.S - len(self.L)} links. current size = {len(self.L)} ::')
            print(f'\n :: final unit testing.    error: {self.E:0.4f}    score: {self.F} / {len(self.U)} ::\n')




# node activation function: f(x) = x - tanh(x)
def Activate(x):
    e = 2.7182818284590452
    return x + 2 / (1 + e ** (2 * x)) - 1




# error function: difference squared (or + binary classification)
def Error(x, y):
    return (x - y) ** 2




#######################################################################
#######################################################################
#######################################################################
#######################################################################

# UNIT TESTING / LEARNING / PRUNING

unit_tests = [
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 0, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1})
]

# initialize
A, B, C = 7, 3, 1 # initial size is 34 (3 * (7 + 3) + 7 - 3)
NN = STAG(A, B, C)
NN.U = unit_tests

# NN with size 17 and 64 / 64 unit tests
# NN.N = {'A0': {'value': 1, 'links': {'B2': 0.68}}, 'A1': {'value': 1, 'links': {'B0': 0.66, 'B2': 0.68}}, 'A2': {'value': 1, 'links': {'B1': -0.55}}, 'A3': {'value': 1, 'links': {'C0': 0.52}}, 'A4': {'value': 1, 'links': {'B0': -0.65, 'B1': 0.49}}, 'A5': {'value': 1, 'links': {'B0': 0.94, 'B1': 1.43, 'C0': 1.39}}, 'A6': {'value': 1, 'links': {'B0': 1.65, 'B1': -1.37, 'C0': 0.31}}, 'B0': {'value': 0, 'links': {'B2': -1.18}}, 'B1': {'value': 0.0, 'links': {'B2': -1.29, 'C0': -0.33}}, 'B2': {'value': 0, 'links': {'C0': 3.68}}, 'C0': {'value': 0}}
# NN.UpdateLinks()

# NN with no bias node, size 19, and 64 / 64 unit tests
# NN.N = {'A0': {'value': 0, 'links': {}}, 'A1': {'value': 1, 'links': {'B2': -0.9101299039340713, 'C0': 2.11935680780992}}, 'A2': {'value': 1, 'links': {'B0': 0.27449992940993273, 'B1': 0.3615585858770243}}, 'A3': {'value': 1, 'links': {'B0': -0.2515507432517288}}, 'A4': {'value': 1, 'links': {'B1': -0.3947052946797957, 'B2': 0.7143520280145191}}, 'A5': {'value': 1, 'links': {'B0': -3.0638016929946885, 'B1': -2.9811032225809737, 'B2': -0.8648428872087242}}, 'A6': {'value': 1, 'links': {'B0': -0.3269780232757363, 'B1': 1.2593160012175701, 'B2': -2.2025940064990235, 'C0': 1.3644440021056727}}, 'B0': {'value': -2.370203291250916, 'links': {'B1': -0.7553940869893714, 'C0': -1.7274383681474168}}, 'B1': {'value': 1.4910004552115552e-05, 'links': {'B2': 1.41723967783659, 'C0': 2.0731282151494366}}, 'B2': {'value': -2.2661179293108598, 'links': {'C0': 2.5409405834590832}}, 'C0': {'value': 0.8712883511016833}}
# NN.UpdateLinks()

# adjust max generations, learning rate, weights, and error threshold
NN.G = 5000
NN.R = 0.001
NN.T = 1.1
NN.W = [-1, 1]

# Run
NN.Learn()
NN.Prune()
