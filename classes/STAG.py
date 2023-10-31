from random import random as R

# Sorted Topological Acyclical Graph

class STAG:
    def __init__(self, input_size, hidden_size, output_size):
        self.A = input_size
        self.B = hidden_size
        self.C = output_size
        self.D = [] # node ids
        self.E = 1 # error
        self.F = 0 # fitness
        self.I = [] # unit tests / input values
        self.L = [] # list of links
        self.N = {} # node data
        self.O = {} # output nodes
        self.P = [] # unit tests / predicted values of output nodes
        self.Q = {} # links selected for random walk backward
        self.R = 0.001 # learning rate
        self.S = 0 # size / length of self.L
        self.T = 0.01 # threshold of error
        self.W = [-1, 1] # initial domain of random weights

        # generate node ids
        for i in range(self.A):
            self.D += ['A' + str(i).zfill(len(str(self.A - 1)))]
        for i in range(self.B):
            self.D += ['B' + str(i).zfill(len(str(self.B - 1)))]
        for i in range(self.C):
            self.D += ['C' + str(i).zfill(len(str(self.C - 1)))]
        self.D = tuple(self.D)

        # generate values and links for each node id
        for i in range(len(self.D)):
            node_id = self.D[i]
            self.N |= {node_id:{'value':0}}
            if node_id[0] != 'C':
                self.N[node_id]['links'] = {}
                j = max(self.A, i + 1)
                while j < len(self.D):
                    node_out = self.D[j]
                    self.L += [(node_id, node_out)]
                    self.N[node_id]['links'][node_out] = self.RandomWeight()
                    j += 1
        self.L = tuple(self.L)
        self.S = len(self.L)

    def RandomWeight(self):
        return R() * (self.W[1] - self.W[0]) + self.W[0]

    def Forward(self):
        for n in self.N:
            if n in self.I:
                self.N[n]['value'] = self.I[n]
            else:
                self.N[n]['value'] = 0
        i = 'A0'
        for (a, b) in self.L:
            if a[0] != 'A' and a != i:
                self.N[a]['value'] = Activate(self.N[a]['value'])
                i = a
            self.N[b]['value'] += self.N[a]['value'] * self.N[a]['links'][b]
        for n in self.N:
            if n[0] == 'C':
                self.N[n]['value'] = Activate(self.N[n]['value'])
        self.O = dict([(node, self.N[node]['value']) for node in self.N if node[0] == 'C'])
        self.E += sum([Error(self.O[i], self.P[i]) for i in self.P])

    def PickQ(self):
        self.Q = {}
        while len(self.Q) < (self.S ** .4) // 1:
            self.Q |= {self.L[int(len(self.L) * R())]: 0}

def Activate(x):
    e = 2.7182818284590452
    a, b = e ** x, e ** (-x)
    return x - (a - b) / (a + b)

def Error(x, y):
    return (x - y) ** 2
    a = (x - y) ** 2
    b = (2 * x - 1) * (2 * y - 1) < 0
    return 1 - (a + b)




#######################################################################
#######################################################################
#######################################################################
#######################################################################

# UNIT TESTING


A, B, C = 7, 1, 1
NN = STAG(A, B, C)

unit_tests = [
    ({'A0':1, 'A1':0, 'A2':0}, {'C0':1}),
    ({'A0':1, 'A1':0, 'A2':1}, {'C0':0}),
    ({'A0':1, 'A1':1, 'A2':0}, {'C0':0}),
    ({'A0':1, 'A1':1, 'A2':1}, {'C0':1})
]

gen = 0
max_gens = 9999

while NN.E > NN.T and gen < max_gens:
    NN.E = 0
    for inputs, outputs in unit_tests:
        NN.I = inputs
        NN.P = outputs
        NN.Forward()
    current_error = NN.E + 0
    print(gen, current_error)
    NN.PickQ()
    for Q in NN.Q:
        a, b = Q
        NN.N[a]['links'][b] += NN.R
        NN.E = 0
        for inputs, outputs in unit_tests:
            NN.I = inputs
            NN.P = outputs
            NN.Forward()
        NN.Q[Q] = (current_error - NN.E) / NN.R
        NN.N[a]['links'][b] -= NN.R
    Q = [(key, val) for (key, val) in NN.Q.items()]
    Q.sort(key = lambda x: abs(x[1]), reverse=True)
    (a, b), c = Q[0]
    NN.N[a]['links'][b] += NN.R * [1, -1][c < 0]
    gen += 1
print(NN.N)

for inputs, outputs in unit_tests:
    NN.I = inputs
    NN.Forward()
    print(NN.I, NN.O)
NN.Forward()
print(NN.N)
