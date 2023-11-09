
import os
os.system('cls')

import numpy as np

# Directed Acyclical Graph Neural Network
class DAGNN:
    def __init__(self, structure={'A':0, 'B':0, 'C':0}):

        # store network structure size
        self.A = structure['A']
        self.B = structure['B']
        self.C = structure['C']

        # node vectors
        self.nodes = [0] * (self.A + self.B + self.C)
        self.expected_output = [0] * self.C

        # link matrices
        self.links = np.triu(np.ones((self.A + self.B, self.B + self.C)), k=1-self.A)

        # hyper parameters
        self.error = 0
        self.fitness = 0
        self.max_generations = 2_000
        self.unit_tests = []
        self.learning_rate = 0.001
        self.threshold = 2

    def Randomize(self):
        #randomized_array = np.random.uniform(lower_bound, upper_bound, size=array_size)
        self.links = np.random.rand(*(self.links).shape)

    # matrix multiplication and node activation
    def Forward(self):

        # set initial B and C nodes from A links to B and C
        self.nodes[self.A:] = np.dot(self.nodes[:self.A], self.links[:self.A])

        # B node activation function
        def Activate(x):
            return x * np.tanh(x)

        # activate B nodes and then compute B and C nodes from B links to B and C
        for i, _ in enumerate(self.nodes[self.A:-1]):
            j = i + self.A
            self.nodes[j] = Activate(self.nodes[j])
            # this might have an issue
            self.nodes[j+1:] += np.dot(self.nodes[j], self.links[j][i+1:])

        # C vector activation function
        vector_activate = np.vectorize(Activate)
        self.nodes[-self.C:] = vector_activate(self.nodes[-self.C:])
        print(self.nodes)

        # mean squared error function for two vectors
        def MSE(vector_a, vector_b):
            vector_error = np.vectorize(lambda x: abs(x) * np.log(abs(x) + 1))
            return sum(vector_error(vector_a - vector_b))

        # compound error
        self.error += MSE(np.array(self.nodes[-self.C:]), np.array(self.expected_output))

        # compare two vectors
        def Score(vector_a, vector_b):
            vector_classify = np.vectorize(lambda x, y: 1 * ((x - 0.5) * (y * 2 - 1) > 0))
            return sum(vector_classify(vector_a, vector_b))

        # compound fitness
        self.fitness += Score(self.nodes[-self.C:], self.expected_output)

    # compute error and fitness from unit tests
    def Test(self):
        self.error, self.fitness = 0, 0
        for inputs, outputs in self.unit_tests:
            self.nodes[:self.A], self.expected_output = inputs, outputs
            self.Forward()

    # machine learning algorithm
    def Learn(self, minimize_error=False):

        # get initial fitness of network
        self.Test()
        learn_again = self.fitness < len(self.unit_tests)
        if minimize_error:
            print('\n :: minimizing error ::\n')
            learn_again = self.fitness == len(self.unit_tests) and self.error > self.threshold
        else:
            print('\n :: learning ::\n')

        # stop learning if reached max generations or fitness is 100%
        initial_learning_rate = self.learning_rate + 0
        current_generation = 0
        while current_generation < self.max_generations and learn_again:

            # get current error
            current_error = self.error + 0
        # incomplete...

NN = DAGNN({'A':7, 'B':3, 'C': 1})
print(NN.links)
unit_tests = [(list(map(int, bin(n)[2:])), [list(map(int, bin(n)[2:]))[1:5][n % 4]]) for n in range(64, 128)]

NN.unit_tests = unit_tests
NN.Test()