
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

        # hyper parameters
        self.error = 0
        self.fitness = 0
        self.max_generations = 2_000
        self.unit_tests = []
        self.learning_rate = 0.01
        self.threshold = 2
        self.weights = (1, -1, -1, 1) # initial weights for links in each quadrant

        # create and populate the links matrix
        self.links = np.zeros((self.A + self.B, self.B + self.C))
        
        # assign weights for each quadrant
        self.links[:self.A, :self.B] = self.weights[0]
        self.links[:self.A, self.B:] = self.weights[1]
        self.links[self.A:, :self.B] = np.triu(np.full((self.B, self.B), self.weights[2]), k=1)
        self.links[self.A:, self.B:] = self.weights[3]




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

            # store changes in error
            current_error = self.error + 0
            Q = []
            for i in range(2):
                j = np.random.randint(0, self.A + self.B)
                k = np.random.randint(max(0, j-self.A+1), self.B + self.C)
                self.links[j][k] += self.learning_rate
                self.Test()
                Q += [[(j, k), (current_error - self.error) / self.learning_rate]]
                self.links[j][k] -= self.learning_rate

            # sort q links
            Q.sort(key = lambda x: abs(x[1]), reverse=True)

            # update link in network with largest error gradient by learning rate
            (i, j), error = Q[0]
            self.links[i][j] += self.learning_rate * [1, -1][error < 0]

            # print every 100th generation
            if current_generation % 100 == 0: print(f'gen: {current_generation}    error: {self.error:0.4f}    score: {self.fitness} / {len(self.unit_tests)}')
            current_generation += 1

            # test for next generation
            self.Test()

            # adjust learning rate based on adjustment success
            if self.error < current_error:
                self.learning_rate *= 1.1
            else:
                if self.learning_rate > initial_learning_rate:
                    self.learning_rate = initial_learning_rate + 0
                else:
                    self.learning_rate /= 1.1

            learn_again = self.fitness < len(self.unit_tests)

            """if minimize_error:
                learn_again = self.fitness == len(self.unit_tests)
                if learn_again:
                    learn_again &= self.error > self.threshold"""

        self.learning_rate = initial_learning_rate + 0

        # incomplete...

        print(f'gen: {current_generation}    error: {self.error:0.4f}    score: {self.fitness} / {len(self.unit_tests)}')
        print(self.links)

# unit testing

NN = DAGNN({'A':7, 'B':3, 'C': 1})
unit_tests = [(list(map(int, bin(n)[2:])), [list(map(int, bin(n)[2:]))[1:5][n % 4]]) for n in range(64, 128)]

print(NN.links)

NN.unit_tests = unit_tests
NN.Learn()
