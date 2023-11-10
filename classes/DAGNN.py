
import os
os.system('cls')

from math import tanh, log
from random import randint


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
        self.max_generations = 10_000
        self.unit_tests = []
        self.learning_rate = 0.01
        self.threshold = 2

        # initial weights for links in each quadrant
        self.weights = (0, 0, 0, 1)

        # create and populate the links matrix, and assign weights for each quadrant
        self.links = [
            [self.weights[0] if i < self.A and j < self.B else
            self.weights[1] if i < self.A else
            0 if j + self.A <= i else
            self.weights[2] if j < self.B else
            self.weights[3]
            for j in range(self.B + self.C)]
            for i in range(self.A + self.B)
        ]
        print(self.links)

    # B node activation function
    def Activate(self, x):
        return x * tanh(x)

    # mean squared error function for two vectors
    def MSE(self, a, b):
        return sum((abs(x - y) * log(abs(x - y) + 1) for x, y in zip(a, b)))

    # compare two vectors
    def Score(self, a, b):
        return sum((1 * ((x - 0.5) * (y * 2 - 1) > 0) for x, y in zip(a, b)))

    # dot product
    def DotProduct(self, a, b):
        return [sum(sublist[i] * j for j, sublist in zip(a, b)) for i in range(len(b[0]))]

    # matrix multiplication and node activation
    def Forward(self):

        # set initial B and C nodes from A links to B and C
        self.nodes[self.A:] = self.DotProduct(self.nodes[:self.A], self.links[:self.A])

        # activate B nodes and then compute B and C nodes from B links to B and C
        for i, _ in enumerate(self.nodes[self.A:-1]):
            j = i + self.A
            self.nodes[j] = self.Activate(self.nodes[j])
            self.nodes[j+1:] = [a + b * self.nodes[j] for a, b in zip(self.nodes[j+1:], self.links[j][i+1:])]

        # C vector activation function
        self.nodes[-self.C:] = [self.Activate(x) for x in self.nodes[-self.C:]]

        # compound error
        self.error += self.MSE(self.nodes[-self.C:], self.expected_output)

        # compound fitness
        self.fitness += self.Score(self.nodes[-self.C:], self.expected_output)

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
            for i in range(1):
                j = randint(0, self.A + self.B - 1)
                k = randint(max(0, j-self.A+1), self.B + self.C - 1)
                self.links[j][k] += self.learning_rate
                self.Test()
                Q += [[(j, k), (current_error - self.error) / self.learning_rate]]
                self.links[j][k] -= self.learning_rate

            # sort q links
            #Q.sort(key = lambda x: abs(x[1]), reverse=True)

            # update link in network with largest error gradient by learning rate
            (i, j), error = Q[0]
            self.links[i][j] += self.learning_rate * [1, -1][error < 0]

            # print every 100th generation
            if current_generation % 100 == 0:
                print(f'gen: {current_generation}    error: {self.error:0.4f}    score: {self.fitness} / {len(self.unit_tests)}')

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

        print(f'gen: {current_generation}    error: {self.error:0.4f}    score: {self.fitness} / {len(self.unit_tests)}')
        for i in self.links: print(i)
        print(self.nodes)

    def Prune(self):

        self.max_generations //= 10
        while self.prune_again:

            print('\n :: pruning ::')

# unit testing

NN = DAGNN({'A':7, 'B':3, 'C': 1})
NN.unit_tests = [([int(i) for i in bin(n)[2:]], [int(bin(n)[2:][1:5][n % 4])]) for n in range(64, 128)]
NN.Learn()
