
from math import tanh, log
from random import randint

# Directed Acyclical Graph Neural Network
class DAGNN:
    def __init__(self, structure={'A':0, 'B':0, 'C':0}):

        # store network structure and size
        self.A, self.B, self.C = structure['A'], structure['B'], structure['C']
        self.size = self.A * (self.B + self.C) + self.B * (self.B - 1) // 2 + self.B * self.C

        # node vectors
        self.nodes = [0] * (self.A + self.B + self.C)
        self.expected_output = [0] * self.C

        # hyper parameters
        self.error, self.fitness = 0, 0
        self.max_generations = 20_000
        self.unit_tests = []
        self.learning_rate = 0.01
        self.threshold = 2
        self.pruning = 0

        # create matrix and assign initial weights for each quadrant
        self.weights = (-.1, .1, -.1, 1)
        self.links = [
            [self.weights[0] if i < self.A and j < self.B else
            self.weights[1] if i < self.A else
            0 if j + self.A <= i else
            self.weights[2] if j < self.B else
            self.weights[3]
            for j in range(self.B + self.C)]
            for i in range(self.A + self.B)
        ]
        self.previous_links = str(self.links)

    def debug_print(self, message, debug):
        if debug:
            print(message)

    # count number of non-zero weights in links
    def Size(self):
        return sum(1 for i in self.links for j in i if j != 0)

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

        # set initial B and C nodes from A links
        self.nodes[self.A:] = self.DotProduct(self.nodes[:self.A], self.links[:self.A])

        # activate each B node and then compute next B and C nodes from B links
        for i, _ in enumerate(self.nodes[self.A:-1]):
            j = i + self.A
            self.nodes[j] = self.Activate(self.nodes[j])
            self.nodes[j+1:] = [a + b * self.nodes[j] for a, b in zip(self.nodes[j+1:], self.links[j][i+1:])]

        # activate all nodes in C
        self.nodes[-self.C:] = [self.Activate(x) for x in self.nodes[-self.C:]]

        # compound error and fitness
        self.error += self.MSE(self.nodes[-self.C:], self.expected_output)
        self.fitness += self.Score(self.nodes[-self.C:], self.expected_output)

    # compute error and fitness from unit tests
    def Test(self):
        self.error, self.fitness = 0, 0
        for inputs, outputs in self.unit_tests:
            self.nodes[:self.A], self.expected_output = inputs, outputs
            self.Forward()


    def Learn(self, debug=False):
        self.debug_print('\n :: minimizing error ::\n', debug) if self.pruning == 3 else self.debug_print('\n :: learning ::\n', debug)
        initial_learning_rate, current_generation = self.learning_rate + 0, 0
        learn_again = True
        current_error = 0
        while learn_again:
            learn_again = self.fitness < len(self.unit_tests) and current_generation <= self.max_generations
            if self.pruning == 3: learn_again = self.fitness == len(self.unit_tests) and current_error <= self.max_generations and self.error > self.threshold
            # get current error and fitness
            self.Test()
            if current_generation % 100 == 0:
                self.debug_print(f'gen: {current_generation}    error: {self.error:0.4f}    score: {self.fitness} / {len(self.unit_tests)}', debug)
            # adjust random weight and learning rate
            current_error = self.error + 0
            i = randint(0, self.A + self.B - 1)
            j = randint(max(0, i - self.A + 1), self.B + self.C - 1)
            if self.links[i][j] != 0:
                self.links[i][j] -= self.learning_rate
                self.Test()
                if self.error < current_error:
                    self.learning_rate *= 1.1
                else:
                    self.learning_rate = max(self.learning_rate / 1.1, initial_learning_rate)
                    if self.error > current_error:
                        self.links[i][j] += self.learning_rate * 2
                    else:
                        self.links[i][j] += self.learning_rate
            current_generation += 1
        # stopped learning
        self.learning_rate = initial_learning_rate + 0
        if current_generation > self.max_generations:
            self.debug_print('\n :: max generations reached ::\n', debug)
            #prompt = 'pruning' if self.pruning == 1 else 'minimizing' if self.pruning == 3 else 'learning'
            #gain = input(f'continue {prompt} (<enter> for yes)?: ')
            again = '0'
            if again == '': self.Learn()
            else:
                if self.pruning == 1:
                    self.debug_print('\n :: finished pruning ::\n', debug)
                else:
                    print('\n :: current failed end-state ::\n')
                self.pruning = 2
                for i in self.links: print(i)
        else:
            if current_generation % 100 != 0: print(f'gen: {current_generation}    error: {self.error:0.4f}    score: {self.fitness} / {len(self.unit_tests)}')
            print(f'\n :: learning complete ::\n') if self.pruning != 3 else print(f'\n :: minimizing complete ::\n')
            for i in self.links: print(i)
            if self.pruning == 0: self.Prune()

    def Prune(self):
        print('\n :: pruning ::')
        self.pruning = 1
        self.max_generations //= 10
        while self.pruning == 1:
            # save previous links
            self.previous_links = str(self.links)
            # find smallest non-zero weight
            smallest_weight = None
            for i, row in enumerate(self.links):
                for j, weight in enumerate(row):
                    if weight != 0:
                        if smallest_weight is None or abs(weight) < smallest_weight[1]:
                            smallest_weight = [[i, j], abs(weight)]
            # delete smallest weight
            i, j = smallest_weight[0]
            self.links[i][j] = 0
            print(f'\nremoving: ({i}, {j})')
            # re-evaluate
            self.Test()
            self.Learn()

        # restore previous links and minimize error
        self.links = eval(self.previous_links)
        self.max_generations *= 10
        self.pruning = 3
        self.Learn()

        # restore last successful links
        self.links = eval(self.previous_links)
        self.Test()
        for i in self.links: print(i)
        current_size = self.Size()
        print(f'\n :: pruning complete. pruned {self.size - current_size} weights. current size = {current_size} ::')
        print(f'\n :: final unit testing.    error: {self.error:0.4f}    score: {self.fitness} / {len(self.unit_tests)} ::\n')




# unit testing
# unit testing
# unit testing

NN = DAGNN({'A':7, 'B':3, 'C': 1})
NN.unit_tests = [([int(i) for i in bin(n)[2:]], [int(bin(n)[2:][1:5][n % 4])]) for n in range(64, 128)]
NN.Learn(debug=False)
