
import os
from math import tanh, log
from random import randint

# Directed Acyclical Graph Neural Network
class DAGNN:
    def __init__(self, structure={'A':0, 'B':0, 'C':0}):
        self.A, self.B, self.C = structure['A'], structure['B'], structure['C']
        self.AB, self.BC = self.A + self.B, self.B + self.C
        self.nodes, self.expected_output = [0] * (self.AB + self.C), [0] * self.C
        self.weights = (.1, -.1, -.1, .1)
        self.links = [[self.weights[0] if i < self.A and j < self.B else self.weights[1] if i < self.A else 0 if j + self.A <= i else self.weights[2] if j < self.B else self.weights[3] for j in range(self.BC)] for i in range(self.AB)]
        self.error, self.fitness, self.current_generation, self.max_generations, self.step_rate = 0, 0, 0, 5_000, 0.00001
        self.unit_tests, self.num_tests, self.previous_links, self.stage = [], None, None, 'learning'

    def Display(self):
        os.system('cls')
        for row in self.links: print('[ ' + ''.join([f'{w: .4f}  ' if w != 0 else '         ' for w in row])[:-2] + ' ]')
        print(f'\n :: {self.stage} ::    ' + [f'gen: {self.current_generation}    ', ''][self.stage == 'finished'] + f'error: {self.error:.4f}    fitness: {self.fitness} / {self.num_tests}    ' + ['', f'size: {sum(1 for i in self.links for j in i if j != 0)}\n'][self.stage == 'finished'])

    def Activate(self, x): return x * tanh(x)

    def Forward(self):
        for i in range(self.BC):
            j = i + self.A
            self.nodes[j] = self.Activate(sum(x * y for x, y in zip(self.nodes[:j], [row[i] for row in self.links[:j]])))
        self.error   += sum((abs(x - y) * log(abs(x - y) + 1) for x, y in zip(self.nodes[-self.C:], self.expected_output)))
        self.fitness += sum((1 * ((x - 0.5) * (y * 2 - 1) > 0) for x, y in zip(self.nodes[-self.C:], self.expected_output)))

    def Test(self, display=''):
        self.error, self.fitness = 0, 0
        for inputs, outputs in self.unit_tests:
            self.nodes[:self.A], self.expected_output = inputs, outputs
            self.Forward()
        if self.fitness == self.num_tests:
            self.previous_links = str(self.links)
        if display == 'display': self.Display()

    def UpdateWeight(self):
        self.current_generation += 1
        i = randint(0, self.AB - 1)
        j = randint(max(0, i - self.A + 1), self.BC - 1)
        if self.links[i][j] != 0:
            self.Test()
            self.links[i][j], current_error = self.links[i][j] + self.step_rate, self.error + 0
            self.Test()
            self.links[i][j], right_error = self.links[i][j] - self.step_rate, self.error + 0
            error_gradient = (current_error - right_error) / self.step_rate
            self.links[i][j] += tanh(current_error * error_gradient / 200) / 1.5
        if self.current_generation % 100 == 0: self.Display()

    def Learn(self):
        self.current_generation = 0
        self.Test('display')
        while self.current_generation < self.max_generations:
            self.UpdateWeight()
            if self.current_generation == self.max_generations:
                if input(f'\n :: max generations reached. <enter> to continue {self.stage} :: ') == '': self.current_generation = 0
            if ((self.stage == "minimizing") ^ (self.fitness == self.num_tests)): break
        if self.stage in ['learning', 'pruning']:
            if self.fitness == self.num_tests: self.Prune()
            elif self.stage == 'learning': print(f'\n :: stopped learning ::\n')
            else:
                self.stage, self.links = 'minimizing', eval(self.previous_links)
                self.Learn()
        elif self.stage == 'minimizing':
            self.stage, self.links = 'finished', eval(self.previous_links)
            self.Learn()
            self.links = eval(self.previous_links)

    def Prune(self):
        self.stage = 'pruning'
        while self.stage == 'pruning':
            self.links[w[0]][w[1]] = 0 if (w := min(((i, j, abs(weight)) for i, row in enumerate(self.links) for j, weight in enumerate(row) if weight != 0), key=lambda x: x[2], default=None)) is not None else self.links[w[0]][w[1]]
            self.Learn()
        self.Test('display')

# initialize and learning
NN = DAGNN({'A':7, 'B':3, 'C': 1})
NN.unit_tests = [([int(i) for i in bin(n)[2:]], [int(bin(n)[2:][1:5][n % 4])]) for n in range(64, 128)]
NN.num_tests = len(NN.unit_tests)
NN.Learn()
