
import os
from math import tanh, log
from random import uniform, randint

# Directed Acyclical Graph Neural Network
class DAGNN:
    def __init__(self, structure={'A':0, 'B':0, 'C':0}):
        self.A, self.B, self.C = structure['A'], structure['B'], structure['C']
        self.AB, self.BC = self.A + self.B, self.B + self.C
        self.nodes, self.expected_output = [0] * (self.AB + self.C), [0] * self.C
        self.weights = (-1, 1)
        self.links = [[0 if j + self.A <= i else uniform(self.weights[0], self.weights[1]) for j in range(self.BC)] for i in range(self.AB)]
        self.error, self.fitness, self.current_generation, self.max_generations, self.delta = 0, 0, 0, 5_000, 0.00001
        self.unit_tests, self.num_tests, self.previous_links, self.stage = [], None, None, 'learning'
        self.error_array = []

    def Display(self):
        os.system('cls')
        for row in self.links: print('[ ' + ''.join([f'{w: .4f}  ' if w != 0 else '         ' for w in row])[:-2] + ' ]')
        gen = [f'  gen: {self.current_generation}  ', ''][self.stage == 'finished']
        print(f'\n :: {self.stage} ::  {gen}  error: {self.error:.4f}    fitness: {self.fitness} / {self.num_tests}    size: {sum(1 for i in self.links for j in i if j != 0)}\n')

    def Activate(self, x): return abs(x)

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
            self.links[i][j], current_error = self.links[i][j] + self.delta, self.error + 0
            self.Test()
            self.links[i][j], right_error = self.links[i][j] - self.delta, self.error + 0
            error_gradient = (current_error - right_error) / self.delta
            step = error_gradient / 400
            self.links[i][j] += step
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
