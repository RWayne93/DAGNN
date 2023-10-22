from random import *

RULES = {}
RULES['population'] = 1 << 7        # 2 ** (X + 1)
RULES['network']    = [7, 7, 1]     # number of nodes in layers A, B, and C
RULES['mutation']   = 0.5           # mutation rate

class Bunch:
    def __init__(self, population=RULES['population'], network=RULES['network'], mutation=RULES['mutation']):
        self.population = population    # how big is the bunch in the burrow
        self.network    = network       # number of nodes in layers A, B, and C
        self.mutation   = mutation      # mutation rate
        self.bunnies = []               # list of bunnies
        self.Populate()
    def Populate(self):
        while len(self.bunnies) < self.population:
            self.bunnies += [Bunny(A=self.network[0], B=self.network[1], C=self.network[2], R=self.mutation)]
    def Selection(self, nomads=0.5):                # create next generation
        self.bunnies.sort(key=lambda bunny: bunny.F, reverse=True)
        next_gen = [self.bunnies[0]]
        while len(next_gen) < self.population * (1 - nomads):
            parents = choices(population=self.bunnies, weights=[bunny.F for bunny in self.bunnies], k=2)
            next_gen += [self.Crossover(parents)]
        while len(next_gen) < self.population:
            next_gen += [Bunny(A=self.network[0], B=self.network[1], C=self.network[2], R=self.mutation)]
        self.bunnies = next_gen
    def Crossover(self, parents):
        child = Bunny(A=self.network[0], B=self.network[1], C=self.network[2], R=self.mutation, rand=False)
        random_slice_1 = randint(1, (1 << child.S) - 1)
        random_slice_2 = ((1 << child.S) - 1) ^ random_slice_1
        child.G = random_slice_1 & parents[0].G | random_slice_2 & parents[1].G
        child.Mutate()
        return child

class Bunny:
    def __init__(self, A, B, C, R=0.5, rand=True):
        self.A = A                      # nodes in layer A
        self.B = B                      # nodes in layer B
        self.C = C                      # nodes in layer C
        self.F = 0                      # fitness
        self.R = R                      # mutation rate
        self.O = 0                      # output binary values
        self.G = 0                      # bunny genome
        self.S = 0                      # store bunny size
        self.N = 1 << self.A - 1        # input and hidden binary values
        self.P = None                   # store memory for propogation
        self.Size()
        if rand:                        # randomizes bunny genome
            self.G = randint(1, (1 << self.S) - 1)
    def Size(self):                     # computes length of bunny genome
        if self.S == 0:
            self.S = self.A * self.C + self.B * (self.A + self.C) + self.B * (self.B - 1) // 2
        return self.S
    def Mutate(self):                   # switches random bit in genome on/off
        if random() < self.R:
            self.G ^= 1 << randint(0, self.S - 1)
    def Transform(self):                # computes P transformation of bits on self.G for propogation
        if self.P == None:
            self.P = self.G + 0
            for i in range(self.B - 1):
                self.P = Cycle(self.P, self.S, self.C * (i + 1), self.A + self.C + self.B + self.B * 2 * i - i * (i - 1) // 2 - 1, self.C)
            self.P = Cycle(self.P, self.S, self.C * self.B, self.A * (self.C + self.B) + self.B * (self.B - 1) // 2, self.A * self.C)
            for i in range(self.C - 1):
                for j in range(self.B - 1):
                    self.P = Cycle(self.P, self.S, 1 + self.B * i + j, self.C - i + j * (self.C - i - 1), 1)
            for i in range(self.C - 1):
                self.P = Cycle(self.P, self.S, i * (self.A + self.B) + self.B, self.B * (self.C - i - 1) + self.A, self.A)
    def Propogate(self):                # computes self.O from transformed self.G, self.S, and self.N
        #self.Transform()
        #P = self.P + 0
        self.O = 0
        P = self.G + 0
        for i in range(self.B):
            j = self.A + i
            self.N += Activate(Bits(self.N & (P & (1 << j) - 1))) * (1 << j)
            P >>= j
        for i in range(self.C):
            j = self.A + self.B
            self.O += Activate(Bits(self.N & (P & (1 << j) - 1))) * (1 << i)
            P >>= j

# a: intenger, b: mask_length, c: start_position, d: cycle_length, e: right_shifts
# pad zero's to the left of the number equal to the difference of mask_length - len(integer)
# keep bits unchanged to the left of the start_position
# keeps bits unchanged to the right of the end_position (start_position + cycle_length)
# the bits in the cycle are what's left in the middle
# perform a circular shift to the right on the bits in the middle of the number
# bits pushed off the right side get filled in to the left side
def Cycle(a, b, c, d, e):
    f = b - c
    g = f - d
    h = a >> g & (1 << d) - 1
    return a >> f << f | (((1 << e) - 1 & h) << d - e | h >> e) << g | (1 << g) - 1 & a

def Activate(n): # return 1 if n is 1 or greater than 4
    return n not in [0, 2, 3]

def Activate2(n): # return 1 if n is a triangle number: else 0
    return (n * 8 + 1) ** .5 % 1 == 0

def Bits(n):
    return (bin(n)).count('1')

def Int2Bin(n, s=-1):
    b = bin(n)[2:]
    if s != -1:
        return ((s - len(b)) * '0') + b
    return b

def Bin2Int(n):
    return int(n, 2)

def Int2Str(n, l):
    return ' ' * max(0, l - len(str(n))) + str(n)

#######################################################
###    64 UNIT TESTS WITH GENERATIONAL MUTATION     ###
#######################################################

NN = Bunch()

for i in range(9999):

    for bunny in NN.bunnies:
        bunny.F = 0
        for j in range(16):
            for k in range(4):
                bunny.N = (1 << 7) + (j << 2) + k
                bunny.Propogate()
                bunny.F += bunny.O == ((j >> k) & 1)
    
    NN.Selection()
    bb = NN.bunnies[0]
    print(i, bb.F, bb.G)

"""
    def Molt(self):
        if randint(0, 2) and self.coat >= 1 << (self.eyes).bit_length():
            self.ShortCoat()
        if randint(0, 2): self.LongCoat()
        if randint(0, 2): self.BrushCoat()

    def ShortCoat(self):
        return None

    def LongCoat(self):
        return None

    def BrushCoat(self):
        return None

    def Shed(self, shed=True):
        return None
"""

TRUTH_TABLES = {
    "False": [0, 0, 0, 0],
    "A and B": [0, 0, 0, 1],
    "B": [0, 0, 1, 1],
    "A and not B": [0, 1, 0, 0],
    "not A and B": [0, 0, 1, 0],
    "not A not B": [1, 0, 1, 0],
    "A": [0, 1, 0, 1],
    "A xor B": [0, 1, 1, 0],
    "A or B": [0, 1, 1, 1],
    "not A and not B": [1, 0, 0, 0],
    "not (A xor B)": [1, 0, 0, 1],
    "not A or B": [1, 0, 1, 1],
    "not B": [1, 1, 0, 0],
    "A or not B": [1, 1, 0, 1],
    "not (A and B)": [1, 1, 1, 0],
    "True": [1, 1, 1, 1]
}

"""
OLD TEST

A = 3
B = 1
C = 1

base = Bunny(A, B, C, False)
S = base.Size()

# for i in range(1 << S):
#     b = Bunny(A, B, C, False)
#     b.G = i

#     for j in range(4):
#         b.N = 4 + j
#         b.Propogate()
#         #       0,0   0,1   1,0   1,1
#         test = [1,    1,    1,    1][j]
#         b.F += b.O == test

#     print(b.F, Int2Str(i, len(str(1 << S))), Int2Bin(b.G, S))

for i in range(1 << S):
    b = Bunny(A, B, C, False)
    b.G = i
    
    for function_name, expected_outputs in TRUTH_TABLES.items():
        correct_outputs = 0
        for j in range(4):
            b.N = 4 + j
            b.Propogate()
            correct_outputs += b.O == expected_outputs[j]
            
        # For debugging, you can print the fitness for each function:
        print(f"{function_name}: {correct_outputs}")


network_configs = [
    [2, 0, 1],
    [2, 1, 1],
    [2, 2, 1],
    [2, 3, 1],
    # ... add as many configurations as you're interested in testing
]

results = {}

for config in network_configs:
    RULES['network'] = config
    # Initialize and simulate/test your Bunch of bunnies here.
    bunch = Bunch()
    # ... your simulation/testing code
    # At the end, store the average fitness (or some other metric) of the bunnies.
    avg_fitness = sum([bunny.F for bunny in bunch.bunnies]) / len(bunch.bunnies)
    results[str(config)] = avg_fitness

# Once all simulations are done, print out the results:
for config, fitness in results.items():
    print(f"Configuration {config}: Average Fitness = {fitness}")
"""
