from random import randint

RULES = {}
RULES['population'] = 1 << 10       # 1_024
RULES['network']    = [2, 0, 1]     # number of nodes in layers A, B, and C
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
        if rand:                        # randomizes bunny genome
            self.G = randint(1, 1 << self.Size())
    def Size(self):                     # computes length of bunny genome
        if self.S == 0:
            self.S = self.A * self.C + self.B * (self.A + self.C) + self.B * (self.B - 1) // 2
        return self.S
    def Propogate(self):                # computes self.O from self.G, self.S, and self.N
        P = self.G + 0
        for i in range(self.B - 1):
            P = Cycle(P, self.S, self.C * (i + 1), self.A + self.C + self.B + self.B * 2 * i - i * (i - 1) // 2 - 1, self.C)
        P = Cycle(P, self.S, self.C * self.B, self.A * (self.C + self.B) + self.B * (self.B - 1) // 2, self.A * self.C)
        for i in range(self.C - 1):
            for j in range(self.B - 1):
                P = Cycle(P, self.S, 1 + self.B * i + j, self.C - i + j * (self.C - i - 1), 1)
        for i in range(self.C - 1):
            P = Cycle(P, self.S, i * (self.A + self.B) + self.B, self.B * (self.C - i - 1) + self.A, self.A)
        for i in range(self.B):
            j = self.A + i
            self.N += Activate(Bits(self.N & (P & (1 << j) - 1))) * (1 << j)
            P >> j
        for i in range(self.C):
            j = self.A + self.B
            self.O += Activate(Bits(self.N & (P & (1 << j) - 1))) * (1 << i)
            P >> j

# a: intenger, b: mask_length, c: start_position, d: cycle_length, e: right_shifts
def Cycle(a, b, c, d, e):
    f = b - c
    g = f - d
    h = a >> g & (1 << d) - 1
    return a >> f << f | (((1 << e) - 1 & h) << d - e | h >> e) << g | (1 << g) - 1 & a

def Activate(n):
    i = 0 if n in [0, 2] else 1
    return i

def Bits(n):
    return (bin(n)).count('1')

def Int2Bin(n):
    return bin(n)[2:]

def Bin2Int(n):
    return int(n, 2)

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
