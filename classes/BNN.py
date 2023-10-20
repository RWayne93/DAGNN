from random import randint

RULES = {}
RULES['population'] = 1 << 7        # 2 ** (X + 1)
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
        self.Size()
        self.N = 1 << self.A - 1        # input and hidden binary values
        if rand:                        # randomizes bunny genome
            self.G = randint(1, (1 << self.S) - 1)
    def Size(self):                     # computes length of bunny genome
        if self.S == 0:
            self.S = self.A * self.C + self.B * (self.A + self.C) + self.B * (self.B - 1) // 2
        return self.S
    def Propogate(self):                # computes self.O from self.G, self.S, and self.N
        self.O = 0
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

def Int2Bin(n, s=-1):
    b = bin(n)[2:]
    if s != -1:
        return ((s - len(b)) * '0') + b
    return b

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

def Int2Str(n, l):
    return ' ' * max(0, l - len(str(n))) + str(n)

A = 3
B = 0
C = 1

"""
    INSTRUMENT         T/F REPRESENTATION    min len of B for 100%
    False              0, 0, 0, 0            0
    A and B            0, 0, 0, 1            1
    not A and B        0, 0, 1, 0            ...
    B                  0, 0, 1, 1            0
    A and not B        0, 1, 0, 0            ...
    A                  0, 1, 0, 1            0
    A xor B            0, 1, 1, 0            0
    A or B             0, 1, 1, 1            ...
    not A and not B    1, 0, 0, 0            ...
    not (A xor B)      1, 0, 0, 1            0
    not A and B        1, 0, 1, 0            0
    not A or B         1, 0, 1, 1            ...
    not B              1, 1, 0, 0            0
    A or not B         1, 1, 0, 1            ...
    not (A and B)      1, 1, 1, 0            ...
    True               1, 1, 1, 1            0
"""

base = Bunny(A, B, C, False)
S = base.Size()

for i in range(1 << S):
    b = Bunny(A, B, C, False)
    b.G = i

    for j in range(4):
        b.N = 4 + j
        b.Propogate()
        #       0,0   0,1   1,0   1,1
        test = [1,    1,    1,    1][j]
        b.F += b.O == test

    print(b.F, Int2Str(i, len(str(1 << S))), Int2Bin(b.G, S))
