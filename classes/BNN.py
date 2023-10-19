from random import randint as R

RULES = {}
RULES['burrow_size'] = 1 << 10      # 1_024
RULES['bunny_shape'] = [2, 0, 1]    # [eyes (inputs), ears (hidden), toes (outputs)]
RULES['glow_rate']   = 0.5          # mutation rate

class Bunch:
    def __init__(self, burrow_size=RULES['burrow_size'], bunny_shape=RULES['bunny_shape'], glow_rate=RULES['glow_rate']):
        self.burrow = burrow_size     # how big is the bunch in the burrow
        self.shape = bunny_shape      # number of nodes in layers A, B, and C (eyes, ears, and toes)
        self.glow = glow_rate         # mutation rate
        self.bunnies = []             # list of bunnies
        self.Populate()
    def Populate(self):
        while len(self.bunnies) < self.burrow:
            self.bunnies += [Bunny(eyes=self.shape[0], ears=self.shape[1], toes=self.shape[2], glow=self.glow)]

class Bunny:
    def __init__(self, eyes, ears, toes, glow=0.5, rand=True):
        self.hops = 0
        self.eyes = eyes
        self.ears = ears
        self.toes = toes
        self.nose = 1 << (self.eyes + self.toes)
        if rand:
            self.coat = R(1, 1 << self.Size())
        else:
            self.coat = 0
        self.glow = glow

    def Size(self):
        return len(self.eyes) * len(self.toes) + len(self.ears) * (len(self.ears) - 1) >> 1 + len(self.ears) * len(self.eyes + self.toes)

    def Molt(self):
        if R(0, 2) and self.coat >= 1 << (self.eyes).bit_length():
            self.ShortCoat()
        if R(0, 2): self.LongCoat()
        if R(0, 2): self.BrushCoat()

    def ShortCoat(self):
        return None
    
    def LongCoat(self):
        return None
    
    def BrushCoat(self):
        return None

    def Shed(self, shed=True):
        return None
    
    def Hop(self):
        NOSE_MASK() # WILL PROBABLY USE TRANSFORM() FROM BOTTOM

"""     OLD BAD CODE. NEW STUFF AT BOTTOM
        a1 =  1 * self.toes
        a2 = a1 + self.eyes
        b1 = a1 * self.eyes
        b2 = b1 + self.eyes
        for i in range(len(self.eyes)):
            mask_nose = ((1 << a2 - 1 << a1) & self.nose) >> a1
            mask_coat = ((1 << b2 - 1 << b1) & self.coat) >> b1
            self.nose += Activate(BinaryWeight(mask_nose & mask_coat)) * (1 << a2)
            a2 += 1
            b1  = self.toes + b2
            b2 += self.toes + self.eyes + 1
        b1 = 0
        b2 = 0 + self.eyes
        b3 = sum([1 << (self.eyes * (self.toes + 1) + (self.eyes + self.toes + 1) * i + i * (i - 1) >> 1) for i in range(self.ears)])
        for i in range(len(self.toes)):
            mask_nose =      ((1 << a2 - 1 << a1) & self.nose) >> a1
            mask_coat = ((b3 + 1 << b2 - 1 << b1) & self.coat) >> b1
            b1   = 0 + b2
            b2  += self.eyes
            b3 <<= 1
"""

def Int2Bin(n):
    return bin(n)[2:]

def Bin2Int(n):
    return int(n, 2)

def BinaryWeight(n):
    return (bin(n)).count('1')

def Activate(n):
    return None



L = [11,10,9,8,7,6,5,5,5,5,5,5,5,5,5,5,5,11,10,9,8,7,6,4,4,4,4,4,4,4,4,4,4,11,10,9,8,7,6,3,3,3,3,3,3,3,3,3,11,10,9,8,7,6,2,2,2,2,2,2,2,2,11,10,9,8,7,6,1,1,1,1,1,1,1,11,11,11,11,11,11,11,10,10,10,10,10,10,10,9,9,9,9,9,9,9,8,8,8,8,8,8,8,7,7,7,7,7,7,7,6,6,6,6,6,6,6]
A = 7
B = 5
C = 6

def Cycle(LIST, POS, LEN, MOVE_R):
    LIST[POS:POS + LEN] = LIST[POS:POS + LEN][-MOVE_R:] + LIST[POS:POS + LEN][:-MOVE_R]
    return LIST

def Transform(L, A, B, C):
    for i in range(B - 1):
        L = Cycle(L, i * C + C, A + B + C - 1 + i * 2 * B - (i * i - i) // 2, C)
    L = Cycle(L, B * C, A * (B + C) + B * (B - 1) // 2, A * C)
    for i in range(C - 1):
        for j in range(B - 1):
            L = Cycle(L, j + i * B + 1, C - i + (C - i - 1) * j, 1)
    for i in range(C - 1):
        L = Cycle(L, B + i * (A + B), A + B * (C - i - 1), A)
    return L

print(Transform(L, A, B, C))