
from DAGNN import *
from math import sin, cos, pi
from random import *

# initialize and learning
C = 3
NN = DAGNN({'A':3, 'B':9, 'C': C})

def parametric(t, r):
    return t * cos(pi*(3*t/2+2*r/C)), t * sin(pi*(2*t/2+2*r/C))

def vectorize(index, size):
    my_list = [0] * size
    my_list[index] = 1
    return my_list

NN.unit_tests = []
for i in range(50):
    classification = randint(0, C - 1)
    class_vector = vectorize(classification, C)
    NN.unit_tests += [((1,) + parametric(.5 + random()*.5, classification), class_vector)]

NN.num_tests = len(NN.unit_tests) * C

NN.Learn()
