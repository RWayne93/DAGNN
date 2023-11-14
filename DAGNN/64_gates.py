
from DAGNN import *

# initialize and learning
NN = DAGNN({'A':7, 'B':3, 'C': 1})
NN.unit_tests = [([int(i) for i in bin(n)[2:]], [int(bin(n)[2:][1:5][n % 4])]) for n in range(64, 128)]
NN.num_tests = len(NN.unit_tests)
NN.Learn()
