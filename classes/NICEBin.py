from random import random as R
import random

class NICEClass:
    def __init__(self):
        self.RULES = {}
        self.RULES['population']         = 32           # brain population size
        self.RULES['network_structure']  = [3, 1, 1]     # nodes in layers 'A', 'B', and 'C'
        self.RULES['randomize_networks'] = True          # gen 0 brains are random or zero'd
        self.RULES['node_space']         = 1             # digits length for each node
        self.RULES['weights_space']      = [-1, 1]       # random value limits for weights
        self.RULES['mutation_rate']      = .3            # each mutation occurance chance
        self.RULES['activation_slopes']  = [0, 1]        # limits of f(x): m*x as x -> inf
        self.RULES['num_parents']        = 2             # parents chosen for each child
        self.NN = []

    def Populate(self, layers=None, pop=None):
        if layers is None:
            layers = self.RULES.get('network_structure', [3, 1, 1])
        if pop is None:
            pop = self.RULES['population']
        return [self.Brain(layers) for i in range(pop)]    

    def Brain(self, layers=None, rand=None):
        # Retrieve layers safely and debug print
        if layers is None:
            layers = self.RULES.get('network_structure', [3, 1, 1])
       #print(f"Debug: layers = {layers}")  # Debugging print. Comment this out later.
        
        # Robustness check for layers
        if not isinstance(layers, list) or len(layers) < 3:
            raise ValueError("The 'layers' value must be a list with at least three elements.")
            
        # Continue as before...
        brain = {}
        brain['fitness'], brain['adjusted_fitness'], brain['species'] = 0, 0, 0
        brain['nodes'] = self.GenerateNodesBinary('A', layers[0])
        if rand:
            brain['nodes'] += self.GenerateNodesBinary('B', layers[1])
        brain['links'] = ''  # Initialize as empty binary string
        brain['nodes'] += self.GenerateNodesBinary('C', layers[2])
        # Append a fixed binary string to indicate a specific node, if necessary
        brain['nodes'] += '1' + '0' * (self.RULES['node_space'] - 1)
        if rand:
            p = 1 / (layers[1] * layers[2])
            for i in range(layers[1] * layers[2]):
                if R() < p:
                    brain['links'] += '1'
                else:
                    brain['links'] += '0'
        return self.Sort(brain) 
    
    def Sort(self, brain):
        brain['nodes'] = ''.join(sorted(brain['nodes']))
        brain['links'] = ''.join(sorted(brain['links']))
        return brain

    def GenerateNodesBinary(self, layer, count):
        return '1' * count + '0' * (self.RULES['node_space'] - count)

    # The RandomWeight function will need to return a binary representation
    def RandomWeight(self):
        min_w, max_w = self.RULES['weights_space']
        val = random.uniform(min_w, max_w)
        return self.FloatToBinary(val)

    def FloatToBinary(self, value):
        # Convert float to binary. You may want to set a fixed precision.
        # This is a placeholder and can be refined.
        if value >= 0:
            return '1' + format(int(value * (2 ** (self.RULES['node_space'] - 1))), '0' + str(self.RULES['node_space'] - 1) + 'b')
        else:
            return '0' + format(int(-value * (2 ** (self.RULES['node_space'] - 1))), '0' + str(self.RULES['node_space'] - 1) + 'b')
        
brains = NICEClass()
print(brains.Brain())
print(brains.Populate())
