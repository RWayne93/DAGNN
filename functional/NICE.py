from random import random as R
from random import choice as RC
from random import choices as RCS

# global variables
# NN is population of brains
# RULES are general construction and behavior settings for brains
NN = []
RULES = {}
RULES['population']         = 32           # brain population size
RULES['network_structure']  = [3, 1, 1]     # nodes in layers 'A', 'B', and 'C'
RULES['randomize_networks'] = True          # gen 0 brains are random or zero'd
RULES['node_space']         = 1             # digits length for each node
RULES['weights_space']      = [-1, 1]       # random value limits for weights
RULES['mutation_rate']      = .3            # each mutation occurance chance
RULES['activation_slopes']  = [0, 1]        # limits of f(x): m*x as x -> inf
RULES['num_parents']        = 2             # parents chosen for each child

# returns list of brains with length of population ->
#   [{<brain>}, {<brain>}, {<brain>}, ..., {<brain>}]
def Populate(layers=RULES['network_structure'], pop=RULES['population']):
    return [Brain(layers) for i in range(pop)]

# returns brain ->
#   {
#     'fitness' : 0,
#     'adjusted_fitness' : 0,
#     'species' : 0,
#     'nodes' : {'A000000':0, ..., 'A999999':0, ..., 'C000000':0, ...},
#     'links' : {{'A000000':{<other_node>:<weight>, ...}}, ...}
#   }
def Brain(layers=RULES['network_structure'], rand=RULES['randomize_networks']):
    brain = {}
    brain['fitness'], brain['adjusted_fitness'], brain['species'] = 0, 0, 0
    brain['nodes'] = GenerateNodes('A', layers[0])
    if rand:
        brain['nodes'] |= GenerateNodes('B', layers[1])
    brain['links'] = dict((node, {}) for node in brain['nodes'])
    brain['nodes'] |= GenerateNodes('C', layers[2])
    brain['nodes']['A'+GenerateID(0)] = 1
    if rand:
        p = 1 / (layers[1] * layers[2])
        for a, b in GetLinks(brain)['free']:
            if R() < p: brain['links'][a][b] = RandomWeight()
    return Sort(brain)

# returns nodes for layer
# GenerateNodes('A', 5) =>
#   {'A000000':0, 'A249999':0, 'A49999':0, 'A749999':0, 'A999999':0}
def GenerateNodes(layer, num_nodes):
    nodes = {}
    for i in range(num_nodes):
        node = '' if layer == 'B' else i / (num_nodes - 1 + Space())
        nodes[layer + GenerateID(node)] = 0
    return nodes

# return node_id string
# GenerateID() =>
#   random fixed length string of digits, ex: '091052'
# GenerateID(0.33) =>
#   '330000'
# GenerateID(('B123456', 'B987654')) =>
#   random node in layer 'B' between nodes, ex: 'B209756'
def GenerateID(node=''):
    if type(node) == type(tuple()):
        a, b = node
        a = 0 if a[0] == 'A' else float('.' + a[1:]) + Space()
        b = 1 if b[0] == 'C' else float('.' + b[1:])
        return 'B' + Fix(R() * (b - a) + a)
    if node == '': node = R()
    return Fix(node % 1)

# returns random weight between rule values
def RandomWeight():
    a, b = RULES['weights_space']
    return R() * (b - a) + a

# returns 'set' and 'free' links as lists of tuple pairs ->
# {
#   'set' : [
#     (<node>, <node>),
#     ...,
#     (<node>, <node>)
#   ],
#   'free': [
#     (<node>, <node>),
#     ...,
#     (<node>, <node>)
#   ]
# }
def GetLinks(brain):
    links, bl, bn = {}, brain['links'], brain['nodes']
    links['set'] = [(a, b) for a in bl for b in bl[a]]
    links['free'] = [(a, b) for a in bn if a[0] != 'C' for b in bn
      if b[0] != 'A' and a < b and (a, b) not in links['set']]
    return links

# return node to formated space
# Fix(0.25) =>
#   '250000'
def Fix(n):
    return '{:.17f}'.format(n)[2 : 2 + RULES['node_space']]

# return brain with nodes and links sorted alphanumerically
def Sort(brain):
    brain['nodes'] = dict(sorted([node for node in brain['nodes'].items()]))
    brain['links'] = dict(sorted([links for links in brain['links'].items()]))
    return brain

# return 10^-x, where x is the node space
def Space():
    return float(f'1e-{RULES['node_space']}')

# return brain with mutations
def Mutate(brain, prune=False, p=RULES['mutation_rate']):
    return Prune(MutateLinks(MutateNodes(brain, p), p), prune, p)

# return brain with mutated nodes of layer 'B'
def MutateNodes(brain, p=RULES['mutation_rate']):
    B = [node for node in brain['nodes'] if node[0] == 'B']
    if R() < p and len(B) > 0: brain, B = DeleteNode(brain)
    if R() < p and len(B) > 1: brain = MoveNode(brain)
    if R() < p: brain = AddNode(brain)
    return Sort(brain)

# return brain and layer 'B' wth a randomly deleted node from layer 'B'
def DeleteNode(brain):
    b = RC([node for node in brain['nodes'] if node[0] == 'B'])
    brain['nodes'].pop(b, '')
    brain['links'].pop(b, '')
    for a in brain['links']:
        brain['links'][a].pop(b, '')
    return brain, [node for node in brain['nodes'] if node[0] == 'B']

# return brain with a randomly moved node in layer 'B'
def MoveNode(brain):
    a = RC([node for node in brain['nodes'] if node[0] == 'B'])
    b, z = 'B' + GenerateID(), []
    while b in brain['nodes']: b = 'B' + GenerateID()
    brain = eval(b.join(str(brain).split(a)))
    for a, b in GetLinks(brain)['set']:
        if a[0] == 'B' and b[0] == 'B' and a > b:
            brain['links'][b][a] = brain['links'][a][b]
            z += [(a, b)]
    for a, b in z:
        brain['links'][a].pop(b, '')
    return brain

# return brain with a randomly added node
def AddNode(brain):
    node = 'B' + GenerateID()
    brain['nodes'][node], brain['links'][node] = 0, {}
    return brain

# return brain with mutated links
def MutateLinks(brain, p=RULES['mutation_rate']):
    links = GetLinks(brain)
    x, y = links['set'], links['free']
    if R() < p and len(x) > 1: brain, x, y = DeleteLink(brain, x, y)
    if R() < p and len(x) > 0: brain, x, y = ChangeLink(brain, x, y)
    if R() < p and len(x) > 0 < len(y): brain, x, y = MoveLink(brain, x, y)
    if R() < p and 0 < len(y): brain, x = AddLink(brain, x, y)
    if R() < p and len(x) > 0: brain = SplitLink(brain, x)
    return Sort(brain)

# return brain with randomly deleted link
def DeleteLink(brain, x, y):
    a, b = RC(x)
    brain['links'][a].pop(b)
    y += [(a, b)]
    x = [links for links in x if links != (a, b)]
    return brain, x, y

# return brain with randomly changed link weight
def ChangeLink(brain, x, y):
    (a, b), rw = RC(x), RandomWeight()
    brain['links'][a][b] = rw
    return brain, x, y

# return brain with randomly moved link
def MoveLink(brain, x, y):
    (a, b), (i, j) = RC(x), RC(y)
    brain['links'][i][j] = brain['links'][a][b]
    brain['links'][a].pop(b)
    x += [(i, j)]
    y += [(a, b)]
    x = [links for links in x if links != (a, b)]
    y = [links for links in y if links != (i, j)]
    return brain, x, y

# return brain with randomly added link
def AddLink(brain, x, y):
    (i, j), rw = RC(y), RandomWeight()
    brain['links'][i][j] = rw
    x += [(i, j)]
    y = [links for links in y if links != (i, j)]
    return brain, x

# return brain with randomly split link
def SplitLink(brain, x):
    (a, b) = RC(x)
    node = GenerateID((a, b))
    if (node not in brain['nodes'] and node not in (a, b)):
        rw_1, rw_2 = RandomWeight(), RandomWeight()
        brain['links'][a].pop(b)
        brain['links'][a][node] = rw_1
        brain['nodes'][node] = 0
        brain['links'][node] = {b:rw_2}
    return brain

# return brain with pruning
def Prune(brain, prune=False, p=RULES['mutation_rate']):
    bn, bl, x = brain['nodes'], brain['links'], GetLinks(brain)['set']
    if (R() < p and len(x) > 1) or prune:
        for node in bn:
            bn[node] = {'A':2, 'B':1, 'C':3}[node[0]]
        x.sort(key = lambda a : a[0])
        for a, b in x:
            if bn[a] % 2 == 0 and bn[b] % 2 != 0 and bl[a][b] != 0: bn[b] *= 2
        x.sort(key = lambda a : a[1], reverse = True)
        for a, b in x:
            if bn[b] % 3 == 0 and bn[a] % 3 != 0 and bl[a][b] != 0: bn[a] *= 3
        z = [node for node in bn if node[0] == 'B' and bn[node] % 6 != 0]
        for node in z:
            bn.pop(node, '')
            bl.pop(node, '')
        for (a, b) in bl.items():
            bl[a] = dict([(i, j) for (i, j) in b.items()
              if i not in z and j != 0])
    return brain

# return brain with node values in layers 'B' and 'C' computed from layer 'A'
def Propogate(brain):
    bn, bl, layers = brain['nodes'], brain['links'], {'A':[], 'B':[], 'C':[]}
    for node in bn:
        if node[0] != 'A': bn[node] = 0
        layers[node[0]] += [node]
    A, B, C = layers['A'], layers['B'], layers['C']
    for node in A:
        for link in bl[node]:
            bn[link] += bn[node] * bl[node][link]
    for node in B:
        bn[node] = Activate(bn[node])
        for link in bl[node]:
            bn[link] += bn[node] * bl[node][link]
    for node in C:
        bn[node] = Activate(bn[node])
    return brain

# activation function
def Activate(x):
    a = RULES['activation_slopes'][1] + RULES['activation_slopes'][0]
    b = RULES['activation_slopes'][1] - RULES['activation_slopes'][0]
    return x * (a + b * Tanh(x)) / 2

# hyperbolic tangent function
def Tanh(x):
    return (Exp(x) - Exp(-x)) / (Exp(x) + Exp(-x))

# exponential function
def Exp(x):
    return 2.71828182845904524 ** x

# difference squared function
def Diff(x, y):
    return (x - y) ** 2

# group brains into species and adjust fitness
def Speciate(NN):
    return Adjust(Group(NN))

# group brain species by size
def Group(NN):
    NN.sort(key = lambda x : Size(x))
    min_size, max_size = Size(NN[0]), Size(NN[-1])
    dif_size = max_size - min_size
    dividers = [0] * int((len(NN) ** .5) // 1)
    for i in range(len(dividers)):
        dividers[i] = min_size + dif_size * (i + 1) / len(dividers)
    i = 0
    for brain in NN:
        while (Size(brain) > dividers[i]):
            i += 1
        brain['species'] = i * 1
    return NN

# return brain total links
def Size(brain):
    return sum([len(links) for links in brain['links'].values()])

# adjust brain fitness by species
def Adjust(NN):
    NN = NormalizeFitness(NN)
    species = {}
    for brain in NN:
        group = brain['species']
        if group not in species:
            species[group] = {'count':1, 'fitness':brain['fitness']}
        else:
            species[group]['count'] += 1
            species[group]['fitness'] += brain['fitness']
    total_species = len(species)
    total_fitness = sum([species[group]['fitness'] for group in species])
    for brain in NN:
        if brain['fitness'] == 0:
            brain['adjusted_fitness'] = 0
        else:
            group = brain['species']
            x = 1 / (species[group]['fitness'] * total_species) + 1 / total_fitness
            brain['adjusted_fitness'] = brain['fitness'] * x / 2
    return NN

# normalize fitness scores
def NormalizeFitness(NN):
    min_fit, max_fit = None, None
    for brain in NN:
        if min_fit == None or brain['fitness'] < min_fit:
            min_fit = brain['fitness']
        if max_fit == None or brain['fitness'] > max_fit:
            max_fit = brain['fitness']
    dif_fit = max_fit - min_fit
    for brain in NN:
        if brain['fitness'] != 0 and dif_fit != 0:
            brain['fitness'] = (brain['fitness'] - min_fit) * max_fit / dif_fit
    return NN

# create next generation
def Selection(NN):
    NN.sort(key = lambda brain: brain['fitness'], reverse=True)
    NEW, fitness = [NN[0]], [brain['adjusted_fitness'] for brain in NN]
    while len(NEW) < len(NN):
        parents = RCS(population=NN, weights=fitness, k=RULES['num_parents'])
        NEW += [Crossover(parents)]
    return NEW[:]

# return child with random weights from parents
def Crossover(parents):
    child = Brain(layers=RULES['network_structure'], rand=False)
    for parent in parents:
        child = Imprint(parent, child)
    for a in child['links']:
        for b in child['links'][a]:
            child['links'][a][b] += [0] * len(parents)
            child['links'][a][b] = RC(child['links'][a][b][:len(parents)])
    x = GetLinks(child)['set']
    for a, b in x:
        if a[0] == 'B' and a not in child['nodes']: child['nodes'][a] = 1
        if b[0] == 'B' and b not in child['nodes']: child['nodes'][b] = 1
    for node in child['nodes']:
        if node not in child['links'] and node[0] != 'C':
            child['links'][node] = {}
    return Prune(Mutate(child), True)

# return child with imprinted brain
def Imprint(parent, child):
    x = GetLinks(parent)['set']
    for a, b in x:
        if a not in child['links']:
            child['links'][a] = {b:[parent['links'][a][b]]}
        elif b not in child['links'][a]:
            child['links'][a][b] = [parent['links'][a][b]]
        else:
            child['links'][a][b] += [parent['links'][a][b]]
    return child

# print brain neural network representation
def Visualize(brain):
    return 0