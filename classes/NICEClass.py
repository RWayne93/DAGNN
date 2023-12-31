from random import random as R
from random import choice as RC
from random import choices as RCS

class NICE:
    def __init__(self):
        self.RULES = {}
        self.RULES['population']         = 32            # brain population size
        self.RULES['network_structure']  = [3, 1, 1]     # nodes in layers 'A', 'B', and 'C'
        self.RULES['randomize_networks'] = True          # gen 0 brains are random or zero'd
        self.RULES['node_space']         = 1             # digits length for each node
        self.RULES['weights_space']      = [0, 4]        # random value limits for weights
        self.RULES['mutation_rate']      = .3            # each mutation occurance chance
        self.RULES['activation_slopes']  = [0, 1]        # limits of f(x): m*x as x -> inf
        self.RULES['num_parents']        = 2             # parents chosen for each child
        self.NN = []

    def Populate(self, layers=None, pop=None):
        if layers is None:
            layers = self.RULES['network_structure']
        if pop is None:
            pop = self.RULES['population']
        return [self.Brain(layers) for i in range(pop)]

    def Brain(self, layers=None, rand=None):
        if layers is None:
            layers = self.RULES['network_structure']
        if rand is None:
            rand = self.RULES['randomize_networks']
        brain = {}
        brain['fitness'], brain['adjusted_fitness'], brain['species'] = 0, 0, 0
        brain['nodes'] = self.GenerateNodes('A', layers[0])
        if rand:
            brain['nodes'] |= self.GenerateNodes('B', layers[1])
        brain['links'] = dict((node, {}) for node in brain['nodes'])
        brain['nodes'] |= self.GenerateNodes('C', layers[2])
        brain['nodes']['A'+self.GenerateID(0)] = 1
        if rand:
            p = 1 / (layers[1] * layers[2])
            for a, b in self.GetLinks(brain)['free']:
                if R() < p: brain['links'][a][b] = self.RandomWeight()
        return self.Sort(brain)

    def GenerateNodes(self, layer, num_nodes):
        nodes = {}
        for i in range(num_nodes):
            node = '' if layer == 'B' else i / (num_nodes - 1 + self.Space())
            nodes[layer + self.GenerateID(node)] = 0
        return nodes

    def GenerateID(self, node=''):
        if isinstance(node, tuple):
            a, b = node
            a = 0 if a[0] == 'A' else float('.' + a[1:]) + self.Space()
            b = 1 if b[0] == 'C' else float('.' + b[1:])
            return 'B' + self.Fix(R() * (b - a) + a)
        if node == '': node = R()
        return self.Fix(node % 1)

    def RandomWeight(self):
        a, b = self.RULES['weights_space']
        return R() * (b - a) + a

    def GetLinks(self, brain):
        links, bl, bn = {}, brain['links'], brain['nodes']
        links['set'] = [(a, b) for a in bl for b in bl[a]]
        links['free'] = [(a, b) for a in bn if a[0] != 'C' for b in bn
          if b[0] != 'A' and a < b and (a, b) not in links['set']]
        return links

    def Fix(self, n):
        return '{:.17f}'.format(n)[2 : 2 + self.RULES['node_space']]

    def Sort(self, brain):
        brain['nodes'] = dict(sorted([node for node in brain['nodes'].items()]))
        brain['links'] = dict(sorted([links for links in brain['links'].items()]))
        return brain

    def Space(self):
        return float(f'1e-{self.RULES['node_space']}')

    def Mutate(self, brain, prune=False, p=None):
        if p is None:
            p = self.RULES['mutation_rate']
        return self.Prune(self.MutateLinks(self.MutateNodes(brain, p), p), prune, p)

    def MutateNodes(self, brain, p=None):
        if p is None:
            p = self.RULES['mutation_rate']
        B = [node for node in brain['nodes'] if node[0] == 'B']
        if R() < p and len(B) > 0: 
            brain, B = self.DeleteNode(brain)
        if R() < p and len(B) > 1: 
            brain = self.MoveNode(brain)
        if R() < p: 
            brain = self.AddNode(brain)
        return self.Sort(brain)

    def DeleteNode(self, brain):
        b = RC([node for node in brain['nodes'] if node[0] == 'B'])
        brain['nodes'].pop(b, '')
        brain['links'].pop(b, '')
        for a in brain['links']:
            brain['links'][a].pop(b, '')
        return brain, [node for node in brain['nodes'] if node[0] == 'B']

    def MoveNode(self, brain):
        a = RC([node for node in brain['nodes'] if node[0] == 'B'])
        b, z = 'B' + self.GenerateID(), []
        while b in brain['nodes']: b = 'B' + self.GenerateID()
        brain = eval(b.join(str(brain).split(a)))
        for a, b in self.GetLinks(brain)['set']:
            if a[0] == 'B' and b[0] == 'B' and a > b:
                brain['links'][b][a] = brain['links'][a][b]
                z += [(a, b)]
        for a, b in z:
            brain['links'][a].pop(b, '')
        return brain

    def AddNode(self, brain):
        node = 'B' + self.GenerateID()
        while node in brain['nodes']:
            node = 'B' + self.GenerateID()
        brain['nodes'][node], brain['links'][node] = 0, {}
        return brain

    def MutateLinks(self, brain, p=None):
        if p is None:
            p = self.RULES['mutation_rate']
        links = self.GetLinks(brain)
        x, y = links['set'], links['free']
        if R() < p and len(x) > 1: brain, x, y = self.DeleteLink(brain, x, y)
        if R() < p and len(x) > 0: brain, x, y = self.ChangeLink(brain, x, y)
        if R() < p and len(x) > 0 < len(y): brain, x, y = self.MoveLink(brain, x, y)
        if R() < p and 0 < len(y): brain, x = self.AddLink(brain, x, y)
        if R() < p and len(x) > 0: brain = self.SplitLink(brain, x)
        return self.Sort(brain)

    def DeleteLink(self, brain, x, y):
        a, b = RC(x)
        brain['links'][a].pop(b)
        y += [(a, b)]
        x = [links for links in x if links != (a, b)]
        return brain, x, y

    def ChangeLink(self, brain, x, y):
        (a, b), rw = RC(x), self.RandomWeight()
        brain['links'][a][b] = rw
        return brain, x, y

    def MoveLink(self, brain, x, y):
        (a, b), (i, j) = RC(x), RC(y)
        brain['links'][i][j] = brain['links'][a][b]
        brain['links'][a].pop(b)
        x += [(i, j)]
        y += [(a, b)]
        x = [links for links in x if links != (a, b)]
        y = [links for links in y if links != (i, j)]
        return brain, x, y

    def AddLink(self, brain, x, y):
        (i, j), rw = RC(y), self.RandomWeight()
        brain['links'][i][j] = rw
        x += [(i, j)]
        y = [links for links in y if links != (i, j)]
        return brain, x

    def SplitLink(self, brain, x):
        (a, b) = RC(x)
        node = self.GenerateID((a, b))
        if (node not in brain['nodes'] and node not in (a, b)):
            rw_1, rw_2 = self.RandomWeight(), self.RandomWeight()
            brain['links'][a].pop(b)
            brain['links'][a][node] = rw_1
            brain['nodes'][node] = 0
            brain['links'][node] = {b:rw_2}
        return brain

    def Prune(self, brain, prune=False, p=None):
        if p is None:
            p = self.RULES['mutation_rate']
        bn, bl, x = brain['nodes'], brain['links'], self.GetLinks(brain)['set']
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

    def Propogate(self, brain):
        bn, bl, layers = brain['nodes'], brain['links'], {'A':[], 'B':[], 'C':[]}
        for node in bn:
            if node[0] != 'A': bn[node] = 0
            layers[node[0]] += [node]
        A, B, C = layers['A'], layers['B'], layers['C']
        for node in A:
            for link in bl[node]:
                bn[link] += bn[node] * bl[node][link]
        for node in B:
            bn[node] = self.Activate(bn[node])
            for link in bl[node]:
                bn[link] += bn[node] * bl[node][link]
        for node in C:
            bn[node] = self.Activate(bn[node])
        return brain

    def Activate(self, x):
        return x - abs(x) + abs(x-1)

    def Diff(self, x, y):
        return (x - y) ** 2

    def Speciate(self, NN):
        return self.Adjust(self.Group(NN))

    def Group(self, NN):
        NN.sort(key = lambda x : self.Size(x))
        min_size, max_size = self.Size(NN[0]), self.Size(NN[-1])
        dif_size = max_size - min_size
        dividers = [0] * int((len(NN) ** .5) // 1)
        for i in range(len(dividers)):
            dividers[i] = min_size + dif_size * (i + 1) / len(dividers)
        i = 0
        for brain in NN:
            while (self.Size(brain) > dividers[i]):
                i += 1
            brain['species'] = i * 1
        return NN

    def Size(self, brain):
        return sum([len(links) for links in brain['links'].values()])

    def Adjust(self, NN):
            NN = self.NormalizeFitness(NN)
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

    def NormalizeFitness(self, NN):
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

    def Selection(self, NN, nomads=0.5):
        self.NN.sort(key = lambda brain: brain['fitness'], reverse=True)
        NEW, fitness = [NN[0]], [brain['adjusted_fitness'] for brain in NN]
        while len(NEW) < len(NN) * (1 - nomads):
            parents = RCS(population=NN, weights=fitness, k=self.RULES['num_parents'])
            NEW += [self.Crossover(parents)]
        while len(NEW) < len(NN):
            NEW += [self.Brain()]
        return NEW[:]

    def Crossover(self, parents):
        child = self.Brain(layers=self.RULES['network_structure'], rand=False)
        for parent in parents:
            child = self.Imprint(parent, child)
        for a in child['links']:
            for b in child['links'][a]:
                child['links'][a][b] += [0] * len(parents)
                child['links'][a][b] = RC(child['links'][a][b][:len(parents)])
        x = self.GetLinks(child)['set']
        for a, b in x:
            if a[0] == 'B' and a not in child['nodes']: child['nodes'][a] = 1
            if b[0] == 'B' and b not in child['nodes']: child['nodes'][b] = 1
        for node in child['nodes']:
            if node not in child['links'] and node[0] != 'C':
                child['links'][node] = {}
        return self.Prune(self.Mutate(child), True)

    def Imprint(self, parent, child):
        x = self.GetLinks(parent)['set']
        for a, b in x:
            if a not in child['links']:
                child['links'][a] = {b:[parent['links'][a][b]]}
            elif b not in child['links'][a]:
                child['links'][a][b] = [parent['links'][a][b]]
            else:
                child['links'][a][b] += [parent['links'][a][b]]
        return child

    def Visualize(self, brain):
        return 0
    