from NICE import *

# create tests
input_nodes  = list(GenerateNodes('A', RULES['network_structure'][0]).keys())
tests = []
for i in range(4):

    # NAND
    tests += [{'inputs':{}, 'outputs':{}}]
    tests[-1]['inputs'][input_nodes[0]] = 1
    tests[-1]['inputs'][input_nodes[1]] = 0
    tests[-1]['inputs'][input_nodes[2]] = 0
    tests[-1]['inputs'][input_nodes[3]] = i % 2
    tests[-1]['inputs'][input_nodes[4]] = i // 2
    tests[-1]['outputs']['C0'] = (not (i % 2 and i // 2)) * 1

    # OR
    tests += [{'inputs':{}, 'outputs':{}}]
    tests[-1]['inputs'][input_nodes[0]] = 1
    tests[-1]['inputs'][input_nodes[1]] = 1
    tests[-1]['inputs'][input_nodes[2]] = 0
    tests[-1]['inputs'][input_nodes[3]] = i % 2
    tests[-1]['inputs'][input_nodes[4]] = i // 2
    tests[-1]['outputs']['C0'] = i % 2 or i // 2

    # XOR
    tests += [{'inputs':{}, 'outputs':{}}]
    tests[-1]['inputs'][input_nodes[0]] = 1
    tests[-1]['inputs'][input_nodes[1]] = 0
    tests[-1]['inputs'][input_nodes[2]] = 1
    tests[-1]['inputs'][input_nodes[3]] = i % 2
    tests[-1]['inputs'][input_nodes[4]] = i // 2
    tests[-1]['outputs']['C0'] = i % 2 ^ i // 2

    # AND
    tests += [{'inputs':{}, 'outputs':{}}]
    tests[-1]['inputs'][input_nodes[0]] = 1
    tests[-1]['inputs'][input_nodes[1]] = 1
    tests[-1]['inputs'][input_nodes[2]] = 1
    tests[-1]['inputs'][input_nodes[3]] = i % 2
    tests[-1]['inputs'][input_nodes[4]] = i // 2
    tests[-1]['outputs']['C0'] = (i % 2 and i // 2) * 1

max_fitness = 15
max_flag = 0

# define test function loop
def Test(NN):
    global max_flag
    local_flag = max_flag * 1
    for brain in NN:
        size = Size(brain)
        brain['fitness'] = 0
        for test in tests:
            for (node, val) in test['inputs'].items():
                brain['nodes'][node] = val
            brain = Propogate(brain)
            brain['fitness'] += 1 - Diff(test['outputs']['C0'], brain['nodes']['C0'])
        if brain['fitness'] > max_fitness and local_flag == 0 and max_flag == 0:
            local_flag = 1
        if max_flag == 1:
            brain['fitness'] /= (1 + size/100)
    if max_flag == 0: max_flag = local_flag * 1
    return NN

# populate
# NN = Populate(RULES['network_structure'], RULES['population'])
NN = []
while len(NN) < RULES['population']:
    NN += [Brain(RULES['network_structure'], RULES['randomize_networks'])]

# test, speciate, selection, repeat
gen = 0
for i in range(9999):
    NN = Selection(Speciate(Test(NN)), nomads=0.5)
    gen += 1
    NN.sort(key = lambda brain: brain['fitness'], reverse=True)
    f = NN[0]['fitness']
    if max_flag == 1:
        f *= (1 + Size(NN[0])/100)
    print(f'generation: {gen}   best fitness: {f:.4f}   brain size: {Size(NN[0])}')

# display best brain
NN = Test(NN)
NN = Speciate(NN)
NN.sort(key = lambda brain: brain['fitness'], reverse=True)
best_brain = NN[0]
print(best_brain)