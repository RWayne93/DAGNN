from NICE import *

# create tests
input_nodes  = list(GenerateNodes('A', RULES['network_structure'][0]).keys())
tests = []
for i in range(4):
    tests += [{'inputs':{}, 'outputs':{}}]
    tests[i]['inputs'][input_nodes[0]] = 1
    tests[i]['inputs'][input_nodes[1]] = i % 2
    tests[i]['inputs'][input_nodes[2]] = i // 2
    tests[i]['outputs']['C0'] = (i % 3 == 0) * 1

# define test function loop
def Test(NN):
    for brain in NN:
        size = Size(brain)
        brain['fitness'] = 0
        for test in range(len(tests)):
            rand_test = tests[test]
            for (node, val) in rand_test['inputs'].items():
                brain['nodes'][node] = val
            brain = Propogate(brain)
            d = Diff(rand_test['outputs']['C0'], brain['nodes']['C0'])
            score = 1 if d < .25 else 1 - d
            if size < 5: score = 0
            brain['fitness'] += score
        if size != 0: brain['fitness'] /= size
    return NN

# populate
NN = Populate(RULES['network_structure'], RULES['population'])

# test, speciate, selection, repeat
for i in range(100):
    NN = Selection(Speciate(Test(NN)))

NN = Test(NN)
NN = Speciate(NN)
NN.sort(key = lambda brain: brain['fitness'], reverse=True)
for brain in NN:
    size = Size(brain)
    brain['fitness'] = '{:.2f}'.format(brain['fitness'] * size)

best_brain = NN[0]
print(best_brain)
