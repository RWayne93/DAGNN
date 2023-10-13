from NICEClass import NICE
from multiprocessing import Pool

Brain = NICE()

# create tests
input_nodes  = list(Brain.GenerateNodes('A', Brain.RULES['network_structure'][0]))
tests = []


for i in range(4):
    tests += [{'inputs':{}, 'outputs':{}}]
    tests[i]['inputs'][input_nodes[0]] = 1
    tests[i]['inputs'][input_nodes[1]] = i % 2
    tests[i]['inputs'][input_nodes[2]] = i // 2
    tests[i]['outputs']['C0'] = (i % 3 == 0) * 1

print(tests)

# # define test function loop
def Test(NN):
    for brain in NN:
        size = Brain.Size(brain)
        brain['fitness'] = 0
        for test in range(len(tests)):
            rand_test = tests[test]
            for (node, val) in rand_test['inputs'].items():
                brain['nodes'][node] = val
            brain = Brain.Propogate(brain)
            d = 1 - Brain.Diff(rand_test['outputs']['C0'], brain['nodes']['C0'])
            e = (round(brain['nodes']['C0']) == rand_test['outputs']['C0']) * 1
            brain['fitness'] += (d + e) / 2
            if size < 3: brain['fitness'] = 0
        if size != 0: brain['fitness'] /= size
    return NN

# # populate
Brain.NN = Brain.Populate(Brain.RULES['network_structure'], Brain.RULES['population'])
#print(population)
# test, speciate, selection, repeat
for i in range(9999):
    Brain.NN = Brain.Selection(Brain.Speciate(Test(Brain.NN)))
    print(f'generation {i} {Brain.NN[0]['fitness']}')

print(Brain.NN[0])
print(Brain.Size(Brain.NN[0]))

# def test_brain(brain):
#     return Test([brain])[0]

# if __name__ == '__main__':
#     with Pool() as p:
#         for i in range(15000):
#             Brain.NN = p.map(test_brain, Brain.NN)
#             Brain.NN = Brain.Selection(Brain.Speciate(Brain.NN))
#             print(f'generation {i} {Brain.NN[0]['fitness']}')

# print(Brain.NN[0])
# print(Brain.Size(Brain.NN[0]))
