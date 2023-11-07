import numpy as np
from numpy import tanh, log

class STAG:
    def __init__(self, input_size, hidden_size, output_size):
        self.A = np.array(['A' + str(i).zfill(len(str(input_size - 1))) for i in range(input_size)]) # input node ids
        self.B = np.array(['B' + str(i).zfill(len(str(hidden_size - 1))) for i in range(hidden_size)]) # hidden node ids
        self.C = np.array(['C' + str(i).zfill(len(str(output_size - 1))) for i in range(output_size)]) # output node ids
        self.E = 0 # error
        self.F = 0 # fitness / score on unit-tests
        self.G = 9999 # max generations
        self.I = {} # unit tests / input values (keep as dict for variable keys)
        self.L = [] # list of links (keep as list for mixed type entries)
        self.N = {} # node data / network (needs to hold various property dicts)
        self.O = {} # output nodes (keep as dict for variable keys)
        self.P = {} # unit tests / predicted values of output nodes (keep as dict)
        self.R = 0.005 # learning rate
        self.T = 1.5 # threshold of error
        self.U = [] # unit tests (keep as list for test pairs)
        self.W = (-1, 1) # initial domain of random weights
        self.Z = True # prune while True
        
        # Create network with numpy
        node_ids = np.concatenate((self.B, self.C))
        for node_1 in np.concatenate((self.A, self.B)):
            self.N[node_1] = {'value': 0, 'links': {}}
            for node_2 in node_ids[node_ids > node_1]:
                self.N[node_1]['links'][node_2] = np.random.uniform(self.W[0], self.W[1])
                self.L.append((node_1, node_2))
        
        for node in self.C:
            self.N[node] = {'value': 0}
        self.L = tuple(self.L)
        self.S = len(self.L)

    # update links and network size
    def UpdateLinks(self):
        # Use NumPy array for efficient computation and slicing
        nodes = np.array([node for node in self.N if node[0] != 'C'])
        self.L = np.array([], dtype='O')  # Create an empty object array for flexibility

        for node_1 in nodes:
            linked_nodes = self.N[node_1]['links']
            for node_2 in linked_nodes:
                self.L = np.append(self.L, (node_1, node_2))

        # Use numpy lexsort for sorting by multiple keys
        indices = np.lexsort((self.L[:, 1], self.L[:, 0]))
        self.L = self.L[indices]
        self.L = tuple(map(tuple, self.L))  # Convert back to a tuple of tuples
        self.S = len(self.L)

    # propogate input node values through the network to compute output node values
    def Forward(self):
        def Activate(x):
            return x * tanh(x) / 2  # This operation is already using numpy for tanh

        # Set input nodes using numpy vectorization
        for node in self.A:
            self.N[node]['value'] = self.I.get(node, 0)  # Default to zero if not in test inputs

        # Reset hidden and output nodes values
        for node in np.concatenate((self.B, self.C)):
            self.N[node]['value'] = 0
        
        # Propagation through the network
        for (node_1, node_2) in self.L:
            if node_1[0] != 'A':
                self.N[node_1]['value'] = Activate(self.N[node_1]['value'])
            self.N[node_2]['value'] += self.N[node_1]['value'] * self.N[node_1]['links'][node_2]

        # Activate output nodes
        for node in self.C:
            self.N[node]['value'] = Activate(self.N[node]['value'])

        # Output and error computation
        self.E = 0  # Reset error
        for node in self.P:  # Assuming self.P is a dict of expected values
            error = abs(self.N[node]['value'] - self.P[node])
            self.E += (error + 1) * log(error + 1) - (error + 1) + 1
            self.F += error < self.T  # Fitness calculation

    # compute error from passing test through network
    def Test(self):
        # Vectorized test operation if possible
        # However, this depends on the nature of Forward method and self.U structure
        # Assuming self.U is a list of tuples
        for inputs, outputs in self.U:
            self.I = inputs
            self.P = outputs
            self.Forward()

    # machine learning for network
    def Learn(self, minimize_error=False):
        initial_R = self.R + 0.0  # Ensure it's a float for safety in division operations

        gen = 0
        self.E, self.F = 0, 0
        self.Test()

        again = self.F < len(self.U)
        if minimize_error:
            print('\n :: minimizing error ::\n')
            again = self.F == len(self.U) and self.E > self.T
        else:
            print('\n :: learning ::\n')

        while gen < self.G and again:
            current_error = self.E + 0.0

            # Use NumPy's random choice for selection
            self.Q = {}
            link_indices = np.random.choice(len(self.L), size=int((self.S ** .4) // 1), replace=False)
            for index in link_indices:
                link = self.L[index]
                self.Q[link] = 0.0

            for Q in self.Q:
                node_1, node_2 = Q
                self.N[node_1]['links'][node_2] += self.R
                self.Test()
                self.Q[Q] = (current_error - self.E) / self.R
                self.N[node_1]['links'][node_2] -= self.R

            # Use NumPy for sorting Q links by error difference magnitude
            self.Q = np.array(list(self.Q.items()))
            errors = np.abs(self.Q[:, 1].astype(float))  # Make sure error values are floats
            indices = np.argsort(-errors)  # Sort by error in descending order
            self.Q = self.Q[indices]

            # Apply the update with largest gradient
            (node_1, node_2), err = self.Q[0]
            self.N[node_1]['links'][node_2] += self.R * np.sign(-err)

            # Output every 100th generation
            if gen % 100 == 0:
                print(f'gen: {gen}    error: {self.E:0.4f}    score: {self.F} / {len(self.U)}')

            gen += 1
            self.Test()

            # Adjust learning rate based on error improvement
            if self.E < current_error:
                self.R *= 1.1
            else:
                self.R = max(self.R / 1.1, initial_R)

            again = self.F < len(self.U) if not minimize_error else self.F == len(self.U) and self.E > self.T

        self.R = initial_R  # Reset learning rate to initial value

        # Final output after learning
        print(f'gen: {gen}    error: {self.E:0.4f}    score: {self.F} / {len(self.U)}')
        if gen == self.G:
            print('\n :: max generations reached ::\n')
        else:
            if minimize_error:
                print('\n :: minimizing complete ::\n')
            else:
                print('\n :: learning complete ::\n')
            print(self.N)


    def Prune(self):
        # Reduce the number of generations for learning during pruning
        original_G = self.G
        self.G //= 10

        while self.Z and self.L:
            print('\n :: pruning ::')

            # If there are no links, end pruning
            if not self.L:  # Check if L is empty or not
                print('No links to prune.')
                break

            # Save current links and network configuration
            self.X = str(self.L)  # Ensure this is a string representation of the tuple
            self.Y = str(self.N)

            # Since self.L is a tuple, convert it to a list for processing
            links_list = list(self.L)

            # Calculate the weights of the links to find the smallest
            link_weights = [abs(self.N[node_1]['links'][node_2]) for node_1, node_2 in links_list]

            # Check if link weights exist before proceeding
            if not link_weights: 
                print('No link weights to evaluate, pruning cannot proceed.')
                break

            smallest_link_idx = link_weights.index(min(link_weights))  # Find index of smallest link weight
            smallest_link = links_list[smallest_link_idx]

            # Print removal information
            node_1, node_2 = smallest_link
            print(f'\nRemoving link: ({node_1},{node_2}) with weight {link_weights[smallest_link_idx]}')

            # Remove the smallest-weight link from the node's link dictionary and the list
            self.N[node_1]['links'].pop(node_2, None)  # Safely attempt to pop
            links_list.pop(smallest_link_idx)  # Now links_list is a list, it's safe to pop
            self.L = tuple(links_list)  # Convert back to a tuple if necessary for other uses

            # Re-evaluate the network to determine impact of removal
            if self.Z:
                self.Learn()

            # Check if the previous state was saved and then restore it
            if self.X is not None and self.Y is not None:
                self.L = eval(self.X)
                self.N = eval(self.Y)
                self.G = original_G  # Restore the original number of generations

                # Minimize error and confirm unit testing
                self.Learn(minimize_error=True)

                # Restore the last successful links and network for testing
                self.L = eval(self.X)
                self.N = eval(self.Y)

                # Run tests twice as in the provided code
                self.Test()
                self.Test()

                # Printing the network status after pruning
                print(self.N)
                pruned_links = original_G - len(self.L)  # Adjust this line if original_G was a typo for original size
                print(f'\n :: pruning complete. pruned {pruned_links} links. current size = {len(self.L)} ::')
                print(f'\n :: final unit testing.    error: {self.E:0.4f}    score: {self.F} / {len(self.U)} ::\n')



#######################################################################
#######################################################################
#######################################################################
#######################################################################

# UNIT TESTING / LEARNING / PRUNING

unit_tests = [
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 0, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 0, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 0, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 0, 'A5': 1, 'A6': 1}, {'C0': 0}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 0, 'A6': 1}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 0}, {'C0': 1}),
    ({'A0': 1, 'A1': 1, 'A2': 1, 'A3': 1, 'A4': 1, 'A5': 1, 'A6': 1}, {'C0': 1})
]

"""unit_tests = [
    ({'A0':1, 'A1':0, 'A2':0}, {'C0':1}),
    ({'A0':1, 'A1':0, 'A2':1}, {'C0':0}),
    ({'A0':1, 'A1':1, 'A2':0}, {'C0':0}),
    ({'A0':1, 'A1':1, 'A2':1}, {'C0':1})
]"""

total_error = []
runs = 1
# initialize
A, B, C = 7, 3, 1 # initial size is 34 (3 * (7 + 3) + 7 - 3)
NN = STAG(A, B, C)
NN.U = unit_tests

# Run
NN.Learn()
NN.Prune()
# for _ in range(runs):
#     NN = STAG(A, B, C)
#     NN.U = unit_tests
#     NN.Learn()
#     NN.Prune()
#     total_error.append(NN.E)

# average_error = sum(total_error) / len(total_error)
# print(f"Average total error after {runs} runs: {average_error}")

# # NN.Test()
# NN.Visualize()

