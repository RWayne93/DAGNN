from random import randint, random

class GeneticRepresentation:
    def __init__(self, length=32):
        # A random binary string of the given length
        self.bin_string = ''.join(['1' if random() > 0.5 else '0' for _ in range(length)])

    def crossover(self, other):
        """Perform a one-point crossover"""
        pos = randint(0, len(self.bin_string) - 1)
        child1 = GeneticRepresentation(length=0)
        child2 = GeneticRepresentation(length=0)
        child1.bin_string = self.bin_string[:pos] + other.bin_string[pos:]
        child2.bin_string = other.bin_string[:pos] + self.bin_string[pos:]
        return child1, child2

    def mutate(self, mutation_rate=0.01):
        """Mutate the binary string based on the mutation rate"""
        self.bin_string = ''.join([bit if random() > mutation_rate else ('1' if bit == '0' else '0') for bit in self.bin_string])

# Example usage:
parent1 = GeneticRepresentation()
parent2 = GeneticRepresentation()
print(f"Parent1: {parent1.bin_string}")
print(f"Parent2: {parent2.bin_string}")

child1, child2 = parent1.crossover(parent2)
child1.mutate()
child2.mutate()

print(f"Child1: {child1.bin_string}")
print(f"Child2: {child2.bin_string}")

# class NeuralNetwork:
#     def __init__(self):
#         # Sample binary representation
#         self.nodes = '11001' # Represents values of nodes
#         self.links = '10101' # Represents presence/absence of links
    
#     def mutate(self):
#         # Toggle a random bit in nodes or links
#         if random.choice([True, False]):
#             idx = random.randint(0, len(self.nodes) - 1)
#             self.nodes = self.nodes[:idx] + ('1' if self.nodes[idx] == '0' else '0') + self.nodes[idx + 1:]
#         else:
#             idx = random.randint(0, len(self.links) - 1)
#             self.links = self.links[:idx] + ('1' if self.links[idx] == '0' else '0') + self.links[idx + 1:]
    
#     def crossover(self, other):
#         # Split at random point and swap parts
#         idx = random.randint(0, len(self.nodes) - 1)
#         self.nodes, other.nodes = self.nodes[:idx] + other.nodes[idx:], other.nodes[:idx] + self.nodes[idx:]
        
#         idx = random.randint(0, len(self.links) - 1)
#         self.links, other.links = self.links[:idx] + other.links[idx:], other.links[:idx] + self.links[idx:]
        
#         return other

#     # ... other functions ...

# # Sample usage
# nn1 = NeuralNetwork()
# nn2 = NeuralNetwork()

# nn1.mutate()
# nn2 = nn1.crossover(nn2)
