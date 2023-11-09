def generate_custom_weights(input_size, hidden_size, output_size, weight_func=None, seed=None):
    if seed is not None:
        import random
        random.seed(seed)  # Set the seed for reproducibility
    
    # Create a dictionary for custom weights
    custom_weights = {}
    
    # Generate weights for input to hidden layer connections
    for i in range(input_size):
        for h in range(hidden_size):
            link = ('A' + str(i).zfill(len(str(input_size - 1))), 'B' + str(h).zfill(len(str(hidden_size - 1))))
            custom_weights[link] = weight_func(link) if weight_func else random.uniform(-1, 1)
    
    # Generate weights for hidden to output layer connections
    for h in range(hidden_size):
        link = ('B' + str(h).zfill(len(str(hidden_size - 1))), 'C0')
        custom_weights[link] = weight_func(link) if weight_func else random.uniform(-1, 1)
    
    return custom_weights

print(generate_custom_weights(7, 3, 1, seed=100))