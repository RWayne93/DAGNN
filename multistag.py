# import json
# from multiprocessing import Pool
# import os

# def run_network_instance(index):
#     # Set up for running an isolated instance of the network.
#     # Import or define STAG and any required variables inside this function
#     from classes.STAG import STAG  # Replace with your actual import

#     # Assuming 'unit_tests' is a variable that should be defined within each process
#     unit_tests = ...  # Define or import your unit_tests here for each process
    
#     A, B, C = 7, 3, 1  # Assuming these are the parameters for the NN initialization

#     # Initialize STAG instance (Replace 'STAG' with your actual neural network class)
#     NN = STAG(A, B, C)
#     NN.U = unit_tests
#     NN.Learn()
#     NN.Prune()
    
#     # Return the state instead of saving it directly to file
#     return NN.get_network_state()  # Ensure this function returns a dictionary

# def init_networks(runs):
#     # Pool of workers equal to the number of CPU cores available
#     with Pool(processes=os.cpu_count()) as pool:
#         results = pool.map(run_network_instance, range(runs))
#     return results

# def save_results_to_json(filename, results):
#     with open(filename, 'w') as f_out:
#         json.dump(results, f_out)

# if __name__ == '__main__':
#     runs = 1000  # Number of runs you want to perform
#     results = init_networks(runs)  # Get the network states after running instances
#     save_results_to_json('network_states.json', results)

import json
from multiprocessing import Pool
import os

def run_network_instance(index):
    from classes.STAG import STAG  # Use your actual import statement
    unit_tests = ...  # Define your unit_tests here
    A, B, C = 7, 3, 1  # Network parameters
    NN = STAG(A, B, C)
    NN.U = unit_tests
    NN.Learn()
    NN.Prune()
    return NN.get_network_state()

def init_networks(start, end):
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(run_network_instance, range(start, end))
    return results

def save_results_to_json(filename, results, mode='w'):
    with open(filename, mode) as f_out:
        json.dump(results, f_out)

def load_results_from_json(filename):
    with open(filename, 'r') as f_in:
        return json.load(f_in)

if __name__ == '__main__':
    filename = 'network_states.json'
    total_runs = 1000  # Total number of desired runs
    
    # Load existing results if file exists, else start from scratch
    if os.path.isfile(filename):
        try:
            existing_results = load_results_from_json(filename)
            start_run = len(existing_results)  # Determine where to resume
            print(f"Resuming from run {start_run}.")
        except json.JSONDecodeError:  # Handle case where the file may be empty or corrupt
            existing_results = []
            start_run = 0
            print("JSON file is corrupted or empty. Starting from run 0.")
    else:
        existing_results = []
        start_run = 0

    if start_run < total_runs: 
        # If there are remaining runs, continue running them
        new_results = init_networks(start_run, total_runs)
        full_results = existing_results + new_results
        save_results_to_json(filename, full_results, 'w')
    else:
        print("All runs are already completed.")
