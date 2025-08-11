# Import relevant libraries
import os
import numpy as np
import json

def simulate_data(batch_runner,
                  simulation_out_path,
                  n_jobs=8):
    """
    Runs a batch of simulations using the provided BatchRunner instance and saves the results.

    Args:
        batch_runner (BatchRunner): An instance of BatchRunner configured with samples to simulate.
        simulation_out_path (str): Directory path where simulation results and metadata will be saved.
        n_jobs (int, optional): Number of parallel jobs to use for running simulations. Defaults to 8.

    Returns:
        tuple:
            - simulations (list): List of simulation results, where each entry corresponds to a sample.
            - bool_indices (list): List of indices in `simulations` where the result is a boolean (indicating failure).

    Side Effects:
        - Saves raw simulation results to a file in `simulation_out_path`.
        - Saves indices of failed simulations (boolean results) to `bool_indices.json` in `simulation_out_path`.
    """

    # Get the number of samples from the batch runner
    n_samples = batch_runner._samples.shape[0]

    # Create the directory for saving simulations
    os.makedirs(os.path.join(simulation_out_path,
                            f'raw_simulations_{n_samples}_samples'),
                            exist_ok=True)

    print(f"Running batch simulation with {n_samples} samples...")
    simulations = batch_runner.run_batch(
            n_jobs=n_jobs,
            output_path=os.path.join(simulation_out_path, f'raw_simulations_{n_samples}_samples')
        )

    # Track the boolean indices of failed simulations
    bool_indices = [index for index, value in enumerate(simulations) if isinstance(value, bool)]

    # Save the boolean indices to a JSON file
    with open(os.path.join(simulation_out_path,"bool_indices.json"), 'w') as f:
        json.dump(bool_indices, f)

    # Print the boolean indices
    print(bool_indices)

    return simulations, bool_indices
