# Import relevant libraries
import os
from ModularCirc import BatchRunner
from ModularCirc.Models.NaghaviModel import NaghaviModel, NaghaviModelParameters, TEMPLATE_TIME_SETUP_DICT
import numpy as np
import json

def sample_input_space(param_path,
                       n_samples=2048,
                       sampling_method='Sobol'):
    """
    Sample the input space based on the parameters defined in the JSON file.

    :param param_path: Path to the JSON file containing model parameters.
    :param n_samples: Number of samples to generate.
    :return: A DataFrame containing the sampled input space.
    """

    # Get the filename from the path, without extension
    param_filename = os.path.splitext(os.path.basename(param_path))[0]

    # Define the location for saving simulation outputs

    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    simulation_out_path = os.path.join(root_dir, f'outputs/simulations/output_{n_samples}_samples_{param_filename}/')

    ## read in save parameters to the folder where simulations r saved.
    with open(param_path, 'r') as f:
        params = json.load(f)

    # save parameters to the folder where simulations are saved.
    with open(os.path.join(simulation_out_path, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=4)

    # Set up the batch runner
    br = BatchRunner(sampling_method, 0) # why are we using 'Sobol' here?
    br.setup_sampler(param_path)
    br.sample(n_samples)

    map_ = {
        'lv.t_tr' : ['lv.t_tr',],
        'la.t_tr' : ['la.t_tr',],
        'la.delay' : ['la.delay',],
        'lv.tau' : ['lv.tau',],
        'la.tau' : ['la.tau',],
        'lv.t_max' : ['lv.t_max',],
        'la.t_max' : ['la.t_max',],
    }

    # Map the sample timings
    br.map_sample_timings(
        ref_time=1000., # double check if 1000 or 1
        map=map_
        )

    # Map the vessel volumes
    br.map_vessel_volume()

    # Save the samples to a CSV file
    br.samples.to_csv(os.path.join(simulation_out_path,
                                   f'input_samples_{n_samples}.csv'),
                                   index=False)

    # Set up the model with the parameters and time setup
    br.setup_model(model=NaghaviModel, po=NaghaviModelParameters,
                    time_setup=TEMPLATE_TIME_SETUP_DICT)

    return br, simulation_out_path
