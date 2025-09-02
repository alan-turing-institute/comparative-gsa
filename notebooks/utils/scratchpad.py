# Import relevant libraries
import os
import pandas as pd

from ModularCirc.Models.NaghaviModel import NaghaviModel, NaghaviModelParameters, TEMPLATE_TIME_SETUP_DICT

import numpy as np

from comparative_gsa.sample_input_space import sample_input_space
from comparative_gsa.simulate_data import simulate_data
from comparative_gsa.calculate_output_features import calculate_output_features
steps = {
    1,
    2,
    3,
}

param_path = 'inputs/parameters_naghavi_constrained_fixed_T_v_tot_v_ref_lower_k_pas.json'
# Get the filename from the path, without extension
param_filename = os.path.splitext(os.path.basename(param_path))[0]

n_samples = 2048


simulation_out_path = f'outputs/simulations/output_{n_samples}_samples_{param_filename}/'
#create folder if it does not exist
os.makedirs(simulation_out_path, exist_ok=True)

if 1 in steps:

    # Run sampling of the input space:
    br, simulation_out_path = sample_input_space(
        param_path=param_path,
        n_samples=n_samples
    )
if 2 in steps:

    simulations, bool_indices = simulate_data(
        batch_runner=br,
        simulation_out_path=simulation_out_path,
        n_jobs=1
    )
