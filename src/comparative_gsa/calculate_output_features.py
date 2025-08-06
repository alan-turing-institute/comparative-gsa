# Import relevant libraries
import os
import pandas as pd
import numpy as np


def calculate_output_features(simulations,
                              simulation_out_path):
    """    Calculate summary statistics for each simulation and save to a CSV file.
    Args:
        simulations (list): List of simulation DataFrames.
        simulation_out_path (str): Path to save the summary CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics for each simulation.
    """

    summary_rows = []

    for i_sim in range(len(simulations)): ## stuck with the length of simulations like fay
        sim_df = simulations[i_sim]

        # Skip if not a DataFrame this was needed because sometimes simulations can return None or other types. or false.
        if not isinstance(sim_df, pd.DataFrame):
            print(f"Simulation {i_sim} is not a DataFrame, skipping.")
            continue

        row = {} ##the 'dictiionary' to hold the summary for each simulation
        for h in sim_df.columns: # We had sim_df the whole time which means h can be used without the headers list!
            row[f"{h}_mean"] = sim_df[h].mean()
            row[f"{h}_max"] = sim_df[h].max() ## using the headers defined above to make the new csv file headers eg v_ao_mean, v_ao_max, etc.
        summary_rows.append(row) # this is a list that is being used to collect summary data for our simulations.
        print(f"Simulation {i_sim} done.") # simple way to track progress

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(simulation_out_path, "simulations_summary.csv"), index=False)  ## the location, nice and organised!
    print("Saved as simulation_summary.csv") # not really needed but nice to show the file has been saved! :D

    return summary_df  # Return the summary DataFrame for further use if needed
