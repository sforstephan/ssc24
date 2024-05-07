import json
import pandas as pd
import numpy as np
import os
from SALib.sample import sobol as sb
from SALib.analyze import sobol
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3)


def main():

    Si, problem = compute_sobol_indices(2**14)
    print_sobol_indices(Si, problem)
    plot_sobol_indices(Si, problem, "sobol_indices")


def print_sobol_indices(Si, problem):

    param_names = ["Interdependence pattern", "Decision-making mode", "Incentive parameter", "Correlation"]

    print("First-order indices:")
    first_order_indices = Si["S1"]
    for i in range(len(first_order_indices)):
        print(f"{param_names[i]}: {first_order_indices[i]:.3f}")

    print("Total-effect indices:")
    total_order_indices = Si["ST"]
    for i in range(len(total_order_indices)):
        print(f"{param_names[i]}: {total_order_indices[i]:.3f}")


def plot_sobol_indices(Si, problem, str):
    # First-order indices
    S1 = Si["S1"]
    # Total-effect indices
    ST = Si["ST"]

    # Setting up the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Names of the parameters
    # param_names = problem["names"]
    param_names = ["Interdependence pattern", "Decision-making mode", "Incentive parameter", "Correlation"]

    # Plotting first-order indices
    ax[0].grid(True, which="both", axis="y", linestyle="--")
    ax[0].bar(param_names, S1)
    ax[0].set_title("First-order Sobol indices")
    ax[0].set_ylabel("Index value")
    ax[0].set_ylim([0, 1])
    ax[0].set_xticklabels(param_names, rotation=45, ha="right")
    

    # Plotting total-effect indices
    ax[1].grid(True, which="both", axis="y", linestyle="--")
    ax[1].bar(param_names, ST)
    ax[1].set_title("Total-effect Sobol indices")
    ax[1].set_ylabel("Index value")
    ax[1].set_ylim([0, 1])
    ax[1].set_xticklabels(param_names, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{str}.jpeg", dpi=750)
    return plt


def compute_sobol_indices(N):

    # read the csv file with all simulated scenarios and coded parameter settings
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'datasets', 'data_sensitivity.csv')
    
    df = pd.read_csv(file_path)

    problem = {
        # number of variables that define the simulated scenarios
        "num_vars": 4,
        # names of these variables
        "names": ["matrix", "coordination", "incentive", "correlation"],
        # Assuming each parameter is treated as continuous for sampling purposes
        # The bounds are set to encompass the discrete values
        "bounds": [
            [1, 2],  # for "matrix"
            [3, 6],  # for "coordination"
            [1, 4],  # for "incentive"
            [1, 3],  # for "correlation"
        ],  
    }

    # randomly generate N parameter combinations based on the problem definition above (i.e., N random combinations of the four paramters)
    param_values = sb.sample(problem, N, calc_second_order=True)

    # Placeholder for model output
    Y = np.zeros([param_values.shape[0]])

    for i, X in enumerate(param_values):
        # Map continuous values back to their discrete counterparts
        discrete_param_values = [round(x) for x in X]
        # define conditions for filtering the simulated scenarios
        condition1 = df["matrix"] == discrete_param_values[0]
        condition2 = df["coordination"] == discrete_param_values[1]
        condition3 = df["incentive"] == discrete_param_values[2]
        condition4 = df["correlation"] == discrete_param_values[3]
        # apply the filters so that in tmp_df only the simulated scenarios for a specific parmaters combination are included
        tmp_df = df[condition1 & condition2 & condition3 & condition4]["fitness"]
        # randomly sample one simulation run out of tmp_df and store the result in the placeholder for model output
        Y[i] = tmp_df.sample()

    # compute and return first- and higher-order Sobol indices
    indices = sobol.analyze(problem, Y, print_to_console=False)
    return indices, problem


if __name__ == "__main__":
    main()
