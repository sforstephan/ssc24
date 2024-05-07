import json
import pandas as pd
import os


def main():

    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'datasets', 'data_variance_stability.csv')
    df = pd.read_csv(file_path)

    # fix conditions following the coding explained in the main file
    condition1 = df["matrix"] == 1
    condition2 = df["coordination"] == 4
    condition3 = df["repetitions"] == 1000

    fitness_cols = [str(i) for i in range(0, 500)]
    df_analysis = df.loc[df[condition1 & condition2 & condition3].index, fitness_cols]

    # Compute the standard deviation, the mean and the Coefficient of Variation
    sigma = df_analysis.values.flatten().std()
    mu = df_analysis.values.flatten().mean()
    # Print the result
    print(f"Coefficient of Variation for selected scenario: {round(sigma / mu, 3)}")



if __name__ == "__main__":
    main()
