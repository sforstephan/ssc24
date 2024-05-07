import json
import pandas as pd
import os
import zipfile


def main():

    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'datasets', 'data_variance_stability.csv')
    file_path_zip = os.path.join(parent_dir, 'datasets', 'data_variance_stability.csv.zip')
    check_and_unzip(file_path, file_path_zip)
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


def check_and_unzip(target_file, zip_file):
    # Check if the target file exists
    if os.path.exists(target_file):
        print(f"Dataset exists.")
    else:
        print(f"Attempting to unzip dataset.")
        # Attempt to unzip the file
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(target_file))
                # print(f"Unzipped files to {os.path.dirname(target_file)}")
        except FileNotFoundError:
            print(f"The zip file was not found.")
        except zipfile.BadZipFile:
            print(f"The file is not a valid zip file.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
