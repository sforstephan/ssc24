import json
import numpy as np
import pandas as pd
import os
from scipy import stats as stats
from statsmodels.tsa.stattools import adfuller
import zipfile

FROM_timeseries = 1
TO_timeseries = 249

def main():

    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, "datasets", "data_main_analysis.csv")
    file_path_zip = os.path.join(parent_dir, "datasets", "data_main_analysis.csv.zip")
    check_and_unzip(file_path, file_path_zip)
    df = pd.read_csv(file_path)

    fitness_cols = [str(i) for i in range(FROM_timeseries, TO_timeseries)]

    scenario_id = 0
    adf_results = dict()
    for matrix in range(1, 3):
        for coordination in range(1, 7):
            for incentive in range(1, 5):
                for correlation in range(1, 4):
                    # fix conditions following the coding explained in the main file
                    condition1 = df["matrix"] == matrix
                    condition2 = df["coordination"] == coordination
                    condition3 = df["incentive"] == incentive
                    condition4 = df["correlation"] == correlation

                    df_analysis = df.loc[
                        df[condition1 & condition2 & condition3 & condition4].index,
                        fitness_cols,
                    ]
                    adf_result = adf_test(df_analysis)
                    # Create a dictionary to store results
                    adf_results[scenario_id] = {
                        "matrix": matrix,
                        "coordination": coordination,
                        "incentive": incentive,
                        "correlation": correlation,
                        "ADF Statistic": adf_result[0],
                        "p-value": adf_result[1],
                        "Critical Value 1%": adf_result[4]["1%"],
                        "Critical Value 5%": adf_result[4]["5%"],
                        "Critical Value 10%": adf_result[4]["10%"],
                        "Number of Lags Used": adf_result[2],
                        "Number of Observations Used": adf_result[3],
                    }
                    scenario_id = scenario_id + 1

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(adf_results).T
    # Write the DataFrame to an Excel file
    df.to_excel("stationarity.xlsx", index=False)


def adf_test(df):
    # compute means
    row_means = df.iloc[:, :250].mean(axis=1)

    # apply the Augmented Dickey-Fuller test
    adf_result = adfuller(row_means)

    return adf_result


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
