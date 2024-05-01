
# Table of Contents
1. [Installation and Usage](#installation-and-usage)
2. [Datasets](#datasets)
3. [Sensitivity analysis](#sensitivity-analysis)
4. [Analysis scripts](#analysis-scripts)

# Installation and Usage

## Model Description
A detailed model description using the Odd Protocol is provided in the file [model_description.pdf](model_description.pdf).

## Source Code
The source code is provided in the folder [source_code](source_code).

## Running the Model
To run the model, follow these steps:

1. Make sure you have Python installed on your system.
2. Install the required packages by running `pip install -r requirements.txt`
3. Download the source code from the repository.
4. Open the file `parameters.py` and set the parameters as described in the model description.
5. Use the command line to run `main.py`: `python main.py`
6. The simulation will create a JSON file named `results.json`, which will be stored in the same folder as `main.py`.

# Datasets

The repository includes several datasets used in the project. These datasets are stored in the `datasets` folder. Here is a brief description of each dataset:

- **data_main_analysis.csv**: This dataset contains the primary data used for the main paper analysis.

- **data_variance_stability.csv**: This dataset contains data used for the analysis of variance stability to determine the required number of simulation runs.

- **data_sensitivity.csv**: This dataset contains data used for a variance-based sensitivity analysis.

# Sensitivity analysis

A comprehensive analysis has been conducted to assess the requied number of simluation runs and the sensitivity of the results to parameter settings. The results of this analysis are provided in the file `sensitivity_analysis.md`. 

The **variance stability analysis** aims to determine the required number of simulation runs to achieve stable results. It provides insights into the variability of the model outputs under different scenarios.

The **variance-based sensitivity analysis** evaluates the impact of input parameters on the model outputs. By quantifying the sensitivity of the simluation output to various factors, it helps identify influential parameters and understand their relative importance in driving the model behavior.

# Analysis scripts

TBA








