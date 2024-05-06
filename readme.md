
# Table of Contents
1. [ODD Model Descriptions](#odd-model-descriptions)
1. [Running the model](#running-the-model)
2. [Datasets](#datasets)
3. [Sensitivity analysis](#sensitivity-analysis)
4. [Analysis scripts](#analysis-scripts)

## ODD Model Description
A detailed model description using the ODD Protocol is provided in the file [model_description.pdf](model_description.pdf).

# Running the model

The source code is provided in the folder [source_code](source_code).

To run the model, follow these steps:

1. Make sure you have Python installed on your system.
2. Install the required packages by running `pip install -r requirements.txt`
3. Download the source code from the repository.
4. Open the file `parameters.py` and set the parameters as described in the model description.
5. Use the command line to run `main.py`: `python main.py`
6. The simulation will create a JSON file named `results.json`, which will be stored in the same folder as `main.py`.

# Datasets

The repository includes several datasets used in the project. These datasets are stored in the [datasets](datasets) folder. Here is a brief description of each dataset:

- **data_main_analysis.csv**: This dataset contains the primary data used for the main paper analysis.

- **data_variance_stability.csv**: This dataset contains data used for the analysis of variance stability to determine the required number of simulation runs.

- **data_sensitivity.csv**: This dataset contains data used for a variance-based sensitivity analysis.

Please note that the uploaded datasets are compressed (in zip-format).

# Sensitivity analysis

A comprehensive analysis has been conducted to assess the requied number of simluation runs and the sensitivity of the results to parameter settings. The results of this analysis are provided in the file [sensitivity_analysis.md](sensitivity_analysis.md). 

The **variance stability analysis** aims to determine the required number of simulation runs to achieve stable results. It provides insights into the variability of the model outputs under different scenarios.

The **variance-based sensitivity analysis** evaluates the impact of input parameters on the model outputs. By quantifying the sensitivity of the simluation output to various factors, it helps identify influential parameters and understand their relative importance in driving the model behavior.

# Analysis scripts

## Analysis of timeseries stationarity
1.  Open the script and fix the variables `FROM_timeseries` and `TO_timeseries` to fix the range of the timeseries for which stationarity should be tested
2.  The script reads the dataset `data_main_analysis.csv` and extracts the timeseries defined by `FROM_timeseries` and `TO_timeseries`
3.  The script performs an Augmented Dickey-Fuller test and writes the rest results in the file `stationarity.xlsx`

## Cluster analysis and posthoc test
1.  Open the script and fix the variable `variable_of_interest` for the posthoc analysis
2. The script reads the dataset `data_main_analysis.csv` and 
- performs the Mann-Whitney U test,
- computes the rank biserial correlation as effect size measure,
- computes the optimal number of clusters for k-means clustering, 
- performs k-means clustering, 
- computes summary statistics for clusters, 
- computes a G-test (log likelihood ratio test), 
- performs a Tuckey's test, 
- test for normality of clusters and homodscedasticity, 
- performs a Kruskal-Wallis test, and 
- performs a Dunn's test.
2. The following output is stored when the script runs: 
- `main_analysis.xlsx` contains results of Mann-Whitney U test and effect sizes, 
- `data_with_clusters.xlsx` contains the results of the cluster analysis, 
- `cluster_descriptions.xlsx` contains the summary statistics of clusters, 
- `dunn_test_results.xlsx` contains the results of a Dunn's test
3. The follwoing output is displayed in the termincal window when the script runs: 
- optimal number of clusters computed using Silhouette scores, Davies-Bouldin scores, and Calinski Harabasz scores, 
- p-values computes using a log-likelihood ratio test for all features, 
- results of the Tuckey's test, 
- results of a Shapiro Wilk test and a Levene test to test for normal distribution of the variable of interest in clusters and homoscedasticity across clusters, 
- results of a Kruskal-Wallis test 

## Variance stability and sensitivity analysis

###  Variance stability
Open the script `variance_stability.py` and fix the parameters `condition1` (filter for interdependence pattern), `condition2` (filter for decision-making mode), and `condition3`(filter for the number of simulation runs you are interested in). When the script is run, the Coefficient of Variation for the selected scenario is displayed in the terminal window. 

### Sensitivity analysis









