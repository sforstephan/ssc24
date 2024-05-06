
# Table of Contents
1. [ODD Model Description](#odd-model-descriptions)
1. [Running the model](#running-the-model)
2. [Datasets](#datasets)
3. [Sensitivity analysis](#sensitivity-analysis)
4. [Analysis scripts](#analysis-scripts)

# ODD Model Description

A detailed model description using the ODD protocol is provided in the file [model_description.pdf](figures_and_pdfs/model_description.pdf).

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

- [data_main_analysis.csv](datasets/data_main_analysis.csv.zip): This dataset contains the primary data used for the main paper analysis.

- [data_variance_stability.csv](datasets/data_variance_stability.csv.zip): This dataset contains data used for the analysis of variance stability to determine the required number of simulation runs.

- [data_sensitivity.csv](datasets/data_sensitivity.csv.zip): This dataset contains data used for a variance-based sensitivity analysis.

Please note that the uploaded datasets are compressed (in zip-format), please download and upack them to view their contents.

# Sensitivity analysis

A comprehensive analysis has been conducted to assess the requied number of simluation runs and the sensitivity of the results to parameter settings. The results of this analysis are provided [here](sensivity_analysis/sensitivity_analysis.md). 

The **variance stability analysis** aims to determine the required number of simulation runs to achieve stable results. It provides insights into the variability of the model outputs under different scenarios.

The **variance-based sensitivity analysis** evaluates the impact of input parameters on the model outputs. By quantifying the sensitivity of the simluation output to various factors, it helps identify influential parameters and understand their relative importance in driving the model behavior.

# Analysis scripts

## Analysis of timeseries stationarity
1. Open the script [timeseries_stationarity.py](analysis_scripts/timeseries_stationarity.py) and set the variables FROM_timeseries and TO_timeseries to specify the range of the time series for which stationarity should be tested.
2. The script reads the dataset [data_main_analysis.csv](datasets/data_main_analysis.csv.zip) and extracts the time series defined by FROM_timeseries and TO_timeseries.
3. The script performs an Augmented Dickey-Fuller test and records the results in the file stationarity.xlsx.

## Cluster analysis and posthoc test
1. Open the script `[clustering_and_posthoc_analysis.py](analysis_scripts/clustering_and_posthoc_analysis.py)` and set the variable `variable_of_interest` for the posthoc analysis.
2. The script reads the dataset `[data_main_analysis.csv](datasets/data_main_analysis.csv.zip)` and performs several analyses:
   - **Mann-Whitney U test**,
   - **Rank biserial correlation** as an effect size measure,
   - **K-means clustering**, including computing the optimal number of clusters,
   - **Summary statistics** for clusters,
   - **G-test (log-likelihood ratio test)**,
   - **Tukey's test**,
   - Tests for **normality of clusters** and **homoscedasticity**,
   - **Kruskal-Wallis test**,
   - **Dunn's test**.

3. **Outputs stored upon script execution**:
   - `main_analysis.xlsx`: Results of Mann-Whitney U test and effect sizes.
   - `data_with_clusters.xlsx`: Results of the cluster analysis.
   - `cluster_descriptions.xlsx`: Summary statistics of clusters.
   - `dunn_test_results.xlsx`: Results of Dunn's test.

4. **Outputs displayed in the terminal window**:
   - Optimal number of clusters computed using **Silhouette scores**, **Davies-Bouldin scores**, and **Calinski-Harabasz scores**.
   - **P-values** computed using a log-likelihood ratio test for all features.
   - Results of the **Tukey's test**.
   - Results of the **Shapiro-Wilk test** and **Levene test** for testing the normal distribution of the variable of interest in clusters and homoscedasticity across clusters.
   - Results of a **Kruskal-Wallis test**.









