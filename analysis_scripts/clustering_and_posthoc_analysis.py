import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from scipy.stats import hypergeom
from scipy.special import comb
import itertools
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from scipy.stats import shapiro, levene
import scipy.stats as stats





# Fix variable of interst for posthoc test
# select from 499_effect_size or 250_effect_size
metric_of_interest = '499_effect_size'  # Example metric, replace with your actual metric column



def main():
    
    # read dataset
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'datasets', 'data_main_analysis.csv')
    df = pd.read_csv(file_path)

    #define cols in which fitness is stored
    fitness_cols = [str(i) for i in range(0, 500)]


    # perform Mann Whithney U tests and compute effect sizes
    scenario_id = 0
    results = dict()
    for matrix in range(1, 3):
        for coordination in range(1, 7):
            for incentive in range(1, 5):
                for correlation in range(1, 4):
                    condition1 = df["matrix"] == matrix
                    condition2 = df["coordination"] == coordination
                    condition3 = df["incentive"] == incentive
                    condition4 = df["correlation"] == correlation

                    df_analysis = df.loc[
                        df[condition1 & condition2 & condition3 & condition4].index,
                        fitness_cols,
                    ]
                    results[scenario_id] = dict()
                    for step in range(250, 500, 50):
                        u_stat, p_value, rank_biserial_correlation = analyze(df_analysis, t_1 = 249, t_2 = step)
                        entries_to_add = {
                            "matrix": matrix,
                            "coordination": coordination,
                            "incentive": incentive,
                            "correlation": correlation,
                            str(step) + "_u_stat": u_stat,
                            str(step) + "_p-value": p_value,
                            str(step) + "_effect_size": rank_biserial_correlation,
                        }
                        results[scenario_id].update(entries_to_add)

                    step = 499
                    u_stat, p_value, rank_biserial_correlation = analyze(df_analysis, t_1 = 249, t_2 = step)
                    entries_to_add = {
                        "matrix": matrix,
                        "coordination": coordination,
                        "incentive": incentive,
                        "correlation": correlation,
                        str(step) + "_u_stat": u_stat,
                        str(step) + "_p-value": p_value,
                        str(step) + "_effect_size": rank_biserial_correlation,
                    }
                    results[scenario_id].update(entries_to_add)

                    scenario_id = scenario_id + 1
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(results).T
    # Write the DataFrame to an Excel file
    df.to_excel("main_analysis.xlsx", index=False)


    # prepare for clustering
    data = df
    # Selecting categorical and numerical columns
    categorical_cols = ['matrix', 'coordination', 'incentive', 'correlation']
    numerical_cols = ['250_effect_size', '350_effect_size', '499_effect_size']

    # Creating transformers for categorical and numerical data
    categorical_transformer = OneHotEncoder()
    numerical_transformer = StandardScaler()

    # Combining transformers into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ])

    # Applying transformations
    data_processed = preprocessor.fit_transform(data)

    # Extracting only the numerical columns (effect sizes) after transformation
    effect_sizes = data_processed[:, -3:]

    # Range of cluster numbers to evaluate
    cluster_range = range(2, 11)

    # Calculate silhouette scores for different numbers of clusters
    silhouette_scores = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(effect_sizes)  # Fit on the original processed data
        silhouette_avg = silhouette_score(effect_sizes, cluster_labels)  # Compute silhouette score on the original data
        silhouette_scores.append(silhouette_avg)

    # Best number of clusters based on silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    optimal_score = max(silhouette_scores)

    print(f'Optimal number of clusters using Silhouette scores: {optimal_clusters}')
    
    db_scores = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(effect_sizes)
        db_index = davies_bouldin_score(effect_sizes, cluster_labels)
        db_scores.append(db_index)
    optimal_clusters_db = cluster_range[np.argmin(db_scores)]
    print(f'Optimal number of clusters using Davies Boulding Score: {optimal_clusters_db}')

  
    ch_scores = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(effect_sizes)
        ch_score = calinski_harabasz_score(effect_sizes, cluster_labels)
        ch_scores.append(ch_score)
    optimal_clusters_ch = cluster_range[np.argmax(ch_scores)]
    print(f'Optimal number of clusters using Calinski-Harabasz Index: {optimal_clusters_ch}')

    optimal_clusters = optimal_clusters_ch

    # Fitting the optimal k-means model
    kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_optimal.fit_predict(effect_sizes)

    # Assign cluster labels to scenarios in the original data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels

    # Export the DataFrame with cluster labels to an Excel file
    data_with_clusters.to_excel('data_with_clusters.xlsx', index=False)

  
    # Generate descriptive statistics per cluster
    cluster_descriptions = data_with_clusters.groupby('Cluster')[numerical_cols + categorical_cols].agg(['mean', 'std', 'count'])
    cluster_descriptions.to_excel('cluster_descriptions.xlsx', engine='openpyxl')


    # Create cross-tabs
    matrix_crosstab = pd.crosstab(data_with_clusters['Cluster'], data_with_clusters['matrix'])
    coordination_crosstab = pd.crosstab(data_with_clusters['Cluster'], data_with_clusters['coordination'])
    incentive_crosstab = pd.crosstab(data_with_clusters['Cluster'], data_with_clusters['incentive'])
    correlation_crosstab = pd.crosstab(data_with_clusters['Cluster'], data_with_clusters['correlation'])

    print('Log likelihood ratio')
    g, p_value, dof, expctd = chi2_contingency(matrix_crosstab, lambda_="log-likelihood")
    print(f'Matrix: {p_value}')
    g, p_value, dof, expctd = chi2_contingency(coordination_crosstab, lambda_="log-likelihood")
    print(f'Coordination: {p_value}')
    g, p_value, dof, expctd = chi2_contingency(incentive_crosstab, lambda_="log-likelihood")
    print(f'Incentive: {p_value}')
    g, p_value, dof, expctd = chi2_contingency(correlation_crosstab, lambda_="log-likelihood")
    print(f'Correlation: {p_value}')

    cluster_column = 'Cluster'  

    # Perform Tukey's HSD
    tukey = pairwise_tukeyhsd(endog=data_with_clusters[metric_of_interest],     # Data
                              groups=data_with_clusters[cluster_column],             # Groups
                              alpha=0.05)                       # Significance level
    # Print Tukey's test result
    print('Tuckey Test results:')
    print(tukey)

    # Testing normality for variable of interest within each cluster using Shapiro-Wilk Test
    normality_results = {}
    for cluster in data_with_clusters[cluster_column].unique():
        cluster_data = data_with_clusters[data_with_clusters[cluster_column] == cluster][metric_of_interest].dropna()
        stat, p_value = shapiro(cluster_data)
        normality_results[cluster] = (stat, p_value)

    normality_results_df = pd.DataFrame(normality_results, index=['Shapiro Stat', 'P-Value']).T
    
    # Testing homogeneity of variances across clusters using Levene's Test
    levene_stat, levene_p_value = levene(*[data_with_clusters[data_with_clusters[cluster_column] == cluster][ metric_of_interest].dropna() for cluster in data_with_clusters[cluster_column].unique()])

    print(f'Normality of variable of interest within clusters:')
    print(normality_results_df)
    print(f'Variance homogeneity across clusters: {levene_p_value}')


    # Group the data by 'Cluster' and apply list to variable of interest
    grouped_data = data_with_clusters.groupby(cluster_column)[metric_of_interest].apply(list)

    # Perform Kruskal-Wallis test
    stat, p_value = stats.kruskal(*grouped_data)
    print(f"Kruskal-Wallis test statistic: {stat}, p-value: {p_value}")

    # Convert grouped data from Series to list of lists for Dunn's test
    data_for_dunns = [group for group in grouped_data]

    # Perform Dunn's test for multiple comparisons
    dunn_results = sp.posthoc_dunn(data_for_dunns, p_adjust='bonferroni')
    results_df = pd.DataFrame(dunn_results)
    results_df.to_excel('dunn_test_results.xlsx')


def analyze(df, t_1, t_2, n1=150, n2=150):
    sample_1 = df[str(t_1)]
    sample_2 = df[str(t_2)]
    u_stat, p_value = stats.mannwhitneyu(sample_1, sample_2, alternative='two-sided')
    rank_biserial_correlation = 1 - (2 * u_stat) / (n1 * n2)

    return u_stat, p_value, rank_biserial_correlation


if __name__ == "__main__":
    main()