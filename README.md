# -ALGORITHMIC_GROWTH
# Required libraries
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn import linear_model
from sklearn.covariance import GraphicalLassoCV
from sklearn.metrics import r2_score
from pandas.plotting import autocorrelation_plot
import math

# --------------------------------------------------------------------------#
# IMPORT DATA + SAVE IMAGES IN PDF
# --------------------------------------------------------------------------#

file_name = 'pseudodata_3'
person = 'NAAM'

# Import data from Excel without first column
skip_cols = [0,1] # ADD skipping nominal expressions (via index list) ; or include them by using GradientBoosters ! (not recommended) (0 = days ; 1 = location)
total_num_cols = 14
keep_cols = [i for i in range(total_num_cols) if i not in skip_cols]
df = pd.read_excel(f'{file_name}.xlsx', usecols=keep_cols)
variables = df.columns.tolist() # List of variables from the DataFrame

pdf_path = '/Users/stijnvanseveren/PycharmProjects/pythonProject/#GROWTH_OFFICIAL/all_figures.pdf'

# Creating a PDF to save figures
with (PdfPages(pdf_path) as pdf):
    # --------------------------------------------------------------------------#
    # 1. CORRELATION MATRIX
    # --------------------------------------------------------------------------#

    plt.figure(figsize=(14, 15))
    correlation_matrix = df.corr()  # Calculating correlation matrix

    # Custom color map for heatmap
    colors = ["red", "white", "blue"]
    cmap = mcolors.LinearSegmentedColormap.from_list('customized map', colors)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, cbar=True)

    # Highlighting the diagonal elements
    mask = np.eye(len(correlation_matrix), dtype=bool)
    sns.heatmap(correlation_matrix, mask=~mask, annot=True, fmt=".2f", cmap=['lightgrey'], cbar=False, linewidths=0.001, linecolor='black')

    # Boxing the correlation matrix
    ax = plt.gca()
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.title(f"CORRELATION MATRIX: {file_name}", size=28, y=1.05, fontweight='bold') # MAKE FRONT PAGE !
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    pdf.savefig()
    plt.close()

    # --------------------------------------------------------------------------#
    # 2. SPARSE NETWORK OF PARTIAL CORRELATIONS
    # --------------------------------------------------------------------------#

    model = GraphicalLassoCV(alphas=10 ** np.linspace(-0.3, 3, 50)) # Use fixed sparsity instead of range_check ! + Also use regular correlations
    model.fit(df)
    precision_matrix = model.precision_
    partial_correlations = -precision_matrix / np.outer(np.sqrt(np.diag(precision_matrix)),np.sqrt(np.diag(precision_matrix)))
    np.fill_diagonal(partial_correlations, 0)

    # Creating network graph
    graph = nx.Graph()
    for i, variable_i in enumerate(variables):
        for j, variable_j in enumerate(variables):
            if i < j and np.abs(partial_correlations[i, j]) > 1e-2:
                graph.add_edge(variable_i, variable_j, weight=partial_correlations[i, j])

    # Drawing the network
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, weight='weight')
    edges = graph.edges()

    original_weights = [partial_correlations[variables.index(u)][variables.index(v)] for u, v in edges]
    edge_colors = ['blue' if weight > 0 else 'red' for weight in original_weights]

    nx.draw_networkx_nodes(graph, pos, node_size=150, node_color='darkgrey', linewidths=2)
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_colors,
                           width=[abs(weight) * 5 for weight in original_weights])
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')

    # Create legend for edge colors
    positive_edge = plt.Line2D([0], [0], color='blue', lw=4, label='Positive Correlation')
    negative_edge = plt.Line2D([0], [0], color='red', lw=4, label='Negative Correlation')
    plt.legend(handles=[positive_edge, negative_edge], loc='lower right')

    plt.title('SPARSE PARTIAL CORRELATION NETWORK', size=26, y=0.94, fontweight='bold')
    plt.axis('off')
    pdf.savefig(bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------------------------#
    # 3. MATRIX_CONSISTENCY
    # --------------------------------------------------------------------------#

    # FIX 4 ERROR TYPES !: 1. ONLY 1 SAMPLE AVAILABLE --> reshape ; 2. RUNTIMEWARNING: invalid value encountered in subtract --> remove NaN/infs ; 3. RUNTIMEWARNING: divide by zero encountered in scalar divide ; 4. RUNTIMEWARNING: invalid value encountered in multiply

    # 3.1 WEEKLY MATRIX CONSISTENCY
    # Divide the data into weekly segments
    weekly_data = [df.iloc[i:i + 7] for i in range(0, 28, 7)]

    # Calculate partial correlation for each week
    partial_correlation_matrices = []
    for week in weekly_data:
        # Initialize Graphical Lasso with a range of alpha values
        model = GraphicalLassoCV(alphas=10 ** np.linspace(-0.5, 4, 50))
        model.fit(week)

        # Obtain the precision matrix
        precision_matrix = model.precision_

        # Calculate the partial correlation matrix
        partial_correlation = -precision_matrix / np.outer(np.sqrt(np.diag(precision_matrix)),
                                                               np.sqrt(np.diag(precision_matrix)))
        np.fill_diagonal(partial_correlation, 0)
        partial_correlation_matrices.append(partial_correlation)

    # Calculate the average partial correlation matrix
    mean_matrix = np.mean(partial_correlation_matrices, axis=0)

    # Compute Frobenius_Norm dissimilarities
    dissimilarities = [np.linalg.norm(matrix - mean_matrix, 'fro') for matrix in partial_correlation_matrices]

    # Calculate mean and standard deviation of the dissimilarities
    FROBENIUS_NORM_avg_dissimilarity_weeks = np.mean(dissimilarities) # F_N_max = 2N
    FROBENIUS_NORM_sd_dissimilarity_weeks = np.std(dissimilarities)

    # 3.2 BOOTSTRAP ANALYSIS FOR MATRICES (add 95% confidence intervals !)

    # Number of bootstrap iterations
    n_iterations = 1 # Use 500 for actual data ; for now set it to 1
    bootstrap_matrices = []

    for _ in range(n_iterations):
        # Sample with replacement from each week
        weekly_samples = [week.sample(n=7, replace=True) for week in weekly_data if not week.empty]
        combined_sample = pd.concat(weekly_samples)

        # Network analysis
        model = GraphicalLassoCV(alphas=10 ** np.linspace(-0.5, 4, 50))
        model.fit(combined_sample)
        precision_matrix = model.precision_
        partial_correlations = -precision_matrix / np.outer(np.sqrt(np.diag(precision_matrix)),
                                                            np.sqrt(np.diag(precision_matrix)))
        np.fill_diagonal(partial_correlations, 0)

        # Store the partial correlation matrix
        bootstrap_matrices.append(partial_correlations)

    # Compute Frobenius_Norm dissimilarities
    mean_matrix = np.mean(bootstrap_matrices, axis=0)
    dissimilarities = [np.linalg.norm(matrix - mean_matrix, 'fro') for matrix in bootstrap_matrices]

    # Calculate mean and standard deviation of the dissimilarities
    bootstrap_avg_dissimilarity = np.mean(dissimilarities)
    bootstrap_st_dissimilarity_deviation = np.std(dissimilarities)

    # --------------------------------------------------------------------------#
    # 4. GRAPH METRICS (locals for now)
    # --------------------------------------------------------------------------#

    # Function that computes top 3 centrality values of nodes
    def top_3_nodes(centrality_dict):
        nodes_quants = zip(centrality_dict.keys(), centrality_dict.values())
        sorted_nodes_quants = sorted(nodes_quants, key=lambda x: x[1], reverse=True)
        top_values = sorted(set(val for node, val in sorted_nodes_quants), reverse=True)[:3]
        return dict((node, val) for node, val in sorted_nodes_quants if val in top_values)

    # 4.1 Degree Centrality
    degree_centrality = nx.degree_centrality(graph)
    TOP_3_DCnodes = top_3_nodes(degree_centrality)

    # 4.2 Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(graph)
    TOP_3_BCnodes = top_3_nodes(betweenness_centrality)

    # 4.3 Eigenvector Centrality
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=2500)
    TOP_3_ECnodes = top_3_nodes(eigenvector_centrality)

    # --------------------------------------------------------------------------#
    # 5. DAILY PROGRESS
    # --------------------------------------------------------------------------#

    n_observations = len(df[df.columns[0]])
    X = range(1, n_observations + 1)

    plots_per_page = 4
    rows, cols = 2, 2  # 2x2 grid for 4 plots per page
    total_pages = (len(variables) + plots_per_page - 1) // plots_per_page

    for page in range(total_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle(f'DAILY PROGRESS', fontsize=16, fontweight = 'bold')


        for i in range(plots_per_page):
            index = page * plots_per_page + i
            if index >= len(variables):
                # Hide unused axes
                axes[i // cols, i % cols].axis('off')
                continue

            variable = variables[index]
            Y = df[variable]

            # List with 7days_Moving_Averages
            Y_series = pd.Series(Y)
            rolling_mean = Y_series.rolling(window=7, min_periods=1, center=True).mean()

            # Linear regression
            slope, intercept, r, p, std_err = stats.linregress(X, Y)
            mymodel = [slope * x + intercept for x in X]

            # Create legend
            #SlOPE = plt.Line2D([0], [0], color='red', lw=2, label='Slope')
            #RAW = plt.scatter([0], [0], color='b', lw=2, label='Raw Data')
            #MA_7 = plt.scatter([0], [0], color='lightgreen', lw=2, label='7Day_Moving_Average')
            #plt.legend(handles=[RAW,MA_7,SlOPE], loc='lower right') (Fix this !)

            # Draw subplot
            ax = axes[i // cols, i % cols]
            ax.scatter(X, Y)
            #ax.scatter(X,rolling_mean,color='lightgreen') # Leave it out for now
            ax.plot(X, mymodel, color='r')
            ax.text(X[-1], mymodel[-1], f'slope={slope:.2f}', color='red', va='center', ha='left')
            ax.plot(X, Y, color='b', linestyle='-', marker='')  # Connect the raw data points
            ax.set_xlabel('DAYS')
            ax.set_ylabel(f'SCORE')
            ax.set_title(f'PROGRESS_{index + 1}: CRITERION {variable}', size=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Save the figure
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------------------------#
    # 6. INTRA-WEEK COMPARISON (+ add 5_workdays/2_weekends)
    # --------------------------------------------------------------------------#

    # Import 'DAYS'
    df_DAYS = pd.read_excel(f'{file_name}.xlsx',usecols=[0])
    # List all weekdays
    weekdays = df_DAYS['DAGEN'].unique()
    domains = df.columns

    # Initialize a dictionary to store the averages
    average_scores_per_domain = {domain: [] for domain in domains}

    # Iterate over each domain and weekday to fill the dictionary
    for domain in domains:
        for weekday in weekdays:
            indices = df_DAYS[df_DAYS['DAGEN'] == weekday].index
            df_weekday = df.iloc[indices]
            avg_score = round(df_weekday[domain].mean(), 2)
            average_scores_per_domain[domain].append(avg_score)

    # SubPlotting parameters
    plots_per_page = 4
    rows, cols = 2, 2  # 2x2 grid for 4 plots per page
    total_pages = (len(domains) + plots_per_page - 1) // plots_per_page

    # Plotting
    for page in range(total_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle(f'INTRA-WEEK COMPARISONS', fontsize=16, fontweight = 'bold')


        for i in range(plots_per_page):
            index = page * plots_per_page + i
            if index >= len(domains):
                axes[i // cols, i % cols].axis('off')
                continue

            domain = domains[index]
            averages = average_scores_per_domain[domain]

            ax = axes[i // cols, i % cols]
            ax.bar(weekdays, averages)
            ax.set_ylabel('Average Score')
            ax.set_title(f'{domain}', size=10)

            # Set x-ticks and x-tick labels
            ax.set_xticks(range(len(weekdays)))
            ax.set_xticklabels([f'{day}S' for day in weekdays], rotation=45)

            for j, avg in enumerate(averages):
                ax.text(j, avg, f'{avg:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        pdf.savefig()
        plt.close()


    # --------------------------------------------------------------------------#
    # 7. AUTOCORRELATION (periodicity)
    # --------------------------------------------------------------------------#

    # Subplotting parameters
    plots_per_page = 4
    rows, cols = 2, 2  # 2x2 grid for 4 plots per page
    total_pages = (len(domains) + plots_per_page - 1) // plots_per_page

    # Autocorrelation plotting using subplots
    for page in range(total_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle('PERIODICITIES', fontsize=16, fontweight = 'bold')

        for i in range(plots_per_page):
            index = page * plots_per_page + i
            if index >= len(domains):
                if i < rows * cols:
                    axes[i // cols, i % cols].axis('off')
                continue

            domain = domains[index]
            ax = axes[i // cols, i % cols]
            autocorrelation_plot(df[domain].dropna(), ax=ax)
            ax.set_title(domain)
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------------------------#
    # 8. SCATTERPLOT OF THE CRITERIA
    # --------------------------------------------------------------------------#

    significant_pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            X = df[variables[i]]
            Y = df[variables[j]]
            slope, intercept, r, p, std_err = stats.linregress(X, Y)
            if p <= 0.05:
                significant_pairs.append((variables[i], variables[j], slope, intercept, r, p))

    # Subplot parameters
    plots_per_page = 4
    rows, cols = 2, 2  # 2x2 grid for 4 plots per page
    total_pages = math.ceil(len(significant_pairs) / plots_per_page)

    for page in range(total_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        fig.suptitle(f'SCATTERPLOTS OF SIGNIFICANT PAIRS', fontsize=16, fontweight='bold')

        for i in range(plots_per_page):
            index = page * plots_per_page + i
            if index >= len(significant_pairs):
                if i < rows * cols:
                    axes[i // cols, i % cols].axis('off')
                continue

            var_i, var_j, slope, intercept, r, p = significant_pairs[index]
            X = df[var_i]
            Y = df[var_j]
            mymodel = [slope * x + intercept for x in X]

            ax = axes[i // cols, i % cols]
            ax.scatter(X, Y, label=f'{var_i} vs {var_j}')
            ax.plot(X, mymodel, color='r')

            # Adjusted significance level calculation
            if p <= 0.001:
                sig_amount = '***'
            elif p <= 0.01:
                sig_amount = '**'
            else:
                sig_amount = '*'  # Significant ; p<0.05

            text_y_position = slope * max(X) + intercept
            ax.text(max(X), text_y_position, f'r={r:.3f}{sig_amount}', color='red', va='center', ha='left')
            ax.set_xlabel(var_i)
            ax.set_ylabel(var_j)
            ax.set_title(f'{var_i} vs {var_j}', size=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------------------------#
    # 10. INTRA-INDIVIDUAL DOMAIN VARIABILITY - AVERAGES
    # --------------------------------------------------------------------------#

    # Calculate and plot the averages for each variable
    list_averages = [(variable, round(np.mean(df[variable]), 2)) for variable in variables]
    sorted_data_dict = dict(sorted(list_averages, key=lambda x: x[1], reverse=True))
    criteria = list(sorted_data_dict.keys())[::-1]
    scores = list(sorted_data_dict.values())[::-1]

    plt.figure(figsize=(13, 8))
    plt.barh(criteria, scores, color='steelblue')
    plt.xlabel('Average Score',size=13)
    plt.title('INTRA-INDIVIDUAL VARIABILITY: AVERAGES', size=20, fontweight='bold', y=1.04)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    Lowest_5 = list(sorted_data_dict.items())[-5:]

    # Annotate bars with scores
    for i, score in enumerate(scores):
        plt.text(score, i, str(score))
    pdf.savefig()
    plt.close()

    # --------------------------------------------------------------------------#
    # 11. INTRA-INDIVIDUAL DOMAIN VARIABILITY - STANDARD DEVIATIONS
    # --------------------------------------------------------------------------#

    # Calculate and plot the standard deviations for each variable
    list_stds = [(variable, round(np.std(df[variable]), 2)) for variable in variables]
    sorted_data_dict = dict(sorted(list_stds, key=lambda x: x[1], reverse=True))
    criteria = list(sorted_data_dict.keys())[::-1]
    scores = list(sorted_data_dict.values())[::-1]

    plt.figure(figsize=(13, 8))
    plt.barh(criteria, scores, color='skyblue')
    plt.xlabel('Sta.dev. Score',size=13)
    plt.title('INTRA-INDIVIDUAL VARIABILITY: STANDARD DEVIATIONS', size=20, fontweight='bold', y=1.04)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate bars with scores
    for i, score in enumerate(scores):
        plt.text(score, i, str(score))

    pdf.savefig()
    plt.close()

    # --------------------------------------------------------------------------#
    # 12. LOCATION-SPECIFIC METRICS
    # --------------------------------------------------------------------------#

    # Import location_info
    df_loc = pd.read_excel(f'{file_name}.xlsx', usecols=[1]) # For now location_info is stored in the second column
    unique_locations = df_loc['LOCATIE'].unique()

    #12.1 AVERAGES
    # Creating a subplot for each location
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    fig.suptitle(f'LOCATION-SPECIFIC AVERAGES', fontsize=16, fontweight='bold',y=0.96)

    plt.subplots_adjust(hspace=0.4, wspace=0.9)

    for i, location in enumerate(unique_locations):
        df_specific = df[df_loc['LOCATIE'] == location]

        list_averages_specific = [(variable, round(np.mean(df_specific[variable]), 2)) for variable in variables]
        sorted_data_dict_specific = dict(sorted(list_averages_specific, key=lambda x: x[1], reverse=True))
        criteria_specific = list(sorted_data_dict_specific.keys())[::-1]
        scores_specific = list(sorted_data_dict_specific.values())[::-1]

        axes[i].barh(criteria_specific, scores_specific, color='steelblue')
        axes[i].set_xlabel('Average', size=13)
        axes[i].set_title(f'{location}', size=14)
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)

        for j, score in enumerate(scores_specific):
            axes[i].text(score, j, str(score))

    pdf.savefig()
    plt.close()

    #12.2 STANDARD DEVIATIONS (# NOW TOO REDUNDANT; ADD M&SD FUNCTIONS)
    # Creating a subplot for each location
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    fig.suptitle(f'LOCATION-SPECIFIC STANDARD DEVIATIONS', fontsize=16, fontweight='bold',y=0.96)

    plt.subplots_adjust(hspace=0.4, wspace=0.9)

    for i, location in enumerate(unique_locations):
        df_specific = df[df_loc['LOCATIE'] == location]

        list_standarddeviations_specific = [(variable, round(np.std(df_specific[variable]), 2)) for variable in variables]
        sorted_data_dict_specific = dict(sorted(list_standarddeviations_specific, key=lambda x: x[1], reverse=True))
        criteria_specific = list(sorted_data_dict_specific.keys())[::-1]
        scores_specific = list(sorted_data_dict_specific.values())[::-1]

        axes[i].barh(criteria_specific, scores_specific, color='skyblue')
        axes[i].set_xlabel('Standard Deviation', size=13)
        axes[i].set_title(f'{location}', size=14)
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)

        for j, score in enumerate(scores_specific):
            axes[i].text(score, j, str(score))

    pdf.savefig()
    plt.close()

    #12.3 CORRELATION MATRICES & SPARSE PARTIAL CORRELATION NETWORKS
    df_loc = pd.read_excel(f'{file_name}.xlsx', usecols=[1]) # Column of location_information
    unique_locations = df_loc['LOCATIE'].unique()

    # Store partial correlations and graphs for each location
    partial_correlation_matrices = {}
    network_graphs = {}

    for location in unique_locations:
        df_specific = df[df_loc['LOCATIE'] == location]

        # Compute partial correlations
        model = GraphicalLassoCV(alphas=10 ** np.linspace(0, 1, 50))
        model.fit(df_specific)
        precision_matrix = model.precision_
        partial_correlations = -precision_matrix / np.outer(np.sqrt(np.diag(precision_matrix)),
                                                            np.sqrt(np.diag(precision_matrix)))
        np.fill_diagonal(partial_correlations, 0)
        partial_correlation_matrices[location] = partial_correlations

        # Create network graph
        graph = nx.Graph()
        for i, variable_i in enumerate(variables):
            for j, variable_j in enumerate(variables):
                if i < j and np.abs(partial_correlations[i, j]) > 1e-2:
                    graph.add_edge(variable_i, variable_j, weight=partial_correlations[i, j])
        network_graphs[location] = graph

    # Plotting the correlation matrices
    locations_per_page = 4
    total_pages = (len(unique_locations) + locations_per_page - 1) // locations_per_page

    for page in range(total_pages):
        fig, axs = plt.subplots(2, 2, figsize=(14, 14))
        fig.suptitle(f'REGULARISED PARTIAL CORRELATION MATRIX PER LOCATION', fontsize=16, fontweight='bold')
        for i in range(locations_per_page):
            index = page * locations_per_page + i
            if index >= len(unique_locations):
                # Hide unused axes
                axs[i // 2, i % 2].axis('off')
                continue

            location = unique_locations[index]
            ax = axs[i // 2, i % 2]
            sns.heatmap(partial_correlation_matrices[location], annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                        cbar=True, ax=ax)
            ax.set_title(f'{location}', size=16)

            # Highlighting the diagonal elements
            mask = np.eye(len(partial_correlation_matrices[location]), dtype=bool)
            sns.heatmap(partial_correlation_matrices[location], mask=~mask, annot=True, fmt=".2f", cmap=['lightgrey'],
                        cbar=False, linewidths=0.001, linecolor='black', ax=ax)

            # Boxing the correlation matrix
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(1.5)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Plotting the network graphs
    for page in range(total_pages):
        fig, axs = plt.subplots(2, 2, figsize=(14, 14))
        plt.suptitle(f'GRAPHICAL LASSO PER LOCATION', size=16, fontweight='bold')

        for i in range(locations_per_page):
            index = page * locations_per_page + i
            if index >= len(unique_locations):
                # Hide unused axes
                axs[i // 2, i % 2].axis('off')
                continue

            location = unique_locations[index]
            graph = network_graphs[location]
            pos = nx.spring_layout(graph, weight='weight')
            edges = graph.edges()
            weights = [graph[u][v]['weight'] for u, v in edges]

            # Design of graphical lasso
            edge_colors = ['blue' if weight > 0 else 'red' for weight in weights]
            nx.draw_networkx_nodes(graph, pos, node_size=200, node_color='lightgrey', linewidths=2,
                                   ax=axs[i // 2, i % 2])
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_colors,
                                   width=[abs(weight) * 5 for weight in weights], ax=axs[i // 2, i % 2])
            nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', ax=axs[i // 2, i % 2])

            axs[i // 2, i % 2].set_title(f'{location}', size=16)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------------------------#
    # 12. MULTIPLE REGRESSION MODELS (add validation metrics: cross-validation + statistical significance test for coefficients)
    # --------------------------------------------------------------------------#

    # 12.1 8-PREDICTORS (now 7 !) as PREDICTORS for 5-CRITERIONS
    Predictors = df.columns[:7]  # First 8 columns as predictors (normally 8 ; but now without location)
    Criterions = df.columns[7:12]  # Last 5 columns as criterions

    X = df[Predictors]

    # Loop through each criterion
    for criterion in Criterions:
        y = df[criterion]

        model = linear_model.Lasso(alpha=0.4)
        model.fit(X, y)

        predictions = model.predict(X)
        r2 = r2_score(y, predictions)

        standard_coefficients = model.coef_
        text_to_display = f"  Proportion Explained Variance: {round(r2 * 100, 1)}%\n"
        text_to_display += "  Coefficient for each Predictor:\n"

        predictor_coef = zip(Predictors, standard_coefficients)
        sorted_predictor_coef = sorted(predictor_coef, key=lambda x: abs(x[1]), reverse=True)

        for predictor, coef in sorted_predictor_coef:
            text_to_display += f"    {predictor}: {coef:.3f}\n"

        plt.figure(figsize=(5, 2))
        plt.text(0.1, 0.9, text_to_display, ha='left', va='top', fontsize=10)
        plt.title(f'Regression Results for CRITERION "{criterion}"', fontweight='bold', size=10, y=0.94)
        plt.axis('off')
        #pdf.savefig() # Only useful for data-analysis
        plt.close()

    # 12.2 4-CRITERIONS as PREDICTORS for 1_CRITERION (change indices !)
    Predictors = df.columns[7:11]  # 4 columns as predictors
    Criterion = df.columns[11]  # One criterion

    X = df[Predictors]
    y = df[Criterion]

    model = linear_model.LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    r2 = r2_score(y, predictions)

    standard_coefficients = model.coef_
    text_to_display = f"  Proportion Explained Variance: {round(r2 * 100, 1)}%\n"
    text_to_display += "  Coefficient for each Predictor:\n"
    predictor_coef = zip(Predictors, standard_coefficients)
    sorted_predictor_coef = sorted(predictor_coef, key=lambda x: abs(x[1]), reverse=True)

    for predictor, coef in sorted_predictor_coef:
        text_to_display += f"    {predictor}: {coef:.3f}\n"

    plt.figure(figsize=(5, 2))
    plt.text(0.1, 0.9, text_to_display, ha='left', va='top', fontsize=10)
    plt.title(f'Regression Results for CRITERION "{Criterion}"', fontweight='bold', size=10, y=0.94)
    plt.axis('off')
    #pdf.savefig() # Only useful for peer-supported data-analysis
    plt.close()
