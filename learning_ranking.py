# %%
import gc  # Import the garbage collector
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
import dask.dataframe as dd  # Import Dask

# Specify here the name of the parquet output by the benchmark
file_name = ''

# Incremental Processing (Best for moderately large files)

def process_parquet(df):
    """Aggregates and extracts data from a DataFrame."""

    # Aggregate
    df = df.groupby(['objective_name', 'solver_name', 'data_name']).agg({
        'objective_score_test': 'mean',
        'objective_score_train': 'mean',
        'objective_score_val': 'mean',
        'objective_score_bench': 'mean',
        'objective_value': 'mean',
        'time': 'sum',
    }).reset_index()
    return df

# List of your parquet file paths
file_paths = [
    f'outputs/{file_name}',
]

# %%
# Process each file in chunks and store the results
processed_dataframes = []
for file_path in file_paths:
    ddf = dd.read_parquet(file_path)
    for chunk in ddf.to_delayed():
        df_chunk = chunk.compute()
        processed_df = process_parquet(df_chunk)
        processed_dataframes.append(processed_df)
        gc.collect()  # Force garbage collection after each chunk

# Concatenate the *results* of the aggregations
data = pd.concat(processed_dataframes)
del processed_dataframes  # Remove intermediate list
gc.collect()

# Now do the string extractions
data[['n_repeats', 'n_splits', 'procedure', 'study_size', 'test_size', 'val_size']] = data['objective_name'].str.extract(r'n_repeats=(.*),n_splits=(.*),procedure=(.*),study_size=(.*),test_size=(.*),val_size=(.*)')
data[['dataset_n_features', 'dataset_n_samples', 'dataset_noise', 'seed']] = data['data_name'].str.extract(r'n_features=(.*),n_samples=(.*),noise=(.*),seed=(.*)')

nb_exp = len(data['seed'].unique())
print(nb_exp)

# %%
data['n_repeats'] = data['n_repeats'].astype(int)
data['n_splits'] = data['n_splits'].astype(int)
data['study_size'] = data['study_size'].astype(int)
data['test_size'] = data['test_size'].astype(float)
data['val_size'] = data['val_size'].str.replace(']','')
data['val_size'] = data['val_size'].astype(float)


data['dataset_n_features'] = data['dataset_n_features'].astype(int)
data['dataset_n_samples'] = data['dataset_n_samples'].astype(int)
data['dataset_noise'] = data['dataset_noise'].astype(float)
data['seed'] = data['seed'].str.replace(']','')
data['seed'] = data['seed'].astype(int)

# %%
data[['extratrees_n_estimators']] = data['solver_name'].str.extract(r'n_estimators=(.*)')
data['extratrees_n_estimators'] = data['extratrees_n_estimators'].str.replace(']','')
data['extratrees_n_estimators'] = data['extratrees_n_estimators'].astype(float)

data_n_estimators = data[data['extratrees_n_estimators'] > -1]

# %%
test_size = float(data['test_size'].unique())

# Find the study sizes where the number of rows is equal to 48000
study_sizes_no_missing = [10, 18, 32, 56, 100, 178, 316, 562, 1000, 1778, 3162,
                       5623, 10000]

# %%
# Initialize the ranking dictionary
ranking_test = {}

# Group the data by the relevant columns
grouped = data_n_estimators.groupby(['procedure', 'n_splits', 'n_repeats', 'seed', 'study_size'])

# Define the possible rankings
rankings = ['2-20-200', '2-200-20', '20-2-200', '20-200-2', '200-2-20', '200-20-2']

# Function to determine the ranking based on objective scores
def determine_ranking(group):
    scores = group.set_index('extratrees_n_estimators')['objective_score_test']
    scores_2, scores_20, scores_200 = scores.get(2, -np.inf), scores.get(20, -np.inf), scores.get(200, -np.inf)

    if scores_2 > scores_20:
        if scores_20 > scores_200:
            return '2-20-200'
        elif scores_2 > scores_200:
            return '2-200-20'
        else:
            return '200-2-20'
    else:
        if scores_2 > scores_200:
            return '20-2-200'
        elif scores_20 > scores_200:
            return '20-200-2'
        else:
            return '200-20-2'

# Apply the ranking function to each group
grouped_ranking = grouped.apply(determine_ranking).reset_index(name='ranking')

# Group by the same columns without 'seed' to aggregate the counts
aggregated_ranking = grouped_ranking.groupby(['procedure', 'n_splits', 'n_repeats', 'study_size', 'ranking']).size().unstack(fill_value=0)

# Convert the aggregated ranking to the desired format
for (procedure, n_splits, n_repeats, study_size), row in aggregated_ranking.iterrows():
    key = f"{procedure}_{n_splits}_{n_repeats}_{study_size}"
    ranking_test[key] = {r: row.get(r, 0) for r in rankings}

    # Determine the quasi-oracle ranking
    quasioracle = max(ranking_test[key], key=ranking_test[key].get)
    print(quasioracle, ranking_test[key][quasioracle], 'study_size:', study_size, 'procedure:', procedure, 'n_splits:', n_splits, 'n_repeats:', n_repeats)

# Create a new dictionary with modified keys
ranking_modified = {}
for key, value in ranking_test.items():
    # Replace '_0' with '.' in the key
    new_key = key.replace('_0.', '.')
    ranking_modified[new_key] = value

ranking_test = ranking_modified

# %%
# Initialize the ranking dictionary
ranking = {}

# Group the data by the relevant columns
grouped = data_n_estimators.groupby(['procedure', 'n_splits', 'n_repeats', 'seed', 'study_size'])

# Define the possible rankings
rankings = ['2-20-200', '2-200-20', '20-2-200', '20-200-2', '200-2-20', '200-20-2']

# Function to determine the ranking based on objective scores
def determine_ranking(group):
    scores = group.set_index('extratrees_n_estimators')['objective_score_bench']
    scores_2, scores_20, scores_200 = scores.get(2, -np.inf), scores.get(20, -np.inf), scores.get(200, -np.inf)

    if scores_2 > scores_20:
        if scores_20 > scores_200:
            return '2-20-200'
        elif scores_2 > scores_200:
            return '2-200-20'
        else:
            return '200-2-20'
    else:
        if scores_2 > scores_200:
            return '20-2-200'
        elif scores_20 > scores_200:
            return '20-200-2'
        else:
            return '200-20-2'

# Apply the ranking function to each group
grouped_ranking = grouped.apply(determine_ranking).reset_index(name='ranking')

# Group by the same columns without 'seed' to aggregate the counts
aggregated_ranking = grouped_ranking.groupby(['procedure', 'n_splits', 'n_repeats', 'study_size', 'ranking']).size().unstack(fill_value=0)

# Convert the aggregated ranking to the desired format
for (procedure, n_splits, n_repeats, study_size), row in aggregated_ranking.iterrows():
    key = f"{procedure}_{n_splits}_{n_repeats}_{study_size}"
    ranking[key] = {r: row.get(r, 0) for r in rankings}

    # Determine the quasi-oracle ranking
    quasioracle = max(ranking[key], key=ranking[key].get)
    print(quasioracle, ranking[key][quasioracle], 'study_size:', study_size, 'procedure:', procedure, 'n_splits:', n_splits, 'n_repeats:', n_repeats)

# Create a new dictionary with modified keys
ranking_modified = {}
for key, value in ranking.items():
    # Replace '_0' with '.' in the key
    new_key = key.replace('_0.', '.')
    ranking_modified[new_key] = value

ranking = ranking_modified

# %%
# Initialize a dictionary to store the 95% thresholds
thresholds_95 = {}
thresholds_95_test = {}

# Convert the ranking dictionary to a Pandas DataFrame
ranking_df = pd.DataFrame.from_dict(ranking, orient='index')
ranking_df['total'] = ranking_df.sum(axis=1)
ranking_df = ranking_df.div(ranking_df['total'], axis=0).drop(columns='total')

# Reset index to make the keys a column
ranking_df = ranking_df.reset_index()
ranking_df = ranking_df.rename(columns={'index': 'key'})

# Extract the group name and study size from the key
ranking_df['group'] = ranking_df['key'].str.rsplit('_', n=1).str[0]
ranking_df['study_size'] = ranking_df['key'].str.rsplit('_', n=1).str[1].astype(int)

# Convert the ranking_test dictionary to a Pandas DataFrame
ranking_test_df = pd.DataFrame.from_dict(ranking_test, orient='index')
ranking_test_df['total'] = ranking_test_df.sum(axis=1)
ranking_test_df = ranking_test_df.div(ranking_test_df['total'], axis=0).drop(columns='total')

# Reset index to make the keys a column
ranking_test_df = ranking_test_df.reset_index()
ranking_test_df = ranking_test_df.rename(columns={'index': 'key'})

# Extract the group name and study size from the key
ranking_test_df['group'] = ranking_test_df['key'].str.rsplit('_', n=1).str[0]
ranking_test_df['study_size'] = ranking_test_df['key'].str.rsplit('_', n=1).str[1].astype(int)

# Define the order of the rankings for consistent plotting
rankings_order = ['200-20-2', '200-2-20', '20-200-2', '20-2-200', '2-200-20', '2-20-200']

# Define custom colors for each ranking
custom_colors = {
    '2-20-200': 'C5',
    '2-200-20': 'C4',
    '20-2-200': 'C3',
    '20-200-2': 'C1',
    '200-2-20': 'C2',
    '200-20-2': 'C0'
}

# Plotting
for group in ranking_df['group'].unique():
    group_data = ranking_df[ranking_df['group'] == group].sort_values('study_size')
    group_data_test = ranking_test_df[ranking_test_df['group'] == group].sort_values('study_size')

    plt.figure(figsize=(6, 4))  # Set figure size for each plot

    for r in rankings_order:
        plt.plot(group_data['study_size'], group_data[r], marker='o', label=f'{r} (bench)', linestyle='-', color=custom_colors[r])
        plt.plot(group_data_test['study_size'], group_data_test[r], marker='x', label=f'{r} (test)', linestyle='--', color=custom_colors[r])

    group_split = group.split('_')
    n_repeats = group_split[-1]
    n_splits = group_split[-2]
    procedure = '_'.join(group_split[:-2])

    #plt.title(f'Évolution des proportions de classements pour la procédure\n{procedure} avec {n_splits} {"partitions" if int(n_splits) > 1 else "partition"} et {n_repeats} {"répétitions" if int(n_repeats) > 1 else "répétition"}', fontsize=14)
    plt.xlabel('Taille du jeu d\'étude', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.ylim(0, 1)  # Set y-axis limit to 0-1 for proportions
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.xticks(study_sizes_no_missing, study_sizes_no_missing, rotation=45)  # Set x-ticks to be study sizes and rotate them
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create custom legend handles for linestyles and colors
    linestyle_handles = [plt.Line2D([0], [0], color='.5', linestyle='-', label='Référence'),
                         plt.Line2D([0], [0], color='.5', linestyle='--', label='Test')]
    color_handles = [plt.Line2D([0], [0], color=color, linestyle='-', label=label) for label, color in reversed(custom_colors.items())]

    # Add the legends to the plot
    legend1 = plt.legend(handles=linestyle_handles, title='Ensemble', fontsize=10, loc='center left', bbox_to_anchor=(1, 0.85))
    plt.legend(handles=color_handles, title='Classement', fontsize=10, loc='center left', bbox_to_anchor=(1, 0.4))
    plt.gca().add_artist(legend1)

    # Find the study size where any ranking reaches 95% or higher for bench
    threshold_reached = group_data[rankings_order].ge(0.95).any(axis=1)
    if threshold_reached.any():
        first_95_study_size = group_data.loc[threshold_reached, 'study_size'].iloc[0]
        thresholds_95[group] = first_95_study_size
        plt.axvline(x=first_95_study_size, color='.6', linestyle='-', label=f'Premier 95% à {first_95_study_size}')
        plt.annotate(f'Référence \nSeuil 95%\nà {first_95_study_size}',
                     xy=(first_95_study_size, 0.5),  # Center vertically at y=0.5
                     xytext=(10, 25),  # Offset horizontally by 10 points
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='.6'),
                     fontsize=10, color='.6',
                     ha='left', va='center')  # Align text horizontally left and vertically center

    # Find the study size where any ranking reaches 95% or higher for test
    threshold_reached_test = group_data_test[rankings_order].ge(0.95).any(axis=1)
    if threshold_reached_test.any():
        first_95_study_size_test = group_data_test.loc[threshold_reached_test, 'study_size'].iloc[0]
        thresholds_95_test[group] = first_95_study_size_test
        plt.axvline(x=first_95_study_size_test, color='.4', linestyle='--', label=f'Premier 95% à {first_95_study_size_test}')
        plt.annotate(f'Test \nSeuil 95%\nà {first_95_study_size_test}',
                     xy=(first_95_study_size_test, 0.5),  # Center vertically at y=0.5
                     xytext=(-10, -25),  # Offset horizontally by -10 points
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='.4'),
                     fontsize=10, color='.4',
                     ha='right', va='center')  # Align text horizontally right and vertically center

    plt.tight_layout()
    plt.savefig(f'rank_{procedure}_{n_splits}_{n_repeats}.pdf')
    plt.show()

# Print the 95% thresholds
print("Seuils de 95% pour chaque combinaison (procédure, n_partitions, n_repetitions) (référence) :")
for key, value in thresholds_95.items():
    print(f"{key}: {value}")

print('\n')

print("Seuils de 95% pour chaque combinaison (procédure, n_splits, n_repeats) (test) :")
for key, value in thresholds_95_test.items():
    print(f"{key}: {value}")

# %%
# Initialize a dictionary to store the 95% thresholds for each combination of (procedure, total_splits)
thresholds_95_combination = {}

# Ensure the 'ranking' column is created
grouped_ranking = grouped.apply(determine_ranking).reset_index(name='ranking')

# Function to determine if any ranking reaches 95% or higher
def check_threshold(group):
    proportions = group['ranking'].value_counts(normalize=True)
    return proportions[proportions >= 0.95].index.tolist()

# Perform bootstrapping to create 20 groups of seeds
n_bootstrap_samples = 100
bootstrap_groups = []

for _ in range(n_bootstrap_samples):
    bootstrap_sample = resample(grouped_ranking['seed'].unique(), replace=True)
    bootstrap_group = grouped_ranking[grouped_ranking['seed'].isin(bootstrap_sample)]
    bootstrap_groups.append(bootstrap_group)

# Apply the threshold check function to each bootstrap group
for i, bootstrap_group in enumerate(bootstrap_groups):
    grouped_packs = bootstrap_group.groupby(['procedure', 'n_splits', 'n_repeats', 'study_size'])
    for (procedure, n_splits, n_repeats, study_size), group in grouped_packs:
        key = f"{procedure}_{n_splits}_{n_repeats}_{i}"
        if key not in thresholds_95_combination:
            thresholds_95_combination[key] = []

        rankings_reaching_95 = check_threshold(group)
        if '200-20-2' in rankings_reaching_95:
            thresholds_95_combination[key].append(study_size)

# Determine the first study size where the 95% threshold is reached for each combination
for key, study_sizes in thresholds_95_combination.items():
    if study_sizes:
        thresholds_95_combination[key] = min(study_sizes)
    else:
        thresholds_95_combination[key] = np.nan

# Print the 95% thresholds for each combination of (procedure, total_splits, bootstrap_sample)
print("95% thresholds for each (procedure, total_splits, bootstrap_sample) combination:")
for key, value in thresholds_95_combination.items():
    print(f"{key}: {value}")

# %%
# Calculate the quotients of the 95% thresholds
quotients = {}
for key, value in thresholds_95_combination.items():
    procedure, n_splits, n_repeats, bootstrap_sample = key.rsplit('_', 3)
    reference_key = f'train_test_split_1_1_{bootstrap_sample}'
    reference_threshold = thresholds_95_combination.get(reference_key, np.nan)
    if reference_threshold != 0:
        quotients[key] = reference_threshold / value

# Convert the quotients dictionary to a DataFrame for plotting
quotients_df = pd.DataFrame.from_dict(quotients, orient='index', columns=['quotient'])
quotients_df = quotients_df.reset_index().rename(columns={'index': 'combination'})

# Extract the procedure, n_splits, n_repeats, and bootstrap_sample from the combination
quotients_df[['procedure', 'n_splits', 'n_repeats', 'bootstrap_sample']] = quotients_df['combination'].str.rsplit('_', n=3, expand=True)

# %%
quotients_df['total_splits'] = quotients_df['n_splits'].astype(int) * quotients_df['n_repeats'].astype(int)

# Sort the DataFrame by total_splits
quotients_df = quotients_df.sort_values(by='total_splits')

quotients_df

# %%
# Create a new column 'total_splits_str' which is a copy of 'total_splits' but as a string
quotients_df['total_splits_str'] = quotients_df['total_splits'].astype(str)

# %%
# Create a limited quotients_df with specific total_splits values
valid_splits = [1, 2, 3, 5, 10, 15, 20, 25, 50, 100]
quotients_df_limited = quotients_df[quotients_df['total_splits'].isin(valid_splits)]

# Plotting
plt.figure(figsize=(6, 4))
sns.violinplot(x='quotient', y='total_splits_str', hue='procedure', density_norm='count', data=quotients_df_limited[quotients_df_limited['procedure'] != 'train_test_split'], inner='quartile', split=True)

#plt.title('Diagrammes en violon des quotients des seuils à 95%', fontsize=14)
plt.xlabel('Gain d\'échantillons', fontsize=12)
plt.ylabel('Nombre total de partitions', fontsize=12)
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.legend(title='Procédure', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('violinplots.pdf')
plt.show()

# %%
# Initialize a dictionary to store the total sum of computing time for each (procedure, n_splits, n_repeats) combination
total_computing_time = {}
threshold_computing_time = {}

# Group the data by procedure, n_splits, and n_repeats
grouped_time = data.groupby(['procedure', 'n_splits', 'n_repeats'])

# Calculate the total computing time for each group
for (procedure, n_splits, n_repeats), group in grouped_time:
    key = f"{procedure}_{n_splits}_{n_repeats}"
    total_computing_time[key] = group['time'].sum()

    # Calculate the computing time at the 95% bench threshold
    threshold_study_size = thresholds_95.get(key, None)
    if threshold_study_size is not None:
        threshold_group = group[group['study_size'] == threshold_study_size]
        threshold_computing_time[key] = threshold_group['time'].sum()

# Print the total computing time for each (procedure, n_splits, n_repeats) combination
print("Total computing time for each (procedure, n_splits, n_repeats) combination:")
for key, value in total_computing_time.items():
    print(f"{key}: {value}")

# Print the computing time at the 95% bench threshold for each (procedure, n_splits, n_repeats) combination
print("Computing time at the 95% bench threshold for each (procedure, n_splits, n_repeats) combination:")
for key, value in threshold_computing_time.items():
    print(f"{key}: {value}")

# %%
# Initialize a list to store the Pareto front points for all (procedure, n_splits, n_repeats) combinations
pareto_front_points = []
mean_pareto_front_points = []

# Get the computing time for the train_test_split procedure
train_test_split_time = total_computing_time.get('train_test_split_1_1', 1)  # Default to 1 if not found

# Calculate the means and standard deviations of the violin plots
means = quotients_df.groupby(['procedure', 'total_splits'])['quotient'].mean().reset_index()
stds = quotients_df.groupby(['procedure', 'total_splits'])['quotient'].std().reset_index()

# Iterate over each (procedure, n_splits, n_repeats) combination
for key, total_time in total_computing_time.items():
    procedure, n_splits, n_repeats = key.rsplit('_', 2)
    total_splits = int(n_splits) * int(n_repeats)
    mean_quotient = means[(means['procedure'] == procedure) & (means['total_splits'] == total_splits)]['quotient'].values
    std_quotient = stds[(stds['procedure'] == procedure) & (stds['total_splits'] == total_splits)]['quotient'].values
    if mean_quotient.size > 0:
        normalized_time = total_time / train_test_split_time
        mean_pareto_front_points.append((normalized_time, mean_quotient[0], std_quotient[0] if std_quotient.size > 0 else 0, procedure, total_splits))

# Sort the points by normalized computing time (x-axis)
mean_pareto_front_points.sort(key=lambda x: x[0])

# Initialize the Pareto front list
mean_pareto_front_std = []

# Iterate over the sorted points to find the Pareto front
for normalized_time, mean_quotient, std_quotient, procedure, total_splits in mean_pareto_front_points:
    if not mean_pareto_front_std or mean_quotient > mean_pareto_front_std[-1][1]:
        mean_pareto_front_std.append((normalized_time, mean_quotient, std_quotient, procedure, total_splits))

# Initialize a list to store the Pareto front points for all (procedure, n_splits, n_repeats) combinations
pareto_front_points = []
non_pareto_front_points = []

mean_pareto_front_points = []
mean_non_pareto_front_points = []

# Get the computing time for the train_test_split procedure
train_test_split_time = total_computing_time.get('train_test_split_1_1', 1)  # Default to 1 if not found

# Calculate the means of the violin plots
means = quotients_df.groupby(['procedure', 'total_splits'])['quotient'].mean().reset_index()
stds = quotients_df.groupby(['procedure', 'total_splits'])['quotient'].std().reset_index()

# Iterate over each (procedure, n_splits, n_repeats) combination
for key, total_time in total_computing_time.items():
    procedure, n_splits, n_repeats = key.rsplit('_', 2)
    total_splits = int(n_splits) * int(n_repeats)
    quotients = quotients_df[(quotients_df['procedure'] == procedure) & (quotients_df['total_splits'] == total_splits)]['quotient'].values
    if quotients.size > 0:
        normalized_time = total_time / train_test_split_time
        for quotient in quotients:
            pareto_front_points.append((normalized_time, quotient, procedure, total_splits))

# Iterate over each (procedure, n_splits, n_repeats) combination
for key, total_time in total_computing_time.items():
    procedure, n_splits, n_repeats = key.rsplit('_', 2)
    total_splits = int(n_splits) * int(n_repeats)
    mean_quotient = means[(means['procedure'] == procedure) & (means['total_splits'] == total_splits)]['quotient'].values
    std_quotient = stds[(stds['procedure'] == procedure) & (stds['total_splits'] == total_splits)]['quotient'].values
    if mean_quotient.size > 0:
        normalized_time = total_time / train_test_split_time
        mean_pareto_front_points.append((normalized_time, mean_quotient[0], procedure, total_splits))


# Sort the points by normalized computing time (x-axis)
pareto_front_points.sort(key=lambda x: x[0])
mean_pareto_front_points.sort(key=lambda x: x[0])

# Initialize the Pareto front list
pareto_front = []
mean_pareto_front = []

# Iterate over the sorted points to find the Pareto front
for normalized_time, mean_quotient, procedure, total_splits in mean_pareto_front_points:
    if not mean_pareto_front or mean_quotient > mean_pareto_front[-1][1]:
        mean_pareto_front.append((normalized_time, mean_quotient, procedure, total_splits))
    else:
        mean_non_pareto_front_points.append((normalized_time, mean_quotient, procedure, total_splits))

# Plotting
plt.figure(figsize=(4, 3))

# Extract computing times, mean quotients, and procedures from the Pareto front
if mean_pareto_front:
    computing_times, mean_quotients, procedures, total_splits_list = zip(*mean_pareto_front)
    # Create a color map for procedures
    color_map = {'train_test_split': 'black', 'RepeatedKFold': 'C1', 'ShuffleSplit': 'C0'}

    # Plot each point with the corresponding color
    for normalized_time, mean_quotient, procedure, total_splits in mean_pareto_front:
        color = color_map.get(procedure, 'C2')  # Default to 'C2' if procedure not in color_map
        plt.plot(normalized_time, mean_quotient, marker='o', color=color)

# Extract computing times, mean quotients, and procedures from the non-Pareto front points
if mean_non_pareto_front_points:
    for normalized_time, mean_quotient, procedure, total_splits in mean_non_pareto_front_points:
        color = color_map.get(procedure, 'C2')  # Default to 'C2' if procedure not in color_map
        plt.plot(normalized_time, mean_quotient, marker='o', color=color)

# Create custom legend handles for procedures
handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='None', label=procedure) for procedure, color in color_map.items()]

# Add linear fits and confidence intervals for RepeatedKFold and ShuffleSplit
for procedure in ['RepeatedKFold', 'ShuffleSplit']:
    procedure_points = [(x, y) for x, y, p, total_splits in pareto_front_points if p == procedure]
    if procedure_points:
        x_vals, y_vals = zip(*procedure_points)
        log_x_vals = np.log10(x_vals)
        log_y_vals = np.log10(y_vals)
        coeffs = np.polyfit(log_x_vals, log_y_vals, 1)
        fit_line = 10 ** (coeffs[1] + coeffs[0] * log_x_vals)
        plt.plot(x_vals, fit_line, label=f'{procedure} Fit', linestyle='-', color=color_map[procedure])

        # Calculate confidence intervals
        residuals = log_y_vals - (coeffs[0] * log_x_vals + coeffs[1])
        std_error = np.std(residuals)
        ci_upper = 10 ** (coeffs[1] + coeffs[0] * log_x_vals + 1.96 * std_error)
        ci_lower = 10 ** (coeffs[1] + coeffs[0] * log_x_vals - 1.96 * std_error)
        plt.fill_between(x_vals, ci_lower, ci_upper, color=color_map[procedure], alpha=0.2, label=f'{procedure} CI')

# Extract computing times, mean quotients, and procedures from the Pareto front
if mean_pareto_front_std:
    computing_times, mean_quotients, std_quotients, procedures, total_splits_list = zip(*mean_pareto_front_std)
    # Create a color map for procedures
    color_map = {'train_test_split': 'black', 'RepeatedKFold': 'C1', 'ShuffleSplit': 'C0'}

    # Plot each point with the corresponding color and error bar
    for normalized_time, mean_quotient, std_quotient, procedure, total_splits in mean_pareto_front_std:
        color = color_map.get(procedure, 'C2')  # Default to 'C2' if procedure not in color_map
        plt.errorbar(normalized_time, mean_quotient, yerr=std_quotient, fmt='o', color=color, label=procedure if procedure not in plt.gca().get_legend_handles_labels()[1] else "")

# Add the legends to the plot
plt.legend(handles=handles, title='Procédure', fontsize=10, loc='lower right')

plt.xlabel('Temps de calcul au seuil normalisé')
plt.ylabel('Gain d\'échantillons attendu')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.yscale('log')  # Set y-axis to logarithmic scale

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
sns.despine()
plt.savefig('newpareto.pdf')
plt.show()