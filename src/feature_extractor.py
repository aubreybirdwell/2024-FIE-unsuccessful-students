"""
Author: Aubrey Birdwell (design inspired by Kristian Tkacik's code for kypo) 
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import groupby
from statistics import mean, median, StatisticsError
from log_parser_edurange import Dataset, DatasetRecord, create_dataset_records, read_file

INFINITY = np.inf
MAX_COMMAND_GAP = 1200
CORR_THRESHOLD = 0.11


def get_unique_commands(record: DatasetRecord) -> set:
    """
    Return a set of unique tools from the command log in a dataset record.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        set: Unique tools used in the given training run.
    """
    return {command["cmd"][0] for command in record.user_log if command.get("cmd") is not None}


def longest_command_repetition(record: DatasetRecord) -> int:
    """
    Calculate the length of the longest sequence of a single repeating tool (command without arguments).

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        int: Length of the longest sequence of repeating tool usage.
    """
    rep_lengths = []
    cmd_seq = [
        command["cmd"]
        for command in record.user_log
        if command.get("cmd") is not None
    ]
    for _, g in groupby(cmd_seq):
        rep_lengths.append(len(list(g)))
    return max(rep_lengths)


def command_repetition_count(record: DatasetRecord) -> int:
    """
    Calculate the number of sequences of a repeating command in the dataset record.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        int: Total count of repeating command sequences in the training run.
    """
    rep_lengths = []
    cmd_seq = [
        command["cmd"]
        for command in record.user_log
        if command.get("cmd") is not None
    ]
    for _, g in groupby(cmd_seq):
        rep_lengths.append(len(list(g)))
    return len(rep_lengths) - rep_lengths.count(1)


def attempt_count(record: DatasetRecord) -> int:
    """
    Count the number of submissions in the training run based on milestone tags.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        int: Number of answer submissions excluding those labeled as 'U' (unidentified).
    """
    return len([event for event in record.user_log if 'U' not in event.get("ms_tag")])


def user_scenario_time(record: DatasetRecord) -> int:
    """
    Calculate the total elapsed time for the scenario by subtracting the first 
    timestamp from the last timestamp.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        int: Total elapsed time for the scenario in seconds.
    """
    log_sorted = sorted(record.user_log, key=lambda d: d['time'])
    t_secs = (int(log_sorted[-1]["time"]) - int(log_sorted[0]["time"]))
    return t_secs


def commands_per_minute(record: DatasetRecord) -> float:
    """
    Calculate the number of executed commands per minute in the training run.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        float: Number of commands per minute in the training run, or 0 if elapsed time is 0.
    """
    log_sorted = sorted(record.user_log, key=lambda d: d['time'])
    t_secs = (int(log_sorted[-1]["time"]) - int(log_sorted[0]["time"]))
    return len(record.user_log) / t_secs


def avg_command_time_gap(record: DatasetRecord) -> float:
    """
    Calculate the average time gap between commands in seconds.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        float: Average time gap between commands in the given dataset record.
    """
    gaps = [
        (int(record.user_log[i]["time"]) - int(record.user_log[i - 1]["time"]))
        for i in range(1, len(record.user_log) - 1)
    ]
    return mean(gaps)


def total_cmd_errors(record: DatasetRecord) -> int:
    """
    Count the total number of errors (bash errors) in the user's log.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.

    Returns:
        int: Total number of log entries with bash errors.
    """
    return len([event for event in record.user_log if 'F' in event.get("ms_tag")])


def total_commands_difference(record: DatasetRecord, median_commands: int) -> int:
    """
    Calculate the difference between the total number of commands in the record 
    and the median number of commands for the training group.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.
        median_commands (int): Median number of commands across the training group.

    Returns:
        int: Difference between the record's total commands and the median.
    """
    return len(record.user_log) - median_commands


def unique_commands_difference(record: DatasetRecord, median_unique_cmds: int) -> int:
    """
    Calculate the difference between the number of unique tools used in the record 
    and the median number of unique tools for the training group.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.
        median_unique_cmds (int): Median unique tool count for the training group.

    Returns:
        int: Difference between the record's unique tool count and the median.
    """
    return len(get_unique_commands(record)) - median_unique_cmds


def training_time_difference(record: DatasetRecord, median_time: int) -> float:
    """
    Calculate the difference between the training time of the dataset record and 
    the median training time for the group.

    Args:
        record (DatasetRecord): DatasetRecord object representing a training run.
        median_time (int): Median training time across the training group.

    Returns:
        float: Difference between the record's training time and the median time.
    """
    return user_scenario_time(record) - median_time


def extract_features(dataset: Dataset) -> pd.DataFrame:
    """
    Extract features from the dataset to create a summary DataFrame of statistics for each record.

    Args:
        dataset (Dataset): A Dataset object containing multiple DatasetRecords.

    Returns:
        pd.DataFrame: DataFrame with extracted features and statistics for each record.
    """
    median_total_commands = median([len(record.user_log) for record in dataset.records])
    median_unique_cmds = median([len(get_unique_commands(record)) for record in dataset.records])
    median_times = median([user_scenario_time(record) for record in dataset.records])

    dataframe = pd.DataFrame({
        
        # ===== total values =====
        "scenario_times": [user_scenario_time(record) for record in dataset.records],
        "executed_cmds": [len(record.user_log) for record in dataset.records],
        "total_cmd_errors": [total_cmd_errors(record) for record in dataset.records],
        "unique_cmds": [len(get_unique_commands(record)) for record in dataset.records],
        "cmd_rep_count": [command_repetition_count(record) for record in dataset.records],
        "attempt_count": [attempt_count(record) for record in dataset.records], #rename to related_command_count

        # ===== Difference from median =====
        "total_cmds_diff": [total_commands_difference(record, median_total_commands) for record in dataset.records],
        "unique_tools_diff": [unique_commands_difference(record, median_unique_cmds)for record in dataset.records],
        "training_time_diff": [training_time_difference(record, median_times)for record in dataset.records],

        "total_unique_tags": [len(record.unique_tags) for record in dataset.records],
        "total_ms_complete": [len(record.milestones_completed) for record in dataset.records],

        
        "avg_unique_cmds_per_ms": [mean(record.ms_unique_cmds) for record in dataset.records],        
        "min_unique_cmds_per_ms": [min(record.ms_unique_cmds) for record in dataset.records],
        "max_unique_cmds_per_ms": [max(record.ms_unique_cmds) for record in dataset.records],
                       
        # ===== features concerning use of commands and tools =====
        "max_cmd_rep": [longest_command_repetition(record) for record in dataset.records],
        "cmds_per_minute": [commands_per_minute(record) for record in dataset.records],
        "avg_cmd_gap": [avg_command_time_gap(record) for record in dataset.records],

        "avg_cmds_per_ms": [mean(list(record.ms_command_count.values())) for record in dataset.records],
        "min_cmds_per_ms": [min(record.ms_command_count.items(), key=lambda x: x[1])[1] for record in dataset.records],
        "max_cmds_per_ms": [max(record.ms_command_count.items(), key=lambda x: x[1])[1] for record in dataset.records],

        # ===== features concerning answer submission =====
        "avg_time_to_ms": [mean(record.ms_times_absolute) for record in dataset.records],
        "min_time_to_ms": [min(record.ms_times_absolute) for record in dataset.records],
        "max_time_to_ms": [max(record.ms_times_absolute) for record in dataset.records],
        "avg_ms_gap": [mean(np.diff(record.ms_times)) for record in dataset.records],
        "avg_errors_per_ms": [mean(record.ms_error_count) for record in dataset.records],
        #"min_errors_per_ms": [min(record.ms_error_count) for record in dataset.records],
        "max_errors_per_ms": [max(record.ms_error_count) for record in dataset.records],
        
        # ===== target variable =====
        "struggled": [int(record.scenario_outcome) for record in dataset.records]
        #"percent_passed": [int(record.percent_completed) for record in dataset.records]
        
    })

    # Perform data cleaning (optional, uncomment if needed)
    # dataframe.replace(np.nan, sys.maxsize, inplace=True)
    # dataframe.apply(lambda col: col.fillna(col.max(), inplace=True), axis=0)
    
    return dataframe


def main(file_name: str):
    """
    Main function to parse raw dataset files, extract features, and
    output correlation matrices.

    Args:
        file_name (str): Name of the dataset file to process.

    Returns:
        None: Outputs descriptive statistics and correlation matrices to specified files.
    """
    dataset = Dataset()
    log = read_file(file_name)
    
    #Only considers the one scenario for edurange
    create_dataset_records(dataset, "file_wrangler", 13, log)
    dataset.print_stats()
    
    df = extract_features(dataset)           
    df.iloc[:, 11:].describe().transpose().to_csv("results/feature-descriptive-stats.csv")
    print(df.head())

    # Correlation matrix of all features
    df_corr_all = df.iloc[:, 9:].corr(method='kendall')
    plt.figure(figsize=(16, 14))
    sns.heatmap(df_corr_all, annot=True, square=True, linewidths=0.5, annot_kws={"fontsize": 8})
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)
    plt.savefig("results/corr_matrix_all.pdf")
    print("Correlation matrix of all extracted features exported to results/corr_matrix_all.pdf")

    # Correlation matrix of best features
    best_features = df_corr_all.loc[
        (df_corr_all['struggled'] >= CORR_THRESHOLD) |
        (df_corr_all['struggled'] <= -CORR_THRESHOLD)
    ].index
    df_corr_best = df[best_features].corr(method='kendall')
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_corr_best, annot=True, square=True, linewidths=0.5, annot_kws={"fontsize": 9})
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("results/corr_matrix_best.pdf")
    print("Correlation matrix with best extracted features exported to results/corr_matrix_best.pdf")    
    
if __name__ == "__main__":
    #proper arg check
    if len(sys.argv) != 2:
        # This script accepts the ./path/file_name and passes this to its call to the log parser
        print('usage: enter file name')
        sys.exit(1)

    file_name = sys.argv[1] 

    main(file_name)
