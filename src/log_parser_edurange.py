"""
This module provides functionality for handling cybersecurity scenario logs,
specifically for the Edurange project. It includes classes for managing datasets
and individual records, offering methods to process and analyze training logs.

Author: Aubrey Birdwell (loosely based on Kristian Tkacik's code)
"""

import csv
import sys
import json
import nltk
import datetime
import numpy
from typing import List, Optional, Any

class Dataset:
    """
    Class representing a dataset consisting of multiple DatasetRecord objects.

    Attributes:
        records (List[DatasetRecord]): A list of DatasetRecord objects representing 
            individual training sessions.
    """
    
    def __init__(self):
        """Initialize a new, empty Dataset."""
        self.records = []

    def print_stats(self):
        """
        Print statistics about the dataset, including the number of training runs,
        struggling runs, and non-struggling runs.
        """
        struggled_runs_count = len(list(filter(lambda rec: rec.scenario_outcome, self.records)))
        print(f"Number of training runs: {len(self.records)}\n"
              f"Number of struggling runs: {struggled_runs_count}\n"
              f"Number of non-struggling runs: {len(self.records) - struggled_runs_count}")

class DatasetRecord:
    """
    Class for representing a single training run, consisting of a command log and milestone events.
    
    Each DatasetRecord corresponds to an individual student's log for a given scenario or training group.
    It contains a list of dictionaries, each representing an individual log event with details about the
    command executed, the time, the working directory, user ID, and other relevant metadata.

    csv organization example:
    INPUT|uniqueID|class|time|uid|cwd|input|output (truncated to 500 char and wrapped in %%)

    
    Attributes:
        training_group (str): Name of the scenario or training group.
        ms_count (int): Number of milestones or tasks within the scenario.
        user_log (list): List of dictionaries, each representing a single command log record.
    """

    def __init__(self, training_group: str, ms_count: int, user_log: list):
        """
        Initialize a DatasetRecord.

        Args:
            training_group (str): The scenario or training group name.
            ms_count (int): The number of milestones in the scenario.
            user_log (list): The list of log entries for the user.
        """
        self.training_group = training_group
        self.ms_count = ms_count
        self.user_log = user_log

    def unique_log(self, log_data) -> list:
        """
        Return the log data after removing entries with duplicate timestamps.

        This method is useful for cases where commands may be echoed multiple times due to
        nested SSH sessions. Note: may not be needed for all scenario datasets.

        Args:
            log_data (list): The list of log entries to process.

        Returns:
            list: The deduplicated list of log entries, sorted by timestamp.
        """
        log = []
        first_entry = log_data[0]
        [log.append(x) for x in log_data[1:] if x['time'] not in log]
        log.append(first_entry)
        log_sorted = sorted(log, key=lambda d: d['time'])
        
        return log_sorted
    
    @property
    def unique_tags(self) -> set:
        """ 
        Return a set of unique tags present in the log. 

        Returns:
            set: A set of unique tags found in the user log.
        """
        return {tag["ms_tag"] for tag in self.user_log if tag.get("ms_tag") is not None}
        
    @property
    def milestones_completed(self) -> set:
        """
        Return a set of unique milestone tags representing milestones achieved in the log. 

        Returns:
            set: A set containing tags like "M1", "M2" that indicate milestones completed.
        """
        ms = set()
        for tag in self.unique_tags:
            if "M" in tag:
                if "F" not in tag:
                    ms.add(tag)
        return ms

    @property
    def milestones_attempted(self) -> set:
        """
        Return a set of unique milestone tags representing milestones attempted in the log. 

        Returns:
            set: A set containing tags like "M1", "M2", "A1" indicating attempted milestones.
        """
        ms = set()
        for tag in self.unique_tags:
            if "M" in tag:
                ms.add(tag)
            if "A" in tag:
                ms.add(tag)                
        return ms

    @property
    def percent_complete(self) -> float:
        """
        Calculate the percentage of the scenario completed.

        Returns:
            float: The completion percentage based on milestones completed out of total milestones.
        """
        return len(self.milestones_completed) / self.ms_count
        
    @property
    def scenario_outcome(self) -> bool:
        """
        Determine if the scenario meets the completion threshold.

        Returns:
            bool: True if the scenario is considered complete, based on a threshold of (6/13 milestones).
        """        
        threshold = 6 / 13
        if self.percent_complete > threshold:
            return False
        else:
            return True

    '''        
    @property
    def ms_times(self) -> dict:
        """
        Get a dictionary of completed milestones and their first appearance times.
            
        Returns:
            dict: Keys are milestone tags, and values are the first timestamps of completion.
        """
        ms_times = {}
        
        for entry in self.user_log:
            if 'M' in entry['ms_tag']:
                if 'F' not in entry['ms_tag']:
                    if entry['ms_tag'] not in ms_times:
                        ms_times[entry['ms_tag']] = int(entry['time'])
                    else:
                        pass
                    
        return ms_times
    '''

    @property
    def ms_times(self) -> [int]:
        """
        Get a list of timestamps for significant log events, including the first log entry and milestone activations.

        Returns:
            list: A list of important timestamps starting with the first log entry.
        """            
        times = []
        tags = []
        times.append(int(self.user_log[0]['time']))
        for entry in self.user_log:
            if 'M' in entry['ms_tag']:
                if 'F' not in entry['ms_tag']:
                    if entry['ms_tag'] not in tags:
                        tags.append(entry['ms_tag'])
                        times.append(int(entry['time']))
                     
        return times

    @property
    def ms_command_count(self) -> dict:
        """
        Get a dictionary of completed milestones with the count of commands issued leading up to completion.
    
        Returns:
            dict: Keys are milestone tags, and values are the command counts.
        """        
        ms_command_count = {}

        cmds = 0
        for entry in self.user_log:
            cmds += 1
            if 'M' in entry['ms_tag']:
                if 'F' not in entry['ms_tag']:
                    if entry['ms_tag'] not in ms_command_count:
                        ms_command_count[entry['ms_tag']] = cmds
                        cmds = 0
                    else:
                        pass
                    
        return ms_command_count

    @property
    def ms_error_count(self) -> [int]:
        """
        Get a list of error counts for each completed milestone.
    
        Returns:
            list: A list of error counts leading up to each milestone.
        """
        ms_labels = []
        ms_errors = []
        
        errors = 0
        for entry in self.user_log:
            if 'F' in entry['ms_tag']:
                errors += 1        
            if 'M' in entry['ms_tag']:
                if 'F' not in entry['ms_tag']:
                    if entry['ms_tag'] not in ms_labels:
                        ms_labels.append(entry['ms_tag'])
                        ms_errors.append(errors)
                        errors = 0
                    else:
                        pass
                    
        return ms_errors

    @property
    def ms_unique_cmds(self) -> [int]:
        """
        Get a list of unique command counts used per milestone activation.

        Returns:
            list: A list of numbers representing unique commands used between milestone activations.
        """
        cmds = set()
        tags = []
        ms_unique_cmds = []
        
        for entry in self.user_log:
            cmds.add(entry['cmd'][0])
            if 'M' in entry['ms_tag']:
                if 'F' not in entry['ms_tag']:
                    if entry['ms_tag'] not in tags:
                        tags.append(entry['ms_tag'])
                        ms_unique_cmds.append(len(cmds))
                        cmds = set()

        return ms_unique_cmds

    @property
    def ms_times_absolute(self) -> [int]:
        """
        Get a list of elapsed times for each milestone activation.

        Returns:
            list: A list of elapsed times for milestone activations.
        """        
        times = []
        tags = []
        
        for entry in self.user_log:            
            if 'M' in entry['ms_tag']:
                if 'F' not in entry['ms_tag']:
                    if entry['ms_tag'] not in tags:
                        tags.append(entry['ms_tag'])
                        times.append(int(entry['time']) - int(self.user_log[0]['time']))
                  
        return times

        
def create_dataset_records(dataset: Dataset, training_group: str, ms_count: int, logs: dict):
    """
    Create DatasetRecord objects from log data and add them to a Dataset.

    This function processes logs from a single scenario type (e.g., file wrangler)
    and creates DatasetRecord objects, which are added to the given Dataset.

    Args:
        dataset (Dataset): The Dataset instance where the created records are added.
        training_group (str): The name of the training group, e.g., "filewrangler".
        ms_count (int): The number of milestones expected in the scenario.
        logs (dict): A dictionary where the key is a user session and the value is a list 
                     of command log records.

    Returns:
        None
    """
    cnt = 0
    for user in logs:
        user_log = []
        for entry in logs[user]:
            cnt += 1
            log_entry = {"user" : entry[4], "ms_tag" : entry[2], "cmd" : separate_command(entry[6]), "time" : entry[3]}
            user_log.append(log_entry)
        dataset_record = DatasetRecord(training_group, ms_count, user_log)
        dataset.records.append(dataset_record)

    print("log entries: " + str(cnt))
    return
    
def read_file(file_name):
    """
    Custom CSV import to handle the way Edurange data is formatted. This function reads
    a CSV file and processes the entries into a dictionary.

    CSV organization example:
    INPUT|uniqueID|class|time|uid|cwd|input|output (truncated to 500 characters, wrapped in %%)

    Args:
        file_name (str): The path to the CSV file to be read.

    Returns:
        dict: A dictionary with all the entries from the CSV file, where the key is the
              user ID and the value is a list of log entries.
    """
    log = {}

    try:
        with open(file_name, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter="|", quotechar="%", quoting=csv.QUOTE_MINIMAL)
            
            for line in reader:
                if len(line) != 8:
                    #optional print bad lines which occasionally appear in the logs for debugging
                    #print(f"Warning: Line has unexpected number of fields and will not be included: {line}")
                    continue
                else:
                    inpt = 'INPUT'
                    uid = line[1]
                    milestone = line[2]
                    timestamp = line[3]
                    user = line[4].lower()
                    path = line[5]
                    command = line[6].split(':')
                    output = line[7].replace('\n', '')
                
                    if user not in log:
                        log[user] = []
                    
                    event = (inpt, uid, milestone, timestamp, user, path, command[1], output)
                    log[user].append(event)
                    
            csv_file.close()
            
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except IOError:
        print(f"Error: Could not read the file {file_name}.")
    
    return log


def separate_command(input_string):
    """
    Parses a command string into its components: command, flags, and options.
    
    This function splits the input string into the command, option flags, and arguments,
    sorting the flags and arguments alphabetically.

    Args:
        input_string (str): The raw command string.

    Returns:
        list: A list containing the command, sorted flags, and sorted arguments.
    """    
    cmd_list = input_string.split(" ")

    command = ""
    command += cmd_list[0]
    
    opt_flags = ""
    args = []

    for field in cmd_list[1:]:
        if "-" in field:
            opt_flags += field.strip(" -")
        else:
            args.append(field)
            
    opt_flags = "".join(sorted(opt_flags))  
    args = "".join(sorted(args))
    command_split = [command,opt_flags,args]
    
    return command_split

def main(file_name: str):
    """
    Opens a csv file and creates a dataset record object which holds
    the individual log entries.
    """
    dataset = Dataset()
    
    log = read_file(file_name)
    create_dataset_records(dataset, "file_wrangler", 13, log)
    
    dataset.print_stats()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: enter file name')
        sys.exit(1)

    file_name = sys.argv[1]

    main(file_name)
