import pandas as pd
import os

import config


def seperate_tracks(df):
    '''
    Seperates individual mosquito tracks from entire track file

    Params:
     - df (DataFrame): DataFrame of data for a specific trial

    Returns:
     - tracks (arr): 2D array that contains seperated mosquito tracks (i.e. grouped datapoints of same mosquito in array) 
    '''
    mossies = df[1].unique()

    tracks = []
    for mossie in mossies:
        tracks.append(df[df[1] == mossie].drop(columns=[1]).values)

    return tracks


def load_file_from_file(path, trial_id):
    '''
    Returns numpy formatted track with following information for datapoint
        - x
        - y
        - z
        - x_dot
        - y_dot
        - z_dot

    Params:
     - path (str): Path to CSV file
     - trial_id (str or int): ID for trial

    Returns:
     - df (arr):  2D array that contains seperated mosquito tracks (i.e. grouped datapoints of same mosquito in array) 
    '''
    df = pd.read_csv(path)
    df.columns = [i for i in range(len(df.columns))]
    col_num = len(df.columns)

    while col_num < 8:
        df = pd.concat([df, pd.DataFrame({col_num: [0 for i in range(len(df))]})], axis=1)
        col_num = col_num + 1

    df = df.drop(columns=[0])
    df = seperate_tracks(df)
    
    return df


def load_trials_from_file_list(files, path_to):
    '''
    Returns all tracks from trials

    Params:
     - files (arr): list of file names
     - path_to (str): path to files directory

    Returns:
     - trials (arr): Array of all trials arrays combined
    '''
    trials = []
    for index, path in enumerate(files):
        trials.append(load_file_from_file(f'{path_to}/{path}', index))

    return trials


def load_all_files():
    '''
    Returns male and couple tracks

    Returns:
     - male_trials (arr): Array of all male trials combined
     - couples_trials (arr): Array of all couples trials combined
    '''
    files = sorted(os.listdir(config.PATHS.path_to_males))
    male_trials = load_trials_from_file_list(files, config.PATHS.path_to_males)
    
    files = sorted(os.listdir(config.PATHS.path_to_couples))
    couples_trials = load_trials_from_file_list(files, config.PATHS.path_to_couples)
    
    return male_trials, couples_trials


if __name__ == '__main__':
    # For debugging
    male, couple = load_all_files()
    print('number of trials: ', len(male))
    print('number of mosquitoes in trial: ', len(male[0]))
    print('number of positions for mosquito in trial: ', len(male[0][0]))
    print('number of datapoints for position: ', len(male[0][0][0]))
