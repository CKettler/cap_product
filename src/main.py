import pandas as pd
import data_preprocessing_race as ppr
import feature_creation_GPS as fcg
import feature_creation_results as fcr
import feature_creation_combined as fcc
import retrieve_files as rf
import dataframe_combiner as dc
from datetime import datetime
import os
import sys
import getopt

def read_input(argv):
    gps_from_raw = True
    results_from_raw = True
    try:
        opts, args = getopt.getopt(argv, "hg:r:")
    except getopt.GetoptError:
        print 'main.py -g <True/False> -r <True/False>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -g <True/False> -r <True/False>'
            sys.exit()
        elif opt in ("-g", "--gps_from_raw"):
            gps_from_raw = arg
        elif opt in ("-r", "--results_from_raw"):
            results_from_raw = arg
    return gps_from_raw, results_from_raw


def retrieve_files():
    # The path that contains all used xls files
    directory_path = '../data/Racedata/'
    # All xls files are retrieved and split in a dictionary containing all filenames that have GPS data and a dictionary
    # containing all filenames that have RESULTS data
    file_retriever = rf.retrieve_files(directory_path)
    file_dict_GPS, file_dict_results = file_retriever.get_xls_files()
    # print str(len(file_dict_GPS)) + " GPS files found", datetime.now().time()
    # print str(len(file_dict_results)) + " results files found", datetime.now().time()
    return file_dict_GPS, file_dict_results


def make_data_GPS(file_dict_GPS):
    """
    :param: file_dict_GPS
    :return: Nothing, just places parsed files in the folders Speeds and Strokes. These folders are in each year containing
     the parsed files for that year
    """
    prep_GPS = ppr.preprocess_race(file_dict_GPS)
    prep_GPS.read_raw_GPS()

def make_data_results(file_dict_results):
    """
        :param: file_dict_results
        :return: Nothing, just places parsed files in the folder Results. These folders are in each year containing
         the parsed files for that year
        """
    prep_results = ppr.preprocess_race(file_dict_results)
    prep_results.read_raw_results()

def merge_two_dicts(x, y):
    """Given two dicts x and y, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def make_dfs_speeds_strokes():
    """
    Creates two dataframes, one containing all speed data and one containing all strokes data
    """
    file_name_speeds = '../data/Racedata/total_file_speeds.csv'
    file_name_strokes = '../data/Racedata/total_file_strokes.csv'

    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print("[%s: Loading speeds and strokes feature file...]" % time)

    # If both total files are present no new files are created, but the df's are taken directly form the csv files
    # (Saves time)
    if os.path.isfile(file_name_speeds) and os.path.isfile(file_name_strokes):
        df_speeds = pd.read_csv(file_name_speeds, index_col=0)
        df_strokes = pd.read_csv(file_name_strokes, index_col=0)

    # If one or both total files are missing, create new ones from the files in the speeds and strokes folders
    else:
        print('Could not find speeds and strokes files')
        all_speeds_files_dict = {}
        all_strokes_files_dict = {}
        # loop through all available years (Hardcoded limited from 2013 to 2016) If one wants to include 2017 change 7 into 8
        for i in range(3, 7):
            path_speeds = '../data/Racedata/201' + str(i) + '/Speeds/'
            path_strokes = '../data/Racedata/201' + str(i) + '/Strokes/'

            # All files are retrieved from the specified speeds folder and added to the all_speeds_files_dict
            file_retriever_speeds = rf.retrieve_files(path_speeds)
            speeds_files_dict = file_retriever_speeds.get_csv_files()
            all_speeds_files_dict = merge_two_dicts(all_speeds_files_dict, speeds_files_dict)

            # All files are retrieved from the specified strokes folder and added to the all_strokes_files_dict
            file_retriever_strokes = rf.retrieve_files(path_strokes)
            strokes_files_dict = file_retriever_strokes.get_csv_files()
            all_strokes_files_dict = merge_two_dicts(all_strokes_files_dict, strokes_files_dict)

        # preproces speeds data to a total df
        data_preparator_speeds = ppr.preprocess_race(all_speeds_files_dict)
        df_speeds = data_preparator_speeds.read_csv_GPS()

        # preproces strokes data to a total df
        data_preparator_strokes = ppr.preprocess_race(all_strokes_files_dict)
        df_strokes = data_preparator_strokes.read_csv_GPS()
    # Is returned to main
    return df_speeds, df_strokes

def make_df_results():
    """
    Creates one dataframe, containing all results data
    """
    file_name_results = '../data/Racedata/total_file_results.csv'

    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print("[%s: Loading results feature file...]" % time)

    # If both total files are present no new files are created, but the df's are taken directly form the csv files
    # (Saves time)
    if os.path.isfile(file_name_results) and os.path.isfile(file_name_results):
        df_results = pd.read_csv(file_name_results, index_col=0)

    # If the file is missing, create a new one from the files in the results folders
    else:
        print('Could not find results file')
        all_results_files_dict = {}
        # loop through all available years (Hardcoded limited from 2013 to 2016) If one wants to include 2017 change 7 into 8
        for i in range(3, 7):
            path_results = '../data/Racedata/201' + str(i) + '/Results/'

            # All files are retrieved from the specified speeds folder and added to the all_speeds_files_dict
            file_retriever_results = rf.retrieve_files(path_results)
            results_files_dict = file_retriever_results.get_csv_files()
            all_results_files_dict = merge_two_dicts(all_results_files_dict, results_files_dict)

        # preproces results data to a total df
        data_preparator_results = ppr.preprocess_race(all_results_files_dict)
        df_results = data_preparator_results.read_csv_results()

    # Is returned to main
    return df_results


def create_all_df():
    """
    :param feature_creation: if True, new features are generated from the parsed data. If False only the parsed data
    will be in df_all
    :return: df_all, a dataframe with both the GPS and the results data
    """
    file_name_all = '../data/Racedata/total_file_all.csv'

    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print("[%s: Loading all data file...]" % time)

    # If the combined total file is present no new files are created
    # (Saves time)
    if os.path.isfile(file_name_all):
        df_all = pd.read_csv(file_name_all, index_col=0)
    else:
        # Create the speeds and strokes dataframe from the csv's that are already created from the excel files
        df_speeds, df_strokes = make_dfs_speeds_strokes()
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: Loaded Speeds and Strokes df...]" % time)

        # Create the results dataframe from the csv's that are already created from the excel files
        df_results = make_df_results()
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: Loaded Results df...]" % time)

        # Combine all dataframes in one dataframe (df_all)
        combinator = dc.combine_dataframes(df_speeds, df_strokes, df_results)
        df_all = combinator.combine_all()
    return df_all

def create_features(df_all):
    file_name_all = '../data/Racedata/total_file_all_feat.csv'

    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print("[%s: Loading strategy+features data file...]" % time)
    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print("[%s: Loading features data file...]" % time)
    # If the feature file without strategy cats is present only the new file with strategies is created
    if os.path.isfile(file_name_all):
        df_all = pd.read_csv(file_name_all, index_col=0)
    else:
        gps_feature_creator = fcg.feature_creation_gps(df_all, 4)
        df_all = gps_feature_creator.create_all_features()

        results_feature_creator = fcr.feature_creation_results(df_all)
        df_all = results_feature_creator.create_all_features()

        combined_feature_creator = fcc.feature_creation_combined(df_all)
        df_all = combined_feature_creator.create_all_features()

        df_all.to_csv('../data/Racedata/total_file_all_feat.csv')
    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print("[%s: Loaded strat+features data file]" % time)
    return df_all


"""Beneath all functions are called"""
if __name__ == '__main__':

    gps_from_raw, results_from_raw = read_input(sys.argv[1:])

    file_dict_GPS, file_dict_results = retrieve_files()

    if gps_from_raw:
        make_data_GPS(file_dict_GPS)
    print('GPS data per year is parsed', datetime.now().time())

    if results_from_raw:
        make_data_results(file_dict_results)
    print('Results data per year is parsed', datetime.now().time())

    df_all = create_all_df()

    df_all = create_features(df_all)
