import pandas as pd
from data_preparation_race import *
from datetime import datetime
import catagorization as cat
import numpy as np


class preprocess_race(dataprep_race):
    def __init__(self, dict_files=None):
        self.df = pd.DataFrame()
        self.df_GPS_strokes = pd.DataFrame()
        self.df_GPS_speeds = pd.DataFrame()
        self.df_processed = pd.DataFrame()
        self.df_results = pd.DataFrame()
        self.dict = dict_files
        self.type_indicator = str
        self.above_count = 0
        self.throw_count = 0
        self.row_count = 0
        self.throw_row_count = 0
        self.empty_pointer = 0
        self.empty_count = 0
        self.not_same_len_count = 0
        self.filtered_count_speeds = 0
        self.filtered_count_strokes = 0
        print "data opened", datetime.now().time()

    def read_raw_GPS(self):
        """
        The raw GPS data from the xls files is transformed to strokes and speeds csv's and saved per year
        :return: nothing, the csv files are saved
        """
        count = 0
        # Loop over all xls(x) files in the files dict
        for key in self.dict:
            print(self.dict[key])
            self.path = self.dict[key]
            # Function prep_raw is in data_preparation_race and parses the file
            self.prep_raw_GPS()
            count += 1
            # Save GPS strokes and speeds in csv's
            key_no_extension = os.path.splitext(key)[0]
            print('../data/Racedata/' + key[0:4] + '/Speeds/' + key_no_extension + '- speeds.csv')
            print(count)
            if not self.df_GPS_strokes.empty:
                column_names = self.df_GPS_strokes.columns.values.tolist()
                column_names = [str(int(col)) if ".0" in str(col) else str(col) for col in column_names]
                self.df_GPS_strokes.columns = column_names
                self.df_GPS_strokes.to_csv(
                    '../data/Racedata/' + key[0:4] + '/Strokes/' + key_no_extension + '- strokes.csv')
            if not self.df_GPS_speeds.empty:
                column_names = self.df_GPS_speeds.columns.values.tolist()
                column_names = [str(int(col)) if ".0" in str(col) else str(col) for col in column_names]
                self.df_GPS_speeds.columns = column_names
                self.df_GPS_speeds.to_csv('../data/Racedata/' + key[0:4] + '/Speeds/' + key_no_extension + ' - speeds.csv')

    def read_raw_results(self):
        """
        The raw results data from the xls files is transformed to new results csv's and saved per year
        :return: nothing, the csv files are saved
        """
        count = 0
        # Loop over all xls(x) files in the files dict
        for key in self.dict:
            # if count < 40:
            print(self.dict[key])
            print(count)
            self.empty_pointer = 0
            self.path = self.dict[key]
            # Function prep_raw is in data_preparation_race and parses the file
            self.prep_raw_results()
            # Save results in csv's
            self.empty_count += self.empty_pointer
            if not self.empty_pointer:
                key_no_extension = os.path.splitext(key)[0]
                key_witout_results = ' '.join(str(x) for x in key_no_extension.split()[:-2])
                print(key_witout_results)
                self.df_results.to_csv(
                    '../data/Racedata/' + key[0:4] + '/Results/' + key_witout_results + ' - res.csv')

            count += 1
        used_count = count - self.not_same_len_count - self.empty_count
        print("[%s: Not same length count]" % self.not_same_len_count)
        print("[%s: used count]" % used_count)

    def read_csv_GPS(self):
        """
        Reads the created speeds and strokes files, filters and completes them and puts them into total files, which are
         saved in the Racedata folder
        :return:
        """
        count = 0
        for key in self.dict:
            self.path = self.dict[key]
            self.csv_to_df(key, count)
            count += 1
        # Calculates the percentage of files that are not used
        throw_percentage = float(self.throw_count)/float(count)
        # Calculates percentage of rows that contain incorrect measurements
        above_row_percentage = float(self.above_count)/float(self.row_count)
        print("[%s: Percentage of files not used]" % throw_percentage)
        # print("[%s: Percentage of the data that contains one or more changes above the threshold]" % above_percentage)
        print("[%s: Percentage of rows that contains one or more changes above the threshold]" % above_row_percentage)
        self.df_processed = self.df_processed.reset_index(drop=True)
        # At this moment there are only speeds files and strokes files. If there are more types this should be extended
        # with an elif statement (elif self.type_indicator == 'strokes')
        catagorizer = cat.catagorization(self.df_processed, self.type_indicator)
        cat_df = catagorizer.catagorize_all()
        if self.type_indicator == 'speeds':
            cat_df.to_csv(
                '../data/Racedata/total_file_speeds.csv')
        else:
            cat_df.to_csv(
                '../data/Racedata/total_file_strokes.csv')
        return cat_df

    def read_csv_results(self):
        """
        Reads the created results files, filters and completes them and puts them into a total file, which is
         saved in the Racedata folder
        :return: the total dataframe
        """
        count = 0
        for key in self.dict:
            self.path = self.dict[key]
            self.csv_to_df(key, count)
            count += 1

        self.df_processed = self.df_processed.reset_index(drop=True)
        # At this moment there are only speeds files and strokes files. If there are more types this should be extended
        # with an elif statement (elif self.type_indicator == 'strokes')
        catagorizer = cat.catagorization(self.df_processed, type='results')
        cat_df = catagorizer.catagorize_all()
        cat_df.to_csv('../data/Racedata/total_file_results.csv')
        return cat_df




