import pandas as pd
import numpy as np
import os
from datetime import datetime


class dataprep_race:
    def __init__(self, dict_files, variables=None):
        self.dict = dict_files
        self.df = pd.DataFrame()
        self.df_GPS_strokes = pd.DataFrame()
        self.df_GPS_speeds = pd.DataFrame()
        self.df_results = pd.DataFrame()
        self.df_processed = pd.DataFrame()
        self.path = str
        self.type_indicator = str
        self.above_count = 0
        self.throw_count = 0
        self.throw_row_count = 0
        self.row_count = 0
        self.empty_pointer = 0
        self.not_same_len_count = 0
        self.filtered_count_speeds = 0
        self.filtered_count_strokes = 0

    def prep_raw_GPS(self):
        self.df = pd.read_excel(self.path, skiprows=8, index=False)
        if self.df.empty:
            print("Dataframe is empty")
            return
        if self.df.columns.values.tolist()[0] != "Dist. [m]":
            self.df = pd.read_excel(self.path, skiprows=7, index=False)
        self.df = self.df.transpose()
        # Use the distances as headers, the nan header is saying whether it is a speed or a stroke

        self.df.columns = self.df.iloc[0]
        self.df = self.df.ix[1:]
        columns = self.df.columns
        if columns[-1] != 2000:
            # The end columns (Internet Service:... FISA Data Service.... etc are floatings and should be deleted from the dataframe)
            self.df = self.df.ix[:, :-4]
        index_values = self.df.index.values
        if 'Speed' in index_values[0]:
            cols = self.df.columns.values.tolist()
            self.df['stroke_speed_col'] = self.df.index
            columns = ['stroke_speed_col'] + cols
            self.df = self.df.loc[:,columns]
            self.df = self.df.reset_index(drop=True)
            index_values = self.df.index.values

        strokes_list = []
        speed_list = []

        # Divide the rows in strokes and speeds at [indexvalue][0] is the type (Speed [m/s] vs. Stroke)
        for index_value in index_values:
            if 'Stroke' in self.df.loc[index_value, columns[0]]:
                strokes_list.append(index_value)
            else:
                speed_list.append(index_value)

        # Create a Strokes dataframe and a Speeds dataframe
        self.df_GPS_strokes = self.df.loc[strokes_list, :]
        self.df_GPS_speeds = self.df.loc[speed_list, :]
        # Change index from the Unnamed values to a real index and remove the Dist [m] as a header name
        self.df_GPS_strokes = self.df_GPS_strokes.reset_index(drop=True)
        self.df_GPS_strokes.index.names = ['index']
        self.df_GPS_strokes = self.df_GPS_strokes.drop(self.df_GPS_strokes.columns[0], axis=1)
        column_values_strokes = self.df_GPS_strokes.columns.values.tolist()

        # Change the index from the countries to a normal index, keeping the countries at the second place
        self.df_GPS_speeds = self.df_GPS_speeds.reset_index()
        self.df_GPS_speeds.index.names = ['index']
        self.df_GPS_speeds.rename(columns={'index': 'countries'}, inplace=True)
        self.df_GPS_speeds = self.df_GPS_speeds.drop(self.df_GPS_speeds.columns[1], axis=1)

        # Delete all rows that contain NaN's (In this data there are only rows filled with only NaN's or completely
        # NaN-less. Therefore we can use 'any'. This is necessary because there is a collumn with countries which is
        # not NaN but 'Unnamed'
        # self.df_GPS_strokes = self.df_GPS_strokes.dropna(how='any', axis=0)
        # self.df_GPS_speeds = self.df_GPS_speeds.dropna(how='any', axis=0)
        # column names type is int
        self.df_GPS_strokes = self.df_GPS_strokes[pd.notnull(self.df_GPS_strokes[1000])]
        self.df_GPS_speeds = self.df_GPS_speeds[pd.notnull(self.df_GPS_speeds[1000])]

        # Add the countries column to the strokes df
        self.df_GPS_strokes['countries'] = self.df_GPS_speeds['countries']
        new_col_order = ['countries'] + column_values_strokes
        self.df_GPS_strokes = self.df_GPS_strokes.loc[:, new_col_order]

    def create_names_ranks_lanes_df(self, not_nan_df, col_names):
        """
        Gets the names, ranks, lanes and countries out of the dataframe and forms a new overseeable dataframe
        :param not_nan_df: Df that shows where in the original df the values are nan. Is a dataframe of the same size
        as the original dataframe
        :param col_names: Names of the columns that we need to loop over
        :return: a DataFrame containing all names, ranks, lanes and countries from the excel file
        """
        name_list1 = []
        name_list2 = []
        name_list3 = []
        name_list4 = []
        name_list5 = []
        name_list6 = []
        name_list7 = []
        name_list8 = []
        rank_list = []
        lane_list = []
        country_list = []
        name_count = 0
        lane_count = 0
        for row_nr in range(self.df_results.iloc[:, 0].shape[0]):
            if not_nan_df.iloc[row_nr, 0]:
                if not isinstance(self.df_results.iloc[row_nr, 0], unicode):
                    name_count = 0
                    lane_count = 0
                    rank_list.append(self.df_results.iloc[row_nr, 0])
            for col_nr, column_name in enumerate(col_names):
                current_cell = self.df_results.iloc[row_nr, col_nr]
                if not_nan_df.iloc[row_nr, col_nr] and col_nr > 0 and lane_count == 0:
                    lane_list.append(self.df_results.iloc[row_nr, col_nr])
                    lane_count += 1
                if isinstance(current_cell, unicode):
                    current_cell_list = []
                    if not_nan_df.iloc[row_nr, col_nr]:
                        # Sometimes there is a newline in a cell, and sometimes it contains two names.
                        # These names should both be considered,
                        if '\n' in current_cell:
                            splitted = current_cell.split('\n')
                            for cell in splitted:
                                if len(cell.split()) > 1:
                                    current_cell_list.append(cell)
                        # Sometimes two names are in one cell
                        elif len(current_cell.split()) == 4:
                            splitted = current_cell.split()
                            sur_name_check = splitted[0] + splitted[1]
                            if not sur_name_check.isupper():
                                first_name = splitted[0] + ' ' + splitted[1]
                                second_name = splitted[2] + ' ' + splitted[3]
                                current_cell_list = [first_name, second_name]
                            else:
                                current_cell_list.append(current_cell)
                        # Sometimes two names, of which one has more than one surname, are in one cell
                        elif len(current_cell.split()) == 5:
                            splitted = current_cell.split()
                            sur_name_check1 = splitted[0] + splitted[1]
                            sur_name_check2 = splitted[2] + splitted[3]
                            if not sur_name_check1.isupper() and sur_name_check2.isupper():
                                first_name = splitted[0] + ' ' + splitted[1]
                                second_name = splitted[2] + ' ' + splitted[3] + ' ' + splitted[4]
                                current_cell_list = [first_name, second_name]
                            elif sur_name_check1.isupper() and not sur_name_check2.isupper():
                                first_name = splitted[0] + ' ' + splitted[1] + ' ' + splitted[2]
                                second_name = splitted[3] + ' ' + splitted[4]
                                current_cell_list = [first_name, second_name]
                            else:
                                current_cell_list.append(current_cell)
                        # Sometimes three names are in one cell
                        elif len(current_cell.split()) == 6:
                            splitted = current_cell.split()
                            first_name = splitted[0] + ' ' + splitted[1]
                            second_name = splitted[2] + ' ' + splitted[3]
                            third_name = splitted[4] + ' ' + splitted[5]
                            current_cell_list = [first_name, second_name, third_name]
                        else:
                            current_cell_list.append(current_cell)
                        # Loop over all possible names in the cell
                        for current_cell in current_cell_list:
                            if 'Point' in current_cell:
                                continue
                            # If it is the first name of a rower in this boat, put it in name_list1
                            # Second name in name_list2, third name in name_list3 etc.
                            # This means that if there are 8 rowers in a boat, all 8 namelists get a name
                            if len(current_cell.split()) > 1 and name_count == 0:
                                name_list1.append(current_cell)
                                name_list2.append('NaN')
                                name_list3.append('NaN')
                                name_list4.append('NaN')
                                name_list5.append('NaN')
                                name_list6.append('NaN')
                                name_list7.append('NaN')
                                name_list8.append('NaN')
                                name_count += 1
                            elif len(current_cell.split()) > 1 and name_count == 1:
                                name_list2[len(name_list2)-1] = current_cell
                                name_count += 1
                            elif len(current_cell.split()) > 1 and name_count == 2:
                                name_list3[len(name_list3)-1] = current_cell
                                name_count += 1
                            elif len(current_cell.split()) > 1 and name_count == 3:
                                name_list4[len(name_list4)-1] = current_cell
                                name_count += 1
                            elif len(current_cell.split()) > 1 and name_count == 4:
                                name_list5[len(name_list5)-1] = current_cell
                                name_count += 1
                            elif len(current_cell.split()) > 1 and name_count == 5:
                                name_list6[len(name_list6)-1] = current_cell
                                name_count += 1
                            elif len(current_cell.split()) > 1 and name_count == 6:
                                name_list7[len(name_list7)-1] = current_cell
                                name_count += 1
                            elif len(current_cell.split()) > 1 and name_count == 7:
                                name_list8[len(name_list8)-1] = current_cell
                                name_count += 1
                            #Get the country name for each boat. This can be 3 letters or 3 letters + a number long
                            elif len(current_cell.split()) == 1 and len(current_cell) == 3 and current_cell.isalpha():
                                # Elm means eliminated and DNS means Did not start. Are strings who can be confused for
                                # countries
                                if current_cell != 'ELM' and current_cell != 'DNS':
                                    country_list.append(current_cell)
                            # Check if the current string is a country
                            elif len(current_cell.split()) == 1 and len(current_cell) == 4 and current_cell[:-1].isalpha():
                                country_list.append(current_cell)
        all_list = [lane_list, name_list1, name_list2, name_list3, name_list4, name_list5, name_list6,
                    name_list7, name_list8, country_list]
        if all(len(x) == len(rank_list) for x in all_list):
            name_df = pd.DataFrame({'countries': country_list, '2000m_rank': rank_list, 'start_lane': lane_list,
                                    'Name1': name_list1, 'Name2': name_list2, 'Name3': name_list3, 'Name4': name_list4,
                                    'Name5': name_list5, 'Name6': name_list6, 'Name7':name_list7, 'Name8':name_list8})
        else:
            print('names not all the same length')
            self.not_same_len_count += 1
            name_df = pd.DataFrame()
        return name_df

    def create_time_betweenranks_df(self, not_nan_df, col_names):
        current_rank = 1
        time_list1 = []
        time_list2 = []
        time_list3 = []
        time_list4 = []
        rank_list1 = []
        rank_list2 = []
        rank_list3 = []
        rank_list4 = []
        between_time_list1 = []
        between_time_list2 = []
        between_time_list3 = []
        between_rank_list1 = []
        between_rank_list2 = []
        between_rank_list3 = []
        for row_nr in range(self.df_results.iloc[:, 0].shape[0]):
            time_count = 0
            rank_count = 0
            rank1 = 0
            rank2 = 0
            rank3 = 0
            rank4 = 0
            time1 = 0
            time2 = 0
            time3 = 0
            time4 = 0
            for col_nr, column_name in enumerate(col_names):
                current_cell = self.df_results.iloc[row_nr, col_nr]
                if isinstance(current_cell, float) or isinstance(current_cell, int):
                    if not_nan_df.iloc[row_nr, col_nr]:
                        if current_cell < 0 and rank_count == 0 and col_nr > 6:
                            rank1 = -current_cell
                            rank_count += 1
                        elif current_cell < 0 and rank_count == 1 and col_nr > 6:
                            rank2 = -current_cell
                            rank_count += 1
                        elif current_cell < 0 and rank_count == 2 and col_nr > 6:
                            rank3 = -current_cell
                            rank_count += 1
                        elif current_cell < 0 and rank_count == 3 and col_nr > 6:
                            rank4 = -current_cell
                            rank_count += 1
                        # Sometimes the final ranks are not in the excel, since the boats are in the order best to worst
                        # we can just count the boats and know the final rank. Unfortunately this does not take into
                        # account whether this is an a/b/c final
                        elif time_count == 4 and rank_count == 3 and col_nr == (len(col_names)-1):
                            rank4 = current_rank
                elif isinstance(current_cell, unicode):
                    if not_nan_df.iloc[row_nr, col_nr]:
                        if ':' in current_cell and time_count == 0:
                            if '\n' in current_cell:
                                current_cell = current_cell[:7]
                            time1 = (datetime.strptime(current_cell, '%M:%S.%f')-datetime(1900,1,1)).total_seconds()
                            time_count += 1
                        elif ':' in current_cell and time_count == 1:
                            if '\n' in current_cell:
                                current_cell = current_cell[:7]
                            time2 = (datetime.strptime(current_cell, '%M:%S.%f')-datetime(1900,1,1)).total_seconds()
                            time_count += 1
                        elif ':' in current_cell and time_count == 2:
                            if '\n' in current_cell:
                                current_cell = current_cell[:7]
                            time3 = (datetime.strptime(current_cell, '%M:%S.%f')-datetime(1900,1,1)).total_seconds()
                            time_count += 1
                        elif ':' in current_cell and time_count == 3:
                            if '\n' in current_cell:
                                current_cell = current_cell[:7]
                            time4 = (datetime.strptime(current_cell, '%M:%S.%f')-datetime(1900,1,1)).total_seconds()
                            time_count += 1
                elif column_name == 'Prog. Code':
                    rank4 = float('NaN')

            if time_count == 4:
                time_list1.append(time1)
                time_list2.append(time2)
                time_list3.append(time3)
                time_list4.append(time4)
                rank_list1.append(rank1)
                rank_list2.append(rank2)
                rank_list3.append(rank3)
                rank_list4.append(rank4)
                current_rank += 1
            elif time_count == 3:
                between_time_list1.append(time1)
                between_time_list2.append(time2)
                between_time_list3.append(time3)
                between_rank_list1.append(rank1)
                between_rank_list2.append(rank2)
                between_rank_list3.append(rank3)

        all_list = [rank_list1, time_list2, rank_list2, time_list3, rank_list3, time_list4, rank_list4,
                    between_time_list1, between_rank_list1, between_time_list2, between_rank_list2, between_time_list3,
                    between_rank_list3]
        if all(len(x) == len(time_list1) for x in all_list):
            time_df = pd.DataFrame({'500m_time': time_list1, '500m_rank': rank_list1, '1000m_time': time_list2,
                                '1000m_rank': rank_list2, '1500m_time': time_list3, '1500m_rank': rank_list3,
                                '2000m_time': time_list4, '500-1000_time': between_time_list1,
                                '500-1000_rank': between_rank_list1, '1000-1500_time': between_time_list2,
                                '1000-1500_rank': between_rank_list2, '1500-2000_time': between_time_list3,
                                '1500-2000_rank': between_rank_list3})
        else:
            print('times/ranks not all the same length')
            self.not_same_len_count += 1
            time_df = pd.DataFrame()
        return time_df

    def prep_raw_results(self):
        date_count = 0
        self.df = pd.read_excel(self.path, skiprows=3, index=False, header=None)
        if self.df.empty or len(self.df.iloc[:,0]) < 8:
            print('empty')
            self.empty_pointer = 1
        else:
            # Chop off unimportant data
            rank_count = 0
            start_row = 0
            column_names = []
            for row_nr in range(self.df.iloc[:,0].shape[0]):
                # Make sure the df only contains the important information
                current_row = self.df.iloc[row_nr,:].tolist()
                if not date_count:
                    try:
                        date = pd.to_datetime(current_row[0])
                        date_count = 1
                    except:
                        print('')

                if 'Rank' in current_row and rank_count == 0:
                    self.df_results = self.df.ix[row_nr:, :]
                    self.df_results.columns = self.df_results.iloc[0]
                    self.df_results = self.df_results[1:]
                    self.df_results = self.df_results.reset_index(drop=True)
                    start_row = row_nr
                    rank_count += 1
                    column_names = self.df_results.columns.values.tolist()
                # If there are many rowers, the pdf consists of 2 pages, which creates a second row of headers after a
                # number of x rowers (For example two whole 8 boats are on the second page)
                elif 'Rank' in current_row and rank_count > 0:
                    deletable_rows = [row_nr-2, row_nr-2, row_nr]
                    self.df_results = self.df_results.drop(self.df_results.index[deletable_rows])
                if isinstance(current_row[0], unicode) and (row_nr-(start_row+2)) > 3 and rank_count > 0:
                    self.df_results = self.df_results.ix[:row_nr-(start_row+2), :]
                    break

            # Put the information under the correct column names

            # create a new df containing the following columns:
            # rank, lane, countries, rower1....8,
            if not self.df_results.empty:
                col_names = self.df_results.columns.values.tolist()
                not_nan_df = pd.notnull(self.df_results)
                # print(self.df_results)
                name_df = self.create_names_ranks_lanes_df(not_nan_df, col_names)
                if not name_df.empty:
                    time_df = self.create_time_betweenranks_df(not_nan_df, col_names)
                else:
                    time_df = pd.DataFrame()
                if name_df.empty or time_df.empty:
                    self.empty_pointer = 1
                else:
                    col_len = len(name_df.iloc[:,0])
                    date_list = [date] * col_len
                    name_df.insert(0, 'date', date_list)
                    self.df_results = pd.concat([name_df, time_df], axis=1)
                    self.df_results.index.names = ['index']

            # print(self.df_results)



    def decimalization(self, row_indexes):
        self.df = self.df.reset_index(drop=True)
        decimalizer = lambda x: float(x) / 10.0
        for row_index in row_indexes:
            self.df.loc[row_index, '50':] = self.df.loc[row_index, '50':].map(decimalizer)

    def cont_round_check(self, cont_round):
        """
        Checks whether cont_round has a value that is actually the same as an other more occuring value (Like S1 and SA1)
        and gives cont_round the more occurring value
        :param cont_round: contest round (Heats/Semifinals/Finals.....)
        :return: consistent cont_round
        """
        if cont_round[0] == ' ':
            cont_round = cont_round[1:]
        elif cont_round[0] == '-':
            cont_round = cont_round[2:]
        if cont_round[-1] == ' ':
            cont_round = cont_round[:-1]
        if cont_round == 'S1':
            cont_round = 'SA1'
        elif cont_round == 'S2' or cont_round == 's2':
            cont_round = 'SA2'
        elif cont_round == 'R' or cont_round == 'r':
            cont_round = 'R1'
        elif cont_round == 'F1':
            cont_round = 'FA'
        elif cont_round == 'EX1':
            cont_round = 'EX'
        return cont_round

    def create_neighborhood(self, c_ind):
        """
        :param c_ind: column index (pointing at a column which has a faulty value)
        :return: A range of numbers around the faulty point to use in creating a suitable value for that point
        """
        if c_ind > 7 and c_ind < 41:
            index_array = range(c_ind - 3, c_ind + 4)
        elif c_ind > 6 and c_ind < 41:
            index_array = range(c_ind - 2, c_ind + 4)
        elif c_ind > 5 and c_ind < 41:
            index_array = range(c_ind - 1, c_ind + 4)
        elif c_ind == 5:
            index_array = range(c_ind, c_ind + 4)
        elif c_ind > 7 and c_ind < 42:
            index_array = range(c_ind - 3, c_ind + 3)
        elif c_ind > 7 and c_ind < 43:
            index_array = range(c_ind - 3, c_ind + 2)
        elif c_ind > 7 and c_ind < 44:
            index_array = range(c_ind - 3, c_ind + 1)
        elif c_ind == 44:
            index_array = range(c_ind - 3, c_ind)
        # Actually there is no way that it comes here
        else:
            print('[%s: weird case]' % c_ind)
        return index_array

    def change_columns(self, r_ind, c_ind):
        """
        :param r_ind: Index of the row in which the faulty columns are situated
        :param c_ind: the index of the faulty column in r_ind
        :return: A row where the faulty values are replaced by an approximation of the true value
        """
        # Since the speed goes up fast at the beginning, and then lowers a great amount after the first 150 meters, we
        # do not want to take these measurements into the filtering, since they are (legitimized) over the threshold
        if c_ind > 2:
            # the index of te columns should start at 5 (has moved a bit since we only took the numerical values for the
            # truthtable)
            c_ind += 5
            neighborhood_list = []

            # Creates a range which we can use to make a new value for the faulty value
            index_array = self.create_neighborhood(c_ind)
            for i in index_array:
                # Do not use the faulty value for the creation of the new value
                if i == c_ind:
                    continue
                neighborhood_list.append(self.df.iloc[r_ind, i])
            # Use the average of the range to replace the faulty value
            average = np.mean(neighborhood_list)

            # If the average is nan: Replace the faulty value by zero. Otherwise give it the new average value
            if np.isnan(average):
                self.df.iloc[r_ind, c_ind] = 0
            else:
                self.df.iloc[r_ind, c_ind] = float(average)

    def filter_data(self, threshold, type_filter):
        # Create a dataframe containing the differences between the values (per row)
        diff = self.df.loc[:,'50':].diff(axis=1)

        # Counting the total number of rows (for percentage later on)
        self.row_count += diff.shape[0]
        # print('row count: %s' % self.row_count)

        # Create a truth table; Is true where the difference is above the threshold
        truth_table = abs(diff) >= threshold
        # Identify where the truthtable is true
        indexes = np.where(truth_table == True)
        row_indexes = indexes[0]
        column_indexes = indexes[1]

        # If there are any true's (meaning faulty values) go into the replacement process
        if len(row_indexes) > 0:
            # if threshold == 1 or threshold == 2:
            #     self.filtered_count_speeds += 1
            #     print('filtered count speeds: %s' % self.filtered_count_speeds)
            # elif threshold == 4 or threshold == 10:
            #     self.filtered_count_strokes += 1
            #     print('filtered count strokes: %s' % self.filtered_count_strokes)
            self.above_count += 1
            last_row_index = np.nan
            count_wrong_per_row = 0
            # Here the row_col_combi_index is the index combining the rows and columns that belong together (forming
            # one faulty value)
            for row_col_combi_index, r_ind in enumerate(row_indexes):
                if last_row_index == r_ind:
                    count_wrong_per_row += 1
                else:
                    count_wrong_per_row = 0
                    throw_pointer = 0
                last_row_index == r_ind
                # Not yet in the control fase, but in the replacement fase (so keeping the rows 'in')
                if type_filter == 'in':
                    # Count the number of columns that are faulty values per row. If more than 5, the column is not used
                    if r_ind == row_indexes[row_col_combi_index-1]:
                        if count_wrong_per_row > 9:
                            self.df.iloc[r_ind, 5:] = [np.nan] * len(self.df.iloc[r_ind, 5:])
                            if not throw_pointer:
                                self.throw_row_count += 1
                                print('throw count: %s' %self.throw_count)
                            throw_pointer = 1
                        else:
                            c_ind = column_indexes[row_col_combi_index]
                            self.change_columns(r_ind, c_ind)
                    else:
                        c_ind = column_indexes[row_col_combi_index]
                        self.change_columns(r_ind, c_ind)

                # In the control fase. If there are still values above a (higher) threshold, do not use these rows
                else:
                    self.df.iloc[r_ind, 5:] = [np.nan] * len(self.df.iloc[r_ind, 5:])
                    self.throw_row_count += 1
                    print('throw count: %s' % self.throw_count)




    def csv_to_df(self, key, count):
        """
        :param: key: The key corresponding to the current file in the files dictionary
        :param: count: The count stating how many files are already processed
        :return: creates a df called df_processed, which represents all data of one type (GPS or Results) in one dataframe

        """
        self.df = pd.read_csv(self.path, index_col='index')
        # Some of the files are empty, therefore we cannot use them. The filename is printed in combination with the
        # message that it is empty
        if self.df.empty:
            print 'empty'
            self.throw_count += 1
        else:
            if 'speeds' in key or 'strokes' in key:
                col_len = len(self.df['1000'])
            else:
                col_len = len(self.df['500m_time'])
            # get the type of the boat, the contest name, the contest round and the year of the contest from the title
            # of the file
            key_list = key.split(' ')
            key_list = [item for item in key_list if item != '']
            if 'speed' in key or 'stroke' in key:
                boattype = key_list[4]
                contest = key_list[2]
                cont_round = key_list[6]
                if cont_round[-1] == '-':
                    cont_round = cont_round[:-1]
            else:
                boattype = key_list[4]
                contest = key_list[2]
                cont_round = key_list[6]
                if cont_round[-1] == '-':
                    cont_round = cont_round[:-1]

            boattype_list = [boattype] * col_len
            contest_list = [contest] * col_len
            # Change the names of the contest round in such a way that all competitions have the same representation
            # and add them to a list of the length of the dataframe
            cont_round = self.cont_round_check(cont_round)
            round_list = [cont_round] * col_len
            year = key[0:4]
            year_list = [year] * col_len

            # put the boattype, round, contest and year in the dataframe that will be concatenated to the total dataframe
            self.df.insert(0, 'boattype', boattype_list)
            self.df.insert(0, 'round', round_list)
            self.df.insert(0, 'contest', contest_list)
            self.df.insert(0, 'year', year_list)

            # The values of the speed should never be above 10 m/s, since in some of the data it is higher
            # it should be divided by 10. (The problem is a missing dot)
            if 'speeds' in key:
                truth_table = self.df.loc[:, '50':] > 10
                row_indexes = set(np.where(truth_table == True)[0])
                self.type_indicator = 'speeds'
                if len(row_indexes) > 0:
                    self.decimalization(row_indexes)
            # The values of the strokes should never be above 100 strokes/minute, for the same reasons as above
            elif 'strokes' in key:
                truth_table = self.df.loc[:, '50':] > 100
                row_indexes = set(np.where(truth_table == True)[0])
                self.type_indicator = 'strokes'
                if len(row_indexes) > 0:
                    self.decimalization(row_indexes)

            # If there is not a total dataframe yet (df_processed) it will be the same as the current dataframe
            if count == 0:
                self.df_processed = self.df
            # If it is not the first df, it should be concatenated to the total dataframe (df_processed)
            else:
                # Sometimes there are some strange values in the beginning and in these cases the value at 50 meters is
                # non existend. We don't want to use these dataframes as they will corrupt the data
                if 'speeds' in key and (pd.isnull(self.df.iloc[:, 5]) == True).any():
                    print 'strange values'
                    print self.dict[key]
                    self.throw_count += 1
                else:
                    # Replace phase
                    if self.type_indicator == 'speeds':
                        threshold = 1
                        self.filter_data(threshold, type_filter='in')
                    elif self.type_indicator == 'strokes':
                        threshold = 4
                        self.filter_data(threshold, type_filter='in')
                    # Control phase
                    if self.type_indicator == 'speeds':
                        threshold = 2
                        self.filter_data(threshold, type_filter='out')
                    elif self.type_indicator == 'strokes':
                        threshold = 10
                        self.filter_data(threshold, type_filter='out')

                    self.df_processed = pd.concat([self.df_processed, self.df])
                    self.df_processed = self.df_processed.reindex_axis(self.df.columns, axis=1)
