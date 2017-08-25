import pandas as pd
import numpy as np
import exp_smoothing as smooth

class feature_creation_gps:
    def __init__(self, dataframe_all, parts):
        self.df_all = dataframe_all
        self.parts = parts
        self.temp_df = pd.DataFrame()
        self.change_pointer = 0
        self.new_all_df = pd.DataFrame()

    def create_all_features(self):
        """
        Creates all features for both the speeds and the strokes that can be made with just the speeds or just the strokes
        :return: The enriched dataframe with all data. New features can be found at the end of the dataframe
        """
        self.create_boatsize_sex_weight_feature()
        self.smooth_strokes()
        self.create_dif_feature()
        self.create_sprint_feature()
        self.create_average_strokepace_feature()
        self.create_average_sprint_feature()
        return self.df_all


    def create_boatsize_sex_weight_feature(self):
        """
        Creates a feature for the sex of the boat and the weight class of the boat, called resprectively 'sex' and
        'weight_cat'. Both are booleans, where 'Men' and 'Heavy' are represented by a 1 and 'Women' and 'Light' are
        represented by a 0. It adds these features to the data frame containing all data.
        """
        count = 0
        groups = self.df_all.groupby(['boattype'])
        for name, group in groups:
            boattype = group.loc[group.index[0], 'boattype']
            if 'M' in boattype:
                group['sex'] = 'M'
                group['sex_cat'] = 1
            else:
                group['sex'] = 'W'
                group['sex_cat'] = 0
            if 'L' in boattype:
                group['weight_class'] = 'Light'
                group['weight_cat'] = 0
            else:
                group['weight_class'] = 'Heavy'
                group['weight_cat'] = 1
            size = [int(s) for s in list(name) if s.isdigit()]
            group['boatsize'] = size[0]
            self.create_dataframe_from_groups(group, count)
            count += 1
        self.df_all = self.new_all_df

    def create_dataframe_from_groups(self, team, count):
        """
        :param team: small dataframe containing all data from one team (a group of the all_df)
        :param count: a number stating how many dataframes are already added to the new dataframe (new_all_df)
        :return: a new dataframe containing the teams that have already been processed and their new features
        """
        if count == 0:
            self.new_all_df = team
        else:
            self.new_all_df = pd.concat([self.new_all_df, team])

    def smooth_strokes(self):
        """
        Smooths the resolution irregularities out of the strokes data
        """
        nr_rows = self.df_all.shape[0]
        col_names = [str(50*x) + '_stroke' for x in range(1,41)]
        name_col_names = ['year', 'countries', 'contest', 'round', 'boattype']
        for i in range(nr_rows):
            strokes_list = self.df_all.loc[i,col_names]
            # the values of the columns used to generate a name for the plot
            name_list = self.df_all.loc[i,name_col_names].values
            name_list[0] = str(name_list[0])
            name = '_'.join(name_list)
            # performing the actual smoothing
            smoothed_strokes = smooth.smooth_plot_strokes_list(strokes_list, name, plot_indicator=False)
            # replacement of stroke rate measurements by smoothed stroke rate measurements
            self.df_all.loc[i, col_names] = smoothed_strokes

    def create_dif_feature(self):
        """
        Create two features. One that describes the average slope over a part of the race and an other that describes
        the difference between the first and the last value of the part
        :return:
        """
        # Get all columns that are stroke data and create a dataframe with all stroke data
        stroke_cols = [col for col in self.df_all.columns if 'stroke' in col and '2000' not in col]
        stroke_df = self.df_all.loc[:, stroke_cols]
        self.calculate_difference(stroke_df, measurement_type='stroke')

        # Get all columns that are speeds data and create a dataframe with all speeds data
        speed_cols = [col for col in self.df_all.columns if 'speed' in col and '2000' not in col]
        speed_df = self.df_all.loc[:, speed_cols]
        self.calculate_difference(speed_df, measurement_type='speed')

    def calculate_difference(self, df, measurement_type):
        # nr columns used in slicing
        col_names = df.columns.values.tolist()
        nr_of_columns = len(col_names)
        # section size is determined by the number of parts decided on when calling the function
        section_size = nr_of_columns/self.parts

        for i in range(self.parts):
            # determine the begin and end point of the slice
            if i == 0:
                begin_point = 0
                end_point = begin_point + section_size
            else:
                begin_point = end_point - 1
                end_point = begin_point + (section_size+2)

            # getting measurements at begin and end point
            df_section = df.iloc[:, begin_point:end_point]
            # df containing the differences between the cells vertically
            difference_df = df_section.diff(axis=1)
            section_col_names = df_section.columns.values.tolist()
            # first last diff contains the difference between the first value and last value of the section per row
            first_last_diff = df_section[[str(section_col_names[0]),
                                          str(section_col_names[len(section_col_names)-1])]].diff(axis=1)
            # the feature name becomes: dif_ + section + measurement type, for example: dif_50-500m_stroke
            name_feature = ['50-500', '500-1000', '1000-1500', '1500-2000']
            self.df_all.insert(len(self.df_all.columns), 'dif_' + name_feature[i] + '_' + measurement_type,
                           first_last_diff[str(section_col_names[len(section_col_names)-1])])
        return df


    def create_sprint_feature(self):
        """Saves the point where a boat starts sprinting (in the last 500m of the race) as
        the feature start_sprint.
        """
        sprint_point_all = []
        # Get all stroke rate data for the last 500m
        stroke_columns = [col for col in self.df_all.columns if 'stroke' in col and '-' not in col and '2000' not in col]
        stroke_columns_1500_to_2000 = stroke_columns[-10:]
        stroke_values = self.df_all[stroke_columns_1500_to_2000]
        # Get the number of rows
        no_datapoints = stroke_values.iloc[:,1].shape[0]
        # Loop over the rows (datapoints) to check whether the boat starts sprinting and if so, add the
        # distance at which they start sprinting in the feature start_sprint. If a crew does not sprint
        # this distance is -1.
        for boat_index in range(no_datapoints):
            boat_strokes = stroke_values.iloc[boat_index,:].values.tolist()
            # difference_list_strokes contains the difference between successive values
            difference_list_strokes = [boat_strokes[index+1] - stroke for index, stroke in enumerate(boat_strokes[:-1])]
            sprint_point = self.determine_sprint_point(difference_list_strokes)
            if sprint_point == 0:
                sprint_dist = -1
            else:
                sprint_col = stroke_columns_1500_to_2000[sprint_point + 1]
                sprint_dist = int(sprint_col[:4])
            sprint_point_all.append(sprint_dist)
        self.df_all.insert(len(self.df_all.columns), 'start_sprint', sprint_point_all)

    def determine_sprint_point(self, difference_list_strokes):
        """Determines where the sprint is started. The beginning of a sprint is marked as a difference
        larger then 1.5 in the stroke rate.
        """
        # Get all indexes of differences larger than 1.5
        index_list_strokes = [index for index, difference in enumerate(difference_list_strokes) if difference > 1.5]
        # If team does not sprint, return 0
        if not index_list_strokes:
            sprint_point = 0
        # If a team does sprint, find out if it is an irregularity (which is if the next stroke rate difference is
        # the same but reversed, marking a little peak). If it is no irregularity, save the (highest) difference as the
        # start point of the sprint.
        else:
            difference_list_strokes = self.filter_irregularities(difference_list_strokes, index_list_strokes)
            if self.change_pointer == 1:
                index_list_strokes = [index for index, difference in enumerate(difference_list_strokes) if difference > 1.5]
            if not index_list_strokes:
                sprint_point = 0
            # if there is only one difference higher then 1.5, save this difference as the sprint point
            elif len(index_list_strokes) == 1:
                sprint_point = index_list_strokes[0]
            # if there is more then one difference higher then 1.5, save the highest difference as the sprint point
            else:
                max_diffs = [difference_list_strokes[index] for index in index_list_strokes]
                biggest_diff = max_diffs.index(max(max_diffs))
                sprint_point = index_list_strokes[biggest_diff]
        return sprint_point

    def filter_irregularities(self, diffs_list, index_list):
        """
        Checks whether possible sprint points are irregularities or not. (which is if the next stroke rate difference is
        the same but reversed, marking a little peak)
        :param diffs_list: list of differences between successive stroke rate measurements
        :param index_list: list of indexes where a sprint possibly starts (if it is not an irregularity)
        :return: diffs_list where the differences that are irregularities are reduced to 0
        """
        self.change_pointer = 0
        # Loop through the differences
        for index, diff in enumerate(diffs_list[:-1]):
            # If it is a sprint point, check whether addition to the next difference
            # or the previous difference leads to 0. If so, reduce the difference to 0.
            # filtering out the irregularities and remaining only the true sprint points
            if index in index_list:
                diff_between_diffs_next = diff + diffs_list[index+1]
                if not diff_between_diffs_next:
                    diffs_list[index] = 0.0
                    diffs_list[index+1] = 0.0
                    self.change_pointer = 1
                elif index > 0:
                    diff_between_diffs_prev = diffs_list[index-1] + diff
                    if not diff_between_diffs_prev:
                        diffs_list[index-1] = 0.0
                        diffs_list[index] = 0.0
                        self.change_pointer = 1
        return diffs_list

    def create_average_strokepace_feature(self):
        """
        Creates the average stroke rate in strokes per minut, the average stroke rate in
        strokes per second and the average speed per boat per race. And the average speed
        and average stroke rate and variance of the stroke rate per team (meaning the average of the average stroke rates and
        speed per boat). And finally the 'effect of a stroke' by dividing the speeds by the
        average stroke per second in the feature called 'speed/stroke'.
        All these features are added to the data frame
        """
        types = ['stroke', 'speed']
        for averaging_type in types:
            used_cols = [str(col) for x, col in enumerate(self.df_all.columns) if averaging_type in col and len(str(col)) < 20]
            teams = self.df_all.groupby(['countries', 'boat_cat'])
            count = 0
            for name, team in teams:
                # average of either the stroke rate or the speed per boat per race
                average_type_team = team.loc[:, used_cols].mean(axis=1)
                num_averages = len(average_type_team)
                # team average stroke rate or speed
                team_average = np.mean(average_type_team)
                team_variance = np.var(average_type_team)
                # create column of length team occurrence to add the feature to the team dataframe
                average_type_feat = [team_average] * num_averages
                variance_type_feat = [team_variance] * num_averages
                # below the features are added to the data frame
                if averaging_type == 'stroke':
                    team['average_stroke_pace'] = average_type_team
                    team['average_stroke_pace_per_second'] = average_type_team/60
                if averaging_type == 'speed':
                    team['average_speed'] = average_type_team
                team['average_' + averaging_type + '_team'] = average_type_feat
                team['variance_' + averaging_type + '_team'] = variance_type_feat
                self.create_dataframe_from_groups(team, count)
                count += 1
            self.df_all = self.new_all_df
        self.df_all['speed/stroke'] = np.where(self.df_all['average_stroke_pace_per_second'] < 1, self.df_all['average_stroke_pace_per_second'],
                                               self.df_all['average_speed'] / self.df_all['average_stroke_pace_per_second'])

    def create_average_sprint_feature(self):
        """
        Creates both a Boolean sprint feature stating whether a boat sprints or not (sprint_bool) and a team based
        feature where the average sprint point per team and the variance of the sprint point  (over all races that team
        has raced in the current data base) is calculated (average_sprint_team, variance_sprint_team) and added as a
        feature to the dataframe.
        :return:
        """
        types = ['start_sprint']
        for averaging_type in types:
            # group per team
            teams = self.df_all.groupby(['countries', 'boat_cat'])
            count = 0
            # loop over teams
            for name, team in teams:
                sprint_team = team.loc[:, 'start_sprint']
                # use only the points where a team has sprinted to calculate the average sprint point
                # and the variance in the sprint point
                sprint_boolean_team = [1.0 if sprint > 0.0 else 0.0 for sprint in sprint_team]
                num_bools = len(sprint_boolean_team)
                team_average = np.mean(sprint_boolean_team)
                team_variance = np.var(sprint_boolean_team)
                average_sprint_feat = [team_average] * num_bools
                variance_sprint_feat = [team_variance] * num_bools
                # add features to the dataframe
                team['sprint_bool'] = sprint_boolean_team
                team['average_' + 'sprint' + '_team'] = average_sprint_feat
                team['variance_' + 'sprint' + '_team'] = variance_sprint_feat
                self.create_dataframe_from_groups(team, count)
                count += 1
            self.df_all = self.new_all_df