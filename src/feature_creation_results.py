import pandas as pd
import numpy as np

#turn off chained warnings
pd.options.mode.chained_assignment = None

class feature_creation_results:
    def __init__(self, dataframe):
        self.df_all = dataframe
        self.temp_df = pd.DataFrame
        self.new_all_df = pd.DataFrame

    def create_all_features(self):
        """
        calls all functions that create features based on only the results data
        :return: a data frame including these features
        """
        self.first_position_feature()
        self.team_ranking_feature()
        return self.df_all


    def first_position_feature(self):
        """
        Creates a Boolean that is 1 if the position at the specified measure point is 1 and is 0 otherwise.
        """
        used_columns = [col for col in self.df_all.columns if 'rank' in col and '-' not in col]
        # data frame containing only the columns with rank information
        rank_df = self.df_all.loc[:, used_columns]
        # data frame that is 1 where rank_df is 1 and 0 where rank_df holds an other value
        one_df = rank_df.where(rank_df == 1)
        first_feature_df = one_df.fillna(0)
        first_feature_df.columns = [str(col) + '_first_bool' for col in first_feature_df.columns]
        # adding the new columns to the data frame
        self.df_all = self.df_all.join(first_feature_df)


    def team_ranking_feature(self):
        """
         Adds the average rank of a team and the variance of the rank of a team as a feature to each occurrence of the
         team, based on the races of that team in the data. These are added to the data frame as average_rank_team and
         variance_rank_team
        """
        race_id_columns = ['boat_cat', 'countries']
        teams = self.df_all.groupby(race_id_columns)
        count = 0
        # loop over teams
        for name, team in teams:
            # collect all ranks per team
            ranks_team = team.loc[:,'2000m_rank'].values.tolist()
            num_ranks = len(ranks_team)
            team_average = np.mean(ranks_team)
            # if the average is nan, skip this team
            if np.isnan(team_average):
                continue
            team_variance = np.var(ranks_team)
            average_rank_feat = [team_average] * num_ranks
            variance_rank_feat = [team_variance] * num_ranks
            # add features to data frame
            team['average_rank_team'] = average_rank_feat
            team['variance_rank_team'] = variance_rank_feat
            self.create_dataframe_from_groups(team, count)
            count += 1
        self.df_all = self.new_all_df

    def create_dataframe_from_groups(self, team, count):
        if count == 0:
            self.new_all_df = team
        else:
            self.new_all_df = pd.concat([self.new_all_df, team])


