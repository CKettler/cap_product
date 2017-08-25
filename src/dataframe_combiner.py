import pandas as pd
import time


class combine_dataframes:
    def __init__(self, speeds_df, strokes_df, results_df):
        self.speeds_df = speeds_df
        self.strokes_df = strokes_df
        self.results_df = results_df

    def combine_all(self):
        """
        Combine the strokes, speeds and results dataframes in one big dataframe
        :return: Dataframe containing all data
        """
        # Join cols are the columns used to combine on. The values of these columns should be unique for each row. In
        # this case the year, competition type, boat type, round type and round number are used, alongside their catagorie
        # values
        join_cols = ['countries', 'year', 'contest', 'contest_cat', 'round', 'round_cat', 'round_number',
                     'boattype', 'boat_cat']

        # ad a suffix of speed or stroke to the data from either the speed or the stroke dataframe to be able to identify
        # where the features belong to
        self.speeds_df.columns = [str(col) + '_speed' if x > 9 else str(col) for x, col in enumerate(self.speeds_df.columns)]
        self.strokes_df.columns = [str(col) + '_stroke' if x > 8 else str(col) for x, col in enumerate(self.strokes_df.columns)]
        if 'country_cat' in self.speeds_df.columns and 'country_cat' in self.strokes_df.columns:
            self.speeds_df = self.speeds_df.drop('country_cat')
        # Merge the speeds dataframe and the strokes dataframe to one
        speeds_strokes_df = pd.merge(self.speeds_df, self.strokes_df, on=join_cols, how='inner')
        print("[%s: speeds df merged with strokes df]")
        # The columns in the results data are not always ordered right and should therefore be reindexed
        new_col_order = ['countries', 'year', 'contest', 'contest_cat', 'date', 'round', 'round_cat',
                         'round_number', 'boattype', 'boat_cat', 'Name1', 'Name2', 'Name3', 'Name4', 'Name5', 'Name6',
                         'Name7', 'Name8', 'start_lane', '500m_rank', '500m_time', '500-1000_time',
                         '500-1000_rank', '1000-1500_rank', '1000-1500_time', '1000m_rank', '1000m_time',
                         '1500-2000_rank', '1500-2000_time', '1500m_rank', '1500m_time', '2000m_rank', '2000m_time']
        self.results_df = self.results_df.reindex(columns=new_col_order)
        # Merge the combined speeds_strokes_df with the results dataframe and save this into a PDF
        all_df = pd.merge(self.results_df, speeds_strokes_df, on=join_cols, how='inner')
        print("[%s: speeds&strokes df merged with results df]")
        time.sleep(1)
        # all_df = all_df.drop(['boat_cat_stroke', 'boat_cat_speed'], 1)
        print(all_df.shape)
        all_df.to_csv('../data/Racedata/total_file_all.csv')

        return all_df
