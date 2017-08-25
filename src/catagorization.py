import pandas as pd
from datetime import datetime

class catagorization:
    def __init__(self, df, type):
        self.df = df
        self.type = type

    def catagorize_contest(self):
        grouped = self.df.groupby(by=['contest'])
        frames = []
        for name, group in grouped:
            group_len = group.shape[0]
            if 'EC' in name:
                contest_cat = [1] * group_len
                group.insert(3, 'contest_cat', contest_cat)
                frames.append(group)
            elif 'WC1' in name:
                contest_cat = [2] * group_len
                group.insert(3, 'contest_cat', contest_cat)
                frames.append(group)
            elif 'WC2' in name:
                contest_cat = [3] * group_len
                group.insert(3, 'contest_cat', contest_cat)
                frames.append(group)
            elif 'WC3' in name:
                contest_cat = [4] * group_len
                group.insert(3, 'contest_cat', contest_cat)
                frames.append(group)
            elif 'WCH' in name:
                contest_cat = [5] * group_len
                group.insert(3, 'contest_cat', contest_cat)
                frames.append(group)
            elif 'OS' in name:
                contest_cat = [6] * group_len
                group.insert(3, 'contest_cat', contest_cat)
                frames.append(group)
            else:
                print('%s is a contest that is not covered in catagorization' % name)
        self.df = pd.concat(frames)
        self.df.reset_index(drop=True)
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: Contests catagorized]" % time)

    def catagorize_boattype(self):
        grouped = self.df.groupby(by=['boattype'])
        frames = []
        for name, group in grouped:
            group_len = group.shape[0]
            if name.upper() == 'HM1X' or name.upper().strip() == 'M1X':
                boat_cat = [1] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HM2-' or name.upper().strip() == 'M2-':
                boat_cat = [2] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HM2+' or name.upper().strip() == 'M2+':
                boat_cat = [3] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HM2X' or name.upper().strip() == 'M2X':
                boat_cat = [4] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HM4-' or name.upper().strip() == 'M4-':
                boat_cat = [5] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HM4X' or name.upper().strip() == 'M4X':
                boat_cat = [6] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HM8+' or name.upper().strip() == 'M8+':
                boat_cat = [7] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HW1X' or name.upper().strip() == 'W1X':
                boat_cat = [8] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HW2-' or name.upper().strip() == 'W2-':
                boat_cat = [9] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HW2X' or name.upper().strip() == 'W2X':
                boat_cat = [10] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HW4-' or name.upper().strip() == 'W4-':
                boat_cat = [11] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HW4X' or name.upper().strip() == 'W4X':
                boat_cat = [12] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'HW8+' or name.upper().strip() == 'W8+':
                boat_cat = [13] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LM1X' or name.upper().strip() == 'M1X':
                boat_cat = [14] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LM2-':
                boat_cat = [15] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LM2X':
                boat_cat = [16] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LM4-':
                boat_cat = [17] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LM4X':
                boat_cat = [18] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LM8+':
                boat_cat = [19] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LW1X':
                boat_cat = [20] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LW2X':
                boat_cat = [21] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            elif name.upper() == 'LW4X':
                boat_cat = [22] * group_len
                group.insert(6, 'boat_cat', boat_cat)
                frames.append(group)
            else:
                print('%s is a boattype that is not covered in catagorization' % name.upper())
        self.df = pd.concat(frames)
        self.df = self.df.reset_index(drop=True)
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: Boattypes catagorized]" % time)


    def catagorize_contestround(self):
        grouped = self.df.groupby(by=['round'])
        frames = []
        for name, group in grouped:
            group_len = group.shape[0]
            if 'H' in name.upper() or 'X' in name.upper():
                name_list = list(name.strip())
                round_number = name_list[len(name_list)-1]
                if 'X' in name.upper():
                    round_number = 1
                round_cat = [1] * group_len
                round_num_col = [round_number] * group_len
                group.insert(5, 'round_cat', round_cat)
                group.insert(6, 'round_number', round_num_col)
                frames.append(group)
            elif 'R' in name.upper():
                name_list = list(name.strip())
                round_number = name_list[len(name_list)-1]
                round_cat = [2] * group_len
                round_num_col = [round_number] * group_len
                group.insert(5, 'round_cat', round_cat)
                group.insert(6, 'round_number', round_num_col)
                frames.append(group)
            elif 'Q' in name.upper():
                name_list = list(name.strip())
                round_number = name_list[len(name_list)-1]
                round_cat = [3] * group_len
                round_num_col = [round_number] * group_len
                group.insert(5, 'round_cat', round_cat)
                group.insert(6, 'round_number', round_num_col)
                frames.append(group)
            elif 'S' in name.upper():
                name_list = list(name.strip())
                round_number = name_list[len(name_list)-1]
                round_cat = [4] * group_len
                round_num_col = [round_number] * group_len
                group.insert(5, 'round_cat', round_cat)
                group.insert(6, 'round_number', round_num_col)
                frames.append(group)
            elif 'F' in name.upper():
                name_list = list(name.strip())
                round_number = name_list[len(name_list)-1]
                round_number = ord(round_number.lower())-96
                round_cat = [5] * group_len
                round_num_col = [round_number] * group_len
                group.insert(5, 'round_cat', round_cat)
                group.insert(6, 'round_number', round_num_col)
                frames.append(group)
            else:
                print('%s is a round that is not covered in catagorization' % name.upper())
        self.df = pd.concat(frames)
        self.df = self.df.reset_index(drop=True)
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: Boattypes catagorized]" % time)

    def catagorize_countries(self):
        cat_count = 0
        grouped = self.df.groupby(by=['countries'])
        frames = []
        for name, group in grouped:
            group_len = group.shape[0]
            country_cat = [cat_count] * group_len
            group.insert(5, 'country_cat', country_cat)
            frames.append(group)
            cat_count += 1
        self.df = pd.concat(frames)
        self.df = self.df.reset_index(drop=True)
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: Countries catagorized]" % time)


    def catagorize_all(self):
        self.catagorize_contest()
        self.catagorize_boattype()
        self.catagorize_contestround()
        if self.type == 'speeds':
            self.catagorize_countries()
        self.df = self.df.sort_values(['year', 'contest_cat', 'boat_cat', 'round_cat', 'round_number'])
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: All data catagorized]" % time)
        self.df = self.df.reset_index(drop=True)
        return self.df
