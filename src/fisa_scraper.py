"""
This file scrapes the Results and GPS PDF's from the World Rowing site
"""

from urllib import urlretrieve
from urllib import urlopen


# All contest types that we want to scrape from www.worldrowing.com
contests = ['ECH', 'WCp1', 'WCp2', 'WCp3', 'WCH', 'OG']

# All years that we want to use
years = [2013, 2014, 2015, 2016, 2017]

# All boattypes we want to use
boattypes = ['ROM112', 'ROM121', 'ROM122', 'ROM141', 'ROM012', 'ROM021', 'ROM022', 'ROM041', 'ROM042', 'ROM083', 'ROW112',
             'ROW122', 'ROW012', 'ROW021', 'ROW022', 'ROW041', 'ROW042', 'ROW083']
# The names corresponding to the boattypes
boatnames = ['LM1X', 'LM2-', 'LM2X', 'LM4-', 'HM1x', 'HM2-', 'HM2x', 'HM4-', 'HM4x', 'HM8+', 'LW1x', 'LW2x', 'HW1x',
             'HW2-', 'HW2x', 'HW4-', 'W4x', 'W8+']
# The rounds we want to use
rounds = ['101', '102', '103', '104', '105', '106', '201', '202', '203', '204', '205', '206', '301', '302', '303', '304', '801', '802', '803',
          '804', '901', '902', '903', '904', '905', '906', 'P01', '851', '852', '853', '854', '701']
# The names corresponding to the rounds
round_names = ['FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'SA1', 'SA2', 'SC1', 'SC2', 'SE1', 'SE2', 'Q1', 'Q2', 'Q3', 'Q4', 'R1',
               'R2', 'R3', 'R4', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'EX', 'R1', 'R2', 'R3', 'R4', 'EX']

# For the Resultsdata
for year in years:
    for contest in contests:
        for i,boattype in enumerate(boattypes):
            for j,contest_round in enumerate(rounds):
                if year == 2017:
                    pdf_url = 'http://www.worldrowing.com/assets/pdfs/' + contest + '_' + str(year) + '_1/' + boattype + contest_round + '_C73.pdf'
                elif year in [2013, 2014] and contest == 'WCp3':
                    pdf_url = 'http://www.worldrowing.com/assets/pdfs/' + contest + 'F_' + str(
                        year) + '/' + boattype + contest_round + '_C73.pdf'
                else:
                    pdf_url = 'http://www.worldrowing.com/assets/pdfs/' + contest + '_' + str(
                        year) + '/' + boattype + contest_round + '_C73.pdf'
                resp = urlopen(pdf_url)
                if resp.code == 200:
                    print(pdf_url)
                    urlretrieve(pdf_url, '../data/Scrapedata/Results/' + str(year) + '_' + contest + '_' + boatnames[i] + '_' + round_names[j] + '.pdf')
                else:
                    continue

# For the GPS data
for year in years:
    for contest in contests:
        for i,boattype in enumerate(boattypes):
            for j,contest_round in enumerate(rounds):
                if year == 2017 and contest == 'WCp1':
                    pdf_url = 'http://www.worldrowing.com/assets/pdfs/' + contest + '_' + str(year) + '_1/' + boattype + contest_round + '_MGPS.pdf'
                elif year in [2013, 2014] and contest == 'WCp3':
                    pdf_url = 'http://www.worldrowing.com/assets/pdfs/' + contest + 'F_' + str(
                        year) + '/' + boattype + contest_round + '_MGPS.pdf'
                else:
                    pdf_url = 'http://www.worldrowing.com/assets/pdfs/' + contest + '_' + str(
                        year) + '/' + boattype + contest_round + '_MGPS.pdf'
                print(pdf_url)
                resp = urlopen(pdf_url)
                if resp.code == 200:
                    urlretrieve(pdf_url, '../data/Scrapedata/GPS/' + str(year) + '_' + contest + '_' + boatnames[i] + '_' + round_names[j] + '.pdf')
                else:
                    continue
