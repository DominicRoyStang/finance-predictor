import pandas
from pathlib import Path


def split_czech(csv_file):
    """
    Converts an aggregate csv file in czech format to multiple csvs (one per account)
    Returns the path to the parent folder holding the csv files
    """
    print('czech format detected')

    # Load from dataset
    data = pandas.read_csv(csv_file, delimiter=';', dtype='unicode')

    # Set date format and sort by date
    # data['Date'] = pandas.to_datetime(data.Date)
    # data.sort_values(by='Date', inplace=True)

    return data
