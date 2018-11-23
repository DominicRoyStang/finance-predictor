import pandas
from pathlib import Path


def handle_preformatted(csv_file):
    """
    Converts csv file in preformatted format to a dataframe
    Returns the csv as a Pandas DataFrame
    """
    print('preformatted format detected')

    # Load from dataset
    data = pandas.read_csv(csv_file)

    # Set date format and sort by date
    data['Date'] = pandas.to_datetime(data.Date)
    data.sort_values(by='Date', inplace=True)

    return data


def handle_indexed_preformatted(csv_file):
    """
    Converts csv file in preformatted format to a dataframe
    Returns the csv as a Pandas DataFrame
    """
    print('indexed_preformatted format detected')

    # Load from dataset
    data = pandas.read_csv(csv_file)

    # Set date format and sort by date
    data['Date'] = pandas.to_datetime(data.Date)
    data.sort_values(by='Date', inplace=True)

    # Initialize index column
    data.set_index('Index', inplace=True)

    return data
