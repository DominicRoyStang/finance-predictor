import yaml
import pandas
from pathlib import Path


def determine_format(csv_file):
    """
    Determines the format of the provided csv file by looking at the first row in the file.
    Returns a string representation of the csv format
    Raises exception if the format is unrecognized
    """
    # Get list of supported csv formats
    with open('csv_formats.yaml') as formats:
        csv_formats = yaml.load(formats)

    # The first row of a csv contains the labels
    first_row = pandas.read_csv(csv_file, nrows=1)
    current_format = first_row.columns.tolist()

    # Search for a corresponding supported format
    if (current_format == csv_formats['mint']):
        return 'mint'
    else:
        raise Exception('CSV format not supported!')


def handle_mint(csv_file):
    print('MINT')


def format_dataset(csv_file):
    """
    Format and edit dataset such that it returns a Pandas DataFrame that
    1. It has two columns: date, net worth
    2. It is sorted by date (YYYY-MM-DD)
    """
    try:
        csv_format = determine_format(csv_file)
    except Exception as e:
        raise

    if csv_format is 'mint':
        return handle_mint(csv_file)
    else:
        raise Exception('CSV format unrecognized!')


datasets_directory = Path.cwd()/"../../datasets"
transactions_file = datasets_directory/"transactions.csv"
format_dataset(transactions_file)
