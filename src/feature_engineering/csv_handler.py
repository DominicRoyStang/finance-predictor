import yaml
import pandas
from pathlib import Path

from .czech_csv import split_czech, handle_czech
from .mint_csv import handle_mint
from .preformatted_csv import handle_preformatted, handle_indexed_preformatted


def determine_format(csv_file):
    """
    Determines the format of the provided csv file by looking at the first row in the file.
    Returns a string representation of the csv format
    Raises exception if the format is unrecognized
    """
    # Get list of supported csv formats
    formats_file = Path(__file__).resolve().parent/'csv_formats.yaml'
    with open(formats_file) as formats:
        csv_formats = yaml.load(formats)

    # The first row of a csv contains the labels which are used to determine the format
    delimiters = [',', ';']
    for delimiter in delimiters:
        first_row = pandas.read_csv(csv_file, nrows=1, dtype='unicode', delimiter=delimiter)
        if len(first_row.columns.tolist()) > 1:
            break

    current_format = first_row.columns.tolist()

    # Search for a corresponding supported format
    if current_format == csv_formats['preformatted']:
        return 'preformatted'
    if current_format == csv_formats['indexed_preformatted']:
        return 'indexed_preformatted'
    if current_format == csv_formats['mint']:
        return 'mint'
    if current_format == csv_formats['czech']:
        return 'czech'
    else:
        raise Exception('CSV format not supported!')


def csv_to_formatted_dataframe(csv_file):
    """
    Format and edit dataset from a csv file.
    Returns a Pandas DataFrame that
        1. It has two columns: date, net worth
        2. It is sorted by date (YYYY-MM-DD)
    """
    # Determine csv format
    try:
        csv_format = determine_format(csv_file)
    except Exception as e:
        raise

    # Handle the format
    if csv_format is 'preformatted':
        return handle_preformatted(csv_file)
    elif csv_format is 'indexed_preformatted':
        return handle_indexed_preformatted(csv_file)
    elif csv_format is 'mint':
        return handle_mint(csv_file)
    elif csv_format is 'czech':
        return handle_czech(csv_file)
    else:
        raise Exception('CSV format unrecognized!')


def split_aggregate_csv(csv_file):
    """
    Receives a csv file with transactions from many individuals,
    and creates a folder containing one csv file per individual.
    Returns a pathlib Path to the created folder.
    """

    # Determine csv format
    try:
        csv_format = determine_format(csv_file)
    except Exception as e:
        raise

    # Handle the format
    if csv_format is 'czech':
        print(split_czech(csv_file))
    else:
        raise Exception('CSV format unrecognized!')
