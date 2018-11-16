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
    formats_file = Path(__file__).resolve().parent/"csv_formats.yaml"
    with open(formats_file) as formats:
        csv_formats = yaml.load(formats)

    # The first row of a csv contains the labels
    first_row = pandas.read_csv(csv_file, nrows=1)
    current_format = first_row.columns.tolist()

    # Search for a corresponding supported format
    if current_format == csv_formats['preformatted']:
        return 'preformatted'
    if current_format == csv_formats['indexed_preformatted']:
        return 'indexed_preformatted'
    if current_format == csv_formats['mint']:
        return 'mint'
    else:
        raise Exception('CSV format not supported!')


def handle_preformatted(csv_file):
    """
    Converts csv file in preformatted format to a dataframe
    Returns the csv as a Pandas DataFrame
    """
    print('preformatted format dected')

    # Load from dataset
    data = pandas.read_csv(csv_file, infer_datetime_format=True)

    # Sort by date
    data.sort_values(by='Date', inplace=True)

    return data


def handle_indexed_preformatted(csv_file):
    """
    Converts csv file in preformatted format to a dataframe
    Returns the csv as a Pandas DataFrame
    """
    print('indexed_preformatted format dected')

    # Load from dataset
    data = pandas.read_csv(csv_file)

    # Set date format and sort by date
    data['Date'] = pandas.to_datetime(data.Date)
    data.sort_values(by='Date', inplace=True)

    # Initialize index column
    data.set_index('Index', inplace=True)

    return data


def handle_mint(csv_file):
    """
    Formats and edits a dataset from a csv file in Mint format
    Returns a Pandas DataFrame that
        1. Has two columns: date, net worth
        2. Is sorted by date (YYYY-MM-DD)
    """
    print('mint format dected')

    # Load relevant data from dataset
    relevant_features = ['Date', 'Transaction Type', 'Amount']
    data = pandas.read_csv(csv_file, usecols=relevant_features)

    # Set date format and sort by date
    data['Date'] = pandas.to_datetime(data.Date)
    data.sort_values(by='Date', inplace=True)

    # Create and initialize index column
    data.insert(1, 'Index', range(1, len(data)+1))
    data.set_index('Index', inplace=True)

    # Add a net worth column with default value of 0
    data.insert(1, 'Net Worth', 0)

    # Calculate net worth
    net_worth = 0
    for index, row in data.iterrows():
        transaction_type = row['Transaction Type']
        amount = row['Amount']
        if row['Transaction Type'] == 'credit':
            net_worth += amount
            data.loc[index, 'Net Worth'] = net_worth
        elif row['Transaction Type'] == 'debit':
            net_worth -= amount
            data.loc[index, 'Net Worth'] = net_worth

    # Remove features that are no longer needed
    data.drop(["Amount", "Transaction Type"], axis=1, inplace=True)

    return data


def format_dataset(csv_file):
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
    else:
        raise Exception('CSV format unrecognized!')
