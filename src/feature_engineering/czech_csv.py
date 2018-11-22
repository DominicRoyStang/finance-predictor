import pandas
from pathlib import Path


def handle_czech(csv_file):
    """
    Formats and edits a dataset from a csv file in Mint format
    Returns a Pandas DataFrame that
        1. Has two columns: date, net worth
        2. Is sorted by date (YYYY-MM-DD)
    """

    # Load relevant data from dataset
    relevant_features = ['date', 'balance']
    data = pandas.read_csv(csv_file, delimiter=';', dtype='unicode', usecols=relevant_features)

    # Set date format and sort by date
    data['date'] = pandas.to_datetime(data.date)
    data.sort_values(by='date', inplace=True)

    data['balance'] = pandas.to_numeric(data.balance)

    # Relabel
    data = data.rename({'date': 'Date', 'balance': 'Net Worth'}, axis='columns')

    return data


def split_czech(csv_file):
    """
    Converts an aggregate csv file in czech format to multiple csvs (one per account)
    Returns the path to the parent folder holding the csv files
    """

    # Load from dataset
    data = pandas.read_csv(csv_file, delimiter=';', dtype='unicode')

    print('This might take several minutes. Be patient.')

    gb = data.groupby('account_id')
    # data = [gb.get_group(x) for x in gb.groups]
    project_root = Path(__file__).resolve().parent.parent.parent
    output_folder = project_root/("datasets/" + str(csv_file.stem) + "_split")
    output_folder.mkdir(parents=True, exist_ok=True)

    # for each account, write its transactions to a csv file
    for account_id in gb.groups:
        transactions = gb.get_group(account_id)
        output_file = output_folder/(account_id + ".csv")
        transactions.to_csv(output_file, sep=';', index=False)

    return output_folder
