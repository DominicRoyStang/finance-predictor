import pandas
from pathlib import Path


def handle_mint(csv_file):
    """
    Formats and edits a dataset from a csv file in Mint format
    Returns a Pandas DataFrame that
        1. Has two columns: date, net worth
        2. Is sorted by date (YYYY-MM-DD)
    """
    print('mint format detected')

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
