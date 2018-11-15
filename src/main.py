import matplotlib
import numpy
import sklearn
import pandas
from pathlib import Path
from csv_handler import csv_handler


project_root = Path(__file__).resolve().parent.parent


def main():
    transactions_file = project_root/"datasets/transactions.csv"
    data = csv_handler.format_dataset(transactions_file)
    print(data)


if __name__ == "__main__":
    main()
