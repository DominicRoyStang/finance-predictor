"""
This is a helper script to split a large datset (of supported format) into smaller sets.
run it as such:
    python split_dataset.py ../datasets/trans.asc
"""
import sys
from pathlib import Path
from feature_engineering.csv_handler import split_aggregate_csv


def main(args):
    if len(args) == 1:
        print('No path to csv dataset provided.')
    elif len(args) == 2:
        split_aggregate_csv(Path(args[1]))
    else:
        print('Too many arguments!')


if __name__ == "__main__":
    main(sys.argv)
