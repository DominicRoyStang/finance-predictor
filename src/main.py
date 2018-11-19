import sys
from pathlib import Path

import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_engineering.csv_handler import csv_to_formatted_dataframe
from graphing.graphing import plot_data, plot_prediction


def test_csv_handler():
    # Load the personal finance dataset
    transactions_file = project_root/"datasets/preformatted.csv"
    data = csv_to_formatted_dataframe(transactions_file)

    # Split into feature and target sets
    X = data["Date"]
    y = data["Net Worth"]

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=False)

    # Plot outputs
    plot_data(X_train, y_train, timeout=3000)


def run_linear_regression(data):
    """
    Receives a formatted pandas dataframe, 
    and performs a linear regression.
    Returns the root mean squared error on the test set.
    """

    # Split into feature and target sets
    X = data["Date"].values
    y = data["Net Worth"]

    # Convert 1-D array to 2-D feature array, as expected by sklearn
    X = X.reshape(len(X), 1)

    # Plot all data
    # plot_data(X, y, timeout=None)

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=False)
    X_numeric = [pandas.to_numeric(example) for example in X]
    X_train_numeric = [pandas.to_numeric(example) for example in X_train]
    X_test_numeric = [pandas.to_numeric(example) for example in X_test]

    # Create linear regression object
    regression = LinearRegression()

    # Train the model using the training sets
    regression.fit(X_train_numeric, y_train)

    # Make predictions using the testing set
    y_pred = regression.predict(X_test_numeric)  # predictions on the domain of the training set
    # y_pred = regression.predict(X_numeric)  # predictions on the domain of X

    # Graph
    plot_prediction(X_test, y_test, y_pred=y_pred, timeout=None)  # plot on the domain of the training set
    # plot_prediction(X, y, y_pred=y_pred, timeout=None)  # plot on the domain of X

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Root mean squared error
    root_mean_squared_error = numpy.sqrt(mean_squared_error(y_test, y_pred))
    print("Root mean square error: %.2f" % root_mean_squared_error)
    # Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % r2_score(y_test, y_pred))

    return root_mean_squared_error


def run_support_vector_regression(data):
    """
    Receives a formatted pandas dataframe,
    and performs a support vector regression.
    Returns the root mean squared error on the test set.
    """
    # TODO
    return 0


def run_gaussian_process_regression(data):
    """
    Receives a formatted pandas dataframe,
    and performs a gaussian process regression.
    Returns the root mean squared error on the test set.
    """
    # TODO
    return 0


def process_files(file_list):
    score_lists = {
        'lr': [],
        'svr': [],
        'gpr': []
    }
    # NOTE: This could be much faster with parallel processing on large file lists
    for file in file_list:
        # Create a dataframe from the current file
        dataframe = csv_to_formatted_dataframe(file)
        # Run regressions on the dataframe
        lr_score = run_linear_regression(dataframe)
        svr_score = run_support_vector_regression(dataframe)
        gpr_score = run_gaussian_process_regression(dataframe)
        # Append regression results to the list
        score_lists['lr'].append(lr_score)
        score_lists['svr'].append(svr_score)
        score_lists['gpr'].append(gpr_score)

    return score_lists


def main(args):
    """
    Receives a file of transactions for one individual or
    a folder containing transaction files for individuals,
    calls helper functions to run regressions.
    """

    results = []

    # Load the provided personal finance dataset
    if len(args) is 1:
        # Default to all *.csv files if no file is provided as an argument
        project_root = Path(__file__).resolve().parent.parent
        datasets_folder = project_root/"datasets/"
        datasets = [x for x in datasets_folder.glob("**/*.csv") if x.is_file()]
        results = process_files(datasets)
    elif len(args) is 2:
        pass
    else:
        print("Too many arguments provided!")
        exit()

    # Output results
    print("Average score (rmse) on linear regression:", numpy.average(results["lr"]))
    print("Average score (rmse) on support vector regression:", numpy.average(results["svr"]))
    print("Average score (rmse) on gaussian process regression:", numpy.average(results["gpr"]))


if __name__ == "__main__":
    main(sys.argv)
