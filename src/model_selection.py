"""
This file is intended to be used for model selection.
It can be used to compare the following three regression models:
    - Linear Regression (lr)
    - Support Vector Regression (svr)
    - Gaussian Process Regression (gpr)

Author: Dominic Roy-Stang
"""

import sys
from pathlib import Path

import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel
from sklearn.svm import SVR
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
    # plot_prediction(X_test, y_test, y_pred=y_pred, timeout=None)  # plot on the domain of the training set
    # plot_prediction(X, y, y_pred=y_pred, timeout=None)  # plot on the domain of X

    # Root mean squared error
    root_mean_squared_error = numpy.sqrt(mean_squared_error(y_test, y_pred))
    print("Root mean square error: %.2f" % root_mean_squared_error)
    # Explained variance score: 1 is perfect prediction
    variance_score = r2_score(y_test, y_pred)
    print("Variance score: %.2f" % variance_score)  # this is actually r squared
    print("\n")

    return variance_score


def run_support_vector_regression(data):
    """
    Receives a formatted pandas dataframe,
    and performs a support vector regression.
    Returns the root mean squared error on the test set.
    """
    # TODO
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
    regression = SVR(gamma='scale', C=10e2)

    # Train the model using the training sets
    regression.fit(X_train_numeric, y_train)

    # Make predictions using the testing set
    y_pred = regression.predict(X_test_numeric)  # predictions on the domain of the training set
    y_pred_all = regression.predict(X_numeric)  # predictions on the domain of X

    # Graph
    # plot_prediction(X_test, y_test, y_pred=y_pred, timeout=None)  # plot on the domain of the training set
    plot_prediction(X, y, X_test=X_test, y_pred=y_pred_all, timeout=None)  # plot on the domain of X

    # Root mean squared error
    root_mean_squared_error = numpy.sqrt(mean_squared_error(y_test, y_pred))
    print("Root mean square error: %.2f" % root_mean_squared_error)
    # Explained variance score: 1 is perfect prediction
    variance_score = r2_score(y_test, y_pred)
    print("Variance score: %.2f" % variance_score)  # this is actually r squared
    print("\n")

    return variance_score


def run_gaussian_process_regression(data):
    """
    Receives a formatted pandas dataframe,
    and performs a gaussian process regression.
    Returns the root mean squared error on the test set.
    """
    # TODO
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
    kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 10e1)) + WhiteKernel(1e-1)
    regression = GaussianProcessRegressor(kernel=kernel, alpha=35)

    # Train the model using the training sets
    regression.fit(X_train_numeric, y_train)

    # Make predictions using the testing set
    y_pred = regression.predict(X_test_numeric)  # predictions on the domain of the training set
    y_pred_all = regression.predict(X_numeric)  # predictions on the domain of X

    # Graph
    # plot_prediction(X_test, y_test, X_test=X_test, y_pred=y_pred, timeout=None)  # plot on the domain of the training set
    plot_prediction(X, y, X_test=X_test, y_pred=y_pred_all, timeout=None)  # plot on the domain of X

    # Root mean squared error
    root_mean_squared_error = numpy.sqrt(mean_squared_error(y_test, y_pred))
    print("Root mean square error: %.2f" % root_mean_squared_error)
    # Explained variance score: 1 is perfect prediction
    variance_score = r2_score(y_test, y_pred)
    print("Variance score: %.2f" % variance_score)  # this is actually r squared
    print(regression.score(X_test, y_test))
    print("\n")

    return variance_score


def process_files(file_list):
    """
    Runs all three regressions on the files in the file_list.
    Returns a dictionary, where each compared regression type is mapped to a list of scores (one for each file)
    """
    score_lists = {
        'lr': [],
        'svr': [],
        'gpr': []
    }
    # NOTE: This could be much faster with parallel processing on large file lists
    for file in file_list:
        # Create a dataframe from the current file
        try:
            dataframe = csv_to_formatted_dataframe(file)
        except Exception as err:
            print(str(err), "Skipping file.")
            continue
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
        # Default to preformatted.csv if no file is provided as an argument
        project_root = Path(__file__).resolve().parent.parent
        default_location = project_root/"datasets/preformatted.csv"
        results = process_files([default_location])

    elif len(args) is 2:
        location = Path(args[1])
        if location.is_file():
            results = process_files([location])
        elif location.is_dir():
            datasets = [x for x in location.glob("**/*") if x.is_file()]
            results = process_files(datasets)

    else:
        print("Too many arguments provided!")
        exit()

    # Output results
    print("Average score (r_sqr) on linear regression:", numpy.average(results["lr"]))
    print("Average score (r_sqr) on support vector regression:", numpy.average(results["svr"]))
    print("Average score (r_sqr) on gaussian process regression:", numpy.average(results["gpr"]))


if __name__ == "__main__":
    main(sys.argv)
