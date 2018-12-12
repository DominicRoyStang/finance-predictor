import sys
from pathlib import Path

import numpy
import pandas
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel

from tseries import TimeSeriesRegressor, time_series_split

from feature_engineering.csv_handler import csv_to_formatted_dataframe
from graphing.graphing import plot_predictions


def run_linear_regression(X, y, datelist):
    """
    Receives a formatted pandas dataframe,
    and performs a linear regression.
    Returns the root mean squared error on the test set.
    """
    print("\nLinear Regression")

    # Format data for fitting
    X_numeric = [pandas.to_numeric(example) for example in X]
    datelist_numeric = [pandas.to_numeric(date) for date in datelist]

    # Create linear regression object
    regression = LinearRegression()

    # Train the model using the training sets
    regression.fit(X_numeric, y)

    # Make predictions using the testing set
    y_pred = regression.predict(X_numeric)  # predictions on the domain of X
    y_pred_all = regression.predict(datelist_numeric)

    return y_pred, y_pred_all


def run_gaussian_process_regression(X, y, datelist):
    """
    Receives a formatted pandas dataframe,
    and performs a gaussian process regression.
    Returns the root mean squared error on the test set.
    """
    print("\nGaussian Process Regression")
    # Format data for fitting
    X_numeric = [pandas.to_numeric(example) for example in X]
    datelist_numeric = [pandas.to_numeric(date) for date in datelist]

    # Create linear regression object
    kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e2)) + WhiteKernel(1e-1)
    regression = GaussianProcessRegressor(kernel=kernel, alpha=35, normalize_y=True)

    # Fit the model
    regression.fit(X_numeric, y)

    # Make predictions using the testing set
    y_pred = regression.predict(X_numeric)  # predictions on the domain of X
    y_pred_all = regression.predict(datelist_numeric)

    return y_pred, y_pred_all


def run_time_series(X, y, datelist):
    """
    Receives a formatted pandas dataframe,
    and performs a support vector regression.
    Returns the root mean squared error on the test set.
    """
    print("\nTime Series")
    # Format data for fitting
    X = X.astype('float64', copy=True)
    datelist = datelist.astype('float64', copy=True)

    # Create time series lasso regressor object
    n_prev = int(len(y)/2)
    empty_values = [numpy.nan for _ in range(0, n_prev)]
    tsr = TimeSeriesRegressor(n_prev=n_prev)  # uses linear regressions

    # Train the model using the training sets
    tsr.fit(X, y)

    # Make predictions using the testing set
    y_pred = tsr.predict(X)
    # y_pred = [*empty_values, *y_pred]
    """
    X = X[n_prev:]

    for date in datelist:
        tsr.fit(X, y_pred)
        X = numpy.append(X, date)
        X = X.reshape(len(X), 1)
        new_y_pred = tsr.predict(X)
        y_pred = numpy.append(y_pred, new_y_pred[-1])
    """

    datelist = numpy.append(X, datelist)
    datelist = datelist.reshape(len(datelist), 1)

    y_pred_all = tsr.predict(datelist)
    y_pred_all = [*empty_values, *y_pred_all]
    #y_pred_all = y_pred.copy()
    #y_pred_all = [*empty_values, *y_pred_all]

    return y_pred, y_pred_all


def process_file(file):
    try:
        dataframe = csv_to_formatted_dataframe(file)
    except Exception as err:
        print(str(err), "Skipping file.")
        exit()

    # Grab graphing data
    X = dataframe["Date"].values
    y = dataframe["Net Worth"]

    # Create a list of dates with a 30-day interval
    datelist = numpy.arange(
        str(X[-1]),
        '2019-11-01T00:00:00.0000000',
        numpy.timedelta64(int(24*1.8), 'h'),
        dtype='datetime64'
    )
    datelist = numpy.delete(datelist, 0)

    # Format inputs
    X = X.reshape(len(X), 1)
    datelist = datelist.reshape(len(datelist), 1)
    X_and_datelist = [*X, *datelist]

    # Get predicted values
    lr_y, lr_y_all = run_linear_regression(X, y, X_and_datelist)
    ts_y, ts_y_all = run_time_series(X, y, datelist)
    gpr_y, gpr_y_all = run_gaussian_process_regression(X, y, X_and_datelist)

    plot_predictions(X, y, pred_dates=X_and_datelist, lines=[lr_y_all, ts_y_all, gpr_y_all])


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
        results = process_file(default_location)

    elif len(args) is 2:
        location = Path(args[1])
        if location.is_file():
            results = process_file(location)
        elif location.is_dir():
            print("Folder provided. Please provide a file.")
            exit()
        else:
            print("Please provide a file as an argument.")
            exit()

    else:
        print("Too many arguments provided!")
        exit()


if __name__ == "__main__":
    main(sys.argv)
