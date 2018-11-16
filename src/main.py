import numpy
import pandas
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from feature_engineering import csv_handler
from graphing.graphing import plot_data, plot_prediction


project_root = Path(__file__).resolve().parent.parent


def test_csv_handler():
    # Load the personal finance dataset
    transactions_file = project_root/"datasets/preformatted.csv"
    data = csv_handler.format_dataset(transactions_file)

    # Split into feature and target sets
    X = data["Date"]
    y = data["Net Worth"]

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=False)

    # Plot outputs
    plot_data(X_train, y_train, timeout=3000)


def run_linear_regression():
    # Load the personal finance dataset
    transactions_file = project_root/"datasets/preformatted.csv"
    data = csv_handler.format_dataset(transactions_file)

    # Split into feature and target sets
    X = data["Date"].values
    y = data["Net Worth"]

    # Convert 1-D array to 2-D feature array, as expected by sklearn
    X = X.reshape(len(X), 1)

    # Plot all data
    plot_data(X, y, timeout=None)

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
    y_pred = regression.predict(X_test_numeric)
    # y_pred = regression.predict(X_numeric)

    # Graph
    plot_prediction(X_test, y_test, y_pred=y_pred, timeout=None)
    # plot_prediction(X, y, y_pred=y_pred, timeout=None)

    # The coefficients
    print('Coefficients: \n', regression.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))


def main():
    # Load the personal finance dataset
    transactions_file = project_root/"datasets/transactions.csv"
    X = csv_handler.format_dataset(transactions_file)

    # Linear Regression
    run_linear_regression()

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(y_test, y_pred))


if __name__ == "__main__":
    main()
