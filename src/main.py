import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from feature_engineering import csv_handler
from graphing.graphing import plot_data


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
    plot_data(X, y, timeout=3000)

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=False)

    # Create linear regression object
    regression = LinearRegression()
    print(len(X_train))
    print(len(y_train))

    # Train the model using the training sets
    regression.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regression.predict(X_test)


def main():
    # Load the personal finance dataset
    transactions_file = project_root/"datasets/transactions.csv"
    X = csv_handler.format_dataset(transactions_file)
    y = X.pop("Net Worth").values
    X = X.pop("Date").values
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Split the data into training/testing sets
    X_train = X[:-50]
    X_test = X[-50:]

    # Split the targets into training/testing sets
    y_train = y[:-50]
    y_test = y[-50:]

    # Plot outputs
    plt.scatter(X_train, y_train,  color='black')
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    # y_pred = regr.predict(X_test)

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(y_test, y_pred))


if __name__ == "__main__":
    run_linear_regression()
