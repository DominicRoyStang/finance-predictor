import numpy
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from pathlib import Path
from feature_engineering import csv_handler
from graphing.graphing import plot_data


project_root = Path(__file__).resolve().parent.parent


def test_csv_handler():
    # Load the personal finance dataset
    transactions_file = project_root/"datasets/preformatted.csv"
    data = csv_handler.format_dataset(transactions_file)
    print(data.shape)
    print(data.head())
    X = data["Net Worth"]
    y = data["Date"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=False)
    # Plot outputs
    plot_data(X_train, y_train, timeout=3000)


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
    regr = linear_model.LinearRegression()

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
    test_csv_handler()
