# Requirements
+ Python3.7
+ Pipenv
    + `pip3 install --user pipenv`


# First time set up
from the root of the project, run `pipenv install`.


# How to run a prediction with your data
1. Download your transactions from Mint
2. Move the transactions.csv file to a `datasets/` folder under the root of the project
3. Navigate to the `src/` folder
4. Either `pipenv run python predict_finances.py ../datasets/transactions.csv`
    Or `pipenv shell` followed by `python predict_finances.py ../datasets/transactions.csv`
    _You can exit pipenv shell with `CTRL+D`_


# Why was linear regression chosen?
After comparing it with support vector regression and gaussian process regression, linear regression was found to have the best r-squared score.
It was compared by running all three processes on the data from 10,000 czech bank accounts, and measuring their performance.

If you wish to **validate my results**, follow these steps:
1. Convert the trans.asc aggregate file into individual csv files for each transaction.
    from the root of the repository, run `pipenv run python src/split_dataset.py datasets/trans.asc`
    This will create a trans_split/ directory containing one file per account's transactions under the datasets folder.
2. Run `model_selection.py` on the new generated file
    from the root of the repository, run `pipenv run python src/model_selection.py datasets/trans_split/`
    This process will take several minutes. Be patient.


## Training set for the regressions
The `trans.asc` file used was obtained from [here](https://github.com/awesomedata/awesome-public-datasets/issues/234)
Data descriptions can be accessed [here](https://web.archive.org/web/20161019192412/http://lisp.vse.cz/pkdd99/berka.htm)


# Points that could be improved
+ Inflation is not accounted for
+ Since there may be lots of transactions on one day, and then a long break, it might make more sense to calculate the area between the predicted, and a line joining every test example.