# Requirements
+ Python3.7
+ Pipenv
    + `pip3 install --user pipenv`


# First time set up
from the root of the project, run `pipenv install`.


# How to run the code with your data
1. Download your transactions from Mint
2. Move the transactions.csv file to a `datasets/` folder under the root of the project
3. Either `pipenv run python main.py`
    Or `pipenv shell` followed by `python main.py`
    _You can exit pipenv shell with `CTRL+D`_


# Training set for the regressions
>Obtained from https://github.com/awesomedata/awesome-public-datasets/issues/234

Data descriptions: https://web.archive.org/web/20161019192412/http://lisp.vse.cz/pkdd99/berka.htm


# Points that could be improved
+ Inflation is not accounted for
+ Since there may be lots of transactions on one day, and then a long break, it might make more sense to calculate the between the predicted, and a line joining every test example.