# Project description
The objective of this research project was to evaluate the feasibility of predicting an individual’s personal finances based on past results with a greater accuracy than a simple linear regression. This sort of result could, in practice, be used to help people predict their retirement date, which would in turn, help them assess if they are living within their means.

see the `docs/` folder for more details.

# Requirements
+ Python 3.7
+ Pipenv
    + `pip3 install --user pipenv`


# First time set up
From the root of the project, run `pipenv install`.

# Quick start
If you just want to see the program at work, here are instructions to quickly get _something_ running with the provided datasets.

**Before you get started**
Please ensure that you have completed the steps under the _Requirements_ and _First time set up_ sections of this README file.

#### Running the model selection script
1. From the terminal at the root of this project directory, run `pipenv shell`
2. Navigate to the `src/` folder: `cd src`
3. Run `python model_selection.py`

This will run the model selection program on a single file (preformatted.csv)

#### Running the predictor
1. From the terminal at the root of this project directory, run `pipenv shell`
2. Navigate to the `src/` folder: `cd src`
3. Run `python finance_predictor.py`

This will run the the finance predictor program on a single file (preformatted.csv)
Note: this file is still unstable, and will likely not work with other transactions files because of some hard-coded values.

#### Using the Czech Bank data
1. From the terminal at the root of this project directory, run `pipenv shell`
2. Navigate to the `src/` folder: `cd src`
3. Split the dataset into a folder of datastes `python  split_dataset.py ../datasets/trans.asc`
4. You can now run the model selection on that entire folder of csv files: `python model_selection.py ../datasets/trans_split/`

Note: This will likely take a while, so I suggest either stopping the process with `CTRL+C` or deleting most of the files in the `../datasets/trans_split/` folder to speed things up.


# How to run a prediction with your data
1. Download your transactions from Mint
2. Move the transactions.csv file to a `datasets/` folder under the root of the project
3. Navigate to the `src/` folder
4. Either `pipenv run python predict_finances.py ../datasets/transactions.csv`
    Or `pipenv shell` followed by `python predict_finances.py ../datasets/transactions.csv`
    _You can exit pipenv shell with `CTRL+D`_

Note: _I have provided some sample files under the datasets folder._

# Results

The results of this experiment are outlined in the research report in the `docs/` folder.

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
+ More improvements related to the models themselves are listed in the project report in the `docs/` folder.


---


Thank you!

**Dominic Roy-Stang**
