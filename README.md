# Requirements
+ Python3.7
+ Pipenv
    + `pip3 install --user pipenv`

# How to run the code
1. Download your transactions from Mint
2. Move the transactions.csv file to a `datasets/` folder under the root of the project  
3. Either `pipenv run python main.py`
    Or `pipenv shell` followed by `python main.py`
    _You can exit pipenv shell with `CTRL + D`

# Points that could be improved
+ Inflation is not accounted for
