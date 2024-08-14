import pandas as pd

def load_holidays_data(filepath='data/raw/holidays_events.csv'):
    return pd.read_csv(filepath)

def load_oil_data(filepath='data/raw/oil.csv'):
    return pd.read_csv(filepath)

def load_stores_data(filepath='data/raw/stores.csv'):
    return pd.read_csv(filepath)

def load_transactions_data(filepath='data/raw/transactions.csv'):
    return pd.read_csv(filepath)

def load_train_data(filepath='data/raw/train.csv'):
    return pd.read_csv(filepath)

def load_test_data(filepath='data/raw/test.csv'):
    return pd.read_csv(filepath)
