import pandas as pd
from scipy import stats
import numpy as np

def clean_train_data(train_df, oil_df, holidays_df, transactions_df):
    # Convertendo colunas de data para o formato datetime em todos os DataFrames
    train_df['date'] = pd.to_datetime(train_df['date'])
    oil_df['date'] = pd.to_datetime(oil_df['date'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    # Preenchendo valores ausentes na coluna 'sales'
    train_df['sales'] = train_df['sales'].fillna(train_df['sales'].mean())
    
    # Mesclando com outros datasets
    train_df = pd.merge(train_df, oil_df, on='date', how='left')
    train_df = pd.merge(train_df, holidays_df, on='date', how='left')
    train_df = pd.merge(train_df, transactions_df, on=['store_nbr', 'date'], how='left')
    
    # Removendo outliers
    z_scores = stats.zscore(train_df['sales'])
    abs_z_scores = np.abs(z_scores)
    train_df = train_df[(abs_z_scores < 3)]
    
    # Codificando variáveis categóricas
    train_df = pd.get_dummies(train_df, columns=['family'], drop_first=True)
    
    return train_df
