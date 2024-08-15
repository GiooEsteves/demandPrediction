import os
import pandas as pd

def build_features(df):
    # Extração de informações temporais
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Agregações baseadas em vendas passadas
    df['sales_lag_7'] = df['sales'].shift(7)
    df['sales_lag_30'] = df['sales'].shift(30)
    df['sales_rolling_mean_7'] = df['sales'].shift(7).rolling(window=7).mean()
    df['sales_rolling_mean_30'] = df['sales'].shift(30).rolling(window=30).mean()

    # Preenchendo valores NaN resultantes das transformações
    df.fillna(0, inplace=True)

    return df

if __name__ == "__main__":
    # Carregar dados processados
    train_path = os.path.join('data', 'processed', 'train_clean.csv')
    train_df = pd.read_csv(train_path)

    # Construir novas features
    train_features_df = build_features(train_df)

    # Salvar dataset com features
    feature_path = os.path.join('data', 'processed', 'train_features.csv')
    train_features_df.to_csv(feature_path, index=False)
    print(f'Dataset com novas features salvo em: {feature_path}')
