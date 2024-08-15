import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Carregar as funções de preprocessamento e seleção de features
from src.data.preprocess import clean_train_data
from src.features.build_features import build_features
from src.features.select_features import select_important_features

def train_model(train_df, oil_df, holidays_df, transactions_df):
    # Limpeza e preprocessamento dos dados
    train_clean_df = clean_train_data(train_df, oil_df, holidays_df, transactions_df)
    
    # Construir novas features
    train_features_df = build_features(train_clean_df)
    
    # Seleção de features importantes
    target = 'sales'
    train_selected_df = select_important_features(train_features_df, target)

    # Separar features e target
    X = train_selected_df.drop(columns=[target])
    y = train_selected_df[target]

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar o modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'RMSE: {rmse}')

    # Salvar o modelo treinado
    model_path = os.path.join('src', 'models', 'demand_forecasting_model.pkl')
    joblib.dump(model, model_path)
    print(f'Modelo salvo em: {model_path}')

    return model, rmse

if __name__ == "__main__":
    # Caminho para os dados
    train_path = os.path.join('data', 'processed', 'train_clean.csv')
    oil_path = os.path.join('data', 'processed', 'oil_clean.csv')
    holidays_path = os.path.join('data', 'processed', 'holidays_clean.csv')
    transactions_path = os.path.join('data', 'processed', 'transactions_clean.csv')

    # Carregar os dados processados
    train_df = pd.read_csv(train_path)
    oil_df = pd.read_csv(oil_path)
    holidays_df = pd.read_csv(holidays_path)
    transactions_df = pd.read_csv(transactions_path)

    # Treinar o modelo
    train_model(train_df, oil_df, holidays_df, transactions_df)
