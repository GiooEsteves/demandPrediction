import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def evaluate_model(model, X_test, y_test):
    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Calcular métricas de avaliação
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R²: {r2}')

    return rmse, mae, r2

if __name__ == "__main__":
    # Caminho para o modelo treinado
    model_path = os.path.join('src', 'models', 'demand_forecasting_model.pkl')
    
    # Caminho para os dados processados
    test_path = os.path.join('data', 'processed', 'test_clean.csv')

    # Carregar o modelo treinado
    model = joblib.load(model_path)
    print(f'Modelo carregado de: {model_path}')

    # Carregar os dados de teste
    test_df = pd.read_csv(test_path)
    
    # Separar features e target
    X_test = test_df.drop(columns=['sales'])
    y_test = test_df['sales']

    # Avaliar o modelo
    evaluate_model(model, X_test, y_test)
