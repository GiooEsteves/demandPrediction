import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def select_important_features(df, target):
    # Verificar se a coluna target está presente
    if target not in df.columns:
        raise ValueError(f"Coluna target '{target}' não encontrada no DataFrame.")
    
    # Remover colunas não numéricas
    non_numeric_cols = df.select_dtypes(exclude=[float, int]).columns
    df = df.drop(columns=non_numeric_cols, errors='ignore')
    
    # Separar features e target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar um modelo básico para avaliação
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Avaliar o modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    
    # Selecionar features importantes baseadas na importância fornecida pelo modelo
    feature_importances = model.feature_importances_
    important_features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
    
    print("Features importantes:")
    print(important_features)
    
    # Selecionar as principais features (ajustar o número conforme necessário)
    top_features = important_features.head(10).index
    selected_df = df[top_features.to_list() + [target]]
    
    return selected_df

if __name__ == "__main__":
    # Caminho para o dataset com novas features
    features_path = os.path.join('data', 'processed', 'train_features.csv')

    # Carregar dataset com novas features
    features_df = pd.read_csv(features_path, low_memory=False)

    # Selecionar features importantes
    target = 'sales'
    selected_features_df = select_important_features(features_df, target)

    # Salvar dataset com features selecionadas
    selected_features_path = os.path.join('data', 'processed', 'train_selected_features.csv')
    selected_features_df.to_csv(selected_features_path, index=False)
    print(f'Dataset com features selecionadas salvo em: {selected_features_path}')
