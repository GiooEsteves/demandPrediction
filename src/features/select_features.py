import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def select_important_features(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # Treinamento de um modelo para determinar a importância das features
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Selecionando as features mais importantes
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    selected_features = feature_importances.nlargest(10).index.tolist()  # Seleciona as 10 features mais importantes

    # Retorna o dataframe apenas com as features selecionadas e a coluna target
    selected_df = df[selected_features + [target]]

    return selected_df

if __name__ == "__main__":
    # Carregar dados com novas features
    feature_path = os.path.join('data', 'processed', 'train_features.csv')
    train_features_df = pd.read_csv(feature_path)

    # Selecionar features importantes
    target = 'sales'
    train_selected_df = select_important_features(train_features_df, target)

    # Salvar dataset com features selecionadas
    selected_feature_path = os.path.join('data', 'processed', 'train_selected_features.csv')
    train_selected_df.to_csv(selected_feature_path, index=False)
    print(f'Dataset com features selecionadas salvo em: {selected_feature_path}')
