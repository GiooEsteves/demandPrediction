import os
from src.data.load_data import (
    load_holidays_data, load_oil_data, load_stores_data,
    load_transactions_data, load_train_data, load_test_data
)
from src.data.preprocess import clean_train_data

def main():
    # Carregar os dados
    holidays_df = load_holidays_data()
    oil_df = load_oil_data()
    stores_df = load_stores_data()
    transactions_df = load_transactions_data()
    train_df = load_train_data()
    test_df = load_test_data()

    # Limpar e pré-processar os dados de treinamento
    train_clean_df = clean_train_data(train_df, oil_df, holidays_df, transactions_df)
    
    # Criar o diretório 'data/processed' se não existir
    processed_dir = 'data/processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Salvar os dados limpos em 'data/processed/train_clean.csv'
    output_path = os.path.join(processed_dir, 'train_clean.csv')
    train_clean_df.to_csv(output_path, index=False)
    print(f"Arquivo CSV salvo em: {output_path}")

if __name__ == "__main__":
    main()
