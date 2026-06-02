import pandas as pd
from sklearn.feature_selection import VarianceThreshold

print("1. Carregando o dataset com headers...")
df = pd.read_csv('RansomwareData_Com_Headers.csv', low_memory=False)

# Define os nomes das 3 colunas iniciais
col_id = df.columns[0]      # 'ID'
col_label = df.columns[1]   # 'Label (1 Ransomware / 0 Goodware)'
col_family = df.columns[2]  # 'Ransomware Family'

# Separa as variáveis (X) e o alvo (y)
X = df.drop(columns=[col_id, col_label, col_family])
y = df[col_label]

print(f"Colunas originais de atributos (X): {X.shape[1]}")

print("\n2. Aplicando o filtro de baixa variância...")
# p = 0.97 remove colunas onde o mesmo valor se repete em mais de 97% das amostras
p = 0.97
selector = VarianceThreshold(threshold=(p * (1 - p)))

# Ajusta o seletor e transforma os dados
X_reduced_array = selector.fit_transform(X)

# Recuper os nomes das colunas apos o filtro
surviving_features = selector.get_feature_names_out(input_features=X.columns)

# Converte o array resultante para um DataFrame com os nomes reais das colunas
X_reduced_df = pd.DataFrame(X_reduced_array, columns=surviving_features)

print("\n3. Estruturando o DataFrame reduzido...")
# Concatena as 3 colunas de metadados com as colunas filtradas
df_reduced = pd.concat([df[[col_id, col_label, col_family]], X_reduced_df], axis=1)

print("\n--- Resumo da Redução ---")
print(f"Colunas totais no dataset original : {df.shape[1]}")
print(f"Colunas totais no 'ransomData_Reduced': {df_reduced.shape[1]}")
print(f"Atributos removidos por baixa variância: {X.shape[1] - len(surviving_features)}")

# Salva o arquivo final reduzido
print("\n4. Salvando o arquivo 'ransomData_Reduced.csv'...")
df_reduced.to_csv('ransomData_Reduced.csv', index=False)
print("Processo concluído com sucesso!")