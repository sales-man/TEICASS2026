import pandas as pd
import csv

print("1. Lendo os nomes das variáveis...")
# VariableNames usa ';' como separador
# Le a segunda coluna (índice 1), contém os nomes das variáveis
names_df = pd.read_csv('VariableNames.txt', sep=';', header=None, usecols=[1], quoting=csv.QUOTE_NONE)

# Converte a coluna em uma lista de strings para usar como cabeçalho
colunas_nomes = names_df[1].tolist()

print(f"Total de nomes de colunas encontrados: {len(colunas_nomes)}")

print("\n2. Carregando o dataset de Ransomware (sem header)...")
# Carrega o CSV informando que ele NÃO possui uma linha de cabeçalho original (header=None)
# Define low_memory=False para evitar alertas devido ao tamanho das colunas
df = pd.read_csv('RansomwareData.csv', sep=',', header=None, low_memory=False)

print(f"Formato original do Dataset: {df.shape[0]} linhas e {df.shape[1]} colunas.")

# Garante que a quantidade de nomes bate com a quantidade de colunas
if len(colunas_nomes) == df.shape[1]:
    print("\n3. Ajustando os nomes das colunas...")
    df.columns = colunas_nomes
    print("Nomes aplicados com sucesso!")
else:
    print(f"\n[Aviso] Divergência detectada: O arquivo de texto tem {len(colunas_nomes)} nomes, "
          f"mas o CSV tem {df.shape[1]} colunas.")
    # Ajusta usando o mínimo possível para não travar o código
    min_cols = min(len(colunas_nomes), df.shape[1])
    df = df.iloc[:, :min_cols]
    df.columns = colunas_nomes[:min_cols]
    print(f"Dataset truncado e ajustado para {min_cols} colunas.")

# Mostra as primeiras 5 colunas com os novos nomes para validação visual
print("\nPrimeiras colunas do DataFrame atualizado:")
print(df.iloc[:, :5].head())

# Salva o arquivo corrigido
print("\n4. Salvando o novo arquivo 'RansomwareData_Com_Headers.csv'...")
df.to_csv('RansomwareData_Com_Headers.csv', index=False)
print("Concluído!")