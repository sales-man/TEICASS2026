import pandas as pd

# Carrega os nomes de famílias
# 'Family Names ID.txt' tem formato de tabela textual, é tratado os espaços
family_mapping = {}
with open('Family Names ID.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # Ignora cabeçalhos ou linhas vazias
        if not line or 'FAMILY NAME' in line or '---' in line:
            continue
        
        # Divide a linha baseado no ID numérico que fica no final
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            name, idx = parts[0].strip().replace("'", ""), int(parts[1])
            family_mapping[idx] = name

# Carrega dataset com os headers configurados
df = pd.read_csv('RansomwareData_Com_Headers.csv', low_memory=False)

# Identifica nomes exatos das colunas com base no VariableNames.txt
col_label = 'Label (1 Ransomware / 0 Goodware)'
col_family = 'Ransomware Family'

# Remove as colunas de ID, Label e Family
num_features = df.shape[1] - 3 

# --- Estatísticas Automáticas ---
total = len(df)
malware = df[df[col_label] == 1].shape[0]
goodware = df[df[col_label] == 0].shape[0]
familias_unicas = df[col_family].nunique()

# --- Abre um .txt para salvar os resultados ---
with open('resultado_estatisticas.txt', 'w', encoding='utf-8') as arquivo_saida:
    
    # Adiciona argumento file=arquivo_saida em cada print
    print(f"Total de Amostras: {total}", file=arquivo_saida)
    print(f"Ransomwares: {malware} ({(malware/total)*100:.2f}%)", file=arquivo_saida)
    print(f"Goodwares: {goodware} ({(goodware/total)*100:.2f}%)", file=arquivo_saida)
    print(f"Famílias Únicas Detectadas: {familias_unicas}", file=arquivo_saida)
    print(f"Total de Atributos Computados (Features): {num_features}", file=arquivo_saida)

    # --- Detalhamento por Família ---
    print("\n--- Amostras por Família de Ransomware ---", file=arquivo_saida)
    family_counts = df[col_family].value_counts()

    for idx, count in family_counts.items():
        nome_familia = family_mapping.get(idx, f"ID Desconhecido ({idx})")
        print(f"- {nome_familia:<20}: {count} amostras", file=arquivo_saida)

print("O resultado foi salvo com sucesso em 'resultado_estatisticas.txt'!")

