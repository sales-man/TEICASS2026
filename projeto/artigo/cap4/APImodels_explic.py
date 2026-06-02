import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

# Modelos e Validação
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)

# Ignorar warnings do Scikit-Learn de forma limpa
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Metricas
metricas = {
    "Acuracia": "accuracy",
    "Precisao": "precision",
    "Recall": "recall",
    "F1": "f1",
}

# Carrega o dataset
def dataset_load():
    print("1. Carregando e embaralhando o dataset reduzido...")
    df_reduced = pd.read_csv('ransomData_Reduced.csv')

    # Embaralhamento completo do dataset (linhas)
    df_reduced_shuf = (
        df_reduced.sample(frac=1, random_state=42).reset_index(drop=True)
    )

    # Salva a base embaralhada
    df_reduced_shuf.to_csv('ransomData_Reduced_Shuffled.csv', index=False)

    # Definição das variáveis (X) e do alvo (y)
    # Remove as 3 primeiras colunas
    col_id, col_label, col_family = df_reduced_shuf.columns[:3]

    X = df_reduced_shuf.drop(columns=[col_id, col_label, col_family])
    y = df_reduced_shuf[col_label]

    print(f"Dataset pronto para treino: {X.shape[0]} amostras e {X.shape[1]} atributos.")

    return X, y

# CrossValidate
# Função de exibição aprimorada e corrigida estatisticamente
def model_crossv_results(model_name, model_crossv, X, y, kfold, total_cv_runs, filename):
    print("\nIniciando testes de validação cruzada...")

    cv_model = cross_validate(
        model_crossv, 
        X, 
        y, 
        cv=kfold, 
        scoring=metricas, 
        n_jobs=-1
    )
    
    # Acumula as linhas de strings do print e salvar em .txt
    lines = []

    lines.append(f"\n===== Resultados: {model_name} =====")
    lines.append(
        f"{'Métrica':<12} | {'Média':<8} | +/- Intervalo de Confiança (95%)"
    )
    lines.append("-" * 55)

    for metric in metricas:
        res_key = "test_" + metric
        scores = cv_model[res_key]

        # Correção Estatística: O desvio padrão deve ser dividido pela raiz do total de rodadas do CV
        # Z-score para 95% de confiança = 1.96
        ci = 1.96 * (scores.std() / np.sqrt(total_cv_runs))

        mean_val = scores.mean() * 100
        ci_val = ci * 100
        lines.append(f"{metric:<12} | {mean_val:>6.2f}% | +/- {ci_val:.2f}%")

    # Printa na console
    for line in lines:
        print(line)

    # Salva no arquivo .txt o modo 'a' faz um "append", adiciona ao final do arquivo sem apagar
    with open(filename, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return cv_model

# GridSearch
def model_gridsearch(model_name, X, y, model_to_tune, model_params, kfold, filename):
    # Lista para acumular as linhas de texto
    lines = []

    lines.append("\n" + "=" * 50)
    lines.append(f"Iniciando ajuste de hiperparâmetros (GridSearchCV) para o Modelo...")

    grid = GridSearchCV(
        estimator=model_to_tune,
        param_grid=model_params,
        scoring=metricas,
        refit="F1",
        cv=kfold,
        verbose=1,
        n_jobs=-1,
    )

    grid.fit(X, y)
    
    lines.append("\nOtimização Concluída!")

    melhor_index = grid.best_index_
    results = grid.cv_results_

    lines.append(f"Melhores parâmetros configurados: {grid.best_params_}\n")
    lines.append("Desempenho do melhor modelo encontrado:")
    lines.append(f" - Acurácia: {results['mean_test_Acuracia'][melhor_index] * 100:.2f}%")
    lines.append(f" - Precisão: {results['mean_test_Precisao'][melhor_index] * 100:.2f}%")
    lines.append(f" - Recall:   {results['mean_test_Recall'][melhor_index] * 100:.2f}%")
    lines.append(f" - F1-Score: {results['mean_test_F1'][melhor_index] * 100:.2f}%")
    lines.append("=" * 50 + "\n")

    # Printa tudo na tela do console
    for line in lines:
        print(line)

    # Salva no arquivo .txt
    with open(filename, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return grid.best_estimator_

# ==============================================================================
#                      GERAÇÃO GRÁFICO BEESWARM
# ==============================================================================
def shap_plot_beeswarm(explainer, shap_values):

    # O SHAP calcula o impacto de todas as colunas de X e plota as 30 mais importantes
    shap.plots.beeswarm(shap_values, max_display=30, show=False)

    # Força o tamanho ideal para 30 variáveis diretamente na figura criada pelo SHAP
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    plt.subplots_adjust(left=0.35, right=0.95, top=0.90, bottom=0.10)

    # Customiza Título e Rótulos
    plt.title(
        "Beeswarm Plot - Impacto das Top 30 APIs (Regressão Logística)",
        fontsize=14,
        pad=25,
        fontweight="bold",
    )
    plt.xlabel("Impacto SHAP (Muda a probabilidade de ser Ransomware)", fontsize=12)

    """
    # Salva a imagem em PNG sem cortar textos longos nas bordas
    plt.savefig(
        "shap_lregression_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("-> Gráfico 'shap_lregression_beeswarm.png' salvo com sucesso!")

    plt.show()
    """

    # Salvar o gráfico em SVG
    plt.savefig("shap_lregression_beeswarm.svg", format="svg", dpi=300)
    print(
        "-> Gráfico nativo 'shap_lregression_beeswarm.svg' salvo em formato vetorial!"
    )
    plt.show()

# ==============================================================================
#                      GERAÇÃO GRÁFICO DECISION PLOT
# ==============================================================================
def shap_plot_decision(explainer, shap_values, X):

    # O Decision Plot mostra como as previsões individuais "viajam" até o resultado final.
    # Plotar as 1524 linhas simultaneamente deixa o gráfico ilegível
    # Melhor prática acadêmica: plotar uma amostra representativa (ex: primeiras 50 instâncias).
    # Definição da amostra (melhor prática acadêmica)
    amostra_idx = 50

    # Extrai o valor base (esperado) do modelo
    expected_value = explainer.expected_value
    # Se o explainer for o novo, o expected_value pode estar dentro do objeto ou ser um array
    if hasattr(expected_value, "__len__") and not isinstance(expected_value, str):
        expected_value = expected_value[1] # Classe 1 (Ransomware) se for array
    elif hasattr(shap_values, "base_values"):
        expected_value = shap_values.base_values[0] # Pega do objeto se disponível

    # Fatia o objeto original do SHAP diretamente.
    # Decision plot aceita o objeto Explanation fatiado ou a matriz pura (.values).
    # Esta abordagem funciona independente da versão do SHAP instalada
    if hasattr(shap_values, "values"):
        # Se for o objeto Explanation moderno, extraímos a matriz NumPy e fatiamos
        shap_matrix = shap_values.values[:amostra_idx]
    else:
        # Se já for um array NumPy puro da API antiga
        shap_matrix = shap_values[:amostra_idx]

    # Garantia de segurança para modelos binários da Regressão Logística:
    # Se a matriz extraída tiver 3 dimensões (amostras, features, classes), filtramos para a classe 1
    if len(shap_matrix.shape) == 3:
        shap_matrix = shap_matrix[:, :, 1]


    # Plotagem do Decision Plot
    # O slice(-30, None) garante a exibição apenas das top 30 variáveis mais impactantes
    shap.plots.decision(
        base_value=expected_value,
        shap_values=shap_matrix,
        features=X.iloc[:amostra_idx],
        feature_display_range=slice(-30, None),
        show=False,
    )

    # Correção nas configuracoes da imagem (12 de largura por 8 de altura)
    fig = plt.gcf()
    fig.set_size_inches(12, 9)
    plt.subplots_adjust(left=0.35, right=0.95, top=0.90, bottom=0.10)

    # Customiza Título e Rótulos (pad=25 evita colisões no topo)
    plt.title(
        f"Decision Plot - Caminho de Decisão para {amostra_idx} Amostras (Top 30 APIs)",
        fontsize=14,
        pad=25,
        fontweight="bold",
    )
    plt.xlabel("Valor de Saída do Modelo (Log-Odds)", fontsize=12)

    """
    # Salva a imagem em PNG sem cortar as APIs na esquerda
    plt.savefig(
        "shap_lregression_bdecision.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("-> Gráfico 'shap_lregression_bdecision.png' salvo com sucesso!")

    plt.show()
    """

    # Salvar o gráfico em SVG
    plt.savefig("shap_lregression_bdecision.svg", format="svg", dpi=300)
    print(
        "-> Gráfico nativo 'shap_lregression_bdecision.svg' salvo em formato vetorial!"
    )
    plt.show()


# ==============================================================================
#               Calculo Nativo Coeficientes Logistc Regression
# ==============================================================================
def plot_calcnativo_lregression(lregression_best_model, X, y):

    # Garante que o modelo está treinado nas features limpas
    lregression_best_model.fit(X, y)

    # Extrai os coeficientes obtidos pelo modelo
    # coef_[0] é usado porque para classificação binária os pesos vêm em uma matriz 1D
    coeficientes = lregression_best_model.coef_[0]

    # Cria DataFrame associando de cada API ao seu respectivo coeficiente
    importancia_df = pd.DataFrame(
        {"API": X.columns, "Coeficiente": coeficientes}
    )

    # Calcula o Valor Absoluto para definir a importância real na decisão
    importancia_df["Importancia_Absoluta"] = it_df = importancia_df[
        "Coeficiente"
    ].abs()

    # Ordena as mais importantes (maior valor absoluto), as top 30
    top30_nativas = importancia_df.sort_values(
        by="Importancia_Absoluta", ascending=False
    ).head(30)

    # --- Exibir a Lista no Terminal ---
    print("\n=== TOP 30 APIs MAIS IMPORTANTES (REGRESSÃO LOGÍSTICA) ===")
    print(
        top30_nativas[
            ["API", "Coeficiente", "Importancia_Absoluta"]
        ].to_string(index=False)
    )

    #Plotar o Gráfico de Barras Nativo
    plt.figure(figsize=(12, 8))

    # Inverte a ordem, a mais importante fica no topo do gráfico
    top30_plot = top30_nativas.iloc[::-1]

    # Definir cores: Vermelho para coeficientes positivos (Ransomware), Azul para negativos (Goodware)
    cores = [
        "darkred" if c > 0 else "darkblue" for c in top30_plot["Coeficiente"]
    ]

    plt.barh(top30_plot["API"], top30_plot["Coeficiente"], color=cores)
    plt.axvline(
        x=0, color="black", linestyle="--", linewidth=1
    )  # Linha divisória no zero

    plt.title(
        "Top 30 APIs por Importância Nativa (Coeficientes da Regressão Logística)",
        fontsize=14,
        pad=20,
    )
    plt.xlabel(
        "Valor do Coeficiente (Positivo = Ransomware | Negativo = Goodware)",
        fontsize=12,
    )
    plt.ylabel("Chamadas de API", fontsize=12)
    #plt.tight_layout()

    """
    # Salvar o gráfico em PNG
    plt.savefig("calcnativo_regressao.png", dpi=300)
    print(
        "\n-> Gráfico nativo 'calcnativa_regressao.png' gerado com sucesso!"
    )
    plt.show()
    """

    # Salvar o gráfico em SVG
    plt.savefig("calcnativo_regressao.svg", format="svg", dpi=300)
    print(
        "-> Gráfico nativo 'calcnativo_regressao.svg' salvo em formato vetorial!"
    )
    plt.show()