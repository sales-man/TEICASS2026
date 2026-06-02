from APImodels_explic import *

# Configuraça a Validação Cruzada
num_folds = 5
num_repeats = 10
total_cv_runs = num_folds * num_repeats  # Total de iterações executadas = 20 (num_folds = 2 * num_repeats = 10)

# Carrega o dataset
X, y = dataset_load()

# Configurar a estratégia de validação cruzada
def kfold_run():
    return RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)

# Execução dos Benchmarks do Modelo
# lregression_crossv = model_crossv_results("Logistic Regression", LogisticRegression(), X, y, kfold_run(), total_cv_runs, "resultados_lregression_crossv1.txt")

# Ajustes de hiperparametros com GridSearch
lregression_params = [
    {
        "solver": ["saga"],
        "penalty": ["l1", "l2"],    # L1 fará a seleção das APIs importantes
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [1000],
    }
]

# LRegression GridSearch
lregression_best_model = model_gridsearch("Logistic Regression GridSearch", X, y, LogisticRegression(), lregression_params, kfold_run(), "resultados_lregression_grid.txt")

# LRegression CrossValidate
model_crossv_results("Logistic Regression Hiperparametros Otim", lregression_best_model, X, y, kfold_run(), total_cv_runs, "resultados_lregression_otim.txt")

# Teste parametros
# Instancia o modelo com os melhores hiperparâmetros do GridSearch
### lregression_model = LogisticRegression(C=1.0, max_iter=1000, penalty='l2', solver='saga')
### model_crossv_results("Logistic Regression Hiperparametros Otim", lregression_model, X, y, kfold_run(), total_cv_runs, "resultados_lregression_testparams.txt")


# Plota os Resultados do Calculo Nativo  na Regressão Logística através dos Coeficientes
print("\n--- Calculando a Importância Nativa (Coeficientes) ---")
plot_calcnativo_lregression(lregression_best_model, X, y)


# Plota os Resultados do SHAP (BEESWARM e DECISION PLOT)
print("\n--- Gerando Gráficos Avançados do SHAP ---")

# Cria um "masker" explícito dizendo para aceitar até 1524 amostras
masker = shap.maskers.Independent(X, max_samples=1524)

# --- Cria o Explainer do SHAP ---
# O SHAP possui otimizadores nativos para cada tipo de modelo:
# Random Forest -> usar: shap.TreeExplainer(modelo_otimizado)
# Regressão Logística -> usar: shap.Explainer(modelo_otimizado, X_features)
# masker configurado para dentro do Explainer
explainer = shap.Explainer(lregression_best_model, masker)

# --- Calcular os SHAP Values ---
# Calcula o impacto de cada chamada de API para cada linha de dados
shap_values = explainer(X)

# Plota grafico BEESWARM
shap_plot_beeswarm(explainer, shap_values)

# Plota grafico 
shap_plot_decision(explainer, shap_values, X)



