from APImodels import *

# Configuração inputs da Validação Cruzada
num_folds = 5
num_repeats = 10
total_cv_runs = num_folds * num_repeats  # Total de iterações executadas = 20 (num_folds = 2 * num_repeats = 10)

# Carrega o dataset
X, y = dataset_load()

# Configurar a estratégia de validação cruzada
def kfold_run():
    return RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)

# Primeira execucao do CrossValidadte
lregression_crossv = model_crossv_results("Logistic Regression", LogisticRegression(), X, y, kfold_run(), total_cv_runs, "resultados_lregression_crossv1.txt")

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

# LRegression Best CrossValidate
model_crossv_results("Logistic Regression Hiperparametros Otim", lregression_best_model, X, y, kfold_run(), total_cv_runs, "resultados_lregression_otim.txt")

# Teste parametros especificos
### lregression_model = LogisticRegression(C=1.0, max_iter=1000, penalty='l2', solver='saga')
### model_crossv_results("Logistic Regression Hiperparametros TestParams", lregression_model, X, y, kfold_run(), total_cv_runs, "resultados_lregression_testparams.txt")
