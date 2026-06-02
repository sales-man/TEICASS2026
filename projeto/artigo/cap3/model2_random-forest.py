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
rforest_crossv = model_crossv_results("Random Forest", RandomForestClassifier(), X, y, kfold_run(), total_cv_runs, "resultados_rforest_crossv1.txt")

# Ajustes de hiperparametros com GridSearch
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],            # Evita que a árvore cresça infinitamente
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced", None,],    # Ajuda se houver desbalanceamento de classes
}

# RForest GridSearch
rforest_best_model = model_gridsearch("Random Forest GridSearch", X, y, RandomForestClassifier(), rf_params, kfold_run(), "resultados_rforest_grid.txt")

# RForest Best CrossValidate
model_crossv_results("Random Forest Hiperparametros Otim", rforest_best_model, X, y, kfold_run(), total_cv_runs, "resultados_rforest_otim.txt")

# Teste parametros especificos
### rforest_model = RandomForestClassifier(class_weight= None, criterion= 'entropy', max_depth= None, n_estimators= 200)
### model_crossv_results("Random Forest Hiperparametros TestParams", rforest_model, X, y, kfold_run(), total_cv_runs, "resultados_rforest_testparams.txt")