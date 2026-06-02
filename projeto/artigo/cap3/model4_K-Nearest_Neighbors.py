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
knn_crossv = model_crossv_results("KNeighbors Classifier", KNeighborsClassifier(), X, y, kfold_run(), total_cv_runs, "resultados_knn_crossv1.txt")

# Ajustes de hiperparametros com GridSearch
knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["manhattan", "euclidean"],  # Manhattan costuma ganhar em dados binários
}

# KNN GridSearch
knn_best_model = model_gridsearch("KNN GridSearch", X, y, KNeighborsClassifier(), knn_params, kfold_run(), "resultados_knn_grid.txt")

# KNN Best CrossValidate
model_crossv_results("KNN Hiperparametros Otim", knn_best_model, X, y, kfold_run(), total_cv_runs, "resultados_knn_otim.txt")

# Teste parametros especificos
### knn_model = KNeighborsClassifier(metric= 'manhattan', n_neighbors= 3, weights= 'distance')
### model_crossv_results("KNN Hiperparametros TestParams", knn_model, X, y, kfold_run(), total_cv_runs, "resultados_knn_testparams.txt")