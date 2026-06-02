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

# Naive Bayes (Excelente Benchmark para dados esparsos/binários)
# Primeira execucao do CrossValidadte
bernoulli_crossv = model_crossv_results("Bernoulli Naive Bayes", BernoulliNB(), X, y, kfold_run(), total_cv_runs, "resultados_bernoulli_crossv1.txt")

# Ajustes de hiperparametros com GridSearch
bernoulli_params = {
    "alpha": [0.001, 0.01, 0.1, 0.5, 1.0],  # Suavização de Laplace
    "binarize": [None, 0.0],                # Removidos limiares altos. Usa 0.0 pois os dados já são binários em vez de limiares altos
    "fit_prior": [True, False],
    }
# Bernoulli GridSearch
bernoulli_best_model = model_gridsearch("Bernoulli Naive Bayes GridSearch", X, y, BernoulliNB(), bernoulli_params, kfold_run(), "resultados_bernoulli_grid.txt")

# Bernoulli Best CrossValidate
model_crossv_results("BernoulliNB Hiperparametros Otim", bernoulli_best_model, X, y, kfold_run(), total_cv_runs, "resultados_bernoulli_otim.txt")

# Teste parametros especificos
### bernoulli_model = BernoulliNB(alpha= 0.001, binarize= None, fit_prior= True)
### model_crossv_results("BernoulliNB Hiperparametros TestParams", bernoulli_model, X, y, kfold_run(), total_cv_runs, "resultados_bernoulli_testparams.txt")
