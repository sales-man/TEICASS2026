```markdown
# Treinamento de Modelos de Machine Learning para Detecçaode Ransomware Baseado em Dataset Publico - Um Estudo de Caso de 4 Modelos

Este repositório contém o projeto completo de pesquisa focado na detecção de ransomware através de análise de modelos de Machine Learning. O trabalho aborda desde o processamento de dados comportamentais até a explicabilidade dos modelos.

## 📌 Visão Geral

Este trabalho apresenta um estudo de caso focado no treinamento e avaliaçãao de quatro modelos de Machine Learning: **BernoulliNB, Random Forest, Logistic Regression e KNN**. Utilizando um dataset de an´alise dinˆamica com 1.524 amostras (582 ransomware e 942 goodwares), a metodologia empregou a técnica de Variance Threshold para reduzira alta dimensionalidade de 30.967 para 485 atributos bin´arios essenciais, como chamadas de API e operac¸ ˜oes de registro. Os modelos foram otimizados via GridSearchCV, resultando em desempenhos superiores para a Regressão Logistica e Random Forest, ambos com F1-score acima de 96%. Adicionalmente, o estudo explorou a interpretabilidade dos modelos atrav´es de coeficientes nativos e do framework SHAP, identificando padr˜oes comportamentais decisivos para a classificaçãao das ameaças.


### 🌟 Destaques do Projeto
* **Análise Dinâmica:** Uso de logs comportamentais (Cuckoo Sandbox) para evitar técnicas de evasão tradicionais baseadas em análise estática.
* **Engenharia de Atributos:** Redução drástica de dimensionalidade de **30.967** para **485** atributos binários utilizando a técnica de *Variance Threshold*.
* **Explicabilidade (XAI):** Implementação de análise interpretável via **SHAP** (*SHapley Additive exPlanations*) e coeficientes de regressão para auditar as decisões tomadas pelos modelos.

---

## 📊 Resultados e Desempenho

Foram testados quatro algoritmos de aprendizado de máquina com otimização rigorosa de hiperparâmetros via `GridSearchCV`. Os resultados obtidos (representando a média com intervalo de confiança de 95%) foram:

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 97,82 ± 0,20 | 97,58 ± 0,36 | 96,70 ± 0,44 | 97,12 ± 0,26 |
| **Logistic Regression** | 97,69 ± 0,23 | 97,35 ± 0,37 | 96,60 ± 0,43 | 96,96 ± 0,30 |
| **KNN** | 95,58 ± 0,23 | 93,99 ± 0,62 | 94,57 ± 0,54 | 94,24 ± 0,29 |
| **BernoulliNB** | 82,09 ± 0,57 | 71,77 ± 0,78 | 87,80 ± 0,79 | 78,94 ± 0,62 |

---

## 📁 Estrutura do Repositório

```text
├── aulas/                        <-- Materiais utilizados em aula
├── projeto/                      <-- Conjunto de todo matreial usado para elaborar o trabalho de conclusao da disciplina
│   └── artigo/                   <-- Raiz de tudo que foi usado para elaborar o artigo  
│       └── ArtigoTEICASS_TeX/    <-- Projeto LaTeX do artigo
│       └── cap#1#2#3#4/          <-- Materiais organizados de acordo com conteudo abordado em cada Capitulo/Secção
│   └── dataset/                  <-- Dataset original utilizado na confecção do trabalho **Sgandurra 2016**
│   └── dataset_tratamento/       <-- Tratamento preliminar do dataset para poder ser utilizado pelos codigos desenvolvidos
│
└── README.md
