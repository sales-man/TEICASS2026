# Treinamento de Modelos de Machine Learning para Detecção de Ransomware Baseado em Dataset Público - Um Estudo de Caso de 4 Modelos

Este repositório contém o projeto completo de pesquisa focado no treinamento de modelos de Machine Learning para detecção de ransomware através de análise de logs de eventos, desenvolvido como trabalho de conclusão da disciplina Tópicos Especiais em Inteligência Computacional Aplicada à Segurança de Sistemas, ministrada em 2026, como parte do Programa de Pós-graduação em Computação Aplicada (PPCA), vinculado ao Núcleo de Desenvolvimento Amazônico em Engenharia (NDAE), da Universidade Federal do Pará (UFPA).
O trabalho aborda desde o processamento de dados comportamentais até a explicabilidade dos modelos.

## 📌 Visão Geral

Este trabalho apresenta um estudo de caso focado no treinamento e avaliação de quatro modelos de Machine Learning: `BernoulliNB`, `Random Forest`, `Logistic Regression` e `KNN`. Utilizando o dataset publico `Sgandurra 2016`, que consiste em logs de eventos de execuço em sandbox (análise dinámica), com `1.524 amostras` (582 ransomware e 942 goodwares), a metodologia empregou a técnica de `Variance Threshold` para reduzir a alta dimensionalidade de `30.967` para `485` atributos binários essenciais, como chamadas de API e operações de registro. Os modelos foram otimizados via `GridSearchCV`, resultando em desempenhos superiores para a `Regressão Logística` e `Random Forest`, ambos com F1-score acima de 96%. Adicionalmente, o estudo explorou a interpretabilidade dos modelos através de `coeficientes nativos` e do framework `SHAP`, identificando padrões comportamentais decisivos para a classificação das ameaças.

---

## 📄 Trabalho Final (Artigo Científico)

O artigo científico completo desenvolvido ao longo desta disciplina, detalhando toda a fundamentação teórica, metodologia de análise dinâmica e discussão dos resultados, está disponível para leitura e download no link abaixo:

👉 [**Acesso direto ao Artigo Completo (PDF)**](https://github.com/sales-man/TEICASS2026/blob/main/projeto/artigo/ArtigoTEICASS.pdf)

---

## 🌟 Destaques do Projeto
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
├── aulas/                       <-- Materiais utilizados em aula
├── projeto/                     <-- Conjunto de todo o material usado para elaborar o trabalho de conclusão da disciplina
│   ├── artigo/                  <-- Raiz de tudo que foi usado para elaborar o artigo
│   │   ├── ArtigoTEICASS_TeX/   <-- Projeto LaTeX do artigo
│   │   └── cap#1#2#3#4/         <-- Materiais organizados de acordo com o conteúdo abordado em cada Capítulo/Secção
│   ├── dataset/                 <-- Dataset original utilizado na confecção do trabalho (Sgandurra, 2016)
│   └── dataset_tratamento/      <-- Tratamento preliminar do dataset para poder ser utilizado pelos códigos desenvolvidos
└── README.md
