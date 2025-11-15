# Previsão Temporal de Carga Elétrica com Modelos Lineares, MLP, LSTM e TFT
### Comparação de arquiteturas simples e complexas aplicadas à ENTSO-E (França, Espanha e Portugal)
**Autor:** Victor M. Bertini  
**Orientador:** Prof. Fernando J. Von Zuben – FEEC / UNICAMP  

---

## 📌 Visão Geral

Este repositório contém todo o código, pré-processamento, experimentos e resultados utilizados no meu Trabalho de Conclusão de Curso (TCC), cujo objetivo foi **comparar modelos de diferentes complexidades** para previsão de carga elétrica horária utilizando dados reais da **ENTSO-E Transparency Platform**.

A pesquisa busca responder:

> **Modelos complexos (como LSTM e Transformers) realmente superam modelos simples (Linear / MLP) em cenários reais e com quantidade moderada de dados?**

---

## 📊 Dados

- **Fonte:** ENTSO-E Transparency Platform  
- **Métrica:** Carga Elétrica Total (Total Load)  
- **Países:** França (FR), Espanha (ES), Portugal (PT)  
- **Período:** 2021–2025  
- **Frequência:** Horária  
- **Total por país:** ~35.000 amostras  

Motivos da escolha:
- Dados reais e confiáveis  
- Alta resolução temporal  
- Baixa necessidade de limpeza  
- Cobertura longa e contínua  

---

## 🧪 Problemas de Previsão (N1, N2, N3)

Os experimentos foram organizados em três níveis progressivos:

### **N1 — Univariado (baseline por país)**
- Um modelo para cada país, olhando apenas seu histórico.
- Mede o desempenho básico da arquitetura.

### **N2 — Multipaís (aprendizado compartilhado)**
- Um único modelo aprende FR + ES + PT juntos.
- Avalia transferência de padrões entre países.

### **N3 — Robustez a Ruído**
- Ruído gaussiano leve adicionado às entradas.
- Testa estabilidade de cada modelo.

### **Variações A/B/C — diferentes lags**
Três janelas de observação para estudar o impacto do lookback.

---

## 🔧 Pré-Processamento

Todo o pipeline utiliza uma **classe única** responsável por:

### ✔ Organização e limpeza
- Ordenação temporal
- Tratamento mínimo de valores ausentes
- Manutenção da integridade das séries

### ✔ Codificação temporal (sen/cos)
- Hora, dia, mês → representações cíclicas  
- Mantém periodicidade natural da carga elétrica

### ✔ Normalização por país
- Z-score independente para FR, ES, PT  
- Estatísticas salvas em `.meta.json`

### ✔ Dados preparados conforme o modelo
- **Linear / MLP:** vetores flatten  
- **LSTM:** tensores 3D (sequências)  
- **TFT:** sequências + identificador do país  

---

## 🧠 Modelos Implementados

### **Regressão Linear**
- Modelo baseline  
- Excelente em curtíssimo prazo  
- Degrada rapidamente em horizontes longos  

### **MLP (Multilayer Perceptron)**
- 3 camadas densas com ReLU  
- Melhor equilíbrio entre simplicidade e desempenho  
- Estável ao longo do horizonte  

### **LSTM**
- 2 camadas recorrentes empilhadas  
- Forte em dependências de longo prazo  
- Requer mais dados e custo computacional maior  

### **TFT (Temporal Fusion Transformer)**
- Implementação via PyTorch Forecasting  
- Inclui atenção, gating e seleção de variáveis  
- Não convergiu adequadamente com o volume atual (~35k por país)  

---

## ⚙ Treinamento e Hiperparâmetros

### Divisão temporal
- Treino: Jan/2021 – Set/2024  
- Validação: Out–Dez/2024  
- Teste: Jan–Mar/2025  

### Ajuste de hiperparâmetros
- Feito com **Optuna**  
- Objetivo: **minimizar erro de validação**  
- Resultados registrados com **MLflow**

### Métricas avaliadas
- MAE  
- MSE  
- RMSE  
- R²  
- Correlação de Pearson  

---

## 📈 Resultados (Resumo)

### **Lineares**
- Melhores nas primeiras 24 horas
- Degradação rápida em leads longos

### **MLP**
- Inicialmente ligeiramente inferior ao linear  
- Muito mais estável ao longo do horizonte  
- Melhor desempenho médio geral  

### **LSTM**
- Fraco no curtíssimo prazo  
- Melhor em previsões longas  
- Adequado quando há dependências estendidas  

### **TFT**
- Não convergiu  
- Requer datasets muito maiores  

---

## 🏁 Conclusões

1. **Modelos simples são extremamente competitivos.**  
2. **A complexidade não garante melhor desempenho.**  
3. **MLP apresentou o melhor equilíbrio entre custo e performance.**  
4. **LSTMs valem a pena apenas em horizontes longos.**  
5. **Transformers (TFT) não são eficazes com datasets moderados.**  

### Recomendações práticas:
- **Curtíssimo prazo:** Linear  
- **Curto/médio prazo:** MLP  
- **Médio/longo prazo:** LSTM  

---

## 📂 Estrutura do Repositório

├── data/
│   ├── raw/                # Dados originais coletados da ENTSO-E
│   ├── processed/          # Dados após pré-processamento
│   └── treinamento/        # TFRecords e Parquets finais usados nos modelos
│
├── notebooks/
│   ├── Coleta_dados.ipynb
│   ├── Modelos_tensorflow.ipynb
│   └── Modelos_pytorch.ipynb
│
├── preprocessor.py         # Classe única de encoding, decoding e normalização
├── models/                 # Arquiteturas lineares, MLP, LSTM e TFT
├── results/                # Gráficos, métricas, validações e relatórios
├── utils/                  # Funções auxiliares
└── README.md


---

## 🔗 Links Úteis

- **Repositório:** https://github.com/vm-bertini/TCC-2025  
- **Resultados (gráficos):** disponível no Google Drive  
- **Paper do TFT:** https://arxiv.org/abs/1912.09363  

---

## 📜 Licença
MIT License – livre para uso acadêmico.

