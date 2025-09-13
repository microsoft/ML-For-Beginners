<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T08:40:23+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "pt"
}
-->
# Previsão de Séries Temporais com Support Vector Regressor

Na lição anterior, aprendeste a usar o modelo ARIMA para fazer previsões de séries temporais. Agora vais explorar o modelo Support Vector Regressor, que é um modelo de regressão utilizado para prever dados contínuos.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/) 

## Introdução

Nesta lição, vais descobrir uma forma específica de construir modelos com [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) para regressão, ou **SVR: Support Vector Regressor**. 

### SVR no contexto de séries temporais [^1]

Antes de compreender a importância do SVR na previsão de séries temporais, aqui estão alguns conceitos importantes que precisas de saber:

- **Regressão:** Técnica de aprendizagem supervisionada para prever valores contínuos a partir de um conjunto de entradas. A ideia é ajustar uma curva (ou linha) no espaço de características que tenha o maior número de pontos de dados. [Clica aqui](https://en.wikipedia.org/wiki/Regression_analysis) para mais informações.
- **Support Vector Machine (SVM):** Um tipo de modelo de aprendizagem supervisionada usado para classificação, regressão e deteção de outliers. O modelo é um hiperplano no espaço de características, que no caso de classificação atua como uma fronteira, e no caso de regressão atua como a linha de melhor ajuste. No SVM, uma função Kernel é geralmente usada para transformar o conjunto de dados para um espaço com maior número de dimensões, de forma a torná-los mais facilmente separáveis. [Clica aqui](https://en.wikipedia.org/wiki/Support-vector_machine) para mais informações sobre SVMs.
- **Support Vector Regressor (SVR):** Um tipo de SVM, que encontra a linha de melhor ajuste (que no caso de SVM é um hiperplano) com o maior número de pontos de dados.

### Porquê SVR? [^1]

Na última lição aprendeste sobre o ARIMA, que é um método estatístico linear muito bem-sucedido para prever dados de séries temporais. No entanto, em muitos casos, os dados de séries temporais apresentam *não-linearidade*, que não pode ser mapeada por modelos lineares. Nestes casos, a capacidade do SVM de considerar a não-linearidade nos dados para tarefas de regressão torna o SVR bem-sucedido na previsão de séries temporais.

## Exercício - construir um modelo SVR

Os primeiros passos para a preparação dos dados são os mesmos da lição anterior sobre [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Abre a pasta [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) nesta lição e encontra o ficheiro [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Executa o notebook e importa as bibliotecas necessárias: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. Carrega os dados do ficheiro `/data/energy.csv` para um dataframe do Pandas e analisa-os: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Faz o gráfico de todos os dados de energia disponíveis de janeiro de 2012 a dezembro de 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![dados completos](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Agora, vamos construir o nosso modelo SVR.

### Criar conjuntos de treino e teste

Agora que os dados estão carregados, podes separá-los em conjuntos de treino e teste. Depois vais remodelar os dados para criar um conjunto de dados baseado em passos temporais, que será necessário para o SVR. Vais treinar o teu modelo no conjunto de treino. Após o modelo terminar o treino, vais avaliar a sua precisão no conjunto de treino, no conjunto de teste e depois no conjunto de dados completo para ver o desempenho geral. É importante garantir que o conjunto de teste cobre um período posterior ao conjunto de treino para assegurar que o modelo não obtém informações de períodos futuros [^2] (uma situação conhecida como *Overfitting*).

1. Aloca um período de dois meses de 1 de setembro a 31 de outubro de 2014 para o conjunto de treino. O conjunto de teste incluirá o período de dois meses de 1 de novembro a 31 de dezembro de 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualiza as diferenças: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![dados de treino e teste](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Preparar os dados para treino

Agora, precisas de preparar os dados para treino, realizando filtragem e escalonamento dos dados. Filtra o conjunto de dados para incluir apenas os períodos de tempo e colunas necessários, e faz o escalonamento para garantir que os dados são projetados no intervalo 0,1.

1. Filtra o conjunto de dados original para incluir apenas os períodos de tempo mencionados por conjunto e apenas a coluna necessária 'load' mais a data: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. Escalona os dados de treino para estarem no intervalo (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Agora, escalona os dados de teste: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Criar dados com passos temporais [^1]

Para o SVR, transformas os dados de entrada para a forma `[batch, timesteps]`. Assim, remodelas os `train_data` e `test_data` existentes de forma a que haja uma nova dimensão que se refere aos passos temporais. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Para este exemplo, usamos `timesteps = 5`. Assim, as entradas para o modelo são os dados dos primeiros 4 passos temporais, e a saída será os dados do 5º passo temporal.

```python
timesteps=5
```

Converter os dados de treino para tensor 2D usando list comprehension aninhada:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Converter os dados de teste para tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Selecionar entradas e saídas dos dados de treino e teste:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### Implementar SVR [^1]

Agora, é hora de implementar o SVR. Para saber mais sobre esta implementação, podes consultar [esta documentação](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Para a nossa implementação, seguimos estes passos:

  1. Define o modelo chamando `SVR()` e passando os hiperparâmetros do modelo: kernel, gamma, c e epsilon
  2. Prepara o modelo para os dados de treino chamando a função `fit()`
  3. Faz previsões chamando a função `predict()`

Agora criamos um modelo SVR. Aqui usamos o [kernel RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), e definimos os hiperparâmetros gamma, C e epsilon como 0.5, 10 e 0.05 respetivamente.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Ajustar o modelo aos dados de treino [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Fazer previsões com o modelo [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Construíste o teu SVR! Agora precisamos de avaliá-lo.

### Avaliar o modelo [^1]

Para avaliação, primeiro vamos escalonar os dados de volta para a escala original. Depois, para verificar o desempenho, vamos fazer o gráfico da série temporal original e prevista, e também imprimir o resultado do MAPE.

Escalona os dados previstos e originais:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### Verificar o desempenho do modelo nos dados de treino e teste [^1]

Extraímos os timestamps do conjunto de dados para mostrar no eixo x do nosso gráfico. Nota que estamos a usar os primeiros ```timesteps-1``` valores como entrada para a primeira saída, então os timestamps para a saída começarão depois disso.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Faz o gráfico das previsões para os dados de treino:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![previsão dos dados de treino](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Imprime o MAPE para os dados de treino

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Faz o gráfico das previsões para os dados de teste

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![previsão dos dados de teste](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Imprime o MAPE para os dados de teste

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Obtiveste um resultado muito bom no conjunto de dados de teste!

### Verificar o desempenho do modelo no conjunto de dados completo [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![previsão dos dados completos](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Gráficos muito bons, mostrando um modelo com boa precisão. Excelente trabalho!

---

## 🚀Desafio

- Tenta ajustar os hiperparâmetros (gamma, C, epsilon) ao criar o modelo e avalia os dados para ver qual conjunto de hiperparâmetros dá os melhores resultados nos dados de teste. Para saber mais sobre estes hiperparâmetros, podes consultar o documento [aqui](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Experimenta usar diferentes funções kernel para o modelo e analisa os seus desempenhos no conjunto de dados. Um documento útil pode ser encontrado [aqui](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Tenta usar diferentes valores para `timesteps` para o modelo olhar para trás e fazer previsões.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Individual

Esta lição foi para introduzir a aplicação de SVR na previsão de séries temporais. Para saber mais sobre SVR, podes consultar [este blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Esta [documentação sobre scikit-learn](https://scikit-learn.org/stable/modules/svm.html) fornece uma explicação mais abrangente sobre SVMs em geral, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) e também outros detalhes de implementação, como as diferentes [funções kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) que podem ser usadas e os seus parâmetros.

## Tarefa

[Um novo modelo SVR](assignment.md)

## Créditos

[^1]: O texto, código e saída nesta seção foram contribuídos por [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: O texto, código e saída nesta seção foram retirados de [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.