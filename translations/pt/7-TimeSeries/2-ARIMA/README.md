<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T08:39:06+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "pt"
}
-->
# Previsão de séries temporais com ARIMA

Na lição anterior, aprendeste um pouco sobre previsão de séries temporais e carregaste um conjunto de dados que mostra as flutuações da carga elétrica ao longo de um período de tempo.

[![Introdução ao ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introdução ao ARIMA")

> 🎥 Clica na imagem acima para um vídeo: Uma breve introdução aos modelos ARIMA. O exemplo é feito em R, mas os conceitos são universais.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Introdução

Nesta lição, vais descobrir uma forma específica de construir modelos com [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Os modelos ARIMA são particularmente adequados para ajustar dados que apresentam [não-estacionaridade](https://wikipedia.org/wiki/Stationary_process).

## Conceitos gerais

Para trabalhar com ARIMA, há alguns conceitos que precisas de conhecer:

- 🎓 **Estacionaridade**. No contexto estatístico, estacionaridade refere-se a dados cuja distribuição não muda ao longo do tempo. Dados não estacionários, por outro lado, apresentam flutuações devido a tendências que precisam ser transformadas para serem analisadas. A sazonalidade, por exemplo, pode introduzir flutuações nos dados e pode ser eliminada através de um processo de 'diferença sazonal'.

- 🎓 **[Diferença](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Diferençar os dados, novamente no contexto estatístico, refere-se ao processo de transformar dados não estacionários para torná-los estacionários, removendo sua tendência não constante. "A diferença remove as mudanças no nível de uma série temporal, eliminando tendência e sazonalidade e, consequentemente, estabilizando a média da série temporal." [Artigo de Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA no contexto de séries temporais

Vamos explorar as partes do ARIMA para entender melhor como ele nos ajuda a modelar séries temporais e a fazer previsões.

- **AR - de AutoRegressivo**. Modelos autorregressivos, como o nome sugere, olham 'para trás' no tempo para analisar valores anteriores nos teus dados e fazer suposições sobre eles. Esses valores anteriores são chamados de 'lags'. Um exemplo seria dados que mostram vendas mensais de lápis. O total de vendas de cada mês seria considerado uma 'variável em evolução' no conjunto de dados. Este modelo é construído como "a variável de interesse em evolução é regredida em seus próprios valores defasados (ou seja, valores anteriores)." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - de Integrado**. Ao contrário dos modelos semelhantes 'ARMA', o 'I' em ARIMA refere-se ao seu aspeto *[integrado](https://wikipedia.org/wiki/Order_of_integration)*. Os dados são 'integrados' quando passos de diferença são aplicados para eliminar a não-estacionaridade.

- **MA - de Média Móvel**. O aspeto de [média móvel](https://wikipedia.org/wiki/Moving-average_model) deste modelo refere-se à variável de saída que é determinada observando os valores atuais e passados dos lags.

Resumindo: o ARIMA é usado para ajustar um modelo o mais próximo possível da forma especial dos dados de séries temporais.

## Exercício - construir um modelo ARIMA

Abre a pasta [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) nesta lição e encontra o ficheiro [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Executa o notebook para carregar a biblioteca Python `statsmodels`; vais precisar dela para os modelos ARIMA.

1. Carrega as bibliotecas necessárias.

1. Agora, carrega mais algumas bibliotecas úteis para a plotagem de dados:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. Carrega os dados do ficheiro `/data/energy.csv` para um dataframe do Pandas e dá uma olhada:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Plota todos os dados de energia disponíveis de janeiro de 2012 a dezembro de 2014. Não deverá haver surpresas, pois já vimos esses dados na última lição:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Agora, vamos construir um modelo!

### Criar conjuntos de treino e teste

Agora que os teus dados estão carregados, podes separá-los em conjuntos de treino e teste. Vais treinar o teu modelo no conjunto de treino. Como de costume, após o modelo ter terminado o treino, vais avaliar a sua precisão usando o conjunto de teste. É necessário garantir que o conjunto de teste cobre um período posterior ao conjunto de treino para garantir que o modelo não obtenha informações de períodos futuros.

1. Aloca um período de dois meses, de 1 de setembro a 31 de outubro de 2014, para o conjunto de treino. O conjunto de teste incluirá o período de dois meses de 1 de novembro a 31 de dezembro de 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Como estes dados refletem o consumo diário de energia, há um forte padrão sazonal, mas o consumo é mais semelhante ao consumo de dias mais recentes.

1. Visualiza as diferenças:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![dados de treino e teste](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Portanto, usar uma janela de tempo relativamente pequena para treinar os dados deve ser suficiente.

    > Nota: Como a função que usamos para ajustar o modelo ARIMA utiliza validação in-sample durante o ajuste, omitiremos os dados de validação.

### Preparar os dados para treino

Agora, precisas de preparar os dados para o treino, realizando filtragem e escalonamento dos dados. Filtra o teu conjunto de dados para incluir apenas os períodos de tempo e colunas necessários, e escala os dados para garantir que estejam no intervalo 0,1.

1. Filtra o conjunto de dados original para incluir apenas os períodos de tempo mencionados por conjunto e apenas a coluna necessária 'load', além da data:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Podes ver a forma dos dados:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Escala os dados para estarem no intervalo (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualiza os dados originais vs. os dados escalados:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Os dados originais

    ![escalado](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Os dados escalados

1. Agora que calibraste os dados escalados, podes escalar os dados de teste:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementar ARIMA

É hora de implementar o ARIMA! Agora vais usar a biblioteca `statsmodels` que instalaste anteriormente.

Agora precisas de seguir vários passos:

   1. Define o modelo chamando `SARIMAX()` e passando os parâmetros do modelo: parâmetros p, d e q, e parâmetros P, D e Q.
   2. Prepara o modelo para os dados de treino chamando a função `fit()`.
   3. Faz previsões chamando a função `forecast()` e especificando o número de passos (o `horizon`) a prever.

> 🎓 Para que servem todos esses parâmetros? Num modelo ARIMA, há 3 parâmetros usados para ajudar a modelar os principais aspetos de uma série temporal: sazonalidade, tendência e ruído. Esses parâmetros são:

`p`: o parâmetro associado ao aspeto autorregressivo do modelo, que incorpora valores *passados*.  
`d`: o parâmetro associado à parte integrada do modelo, que afeta a quantidade de *diferença* (🎓 lembra-te da diferença 👆?) a aplicar a uma série temporal.  
`q`: o parâmetro associado à parte de média móvel do modelo.  

> Nota: Se os teus dados tiverem um aspeto sazonal - como este tem -, usamos um modelo ARIMA sazonal (SARIMA). Nesse caso, precisas de usar outro conjunto de parâmetros: `P`, `D` e `Q`, que descrevem as mesmas associações que `p`, `d` e `q`, mas correspondem aos componentes sazonais do modelo.

1. Começa por definir o teu valor de horizonte preferido. Vamos tentar 3 horas:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Selecionar os melhores valores para os parâmetros de um modelo ARIMA pode ser desafiador, pois é algo subjetivo e demorado. Podes considerar usar uma função `auto_arima()` da [biblioteca `pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Por agora, tenta algumas seleções manuais para encontrar um bom modelo.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Uma tabela de resultados é exibida.

Construíste o teu primeiro modelo! Agora precisamos de encontrar uma forma de avaliá-lo.

### Avaliar o teu modelo

Para avaliar o teu modelo, podes realizar a chamada validação `walk forward`. Na prática, os modelos de séries temporais são re-treinados sempre que novos dados ficam disponíveis. Isso permite que o modelo faça a melhor previsão em cada passo de tempo.

Começando no início da série temporal, usando esta técnica, treina o modelo no conjunto de treino. Depois, faz uma previsão para o próximo passo de tempo. A previsão é avaliada em relação ao valor conhecido. O conjunto de treino é então expandido para incluir o valor conhecido e o processo é repetido.

> Nota: Deves manter a janela do conjunto de treino fixa para um treino mais eficiente, de modo que, sempre que adicionares uma nova observação ao conjunto de treino, removes a observação do início do conjunto.

Este processo fornece uma estimativa mais robusta de como o modelo irá desempenhar-se na prática. No entanto, tem o custo computacional de criar tantos modelos. Isso é aceitável se os dados forem pequenos ou se o modelo for simples, mas pode ser um problema em escala.

A validação walk-forward é o padrão ouro para avaliação de modelos de séries temporais e é recomendada para os teus próprios projetos.

1. Primeiro, cria um ponto de dados de teste para cada passo do HORIZON.

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    Os dados são deslocados horizontalmente de acordo com o seu ponto de horizonte.

1. Faz previsões nos teus dados de teste usando esta abordagem de janela deslizante num loop do tamanho do comprimento dos dados de teste:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    Podes observar o treino a ocorrer:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Compara as previsões com a carga real:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Saída  
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Observa a previsão dos dados horários em comparação com a carga real. Quão precisa é esta previsão?

### Verificar a precisão do modelo

Verifica a precisão do teu modelo testando o seu erro percentual absoluto médio (MAPE) em todas as previsões.
> **🧮 Mostra-me os cálculos**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) é utilizado para mostrar a precisão das previsões como uma razão definida pela fórmula acima. A diferença entre o valor real e o previsto é dividida pelo valor real. "O valor absoluto deste cálculo é somado para cada ponto previsto no tempo e dividido pelo número de pontos ajustados n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Exprimir a equação em código:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Calcular o MAPE de um passo:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE da previsão de um passo:  0.5570581332313952 %

1. Imprimir o MAPE da previsão de múltiplos passos:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Um número baixo é o ideal: considere que uma previsão com um MAPE de 10 está errada em 10%.

1. Mas, como sempre, é mais fácil visualizar este tipo de medição de precisão, então vamos representá-lo graficamente:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![um modelo de séries temporais](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Um gráfico muito bom, mostrando um modelo com boa precisão. Excelente trabalho!

---

## 🚀Desafio

Explore as formas de testar a precisão de um modelo de séries temporais. Abordamos o MAPE nesta lição, mas existem outros métodos que poderia usar? Pesquise sobre eles e anote-os. Um documento útil pode ser encontrado [aqui](https://otexts.com/fpp2/accuracy.html)

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Individual

Esta lição aborda apenas os fundamentos da previsão de séries temporais com ARIMA. Dedique algum tempo para aprofundar o seu conhecimento explorando [este repositório](https://microsoft.github.io/forecasting/) e os seus vários tipos de modelos para aprender outras formas de construir modelos de séries temporais.

## Tarefa

[Um novo modelo ARIMA](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.