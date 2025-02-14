# Introdução à previsão de séries temporais

![Resumo de séries temporais em um sketchnote](../../../../translated_images/ml-timeseries.fb98d25f1013fc0c59090030080b5d1911ff336427bec31dbaf1ad08193812e9.pt.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

Nesta lição e na próxima, você aprenderá um pouco sobre a previsão de séries temporais, uma parte interessante e valiosa do repertório de um cientista de ML que é um pouco menos conhecida do que outros tópicos. A previsão de séries temporais é uma espécie de 'bola de cristal': com base no desempenho passado de uma variável, como o preço, você pode prever seu potencial valor futuro.

[![Introdução à previsão de séries temporais](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introdução à previsão de séries temporais")

> 🎥 Clique na imagem acima para assistir a um vídeo sobre previsão de séries temporais

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/41/)

É um campo útil e interessante com valor real para os negócios, dada sua aplicação direta a problemas de precificação, inventário e questões da cadeia de suprimentos. Embora técnicas de aprendizado profundo tenham começado a ser usadas para obter mais insights e prever melhor o desempenho futuro, a previsão de séries temporais continua sendo um campo amplamente informado por técnicas clássicas de ML.

> O currículo útil de séries temporais da Penn State pode ser encontrado [aqui](https://online.stat.psu.edu/stat510/lesson/1)

## Introdução

Suponha que você mantenha um conjunto de parquímetros inteligentes que fornecem dados sobre com que frequência são usados e por quanto tempo ao longo do tempo.

> E se você pudesse prever, com base no desempenho passado do parquímetro, seu valor futuro de acordo com as leis de oferta e demanda?

Prever com precisão quando agir para alcançar seu objetivo é um desafio que pode ser enfrentado pela previsão de séries temporais. Não seria agradável para as pessoas serem cobradas mais em horários de pico quando estão procurando uma vaga de estacionamento, mas seria uma maneira segura de gerar receita para limpar as ruas!

Vamos explorar alguns dos tipos de algoritmos de séries temporais e começar um notebook para limpar e preparar alguns dados. Os dados que você analisará são provenientes da competição de previsão GEFCom2014. Eles consistem em 3 anos de valores de carga elétrica e temperatura horária entre 2012 e 2014. Dado os padrões históricos de carga elétrica e temperatura, você pode prever os valores futuros da carga elétrica.

Neste exemplo, você aprenderá como prever um passo de tempo à frente, usando apenas dados históricos de carga. Antes de começar, no entanto, é útil entender o que está acontecendo nos bastidores.

## Algumas definições

Ao encontrar o termo 'série temporal', você precisa entender seu uso em vários contextos diferentes.

🎓 **Série temporal**

Na matemática, "uma série temporal é uma série de pontos de dados indexados (ou listados ou grafados) em ordem temporal. Mais comumente, uma série temporal é uma sequência tomada em pontos sucessivos igualmente espaçados no tempo." Um exemplo de uma série temporal é o valor de fechamento diário do [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). O uso de gráficos de séries temporais e modelagem estatística é frequentemente encontrado em processamento de sinais, previsão do tempo, previsão de terremotos e outros campos onde eventos ocorrem e pontos de dados podem ser plotados ao longo do tempo.

🎓 **Análise de séries temporais**

A análise de séries temporais é a análise dos dados de séries temporais mencionados acima. Os dados de séries temporais podem assumir formas distintas, incluindo 'séries temporais interrompidas', que detectam padrões na evolução de uma série temporal antes e depois de um evento interruptivo. O tipo de análise necessária para a série temporal depende da natureza dos dados. Os dados de séries temporais em si podem assumir a forma de séries de números ou caracteres.

A análise a ser realizada utiliza uma variedade de métodos, incluindo domínio de frequência e domínio do tempo, linear e não linear, e mais. [Saiba mais](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) sobre as muitas maneiras de analisar esse tipo de dado.

🎓 **Previsão de séries temporais**

A previsão de séries temporais é o uso de um modelo para prever valores futuros com base em padrões exibidos por dados coletados anteriormente, conforme ocorreram no passado. Embora seja possível usar modelos de regressão para explorar dados de séries temporais, com índices de tempo como variáveis x em um gráfico, tais dados são melhor analisados usando tipos especiais de modelos.

Os dados de séries temporais são uma lista de observações ordenadas, ao contrário de dados que podem ser analisados por regressão linear. O mais comum é o ARIMA, um acrônimo que significa "Média Móvel Integrada Autoregressiva".

[Modelos ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relacionam o valor presente de uma série a valores passados e erros de previsão passados." Eles são mais apropriados para analisar dados no domínio do tempo, onde os dados estão ordenados ao longo do tempo.

> Existem vários tipos de modelos ARIMA, sobre os quais você pode aprender [aqui](https://people.duke.edu/~rnau/411arim.htm) e que você tocará na próxima lição.

Na próxima lição, você construirá um modelo ARIMA usando [Séries Temporais Univariadas](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), que se concentra em uma variável que muda seu valor ao longo do tempo. Um exemplo desse tipo de dado é [este conjunto de dados](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) que registra a concentração mensal de CO2 no Observatório de Mauna Loa:

|  CO2   | AnoMês | Ano  | Mês |
| :----: | :----: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Identifique a variável que muda ao longo do tempo neste conjunto de dados.

## Características dos dados de séries temporais a considerar

Ao olhar para dados de séries temporais, você pode notar que eles têm [certas características](https://online.stat.psu.edu/stat510/lesson/1/1.1) que você precisa levar em conta e mitigar para entender melhor seus padrões. Se você considerar os dados de séries temporais como potencialmente fornecendo um 'sinal' que deseja analisar, essas características podem ser pensadas como 'ruído'. Você muitas vezes precisará reduzir esse 'ruído' compensando algumas dessas características usando algumas técnicas estatísticas.

Aqui estão alguns conceitos que você deve conhecer para poder trabalhar com séries temporais:

🎓 **Tendências**

Tendências são definidas como aumentos e diminuições mensuráveis ao longo do tempo. [Leia mais](https://machinelearningmastery.com/time-series-trends-in-python). No contexto de séries temporais, trata-se de como usar e, se necessário, remover tendências de sua série temporal.

🎓 **[Sazonalidade](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sazonalidade é definida como flutuações periódicas, como corridas de férias que podem afetar as vendas, por exemplo. [Dê uma olhada](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) em como diferentes tipos de gráficos exibem sazonalidade nos dados.

🎓 **Outliers**

Outliers são valores que estão muito distantes da variância padrão dos dados.

🎓 **Ciclo de longo prazo**

Independente da sazonalidade, os dados podem exibir um ciclo de longo prazo, como uma recessão econômica que dura mais de um ano.

🎓 **Variância constante**

Ao longo do tempo, alguns dados exibem flutuações constantes, como o uso de energia por dia e noite.

🎓 **Mudanças abruptas**

Os dados podem exibir uma mudança abrupta que pode precisar de uma análise mais aprofundada. O fechamento abrupto de empresas devido à COVID, por exemplo, causou mudanças nos dados.

✅ Aqui está um [gráfico de séries temporais de exemplo](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) mostrando a moeda do jogo gasta diariamente ao longo de alguns anos. Você consegue identificar alguma das características listadas acima nesses dados?

![Gastos em moeda do jogo](../../../../translated_images/currency.e7429812bfc8c6087b2d4c410faaa4aaa11b2fcaabf6f09549b8249c9fbdb641.pt.png)

## Exercício - começando com dados de uso de energia

Vamos começar a criar um modelo de séries temporais para prever o uso futuro de energia, dado o uso passado.

> Os dados neste exemplo são provenientes da competição de previsão GEFCom2014. Eles consistem em 3 anos de valores de carga elétrica e temperatura horária entre 2012 e 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli e Rob J. Hyndman, "Previsão de energia probabilística: Competição Global de Previsão de Energia 2014 e além", International Journal of Forecasting, vol.32, no.3, pp 896-913, julho-setembro, 2016.

1. Na pasta `working` desta lição, abra o arquivo _notebook.ipynb_. Comece adicionando bibliotecas que ajudarão você a carregar e visualizar os dados.

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Observe que você está usando os arquivos da função incluída `common` folder which set up your environment and handle downloading the data.

2. Next, examine the data as a dataframe calling `load_data()` and `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Você pode ver que há duas colunas representando a data e a carga:

    |                     |  carga  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Agora, plote os dados chamando `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![gráfico de energia](../../../../translated_images/energy-plot.5fdac3f397a910bc6070602e9e45bea8860d4c239354813fa8fc3c9d556f5bad.pt.png)

4. Agora, plote a primeira semana de julho de 2014, fornecendo-a como entrada para o padrão `energia` in `[de data]: [até data]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![julho](../../../../translated_images/july-2014.9e1f7c318ec6d5b30b0d7e1e20be3643501f64a53f3d426d7c7d7b62addb335e.pt.png)

    Um gráfico lindo! Dê uma olhada nesses gráficos e veja se consegue determinar alguma das características listadas acima. O que podemos inferir ao visualizar os dados?

Na próxima lição, você criará um modelo ARIMA para gerar algumas previsões.

---

## 🚀Desafio

Faça uma lista de todas as indústrias e áreas de pesquisa que você consegue pensar que se beneficiariam da previsão de séries temporais. Você consegue pensar em uma aplicação dessas técnicas nas artes? Em Econometria? Ecologia? Varejo? Indústria? Finanças? Onde mais?

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/42/)

## Revisão e Estudo Pessoal

Embora não os abordemos aqui, redes neurais são às vezes usadas para aprimorar métodos clássicos de previsão de séries temporais. Leia mais sobre elas [neste artigo](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Tarefa

[Visualize mais séries temporais](assignment.md)

**Isenção de responsabilidade**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.