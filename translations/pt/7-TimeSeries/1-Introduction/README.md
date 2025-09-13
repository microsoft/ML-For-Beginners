<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T08:39:49+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "pt"
}
-->
# Introdução à previsão de séries temporais

![Resumo de séries temporais em um sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

Nesta lição e na próxima, vais aprender um pouco sobre previsão de séries temporais, uma parte interessante e valiosa do repertório de um cientista de ML que é um pouco menos conhecida do que outros tópicos. A previsão de séries temporais é como uma espécie de "bola de cristal": com base no desempenho passado de uma variável, como o preço, podes prever o seu valor potencial futuro.

[![Introdução à previsão de séries temporais](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introdução à previsão de séries temporais")

> 🎥 Clica na imagem acima para ver um vídeo sobre previsão de séries temporais

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

É um campo útil e interessante com valor real para os negócios, dado a sua aplicação direta em problemas de preços, inventário e questões de cadeia de abastecimento. Embora técnicas de aprendizagem profunda tenham começado a ser usadas para obter mais insights e prever melhor o desempenho futuro, a previsão de séries temporais continua a ser um campo amplamente informado por técnicas clássicas de ML.

> O currículo útil de séries temporais da Penn State pode ser encontrado [aqui](https://online.stat.psu.edu/stat510/lesson/1)

## Introdução

Suponha que geres uma rede de parquímetros inteligentes que fornecem dados sobre a frequência e duração de uso ao longo do tempo.

> E se pudesses prever, com base no desempenho passado do parquímetro, o seu valor futuro de acordo com as leis de oferta e procura?

Prever com precisão quando agir para alcançar o teu objetivo é um desafio que pode ser abordado com previsão de séries temporais. Não seria agradável para as pessoas serem cobradas mais em horários de pico quando procuram um lugar para estacionar, mas seria uma forma eficaz de gerar receita para limpar as ruas!

Vamos explorar alguns dos tipos de algoritmos de séries temporais e começar um notebook para limpar e preparar alguns dados. Os dados que vais analisar foram retirados da competição de previsão GEFCom2014. Consistem em 3 anos de valores horários de carga elétrica e temperatura entre 2012 e 2014. Dado os padrões históricos de carga elétrica e temperatura, podes prever valores futuros de carga elétrica.

Neste exemplo, vais aprender a prever um passo temporal à frente, usando apenas dados históricos de carga. Antes de começar, no entanto, é útil entender o que está a acontecer nos bastidores.

## Algumas definições

Ao encontrar o termo "séries temporais", precisas de entender o seu uso em vários contextos diferentes.

🎓 **Séries temporais**

Em matemática, "uma série temporal é uma série de pontos de dados indexados (ou listados ou representados graficamente) em ordem temporal. Mais comumente, uma série temporal é uma sequência tomada em pontos sucessivos igualmente espaçados no tempo." Um exemplo de série temporal é o valor de fecho diário do [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). O uso de gráficos de séries temporais e modelagem estatística é frequentemente encontrado em processamento de sinais, previsão do tempo, previsão de terremotos e outros campos onde eventos ocorrem e pontos de dados podem ser representados ao longo do tempo.

🎓 **Análise de séries temporais**

A análise de séries temporais é a análise dos dados de séries temporais mencionados acima. Os dados de séries temporais podem assumir formas distintas, incluindo "séries temporais interrompidas", que detectam padrões na evolução de uma série temporal antes e depois de um evento interruptor. O tipo de análise necessário para a série temporal depende da natureza dos dados. Os dados de séries temporais podem assumir a forma de séries de números ou caracteres.

A análise a ser realizada utiliza uma variedade de métodos, incluindo domínio de frequência e domínio temporal, linear e não linear, entre outros. [Saiba mais](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) sobre as várias formas de analisar este tipo de dados.

🎓 **Previsão de séries temporais**

A previsão de séries temporais é o uso de um modelo para prever valores futuros com base em padrões exibidos por dados previamente recolhidos à medida que ocorreram no passado. Embora seja possível usar modelos de regressão para explorar dados de séries temporais, com índices temporais como variáveis x num gráfico, esses dados são melhor analisados usando tipos especiais de modelos.

Os dados de séries temporais são uma lista de observações ordenadas, diferente de dados que podem ser analisados por regressão linear. O mais comum é o ARIMA, um acrónimo que significa "Autoregressive Integrated Moving Average".

[Modelos ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relacionam o valor presente de uma série com valores passados e erros de previsão passados." Eles são mais apropriados para analisar dados no domínio temporal, onde os dados são ordenados ao longo do tempo.

> Existem vários tipos de modelos ARIMA, sobre os quais podes aprender [aqui](https://people.duke.edu/~rnau/411arim.htm) e que vais abordar na próxima lição.

Na próxima lição, vais construir um modelo ARIMA usando [Séries Temporais Univariadas](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), que se concentram numa variável que muda o seu valor ao longo do tempo. Um exemplo deste tipo de dados é [este conjunto de dados](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) que regista a concentração mensal de CO2 no Observatório Mauna Loa:

|   CO2   | YearMonth | Year  | Month |
| :-----: | :-------: | :---: | :---: |
| 330.62  |  1975.04  | 1975  |   1   |
| 331.40  |  1975.13  | 1975  |   2   |
| 331.87  |  1975.21  | 1975  |   3   |
| 333.18  |  1975.29  | 1975  |   4   |
| 333.92  |  1975.38  | 1975  |   5   |
| 333.43  |  1975.46  | 1975  |   6   |
| 331.85  |  1975.54  | 1975  |   7   |
| 330.01  |  1975.63  | 1975  |   8   |
| 328.51  |  1975.71  | 1975  |   9   |
| 328.41  |  1975.79  | 1975  |  10   |
| 329.25  |  1975.88  | 1975  |  11   |
| 330.97  |  1975.96  | 1975  |  12   |

✅ Identifica a variável que muda ao longo do tempo neste conjunto de dados.

## Características dos dados de séries temporais a considerar

Ao observar dados de séries temporais, podes notar que eles possuem [certas características](https://online.stat.psu.edu/stat510/lesson/1/1.1) que precisas levar em conta e mitigar para entender melhor os seus padrões. Se considerares os dados de séries temporais como potencialmente fornecendo um "sinal" que desejas analisar, essas características podem ser vistas como "ruído". Muitas vezes, será necessário reduzir esse "ruído" compensando algumas dessas características usando técnicas estatísticas.

Aqui estão alguns conceitos que deves conhecer para trabalhar com séries temporais:

🎓 **Tendências**

Tendências são definidas como aumentos e diminuições mensuráveis ao longo do tempo. [Lê mais](https://machinelearningmastery.com/time-series-trends-in-python). No contexto de séries temporais, trata-se de como usar e, se necessário, remover tendências da tua série temporal.

🎓 **[Sazonalidade](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sazonalidade é definida como flutuações periódicas, como picos de vendas durante as férias, por exemplo. [Dá uma olhada](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) em como diferentes tipos de gráficos exibem sazonalidade nos dados.

🎓 **Outliers**

Outliers são valores que estão muito distantes da variância padrão dos dados.

🎓 **Ciclo de longo prazo**

Independentemente da sazonalidade, os dados podem exibir um ciclo de longo prazo, como uma recessão económica que dura mais de um ano.

🎓 **Variância constante**

Ao longo do tempo, alguns dados exibem flutuações constantes, como o uso de energia durante o dia e a noite.

🎓 **Mudanças abruptas**

Os dados podem exibir uma mudança abrupta que pode precisar de análise adicional. O encerramento repentino de negócios devido à COVID, por exemplo, causou mudanças nos dados.

✅ Aqui está um [exemplo de gráfico de séries temporais](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) mostrando o gasto diário em moeda dentro de um jogo ao longo de alguns anos. Consegues identificar alguma das características listadas acima nesses dados?

![Gasto em moeda no jogo](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Exercício - começando com dados de uso de energia

Vamos começar a criar um modelo de séries temporais para prever o uso futuro de energia com base no uso passado.

> Os dados neste exemplo foram retirados da competição de previsão GEFCom2014. Consistem em 3 anos de valores horários de carga elétrica e temperatura entre 2012 e 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli e Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, julho-setembro, 2016.

1. Na pasta `working` desta lição, abre o ficheiro _notebook.ipynb_. Começa por adicionar bibliotecas que te ajudarão a carregar e visualizar os dados.

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Nota que estás a usar os ficheiros da pasta `common` incluída, que configuram o teu ambiente e tratam do download dos dados.

2. Em seguida, examina os dados como um dataframe chamando `load_data()` e `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Podes ver que há duas colunas representando data e carga:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Agora, representa os dados graficamente chamando `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![gráfico de energia](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Agora, representa graficamente a primeira semana de julho de 2014, fornecendo-a como entrada para o `energy` no padrão `[data inicial]:[data final]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![julho](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Um gráfico bonito! Dá uma olhada nesses gráficos e vê se consegues determinar alguma das características listadas acima. O que podemos concluir ao visualizar os dados?

Na próxima lição, vais criar um modelo ARIMA para fazer algumas previsões.

---

## 🚀Desafio

Faz uma lista de todas as indústrias e áreas de estudo que consegues pensar que poderiam beneficiar da previsão de séries temporais. Consegues pensar numa aplicação dessas técnicas nas artes? Em Econometria? Ecologia? Retalho? Indústria? Finanças? Onde mais?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Embora não os abordemos aqui, redes neurais são por vezes usadas para melhorar os métodos clássicos de previsão de séries temporais. Lê mais sobre isso [neste artigo](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Tarefa

[Visualiza mais séries temporais](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante notar que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.