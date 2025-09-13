<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T08:38:38+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "pt"
}
-->
# Construir um modelo de regressão usando Scikit-learn: preparar e visualizar dados

![Infográfico de visualização de dados](../../../../2-Regression/2-Data/images/data-visualization.png)

Infográfico por [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introdução

Agora que já está equipado com as ferramentas necessárias para começar a construir modelos de machine learning com Scikit-learn, está pronto para começar a fazer perguntas aos seus dados. Ao trabalhar com dados e aplicar soluções de ML, é muito importante saber como formular a pergunta certa para desbloquear adequadamente o potencial do seu conjunto de dados.

Nesta lição, irá aprender:

- Como preparar os seus dados para a construção de modelos.
- Como usar o Matplotlib para visualização de dados.

## Fazer a pergunta certa aos seus dados

A pergunta que precisa de responder determinará o tipo de algoritmos de ML que irá utilizar. E a qualidade da resposta que obtém dependerá muito da natureza dos seus dados.

Veja os [dados](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) fornecidos para esta lição. Pode abrir este ficheiro .csv no VS Code. Uma rápida análise mostra imediatamente que há espaços em branco e uma mistura de dados em formato texto e numérico. Há também uma coluna estranha chamada 'Package', onde os dados são uma mistura entre 'sacks', 'bins' e outros valores. Os dados, na verdade, estão um pouco desorganizados.

[![ML para iniciantes - Como analisar e limpar um conjunto de dados](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML para iniciantes - Como analisar e limpar um conjunto de dados")

> 🎥 Clique na imagem acima para assistir a um vídeo curto sobre como preparar os dados para esta lição.

De facto, não é muito comum receber um conjunto de dados completamente pronto para criar um modelo de ML diretamente. Nesta lição, irá aprender como preparar um conjunto de dados bruto usando bibliotecas padrão do Python. Também aprenderá várias técnicas para visualizar os dados.

## Estudo de caso: 'o mercado de abóboras'

Nesta pasta encontrará um ficheiro .csv na pasta raiz `data` chamado [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), que inclui 1757 linhas de dados sobre o mercado de abóboras, organizados por cidade. Estes são dados brutos extraídos dos [Relatórios Padrão dos Mercados de Culturas Especiais](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuídos pelo Departamento de Agricultura dos Estados Unidos.

### Preparar os dados

Estes dados estão em domínio público. Podem ser descarregados em vários ficheiros separados, por cidade, no site do USDA. Para evitar muitos ficheiros separados, concatenámos todos os dados das cidades num único ficheiro, portanto já _preparámos_ os dados um pouco. Agora, vamos analisar os dados mais de perto.

### Os dados das abóboras - primeiras conclusões

O que nota sobre estes dados? Já viu que há uma mistura de texto, números, espaços em branco e valores estranhos que precisa de interpretar.

Que pergunta pode fazer a estes dados, usando uma técnica de regressão? Que tal "Prever o preço de uma abóbora à venda durante um determinado mês"? Olhando novamente para os dados, há algumas alterações que precisa de fazer para criar a estrutura de dados necessária para esta tarefa.

## Exercício - analisar os dados das abóboras

Vamos usar o [Pandas](https://pandas.pydata.org/) (o nome significa `Python Data Analysis`), uma ferramenta muito útil para moldar dados, para analisar e preparar estes dados das abóboras.

### Primeiro, verificar datas em falta

Primeiro, precisará de tomar medidas para verificar se há datas em falta:

1. Converter as datas para um formato de mês (estas são datas dos EUA, então o formato é `MM/DD/YYYY`).
2. Extrair o mês para uma nova coluna.

Abra o ficheiro _notebook.ipynb_ no Visual Studio Code e importe a folha de cálculo para um novo dataframe do Pandas.

1. Use a função `head()` para visualizar as primeiras cinco linhas.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Que função usaria para visualizar as últimas cinco linhas?

1. Verifique se há dados em falta no dataframe atual:

    ```python
    pumpkins.isnull().sum()
    ```

    Há dados em falta, mas talvez isso não seja relevante para a tarefa em questão.

1. Para tornar o seu dataframe mais fácil de trabalhar, selecione apenas as colunas necessárias, usando a função `loc`, que extrai do dataframe original um grupo de linhas (passado como primeiro parâmetro) e colunas (passado como segundo parâmetro). A expressão `:` no caso abaixo significa "todas as linhas".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Segundo, determinar o preço médio da abóbora

Pense em como determinar o preço médio de uma abóbora num determinado mês. Que colunas escolheria para esta tarefa? Dica: precisará de 3 colunas.

Solução: calcule a média das colunas `Low Price` e `High Price` para preencher a nova coluna Price e converta a coluna Date para mostrar apenas o mês. Felizmente, de acordo com a verificação acima, não há dados em falta para datas ou preços.

1. Para calcular a média, adicione o seguinte código:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Sinta-se à vontade para imprimir quaisquer dados que queira verificar usando `print(month)`.

2. Agora, copie os seus dados convertidos para um novo dataframe do Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Ao imprimir o seu dataframe, verá um conjunto de dados limpo e organizado, no qual pode construir o seu novo modelo de regressão.

### Mas espere! Há algo estranho aqui

Se olhar para a coluna `Package`, verá que as abóboras são vendidas em muitas configurações diferentes. Algumas são vendidas em medidas de '1 1/9 bushel', outras em '1/2 bushel', algumas por abóbora, outras por peso, e algumas em grandes caixas com larguras variadas.

> Parece que as abóboras são muito difíceis de pesar de forma consistente

Ao analisar os dados originais, é interessante notar que qualquer coisa com `Unit of Sale` igual a 'EACH' ou 'PER BIN' também tem o tipo `Package` por polegada, por bin ou 'each'. Parece que as abóboras são muito difíceis de pesar de forma consistente, então vamos filtrá-las selecionando apenas abóboras com a string 'bushel' na coluna `Package`.

1. Adicione um filtro no topo do ficheiro, sob a importação inicial do .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Se imprimir os dados agora, verá que está a obter apenas cerca de 415 linhas de dados contendo abóboras por bushel.

### Mas espere! Há mais uma coisa a fazer

Notou que a quantidade de bushel varia por linha? Precisa de normalizar os preços para mostrar o preço por bushel, então faça alguns cálculos para padronizá-lo.

1. Adicione estas linhas após o bloco que cria o dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ De acordo com [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), o peso de um bushel depende do tipo de produto, pois é uma medida de volume. "Um bushel de tomates, por exemplo, deve pesar 56 libras... Folhas e verduras ocupam mais espaço com menos peso, então um bushel de espinafre pesa apenas 20 libras." É tudo bastante complicado! Vamos evitar fazer uma conversão de bushel para libra e, em vez disso, calcular o preço por bushel. Todo este estudo sobre bushels de abóboras, no entanto, mostra como é muito importante entender a natureza dos seus dados!

Agora, pode analisar o preço por unidade com base na medida de bushel. Se imprimir os dados mais uma vez, verá como estão padronizados.

✅ Reparou que as abóboras vendidas por meio bushel são muito caras? Consegue descobrir porquê? Dica: abóboras pequenas são muito mais caras do que grandes, provavelmente porque há muito mais delas por bushel, dado o espaço não utilizado ocupado por uma grande abóbora oca para torta.

## Estratégias de Visualização

Parte do papel do cientista de dados é demonstrar a qualidade e a natureza dos dados com os quais está a trabalhar. Para isso, frequentemente criam visualizações interessantes, como gráficos, diagramas e tabelas, mostrando diferentes aspetos dos dados. Desta forma, conseguem mostrar visualmente relações e lacunas que, de outra forma, seriam difíceis de identificar.

[![ML para iniciantes - Como visualizar dados com Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML para iniciantes - Como visualizar dados com Matplotlib")

> 🎥 Clique na imagem acima para assistir a um vídeo curto sobre como visualizar os dados para esta lição.

As visualizações também podem ajudar a determinar a técnica de machine learning mais apropriada para os dados. Um gráfico de dispersão que parece seguir uma linha, por exemplo, indica que os dados são bons candidatos para um exercício de regressão linear.

Uma biblioteca de visualização de dados que funciona bem em notebooks Jupyter é [Matplotlib](https://matplotlib.org/) (que também viu na lição anterior).

> Obtenha mais experiência com visualização de dados nestes [tutoriais](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Exercício - experimentar com Matplotlib

Tente criar alguns gráficos básicos para exibir o novo dataframe que acabou de criar. O que um gráfico de linha básico mostraria?

1. Importe o Matplotlib no topo do ficheiro, sob a importação do Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Execute novamente todo o notebook para atualizar.
1. No final do notebook, adicione uma célula para plotar os dados como um boxplot:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Um gráfico de dispersão mostrando a relação entre preço e mês](../../../../2-Regression/2-Data/images/scatterplot.png)

    Este gráfico é útil? Há algo nele que o surpreenda?

    Não é particularmente útil, pois apenas exibe os seus dados como uma dispersão de pontos num determinado mês.

### Torná-lo útil

Para que os gráficos exibam dados úteis, geralmente é necessário agrupar os dados de alguma forma. Vamos tentar criar um gráfico onde o eixo y mostra os meses e os dados demonstram a distribuição.

1. Adicione uma célula para criar um gráfico de barras agrupado:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Um gráfico de barras mostrando a relação entre preço e mês](../../../../2-Regression/2-Data/images/barchart.png)

    Este é um gráfico de visualização de dados mais útil! Parece indicar que o preço mais alto das abóboras ocorre em setembro e outubro. Isso corresponde às suas expectativas? Porquê ou porquê não?

---

## 🚀Desafio

Explore os diferentes tipos de visualização que o Matplotlib oferece. Quais tipos são mais apropriados para problemas de regressão?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Explore as várias formas de visualizar dados. Faça uma lista das diferentes bibliotecas disponíveis e anote quais são melhores para determinados tipos de tarefas, por exemplo, visualizações 2D vs. visualizações 3D. O que descobre?

## Tarefa

[Explorar visualização](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.