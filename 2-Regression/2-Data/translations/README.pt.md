# Crie um modelo de regressão usando o Scikit-learn: preparar e visualizar dados

![Infográfico de visualização de dados](../images/data-visualization.png)

Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Teste de pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/11/)

> ### [Esta lição está disponível em R!](./solution/R/lesson_2-R.ipynb)

## Introdução

Agora que você está configurado com as ferramentas necessárias para começar a lidar com a construção de modelos de aprendizagem automática com o Scikit-learn, você está pronto para começar a fazer perguntas sobre seus dados. Como você trabalha com dados e aplica soluções ML, é muito importante entender como fazer a pergunta certa para desbloquear adequadamente os potenciais de seu conjunto de dados.

Nesta lição, você aprenderá:

- Como preparar seus dados para a criação de modelos.
- Como usar Matplotlib para visualização de dados.

[![Preparação e Visualização de Dados](https://img.youtube.com/vi/11AnOn_OAcE/0.jpg)](https://youtu.be/11AnOn_OAcE "Preparando e Visualizando vídeo de dados - Clique para Assistir!")
> 🎥 Clique na imagem acima para ver um vídeo que aborda os principais aspectos desta lição


## Fazendo a pergunta certa sobre seus dados

A pergunta que você precisa responder determinará que tipo de algoritmos de ML você utilizará. E a qualidade da resposta que você recebe de volta será fortemente dependente da natureza de seus dados.

Dê uma olhada nos [dados](../data/US-pumpkins.csv) fornecidos para esta lição. Você pode abrir este arquivo .csv no Código VS. Um skim rápido imediatamente mostra que há espaços em branco e uma mistura de strings e dados numéricos. Há também uma coluna estranha chamada 'Package' onde os dados são uma mistura entre 'sacks', 'bins' e outros valores. Os dados, de fato, são um pouco confusos.

Na verdade, não é muito comum ser dotado de um conjunto de dados que está completamente pronto para usar para criar um modelo ML pronto para uso. Nesta lição, você aprenderá como preparar um conjunto de dados bruto usando bibliotecas Python padrão. Você também aprenderá várias técnicas para visualizar os dados.

## Estudo de caso: "mercado da abóbora"

Nesta pasta você encontrará um arquivo .csv na pasta raiz `data` chamada [US-pumpkins.csv](../data/US-pumpkins.csv) que inclui 1757 linhas de dados sobre o mercado de abóboras, classificadas em agrupamentos por cidade. Estes são dados brutos extraídos dos [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuídos pelo Departamento de Agricultura dos Estados Unidos.

### Preparando dados

Estes dados são do domínio público. Ele pode ser baixado em muitos arquivos separados, por cidade, a partir do site USDA. Para evitar muitos arquivos separados, nós concatenamos todos os dados da cidade em uma planilha, assim nós já _preparamos_ os dados um pouco. A seguir, vamos dar uma olhada nos dados.

### Os dados da abóbora - primeiras conclusões

O que você nota sobre esses dados? Vocês já viram que há uma mistura de strings, números, espaços em branco e valores estranhos que você precisa entender.

Que pergunta você pode fazer sobre esses dados, usando uma técnica de Regressão? E quanto a "Prever o preço de uma abóbora à venda durante um determinado mês". Observando novamente os dados, há algumas alterações que você precisa fazer para criar a estrutura de dados necessária para a tarefa.
## Exercício - analisar os dados da abóbora

Vamos usar [Pandas](https://pandas.pydata.org/), (o nome significa `Python Data Analysis`) uma ferramenta muito útil para moldar dados, para analisar e preparar esses dados de abóbora.

### Primeiro, verifique se há datas ausentes

Primeiro, você precisará seguir as etapas para verificar se há datas ausentes:

1. Converta as datas em um formato de mês (essas são datas americanas, portanto o formato é `MM/DD/AAAA`).
2. Extraia o mês para uma nova coluna.

Abra o arquivo _notebook.ipynb_ no Visual Studio Code e importe a planilha para um novo quadro de dados do Pandas.

1. Use a função `head()` para exibir as cinco primeiras linhas.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    Qual função você usaria para exibir as últimas cinco linhas?

1. Verifique se há dados ausentes no banco de dados atual:

    ```python
    pumpkins.isnull().sum()
    ```

    Faltam dados, mas talvez não seja importante para a tarefa em questão.

1. Para facilitar o trabalho com seu banco de dados, solte várias de suas colunas, usando `drop()`, mantendo apenas as colunas necessárias:

    ```python
    new_columns = ['Package', 'Month', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)
    ```

### Segundo, determinar o preço médio da abóbora

Pense sobre como determinar o preço médio de uma abóbora em um determinado mês. Que colunas você escolheria para esta tarefa? Dica: você precisará de 3 colunas.

Solução: utilize a média das colunas `Preço Baixo` e Preço Alto` para preencher a nova coluna Preço e converta a coluna Data para mostrar apenas o mês. Felizmente, de acordo com a verificação acima, não há dados ausentes para datas ou preços.

1. Para calcular a média, adicione o seguinte código:
2. 
    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅Sinta-se à vontade para imprimir quaisquer dados que você gostaria de verificar usando `print(month)`.

3. Agora, copie seus dados convertidos em um novo dataframe Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    A impressão do seu dataframe mostrará um conjunto de dados limpo e organizado no qual você pode construir seu novo modelo de regressão.

### Mas espere! Há algo estranho aqui

Se você observar a coluna `Package`, as abóboras são vendidas em várias configurações diferentes. Algumas são vendidas em medidas de "1 1/9 bushel", e algumas em medidas de "1/2 bushel", algumas por abóbora, algumas por libra, e algumas em caixas grandes com larguras variadas.

> Abóboras parecem muito difíceis de pesar consistentemente

Analisando os dados originais, é interessante que qualquer coisa com `Unidade de Venda` igual a 'CADA' ou 'POR CAIXA' também tenha o tipo `Pacote` por polegada, por caixa ou 'cada'. As abóboras parecem ser muito difíceis de pesar consistentemente, então vamos filtrá-las selecionando apenas as abóboras com a cadeia "bushel" na coluna `Pacote`.

1. Adicione um filtro na parte superior do arquivo, sob a importação .csv inicial:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

  Se você imprimir os dados agora, você pode ver que você está apenas recebendo as 415 ou mais linhas de dados contendo abóboras pelo bushel.

### Mas espere! Há mais uma coisa a fazer

Você notou que o montante de bushel varia por linha? Você precisa normalizar o preço para que você mostre o preço por bushel, então faça algumas contas para padronizá-lo.

1. Adicione estas linhas após o bloco criar o dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ De acordo com [O Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), o peso de um bushel depende do tipo de produto, pois é uma medida de volume. "Um bushel de tomates, por exemplo, deve pesar 56 libras... Folhas e verdes ocupam mais espaço com menos peso, então um alqueire de espinafre pesa apenas 20 libras." É tudo muito complicado! Não nos preocupemos em fazer uma conversão bushel-to-pound, e em vez disso o preço pelo bushel. Todo este estudo de bushels de abóboras, no entanto, vai para mostrar como é muito importante entender a natureza dos seus dados!

Agora, você pode analisar o preço por unidade com base em sua medição de bushel. Se você imprimir os dados mais uma vez, poderá ver como eles são padronizados.

Você notou que as abóboras vendidas pela metade do bushel são muito caras? Você pode descobrir por quê? Dica: abóboras pequenas são muito mais caras do que as grandes, provavelmente porque há muito mais delas por bushel, dado o espaço não utilizado tomado por uma grande abóbora de torta oca.

## Estratégias de visualização

Parte do papel do cientista de dados é demonstrar a qualidade e a natureza dos dados com os quais eles estão trabalhando. Para fazer isso, muitas vezes criam visualizações interessantes, ou gráficos, gráficos e gráficos, mostrando diferentes aspectos dos dados. Dessa forma, eles são capazes de mostrar visualmente relacionamentos e lacunas que de outra forma são difíceis de descobrir.

Visualizações também podem ajudar a determinar a técnica de aprendizado de máquina mais apropriada para os dados. Um gráfico de dispersão que parece seguir uma linha, por exemplo, indica que os dados são um bom candidato para um exercício de regressão linear.

Uma biblioteca de visualização de dados que funciona bem em notebooks Jupyter é [Matplotlib](https://matplotlib.org/) (que você também viu na lição anterior).

> Obtenha mais experiência com visualização de dados em [estes tutoriais](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=university-15963-cxa).

## Exercício - experimente com Matplotlib

Tente criar alguns gráficos básicos para exibir o novo banco de dados que você acabou de criar. O que um gráfico de linha básica mostraria?

1. Importar Matplotlib no topo do arquivo, sob a importação Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Execute novamente todo o bloco de anotações para atualizar.
1. Na parte inferior do notebook, adicione uma célula para plotar os dados como uma caixa:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

  ![Uma distribuição que mostra a relação preço/mês](../images/scatterplot.png)

Será isto um enredo útil? Alguma coisa sobre isso o surpreende?

Não é particularmente útil, uma vez que tudo o que apresenta nos seus dados como uma distribuição de pontos num determinado mês.

### Tornar útil

Para que os gráficos apresentem dados úteis, normalmente é necessário agrupar os dados de alguma forma. Vamos tentar criar um desenho onde o eixo Y mostre os meses e os dados demonstram a distribuição de dados.

1. Adicionar uma célula para criar um gráfico de barras agrupado:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

![Um gráfico de barras que mostra a relação preço/mês](../images/barchart.png)

Esta é uma visualização de dados mais útil! Parece indicar que o preço mais alto para as abrigas ocorre em setembro e outubro. Isso atende às suas expetativas? Porque ou porque não?

—

## 🚀 desafio

Explore os diferentes tipos de visualização que o Matplotlib oferece. Que tipos são mais apropriados para problemas de regressão?

## [Questionário pós-palestra](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/12/)

## Revisão e Estudo Automático

Dê uma vista de olhos às muitas formas de visualizar dados. Disponibilize uma lista das várias bibliotecas e anote quais as melhores para determinados tipos de tarefas, por exemplo, visualizações 2D vs. visualizações 3D. O que você descobre?

## Atribuição

[A explorar visualização](assignment.md)
