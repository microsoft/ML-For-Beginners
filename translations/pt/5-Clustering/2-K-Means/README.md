<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T08:41:51+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "pt"
}
-->
# K-Means clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Nesta lição, vais aprender a criar clusters utilizando Scikit-learn e o conjunto de dados de música nigeriana que importaste anteriormente. Vamos abordar os fundamentos do K-Means para Clustering. Lembra-te de que, como aprendeste na lição anterior, existem várias formas de trabalhar com clusters e o método que utilizas depende dos teus dados. Vamos experimentar o K-Means, pois é a técnica de clustering mais comum. Vamos começar!

Termos que vais aprender:

- Pontuação de Silhouette
- Método do cotovelo
- Inércia
- Variância

## Introdução

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) é um método derivado do domínio do processamento de sinais. É utilizado para dividir e agrupar conjuntos de dados em 'k' clusters utilizando uma série de observações. Cada observação trabalha para agrupar um determinado ponto de dados mais próximo do seu 'média' mais próxima, ou o ponto central de um cluster.

Os clusters podem ser visualizados como [diagramas de Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), que incluem um ponto (ou 'semente') e a sua região correspondente.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infográfico por [Jen Looper](https://twitter.com/jenlooper)

O processo de clustering K-Means [executa-se em três etapas](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. O algoritmo seleciona um número k de pontos centrais ao amostrar do conjunto de dados. Depois disso, ele repete:
    1. Atribui cada amostra ao centróide mais próximo.
    2. Cria novos centróides ao calcular o valor médio de todas as amostras atribuídas aos centróides anteriores.
    3. Em seguida, calcula a diferença entre os novos e os antigos centróides e repete até que os centróides se estabilizem.

Uma desvantagem do uso do K-Means é o facto de que precisas de estabelecer 'k', ou seja, o número de centróides. Felizmente, o 'método do cotovelo' ajuda a estimar um bom valor inicial para 'k'. Vais experimentá-lo em breve.

## Pré-requisito

Vais trabalhar no ficheiro [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) desta lição, que inclui a importação de dados e a limpeza preliminar que fizeste na última lição.

## Exercício - preparação

Começa por dar outra olhada nos dados das músicas.

1. Cria um boxplot, chamando `boxplot()` para cada coluna:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    Estes dados são um pouco ruidosos: ao observar cada coluna como um boxplot, podes ver valores atípicos.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Poderias percorrer o conjunto de dados e remover esses valores atípicos, mas isso tornaria os dados bastante reduzidos.

1. Por agora, escolhe quais colunas vais usar para o teu exercício de clustering. Escolhe aquelas com intervalos semelhantes e codifica a coluna `artist_top_genre` como dados numéricos:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Agora precisas de escolher quantos clusters vais segmentar. Sabes que existem 3 géneros musicais que extraímos do conjunto de dados, então vamos tentar 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

Vês um array impresso com clusters previstos (0, 1 ou 2) para cada linha do dataframe.

1. Usa este array para calcular uma 'pontuação de silhouette':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Pontuação de Silhouette

Procura uma pontuação de silhouette mais próxima de 1. Esta pontuação varia de -1 a 1, e se a pontuação for 1, o cluster é denso e bem separado dos outros clusters. Um valor próximo de 0 representa clusters sobrepostos com amostras muito próximas da fronteira de decisão dos clusters vizinhos. [(Fonte)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

A nossa pontuação é **0.53**, ou seja, está no meio. Isso indica que os nossos dados não são particularmente adequados para este tipo de clustering, mas vamos continuar.

### Exercício - construir um modelo

1. Importa `KMeans` e inicia o processo de clustering.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Existem algumas partes aqui que merecem explicação.

    > 🎓 range: Estas são as iterações do processo de clustering.

    > 🎓 random_state: "Determina a geração de números aleatórios para a inicialização dos centróides." [Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "soma dos quadrados dentro do cluster" mede a distância média quadrada de todos os pontos dentro de um cluster ao centróide do cluster. [Fonte](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inércia: Os algoritmos K-Means tentam escolher centróides para minimizar a 'inércia', "uma medida de quão internamente coerentes são os clusters." [Fonte](https://scikit-learn.org/stable/modules/clustering.html). O valor é adicionado à variável wcss em cada iteração.

    > 🎓 k-means++: Em [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means), podes usar a otimização 'k-means++', que "inicializa os centróides para estarem (geralmente) distantes uns dos outros, levando a resultados provavelmente melhores do que a inicialização aleatória."

### Método do cotovelo

Anteriormente, deduziste que, como segmentaste 3 géneros musicais, deverias escolher 3 clusters. Mas será que é mesmo o caso?

1. Usa o 'método do cotovelo' para ter certeza.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Usa a variável `wcss` que construíste no passo anterior para criar um gráfico que mostra onde está a 'curvatura' no cotovelo, indicando o número ótimo de clusters. Talvez seja mesmo **3**!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Exercício - exibir os clusters

1. Experimenta o processo novamente, desta vez definindo três clusters, e exibe os clusters como um scatterplot:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. Verifica a precisão do modelo:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    A precisão deste modelo não é muito boa, e a forma dos clusters dá-te uma pista do porquê.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Estes dados são demasiado desequilibrados, pouco correlacionados e há demasiada variância entre os valores das colunas para formar bons clusters. Na verdade, os clusters que se formam provavelmente são fortemente influenciados ou enviesados pelas três categorias de géneros que definimos acima. Foi um processo de aprendizagem!

    Na documentação do Scikit-learn, podes ver que um modelo como este, com clusters não muito bem demarcados, tem um problema de 'variância':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infográfico do Scikit-learn

## Variância

A variância é definida como "a média das diferenças quadradas em relação à média" [(Fonte)](https://www.mathsisfun.com/data/standard-deviation.html). No contexto deste problema de clustering, refere-se a dados em que os números do nosso conjunto tendem a divergir um pouco demais da média.

✅ Este é um ótimo momento para pensar em todas as formas de corrigir este problema. Ajustar os dados um pouco mais? Usar colunas diferentes? Utilizar um algoritmo diferente? Dica: Experimenta [escalar os teus dados](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) para normalizá-los e testar outras colunas.

> Experimenta este '[calculador de variância](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' para entender melhor o conceito.

---

## 🚀Desafio

Passa algum tempo com este notebook, ajustando os parâmetros. Consegues melhorar a precisão do modelo ao limpar mais os dados (removendo valores atípicos, por exemplo)? Podes usar pesos para dar mais importância a determinadas amostras de dados. O que mais podes fazer para criar melhores clusters?

Dica: Experimenta escalar os teus dados. Há código comentado no notebook que adiciona escalonamento padrão para fazer com que as colunas de dados se assemelhem mais em termos de intervalo. Vais perceber que, embora a pontuação de silhouette diminua, a 'curvatura' no gráfico do cotovelo suaviza-se. Isso acontece porque deixar os dados sem escala permite que dados com menos variância tenham mais peso. Lê mais sobre este problema [aqui](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Dá uma olhada num Simulador de K-Means [como este](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Podes usar esta ferramenta para visualizar pontos de dados de amostra e determinar os seus centróides. Podes editar a aleatoriedade dos dados, o número de clusters e o número de centróides. Isso ajuda-te a ter uma ideia de como os dados podem ser agrupados?

Além disso, consulta [este documento sobre K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) da Stanford.

## Tarefa

[Experimenta diferentes métodos de clustering](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.