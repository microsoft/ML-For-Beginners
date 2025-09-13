<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-04T22:18:39+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "es"
}
-->
# Agrupamiento K-Means

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

En esta lección, aprenderás a crear agrupaciones utilizando Scikit-learn y el conjunto de datos de música nigeriana que importaste anteriormente. Cubriremos los conceptos básicos de K-Means para el agrupamiento. Ten en cuenta que, como aprendiste en la lección anterior, hay muchas formas de trabajar con agrupaciones y el método que utilices dependerá de tus datos. Probaremos K-Means, ya que es la técnica de agrupamiento más común. ¡Comencemos!

Términos que aprenderás:

- Puntuación de silueta
- Método del codo
- Inercia
- Varianza

## Introducción

[El agrupamiento K-Means](https://wikipedia.org/wiki/K-means_clustering) es un método derivado del ámbito del procesamiento de señales. Se utiliza para dividir y particionar grupos de datos en 'k' agrupaciones utilizando una serie de observaciones. Cada observación trabaja para agrupar un punto de datos dado al 'promedio' más cercano, o el punto central de una agrupación.

Las agrupaciones pueden visualizarse como [diagramas de Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), que incluyen un punto (o 'semilla') y su región correspondiente.

![diagrama de voronoi](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografía por [Jen Looper](https://twitter.com/jenlooper)

El proceso de agrupamiento K-Means [se ejecuta en tres pasos](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. El algoritmo selecciona un número k de puntos centrales muestreando del conjunto de datos. Después, entra en un bucle:
    1. Asigna cada muestra al centroide más cercano.
    2. Crea nuevos centroides tomando el valor promedio de todas las muestras asignadas a los centroides anteriores.
    3. Luego, calcula la diferencia entre los nuevos y antiguos centroides y repite hasta que los centroides se estabilicen.

Una desventaja de usar K-Means es que necesitas establecer 'k', es decir, el número de centroides. Afortunadamente, el 'método del codo' ayuda a estimar un buen valor inicial para 'k'. Lo probarás en un momento.

## Prerrequisito

Trabajarás en el archivo [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) de esta lección, que incluye la importación de datos y la limpieza preliminar que realizaste en la lección anterior.

## Ejercicio - preparación

Comienza revisando nuevamente los datos de las canciones.

1. Crea un diagrama de caja llamando a `boxplot()` para cada columna:

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

    Estos datos son un poco ruidosos: al observar cada columna como un diagrama de caja, puedes ver valores atípicos.

    ![valores atípicos](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Podrías recorrer el conjunto de datos y eliminar estos valores atípicos, pero eso haría que los datos sean bastante mínimos.

1. Por ahora, elige qué columnas usarás para tu ejercicio de agrupamiento. Escoge aquellas con rangos similares y codifica la columna `artist_top_genre` como datos numéricos:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Ahora necesitas elegir cuántas agrupaciones apuntar. Sabes que hay 3 géneros musicales que extrajimos del conjunto de datos, así que probemos con 3:

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

Verás un arreglo impreso con las agrupaciones predichas (0, 1 o 2) para cada fila del dataframe.

1. Usa este arreglo para calcular una 'puntuación de silueta':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Puntuación de silueta

Busca una puntuación de silueta cercana a 1. Esta puntuación varía de -1 a 1, y si la puntuación es 1, la agrupación es densa y está bien separada de otras agrupaciones. Un valor cercano a 0 representa agrupaciones superpuestas con muestras muy cercanas al límite de decisión de las agrupaciones vecinas. [(Fuente)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Nuestra puntuación es **0.53**, justo en el medio. Esto indica que nuestros datos no son particularmente adecuados para este tipo de agrupamiento, pero sigamos adelante.

### Ejercicio - construir un modelo

1. Importa `KMeans` y comienza el proceso de agrupamiento.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Hay algunas partes aquí que merecen explicación.

    > 🎓 range: Estas son las iteraciones del proceso de agrupamiento.

    > 🎓 random_state: "Determina la generación de números aleatorios para la inicialización de centroides." [Fuente](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "suma de cuadrados dentro de la agrupación" mide la distancia promedio al cuadrado de todos los puntos dentro de una agrupación al centroide de la agrupación. [Fuente](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inercia: Los algoritmos K-Means intentan elegir centroides para minimizar la 'inercia', "una medida de cuán coherentes son internamente las agrupaciones." [Fuente](https://scikit-learn.org/stable/modules/clustering.html). El valor se agrega a la variable wcss en cada iteración.

    > 🎓 k-means++: En [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) puedes usar la optimización 'k-means++', que "inicializa los centroides para que estén (generalmente) distantes entre sí, lo que lleva a resultados probablemente mejores que la inicialización aleatoria."

### Método del codo

Anteriormente, dedujiste que, dado que apuntaste a 3 géneros musicales, deberías elegir 3 agrupaciones. ¿Pero es ese el caso?

1. Usa el 'método del codo' para asegurarte.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Usa la variable `wcss` que construiste en el paso anterior para crear un gráfico que muestre dónde está el 'doblez' en el codo, lo que indica el número óptimo de agrupaciones. ¡Quizás sí sea 3!

    ![método del codo](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Ejercicio - mostrar las agrupaciones

1. Intenta el proceso nuevamente, esta vez configurando tres agrupaciones, y muestra las agrupaciones como un diagrama de dispersión:

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

1. Verifica la precisión del modelo:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    La precisión de este modelo no es muy buena, y la forma de las agrupaciones te da una pista del porqué.

    ![agrupaciones](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Estos datos están demasiado desequilibrados, poco correlacionados y hay demasiada varianza entre los valores de las columnas para agruparlos bien. De hecho, las agrupaciones que se forman probablemente están muy influenciadas o sesgadas por las tres categorías de géneros que definimos anteriormente. ¡Fue un proceso de aprendizaje!

    En la documentación de Scikit-learn, puedes ver que un modelo como este, con agrupaciones no muy bien delimitadas, tiene un problema de 'varianza':

    ![modelos problemáticos](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografía de Scikit-learn

## Varianza

La varianza se define como "el promedio de las diferencias al cuadrado respecto a la media" [(Fuente)](https://www.mathsisfun.com/data/standard-deviation.html). En el contexto de este problema de agrupamiento, se refiere a datos cuyos números tienden a divergir demasiado de la media.

✅ Este es un buen momento para pensar en todas las formas en que podrías corregir este problema. ¿Ajustar un poco más los datos? ¿Usar diferentes columnas? ¿Usar un algoritmo diferente? Pista: Prueba [escalar tus datos](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) para normalizarlos y prueba con otras columnas.

> Prueba este '[calculador de varianza](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' para entender mejor el concepto.

---

## 🚀Desafío

Dedica algo de tiempo a este notebook ajustando parámetros. ¿Puedes mejorar la precisión del modelo limpiando más los datos (eliminando valores atípicos, por ejemplo)? Puedes usar pesos para dar más importancia a ciertas muestras de datos. ¿Qué más puedes hacer para crear mejores agrupaciones?

Pista: Prueba escalar tus datos. Hay código comentado en el notebook que agrega escalado estándar para hacer que las columnas de datos se parezcan más en términos de rango. Descubrirás que, aunque la puntuación de silueta disminuye, el 'doblez' en el gráfico del codo se suaviza. Esto se debe a que dejar los datos sin escalar permite que los datos con menos varianza tengan más peso. Lee un poco más sobre este problema [aquí](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Repaso y autoestudio

Echa un vistazo a un simulador de K-Means [como este](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Puedes usar esta herramienta para visualizar puntos de datos de muestra y determinar sus centroides. Puedes editar la aleatoriedad de los datos, el número de agrupaciones y el número de centroides. ¿Te ayuda esto a tener una idea de cómo se pueden agrupar los datos?

Además, revisa [este documento sobre K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) de Stanford.

## Tarea

[Prueba diferentes métodos de agrupamiento](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.