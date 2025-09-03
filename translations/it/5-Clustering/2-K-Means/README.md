<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "085d571097d201810720df4cd379f8c2",
  "translation_date": "2025-08-29T21:06:45+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "it"
}
-->
# Clustering K-Means

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/29/)

In questa lezione, imparerai a creare cluster utilizzando Scikit-learn e il dataset di musica nigeriana che hai importato in precedenza. Tratteremo le basi del K-Means per il clustering. Tieni presente che, come hai appreso nella lezione precedente, ci sono molti modi per lavorare con i cluster e il metodo che utilizzi dipende dai tuoi dati. Proveremo il K-Means poiché è la tecnica di clustering più comune. Iniziamo!

Termini che imparerai:

- Punteggio silhouette
- Metodo del gomito
- Inerzia
- Varianza

## Introduzione

[Clustering K-Means](https://wikipedia.org/wiki/K-means_clustering) è un metodo derivato dal campo dell'elaborazione del segnale. Viene utilizzato per dividere e raggruppare gruppi di dati in 'k' cluster utilizzando una serie di osservazioni. Ogni osservazione lavora per raggruppare un determinato punto dati al centroide più vicino, ovvero il punto centrale di un cluster.

I cluster possono essere visualizzati come [diagrammi di Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), che includono un punto (o 'seme') e la sua regione corrispondente.

![diagramma di voronoi](../../../../translated_images/voronoi.1dc1613fb0439b9564615eca8df47a4bcd1ce06217e7e72325d2406ef2180795.it.png)

> Infografica di [Jen Looper](https://twitter.com/jenlooper)

Il processo di clustering K-Means [si esegue in tre passaggi](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. L'algoritmo seleziona un numero k di punti centrali campionandoli dal dataset. Successivamente, esegue un ciclo:
    1. Assegna ogni campione al centroide più vicino.
    2. Crea nuovi centroidi calcolando il valore medio di tutti i campioni assegnati ai centroidi precedenti.
    3. Calcola la differenza tra i nuovi e i vecchi centroidi e ripete il processo fino a quando i centroidi si stabilizzano.

Uno svantaggio dell'utilizzo del K-Means è che è necessario stabilire 'k', ovvero il numero di centroidi. Fortunatamente, il 'metodo del gomito' aiuta a stimare un buon valore iniziale per 'k'. Lo proverai tra poco.

## Prerequisiti

Lavorerai nel file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) di questa lezione, che include l'importazione dei dati e la pulizia preliminare che hai effettuato nella lezione precedente.

## Esercizio - preparazione

Inizia dando un'altra occhiata ai dati delle canzoni.

1. Crea un boxplot, chiamando `boxplot()` per ogni colonna:

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

    Questi dati sono un po' rumorosi: osservando ogni colonna come boxplot, puoi vedere i valori anomali.

    ![outliers](../../../../translated_images/boxplots.8228c29dabd0f29227dd38624231a175f411f1d8d4d7c012cb770e00e4fdf8b6.it.png)

Potresti esaminare il dataset e rimuovere questi valori anomali, ma ciò renderebbe i dati piuttosto ridotti.

1. Per ora, scegli quali colonne utilizzare per il tuo esercizio di clustering. Seleziona quelle con intervalli simili e codifica la colonna `artist_top_genre` come dati numerici:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Ora devi scegliere quanti cluster targetizzare. Sai che ci sono 3 generi musicali che abbiamo estratto dal dataset, quindi proviamo con 3:

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

Vedrai stampato un array con i cluster previsti (0, 1 o 2) per ogni riga del dataframe.

1. Usa questo array per calcolare un 'punteggio silhouette':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Punteggio silhouette

Cerca un punteggio silhouette vicino a 1. Questo punteggio varia da -1 a 1, e se il punteggio è 1, il cluster è denso e ben separato dagli altri cluster. Un valore vicino a 0 rappresenta cluster sovrapposti con campioni molto vicini al confine decisionale dei cluster vicini. [(Fonte)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Il nostro punteggio è **0.53**, quindi a metà strada. Questo indica che i nostri dati non sono particolarmente adatti a questo tipo di clustering, ma continuiamo.

### Esercizio - costruire un modello

1. Importa `KMeans` e avvia il processo di clustering.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Ci sono alcune parti qui che meritano una spiegazione.

    > 🎓 range: Queste sono le iterazioni del processo di clustering.

    > 🎓 random_state: "Determina la generazione di numeri casuali per l'inizializzazione dei centroidi." [Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "Somma dei quadrati intra-cluster" misura la distanza media quadrata di tutti i punti all'interno di un cluster rispetto al centroide del cluster. [Fonte](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inerzia: Gli algoritmi K-Means tentano di scegliere i centroidi per minimizzare l'inerzia, "una misura di quanto siano coerenti internamente i cluster." [Fonte](https://scikit-learn.org/stable/modules/clustering.html). Il valore viene aggiunto alla variabile wcss a ogni iterazione.

    > 🎓 k-means++: In [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) puoi utilizzare l'ottimizzazione 'k-means++', che "inizializza i centroidi in modo che siano (generalmente) distanti tra loro, portando probabilmente a risultati migliori rispetto all'inizializzazione casuale."

### Metodo del gomito

In precedenza, hai ipotizzato che, poiché hai targetizzato 3 generi musicali, dovresti scegliere 3 cluster. Ma è davvero così?

1. Usa il 'metodo del gomito' per esserne sicuro.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Usa la variabile `wcss` che hai costruito nel passaggio precedente per creare un grafico che mostri dove si trova la 'curvatura' nel gomito, che indica il numero ottimale di cluster. Forse è davvero **3**!

    ![metodo del gomito](../../../../translated_images/elbow.72676169eed744ff03677e71334a16c6b8f751e9e716e3d7f40dd7cdef674cca.it.png)

## Esercizio - visualizzare i cluster

1. Prova di nuovo il processo, questa volta impostando tre cluster, e visualizza i cluster come uno scatterplot:

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

1. Controlla l'accuratezza del modello:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    L'accuratezza di questo modello non è molto buona, e la forma dei cluster ti dà un indizio sul perché.

    ![cluster](../../../../translated_images/clusters.b635354640d8e4fd4a49ef545495518e7be76172c97c13bd748f5b79f171f69a.it.png)

    Questi dati sono troppo sbilanciati, poco correlati e c'è troppa varianza tra i valori delle colonne per formare cluster ben definiti. Infatti, i cluster che si formano sono probabilmente fortemente influenzati o distorti dalle tre categorie di genere che abbiamo definito sopra. È stato un processo di apprendimento!

    Nella documentazione di Scikit-learn, puoi vedere che un modello come questo, con cluster non molto ben definiti, ha un problema di 'varianza':

    ![modelli problematici](../../../../translated_images/problems.f7fb539ccd80608e1f35c319cf5e3ad1809faa3c08537aead8018c6b5ba2e33a.it.png)
    > Infografica da Scikit-learn

## Varianza

La varianza è definita come "la media delle differenze al quadrato rispetto alla media" [(Fonte)](https://www.mathsisfun.com/data/standard-deviation.html). Nel contesto di questo problema di clustering, si riferisce al fatto che i numeri del nostro dataset tendono a divergere un po' troppo dalla media.

✅ Questo è un ottimo momento per pensare a tutti i modi in cui potresti correggere questo problema. Modificare un po' i dati? Usare colonne diverse? Utilizzare un algoritmo diverso? Suggerimento: prova a [scalare i tuoi dati](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) per normalizzarli e testare altre colonne.

> Prova questo '[calcolatore di varianza](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' per comprendere meglio il concetto.

---

## 🚀Sfida

Dedica un po' di tempo a questo notebook, modificando i parametri. Riesci a migliorare l'accuratezza del modello pulendo ulteriormente i dati (ad esempio, rimuovendo i valori anomali)? Puoi usare pesi per dare maggiore importanza a determinati campioni di dati. Cos'altro puoi fare per creare cluster migliori?

Suggerimento: prova a scalare i tuoi dati. Nel notebook c'è del codice commentato che aggiunge la scalatura standard per far sì che le colonne dei dati si somiglino di più in termini di intervallo. Scoprirai che, mentre il punteggio silhouette diminuisce, la 'curvatura' nel grafico del gomito si appiana. Questo perché lasciare i dati non scalati consente ai dati con meno varianza di avere un peso maggiore. Leggi un po' di più su questo problema [qui](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/30/)

## Revisione e studio autonomo

Dai un'occhiata a un simulatore di K-Means [come questo](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Puoi utilizzare questo strumento per visualizzare punti dati di esempio e determinare i loro centroidi. Puoi modificare la casualità dei dati, il numero di cluster e il numero di centroidi. Ti aiuta a capire come i dati possono essere raggruppati?

Inoltre, dai un'occhiata a [questo documento sul K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) di Stanford.

## Compito

[Prova metodi di clustering diversi](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.