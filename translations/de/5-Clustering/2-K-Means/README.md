<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-04T21:56:30+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "de"
}
-->
# K-Means Clustering

## [Quiz vor der Lektion](https://ff-quizzes.netlify.app/en/ml/)

In dieser Lektion lernst du, wie man mit Scikit-learn und dem nigerianischen Musikdatensatz, den du zuvor importiert hast, Cluster erstellt. Wir behandeln die Grundlagen von K-Means für das Clustering. Denke daran, dass es, wie du in der vorherigen Lektion gelernt hast, viele Möglichkeiten gibt, mit Clustern zu arbeiten, und die Methode, die du wählst, hängt von deinen Daten ab. Wir werden K-Means ausprobieren, da es die gängigste Clustering-Technik ist. Los geht's!

Begriffe, die du kennenlernen wirst:

- Silhouette-Score
- Elbow-Methode
- Trägheit (Inertia)
- Varianz

## Einführung

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) ist eine Methode aus dem Bereich der Signalverarbeitung. Sie wird verwendet, um Datengruppen in 'k' Cluster zu unterteilen, basierend auf einer Reihe von Beobachtungen. Jede Beobachtung dient dazu, einen bestimmten Datenpunkt dem nächstgelegenen 'Mittelwert' oder dem Mittelpunkt eines Clusters zuzuordnen.

Die Cluster können als [Voronoi-Diagramme](https://wikipedia.org/wiki/Voronoi_diagram) visualisiert werden, die einen Punkt (oder 'Seed') und dessen zugehörige Region umfassen.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografik von [Jen Looper](https://twitter.com/jenlooper)

Der K-Means-Clustering-Prozess [läuft in einem dreistufigen Verfahren ab](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Der Algorithmus wählt eine Anzahl von k-Mittelpunkten aus, indem er Stichproben aus dem Datensatz zieht. Danach wiederholt er:
    1. Er ordnet jede Stichprobe dem nächstgelegenen Schwerpunkt zu.
    2. Er erstellt neue Schwerpunkte, indem er den Mittelwert aller Stichproben berechnet, die den vorherigen Schwerpunkten zugeordnet wurden.
    3. Dann berechnet er die Differenz zwischen den neuen und alten Schwerpunkten und wiederholt den Vorgang, bis die Schwerpunkte stabilisiert sind.

Ein Nachteil der Verwendung von K-Means ist, dass du 'k', also die Anzahl der Schwerpunkte, festlegen musst. Glücklicherweise hilft die 'Elbow-Methode', einen guten Ausgangswert für 'k' zu schätzen. Du wirst sie gleich ausprobieren.

## Voraussetzung

Du wirst in der Datei [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) arbeiten, die den Datenimport und die vorläufige Bereinigung enthält, die du in der letzten Lektion durchgeführt hast.

## Übung - Vorbereitung

Beginne damit, die Song-Daten noch einmal anzusehen.

1. Erstelle ein Boxplot, indem du `boxplot()` für jede Spalte aufrufst:

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

    Diese Daten sind etwas verrauscht: Wenn du jede Spalte als Boxplot betrachtest, kannst du Ausreißer erkennen.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Du könntest den Datensatz durchgehen und diese Ausreißer entfernen, aber das würde die Daten ziemlich minimieren.

1. Wähle vorerst aus, welche Spalten du für deine Clustering-Übung verwenden möchtest. Wähle solche mit ähnlichen Bereichen und kodiere die Spalte `artist_top_genre` als numerische Daten:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Jetzt musst du festlegen, wie viele Cluster du anstreben möchtest. Du weißt, dass es 3 Song-Genres gibt, die wir aus dem Datensatz herausgearbeitet haben, also probiere es mit 3:

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

Du siehst ein Array, das die vorhergesagten Cluster (0, 1 oder 2) für jede Zeile des Dataframes ausgibt.

1. Verwende dieses Array, um einen 'Silhouette-Score' zu berechnen:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette-Score

Suche nach einem Silhouette-Score, der näher bei 1 liegt. Dieser Score variiert zwischen -1 und 1, und wenn der Score 1 ist, ist das Cluster dicht und gut von anderen Clustern getrennt. Ein Wert nahe 0 repräsentiert sich überlappende Cluster mit Stichproben, die sehr nahe an der Entscheidungsgrenze der benachbarten Cluster liegen. [(Quelle)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Unser Score ist **0,53**, also genau in der Mitte. Das zeigt, dass unsere Daten nicht besonders gut für diese Art von Clustering geeignet sind, aber lass uns weitermachen.

### Übung - Ein Modell erstellen

1. Importiere `KMeans` und starte den Clustering-Prozess.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Es gibt einige Teile, die einer Erklärung bedürfen.

    > 🎓 range: Dies sind die Iterationen des Clustering-Prozesses.

    > 🎓 random_state: "Bestimmt die Zufallszahlengenerierung für die Initialisierung der Schwerpunkte." [Quelle](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" misst den quadrierten durchschnittlichen Abstand aller Punkte innerhalb eines Clusters zum Cluster-Schwerpunkt. [Quelle](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Trägheit (Inertia): K-Means-Algorithmen versuchen, Schwerpunkte so zu wählen, dass die 'Trägheit' minimiert wird, "ein Maß dafür, wie intern kohärent Cluster sind." [Quelle](https://scikit-learn.org/stable/modules/clustering.html). Der Wert wird bei jeder Iteration zur WCSS-Variablen hinzugefügt.

    > 🎓 k-means++: In [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) kannst du die 'k-means++'-Optimierung verwenden, die "die Schwerpunkte so initialisiert, dass sie (im Allgemeinen) weit voneinander entfernt sind, was wahrscheinlich bessere Ergebnisse als eine zufällige Initialisierung liefert."

### Elbow-Methode

Zuvor hast du angenommen, dass du 3 Cluster wählen solltest, da du 3 Song-Genres anvisiert hast. Aber ist das wirklich der Fall?

1. Verwende die 'Elbow-Methode', um sicherzugehen.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Verwende die `wcss`-Variable, die du im vorherigen Schritt erstellt hast, um ein Diagramm zu erstellen, das zeigt, wo der 'Knick' im Ellbogen liegt, der die optimale Anzahl von Clustern anzeigt. Vielleicht sind es tatsächlich **3**!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Übung - Die Cluster anzeigen

1. Wiederhole den Prozess, diesmal mit drei Clustern, und zeige die Cluster als Streudiagramm an:

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

1. Überprüfe die Genauigkeit des Modells:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Die Genauigkeit dieses Modells ist nicht sehr gut, und die Form der Cluster gibt dir einen Hinweis, warum.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Diese Daten sind zu unausgewogen, zu wenig korreliert, und es gibt zu viel Varianz zwischen den Spaltenwerten, um gut zu clustern. Tatsächlich werden die Cluster, die sich bilden, wahrscheinlich stark von den drei Genre-Kategorien beeinflusst, die wir oben definiert haben. Das war ein Lernprozess!

    In der Dokumentation von Scikit-learn kannst du sehen, dass ein Modell wie dieses, bei dem die Cluster nicht sehr gut abgegrenzt sind, ein 'Varianz'-Problem hat:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografik von Scikit-learn

## Varianz

Varianz wird definiert als "der Durchschnitt der quadrierten Abweichungen vom Mittelwert" [(Quelle)](https://www.mathsisfun.com/data/standard-deviation.html). Im Kontext dieses Clustering-Problems bezieht sich dies darauf, dass die Zahlen unseres Datensatzes dazu neigen, sich etwas zu stark vom Mittelwert zu entfernen.

✅ Dies ist ein guter Moment, um über alle Möglichkeiten nachzudenken, wie du dieses Problem beheben könntest. Die Daten etwas mehr anpassen? Andere Spalten verwenden? Einen anderen Algorithmus ausprobieren? Tipp: Versuche, [deine Daten zu skalieren](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/), um sie zu normalisieren, und teste andere Spalten.

> Probiere diesen '[Varianzrechner](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' aus, um das Konzept besser zu verstehen.

---

## 🚀Herausforderung

Verbringe etwas Zeit mit diesem Notebook und passe die Parameter an. Kannst du die Genauigkeit des Modells verbessern, indem du die Daten weiter bereinigst (z. B. Ausreißer entfernst)? Du kannst Gewichte verwenden, um bestimmten Datenproben mehr Gewicht zu geben. Was kannst du sonst noch tun, um bessere Cluster zu erstellen?

Tipp: Versuche, deine Daten zu skalieren. Im Notebook gibt es auskommentierten Code, der eine Standard-Skalierung hinzufügt, um die Daten-Spalten in Bezug auf den Bereich einander ähnlicher zu machen. Du wirst feststellen, dass der Silhouette-Score zwar sinkt, aber der 'Knick' im Ellbogen-Diagramm glatter wird. Das liegt daran, dass unskalierte Daten es Daten mit weniger Varianz erlauben, mehr Gewicht zu tragen. Lies mehr über dieses Problem [hier](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Quiz nach der Lektion](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Schau dir einen K-Means-Simulator [wie diesen hier](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) an. Mit diesem Tool kannst du Beispieldatenpunkte visualisieren und deren Schwerpunkte bestimmen. Du kannst die Zufälligkeit der Daten, die Anzahl der Cluster und die Anzahl der Schwerpunkte bearbeiten. Hilft dir das, eine Vorstellung davon zu bekommen, wie die Daten gruppiert werden können?

Sieh dir auch [dieses Handout zu K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) von Stanford an.

## Aufgabe

[Probiere verschiedene Clustering-Methoden aus](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.