<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T19:16:40+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "nl"
}
-->
# K-Means clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

In deze les leer je hoe je clusters kunt maken met behulp van Scikit-learn en de Nigeriaanse muziekdataset die je eerder hebt geïmporteerd. We behandelen de basisprincipes van K-Means voor clustering. Onthoud dat, zoals je in de vorige les hebt geleerd, er veel manieren zijn om met clusters te werken en dat de methode die je gebruikt afhankelijk is van je data. We gaan K-Means proberen, omdat dit de meest gebruikte clusteringtechniek is. Laten we beginnen!

Termen die je zult leren:

- Silhouettescore
- Elbow-methode
- Inertie
- Variantie

## Introductie

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) is een methode afkomstig uit het domein van signaalverwerking. Het wordt gebruikt om groepen data te verdelen en te partitioneren in 'k' clusters met behulp van een reeks observaties. Elke observatie werkt om een gegeven datapunt te groeperen bij het dichtstbijzijnde 'gemiddelde', of het middelpunt van een cluster.

De clusters kunnen worden gevisualiseerd als [Voronoi-diagrammen](https://wikipedia.org/wiki/Voronoi_diagram), die een punt (of 'zaad') en de bijbehorende regio bevatten.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> infographic door [Jen Looper](https://twitter.com/jenlooper)

Het K-Means clusteringproces [wordt uitgevoerd in een driestapsproces](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Het algoritme selecteert een k-aantal middelpuntpunten door te sampelen uit de dataset. Daarna herhaalt het:
    1. Het wijst elke sample toe aan het dichtstbijzijnde middelpunt.
    2. Het creëert nieuwe middelpunten door de gemiddelde waarde te nemen van alle samples die aan de vorige middelpunten zijn toegewezen.
    3. Vervolgens berekent het het verschil tussen de nieuwe en oude middelpunten en herhaalt dit totdat de middelpunten gestabiliseerd zijn.

Een nadeel van het gebruik van K-Means is dat je 'k' moet vaststellen, dat wil zeggen het aantal middelpunten. Gelukkig helpt de 'elbow-methode' om een goede startwaarde voor 'k' te schatten. Je zult dit zo proberen.

## Vereisten

Je werkt in het [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) bestand van deze les, dat de data-import en voorlopige schoonmaak bevat die je in de vorige les hebt gedaan.

## Oefening - voorbereiding

Begin met opnieuw naar de liedjesdata te kijken.

1. Maak een boxplot door `boxplot()` aan te roepen voor elke kolom:

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

    Deze data is een beetje rommelig: door elke kolom als een boxplot te observeren, kun je uitschieters zien.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Je zou door de dataset kunnen gaan en deze uitschieters kunnen verwijderen, maar dat zou de data behoorlijk minimaliseren.

1. Kies voorlopig welke kolommen je zult gebruiken voor je clusteringoefening. Kies kolommen met vergelijkbare bereiken en codeer de `artist_top_genre` kolom als numerieke data:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Nu moet je bepalen hoeveel clusters je wilt targeten. Je weet dat er 3 muziekgenres zijn die we uit de dataset hebben gehaald, dus laten we 3 proberen:

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

Je ziet een array die wordt afgedrukt met voorspelde clusters (0, 1 of 2) voor elke rij van de dataframe.

1. Gebruik deze array om een 'silhouettescore' te berekenen:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouettescore

Zoek naar een silhouettescore dichter bij 1. Deze score varieert van -1 tot 1, en als de score 1 is, is de cluster dicht en goed gescheiden van andere clusters. Een waarde dicht bij 0 vertegenwoordigt overlappende clusters met samples die erg dicht bij de beslissingsgrens van de naburige clusters liggen. [(Bron)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Onze score is **.53**, dus precies in het midden. Dit geeft aan dat onze data niet bijzonder geschikt is voor dit type clustering, maar laten we doorgaan.

### Oefening - bouw een model

1. Importeer `KMeans` en start het clusteringproces.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Er zijn een paar onderdelen die uitleg verdienen.

    > 🎓 range: Dit zijn de iteraties van het clusteringproces

    > 🎓 random_state: "Bepaalt willekeurige nummergeneratie voor middelpuntinitialisatie." [Bron](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" meet de gemiddelde kwadratische afstand van alle punten binnen een cluster tot het cluster-middelpunt. [Bron](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Inertie: K-Means algoritmes proberen middelpunten te kiezen om 'inertie' te minimaliseren, "een maatstaf voor hoe intern coherent clusters zijn." [Bron](https://scikit-learn.org/stable/modules/clustering.html). De waarde wordt bij elke iteratie toegevoegd aan de wcss-variabele.

    > 🎓 k-means++: In [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) kun je de 'k-means++' optimalisatie gebruiken, die "de middelpunten initialiseert om (over het algemeen) ver van elkaar te liggen, wat waarschijnlijk betere resultaten oplevert dan willekeurige initialisatie."

### Elbow-methode

Eerder vermoedde je dat, omdat je 3 muziekgenres hebt getarget, je 3 clusters zou moeten kiezen. Maar is dat wel zo?

1. Gebruik de 'elbow-methode' om dit te controleren.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Gebruik de `wcss`-variabele die je in de vorige stap hebt gebouwd om een grafiek te maken die laat zien waar de 'knik' in de elleboog zit, wat het optimale aantal clusters aangeeft. Misschien is het **wel** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Oefening - toon de clusters

1. Probeer het proces opnieuw, stel deze keer drie clusters in en toon de clusters als een scatterplot:

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

1. Controleer de nauwkeurigheid van het model:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    De nauwkeurigheid van dit model is niet erg goed, en de vorm van de clusters geeft je een hint waarom.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Deze data is te onevenwichtig, te weinig gecorreleerd en er is te veel variatie tussen de kolomwaarden om goed te clusteren. In feite worden de clusters die zich vormen waarschijnlijk sterk beïnvloed of scheefgetrokken door de drie genrecategorieën die we hierboven hebben gedefinieerd. Dat was een leerproces!

    In de documentatie van Scikit-learn kun je zien dat een model zoals dit, met clusters die niet goed afgebakend zijn, een 'variantie'-probleem heeft:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infographic van Scikit-learn

## Variantie

Variantie wordt gedefinieerd als "het gemiddelde van de kwadratische verschillen ten opzichte van het gemiddelde" [(Bron)](https://www.mathsisfun.com/data/standard-deviation.html). In de context van dit clusteringprobleem verwijst het naar data waarbij de waarden van onze dataset de neiging hebben om te veel af te wijken van het gemiddelde.

✅ Dit is een goed moment om na te denken over alle manieren waarop je dit probleem zou kunnen oplossen. De data nog wat meer aanpassen? Andere kolommen gebruiken? Een ander algoritme gebruiken? Tip: Probeer [je data te schalen](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) om deze te normaliseren en test andere kolommen.

> Probeer deze '[variantiecalculator](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' om het concept wat beter te begrijpen.

---

## 🚀Uitdaging

Besteed wat tijd aan deze notebook en pas parameters aan. Kun je de nauwkeurigheid van het model verbeteren door de data meer te schonen (bijvoorbeeld uitschieters verwijderen)? Je kunt gewichten gebruiken om bepaalde datasamples meer gewicht te geven. Wat kun je nog meer doen om betere clusters te maken?

Tip: Probeer je data te schalen. Er is gecommentarieerde code in de notebook die standaard schaling toevoegt om de datakolommen meer op elkaar te laten lijken qua bereik. Je zult merken dat terwijl de silhouettescore daalt, de 'knik' in de ellebooggrafiek gladder wordt. Dit komt omdat het laten staan van de data zonder schaling ervoor zorgt dat data met minder variatie meer gewicht krijgt. Lees hier meer over dit probleem [hier](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Bekijk een K-Means Simulator [zoals deze](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Je kunt deze tool gebruiken om sampledatapunten te visualiseren en de middelpunten te bepalen. Je kunt de willekeurigheid van de data, het aantal clusters en het aantal middelpunten aanpassen. Helpt dit je om een idee te krijgen van hoe de data kan worden gegroepeerd?

Bekijk ook [deze hand-out over K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) van Stanford.

## Opdracht

[Probeer verschillende clusteringmethoden](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.