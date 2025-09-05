<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T21:29:18+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "sv"
}
-->
# K-Means klustring

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

I den här lektionen kommer du att lära dig hur man skapar kluster med hjälp av Scikit-learn och den nigerianska musikdatabasen som du importerade tidigare. Vi kommer att gå igenom grunderna i K-Means för klustring. Kom ihåg att, som du lärde dig i den tidigare lektionen, finns det många sätt att arbeta med kluster och metoden du använder beror på din data. Vi kommer att testa K-Means eftersom det är den vanligaste klustringstekniken. Låt oss sätta igång!

Begrepp du kommer att lära dig om:

- Silhouettescore
- Elbow-metoden
- Inertia
- Varians

## Introduktion

[K-Means klustring](https://wikipedia.org/wiki/K-means_clustering) är en metod som härstammar från signalbehandlingsområdet. Den används för att dela och partitionera grupper av data i 'k' kluster med hjälp av en serie observationer. Varje observation arbetar för att gruppera en given datapunkt närmast dess närmaste 'medelvärde', eller mittpunkten av ett kluster.

Klustrerna kan visualiseras som [Voronoi-diagram](https://wikipedia.org/wiki/Voronoi_diagram), som inkluderar en punkt (eller 'frö') och dess motsvarande område.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografik av [Jen Looper](https://twitter.com/jenlooper)

K-Means klustringsprocessen [utförs i tre steg](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritmen väljer k antal mittpunkter genom att ta ett urval från datasetet. Därefter loopar den:
    1. Den tilldelar varje prov till den närmaste mittpunkten.
    2. Den skapar nya mittpunkter genom att ta medelvärdet av alla prover som tilldelats de tidigare mittpunkterna.
    3. Sedan beräknar den skillnaden mellan de nya och gamla mittpunkterna och upprepar tills mittpunkterna stabiliseras.

En nackdel med att använda K-Means är att du måste fastställa 'k', det vill säga antalet mittpunkter. Lyckligtvis hjälper 'elbow-metoden' till att uppskatta ett bra startvärde för 'k'. Du kommer att testa detta snart.

## Förutsättningar

Du kommer att arbeta i den här lektionens [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb)-fil som inkluderar dataimporten och den preliminära datarensningen du gjorde i den förra lektionen.

## Övning - förberedelse

Börja med att ta en ny titt på låtdatabasen.

1. Skapa ett boxplot genom att kalla på `boxplot()` för varje kolumn:

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

    Den här datan är lite brusig: genom att observera varje kolumn som ett boxplot kan du se avvikare.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Du skulle kunna gå igenom datasetet och ta bort dessa avvikare, men det skulle göra datan ganska minimal.

1. För tillfället, välj vilka kolumner du ska använda för din klustringsövning. Välj sådana med liknande intervall och koda kolumnen `artist_top_genre` som numerisk data:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Nu behöver du välja hur många kluster du ska rikta in dig på. Du vet att det finns 3 låtgenrer som vi har tagit fram ur datasetet, så låt oss testa med 3:

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

Du ser en array som skrivs ut med förutsagda kluster (0, 1 eller 2) för varje rad i dataramen.

1. Använd denna array för att beräkna en 'silhouettescore':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouettescore

Sök efter en silhouettescore nära 1. Denna score varierar från -1 till 1, och om scoren är 1 är klustret tätt och väl separerat från andra kluster. Ett värde nära 0 representerar överlappande kluster med prover som ligger mycket nära beslutsgränsen för de närliggande klustren. [(Källa)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Vår score är **.53**, alltså mitt i mellan. Detta indikerar att vår data inte är särskilt väl lämpad för denna typ av klustring, men låt oss fortsätta.

### Övning - bygg en modell

1. Importera `KMeans` och starta klustringsprocessen.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Det finns några delar här som förtjänar en förklaring.

    > 🎓 range: Detta är iterationerna av klustringsprocessen.

    > 🎓 random_state: "Bestämmer slumpmässig nummergenerering för initialisering av mittpunkter." [Källa](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" mäter det kvadrerade genomsnittliga avståndet för alla punkter inom ett kluster till klustrets mittpunkt. [Källa](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inertia: K-Means-algoritmer försöker välja mittpunkter för att minimera 'inertia', "ett mått på hur internt sammanhängande kluster är." [Källa](https://scikit-learn.org/stable/modules/clustering.html). Värdet läggs till i wcss-variabeln vid varje iteration.

    > 🎓 k-means++: I [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) kan du använda 'k-means++'-optimering, som "initialiserar mittpunkterna så att de (generellt sett) är långt ifrån varandra, vilket leder till troligen bättre resultat än slumpmässig initialisering."

### Elbow-metoden

Tidigare antog du att eftersom du har riktat in dig på 3 låtgenrer, bör du välja 3 kluster. Men är det verkligen så?

1. Använd 'elbow-metoden' för att vara säker.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Använd variabeln `wcss` som du byggde i föregående steg för att skapa ett diagram som visar var 'böjen' i armbågen är, vilket indikerar det optimala antalet kluster. Kanske är det **verkligen** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Övning - visa klustren

1. Testa processen igen, den här gången med tre kluster, och visa klustren som ett scatterplot:

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

1. Kontrollera modellens noggrannhet:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Den här modellens noggrannhet är inte särskilt bra, och formen på klustren ger dig en ledtråd om varför.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Den här datan är för obalanserad, för lite korrelerad och det finns för mycket varians mellan kolumnvärdena för att klustras väl. Faktum är att klustren som bildas förmodligen är starkt påverkade eller snedvridna av de tre genrekategorier vi definierade ovan. Det var en lärandeprocess!

    I Scikit-learns dokumentation kan du se att en modell som denna, med kluster som inte är särskilt väl avgränsade, har ett 'varians'-problem:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografik från Scikit-learn

## Varians

Varians definieras som "genomsnittet av de kvadrerade skillnaderna från medelvärdet" [(Källa)](https://www.mathsisfun.com/data/standard-deviation.html). I kontexten av detta klustringsproblem hänvisar det till data där siffrorna i vårt dataset tenderar att avvika lite för mycket från medelvärdet.

✅ Detta är ett bra tillfälle att fundera på alla sätt du kan korrigera detta problem. Justera datan lite mer? Använd andra kolumner? Använd en annan algoritm? Tips: Testa att [skala din data](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) för att normalisera den och testa andra kolumner.

> Testa denna '[varianskalkylator](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' för att förstå konceptet lite bättre.

---

## 🚀Utmaning

Tillbringa lite tid med denna notebook och justera parametrar. Kan du förbättra modellens noggrannhet genom att rensa datan mer (till exempel ta bort avvikare)? Du kan använda vikter för att ge mer vikt åt vissa dataprov. Vad mer kan du göra för att skapa bättre kluster?

Tips: Testa att skala din data. Det finns kommenterad kod i notebooken som lägger till standardisering för att få datakolumnerna att likna varandra mer i termer av intervall. Du kommer att märka att även om silhouettescoren går ner, så jämnas 'böjen' i armbågsdiagrammet ut. Detta beror på att om datan lämnas oskalad tillåts data med mindre varians att väga tyngre. Läs mer om detta problem [här](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Ta en titt på en K-Means Simulator [som denna](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Du kan använda detta verktyg för att visualisera exempeldata och bestämma dess mittpunkter. Du kan redigera datans slumpmässighet, antal kluster och antal mittpunkter. Hjälper detta dig att få en idé om hur datan kan grupperas?

Ta också en titt på [detta handout om K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) från Stanford.

## Uppgift

[Testa olika klustringsmetoder](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.