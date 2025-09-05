<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T00:05:35+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "da"
}
-->
# K-Means clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

I denne lektion vil du lære, hvordan du opretter klynger ved hjælp af Scikit-learn og det nigerianske musikdatasæt, du importerede tidligere. Vi vil dække det grundlæggende i K-Means til klyngedannelse. Husk, som du lærte i den tidligere lektion, at der er mange måder at arbejde med klynger på, og den metode, du bruger, afhænger af dine data. Vi vil prøve K-Means, da det er den mest almindelige teknik til klyngedannelse. Lad os komme i gang!

Begreber, du vil lære om:

- Silhouettescore
- Albue-metoden
- Inerti
- Varians

## Introduktion

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) er en metode, der stammer fra signalbehandlingsområdet. Den bruges til at opdele og partitionere grupper af data i 'k' klynger ved hjælp af en række observationer. Hver observation arbejder på at gruppere et givet datapunkt tættest på dets nærmeste 'gennemsnit', eller midtpunktet af en klynge.

Klyngerne kan visualiseres som [Voronoi-diagrammer](https://wikipedia.org/wiki/Voronoi_diagram), som inkluderer et punkt (eller 'frø') og dets tilsvarende område.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografik af [Jen Looper](https://twitter.com/jenlooper)

K-Means klyngedannelsesprocessen [udføres i en tretrinsproces](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritmen vælger k-antal midtpunkter ved at tage stikprøver fra datasættet. Derefter gentager den:
    1. Den tildeler hver prøve til det nærmeste midtpunkt.
    2. Den skaber nye midtpunkter ved at tage gennemsnitsværdien af alle prøver, der er tildelt de tidligere midtpunkter.
    3. Derefter beregner den forskellen mellem de nye og gamle midtpunkter og gentager, indtil midtpunkterne stabiliseres.

En ulempe ved at bruge K-Means er, at du skal fastsætte 'k', altså antallet af midtpunkter. Heldigvis hjælper 'albue-metoden' med at estimere en god startværdi for 'k'. Du vil prøve det om lidt.

## Forudsætning

Du vil arbejde i denne lektions [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) fil, som inkluderer dataimporten og den indledende rengøring, du lavede i den sidste lektion.

## Øvelse - forberedelse

Start med at tage et nyt kig på sangdataene.

1. Opret et boxplot ved at kalde `boxplot()` for hver kolonne:

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

    Disse data er lidt støjende: ved at observere hver kolonne som et boxplot kan du se outliers.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Du kunne gennemgå datasættet og fjerne disse outliers, men det ville gøre dataene ret minimale.

1. Vælg for nu, hvilke kolonner du vil bruge til din klyngedannelsesøvelse. Vælg dem med lignende intervaller og kod kolonnen `artist_top_genre` som numeriske data:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Nu skal du vælge, hvor mange klynger du vil målrette. Du ved, at der er 3 sanggenrer, som vi har udskilt fra datasættet, så lad os prøve med 3:

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

Du ser en array udskrevet med forudsagte klynger (0, 1 eller 2) for hver række i dataframen.

1. Brug denne array til at beregne en 'silhouettescore':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouettescore

Søg efter en silhouettescore tættere på 1. Denne score varierer fra -1 til 1, og hvis scoren er 1, er klyngen tæt og godt adskilt fra andre klynger. En værdi nær 0 repræsenterer overlappende klynger med prøver meget tæt på beslutningsgrænsen for de nærliggende klynger. [(Kilde)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Vores score er **.53**, altså midt i mellem. Dette indikerer, at vores data ikke er særligt velegnede til denne type klyngedannelse, men lad os fortsætte.

### Øvelse - byg en model

1. Importér `KMeans` og start klyngedannelsesprocessen.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Der er nogle dele her, der kræver forklaring.

    > 🎓 range: Dette er iterationerne af klyngedannelsesprocessen.

    > 🎓 random_state: "Bestemmer tilfældig talgenerering til initialisering af midtpunkter." [Kilde](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" måler den kvadrerede gennemsnitlige afstand af alle punkter inden for en klynge til klyngens midtpunkt. [Kilde](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inerti: K-Means algoritmer forsøger at vælge midtpunkter for at minimere 'inerti', "et mål for, hvor internt sammenhængende klynger er." [Kilde](https://scikit-learn.org/stable/modules/clustering.html). Værdien tilføjes til wcss-variablen ved hver iteration.

    > 🎓 k-means++: I [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) kan du bruge 'k-means++'-optimeringen, som "initialiserer midtpunkterne til at være (generelt) fjernt fra hinanden, hvilket sandsynligvis fører til bedre resultater end tilfældig initialisering."

### Albue-metoden

Tidligere antog du, at fordi du har målrettet 3 sanggenrer, bør du vælge 3 klynger. Men er det tilfældet?

1. Brug 'albue-metoden' for at være sikker.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Brug den `wcss`-variabel, du byggede i det foregående trin, til at oprette et diagram, der viser, hvor 'knækket' i albuen er, hvilket indikerer det optimale antal klynger. Måske er det **faktisk** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Øvelse - vis klyngerne

1. Prøv processen igen, denne gang med tre klynger, og vis klyngerne som et scatterplot:

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

1. Tjek modellens nøjagtighed:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Denne models nøjagtighed er ikke særlig god, og formen på klyngerne giver dig et hint om hvorfor.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Disse data er for ubalancerede, for lidt korrelerede, og der er for meget varians mellem kolonneværdierne til at danne gode klynger. Faktisk er de klynger, der dannes, sandsynligvis stærkt påvirket eller skævvredet af de tre genrekategorier, vi definerede ovenfor. Det var en læringsproces!

    I Scikit-learns dokumentation kan du se, at en model som denne, med klynger, der ikke er særlig godt afgrænsede, har et 'varians'-problem:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografik fra Scikit-learn

## Varians

Varians defineres som "gennemsnittet af de kvadrerede forskelle fra gennemsnittet" [(Kilde)](https://www.mathsisfun.com/data/standard-deviation.html). I konteksten af dette klyngedannelsesproblem refererer det til data, hvor tallene i vores datasæt har en tendens til at afvige lidt for meget fra gennemsnittet.

✅ Dette er et godt tidspunkt at tænke over alle de måder, du kunne rette dette problem på. Justere dataene lidt mere? Bruge andre kolonner? Bruge en anden algoritme? Tip: Prøv at [skalere dine data](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) for at normalisere dem og teste andre kolonner.

> Prøv denne '[variansberegner](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' for at forstå konceptet lidt bedre.

---

## 🚀Udfordring

Brug lidt tid med denne notebook og juster parametrene. Kan du forbedre modellens nøjagtighed ved at rense dataene mere (fjerne outliers, for eksempel)? Du kan bruge vægte til at give mere vægt til bestemte datapunkter. Hvad kan du ellers gøre for at skabe bedre klynger?

Tip: Prøv at skalere dine data. Der er kommenteret kode i notebooken, der tilføjer standard skalering for at få datakolonnerne til at ligne hinanden mere i forhold til interval. Du vil opdage, at mens silhouettescoren går ned, udjævnes 'knækket' i albuegrafen. Dette skyldes, at hvis dataene ikke skaleres, får data med mindre varians mere vægt. Læs lidt mere om dette problem [her](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Tag et kig på en K-Means Simulator [som denne](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Du kan bruge dette værktøj til at visualisere prøvepunkter og bestemme deres midtpunkter. Du kan redigere dataenes tilfældighed, antal klynger og antal midtpunkter. Hjælper dette dig med at få en idé om, hvordan dataene kan grupperes?

Tag også et kig på [dette handout om K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) fra Stanford.

## Opgave

[Prøv forskellige klyngedannelsesmetoder](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på at sikre nøjagtighed, skal det bemærkes, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.