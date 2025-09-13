<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T21:29:46+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "no"
}
-->
# K-Means klynging

## [Pre-forelesningsquiz](https://ff-quizzes.netlify.app/en/ml/)

I denne leksjonen vil du lære hvordan du lager klynger ved hjelp av Scikit-learn og det nigerianske musikkdatasettet du importerte tidligere. Vi skal dekke det grunnleggende om K-Means for klynging. Husk at, som du lærte i den forrige leksjonen, finnes det mange måter å jobbe med klynger på, og metoden du bruker avhenger av dataene dine. Vi skal prøve K-Means siden det er den mest vanlige klyngemetoden. La oss komme i gang!

Begreper du vil lære om:

- Silhuett-score
- Albuemetoden
- Inertia
- Varians

## Introduksjon

[K-Means klynging](https://wikipedia.org/wiki/K-means_clustering) er en metode som stammer fra signalbehandling. Den brukes til å dele og gruppere datasett i 'k' klynger basert på en serie observasjoner. Hver observasjon jobber for å gruppere et gitt datapunkt nærmest sin 'gjennomsnittlige verdi', eller midtpunktet i en klynge.

Klyngene kan visualiseres som [Voronoi-diagrammer](https://wikipedia.org/wiki/Voronoi_diagram), som inkluderer et punkt (eller 'frø') og dets tilhørende område.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografikk av [Jen Looper](https://twitter.com/jenlooper)

K-Means klyngingsprosessen [utføres i tre trinn](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritmen velger et k-antall midtpunkter ved å ta prøver fra datasettet. Deretter gjentas følgende:
    1. Den tilordner hver prøve til det nærmeste midtpunktet.
    2. Den lager nye midtpunkter ved å ta gjennomsnittsverdien av alle prøvene som er tilordnet de tidligere midtpunktene.
    3. Deretter beregner den forskjellen mellom de nye og gamle midtpunktene og gjentar til midtpunktene stabiliseres.

En ulempe med å bruke K-Means er at du må bestemme 'k', altså antallet midtpunkter. Heldigvis kan 'albuemetoden' hjelpe med å estimere en god startverdi for 'k'. Du skal prøve det om et øyeblikk.

## Forutsetninger

Du vil jobbe i denne leksjonens [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb)-fil, som inkluderer dataimporten og den innledende rengjøringen du gjorde i forrige leksjon.

## Øvelse - forberedelse

Start med å ta en ny titt på sangdataene.

1. Lag et boksplott ved å kalle `boxplot()` for hver kolonne:

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

    Disse dataene er litt støyete: ved å observere hver kolonne som et boksplott kan du se uteliggere.

    ![uteliggere](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Du kunne gått gjennom datasettet og fjernet disse uteliggerne, men det ville gjort dataene ganske minimale.

1. For nå, velg hvilke kolonner du vil bruke til klyngingsøvelsen. Velg kolonner med lignende verdier og kod `artist_top_genre`-kolonnen som numeriske data:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Nå må du velge hvor mange klynger du vil målrette. Du vet at det er 3 sangsjangre som vi har skilt ut fra datasettet, så la oss prøve med 3:

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

Du ser en matrise som skrives ut med forutsagte klynger (0, 1 eller 2) for hver rad i dataframen.

1. Bruk denne matrisen til å beregne en 'silhuett-score':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhuett-score

Se etter en silhuett-score nærmere 1. Denne scoren varierer fra -1 til 1, og hvis scoren er 1, er klyngen tett og godt adskilt fra andre klynger. En verdi nær 0 representerer overlappende klynger med prøver som er veldig nær beslutningsgrensen til naboklyngene. [(Kilde)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Vår score er **0,53**, altså midt på treet. Dette indikerer at dataene våre ikke er spesielt godt egnet for denne typen klynging, men la oss fortsette.

### Øvelse - bygg en modell

1. Importer `KMeans` og start klyngingsprosessen.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Det er noen deler her som fortjener en forklaring.

    > 🎓 range: Dette er iterasjonene av klyngingsprosessen.

    > 🎓 random_state: "Bestemmer tilfeldig tallgenerering for initialisering av midtpunkter." [Kilde](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "Summen av kvadrerte avstander innenfor klyngen" måler den kvadrerte gjennomsnittsavstanden for alle punktene innenfor en klynge til klyngens midtpunkt. [Kilde](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inertia: K-Means-algoritmer forsøker å velge midtpunkter for å minimere 'inertia', "et mål på hvor internt sammenhengende klynger er." [Kilde](https://scikit-learn.org/stable/modules/clustering.html). Verdien legges til variabelen wcss ved hver iterasjon.

    > 🎓 k-means++: I [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) kan du bruke 'k-means++'-optimalisering, som "initialiserer midtpunktene til å være (generelt) langt fra hverandre, noe som gir sannsynligvis bedre resultater enn tilfeldig initialisering."

### Albuemetoden

Tidligere antok du at, siden du har målrettet 3 sangsjangre, bør du velge 3 klynger. Men er det tilfelle?

1. Bruk 'albuemetoden' for å være sikker.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Bruk variabelen `wcss` som du bygde i forrige steg til å lage et diagram som viser hvor 'knekkpunktet' i albuen er, som indikerer det optimale antallet klynger. Kanskje det **er** 3!

    ![albuemetoden](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Øvelse - vis klyngene

1. Prøv prosessen igjen, denne gangen med tre klynger, og vis klyngene som et spredningsdiagram:

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

1. Sjekk modellens nøyaktighet:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Denne modellens nøyaktighet er ikke særlig god, og formen på klyngene gir deg et hint om hvorfor.

    ![klynger](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Disse dataene er for ubalanserte, for lite korrelerte, og det er for mye varians mellom kolonneverdiene til å klynge godt. Faktisk er klyngene som dannes sannsynligvis sterkt påvirket eller skjevfordelt av de tre sjangerkategoriene vi definerte ovenfor. Det var en læringsprosess!

    I Scikit-learns dokumentasjon kan du se at en modell som denne, med klynger som ikke er særlig godt avgrenset, har et 'varians'-problem:

    ![problematiske modeller](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografikk fra Scikit-learn

## Varians

Varians er definert som "gjennomsnittet av de kvadrerte forskjellene fra gjennomsnittet" [(Kilde)](https://www.mathsisfun.com/data/standard-deviation.html). I konteksten av dette klyngeproblemet refererer det til data der tallene i datasettet har en tendens til å avvike litt for mye fra gjennomsnittet.

✅ Dette er et godt tidspunkt å tenke på alle måtene du kan rette opp dette problemet. Justere dataene litt mer? Bruke andre kolonner? Bruke en annen algoritme? Hint: Prøv [å skalere dataene dine](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) for å normalisere dem og teste andre kolonner.

> Prøv denne '[varianskalkulatoren](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' for å forstå konseptet litt bedre.

---

## 🚀Utfordring

Bruk litt tid på denne notebooken og juster parametere. Kan du forbedre modellens nøyaktighet ved å rense dataene mer (for eksempel fjerne uteliggere)? Du kan bruke vekter for å gi mer vekt til visse datapunkter. Hva annet kan du gjøre for å lage bedre klynger?

Hint: Prøv å skalere dataene dine. Det er kommentert kode i notebooken som legger til standard skalering for å få datakolonnene til å ligne hverandre mer når det gjelder verdier. Du vil oppdage at selv om silhuett-scoren går ned, jevner 'knekkpunktet' i albuegrafen seg ut. Dette skyldes at å la dataene være uskalert gjør at data med mindre varians får mer vekt. Les mer om dette problemet [her](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Etter-forelesningsquiz](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Ta en titt på en K-Means-simulator [som denne](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Du kan bruke dette verktøyet til å visualisere prøvedatapunkter og bestemme midtpunktene deres. Du kan redigere dataens tilfeldighet, antall klynger og antall midtpunkter. Hjelper dette deg med å få en idé om hvordan dataene kan grupperes?

Ta også en titt på [dette handoutet om K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) fra Stanford.

## Oppgave

[Prøv forskjellige klyngemetoder](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.