<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T00:06:37+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "fi"
}
-->
# K-Means-klusterointi

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Tässä oppitunnissa opit luomaan klustereita Scikit-learnin avulla ja käyttämällä aiemmin tuomaasi Nigerian musiikkidataa. Käymme läpi K-Meansin perusteet klusterointia varten. Muista, että kuten opit aiemmassa oppitunnissa, klustereiden kanssa työskentelyyn on monia tapoja, ja käyttämäsi menetelmä riippuu datastasi. Kokeilemme K-Meansia, koska se on yleisin klusterointitekniikka. Aloitetaan!

Termit, joista opit lisää:

- Silhouette-pisteytys
- Kyynärpäämenetelmä
- Inertia
- Varianssi

## Johdanto

[K-Means-klusterointi](https://wikipedia.org/wiki/K-means_clustering) on menetelmä, joka on peräisin signaalinkäsittelyn alalta. Sitä käytetään jakamaan ja ryhmittelemään dataa 'k' klusteriin havaintojen avulla. Jokainen havainto pyrkii ryhmittelemään tietyn datapisteen lähimpään 'keskiarvoon' eli klusterin keskipisteeseen.

Klusterit voidaan visualisoida [Voronoi-diagrammeina](https://wikipedia.org/wiki/Voronoi_diagram), jotka sisältävät pisteen (tai 'siemenen') ja sen vastaavan alueen.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

K-Means-klusterointiprosessi [etenee kolmivaiheisessa prosessissa](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritmi valitsee k-määrän keskipisteitä ottamalla näytteitä datasta. Tämän jälkeen se toistaa:
    1. Se määrittää jokaisen näytteen lähimpään keskipisteeseen.
    2. Se luo uusia keskipisteitä laskemalla kaikkien edellisiin keskipisteisiin määritettyjen näytteiden keskiarvon.
    3. Sitten se laskee eron uusien ja vanhojen keskipisteiden välillä ja toistaa, kunnes keskipisteet vakiintuvat.

Yksi K-Meansin käytön haittapuoli on se, että sinun täytyy määrittää 'k', eli keskipisteiden määrä. Onneksi 'kyynärpäämenetelmä' auttaa arvioimaan hyvän lähtöarvon 'k':lle. Kokeilet sitä pian.

## Esitiedot

Työskentelet tämän oppitunnin [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb)-tiedostossa, joka sisältää datan tuonnin ja alustavan puhdistuksen, jonka teit edellisessä oppitunnissa.

## Harjoitus - valmistelu

Aloita tarkastelemalla uudelleen kappaledataa.

1. Luo laatikkokaavio kutsumalla `boxplot()` jokaiselle sarakkeelle:

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

    Tämä data on hieman meluisaa: tarkastelemalla kutakin saraketta laatikkokaaviona voit nähdä poikkeamia.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Voisit käydä datasetin läpi ja poistaa nämä poikkeamat, mutta se tekisi datasta melko vähäistä.

1. Valitse toistaiseksi, mitkä sarakkeet käytät klusterointiharjoituksessa. Valitse sarakkeet, joilla on samanlaiset vaihteluvälit, ja koodaa `artist_top_genre`-sarake numeeriseksi dataksi:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Nyt sinun täytyy päättää, kuinka monta klusteria kohdistat. Tiedät, että datasetistä on erotettu kolme kappaletyyliä, joten kokeillaan kolmea:

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

Näet tulostettuna taulukon, jossa on ennustetut klusterit (0, 1 tai 2) jokaiselle dataframe-riville.

1. Käytä tätä taulukkoa laskeaksesi 'silhouette-pisteen':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette-pisteytys

Etsi silhouette-piste, joka on lähempänä 1:tä. Tämä piste vaihtelee -1:stä 1:een, ja jos piste on 1, klusteri on tiivis ja hyvin erotettu muista klustereista. Arvo lähellä 0:aa edustaa päällekkäisiä klustereita, joissa näytteet ovat hyvin lähellä naapuriklusterien päätösrajaa. [(Lähde)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Meidän pisteemme on **.53**, eli keskivaiheilla. Tämä osoittaa, että datamme ei ole erityisen hyvin soveltuvaa tämän tyyppiseen klusterointiin, mutta jatketaan.

### Harjoitus - mallin rakentaminen

1. Tuo `KMeans` ja aloita klusterointiprosessi.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Tässä on muutama osa, jotka vaativat selitystä.

    > 🎓 range: Nämä ovat klusterointiprosessin iteroinnit.

    > 🎓 random_state: "Määrittää satunnaislukugeneraation keskipisteiden alustamiseen." [Lähde](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "klusterin sisäiset neliösummat" mittaa kaikkien klusterin sisällä olevien pisteiden keskipisteeseen kohdistuvan keskimääräisen etäisyyden neliön. [Lähde](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inertia: K-Means-algoritmit pyrkivät valitsemaan keskipisteet minimoimaan 'inertiaa', "mitta, kuinka sisäisesti yhtenäisiä klusterit ovat." [Lähde](https://scikit-learn.org/stable/modules/clustering.html). Arvo lisätään wcss-muuttujaan jokaisella iteroinnilla.

    > 🎓 k-means++: [Scikit-learnissa](https://scikit-learn.org/stable/modules/clustering.html#k-means) voit käyttää 'k-means++'-optimointia, joka "alustaa keskipisteet olemaan (yleensä) kaukana toisistaan, mikä johtaa todennäköisesti parempiin tuloksiin kuin satunnainen alustus."

### Kyynärpäämenetelmä

Aiemmin oletit, että koska olet kohdistanut kolme kappaletyyliä, sinun pitäisi valita kolme klusteria. Mutta onko näin?

1. Käytä 'kyynärpäämenetelmää' varmistaaksesi.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Käytä `wcss`-muuttujaa, jonka rakensit edellisessä vaiheessa, luodaksesi kaavion, joka näyttää, missä kyynärpään "taite" on, mikä osoittaa optimaalisen klusterien määrän. Ehkä se **onkin** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Harjoitus - klustereiden näyttäminen

1. Kokeile prosessia uudelleen, tällä kertaa asettamalla kolme klusteria, ja näytä klusterit hajontakaaviona:

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

1. Tarkista mallin tarkkuus:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Tämän mallin tarkkuus ei ole kovin hyvä, ja klustereiden muoto antaa vihjeen miksi.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Tämä data on liian epätasapainoista, liian vähän korreloitua ja sarakearvojen välillä on liikaa vaihtelua, jotta klusterointi onnistuisi hyvin. Itse asiassa muodostuvat klusterit ovat todennäköisesti vahvasti vaikuttuneita tai vinoutuneita yllä määrittelemiemme kolmen tyylilajin mukaan. Tämä oli oppimisprosessi!

    Scikit-learnin dokumentaatiosta voit nähdä, että tällaisella mallilla, jossa klusterit eivät ole kovin hyvin rajattuja, on 'varianssi'-ongelma:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografiikka: Scikit-learn

## Varianssi

Varianssi määritellään "keskiarvona neliöllisistä eroista keskiarvosta" [(Lähde)](https://www.mathsisfun.com/data/standard-deviation.html). Tässä klusterointiongelman kontekstissa se viittaa dataan, jossa datasetin numerot poikkeavat hieman liikaa keskiarvosta.

✅ Tämä on hyvä hetki miettiä kaikkia tapoja, joilla voisit korjata tämän ongelman. Voisitko muokata dataa hieman enemmän? Käyttää eri sarakkeita? Käyttää eri algoritmia? Vinkki: Kokeile [skaalata dataasi](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) normalisoidaksesi sen ja testataksesi muita sarakkeita.

> Kokeile tätä '[varianssilaskuria](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' ymmärtääksesi konseptia hieman paremmin.

---

## 🚀Haaste

Käytä aikaa tämän notebookin parissa ja säädä parametreja. Voitko parantaa mallin tarkkuutta puhdistamalla dataa enemmän (esimerkiksi poistamalla poikkeamat)? Voit käyttää painotuksia antaaksesi enemmän painoarvoa tietyille datanäytteille. Mitä muuta voisit tehdä luodaksesi parempia klustereita?

Vinkki: Kokeile skaalata dataasi. Notebookissa on kommentoitua koodia, joka lisää standardisoinnin, jotta datan sarakkeet muistuttaisivat toisiaan enemmän vaihteluvälin osalta. Huomaat, että vaikka silhouette-piste laskee, kyynärpääkaavion "taite" tasoittuu. Tämä johtuu siitä, että jättämällä datan skaalaamatta, data, jolla on vähemmän varianssia, saa enemmän painoarvoa. Lue lisää tästä ongelmasta [täältä](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Tutustu K-Means-simulaattoriin [kuten tähän](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Voit käyttää tätä työkalua visualisoidaksesi näytepisteitä ja määrittääksesi niiden keskipisteet. Voit muokata datan satunnaisuutta, klusterien määrää ja keskipisteiden määrää. Auttaako tämä sinua saamaan käsityksen siitä, miten data voidaan ryhmitellä?

Tutustu myös [tähän K-Means-materiaaliin](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) Stanfordilta.

## Tehtävä

[Kokeile eri klusterointimenetelmiä](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.