<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T15:45:56+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "hu"
}
-->
# K-Means klaszterezés

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

Ebben a leckében megtanulod, hogyan hozz létre klasztereket a Scikit-learn és az előzőekben importált nigériai zenei adatállomány segítségével. Áttekintjük a K-Means alapjait a klaszterezéshez. Ne feledd, ahogy az előző leckében tanultad, számos módja van a klaszterekkel való munkának, és az alkalmazott módszer az adataidtól függ. Kipróbáljuk a K-Means-t, mivel ez a leggyakoribb klaszterezési technika. Kezdjük!

Fogalmak, amelyeket megismerhetsz:

- Silhouette pontszám
- Könyökmódszer
- Inertia
- Variancia

## Bevezetés

A [K-Means klaszterezés](https://wikipedia.org/wiki/K-means_clustering) a jelfeldolgozás területéről származó módszer. Arra használják, hogy az adatokat 'k' klaszterekbe osszák és csoportosítsák megfigyelések sorozata alapján. Minden megfigyelés arra törekszik, hogy az adott adatpontot a legközelebbi 'átlaghoz', vagyis egy klaszter középpontjához csoportosítsa.

A klaszterek [Voronoi diagramokként](https://wikipedia.org/wiki/Voronoi_diagram) is megjeleníthetők, amelyek tartalmaznak egy pontot (vagy 'magot') és annak megfelelő régióját.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografika: [Jen Looper](https://twitter.com/jenlooper)

A K-Means klaszterezési folyamat [három lépésben zajlik](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Az algoritmus kiválasztja az adatállományból k számú középpontot mintavételezéssel. Ezután ismétlődően:
    1. Minden mintát hozzárendel a legközelebbi centroidhoz.
    2. Új centroidokat hoz létre az előző centroidokhoz rendelt minták átlagértéke alapján.
    3. Ezután kiszámítja az új és régi centroidok közötti különbséget, és addig ismétli, amíg a centroidok stabilizálódnak.

A K-Means használatának egyik hátránya, hogy meg kell határoznod 'k'-t, azaz a centroidok számát. Szerencsére a 'könyökmódszer' segít egy jó kiindulási érték becslésében 'k'-hoz. Mindjárt kipróbáljuk.

## Előfeltétel

Ebben a lecke [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) fájljában fogsz dolgozni, amely tartalmazza az előző leckében végzett adatimportálást és előzetes tisztítást.

## Gyakorlat - előkészítés

Kezdd azzal, hogy újra megnézed a dalok adatait.

1. Hozz létre egy boxplotot, és hívd meg a `boxplot()` függvényt minden oszlopra:

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

    Ez az adat kissé zajos: az egyes oszlopok boxplotjait megfigyelve láthatod a kiugró értékeket.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Az adatállományt átnézve eltávolíthatnád ezeket a kiugró értékeket, de ez az adatokat eléggé minimalizálná.

1. Egyelőre válaszd ki, mely oszlopokat fogod használni a klaszterezési gyakorlatban. Válassz olyanokat, amelyek hasonló tartományokkal rendelkeznek, és kódold az `artist_top_genre` oszlopot numerikus adatként:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Most meg kell határoznod, hány klasztert célozz meg. Tudod, hogy az adatállományból 3 zenei műfajt választottunk ki, így próbáljuk meg a 3-at:

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

Egy tömböt látsz, amely az adatkeret minden sorára előrejelzett klasztereket (0, 1 vagy 2) tartalmaz.

1. Használd ezt a tömböt egy 'silhouette pontszám' kiszámításához:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette pontszám

Keress egy silhouette pontszámot, amely közelebb van az 1-hez. Ez a pontszám -1 és 1 között változik, és ha a pontszám 1, akkor a klaszter sűrű és jól elkülönül a többi klasztertől. A 0-hoz közeli érték átfedő klasztereket jelöl, ahol a minták nagyon közel vannak a szomszédos klaszterek döntési határához. [(Forrás)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

A pontszámunk **.53**, tehát középen van. Ez azt jelzi, hogy az adataink nem különösebben alkalmasak erre a klaszterezési típusra, de folytassuk.

### Gyakorlat - modell építése

1. Importáld a `KMeans`-t, és kezdj bele a klaszterezési folyamatba.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Néhány részlet magyarázatra szorul.

    > 🎓 range: Ezek a klaszterezési folyamat iterációi.

    > 🎓 random_state: "Meghatározza a véletlenszám-generálást a centroid inicializálásához." [Forrás](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "a klaszteren belüli négyzetes összeg" méri az összes pont átlagos négyzetes távolságát a klaszter centroidjától. [Forrás](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inertia: A K-Means algoritmusok arra törekednek, hogy olyan centroidokat válasszanak, amelyek minimalizálják az 'inertia'-t, "a klaszterek belső koherenciájának mértékét." [Forrás](https://scikit-learn.org/stable/modules/clustering.html). Az értéket minden iteráció során hozzáadjuk a wcss változóhoz.

    > 🎓 k-means++: A [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) használatával alkalmazhatod a 'k-means++' optimalizálást, amely "általában egymástól távoli centroidokat inicializál, valószínűleg jobb eredményeket eredményezve, mint a véletlenszerű inicializálás."

### Könyökmódszer

Korábban feltételezted, hogy mivel 3 zenei műfajt céloztál meg, 3 klasztert kell választanod. De valóban így van?

1. Használd a 'könyökmódszert', hogy megbizonyosodj róla.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Használd a korábbi lépésben létrehozott `wcss` változót, hogy készíts egy diagramot, amely megmutatja, hol van a 'kanyar' a könyökben, ami a klaszterek optimális számát jelzi. Talán tényleg **3**!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Gyakorlat - klaszterek megjelenítése

1. Próbáld újra a folyamatot, ezúttal három klasztert beállítva, és jelenítsd meg a klasztereket szórásdiagramként:

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

1. Ellenőrizd a modell pontosságát:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Ennek a modellnek a pontossága nem túl jó, és a klaszterek alakja ad egy tippet, hogy miért.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Ezek az adatok túl kiegyensúlyozatlanok, túl kevéssé korreláltak, és az oszlopértékek között túl nagy a variancia ahhoz, hogy jól klaszterezhetők legyenek. Valójában az általunk meghatározott három műfajkategória valószínűleg erősen befolyásolja vagy torzítja a kialakuló klasztereket. Ez egy tanulási folyamat volt!

    A Scikit-learn dokumentációjában láthatod, hogy egy ilyen modell, ahol a klaszterek nem nagyon jól elkülönültek, 'variancia' problémával küzd:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografika: Scikit-learn

## Variancia

A variancia úgy definiálható, mint "az átlagtól való négyzetes eltérések átlaga" [(Forrás)](https://www.mathsisfun.com/data/standard-deviation.html). Ebben a klaszterezési problémában arra utal, hogy az adatállomány számai hajlamosak túlzottan eltérni az átlagtól.

✅ Ez egy remek pillanat arra, hogy átgondold, milyen módokon javíthatnád ezt a problémát. Finomítsd az adatokat? Használj más oszlopokat? Próbálj ki egy másik algoritmust? Tipp: Próbáld meg [normalizálni az adatokat](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) és tesztelj más oszlopokat.

> Próbáld ki ezt a '[variancia kalkulátort](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)', hogy jobban megértsd a fogalmat.

---

## 🚀Kihívás

Tölts el egy kis időt ezzel a notebookkal, és finomítsd a paramétereket. Javíthatod-e a modell pontosságát az adatok további tisztításával (például a kiugró értékek eltávolításával)? Használhatsz súlyokat, hogy bizonyos adatmintáknak nagyobb súlyt adj. Mit tehetsz még a jobb klaszterek létrehozása érdekében?

Tipp: Próbáld meg skálázni az adatokat. A notebookban van kommentált kód, amely hozzáadja a standard skálázást, hogy az adatállomány oszlopai jobban hasonlítsanak egymásra tartomány szempontjából. Meg fogod látni, hogy bár a silhouette pontszám csökken, a könyök grafikon 'kanyarja' kisimul. Ez azért van, mert az adatok skálázatlanul hagyása lehetővé teszi, hogy a kisebb varianciájú adatok nagyobb súlyt kapjanak. Olvass többet erről a problémáról [itt](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Nézd meg egy K-Means szimulátort [például ezt](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Ezzel az eszközzel vizualizálhatod a mintapontokat és meghatározhatod a centroidokat. Szerkesztheted az adatok véletlenszerűségét, a klaszterek számát és a centroidok számát. Segít ez abban, hogy jobban megértsd, hogyan csoportosíthatók az adatok?

Nézd meg [ezt a K-Means segédanyagot](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) a Stanfordtól.

## Feladat

[Próbálj ki különböző klaszterezési módszereket](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.