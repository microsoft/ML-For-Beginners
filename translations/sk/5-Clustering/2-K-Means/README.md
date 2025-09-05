<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T15:46:29+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "sk"
}
-->
# K-Means zhlukovanie

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

V tejto lekcii sa naučíte, ako vytvárať zhluky pomocou Scikit-learn a datasetu nigérijskej hudby, ktorý ste importovali skôr. Pokryjeme základy K-Means pre zhlukovanie. Pamätajte, že ako ste sa naučili v predchádzajúcej lekcii, existuje mnoho spôsobov, ako pracovať so zhlukmi, a metóda, ktorú použijete, závisí od vašich dát. Skúsime K-Means, pretože je to najbežnejšia technika zhlukovania. Poďme na to!

Pojmy, o ktorých sa dozviete:

- Silhouette skóre
- Metóda lakťa
- Inercia
- Variancia

## Úvod

[K-Means zhlukovanie](https://wikipedia.org/wiki/K-means_clustering) je metóda odvodená z oblasti spracovania signálov. Používa sa na rozdelenie a rozčlenenie skupín dát do 'k' zhlukov pomocou série pozorovaní. Každé pozorovanie pracuje na zoskupení daného dátového bodu najbližšie k jeho najbližšiemu 'priemeru', alebo stredovému bodu zhluku.

Zhluky je možné vizualizovať ako [Voronoi diagramy](https://wikipedia.org/wiki/Voronoi_diagram), ktoré zahŕňajú bod (alebo 'semienko') a jeho zodpovedajúcu oblasť.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografika od [Jen Looper](https://twitter.com/jenlooper)

Proces K-Means zhlukovania [prebieha v trojstupňovom procese](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritmus vyberie k-počet stredových bodov vzorkovaním z datasetu. Potom cykluje:
    1. Priradí každú vzorku k najbližšiemu centroidu.
    2. Vytvorí nové centroidy vypočítaním priemernej hodnoty všetkých vzoriek priradených k predchádzajúcim centroidom.
    3. Potom vypočíta rozdiel medzi novými a starými centroidmi a opakuje, kým sa centroidy nestabilizujú.

Jednou nevýhodou používania K-Means je fakt, že budete musieť určiť 'k', teda počet centroidov. Našťastie metóda 'lakťa' pomáha odhadnúť dobrú počiatočnú hodnotu pre 'k'. Hneď si to vyskúšate.

## Predpoklad

Budete pracovať v súbore [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb), ktorý obsahuje import dát a predbežné čistenie, ktoré ste vykonali v poslednej lekcii.

## Cvičenie - príprava

Začnite tým, že sa znova pozriete na dáta piesní.

1. Vytvorte boxplot, zavolaním `boxplot()` pre každý stĺpec:

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

    Tieto dáta sú trochu hlučné: pozorovaním každého stĺpca ako boxplotu môžete vidieť odľahlé hodnoty.

    ![odľahlé hodnoty](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Môžete prejsť dataset a odstrániť tieto odľahlé hodnoty, ale to by spravilo dáta dosť minimálne.

1. Zatiaľ si vyberte, ktoré stĺpce použijete pre vaše cvičenie zhlukovania. Vyberte tie s podobnými rozsahmi a zakódujte stĺpec `artist_top_genre` ako numerické dáta:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Teraz musíte vybrať, koľko zhlukov chcete cieliť. Viete, že existujú 3 hudobné žánre, ktoré sme vyčlenili z datasetu, takže skúsme 3:

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

Vidíte vytlačené pole s predpovedanými zhlukmi (0, 1 alebo 2) pre každý riadok dataframe.

1. Použite toto pole na výpočet 'silhouette skóre':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette skóre

Hľadajte silhouette skóre bližšie k 1. Toto skóre sa pohybuje od -1 do 1, a ak je skóre 1, zhluk je hustý a dobre oddelený od ostatných zhlukov. Hodnota blízka 0 predstavuje prekrývajúce sa zhluky so vzorkami veľmi blízko rozhodovacej hranice susedných zhlukov. [(Zdroj)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Naše skóre je **.53**, takže je presne v strede. To naznačuje, že naše dáta nie sú obzvlášť vhodné pre tento typ zhlukovania, ale pokračujme.

### Cvičenie - vytvorte model

1. Importujte `KMeans` a začnite proces zhlukovania.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Tu je niekoľko častí, ktoré si zaslúžia vysvetlenie.

    > 🎓 range: Toto sú iterácie procesu zhlukovania.

    > 🎓 random_state: "Určuje generovanie náhodných čísel pre inicializáciu centroidov." [Zdroj](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "súčet štvorcov v rámci zhluku" meria štvorcovú priemernú vzdialenosť všetkých bodov v rámci zhluku od centroidu zhluku. [Zdroj](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inercia: Algoritmy K-Means sa snažia vybrať centroidy tak, aby minimalizovali 'inerciu', "mieru toho, ako sú zhluky vnútorne koherentné." [Zdroj](https://scikit-learn.org/stable/modules/clustering.html). Hodnota sa pridáva do premennej wcss pri každej iterácii.

    > 🎓 k-means++: V [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) môžete použiť optimalizáciu 'k-means++', ktorá "inicializuje centroidy tak, aby boli (zvyčajne) vzdialené od seba, čo vedie k pravdepodobne lepším výsledkom ako náhodná inicializácia."

### Metóda lakťa

Predtým ste predpokladali, že keďže ste cielili 3 hudobné žánre, mali by ste zvoliť 3 zhluky. Ale je to tak?

1. Použite metódu 'lakťa', aby ste si boli istí.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Použite premennú `wcss`, ktorú ste vytvorili v predchádzajúcom kroku, na vytvorenie grafu, ktorý ukazuje, kde je 'ohyb' v lakti, čo naznačuje optimálny počet zhlukov. Možno je to **naozaj** 3!

    ![metóda lakťa](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Cvičenie - zobrazte zhluky

1. Skúste proces znova, tentoraz nastavte tri zhluky a zobrazte zhluky ako scatterplot:

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

1. Skontrolujte presnosť modelu:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Presnosť tohto modelu nie je veľmi dobrá a tvar zhlukov vám dáva náznak prečo.

    ![zhluky](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Tieto dáta sú príliš nevyvážené, málo korelované a medzi hodnotami stĺpcov je príliš veľká variancia na to, aby sa dobre zhlukovali. V skutočnosti sú zhluky, ktoré sa tvoria, pravdepodobne silne ovplyvnené alebo skreslené tromi kategóriami žánrov, ktoré sme definovali vyššie. To bol proces učenia!

    V dokumentácii Scikit-learn môžete vidieť, že model ako tento, s nie veľmi dobre vyznačenými zhlukmi, má problém s 'varianciou':

    ![problémové modely](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografika zo Scikit-learn

## Variancia

Variancia je definovaná ako "priemer štvorcových rozdielov od priemeru" [(Zdroj)](https://www.mathsisfun.com/data/standard-deviation.html). V kontexte tohto problému zhlukovania sa vzťahuje na dáta, kde čísla nášho datasetu majú tendenciu odchýliť sa trochu príliš od priemeru.

✅ Toto je skvelý moment na zamyslenie sa nad všetkými spôsobmi, ako by ste mohli tento problém opraviť. Upraviť dáta trochu viac? Použiť iné stĺpce? Použiť iný algoritmus? Tip: Skúste [škálovať vaše dáta](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) na ich normalizáciu a otestovať iné stĺpce.

> Skúste tento '[kalkulátor variancie](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)', aby ste lepšie pochopili tento koncept.

---

## 🚀Výzva

Strávte nejaký čas s týmto notebookom, upravujte parametre. Dokážete zlepšiť presnosť modelu čistením dát (napríklad odstránením odľahlých hodnôt)? Môžete použiť váhy na pridanie väčšej váhy určitým vzorkám dát. Čo ešte môžete urobiť na vytvorenie lepších zhlukov?

Tip: Skúste škálovať vaše dáta. V notebooku je komentovaný kód, ktorý pridáva štandardné škálovanie, aby sa stĺpce dát viac podobali z hľadiska rozsahu. Zistíte, že zatiaľ čo silhouette skóre klesá, 'ohyb' v grafe lakťa sa vyhladzuje. Je to preto, že ponechanie dát neškálovaných umožňuje dátam s menšou varianciou niesť väčšiu váhu. Prečítajte si o tomto probléme [tu](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Pozrite sa na simulátor K-Means [ako je tento](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Môžete použiť tento nástroj na vizualizáciu vzoriek dátových bodov a určenie ich centroidov. Môžete upraviť náhodnosť dát, počet zhlukov a počet centroidov. Pomáha vám to získať predstavu o tom, ako môžu byť dáta zoskupené?

Tiež sa pozrite na [tento materiál o K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) zo Stanfordu.

## Zadanie

[Vyskúšajte rôzne metódy zhlukovania](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.