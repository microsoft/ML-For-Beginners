<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T00:05:07+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "cs"
}
-->
# K-Means clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

V této lekci se naučíte, jak vytvářet klastery pomocí Scikit-learn a nigerijského hudebního datasetu, který jste importovali dříve. Probereme základy K-Means pro klastrování. Mějte na paměti, že jak jste se naučili v předchozí lekci, existuje mnoho způsobů, jak pracovat s klastery, a metoda, kterou použijete, závisí na vašich datech. Vyzkoušíme K-Means, protože je to nejběžnější technika klastrování. Pojďme začít!

Pojmy, o kterých se dozvíte:

- Silhouette skóre
- Metoda lokte
- Inerciální hodnota
- Variance

## Úvod

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) je metoda odvozená z oblasti zpracování signálů. Používá se k rozdělení a seskupení dat do 'k' klastrů pomocí série pozorování. Každé pozorování pracuje na seskupení daného datového bodu k nejbližšímu 'průměru', tedy středovému bodu klastru.

Klastery lze vizualizovat jako [Voronoi diagramy](https://wikipedia.org/wiki/Voronoi_diagram), které zahrnují bod (nebo 'semínko') a jeho odpovídající oblast.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografika od [Jen Looper](https://twitter.com/jenlooper)

Proces K-Means klastrování [probíhá ve třech krocích](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritmus vybere k-počet středových bodů vzorkováním z datasetu. Poté opakuje:
    1. Přiřadí každý vzorek k nejbližšímu centroidu.
    2. Vytvoří nové centroidy vypočítáním průměrné hodnoty všech vzorků přiřazených k předchozím centroidům.
    3. Poté vypočítá rozdíl mezi novými a starými centroidy a opakuje, dokud se centroidy nestabilizují.

Jednou z nevýhod použití K-Means je nutnost stanovit 'k', tedy počet centroidů. Naštěstí metoda 'lokte' pomáhá odhadnout dobrý výchozí počet 'k'. Za chvíli si ji vyzkoušíte.

## Předpoklady

Budete pracovat v souboru [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb), který obsahuje import dat a předběžné čištění, které jste provedli v minulé lekci.

## Cvičení - příprava

Začněte tím, že se znovu podíváte na data o písních.

1. Vytvořte boxplot, zavolejte `boxplot()` pro každý sloupec:

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

    Tato data jsou trochu hlučná: při pozorování každého sloupce jako boxplotu můžete vidět odlehlé hodnoty.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Můžete projít dataset a odstranit tyto odlehlé hodnoty, ale to by data značně zredukovalo.

1. Prozatím vyberte, které sloupce použijete pro své cvičení klastrování. Vyberte ty s podobnými rozsahy a zakódujte sloupec `artist_top_genre` jako číselná data:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Nyní musíte vybrat, kolik klastrů budete cílit. Víte, že existují 3 hudební žánry, které jsme vyčlenili z datasetu, takže zkusme 3:

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

Vidíte vytištěné pole s předpovězenými klastery (0, 1 nebo 2) pro každý řádek datového rámce.

1. Použijte toto pole k výpočtu 'silhouette skóre':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette skóre

Hledejte silhouette skóre blíže k 1. Toto skóre se pohybuje od -1 do 1, a pokud je skóre 1, klastr je hustý a dobře oddělený od ostatních klastrů. Hodnota blízko 0 představuje překrývající se klastery s vzorky velmi blízko rozhodovací hranice sousedních klastrů. [(Zdroj)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Naše skóre je **.53**, tedy přímo uprostřed. To naznačuje, že naše data nejsou pro tento typ klastrování příliš vhodná, ale pokračujme.

### Cvičení - vytvoření modelu

1. Importujte `KMeans` a začněte proces klastrování.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Několik částí zde si zaslouží vysvětlení.

    > 🎓 range: Toto jsou iterace procesu klastrování.

    > 🎓 random_state: "Určuje generování náhodných čísel pro inicializaci centroidů." [Zdroj](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "součet čtverců uvnitř klastrů" měří průměrnou čtvercovou vzdálenost všech bodů v rámci klastru od centroidu klastru. [Zdroj](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inerciální hodnota: Algoritmy K-Means se snaží vybrat centroidy tak, aby minimalizovaly 'inerciální hodnotu', "měřítko toho, jak jsou klastery interně koherentní." [Zdroj](https://scikit-learn.org/stable/modules/clustering.html). Hodnota je připojena k proměnné wcss při každé iteraci.

    > 🎓 k-means++: V [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) můžete použít optimalizaci 'k-means++', která "inicializuje centroidy tak, aby byly (obecně) vzdálené od sebe, což vede pravděpodobně k lepším výsledkům než náhodná inicializace."

### Metoda lokte

Dříve jste předpokládali, že protože jste cílovali 3 hudební žánry, měli byste zvolit 3 klastery. Ale je tomu tak?

1. Použijte metodu 'lokte', abyste si byli jistí.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Použijte proměnnou `wcss`, kterou jste vytvořili v předchozím kroku, k vytvoření grafu ukazujícího, kde je 'ohyb' v lokti, což naznačuje optimální počet klastrů. Možná to **opravdu jsou** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Cvičení - zobrazení klastrů

1. Zkuste proces znovu, tentokrát nastavte tři klastery a zobrazte klastery jako scatterplot:

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

1. Zkontrolujte přesnost modelu:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Přesnost tohoto modelu není příliš dobrá a tvar klastrů vám naznačuje proč.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Tato data jsou příliš nevyvážená, málo korelovaná a mezi hodnotami sloupců je příliš velká variance na to, aby se dobře klastrovala. Ve skutečnosti jsou klastery, které se tvoří, pravděpodobně silně ovlivněny nebo zkresleny třemi kategoriemi žánrů, které jsme definovali výše. To byl proces učení!

    V dokumentaci Scikit-learn můžete vidět, že model jako tento, s klastery, které nejsou příliš dobře vymezené, má problém s 'variancí':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografika ze Scikit-learn

## Variance

Variance je definována jako "průměr čtvercových rozdílů od průměru" [(Zdroj)](https://www.mathsisfun.com/data/standard-deviation.html). V kontextu tohoto problému klastrování se jedná o data, kde čísla našeho datasetu mají tendenci se příliš odchylovat od průměru.

✅ Toto je skvělý moment k zamyšlení nad všemi způsoby, jak byste mohli tento problém napravit. Upravit data trochu více? Použít jiné sloupce? Použít jiný algoritmus? Tip: Zkuste [škálovat svá data](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) pro jejich normalizaci a otestujte jiné sloupce.

> Vyzkoušejte tento '[kalkulátor variance](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)', abyste lépe pochopili tento koncept.

---

## 🚀Výzva

Stravte nějaký čas s tímto notebookem a upravujte parametry. Dokážete zlepšit přesnost modelu tím, že data více vyčistíte (například odstraníte odlehlé hodnoty)? Můžete použít váhy, abyste dali větší váhu určitým vzorkům dat. Co dalšího můžete udělat pro vytvoření lepších klastrů?

Tip: Zkuste škálovat svá data. V notebooku je komentovaný kód, který přidává standardní škálování, aby se sloupce dat více podobaly z hlediska rozsahu. Zjistíte, že zatímco silhouette skóre klesá, 'ohyb' v grafu lokte se vyhlazuje. To je proto, že ponechání dat neškálovaných umožňuje datům s menší variancí mít větší váhu. Přečtěte si o tomto problému více [zde](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Podívejte se na simulátor K-Means [jako je tento](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Můžete použít tento nástroj k vizualizaci vzorových datových bodů a určení jejich centroidů. Můžete upravit náhodnost dat, počet klastrů a počet centroidů. Pomáhá vám to získat představu o tom, jak lze data seskupit?

Také se podívejte na [tento materiál o K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) ze Stanfordu.

## Zadání

[Vyzkoušejte různé metody klastrování](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.