<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T15:46:59+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "ro"
}
-->
# Gruparea K-Means

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

În această lecție, vei învăța cum să creezi grupuri folosind Scikit-learn și setul de date despre muzica nigeriană pe care l-ai importat anterior. Vom acoperi elementele de bază ale K-Means pentru grupare. Ține minte că, așa cum ai învățat în lecția anterioară, există multe moduri de a lucra cu grupuri, iar metoda pe care o alegi depinde de datele tale. Vom încerca K-Means, deoarece este cea mai comună tehnică de grupare. Să începem!

Termeni pe care îi vei învăța:

- Scorul siluetei
- Metoda cotului
- Inerție
- Variație

## Introducere

[Gruparea K-Means](https://wikipedia.org/wiki/K-means_clustering) este o metodă derivată din domeniul procesării semnalelor. Este utilizată pentru a împărți și partitiona grupuri de date în 'k' grupuri folosind o serie de observații. Fiecare observație funcționează pentru a grupa un punct de date dat cât mai aproape de 'media' sa cea mai apropiată, sau punctul central al unui grup.

Grupurile pot fi vizualizate ca [diagrame Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), care includ un punct (sau 'sămânță') și regiunea corespunzătoare acestuia.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> infografic de [Jen Looper](https://twitter.com/jenlooper)

Procesul de grupare K-Means [se execută în trei pași](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritmul selectează un număr k de puncte centrale prin eșantionare din setul de date. După aceasta, se repetă:
    1. Atribuie fiecare eșantion celui mai apropiat centroid.
    2. Creează noi centroizi calculând valoarea medie a tuturor eșantioanelor atribuite centroidelor anterioare.
    3. Apoi, calculează diferența dintre noii și vechii centroizi și repetă până când centroizii se stabilizează.

Un dezavantaj al utilizării K-Means este faptul că trebuie să stabilești 'k', adică numărul de centroizi. Din fericire, 'metoda cotului' ajută la estimarea unei valori bune de început pentru 'k'. Vei încerca acest lucru în curând.

## Prerechizite

Vei lucra în fișierul [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) al acestei lecții, care include importul de date și curățarea preliminară pe care ai făcut-o în lecția anterioară.

## Exercițiu - pregătire

Începe prin a analiza din nou datele despre melodii.

1. Creează un boxplot, apelând `boxplot()` pentru fiecare coloană:

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

    Aceste date sunt puțin zgomotoase: observând fiecare coloană ca boxplot, poți vedea valori extreme.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Ai putea parcurge setul de date și elimina aceste valori extreme, dar acest lucru ar face datele destul de minimale.

1. Deocamdată, alege coloanele pe care le vei folosi pentru exercițiul de grupare. Selectează coloane cu intervale similare și codifică coloana `artist_top_genre` ca date numerice:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Acum trebuie să alegi câte grupuri să vizezi. Știi că există 3 genuri muzicale pe care le-am extras din setul de date, așa că să încercăm cu 3:

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

Vei vedea un array afișat cu grupurile prezise (0, 1 sau 2) pentru fiecare rând din dataframe.

1. Folosește acest array pentru a calcula un 'scor al siluetei':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Scorul siluetei

Caută un scor al siluetei mai aproape de 1. Acest scor variază de la -1 la 1, iar dacă scorul este 1, grupul este dens și bine separat de celelalte grupuri. O valoare apropiată de 0 reprezintă grupuri suprapuse, cu eșantioane foarte apropiate de limita de decizie a grupurilor vecine. [(Sursă)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Scorul nostru este **.53**, deci chiar la mijloc. Acest lucru indică faptul că datele noastre nu sunt deosebit de potrivite pentru acest tip de grupare, dar să continuăm.

### Exercițiu - construiește un model

1. Importă `KMeans` și începe procesul de grupare.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Există câteva părți aici care merită explicate.

    > 🎓 range: Acestea sunt iterațiile procesului de grupare.

    > 🎓 random_state: "Determină generarea de numere aleatoare pentru inițializarea centroizilor." [Sursă](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "suma pătratelor din interiorul grupurilor" măsoară distanța medie pătrată a tuturor punctelor dintr-un grup față de centrul grupului. [Sursă](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inerție: Algoritmii K-Means încearcă să aleagă centroizi pentru a minimiza 'inerția', "o măsură a cât de coerente sunt grupurile intern." [Sursă](https://scikit-learn.org/stable/modules/clustering.html). Valoarea este adăugată la variabila wcss la fiecare iterație.

    > 🎓 k-means++: În [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) poți folosi optimizarea 'k-means++', care "inițializează centroizii pentru a fi (în general) distanțați unul de celălalt, conducând probabil la rezultate mai bune decât inițializarea aleatorie."

### Metoda cotului

Anterior, ai presupus că, deoarece ai vizat 3 genuri muzicale, ar trebui să alegi 3 grupuri. Dar este acesta cazul?

1. Folosește 'metoda cotului' pentru a te asigura.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Folosește variabila `wcss` pe care ai construit-o în pasul anterior pentru a crea un grafic care arată unde este 'cotul', ceea ce indică numărul optim de grupuri. Poate chiar **este** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Exercițiu - afișează grupurile

1. Încearcă procesul din nou, de data aceasta setând trei grupuri și afișează grupurile ca un scatterplot:

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

1. Verifică acuratețea modelului:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Acuratețea acestui model nu este foarte bună, iar forma grupurilor îți oferă un indiciu de ce.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Aceste date sunt prea dezechilibrate, prea puțin corelate și există prea multă variație între valorile coloanelor pentru a grupa bine. De fapt, grupurile care se formează sunt probabil influențate sau distorsionate puternic de cele trei categorii de genuri pe care le-am definit mai sus. A fost un proces de învățare!

    În documentația Scikit-learn, poți vedea că un model ca acesta, cu grupuri care nu sunt foarte bine delimitate, are o problemă de 'variație':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografic din Scikit-learn

## Variație

Variația este definită ca "media diferențelor pătrate față de medie" [(Sursă)](https://www.mathsisfun.com/data/standard-deviation.html). În contextul acestei probleme de grupare, se referă la datele ale căror valori tind să se abată prea mult de la medie.

✅ Acesta este un moment excelent pentru a te gândi la toate modurile în care ai putea corecta această problemă. Ajustezi datele puțin mai mult? Folosești alte coloane? Utilizezi un alt algoritm? Sugestie: Încearcă [scalarea datelor](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) pentru a le normaliza și testează alte coloane.

> Încearcă acest '[calculator de variație](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' pentru a înțelege mai bine conceptul.

---

## 🚀Provocare

Petrece ceva timp cu acest notebook, ajustând parametrii. Poți îmbunătăți acuratețea modelului prin curățarea mai bună a datelor (eliminând valorile extreme, de exemplu)? Poți folosi ponderi pentru a acorda mai multă greutate anumitor eșantioane de date. Ce altceva poți face pentru a crea grupuri mai bune?

Sugestie: Încearcă să scalezi datele. Există cod comentat în notebook care adaugă scalarea standard pentru a face ca valorile coloanelor să semene mai mult între ele în termeni de interval. Vei descoperi că, deși scorul siluetei scade, 'cotul' din graficul cotului se netezește. Acest lucru se întâmplă deoarece lăsarea datelor nescalate permite datelor cu mai puțină variație să aibă mai multă greutate. Citește mai multe despre această problemă [aici](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Aruncă o privire la un simulator K-Means [cum ar fi acesta](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Poți folosi acest instrument pentru a vizualiza punctele de date eșantion și pentru a determina centroizii acestora. Poți edita aleatoritatea datelor, numărul de grupuri și numărul de centroizi. Te ajută acest lucru să îți faci o idee despre cum pot fi grupate datele?

De asemenea, aruncă o privire la [acest material despre K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) de la Stanford.

## Temă

[Încearcă metode diferite de grupare](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.