<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T12:18:40+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "hr"
}
-->
# K-Means klasteriranje

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

U ovoj lekciji naučit ćete kako kreirati klastere koristeći Scikit-learn i nigerijski glazbeni dataset koji ste ranije uvezli. Pokrit ćemo osnove K-Means metode za klasteriranje. Imajte na umu da, kao što ste naučili u prethodnoj lekciji, postoji mnogo načina za rad s klasterima, a metoda koju koristite ovisi o vašim podacima. Isprobat ćemo K-Means jer je to najčešće korištena tehnika klasteriranja. Krenimo!

Pojmovi koje ćete naučiti:

- Silhouette ocjenjivanje
- Metoda lakta
- Inercija
- Varijanca

## Uvod

[K-Means klasteriranje](https://wikipedia.org/wiki/K-means_clustering) je metoda koja potječe iz područja obrade signala. Koristi se za podjelu i grupiranje podataka u 'k' klastera pomoću niza opažanja. Svako opažanje radi na grupiranju određenog podatka najbližem njegovom 'prosjeku', odnosno središnjoj točki klastera.

Klasteri se mogu vizualizirati kao [Voronoi dijagrami](https://wikipedia.org/wiki/Voronoi_diagram), koji uključuju točku (ili 'sjeme') i njezinu odgovarajuću regiju.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografika od [Jen Looper](https://twitter.com/jenlooper)

Proces K-Means klasteriranja [izvodi se u tri koraka](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritam odabire k-broj središnjih točaka uzorkovanjem iz skupa podataka. Nakon toga, ponavlja:
    1. Dodjeljuje svaki uzorak najbližem centroidu.
    2. Stvara nove centre uzimajući prosječnu vrijednost svih uzoraka dodijeljenih prethodnim centrima.
    3. Zatim izračunava razliku između novih i starih centara i ponavlja dok se centri ne stabiliziraju.

Jedan nedostatak korištenja K-Means metode je činjenica da morate odrediti 'k', odnosno broj centara. Srećom, 'metoda lakta' pomaže procijeniti dobru početnu vrijednost za 'k'. Uskoro ćete je isprobati.

## Preduvjeti

Radit ćete u datoteci [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) koja uključuje uvoz podataka i preliminarno čišćenje koje ste obavili u prethodnoj lekciji.

## Vježba - priprema

Započnite tako da ponovno pogledate podatke o pjesmama.

1. Kreirajte boxplot pozivajući `boxplot()` za svaki stupac:

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

    Ovi podaci su malo bučni: promatrajući svaki stupac kao boxplot, možete vidjeti outliere.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Možete proći kroz skup podataka i ukloniti ove outliere, ali to bi učinilo podatke prilično minimalnima.

1. Za sada odaberite koje stupce ćete koristiti za vježbu klasteriranja. Odaberite one sa sličnim rasponima i kodirajte stupac `artist_top_genre` kao numeričke podatke:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Sada trebate odabrati koliko klastera ciljati. Znate da postoje 3 glazbena žanra koja smo izdvojili iz skupa podataka, pa pokušajmo s 3:

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

Vidite ispisan niz s predviđenim klasterima (0, 1 ili 2) za svaki redak dataframea.

1. Koristite ovaj niz za izračunavanje 'silhouette ocjene':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette ocjena

Tražite silhouette ocjenu bližu 1. Ova ocjena varira od -1 do 1, a ako je ocjena 1, klaster je gust i dobro odvojen od drugih klastera. Vrijednost blizu 0 predstavlja preklapajuće klastere s uzorcima vrlo blizu granice odluke susjednih klastera. [(Izvor)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Naša ocjena je **.53**, dakle točno u sredini. To ukazuje da naši podaci nisu osobito prikladni za ovu vrstu klasteriranja, ali nastavimo.

### Vježba - izgradnja modela

1. Uvezite `KMeans` i započnite proces klasteriranja.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Postoji nekoliko dijelova koji zaslužuju objašnjenje.

    > 🎓 range: Ovo su iteracije procesa klasteriranja

    > 🎓 random_state: "Određuje generiranje slučajnih brojeva za inicijalizaciju centara." [Izvor](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "sume kvadrata unutar klastera" mjere prosječnu kvadratnu udaljenost svih točaka unutar klastera do centra klastera. [Izvor](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Inercija: K-Means algoritmi pokušavaju odabrati centre kako bi minimizirali 'inerciju', "mjeru koliko su klasteri interno koherentni." [Izvor](https://scikit-learn.org/stable/modules/clustering.html). Vrijednost se dodaje varijabli wcss pri svakoj iteraciji.

    > 🎓 k-means++: U [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) možete koristiti 'k-means++' optimizaciju, koja "inicijalizira centre tako da budu (općenito) udaljeni jedni od drugih, što dovodi do vjerojatno boljih rezultata od slučajne inicijalizacije.

### Metoda lakta

Ranije ste pretpostavili da, budući da ste ciljali 3 glazbena žanra, trebate odabrati 3 klastera. Ali je li to slučaj?

1. Koristite 'metodu lakta' da budete sigurni.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Koristite varijablu `wcss` koju ste izgradili u prethodnom koraku za kreiranje grafikona koji pokazuje gdje je 'savijanje' u laktu, što ukazuje na optimalan broj klastera. Možda je to **zaista** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Vježba - prikaz klastera

1. Pokušajte ponovno proces, ovaj put postavljajući tri klastera, i prikažite klastere kao scatterplot:

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

1. Provjerite točnost modela:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Točnost ovog modela nije baš dobra, a oblik klastera daje vam nagovještaj zašto.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Ovi podaci su previše neuravnoteženi, premalo korelirani i postoji prevelika varijanca između vrijednosti stupaca da bi se dobro klasterirali. Zapravo, klasteri koji se formiraju vjerojatno su jako pod utjecajem ili iskrivljeni zbog tri kategorije žanrova koje smo gore definirali. To je bio proces učenja!

    U dokumentaciji Scikit-learn možete vidjeti da model poput ovog, s klasterima koji nisu dobro razgraničeni, ima problem 'varijance':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografika iz Scikit-learn

## Varijanca

Varijanca je definirana kao "prosjek kvadrata razlika od srednje vrijednosti" [(Izvor)](https://www.mathsisfun.com/data/standard-deviation.html). U kontekstu ovog problema klasteriranja, odnosi se na podatke kod kojih brojevi našeg skupa podataka imaju tendenciju previše odstupati od srednje vrijednosti.

✅ Ovo je odličan trenutak da razmislite o svim načinima na koje biste mogli ispraviti ovaj problem. Malo više prilagoditi podatke? Koristiti različite stupce? Koristiti drugačiji algoritam? Savjet: Pokušajte [skalirati svoje podatke](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) kako biste ih normalizirali i testirali druge stupce.

> Pokušajte ovaj '[kalkulator varijance](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' kako biste bolje razumjeli koncept.

---

## 🚀Izazov

Provedite neko vrijeme s ovim notebookom, prilagođavajući parametre. Možete li poboljšati točnost modela čišćenjem podataka (na primjer, uklanjanjem outliera)? Možete koristiti težine kako biste dali veću težinu određenim uzorcima podataka. Što još možete učiniti kako biste stvorili bolje klastere?

Savjet: Pokušajte skalirati svoje podatke. U notebooku postoji komentirani kod koji dodaje standardno skaliranje kako bi stupci podataka više nalikovali jedni drugima u smislu raspona. Primijetit ćete da, iako silhouette ocjena opada, 'savijanje' u grafu lakta postaje glađe. To je zato što ostavljanje podataka neskaliranima omogućuje podacima s manje varijance da imaju veću težinu. Pročitajte malo više o ovom problemu [ovdje](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Pogledajte simulator K-Means [poput ovog](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Možete koristiti ovaj alat za vizualizaciju uzoraka podataka i određivanje njihovih centara. Možete uređivati slučajnost podataka, broj klastera i broj centara. Pomaže li vam ovo da steknete ideju o tome kako se podaci mogu grupirati?

Također, pogledajte [ovaj materijal o K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) sa Stanforda.

## Zadatak

[Isprobajte različite metode klasteriranja](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.