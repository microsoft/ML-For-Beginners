<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T15:43:36+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "ro"
}
-->
# Introducere în clustering

Clustering-ul este un tip de [Învățare Nesupervizată](https://wikipedia.org/wiki/Unsupervised_learning) care presupune că un set de date nu este etichetat sau că intrările sale nu sunt asociate cu ieșiri predefinite. Folosește diverse algoritmi pentru a analiza datele neetichetate și a oferi grupări bazate pe tiparele identificate în date.

[![No One Like You de PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You de PSquare")

> 🎥 Click pe imaginea de mai sus pentru un videoclip. În timp ce studiezi învățarea automată cu clustering, bucură-te de câteva piese de Dance Hall nigerian - aceasta este o melodie foarte apreciată din 2014 de PSquare.

## [Quiz înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

### Introducere

[Clustering-ul](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) este foarte util pentru explorarea datelor. Să vedem dacă poate ajuta la descoperirea tendințelor și tiparelor în modul în care publicul nigerian consumă muzică.

✅ Ia un minut să te gândești la utilizările clustering-ului. În viața reală, clustering-ul se întâmplă ori de câte ori ai o grămadă de rufe și trebuie să sortezi hainele membrilor familiei 🧦👕👖🩲. În știința datelor, clustering-ul se întâmplă atunci când încerci să analizezi preferințele unui utilizator sau să determini caracteristicile unui set de date neetichetat. Clustering-ul, într-un fel, ajută la a face ordine în haos, ca un sertar de șosete.

[![Introducere în ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introducere în Clustering")

> 🎥 Click pe imaginea de mai sus pentru un videoclip: John Guttag de la MIT introduce clustering-ul.

Într-un mediu profesional, clustering-ul poate fi utilizat pentru a determina lucruri precum segmentarea pieței, identificarea grupelor de vârstă care cumpără anumite produse, de exemplu. O altă utilizare ar fi detectarea anomaliilor, poate pentru a identifica fraude într-un set de date cu tranzacții de carduri de credit. Sau ai putea folosi clustering-ul pentru a identifica tumori într-un lot de scanări medicale.

✅ Gândește-te un minut la cum ai putea întâlni clustering-ul 'în sălbăticie', într-un mediu bancar, de e-commerce sau de afaceri.

> 🎓 Interesant, analiza clusterelor a apărut în domeniile Antropologiei și Psihologiei în anii 1930. Îți poți imagina cum ar fi fost utilizată?

Alternativ, ai putea să-l folosești pentru gruparea rezultatelor căutării - de exemplu, după linkuri de cumpărături, imagini sau recenzii. Clustering-ul este util atunci când ai un set de date mare pe care vrei să-l reduci și pe care vrei să efectuezi o analiză mai detaliată, astfel încât tehnica poate fi utilizată pentru a învăța despre date înainte de a construi alte modele.

✅ Odată ce datele tale sunt organizate în clustere, le atribui un Id de cluster, iar această tehnică poate fi utilă pentru a păstra confidențialitatea unui set de date; poți să te referi la un punct de date prin Id-ul său de cluster, mai degrabă decât prin date identificabile mai revelatoare. Poți să te gândești la alte motive pentru care ai prefera să te referi la un Id de cluster în loc de alte elemente ale clusterului pentru a-l identifica?

Aprofundează-ți înțelegerea tehnicilor de clustering în acest [modul de învățare](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Începerea cu clustering

[Scikit-learn oferă o gamă largă](https://scikit-learn.org/stable/modules/clustering.html) de metode pentru a efectua clustering. Tipul pe care îl alegi va depinde de cazul tău de utilizare. Conform documentației, fiecare metodă are diverse beneficii. Iată un tabel simplificat al metodelor suportate de Scikit-learn și cazurile lor de utilizare adecvate:

| Numele metodei               | Caz de utilizare                                                     |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | scop general, inductiv                                               |
| Propagarea afinității        | multe, clustere inegale, inductiv                                    |
| Mean-shift                   | multe, clustere inegale, inductiv                                    |
| Clustering spectral          | puține, clustere egale, transductiv                                 |
| Clustering ierarhic Ward     | multe, clustere constrânse, transductiv                             |
| Clustering aglomerativ       | multe, constrânse, distanțe non-euclidiene, transductiv             |
| DBSCAN                       | geometrie non-plană, clustere inegale, transductiv                  |
| OPTICS                       | geometrie non-plană, clustere inegale cu densitate variabilă, transductiv |
| Amestecuri Gaussiene         | geometrie plană, inductiv                                           |
| BIRCH                        | set de date mare cu outlieri, inductiv                              |

> 🎓 Modul în care creăm clustere are mult de-a face cu modul în care grupăm punctele de date în grupuri. Să descompunem câteva vocabular:
>
> 🎓 ['Transductiv' vs. 'inductiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Inferența transductivă este derivată din cazuri de antrenament observate care se mapează la cazuri de testare specifice. Inferența inductivă este derivată din cazuri de antrenament care se mapează la reguli generale care sunt aplicate ulterior cazurilor de testare.
> 
> Un exemplu: Imaginează-ți că ai un set de date care este doar parțial etichetat. Unele lucruri sunt 'discuri', altele 'cd-uri', iar unele sunt goale. Sarcina ta este să oferi etichete pentru cele goale. Dacă alegi o abordare inductivă, ai antrena un model căutând 'discuri' și 'cd-uri' și ai aplica acele etichete datelor neetichetate. Această abordare va avea dificultăți în clasificarea lucrurilor care sunt de fapt 'casete'. O abordare transductivă, pe de altă parte, gestionează aceste date necunoscute mai eficient, deoarece lucrează pentru a grupa elemente similare împreună și apoi aplică o etichetă unui grup. În acest caz, clusterele ar putea reflecta 'lucruri muzicale rotunde' și 'lucruri muzicale pătrate'.
> 
> 🎓 ['Geometrie non-plană' vs. 'plană'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derivat din terminologia matematică, geometria non-plană vs. plană se referă la măsurarea distanțelor între puncte fie prin metode geometrice 'plane' ([Euclidiene](https://wikipedia.org/wiki/Euclidean_geometry)), fie 'non-plane' (non-Euclidiene).
>
>'Plană' în acest context se referă la geometria Euclidiană (părți din care sunt predate ca geometrie 'plană'), iar non-plană se referă la geometria non-Euclidiană. Ce legătură are geometria cu învățarea automată? Ei bine, ca două domenii care sunt bazate pe matematică, trebuie să existe o modalitate comună de a măsura distanțele între puncte în clustere, iar aceasta poate fi făcută într-un mod 'plan' sau 'non-plan', în funcție de natura datelor. [Distanțele Euclidiene](https://wikipedia.org/wiki/Euclidean_distance) sunt măsurate ca lungimea unui segment de linie între două puncte. [Distanțele non-Euclidiene](https://wikipedia.org/wiki/Non-Euclidean_geometry) sunt măsurate de-a lungul unei curbe. Dacă datele tale, vizualizate, par să nu existe pe un plan, s-ar putea să fie nevoie să folosești un algoritm specializat pentru a le gestiona.
>
![Infografic Geometrie Plană vs Non-plană](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografic de [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distanțe'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusterele sunt definite de matricea lor de distanțe, de exemplu, distanțele între puncte. Această distanță poate fi măsurată în câteva moduri. Clusterele Euclidiene sunt definite de media valorilor punctelor și conțin un 'centroid' sau punct central. Distanțele sunt astfel măsurate prin distanța față de acel centroid. Distanțele non-Euclidiene se referă la 'clustroizi', punctul cel mai apropiat de alte puncte. Clustroizii, la rândul lor, pot fi definiți în diverse moduri.
> 
> 🎓 ['Constrâns'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Clustering-ul Constrâns](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduce 'învățarea semi-supervizată' în această metodă nesupervizată. Relațiile dintre puncte sunt marcate ca 'nu se pot lega' sau 'trebuie să se lege', astfel încât unele reguli sunt impuse setului de date.
>
>Un exemplu: Dacă un algoritm este lăsat liber pe un lot de date neetichetate sau semi-etichetate, clusterele pe care le produce pot fi de calitate slabă. În exemplul de mai sus, clusterele ar putea grupa 'lucruri muzicale rotunde', 'lucruri muzicale pătrate', 'lucruri triunghiulare' și 'fursecuri'. Dacă i se oferă câteva constrângeri sau reguli de urmat ("elementul trebuie să fie din plastic", "elementul trebuie să poată produce muzică"), acest lucru poate ajuta la 'constrângerea' algoritmului să facă alegeri mai bune.
> 
> 🎓 'Densitate'
> 
> Datele care sunt 'zgomotoase' sunt considerate a fi 'dense'. Distanțele între punctele din fiecare cluster pot fi, la examinare, mai mult sau mai puțin dense sau 'aglomerate', și astfel aceste date trebuie analizate cu metoda de clustering adecvată. [Acest articol](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrează diferența dintre utilizarea clustering-ului K-Means vs. algoritmii HDBSCAN pentru a explora un set de date zgomotos cu densitate inegală a clusterelor.

## Algoritmi de clustering

Există peste 100 de algoritmi de clustering, iar utilizarea lor depinde de natura datelor disponibile. Să discutăm despre câțiva dintre cei mai importanți:

- **Clustering ierarhic**. Dacă un obiect este clasificat prin proximitatea sa față de un obiect apropiat, mai degrabă decât față de unul mai îndepărtat, clusterele sunt formate pe baza distanței membrilor față de și de la alte obiecte. Clustering-ul aglomerativ din Scikit-learn este ierarhic.

   ![Infografic Clustering Ierarhic](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografic de [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering pe bază de centroid**. Acest algoritm popular necesită alegerea 'k', sau numărul de clustere de format, după care algoritmul determină punctul central al unui cluster și adună date în jurul acelui punct. [Clustering-ul K-means](https://wikipedia.org/wiki/K-means_clustering) este o versiune populară a clustering-ului pe bază de centroid. Centrul este determinat de media cea mai apropiată, de aici și numele. Distanța pătrată față de cluster este minimizată.

   ![Infografic Clustering pe bază de Centroid](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografic de [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering bazat pe distribuție**. Bazat pe modelarea statistică, clustering-ul bazat pe distribuție se concentrează pe determinarea probabilității ca un punct de date să aparțină unui cluster și îl atribuie corespunzător. Metodele de amestecuri Gaussiene aparțin acestui tip.

- **Clustering bazat pe densitate**. Punctele de date sunt atribuite clusterelor pe baza densității lor sau a grupării lor în jurul altor puncte. Punctele de date aflate departe de grup sunt considerate outlieri sau zgomot. DBSCAN, Mean-shift și OPTICS aparțin acestui tip de clustering.

- **Clustering bazat pe grilă**. Pentru seturi de date multidimensionale, se creează o grilă, iar datele sunt împărțite între celulele grilei, creând astfel clustere.

## Exercițiu - grupează datele tale

Clustering-ul ca tehnică este foarte ajutat de o vizualizare adecvată, așa că să începem prin a vizualiza datele noastre muzicale. Acest exercițiu ne va ajuta să decidem care dintre metodele de clustering ar trebui să folosim cel mai eficient pentru natura acestor date.

1. Deschide fișierul [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) din acest folder.

1. Importă pachetul `Seaborn` pentru o vizualizare bună a datelor.

    ```python
    !pip install seaborn
    ```

1. Adaugă datele despre melodii din [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Încarcă un dataframe cu câteva date despre melodii. Pregătește-te să explorezi aceste date importând bibliotecile și afișând datele:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Verifică primele câteva linii de date:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obține informații despre dataframe, apelând `info()`:

    ```python
    df.info()
    ```

   Rezultatul arată astfel:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Verifică din nou valorile nule, apelând `isnull()` și verificând că suma este 0:

    ```python
    df.isnull().sum()
    ```

    Totul arată bine:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Descrie datele:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Dacă lucrăm cu clustering, o metodă nesupravegheată care nu necesită date etichetate, de ce arătăm aceste date cu etichete? În faza de explorare a datelor, acestea sunt utile, dar nu sunt necesare pentru ca algoritmii de clustering să funcționeze. Ai putea la fel de bine să elimini anteturile coloanelor și să te referi la date prin numărul coloanei.

Privește valorile generale ale datelor. Observă că popularitatea poate fi '0', ceea ce indică melodii care nu au un clasament. Să eliminăm aceste valori în curând.

1. Folosește un barplot pentru a afla cele mai populare genuri:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Dacă dorești să vezi mai multe valori de top, schimbă topul `[:5]` la o valoare mai mare sau elimină-l pentru a vedea totul.

Observă că atunci când genul de top este descris ca 'Missing', înseamnă că Spotify nu l-a clasificat, așa că să scăpăm de acesta.

1. Elimină datele lipsă prin filtrarea lor

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Acum verifică din nou genurile:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Cele trei genuri de top domină acest set de date. Să ne concentrăm pe `afro dancehall`, `afropop` și `nigerian pop`, și să filtrăm suplimentar setul de date pentru a elimina orice valoare de popularitate 0 (ceea ce înseamnă că nu a fost clasificată cu o popularitate în setul de date și poate fi considerată zgomot pentru scopurile noastre):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Fă un test rapid pentru a vedea dacă datele corelează într-un mod deosebit de puternic:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Singura corelație puternică este între `energy` și `loudness`, ceea ce nu este prea surprinzător, având în vedere că muzica tare este de obicei destul de energică. În rest, corelațiile sunt relativ slabe. Va fi interesant să vedem ce poate face un algoritm de clustering cu aceste date.

    > 🎓 Reține că corelația nu implică cauzalitate! Avem dovada corelației, dar nu și dovada cauzalității. Un [site web amuzant](https://tylervigen.com/spurious-correlations) are câteva vizualizări care subliniază acest punct.

Există vreo convergență în acest set de date în jurul popularității percepute a unei melodii și a dansabilității? Un FacetGrid arată că există cercuri concentrice care se aliniază, indiferent de gen. Ar putea fi că gusturile nigeriene converg la un anumit nivel de dansabilitate pentru acest gen?

✅ Încearcă diferite puncte de date (energy, loudness, speechiness) și mai multe sau diferite genuri muzicale. Ce poți descoperi? Privește tabelul `df.describe()` pentru a vedea răspândirea generală a punctelor de date.

### Exercițiu - distribuția datelor

Aceste trei genuri sunt semnificativ diferite în percepția dansabilității lor, bazată pe popularitate?

1. Examinează distribuția datelor pentru genurile noastre de top în ceea ce privește popularitatea și dansabilitatea de-a lungul unei axe x și y date.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Poți descoperi cercuri concentrice în jurul unui punct general de convergență, arătând distribuția punctelor.

    > 🎓 Reține că acest exemplu folosește un grafic KDE (Kernel Density Estimate) care reprezintă datele folosind o curbă continuă de densitate a probabilității. Acest lucru ne permite să interpretăm datele atunci când lucrăm cu distribuții multiple.

    În general, cele trei genuri se aliniază vag în ceea ce privește popularitatea și dansabilitatea. Determinarea clusterelor în aceste date vag aliniate va fi o provocare:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Creează un scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Un scatterplot pe aceleași axe arată un model similar de convergență

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

În general, pentru clustering, poți folosi scatterplots pentru a arăta clusterele de date, așa că stăpânirea acestui tip de vizualizare este foarte utilă. În lecția următoare, vom lua aceste date filtrate și vom folosi clustering-ul k-means pentru a descoperi grupuri în aceste date care par să se suprapună în moduri interesante.

---

## 🚀Provocare

În pregătirea pentru lecția următoare, creează un grafic despre diferitele algoritmi de clustering pe care i-ai putea descoperi și folosi într-un mediu de producție. Ce tipuri de probleme încearcă să abordeze clustering-ul?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Studiu Individual

Înainte de a aplica algoritmi de clustering, așa cum am învățat, este o idee bună să înțelegi natura setului tău de date. Citește mai multe despre acest subiect [aici](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Acest articol util](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) te ghidează prin diferitele moduri în care algoritmii de clustering se comportă, având în vedere diferite forme de date.

## Temă

[Cercetează alte vizualizări pentru clustering](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.