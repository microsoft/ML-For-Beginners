<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T15:42:19+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "sk"
}
-->
# Úvod do zhlukovania

Zhlukovanie je typ [neučenej metódy](https://wikipedia.org/wiki/Unsupervised_learning), ktorá predpokladá, že dataset nie je označený alebo že jeho vstupy nie sú spojené s preddefinovanými výstupmi. Používa rôzne algoritmy na triedenie neoznačených dát a poskytuje skupiny na základe vzorov, ktoré rozpozná v dátach.

[![No One Like You od PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You od PSquare")

> 🎥 Kliknite na obrázok vyššie pre video. Kým študujete strojové učenie so zhlukovaním, užite si niektoré nigerijské Dance Hall skladby - toto je vysoko hodnotená skladba z roku 2014 od PSquare.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

### Úvod

[Zhlukovanie](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je veľmi užitočné pre prieskum dát. Pozrime sa, či nám môže pomôcť objaviť trendy a vzory v tom, ako nigerijské publikum konzumuje hudbu.

✅ Zamyslite sa na chvíľu nad využitím zhlukovania. V reálnom živote sa zhlukovanie deje vždy, keď máte hromadu bielizne a potrebujete roztriediť oblečenie členov rodiny 🧦👕👖🩲. V dátovej vede sa zhlukovanie deje pri analýze preferencií používateľov alebo pri určovaní charakteristík akéhokoľvek neoznačeného datasetu. Zhlukovanie, do istej miery, pomáha urobiť poriadok z chaosu, ako napríklad zásuvka na ponožky.

[![Úvod do ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Úvod do zhlukovania")

> 🎥 Kliknite na obrázok vyššie pre video: John Guttag z MIT predstavuje zhlukovanie.

V profesionálnom prostredí môže byť zhlukovanie použité na určenie vecí, ako je segmentácia trhu, napríklad na určenie, ktoré vekové skupiny kupujú aké položky. Ďalším využitím by mohlo byť odhaľovanie anomálií, napríklad na detekciu podvodov z datasetu transakcií kreditných kariet. Alebo by ste mohli použiť zhlukovanie na určenie nádorov v dávke medicínskych skenov.

✅ Zamyslite sa na chvíľu nad tým, ako ste sa mohli stretnúť so zhlukovaním „v divočine“, v bankovníctve, e-commerce alebo obchodnom prostredí.

> 🎓 Zaujímavosť: Analýza zhlukov vznikla v oblasti antropológie a psychológie v 30. rokoch 20. storočia. Dokážete si predstaviť, ako mohla byť použitá?

Alternatívne by ste ju mohli použiť na zoskupenie výsledkov vyhľadávania - napríklad podľa nákupných odkazov, obrázkov alebo recenzií. Zhlukovanie je užitočné, keď máte veľký dataset, ktorý chcete zredukovať a na ktorom chcete vykonať podrobnejšiu analýzu, takže táto technika môže byť použitá na získanie informácií o dátach pred vytvorením iných modelov.

✅ Keď sú vaše dáta organizované do zhlukov, priradíte im identifikátor zhluku, a táto technika môže byť užitočná pri zachovaní súkromia datasetu; namiesto odkazovania na konkrétne údaje môžete odkazovať na identifikátor zhluku. Dokážete si predstaviť ďalšie dôvody, prečo by ste odkazovali na identifikátor zhluku namiesto iných prvkov zhluku na jeho identifikáciu?

Prehĺbte svoje pochopenie techník zhlukovania v tomto [učebnom module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Začíname so zhlukovaním

[Scikit-learn ponúka širokú škálu](https://scikit-learn.org/stable/modules/clustering.html) metód na vykonávanie zhlukovania. Typ, ktorý si vyberiete, bude závisieť od vášho prípadu použitia. Podľa dokumentácie má každá metóda rôzne výhody. Tu je zjednodušená tabuľka metód podporovaných Scikit-learn a ich vhodné prípady použitia:

| Názov metódy                 | Prípad použitia                                                      |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | všeobecné použitie, induktívne                                       |
| Affinity propagation         | mnoho, nerovnomerné zhluky, induktívne                              |
| Mean-shift                   | mnoho, nerovnomerné zhluky, induktívne                              |
| Spectral clustering          | málo, rovnomerné zhluky, transduktívne                              |
| Ward hierarchical clustering | mnoho, obmedzené zhluky, transduktívne                              |
| Agglomerative clustering     | mnoho, obmedzené, ne-Euklidovské vzdialenosti, transduktívne        |
| DBSCAN                       | neplochá geometria, nerovnomerné zhluky, transduktívne              |
| OPTICS                       | neplochá geometria, nerovnomerné zhluky s variabilnou hustotou, transduktívne |
| Gaussian mixtures            | plochá geometria, induktívne                                        |
| BIRCH                        | veľký dataset s odľahlými bodmi, induktívne                         |

> 🎓 Ako vytvárame zhluky, má veľa spoločného s tým, ako zhromažďujeme dátové body do skupín. Poďme si rozobrať niektoré pojmy:
>
> 🎓 ['Transduktívne' vs. 'induktívne'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktívna inferencia je odvodená z pozorovaných tréningových prípadov, ktoré sa mapujú na konkrétne testovacie prípady. Induktívna inferencia je odvodená z tréningových prípadov, ktoré sa mapujú na všeobecné pravidlá, ktoré sa potom aplikujú na testovacie prípady.
> 
> Príklad: Predstavte si, že máte dataset, ktorý je len čiastočne označený. Niektoré veci sú „platne“, niektoré „CD“ a niektoré sú prázdne. Vašou úlohou je poskytnúť označenia pre prázdne miesta. Ak si vyberiete induktívny prístup, trénovali by ste model hľadajúci „platne“ a „CD“ a aplikovali tieto označenia na vaše neoznačené dáta. Tento prístup bude mať problémy s klasifikáciou vecí, ktoré sú vlastne „kazety“. Transduktívny prístup, na druhej strane, efektívnejšie spracováva tieto neznáme dáta, pretože pracuje na zoskupení podobných položiek a potom aplikuje označenie na skupinu. V tomto prípade by zhluky mohli odrážať „okrúhle hudobné veci“ a „štvorcové hudobné veci“.
> 
> 🎓 ['Neplochá' vs. 'plochá' geometria](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Odvodené z matematickej terminológie, neplochá vs. plochá geometria sa týka merania vzdialeností medzi bodmi buď „plochými“ ([Euklidovskými](https://wikipedia.org/wiki/Euclidean_geometry)) alebo „neplochými“ (ne-Euklidovskými) geometrickými metódami.
>
>'Plochá' v tomto kontexte odkazuje na Euklidovskú geometriu (časti ktorej sa učia ako „rovinná“ geometria), a neplochá odkazuje na ne-Euklidovskú geometriu. Čo má geometria spoločné so strojovým učením? Ako dve oblasti, ktoré sú zakorenené v matematike, musí existovať spoločný spôsob merania vzdialeností medzi bodmi v zhlukoch, a to môže byť vykonané „plochým“ alebo „neplochým“ spôsobom, v závislosti od povahy dát. [Euklidovské vzdialenosti](https://wikipedia.org/wiki/Euclidean_distance) sa merajú ako dĺžka úsečky medzi dvoma bodmi. [Ne-Euklidovské vzdialenosti](https://wikipedia.org/wiki/Non-Euclidean_geometry) sa merajú pozdĺž krivky. Ak vaše dáta, vizualizované, neexistujú na rovine, možno budete potrebovať použiť špecializovaný algoritmus na ich spracovanie.
>
![Infografika plochá vs. neplochá geometria](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Vzdialenosti'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Zhluky sú definované ich maticou vzdialeností, napr. vzdialenosti medzi bodmi. Táto vzdialenosť môže byť meraná niekoľkými spôsobmi. Euklidovské zhluky sú definované priemerom hodnôt bodov a obsahujú „centroid“ alebo stredový bod. Vzdialenosti sú teda merané vzdialenosťou k tomuto centroidu. Ne-Euklidovské vzdialenosti odkazujú na „clustroidy“, bod najbližší k ostatným bodom. Clustroidy môžu byť definované rôznymi spôsobmi.
> 
> 🎓 ['Obmedzené'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Obmedzené zhlukovanie](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) zavádza „semi-učené“ učenie do tejto neučenej metódy. Vzťahy medzi bodmi sú označené ako „nemôže byť prepojené“ alebo „musí byť prepojené“, takže na dataset sú aplikované určité pravidlá.
>
>Príklad: Ak je algoritmus voľne spustený na dávke neoznačených alebo semi-označených dát, zhluky, ktoré vytvorí, môžu byť nekvalitné. V príklade vyššie by zhluky mohli zoskupovať „okrúhle hudobné veci“ a „štvorcové hudobné veci“ a „trojuholníkové veci“ a „sušienky“. Ak sú dané nejaké obmedzenia alebo pravidlá, ktoré treba dodržiavať („položka musí byť vyrobená z plastu“, „položka musí byť schopná produkovať hudbu“), môže to pomôcť „obmedziť“ algoritmus, aby robil lepšie rozhodnutia.
> 
> 🎓 'Hustota'
> 
> Dáta, ktoré sú „hlučné“, sa považujú za „husté“. Vzdialenosti medzi bodmi v každom z jeho zhlukov môžu byť pri skúmaní viac alebo menej husté, alebo „preplnené“, a preto tieto dáta potrebujú byť analyzované vhodnou metódou zhlukovania. [Tento článok](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonštruje rozdiel medzi použitím K-Means zhlukovania vs. HDBSCAN algoritmov na preskúmanie hlučného datasetu s nerovnomernou hustotou zhlukov.

## Algoritmy zhlukovania

Existuje viac ako 100 algoritmov zhlukovania a ich použitie závisí od povahy dostupných dát. Poďme diskutovať o niektorých hlavných:

- **Hierarchické zhlukovanie**. Ak je objekt klasifikovaný podľa jeho blízkosti k blízkemu objektu, namiesto vzdialeného, zhluky sú tvorené na základe vzdialenosti členov od ostatných objektov. Agglomeratívne zhlukovanie v Scikit-learn je hierarchické.

   ![Infografika hierarchického zhlukovania](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Zhlukovanie podľa centroidu**. Tento populárny algoritmus vyžaduje výber 'k', alebo počet zhlukov, ktoré sa majú vytvoriť, po ktorom algoritmus určí stredový bod zhluku a zhromaždí dáta okolo tohto bodu. [K-means zhlukovanie](https://wikipedia.org/wiki/K-means_clustering) je populárna verzia zhlukovania podľa centroidu. Stred je určený najbližším priemerom, odtiaľ názov. Štvorcová vzdialenosť od zhluku je minimalizovaná.

   ![Infografika zhlukovania podľa centroidu](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribučné zhlukovanie**. Založené na štatistickom modelovaní, distribučné zhlukovanie sa zameriava na určenie pravdepodobnosti, že dátový bod patrí do zhluku, a jeho priradenie. Metódy Gaussovských zmesí patria do tohto typu.

- **Zhlukovanie podľa hustoty**. Dátové body sú priradené do zhlukov na základe ich hustoty, alebo ich zoskupenia okolo seba. Dátové body vzdialené od skupiny sú považované za odľahlé body alebo šum. DBSCAN, Mean-shift a OPTICS patria do tohto typu zhlukovania.

- **Zhlukovanie podľa mriežky**. Pre multidimenzionálne datasety sa vytvorí mriežka a dáta sa rozdelia medzi bunky mriežky, čím sa vytvoria zhluky.

## Cvičenie - zhlukujte svoje dáta

Zhlukovanie ako technika je výrazne podporené správnou vizualizáciou, takže začnime vizualizáciou našich hudobných dát. Toto cvičenie nám pomôže rozhodnúť, ktorú z metód zhlukovania by sme mali najefektívnejšie použiť pre povahu týchto dát.

1. Otvorte súbor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) v tomto priečinku.

1. Importujte balík `Seaborn` pre dobrú vizualizáciu dát.

    ```python
    !pip install seaborn
    ```

1. Pripojte hudobné dáta zo súboru [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Načítajte dataframe s niektorými dátami o skladbách. Pripravte sa na preskúmanie týchto dát importovaním knižníc a vypísaním dát:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Skontrolujte prvých pár riadkov dát:

    |     | názov                   | album                        | umelec              | hlavný žáner umelca | dátum vydania | dĺžka | popularita | tanečnosť    | akustickosť  | energia | inštrumentálnosť | živosť   | hlasitosť | rečovosť    | tempo   | takt           |
    | --- | ----------------------- | ---------------------------- | ------------------- | ------------------- | ------------- | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                  | Mandy & The Jungle           | Cruel Santino       | alternatívne r&b    | 2019          | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush              | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop             | 2020          | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Získajte niektoré informácie o dátovom rámci, zavolaním `info()`:

    ```python
    df.info()
    ```

   Výstup vyzerá takto:

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

1. Skontrolujte, či neobsahuje nulové hodnoty, zavolaním `isnull()` a overením, že súčet je 0:

    ```python
    df.isnull().sum()
    ```

    Vyzerá to dobre:

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

1. Popíšte údaje:

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

> 🤔 Ak pracujeme s klastrovaním, nesupervidovanou metódou, ktorá nevyžaduje označené údaje, prečo zobrazujeme tieto údaje s označeniami? Počas fázy skúmania údajov sú užitočné, ale pre fungovanie algoritmov klastrovania nie sú potrebné. Mohli by ste rovnako odstrániť hlavičky stĺpcov a odkazovať na údaje podľa čísla stĺpca.

Pozrite sa na všeobecné hodnoty údajov. Všimnite si, že popularita môže byť '0', čo ukazuje piesne, ktoré nemajú žiadne hodnotenie. Tieto odstránime čoskoro.

1. Použite stĺpcový graf na zistenie najpopulárnejších žánrov:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![najpopulárnejšie](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Ak chcete vidieť viac najlepších hodnôt, zmeňte top `[:5]` na väčšiu hodnotu alebo ho odstráňte, aby ste videli všetky.

Všimnite si, že keď je najlepší žáner opísaný ako 'Missing', znamená to, že Spotify ho neklasifikoval, takže ho odstránime.

1. Odstráňte chýbajúce údaje ich filtrovaním

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Teraz znova skontrolujte žánre:

    ![všetky žánre](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Tri najlepšie žánre jednoznačne dominujú tejto množine údajov. Zamerajme sa na `afro dancehall`, `afropop` a `nigerian pop`, a navyše filtrovaním odstráňme všetko s hodnotou popularity 0 (čo znamená, že nebolo klasifikované s popularitou v množine údajov a môže byť považované za šum pre naše účely):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Rýchlo otestujte, či údaje korelujú nejakým výrazným spôsobom:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korelácie](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Jediná silná korelácia je medzi `energy` a `loudness`, čo nie je príliš prekvapujúce, keďže hlasná hudba je zvyčajne dosť energická. Inak sú korelácie relatívne slabé. Bude zaujímavé vidieť, čo s týmito údajmi dokáže algoritmus klastrovania.

    > 🎓 Všimnite si, že korelácia neimplikuje kauzalitu! Máme dôkaz o korelácii, ale žiadny dôkaz o kauzalite. [Zábavná webová stránka](https://tylervigen.com/spurious-correlations) obsahuje niekoľko vizuálov, ktoré zdôrazňujú tento bod.

Existuje nejaká konvergencia v tejto množine údajov okolo vnímania popularity a tanečnosti piesne? FacetGrid ukazuje, že existujú sústredné kruhy, ktoré sa zhodujú, bez ohľadu na žáner. Mohlo by to byť tak, že nigérijské chute sa zhodujú na určitej úrovni tanečnosti pre tento žáner?  

✅ Vyskúšajte rôzne dátové body (energy, loudness, speechiness) a viac alebo iné hudobné žánre. Čo môžete objaviť? Pozrite sa na tabuľku `df.describe()`, aby ste videli všeobecné rozloženie dátových bodov.

### Cvičenie - rozloženie údajov

Sú tieto tri žánre významne odlišné vo vnímaní ich tanečnosti na základe ich popularity?

1. Preskúmajte rozloženie údajov našich troch najlepších žánrov pre popularitu a tanečnosť pozdĺž danej osi x a y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Môžete objaviť sústredné kruhy okolo všeobecného bodu konvergencie, ktoré ukazujú rozloženie bodov.

    > 🎓 Tento príklad používa graf KDE (Kernel Density Estimate), ktorý reprezentuje údaje pomocou spojitej krivky hustoty pravdepodobnosti. To nám umožňuje interpretovať údaje pri práci s viacerými rozloženiami.

    Vo všeobecnosti sa tri žánre voľne zhodujú, pokiaľ ide o ich popularitu a tanečnosť. Určenie klastrov v týchto voľne zarovnaných údajoch bude výzvou:

    ![rozloženie](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Vytvorte bodový graf:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Bodový graf rovnakých osí ukazuje podobný vzor konvergencie

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Vo všeobecnosti môžete pre klastrovanie použiť bodové grafy na zobrazenie klastrov údajov, takže zvládnutie tohto typu vizualizácie je veľmi užitočné. V ďalšej lekcii použijeme tieto filtrované údaje a použijeme klastrovanie k-means na objavenie skupín v týchto údajoch, ktoré sa zaujímavo prekrývajú.

---

## 🚀Výzva

V rámci prípravy na ďalšiu lekciu vytvorte graf o rôznych algoritmoch klastrovania, ktoré by ste mohli objaviť a použiť v produkčnom prostredí. Aké problémy sa klastrovanie snaží riešiť?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samoštúdium

Pred aplikáciou algoritmov klastrovania, ako sme sa naučili, je dobré pochopiť povahu vašej množiny údajov. Prečítajte si viac na túto tému [tu](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Tento užitočný článok](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vás prevedie rôznymi spôsobmi, ako sa rôzne algoritmy klastrovania správajú pri rôznych tvaroch údajov.

## Zadanie

[Preskúmajte ďalšie vizualizácie pre klastrovanie](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.