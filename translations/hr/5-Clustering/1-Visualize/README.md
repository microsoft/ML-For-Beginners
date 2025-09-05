<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T12:10:57+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "hr"
}
-->
# Uvod u klasteriranje

Klasteriranje je vrsta [Nenadziranog učenja](https://wikipedia.org/wiki/Unsupervised_learning) koja pretpostavlja da je skup podataka neoznačen ili da njegovi ulazi nisu povezani s unaprijed definiranim izlazima. Koristi razne algoritme za analizu neoznačenih podataka i pruža grupiranja prema obrascima koje prepoznaje u podacima.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Kliknite na sliku iznad za video. Dok proučavate strojno učenje s klasteriranjem, uživajte u nigerijskim Dance Hall pjesmama - ovo je visoko ocijenjena pjesma iz 2014. od PSquare.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

### Uvod

[Klasteriranje](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je vrlo korisno za istraživanje podataka. Pogledajmo može li pomoći u otkrivanju trendova i obrazaca u načinu na koji nigerijska publika konzumira glazbu.

✅ Odvojite trenutak da razmislite o primjenama klasteriranja. U stvarnom životu, klasteriranje se događa kad imate hrpu rublja i trebate razvrstati odjeću članova obitelji 🧦👕👖🩲. U podatkovnoj znanosti, klasteriranje se događa kada pokušavate analizirati korisničke preferencije ili odrediti karakteristike bilo kojeg neoznačenog skupa podataka. Klasteriranje, na neki način, pomaže u stvaranju reda iz kaosa, poput ladice za čarape.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Kliknite na sliku iznad za video: John Guttag s MIT-a uvodi klasteriranje

U profesionalnom okruženju, klasteriranje se može koristiti za određivanje stvari poput segmentacije tržišta, utvrđivanja koje dobne skupine kupuju koje proizvode, na primjer. Druga primjena bila bi otkrivanje anomalija, možda za otkrivanje prijevara iz skupa podataka o transakcijama kreditnim karticama. Ili biste mogli koristiti klasteriranje za određivanje tumora u seriji medicinskih skenova.

✅ Razmislite na trenutak o tome kako ste možda naišli na klasteriranje 'u divljini', u bankarstvu, e-trgovini ili poslovnom okruženju.

> 🎓 Zanimljivo je da analiza klastera potječe iz područja antropologije i psihologije 1930-ih. Možete li zamisliti kako se mogla koristiti?

Alternativno, mogli biste ga koristiti za grupiranje rezultata pretraživanja - prema poveznicama za kupovinu, slikama ili recenzijama, na primjer. Klasteriranje je korisno kada imate veliki skup podataka koji želite smanjiti i na kojem želite provesti detaljniju analizu, pa se tehnika može koristiti za upoznavanje podataka prije nego što se izgrade drugi modeli.

✅ Kada su vaši podaci organizirani u klastere, dodjeljujete im ID klastera, a ova tehnika može biti korisna pri očuvanju privatnosti skupa podataka; umjesto toga možete se referirati na podatkovnu točku prema njenom ID-u klastera, umjesto prema otkrivanju identifikacijskih podataka. Možete li smisliti druge razloge zašto biste koristili ID klastera umjesto drugih elemenata klastera za identifikaciju?

Produbite svoje razumijevanje tehnika klasteriranja u ovom [modulu za učenje](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Početak rada s klasteriranjem

[Scikit-learn nudi širok raspon](https://scikit-learn.org/stable/modules/clustering.html) metoda za izvođenje klasteriranja. Vrsta koju odaberete ovisit će o vašem slučaju upotrebe. Prema dokumentaciji, svaka metoda ima različite prednosti. Evo pojednostavljene tablice metoda koje podržava Scikit-learn i njihovih odgovarajućih slučajeva upotrebe:

| Naziv metode                | Slučaj upotrebe                                                      |
| :--------------------------- | :------------------------------------------------------------------ |
| K-Means                      | opća namjena, induktivno                                            |
| Affinity propagation         | mnogi, nejednaki klasteri, induktivno                              |
| Mean-shift                   | mnogi, nejednaki klasteri, induktivno                              |
| Spectral clustering          | malo, jednaki klasteri, transduktivno                              |
| Ward hierarchical clustering | mnogi, ograničeni klasteri, transduktivno                          |
| Agglomerative clustering     | mnogi, ograničeni, ne-Euklidske udaljenosti, transduktivno         |
| DBSCAN                       | ne-ravna geometrija, nejednaki klasteri, transduktivno             |
| OPTICS                       | ne-ravna geometrija, nejednaki klasteri s promjenjivom gustoćom, transduktivno |
| Gaussian mixtures            | ravna geometrija, induktivno                                       |
| BIRCH                        | veliki skup podataka s iznimkama, induktivno                       |

> 🎓 Kako stvaramo klastere ima puno veze s načinom na koji grupiramo podatkovne točke u skupine. Razjasnimo neke pojmove:
>
> 🎓 ['Transduktivno' vs. 'induktivno'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktivno zaključivanje proizlazi iz promatranih slučajeva treninga koji se mapiraju na specifične testne slučajeve. Induktivno zaključivanje proizlazi iz slučajeva treninga koji se mapiraju na opća pravila koja se tek tada primjenjuju na testne slučajeve. 
> 
> Primjer: Zamislite da imate skup podataka koji je samo djelomično označen. Neke stvari su 'ploče', neke 'CD-i', a neke su prazne. Vaš zadatak je dodijeliti oznake praznima. Ako odaberete induktivni pristup, trenirali biste model tražeći 'ploče' i 'CD-e' i primijenili te oznake na neoznačene podatke. Ovaj pristup će imati problema s klasifikacijom stvari koje su zapravo 'kasete'. Transduktivni pristup, s druge strane, učinkovitije se nosi s ovim nepoznatim podacima jer radi na grupiranju sličnih stavki zajedno i zatim primjenjuje oznaku na grupu. U ovom slučaju, klasteri bi mogli odražavati 'okrugle glazbene stvari' i 'kvadratne glazbene stvari'. 
> 
> 🎓 ['Ne-ravna' vs. 'ravna' geometrija](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Izvedeno iz matematičke terminologije, ne-ravna vs. ravna geometrija odnosi se na mjerenje udaljenosti između točaka bilo 'ravnim' ([Euklidskim](https://wikipedia.org/wiki/Euclidean_geometry)) ili 'ne-ravnim' (ne-Euklidskim) geometrijskim metodama. 
>
>'Ravna' u ovom kontekstu odnosi se na Euklidsku geometriju (dijelovi koje se uče kao 'ravninska' geometrija), a ne-ravna se odnosi na ne-Euklidsku geometriju. Što geometrija ima veze sa strojnim učenjem? Pa, kao dva područja koja su ukorijenjena u matematici, mora postojati zajednički način mjerenja udaljenosti između točaka u klasterima, a to se može učiniti na 'ravni' ili 'ne-ravni' način, ovisno o prirodi podataka. [Euklidske udaljenosti](https://wikipedia.org/wiki/Euclidean_distance) mjere se kao duljina segmenta linije između dvije točke. [Ne-Euklidske udaljenosti](https://wikipedia.org/wiki/Non-Euclidean_geometry) mjere se duž krivulje. Ako se vaši podaci, vizualizirani, čine da ne postoje na ravnini, možda ćete trebati koristiti specijalizirani algoritam za njihovu obradu.
>
![Infografika ravne vs. ne-ravne geometrije](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Udaljenosti'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klasteri su definirani njihovom matricom udaljenosti, npr. udaljenostima između točaka. Ova udaljenost može se mjeriti na nekoliko načina. Euklidski klasteri definirani su prosjekom vrijednosti točaka i sadrže 'centroid' ili središnju točku. Udaljenosti se stoga mjere udaljenosti do tog centroida. Ne-Euklidske udaljenosti odnose se na 'clustroid', točku najbližu drugim točkama. Clustroidi se pak mogu definirati na različite načine.
> 
> 🎓 ['Ograničeno'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Ograničeno klasteriranje](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) uvodi 'polu-nadzirano' učenje u ovu nenadziranu metodu. Odnosi između točaka označeni su kao 'ne može se povezati' ili 'mora se povezati' pa se neka pravila nameću skupu podataka.
>
>Primjer: Ako se algoritam pusti na skup neoznačenih ili polu-označenih podataka, klasteri koje proizvede mogu biti loše kvalitete. U gore navedenom primjeru, klasteri bi mogli grupirati 'okrugle glazbene stvari' i 'kvadratne glazbene stvari' i 'trokutaste stvari' i 'kolačiće'. Ako se daju neka ograničenja ili pravila koja treba slijediti ("stavka mora biti izrađena od plastike", "stavka mora moći proizvoditi glazbu") to može pomoći 'ograničiti' algoritam da donosi bolje odluke.
> 
> 🎓 'Gustoća'
> 
> Podaci koji su 'bučni' smatraju se 'gustima'. Udaljenosti između točaka u svakom od njegovih klastera mogu se pokazati, pri ispitivanju, više ili manje gustima, ili 'zbijenima', pa se ti podaci moraju analizirati odgovarajućom metodom klasteriranja. [Ovaj članak](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) pokazuje razliku između korištenja K-Means klasteriranja i HDBSCAN algoritama za istraživanje bučnog skupa podataka s nejednakom gustoćom klastera.

## Algoritmi klasteriranja

Postoji preko 100 algoritama klasteriranja, a njihova upotreba ovisi o prirodi podataka. Razgovarajmo o nekima od glavnih:

- **Hijerarhijsko klasteriranje**. Ako se objekt klasificira prema njegovoj blizini obližnjem objektu, a ne onom udaljenijem, klasteri se formiraju na temelju udaljenosti članova od i prema drugim objektima. Scikit-learn-ovo aglomerativno klasteriranje je hijerarhijsko.

   ![Infografika hijerarhijskog klasteriranja](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid klasteriranje**. Ovaj popularni algoritam zahtijeva odabir 'k', ili broja klastera koje treba formirati, nakon čega algoritam određuje središnju točku klastera i okuplja podatke oko te točke. [K-means klasteriranje](https://wikipedia.org/wiki/K-means_clustering) je popularna verzija centroid klasteriranja. Središte se određuje prema najbližem prosjeku, otuda i naziv. Kvadratna udaljenost od klastera se minimizira.

   ![Infografika centroid klasteriranja](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Klasteriranje temeljeno na distribuciji**. Temeljeno na statističkom modeliranju, klasteriranje temeljeno na distribuciji usredotočuje se na određivanje vjerojatnosti da podatkovna točka pripada klasteru i dodjeljuje je u skladu s tim. Metode Gaussian mješavina pripadaju ovom tipu.

- **Klasteriranje temeljeno na gustoći**. Podatkovne točke dodjeljuju se klasterima na temelju njihove gustoće, odnosno njihovog grupiranja jedne oko drugih. Podatkovne točke udaljene od grupe smatraju se iznimkama ili šumom. DBSCAN, Mean-shift i OPTICS pripadaju ovom tipu klasteriranja.

- **Klasteriranje temeljeno na mreži**. Za višedimenzionalne skupove podataka, stvara se mreža i podaci se dijele među ćelijama mreže, čime se stvaraju klasteri.

## Vježba - klasterirajte svoje podatke

Klasteriranje kao tehnika uvelike se olakšava pravilnom vizualizacijom, pa krenimo s vizualizacijom naših glazbenih podataka. Ova vježba pomoći će nam odlučiti koju od metoda klasteriranja najefikasnije koristiti za prirodu ovih podataka.

1. Otvorite datoteku [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) u ovoj mapi.

1. Uvezite paket `Seaborn` za dobru vizualizaciju podataka.

    ```python
    !pip install seaborn
    ```

1. Dodajte podatke o pjesmama iz [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Učitajte dataframe s nekim podacima o pjesmama. Pripremite se za istraživanje ovih podataka uvozom biblioteka i ispisivanjem podataka:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Provjerite prvih nekoliko redaka podataka:

    |     | naziv                    | album                        | izvođač             | glavni žanr izvođača | datum izlaska | duljina | popularnost | plesnost     | akustičnost | energija | instrumentalnost | živost   | glasnoća | govornost   | tempo   | vremenski potpis |
    | --- | ------------------------ | ---------------------------- | ------------------- | -------------------- | ------------- | ------- | ----------- | ------------ | ----------- | -------- | ---------------- | -------- | -------- | ----------- | ------- | ---------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternativni r&b     | 2019          | 144000  | 48          | 0.666        | 0.851       | 0.42     | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop              | 2020          | 89488   | 30          | 0.71         | 0.0822      | 0.683    | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dobijte informacije o dataframeu pozivajući `info()`:

    ```python
    df.info()
    ```

   Izlaz izgleda ovako:

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

1. Provjerite ima li null vrijednosti pozivajući `isnull()` i provjerite da je zbroj 0:

    ```python
    df.isnull().sum()
    ```

    Izgleda dobro:

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

1. Opis podataka:

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

> 🤔 Ako radimo s klasteriranjem, nesuperviziranom metodom koja ne zahtijeva označene podatke, zašto prikazujemo ove podatke s oznakama? Tijekom faze istraživanja podataka, oznake su korisne, ali nisu nužne za rad algoritama klasteriranja. Mogli biste jednostavno ukloniti zaglavlja stupaca i referirati se na podatke prema broju stupca.

Pogledajte opće vrijednosti podataka. Primijetite da popularnost može biti '0', što pokazuje pjesme koje nemaju rangiranje. Uskoro ćemo ih ukloniti.

1. Koristite barplot za otkrivanje najpopularnijih žanrova:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![najpopularniji](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Ako želite vidjeti više top vrijednosti, promijenite top `[:5]` na veću vrijednost ili ga uklonite da vidite sve.

Napomena: kada je top žanr opisan kao 'Missing', to znači da ga Spotify nije klasificirao, pa ga uklonimo.

1. Uklonite nedostajuće podatke filtriranjem:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sada ponovno provjerite žanrove:

    ![svi žanrovi](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Tri najpopularnija žanra dominiraju ovim skupom podataka. Usredotočimo se na `afro dancehall`, `afropop` i `nigerian pop`, dodatno filtrirajmo skup podataka kako bismo uklonili sve s vrijednošću popularnosti 0 (što znači da nije klasificirano s popularnošću u skupu podataka i može se smatrati šumom za naše svrhe):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Brzo testirajte koreliraju li podaci na neki posebno jak način:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korelacije](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Jedina jaka korelacija je između `energy` i `loudness`, što nije previše iznenađujuće, s obzirom na to da je glasna glazba obično prilično energična. Inače, korelacije su relativno slabe. Bit će zanimljivo vidjeti što algoritam klasteriranja može napraviti s ovim podacima.

    > 🎓 Napomena: korelacija ne implicira uzročnost! Imamo dokaz korelacije, ali ne i dokaz uzročnosti. [Zabavna web stranica](https://tylervigen.com/spurious-correlations) ima vizuale koji naglašavaju ovu točku.

Postoji li konvergencija u ovom skupu podataka oko percepcije popularnosti i plesnosti pjesme? FacetGrid pokazuje da postoje koncentrični krugovi koji se podudaraju, bez obzira na žanr. Može li biti da se nigerijski ukusi konvergiraju na određenoj razini plesnosti za ovaj žanr?

✅ Isprobajte različite podatkovne točke (energy, loudness, speechiness) i više ili različite glazbene žanrove. Što možete otkriti? Pogledajte tablicu `df.describe()` kako biste vidjeli opći raspon podatkovnih točaka.

### Vježba - distribucija podataka

Jesu li ova tri žanra značajno različita u percepciji njihove plesnosti, na temelju njihove popularnosti?

1. Ispitajte distribuciju podataka za naša tri najbolja žanra za popularnost i plesnost duž zadane x i y osi.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Možete otkriti koncentrične krugove oko opće točke konvergencije, pokazujući distribuciju točaka.

    > 🎓 Napomena: ovaj primjer koristi KDE (Kernel Density Estimate) graf koji predstavlja podatke koristeći kontinuiranu krivulju gustoće vjerojatnosti. To nam omogućuje interpretaciju podataka pri radu s višestrukim distribucijama.

    Općenito, tri žanra se labavo usklađuju u smislu njihove popularnosti i plesnosti. Određivanje klastera u ovim labavo usklađenim podacima bit će izazov:

    ![distribucija](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Napravite scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot istih osi pokazuje sličan obrazac konvergencije.

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Općenito, za klasteriranje možete koristiti scatterplotove za prikaz klastera podataka, pa je ovladavanje ovom vrstom vizualizacije vrlo korisno. U sljedećoj lekciji, uzet ćemo ove filtrirane podatke i koristiti k-means klasteriranje za otkrivanje grupa u ovim podacima koje se preklapaju na zanimljive načine.

---

## 🚀Izazov

U pripremi za sljedeću lekciju, napravite grafikon o raznim algoritmima klasteriranja koje biste mogli otkriti i koristiti u produkcijskom okruženju. Koje vrste problema klasteriranje pokušava riješiti?

## [Post-lecture kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Prije nego što primijenite algoritme klasteriranja, kao što smo naučili, dobro je razumjeti prirodu vašeg skupa podataka. Pročitajte više o ovoj temi [ovdje](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ovaj koristan članak](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vodi vas kroz različite načine na koje se različiti algoritmi klasteriranja ponašaju, s obzirom na različite oblike podataka.

## Zadatak

[Istrazite druge vizualizacije za klasteriranje](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane ljudskog prevoditelja. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.