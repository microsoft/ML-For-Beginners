<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T12:12:07+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "sl"
}
-->
# Uvod v razvrščanje v skupine

Razvrščanje v skupine je vrsta [nenadzorovanega učenja](https://wikipedia.org/wiki/Unsupervised_learning), ki predpostavlja, da je podatkovni niz neoznačen ali da njegovi vnosi niso povezani z vnaprej določenimi izhodi. Uporablja različne algoritme za razvrščanje neoznačenih podatkov in zagotavlja skupine glede na vzorce, ki jih zazna v podatkih.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Kliknite zgornjo sliko za video. Medtem ko študirate strojno učenje z razvrščanjem v skupine, uživajte ob nigerijskih plesnih skladbah - to je visoko ocenjena pesem iz leta 2014 skupine PSquare.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

### Uvod

[Razvrščanje v skupine](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) je zelo uporabno za raziskovanje podatkov. Poglejmo, ali lahko pomaga odkriti trende in vzorce v načinu, kako nigerijsko občinstvo uživa glasbo.

✅ Vzemite si trenutek in razmislite o uporabi razvrščanja v skupine. V resničnem življenju se razvrščanje zgodi, kadar imate kup perila in morate razvrstiti oblačila družinskih članov 🧦👕👖🩲. V podatkovni znanosti se razvrščanje zgodi, ko poskušate analizirati uporabnikove preference ali določiti značilnosti katerega koli neoznačenega podatkovnega niza. Razvrščanje na nek način pomaga razumeti kaos, kot je predal za nogavice.

[![Uvod v strojno učenje](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Kliknite zgornjo sliko za video: John Guttag z MIT-a predstavlja razvrščanje v skupine.

V profesionalnem okolju se razvrščanje lahko uporablja za določanje stvari, kot je segmentacija trga, na primer za ugotavljanje, katere starostne skupine kupujejo določene izdelke. Druga uporaba bi bila odkrivanje anomalij, morda za zaznavanje goljufij iz podatkovnega niza transakcij s kreditnimi karticami. Lahko pa uporabite razvrščanje za določanje tumorjev v seriji medicinskih skenov.

✅ Razmislite za trenutek, kako ste morda naleteli na razvrščanje 'v naravi', v bančništvu, e-trgovini ali poslovnem okolju.

> 🎓 Zanimivo je, da analiza skupin izvira iz področij antropologije in psihologije v 30. letih prejšnjega stoletja. Si lahko predstavljate, kako bi jo takrat uporabljali?

Druga možnost je, da jo uporabite za razvrščanje rezultatov iskanja - na primer po nakupovalnih povezavah, slikah ali ocenah. Razvrščanje je uporabno, kadar imate velik podatkovni niz, ki ga želite zmanjšati in na katerem želite opraviti bolj podrobno analizo, zato se tehnika lahko uporablja za spoznavanje podatkov, preden se zgradijo drugi modeli.

✅ Ko so vaši podatki organizirani v skupine, jim dodelite ID skupine, kar je lahko uporabno pri ohranjanju zasebnosti podatkovnega niza; namesto bolj razkrivajočih identifikacijskih podatkov se lahko sklicujete na podatkovno točko z njenim ID-jem skupine. Ali lahko pomislite na druge razloge, zakaj bi se sklicevali na ID skupine namesto na druge elemente skupine za identifikacijo?

Poglobite svoje razumevanje tehnik razvrščanja v tem [učnem modulu](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Začetek z razvrščanjem v skupine

[Scikit-learn ponuja širok nabor](https://scikit-learn.org/stable/modules/clustering.html) metod za izvajanje razvrščanja v skupine. Vrsta, ki jo izberete, bo odvisna od vašega primera uporabe. Po dokumentaciji ima vsaka metoda različne prednosti. Tukaj je poenostavljena tabela metod, ki jih podpira Scikit-learn, in njihovih ustreznih primerov uporabe:

| Ime metode                 | Primer uporabe                                                        |
| :-------------------------- | :-------------------------------------------------------------------- |
| K-Means                    | splošna uporaba, induktivna                                           |
| Affinity propagation       | številne, neenakomerne skupine, induktivna                            |
| Mean-shift                 | številne, neenakomerne skupine, induktivna                            |
| Spectral clustering        | malo, enakomerne skupine, transduktivna                               |
| Ward hierarchical clustering | številne, omejene skupine, transduktivna                             |
| Agglomerative clustering   | številne, omejene, neevklidske razdalje, transduktivna                |
| DBSCAN                     | neploska geometrija, neenakomerne skupine, transduktivna              |
| OPTICS                     | neploska geometrija, neenakomerne skupine z različno gostoto, transduktivna |
| Gaussian mixtures          | ploska geometrija, induktivna                                         |
| BIRCH                      | velik podatkovni niz z odstopanji, induktivna                         |

> 🎓 Kako ustvarjamo skupine, je močno povezano s tem, kako združujemo podatkovne točke v skupine. Razložimo nekaj terminologije:
>
> 🎓 ['Transduktivno' vs. 'induktivno'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktivno sklepanje izhaja iz opazovanih primerov usposabljanja, ki se preslikajo na specifične testne primere. Induktivno sklepanje izhaja iz primerov usposabljanja, ki se preslikajo na splošna pravila, ki se nato uporabijo na testnih primerih.
> 
> Primer: Predstavljajte si, da imate podatkovni niz, ki je le delno označen. Nekatere stvari so 'plošče', nekatere 'CD-ji', nekatere pa so prazne. Vaša naloga je zagotoviti oznake za prazne. Če izberete induktivni pristop, bi usposobili model, ki išče 'plošče' in 'CD-je', ter te oznake uporabili na neoznačenih podatkih. Ta pristop bo imel težave pri razvrščanju stvari, ki so dejansko 'kasete'. Transduktivni pristop pa učinkoviteje obravnava te neznane podatke, saj deluje na združevanju podobnih predmetov in nato dodeli oznako skupini. V tem primeru bi skupine lahko odražale 'okrogle glasbene stvari' in 'kvadratne glasbene stvari'.
> 
> 🎓 ['Neploska' vs. 'ploska' geometrija](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Izpeljano iz matematične terminologije, neploska vs. ploska geometrija se nanaša na merjenje razdalj med točkami bodisi s 'plosko' ([evklidsko](https://wikipedia.org/wiki/Euclidean_geometry)) bodisi z 'neplosko' (neevklidsko) geometrijsko metodo.
>
>'Ploska' v tem kontekstu se nanaša na evklidsko geometrijo (deli katere se učijo kot 'ravninska' geometrija), medtem ko se 'neploska' nanaša na neevklidsko geometrijo. Kaj ima geometrija skupnega s strojno inteligenco? Kot dve področji, ki temeljita na matematiki, mora obstajati skupen način merjenja razdalj med točkami v skupinah, kar se lahko izvede na 'ploski' ali 'neploski' način, odvisno od narave podatkov. [Evklidske razdalje](https://wikipedia.org/wiki/Euclidean_distance) se merijo kot dolžina odseka med dvema točkama. [Neevklidske razdalje](https://wikipedia.org/wiki/Non-Euclidean_geometry) se merijo vzdolž krivulje. Če se vaši podatki, vizualizirani, ne nahajajo na ravnini, boste morda morali uporabiti specializiran algoritem za obravnavo.
>
![Ploska vs. neploska geometrija Infografika](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Razdalje'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Skupine so opredeljene z matriko razdalj, npr. razdaljami med točkami. Te razdalje je mogoče meriti na več načinov. Evklidske skupine so opredeljene z povprečjem vrednosti točk in vsebujejo 'centroid' ali osrednjo točko. Razdalje se tako merijo glede na razdaljo do tega centroida. Neevklidske razdalje se nanašajo na 'clustroid', točko, ki je najbližja drugim točkam. Clustroidi so lahko opredeljeni na različne načine.
> 
> 🎓 ['Omejeno'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Omejeno razvrščanje](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) uvaja 'polnadzorovano' učenje v to nenadzorovano metodo. Razmerja med točkami so označena kot 'ne smejo se povezati' ali 'morajo se povezati', tako da se na podatkovni niz vsilijo nekatera pravila.
>
>Primer: Če je algoritem sproščen na seriji neoznačenih ali delno označenih podatkov, so lahko skupine, ki jih ustvari, slabe kakovosti. V zgornjem primeru bi skupine lahko združevale 'okrogle glasbene stvari', 'kvadratne glasbene stvari', 'trikotne stvari' in 'piškote'. Če so podane nekatere omejitve ali pravila ("predmet mora biti iz plastike", "predmet mora biti sposoben proizvajati glasbo"), to lahko pomaga 'omejiti' algoritem, da sprejme boljše odločitve.
> 
> 🎓 'Gostota'
> 
> Podatki, ki so 'hrupni', se štejejo za 'goste'. Razdalje med točkami v vsaki od njihovih skupin se lahko ob pregledu izkažejo za bolj ali manj goste ali 'natrpane', zato je treba te podatke analizirati z ustrezno metodo razvrščanja. [Ta članek](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) prikazuje razliko med uporabo algoritmov K-Means in HDBSCAN za raziskovanje hrupnega podatkovnega niza z neenakomerno gostoto skupin.

## Algoritmi za razvrščanje v skupine

Obstaja več kot 100 algoritmov za razvrščanje v skupine, njihova uporaba pa je odvisna od narave podatkov. Oglejmo si nekatere glavne:

- **Hierarhično razvrščanje**. Če je predmet razvrščen glede na svojo bližino bližnjemu predmetu, namesto bolj oddaljenemu, se skupine oblikujejo na podlagi razdalje njihovih članov do in od drugih predmetov. Scikit-learnova aglomerativna razvrstitev je hierarhična.

   ![Hierarhično razvrščanje Infografika](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Razvrščanje po centroidih**. Ta priljubljen algoritem zahteva izbiro 'k', ali število skupin, ki jih je treba oblikovati, nato pa algoritem določi osrednjo točko skupine in zbira podatke okoli te točke. [K-means razvrščanje](https://wikipedia.org/wiki/K-means_clustering) je priljubljena različica razvrščanja po centroidih. Center je določen glede na najbližje povprečje, od tod tudi ime. Kvadratna razdalja od skupine je minimizirana.

   ![Razvrščanje po centroidih Infografika](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografika avtorja [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Razvrščanje na podlagi porazdelitve**. Temelji na statističnem modeliranju, razvrščanje na podlagi porazdelitve se osredotoča na določanje verjetnosti, da podatkovna točka pripada skupini, in ji ustrezno dodeli mesto. Metode Gaussovih mešanic spadajo v to vrsto.

- **Razvrščanje na podlagi gostote**. Podatkovne točke so dodeljene skupinam glede na njihovo gostoto ali njihovo združevanje okoli drugih točk. Podatkovne točke, ki so daleč od skupine, se štejejo za odstopanja ali hrup. DBSCAN, Mean-shift in OPTICS spadajo v to vrsto razvrščanja.

- **Razvrščanje na podlagi mreže**. Za večdimenzionalne podatkovne nize se ustvari mreža, podatki pa se razdelijo med celice mreže, s čimer se ustvarijo skupine.

## Vaja - razvrstite svoje podatke

Razvrščanje kot tehnika je močno podprto z ustrezno vizualizacijo, zato začnimo z vizualizacijo naših glasbenih podatkov. Ta vaja nam bo pomagala odločiti, katero metodo razvrščanja bi bilo najbolj učinkovito uporabiti glede na naravo teh podatkov.

1. Odprite datoteko [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) v tej mapi.

1. Uvozite paket `Seaborn` za dobro vizualizacijo podatkov.

    ```python
    !pip install seaborn
    ```

1. Dodajte podatke o pesmih iz [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Naložite podatkovni okvir z nekaterimi podatki o pesmih. Pripravite se na raziskovanje teh podatkov z uvozom knjižnic in izpisom podatkov:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Preverite prvih nekaj vrstic podatkov:

    |     | ime                     | album                        | izvajalec           | glavni žanr izvajalca | datum izdaje | dolžina | priljubljenost | plesnost     | akustičnost | energija | instrumentalnost | živost  | glasnost | govornost   | tempo   | časovni podpis |
    | --- | ------------------------ | ---------------------------- | ------------------- | --------------------- | ------------ | ------ | ------------- | ------------ | ------------ | ------ | ---------------- | ------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternativni r&b      | 2019         | 144000 | 48            | 0.666        | 0.851        | 0.42   | 0.534            | 0.11    | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop               | 2020         | 89488  | 30            | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101   | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Pridobite nekaj informacij o podatkovnem okviru z uporabo `info()`:

    ```python
    df.info()
    ```

   Izhod je videti takole:

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

1. Dvakrat preverite, ali obstajajo manjkajoče vrednosti, tako da pokličete `isnull()` in preverite, da je vsota 0:

    ```python
    df.isnull().sum()
    ```

    Videti je dobro:

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

1. Opis podatkov:

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

> 🤔 Če delamo s klastriranjem, nenadzorovano metodo, ki ne zahteva označenih podatkov, zakaj prikazujemo te podatke z oznakami? V fazi raziskovanja podatkov so koristne, vendar za delovanje algoritmov klastriranja niso nujno potrebne. Stolpčne oznake bi lahko odstranili in se sklicevali na podatke po številki stolpca.

Poglejte splošne vrednosti podatkov. Upoštevajte, da priljubljenost lahko znaša '0', kar kaže na pesmi, ki nimajo uvrstitve. Te bomo kmalu odstranili.

1. Uporabite stolpčni graf za ugotavljanje najbolj priljubljenih žanrov:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![najbolj priljubljeni](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Če želite videti več najvišjih vrednosti, spremenite zgornji `[:5]` v večjo vrednost ali ga odstranite, da vidite vse.

Upoštevajte, da ko je najvišji žanr opisan kot 'Missing', to pomeni, da ga Spotify ni razvrstil, zato ga odstranimo.

1. Odstranite manjkajoče podatke z njihovo filtracijo

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Zdaj ponovno preverite žanre:

    ![vsi žanri](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Trije najboljši žanri močno prevladujejo v tem naboru podatkov. Osredotočimo se na `afro dancehall`, `afropop` in `nigerian pop`, dodatno filtrirajmo nabor podatkov, da odstranimo vse z vrednostjo priljubljenosti 0 (kar pomeni, da ni bilo razvrščeno glede na priljubljenost v naboru podatkov in se lahko za naše namene šteje kot šum):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Hitro preverite, ali podatki močno korelirajo na kakšen poseben način:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korelacije](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Edina močna korelacija je med `energy` in `loudness`, kar ni preveč presenetljivo, saj je glasna glasba običajno precej energična. Sicer so korelacije razmeroma šibke. Zanimivo bo videti, kaj lahko algoritem klastriranja naredi iz teh podatkov.

    > 🎓 Upoštevajte, da korelacija ne pomeni vzročnosti! Imamo dokaz korelacije, vendar ne dokaz vzročnosti. [Zabavna spletna stran](https://tylervigen.com/spurious-correlations) ima nekaj vizualizacij, ki poudarjajo to točko.

Ali obstaja kakšna konvergenca v tem naboru podatkov glede na zaznano priljubljenost pesmi in plesnost? FacetGrid kaže, da obstajajo koncentrični krogi, ki se ujemajo, ne glede na žanr. Ali je mogoče, da se nigerijski okusi za ta žanr konvergirajo na določeni ravni plesnosti?

✅ Preizkusite različne podatkovne točke (energija, glasnost, govorljivost) in več ali različne glasbene žanre. Kaj lahko odkrijete? Oglejte si tabelo `df.describe()` za splošno razporeditev podatkovnih točk.

### Naloga - razporeditev podatkov

Ali se ti trije žanri bistveno razlikujejo v zaznavanju njihove plesnosti glede na njihovo priljubljenost?

1. Preučite razporeditev podatkov za priljubljenost in plesnost naših treh najboljših žanrov vzdolž dane osi x in y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Lahko odkrijete koncentrične kroge okoli splošne točke konvergence, ki prikazujejo razporeditev točk.

    > 🎓 Upoštevajte, da ta primer uporablja graf KDE (Kernel Density Estimate), ki predstavlja podatke z uporabo kontinuirane krivulje gostote verjetnosti. To nam omogoča interpretacijo podatkov pri delu z več razporeditvami.

    Na splošno se trije žanri ohlapno uskladijo glede na njihovo priljubljenost in plesnost. Določanje skupin v teh ohlapno usklajenih podatkih bo izziv:

    ![razporeditev](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Ustvarite razpršeni graf:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Razpršeni graf na istih oseh kaže podoben vzorec konvergence

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Na splošno lahko za klastriranje uporabite razpršene grafe za prikaz skupin podatkov, zato je obvladovanje te vrste vizualizacije zelo koristno. V naslednji lekciji bomo uporabili te filtrirane podatke in uporabili klastriranje k-means za odkrivanje skupin v teh podatkih, ki se zanimivo prekrivajo.

---

## 🚀Izziv

V pripravi na naslednjo lekcijo naredite graf o različnih algoritmih klastriranja, ki jih lahko odkrijete in uporabite v produkcijskem okolju. Kakšne vrste težav poskuša klastriranje rešiti?

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Preden uporabite algoritme klastriranja, kot smo se naučili, je dobro razumeti naravo vašega nabora podatkov. Preberite več o tej temi [tukaj](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ta koristen članek](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vas vodi skozi različne načine, kako se različni algoritmi klastriranja obnašajo glede na različne oblike podatkov.

## Naloga

[Raziskujte druge vizualizacije za klastriranje](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za prevajanje z umetno inteligenco [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni človeški prevod. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki bi nastale zaradi uporabe tega prevoda.