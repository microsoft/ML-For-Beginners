<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T15:40:02+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "sw"
}
-->
# Utangulizi wa clustering

Clustering ni aina ya [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning) inayodhani kuwa dataset haina lebo au kwamba maingizo yake hayajafungamanishwa na matokeo yaliyotanguliwa. Inatumia algorithmi mbalimbali kuchambua data isiyo na lebo na kutoa makundi kulingana na mifumo inayotambua kwenye data.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Bofya picha hapo juu kwa video. Unapojifunza machine learning kwa kutumia clustering, furahia nyimbo za Dance Hall za Nigeria - hii ni wimbo uliopendwa sana kutoka 2014 na PSquare.

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

### Utangulizi

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) ni muhimu sana kwa uchunguzi wa data. Hebu tuone kama inaweza kusaidia kugundua mitindo na mifumo katika jinsi hadhira ya Nigeria inavyotumia muziki.

✅ Chukua dakika moja kufikiria matumizi ya clustering. Katika maisha ya kila siku, clustering hutokea kila unapokuwa na rundo la nguo na unahitaji kupanga nguo za wanafamilia wako 🧦👕👖🩲. Katika data science, clustering hutokea unapojaribu kuchambua mapendeleo ya mtumiaji, au kubaini sifa za dataset yoyote isiyo na lebo. Kwa namna fulani, clustering husaidia kuleta mpangilio kwenye machafuko, kama droo ya soksi.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Bofya picha hapo juu kwa video: John Guttag wa MIT anatambulisha clustering.

Katika mazingira ya kitaalamu, clustering inaweza kutumika kubaini mambo kama mgawanyiko wa soko, kubaini ni makundi ya umri gani yanayonunua bidhaa fulani, kwa mfano. Matumizi mengine yanaweza kuwa kugundua hali zisizo za kawaida, labda kugundua udanganyifu kutoka dataset ya miamala ya kadi za mkopo. Au unaweza kutumia clustering kubaini uvimbe katika kundi la picha za uchunguzi wa matibabu.

✅ Fikiria kwa dakika moja jinsi unavyoweza kuwa umekutana na clustering 'katika mazingira halisi', katika benki, e-commerce, au mazingira ya biashara.

> 🎓 Kwa kushangaza, uchambuzi wa makundi ulianzia katika nyanja za Anthropolojia na Saikolojia katika miaka ya 1930. Je, unaweza kufikiria jinsi ulivyotumika?

Vinginevyo, unaweza kuitumia kwa kupanga matokeo ya utafutaji - kwa viungo vya ununuzi, picha, au hakiki, kwa mfano. Clustering ni muhimu unapokuwa na dataset kubwa unayotaka kupunguza na ambayo unataka kufanya uchambuzi wa kina zaidi, hivyo mbinu hii inaweza kutumika kujifunza kuhusu data kabla ya kujenga mifano mingine.

✅ Mara data yako inapopangwa katika makundi, unaiwekea kitambulisho cha kundi, na mbinu hii inaweza kuwa muhimu katika kuhifadhi faragha ya dataset; badala yake unaweza kurejelea data kwa kitambulisho cha kundi, badala ya data inayoweza kufichua zaidi. Je, unaweza kufikiria sababu nyingine za kutumia kitambulisho cha kundi badala ya vipengele vingine vya kundi kuvitambua?

Panua uelewa wako wa mbinu za clustering katika [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Kuanza na clustering

[Scikit-learn inatoa mbinu nyingi](https://scikit-learn.org/stable/modules/clustering.html) za kufanya clustering. Aina unayochagua itategemea matumizi yako. Kulingana na nyaraka, kila mbinu ina faida mbalimbali. Hapa kuna jedwali rahisi la mbinu zinazoungwa mkono na Scikit-learn na matumizi yake yanayofaa:

| Jina la mbinu               | Matumizi                                                                |
| :--------------------------- | :---------------------------------------------------------------------- |
| K-Means                      | matumizi ya jumla, inductive                                            |
| Affinity propagation         | makundi mengi, yasiyo sawa, inductive                                   |
| Mean-shift                   | makundi mengi, yasiyo sawa, inductive                                   |
| Spectral clustering          | makundi machache, sawa, transductive                                    |
| Ward hierarchical clustering | makundi mengi, yaliyowekewa mipaka, transductive                        |
| Agglomerative clustering     | makundi mengi, yaliyowekewa mipaka, umbali usio wa Euclidean, transductive |
| DBSCAN                       | jiometri isiyo tambarare, makundi yasiyo sawa, transductive             |
| OPTICS                       | jiometri isiyo tambarare, makundi yasiyo sawa yenye msongamano tofauti, transductive |
| Gaussian mixtures            | jiometri tambarare, inductive                                          |
| BIRCH                        | dataset kubwa yenye outliers, inductive                                |

> 🎓 Jinsi tunavyounda makundi inahusiana sana na jinsi tunavyokusanya pointi za data katika vikundi. Hebu tuchambue baadhi ya istilahi:
>
> 🎓 ['Transductive' vs. 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Utoaji wa hitimisho wa transductive hutokana na kesi za mafunzo zilizotazamwa ambazo zinahusiana na kesi maalum za majaribio. Utoaji wa hitimisho wa inductive hutokana na kesi za mafunzo ambazo zinahusiana na sheria za jumla ambazo baadaye tu zinatumika kwa kesi za majaribio.
> 
> Mfano: Fikiria una dataset ambayo imewekwa lebo kwa sehemu tu. Vitu vingine ni 'rekodi', vingine 'cds', na vingine havina lebo. Kazi yako ni kutoa lebo kwa data isiyo na lebo. Ukichagua mbinu ya inductive, ungefundisha mfano ukitafuta 'rekodi' na 'cds', na kutumia lebo hizo kwa data yako isiyo na lebo. Mbinu hii itakuwa na shida kuainisha vitu ambavyo kwa kweli ni 'kanda'. Mbinu ya transductive, kwa upande mwingine, hushughulikia data isiyojulikana kwa ufanisi zaidi kwani inafanya kazi kuunda vikundi vya vitu vinavyofanana na kisha kutumia lebo kwa kundi. Katika kesi hii, makundi yanaweza kuonyesha 'vitu vya muziki vya mviringo' na 'vitu vya muziki vya mraba'.
> 
> 🎓 ['Jiometri isiyo tambarare' vs. 'jiometri tambarare'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Imetokana na istilahi za hisabati, jiometri isiyo tambarare vs. tambarare inahusu kipimo cha umbali kati ya pointi kwa njia ya 'tambarare' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) au 'isiyo tambarare' (isiyo ya Euclidean).
>
>'Tambarare' katika muktadha huu inahusu jiometri ya Euclidean (sehemu zake hufundishwa kama jiometri ya 'plane'), na isiyo tambarare inahusu jiometri isiyo ya Euclidean. Jiometri inahusiana vipi na machine learning? Kweli, kama nyanja mbili zinazotokana na hisabati, lazima kuwe na njia ya kawaida ya kupima umbali kati ya pointi katika makundi, na hiyo inaweza kufanywa kwa njia ya 'tambarare' au 'isiyo tambarare', kulingana na asili ya data. [Umbali wa Euclidean](https://wikipedia.org/wiki/Euclidean_distance) hupimwa kama urefu wa sehemu ya mstari kati ya pointi mbili. [Umbali usio wa Euclidean](https://wikipedia.org/wiki/Non-Euclidean_geometry) hupimwa kando ya mkurva. Ikiwa data yako, ikionyeshwa, inaonekana haipo kwenye plane, unaweza kuhitaji kutumia algorithmi maalum kuishughulikia.
>
![Flat vs Nonflat Geometry Infographic](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infographic na [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Umbali'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Makundi yanafafanuliwa na matrix ya umbali, yaani umbali kati ya pointi. Umbali huu unaweza kupimwa kwa njia kadhaa. Makundi ya Euclidean yanafafanuliwa na wastani wa thamani za pointi, na yana 'centroid' au pointi ya katikati. Umbali hupimwa kwa umbali hadi centroid hiyo. Umbali usio wa Euclidean unahusu 'clustroids', pointi iliyo karibu zaidi na pointi nyingine. Clustroids kwa upande wake zinaweza kufafanuliwa kwa njia mbalimbali.
> 
> 🎓 ['Yaliyowekewa mipaka'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Clustering iliyowekewa mipaka](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) huanzisha 'semi-supervised' learning katika mbinu hii isiyo na usimamizi. Mahusiano kati ya pointi yanawekwa alama kama 'haiwezi kuunganishwa' au 'lazima yaunganishwe' hivyo sheria fulani zinapewa dataset.
>
>Mfano: Ikiwa algorithmi imeachwa huru kwenye kundi la data isiyo na lebo au yenye lebo kwa sehemu, makundi inayozalisha yanaweza kuwa ya ubora duni. Katika mfano hapo juu, makundi yanaweza kuunda 'vitu vya muziki vya mviringo' na 'vitu vya muziki vya mraba' na 'vitu vya pembetatu' na 'biskuti'. Ikiwa imepewa mipaka fulani, au sheria za kufuata ("kitu lazima kiwe cha plastiki", "kitu kinahitaji kuwa na uwezo wa kutoa muziki") hii inaweza kusaidia 'kuweka mipaka' kwa algorithmi kufanya chaguo bora.
> 
> 🎓 'Msongamano'
> 
> Data iliyo na 'kelele' inachukuliwa kuwa 'yenye msongamano'. Umbali kati ya pointi katika kila moja ya makundi yake unaweza kuonyesha, kwa uchunguzi, kuwa na msongamano zaidi au mdogo, au 'imejaa' na hivyo data hii inahitaji kuchambuliwa kwa mbinu sahihi ya clustering. [Makala hii](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) inaonyesha tofauti kati ya kutumia K-Means clustering vs. HDBSCAN algorithmi kuchunguza dataset yenye kelele na msongamano usio sawa.

## Algorithmi za clustering

Kuna zaidi ya algorithmi 100 za clustering, na matumizi yake yanategemea asili ya data iliyopo. Hebu tujadili baadhi ya zile kuu:

- **Hierarchical clustering**. Ikiwa kitu kinaainishwa kwa ukaribu wake na kitu kilicho karibu, badala ya kile kilicho mbali zaidi, makundi yanaundwa kulingana na umbali wa wanachama wake kwa na kutoka kwa vitu vingine. Agglomerative clustering ya Scikit-learn ni hierarchical.

   ![Hierarchical clustering Infographic](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infographic na [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. Algorithmi hii maarufu inahitaji kuchagua 'k', au idadi ya makundi ya kuunda, baada ya hapo algorithmi huamua pointi ya katikati ya kundi na kukusanya data karibu na pointi hiyo. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) ni toleo maarufu la centroid clustering. Kituo kinaamuliwa na wastani wa karibu, hivyo jina. Umbali wa mraba kutoka kwa kundi hupunguzwa.

   ![Centroid clustering Infographic](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infographic na [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribution-based clustering**. Ikitokana na uundaji wa takwimu, distribution-based clustering inazingatia kubaini uwezekano kwamba pointi ya data inahusiana na kundi, na kuipangia ipasavyo. Mbinu za Gaussian mixture zinahusiana na aina hii.

- **Density-based clustering**. Pointi za data zinapangiwa makundi kulingana na msongamano wao, au jinsi zinavyokusanyika karibu na kila moja. Pointi za data zilizo mbali na kundi zinachukuliwa kuwa outliers au kelele. DBSCAN, Mean-shift na OPTICS zinahusiana na aina hii ya clustering.

- **Grid-based clustering**. Kwa datasets zenye vipimo vingi, gridi huundwa na data hugawanywa kati ya seli za gridi hiyo, hivyo kuunda makundi.

## Zoezi - panga data yako

Clustering kama mbinu inasaidiwa sana na uonyeshaji sahihi wa data, kwa hivyo hebu tuanze kwa kuonyesha data yetu ya muziki. Zoezi hili litatusaidia kuamua ni mbinu gani za clustering tunazopaswa kutumia kwa ufanisi zaidi kwa asili ya data hii.

1. Fungua faili [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) katika folda hii.

1. Leta pakiti ya `Seaborn` kwa uonyeshaji mzuri wa data.

    ```python
    !pip install seaborn
    ```

1. Ongeza data ya nyimbo kutoka [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Pakia dataframe yenye data fulani kuhusu nyimbo. Jiandae kuchunguza data hii kwa kuleta maktaba na kutoa data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Angalia mistari michache ya data:

    |     | jina                     | albamu                       | msanii              | aina kuu ya msanii | tarehe ya kutolewa | urefu | umaarufu | uwezo wa kucheza | acousticness | nguvu | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ------------------ | ------------------ | ----- | -------- | ---------------- | ------------ | ----- | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b    | 2019               | 144000 | 48       | 0.666            | 0.851        | 0.42  | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop            | 2020               | 89488  | 30       | 0.71             | 0.0822       | 0.683 | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Pata maelezo kuhusu dataframe, kwa kutumia `info()`:

    ```python
    df.info()
    ```

   Matokeo yanaonekana kama hivi:

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

1. Hakikisha hakuna thamani za null, kwa kutumia `isnull()` na kuthibitisha jumla ni 0:

    ```python
    df.isnull().sum()
    ```

    Inaonekana vizuri:

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

1. Eleza data:

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

> 🤔 Ikiwa tunafanya kazi na clustering, mbinu isiyo ya usimamizi ambayo haihitaji data yenye lebo, kwa nini tunaonyesha data hii yenye lebo? Katika awamu ya uchunguzi wa data, zinafaa, lakini hazihitajiki kwa algorithms za clustering kufanya kazi. Unaweza tu kuondoa vichwa vya safu na kurejelea data kwa nambari ya safu.

Angalia maadili ya jumla ya data. Kumbuka kuwa popularity inaweza kuwa '0', ambayo inaonyesha nyimbo ambazo hazina kiwango. Wacha tuondoe hizo muda mfupi.

1. Tumia barplot kujua aina za muziki maarufu zaidi:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Ikiwa ungependa kuona maadili zaidi ya juu, badilisha juu `[:5]` kwa thamani kubwa, au iondoe ili kuona yote.

Kumbuka, wakati aina ya juu ya muziki inaelezewa kama 'Missing', hiyo inamaanisha kuwa Spotify haikuiweka daraja, kwa hivyo wacha tuiondoe.

1. Ondoa data iliyokosekana kwa kuichuja:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sasa angalia tena aina za muziki:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Kwa mbali, aina tatu za juu za muziki zinatawala dataset hii. Wacha tuzingatie `afro dancehall`, `afropop`, na `nigerian pop`, na pia tuchuje dataset ili kuondoa chochote chenye thamani ya popularity ya 0 (inamaanisha haikuwekwa daraja na popularity katika dataset na inaweza kuchukuliwa kama kelele kwa madhumuni yetu):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Fanya jaribio la haraka kuona ikiwa data inahusiana kwa njia yenye nguvu:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Uhusiano pekee wenye nguvu ni kati ya `energy` na `loudness`, ambayo si ya kushangaza sana, ikizingatiwa kuwa muziki wenye sauti kubwa kawaida huwa na nguvu. Vinginevyo, uhusiano ni dhaifu. Itakuwa ya kuvutia kuona kile algorithm ya clustering inaweza kufanya na data hii.

    > 🎓 Kumbuka kuwa uhusiano hauonyeshi sababu! Tuna ushahidi wa uhusiano lakini hakuna ushahidi wa sababu. [Tovuti ya kuchekesha](https://tylervigen.com/spurious-correlations) ina visuals zinazoonyesha hoja hii.

Je, kuna mwelekeo wowote katika dataset hii kuhusu umaarufu wa wimbo na uwezo wake wa kuchezeka? FacetGrid inaonyesha kuwa kuna miduara inayojipanga, bila kujali aina ya muziki. Inaweza kuwa ladha za Nigeria zinajipanga katika kiwango fulani cha uwezo wa kuchezeka kwa aina hii ya muziki?  

✅ Jaribu pointi tofauti za data (energy, loudness, speechiness) na aina zaidi au tofauti za muziki. Unaweza kugundua nini? Angalia jedwali la `df.describe()` ili kuona mwelekeo wa jumla wa pointi za data.

### Zoezi - usambazaji wa data

Je, aina hizi tatu za muziki zinatofautiana kwa kiasi kikubwa katika mtazamo wa uwezo wa kuchezeka, kulingana na umaarufu wao?

1. Chunguza usambazaji wa data wa aina zetu tatu za juu kwa umaarufu na uwezo wa kuchezeka kwenye x na y axis fulani.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Unaweza kugundua miduara inayozunguka sehemu ya mwelekeo wa jumla, ikionyesha usambazaji wa pointi.

    > 🎓 Kumbuka kuwa mfano huu unatumia grafu ya KDE (Kernel Density Estimate) ambayo inawakilisha data kwa kutumia curve ya probability density inayoendelea. Hii inatuwezesha kufasiri data tunapofanya kazi na usambazaji mwingi.

    Kwa ujumla, aina hizi tatu za muziki zinajipanga kwa kiasi fulani kulingana na umaarufu wao na uwezo wa kuchezeka. Kuamua makundi katika data hii inayojipanga kwa kiasi fulani itakuwa changamoto:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Unda scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot ya axes zile zile inaonyesha mwelekeo sawa wa mwelekeo

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Kwa ujumla, kwa clustering, unaweza kutumia scatterplots kuonyesha makundi ya data, kwa hivyo kujifunza aina hii ya visualisation ni muhimu sana. Katika somo linalofuata, tutachukua data hii iliyochujwa na kutumia k-means clustering kugundua makundi katika data hii ambayo yanaonekana kuingiliana kwa njia za kuvutia.

---

## 🚀Changamoto

Kwa maandalizi ya somo linalofuata, tengeneza chati kuhusu algorithms mbalimbali za clustering ambazo unaweza kugundua na kutumia katika mazingira ya uzalishaji. Ni aina gani za matatizo ambayo clustering inajaribu kushughulikia?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kujisomea

Kabla ya kutumia algorithms za clustering, kama tulivyojifunza, ni wazo nzuri kuelewa asili ya dataset yako. Soma zaidi kuhusu mada hii [hapa](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Makala hii ya msaada](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) inakutembeza kupitia njia tofauti ambazo algorithms za clustering zinavyofanya kazi, ikizingatiwa maumbo tofauti ya data.

## Kazi

[Chunguza visualizations nyingine za clustering](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.