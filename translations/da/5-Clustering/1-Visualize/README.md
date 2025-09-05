<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T00:00:40+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "da"
}
-->
# Introduktion til clustering

Clustering er en type [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning), der antager, at et datasæt er ulabeleret, eller at dets input ikke er matchet med foruddefinerede output. Det bruger forskellige algoritmer til at sortere gennem ulabeleret data og levere grupperinger baseret på mønstre, det identificerer i dataene.

[![No One Like You af PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You af PSquare")

> 🎥 Klik på billedet ovenfor for en video. Mens du studerer maskinlæring med clustering, kan du nyde nogle nigerianske Dance Hall-numre - dette er en højt vurderet sang fra 2014 af PSquare.

## [Quiz før forelæsning](https://ff-quizzes.netlify.app/en/ml/)

### Introduktion

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) er meget nyttigt til dataudforskning. Lad os se, om det kan hjælpe med at opdage tendenser og mønstre i den måde, nigerianske publikum forbruger musik på.

✅ Tag et øjeblik til at tænke over anvendelserne af clustering. I hverdagen sker clustering, når du har en bunke vasketøj og skal sortere familiens tøj 🧦👕👖🩲. I datavidenskab sker clustering, når man forsøger at analysere en brugers præferencer eller bestemme egenskaberne for et ulabeleret datasæt. Clustering hjælper på en måde med at skabe orden i kaos, som en sokkeskuffe.

[![Introduktion til ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduktion til Clustering")

> 🎥 Klik på billedet ovenfor for en video: MIT's John Guttag introducerer clustering

I en professionel kontekst kan clustering bruges til at bestemme ting som markedssegmentering, f.eks. hvilke aldersgrupper der køber hvilke varer. En anden anvendelse kunne være anomalidetektion, måske for at opdage svindel i et datasæt med kreditkorttransaktioner. Eller du kunne bruge clustering til at identificere tumorer i en samling af medicinske scanninger.

✅ Tænk et øjeblik over, hvordan du måske har stødt på clustering 'i det virkelige liv', i en bank-, e-handels- eller forretningskontekst.

> 🎓 Interessant nok stammer clusteranalyse fra antropologi og psykologi i 1930'erne. Kan du forestille dig, hvordan det kunne være blevet brugt?

Alternativt kunne du bruge det til at gruppere søgeresultater - f.eks. efter shoppinglinks, billeder eller anmeldelser. Clustering er nyttigt, når du har et stort datasæt, som du vil reducere og udføre mere detaljeret analyse på, så teknikken kan bruges til at lære om data, før andre modeller konstrueres.

✅ Når dine data er organiseret i klynger, tildeler du dem et cluster-id, og denne teknik kan være nyttig til at bevare et datasæts privatliv; du kan i stedet referere til et datapunkt ved dets cluster-id frem for mere afslørende identificerbare data. Kan du komme på andre grunde til, hvorfor du ville referere til et cluster-id frem for andre elementer i klyngen for at identificere det?

Uddyb din forståelse af clustering-teknikker i dette [Learn-modul](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Kom godt i gang med clustering

[Scikit-learn tilbyder et stort udvalg](https://scikit-learn.org/stable/modules/clustering.html) af metoder til at udføre clustering. Den type, du vælger, afhænger af din anvendelsessituation. Ifølge dokumentationen har hver metode forskellige fordele. Her er en forenklet tabel over de metoder, der understøttes af Scikit-learn, og deres passende anvendelsessituationer:

| Metodenavn                   | Anvendelsessituation                                                  |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | generelt formål, induktiv                                             |
| Affinity propagation         | mange, ujævne klynger, induktiv                                       |
| Mean-shift                   | mange, ujævne klynger, induktiv                                       |
| Spectral clustering          | få, jævne klynger, transduktiv                                       |
| Ward hierarchical clustering | mange, begrænsede klynger, transduktiv                               |
| Agglomerative clustering     | mange, begrænsede, ikke-Euklidiske afstande, transduktiv             |
| DBSCAN                       | ikke-flad geometri, ujævne klynger, transduktiv                      |
| OPTICS                       | ikke-flad geometri, ujævne klynger med variabel tæthed, transduktiv  |
| Gaussian mixtures            | flad geometri, induktiv                                              |
| BIRCH                        | stort datasæt med outliers, induktiv                                 |

> 🎓 Hvordan vi skaber klynger har meget at gøre med, hvordan vi samler datapunkterne i grupper. Lad os pakke noget terminologi ud:
>
> 🎓 ['Transduktiv' vs. 'induktiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiv inferens er afledt af observerede træningstilfælde, der kortlægges til specifikke testtilfælde. Induktiv inferens er afledt af træningstilfælde, der kortlægges til generelle regler, som først derefter anvendes på testtilfælde.
> 
> Et eksempel: Forestil dig, at du har et datasæt, der kun delvist er labeleret. Nogle ting er 'plader', nogle 'cd'er', og nogle er tomme. Din opgave er at give labels til de tomme. Hvis du vælger en induktiv tilgang, ville du træne en model, der leder efter 'plader' og 'cd'er', og anvende disse labels på dine ulabelerede data. Denne tilgang vil have svært ved at klassificere ting, der faktisk er 'kassetter'. En transduktiv tilgang, derimod, håndterer disse ukendte data mere effektivt, da den arbejder på at gruppere lignende ting sammen og derefter anvender en label til en gruppe. I dette tilfælde kunne klynger afspejle 'runde musikting' og 'firkantede musikting'.
> 
> 🎓 ['Ikke-flad' vs. 'flad' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Afledt af matematisk terminologi refererer ikke-flad vs. flad geometri til målingen af afstande mellem punkter ved enten 'flade' ([Euklidiske](https://wikipedia.org/wiki/Euclidean_geometry)) eller 'ikke-flade' (ikke-Euklidiske) geometriske metoder.
>
>'Flad' i denne sammenhæng refererer til Euklidisk geometri (dele af det undervises som 'plan' geometri), og ikke-flad refererer til ikke-Euklidisk geometri. Hvad har geometri med maskinlæring at gøre? Som to felter, der er rodfæstet i matematik, skal der være en fælles måde at måle afstande mellem punkter i klynger, og det kan gøres på en 'flad' eller 'ikke-flad' måde, afhængigt af dataens natur. [Euklidiske afstande](https://wikipedia.org/wiki/Euclidean_distance) måles som længden af en linjesegment mellem to punkter. [Ikke-Euklidiske afstande](https://wikipedia.org/wiki/Non-Euclidean_geometry) måles langs en kurve. Hvis dine data, visualiseret, synes ikke at eksistere på et plan, kan du have brug for en specialiseret algoritme til at håndtere det.
>
![Flad vs Ikke-flad Geometri Infografik](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Afstande'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klynger defineres af deres afstandsmatrix, f.eks. afstandene mellem punkter. Denne afstand kan måles på flere måder. Euklidiske klynger defineres af gennemsnittet af punktværdierne og indeholder et 'centroid' eller midtpunkt. Afstande måles således ved afstanden til dette centroid. Ikke-Euklidiske afstande refererer til 'clustroids', punktet tættest på andre punkter. Clustroids kan igen defineres på forskellige måder.
> 
> 🎓 ['Begrænset'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Begrænset Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introducerer 'semi-supervised' læring i denne unsupervised metode. Forholdene mellem punkter markeres som 'kan ikke linkes' eller 'skal linkes', så nogle regler tvinges på datasættet.
>
>Et eksempel: Hvis en algoritme sættes fri på en batch af ulabelerede eller semi-labelerede data, kan de klynger, den producerer, være af dårlig kvalitet. I eksemplet ovenfor kunne klyngerne gruppere 'runde musikting' og 'firkantede musikting' og 'trekantede ting' og 'kager'. Hvis der gives nogle begrænsninger eller regler at følge ("genstanden skal være lavet af plastik", "genstanden skal kunne producere musik"), kan dette hjælpe med at 'begrænse' algoritmen til at træffe bedre valg.
> 
> 🎓 'Tæthed'
> 
> Data, der er 'støjende', betragtes som 'tæt'. Afstandene mellem punkter i hver af dets klynger kan vise sig, ved undersøgelse, at være mere eller mindre tætte eller 'overfyldte', og derfor skal disse data analyseres med den passende clustering-metode. [Denne artikel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrerer forskellen mellem at bruge K-Means clustering vs. HDBSCAN-algoritmer til at udforske et støjende datasæt med ujævn klyngedensitet.

## Clustering-algoritmer

Der findes over 100 clustering-algoritmer, og deres anvendelse afhænger af dataens natur. Lad os diskutere nogle af de vigtigste:

- **Hierarkisk clustering**. Hvis et objekt klassificeres efter dets nærhed til et nærliggende objekt frem for et længere væk, dannes klynger baseret på deres medlemmers afstand til og fra andre objekter. Scikit-learns agglomerative clustering er hierarkisk.

   ![Hierarkisk clustering Infografik](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. Denne populære algoritme kræver valg af 'k', eller antallet af klynger, der skal dannes, hvorefter algoritmen bestemmer midtpunktet for en klynge og samler data omkring dette punkt. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) er en populær version af centroid clustering. Midtpunktet bestemmes af det nærmeste gennemsnit, deraf navnet. Den kvadrerede afstand fra klyngen minimeres.

   ![Centroid clustering Infografik](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distributionsbaseret clustering**. Baseret på statistisk modellering fokuserer distributionsbaseret clustering på at bestemme sandsynligheden for, at et datapunkt tilhører en klynge, og tildeler det derefter. Gaussian mixture-metoder tilhører denne type.

- **Tæthetsbaseret clustering**. Datapunkter tildeles klynger baseret på deres tæthed eller deres gruppering omkring hinanden. Datapunkter langt fra gruppen betragtes som outliers eller støj. DBSCAN, Mean-shift og OPTICS tilhører denne type clustering.

- **Grid-baseret clustering**. For multidimensionelle datasæt oprettes et gitter, og dataene opdeles blandt gitterets celler, hvilket skaber klynger.

## Øvelse - cluster dine data

Clustering som teknik understøttes i høj grad af korrekt visualisering, så lad os komme i gang med at visualisere vores musikdata. Denne øvelse vil hjælpe os med at beslutte, hvilken af metoderne til clustering vi mest effektivt bør bruge til dataens natur.

1. Åbn filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) i denne mappe.

1. Importér pakken `Seaborn` for god datavisualisering.

    ```python
    !pip install seaborn
    ```

1. Tilføj sangdataene fra [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Indlæs en dataframe med nogle data om sangene. Gør dig klar til at udforske disse data ved at importere bibliotekerne og udskrive dataene:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Tjek de første par linjer af data:

    |     | navn                     | album                        | kunstner            | kunstner_top_genre | udgivelsesdato | længde | popularitet | dansbarhed   | akustiskhed | energi | instrumentalisme | livlighed | lydstyrke | talbarhed   | tempo   | taktart         |
    | --- | ------------------------ | ---------------------------- | ------------------- | ------------------ | -------------- | ------ | ----------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternativ r&b     | 2019           | 144000 | 48          | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop            | 2020           | 89488  | 30          | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Få nogle oplysninger om dataframe ved at kalde `info()`:

    ```python
    df.info()
    ```

   Output ser sådan ud:

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

1. Dobbelt-tjek for null-værdier ved at kalde `isnull()` og verificere, at summen er 0:

    ```python
    df.isnull().sum()
    ```

    Ser godt ud:

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

1. Beskriv data:

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

> 🤔 Hvis vi arbejder med clustering, en usuperviseret metode, der ikke kræver mærkede data, hvorfor viser vi så disse data med labels? I dataudforskningsfasen er de nyttige, men de er ikke nødvendige for, at clustering-algoritmerne fungerer. Du kunne lige så godt fjerne kolonneoverskrifterne og referere til dataene ved kolonnenummer.

Se på de generelle værdier i dataene. Bemærk, at popularitet kan være '0', hvilket viser sange, der ikke har nogen rangering. Lad os fjerne dem snart.

1. Brug et søjlediagram til at finde de mest populære genrer:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![mest populære](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Hvis du vil se flere topværdier, ændr `[:5]` til en større værdi, eller fjern det for at se alle.

Bemærk, når den øverste genre er beskrevet som 'Missing', betyder det, at Spotify ikke har klassificeret den, så lad os fjerne den.

1. Fjern manglende data ved at filtrere dem ud

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Tjek nu genrerne igen:

    ![mest populære](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. De tre øverste genrer dominerer klart dette datasæt. Lad os koncentrere os om `afro dancehall`, `afropop` og `nigerian pop`, og yderligere filtrere datasættet for at fjerne alt med en popularitetsværdi på 0 (hvilket betyder, at det ikke blev klassificeret med en popularitet i datasættet og kan betragtes som støj for vores formål):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Lav en hurtig test for at se, om dataene korrelerer på nogen særlig stærk måde:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korrelationer](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Den eneste stærke korrelation er mellem `energy` og `loudness`, hvilket ikke er så overraskende, da høj musik normalt er ret energisk. Ellers er korrelationerne relativt svage. Det vil være interessant at se, hvad en clustering-algoritme kan få ud af disse data.

    > 🎓 Bemærk, at korrelation ikke indebærer årsagssammenhæng! Vi har bevis for korrelation, men ingen bevis for årsagssammenhæng. En [sjov hjemmeside](https://tylervigen.com/spurious-correlations) har nogle visualiseringer, der understreger dette punkt.

Er der nogen konvergens i dette datasæt omkring en sangs opfattede popularitet og dansbarhed? En FacetGrid viser, at der er koncentriske cirkler, der stemmer overens, uanset genre. Kunne det være, at nigerianske smag konvergerer på et bestemt niveau af dansbarhed for denne genre?  

✅ Prøv forskellige datapunkter (energy, loudness, speechiness) og flere eller andre musikgenrer. Hvad kan du opdage? Tag et kig på `df.describe()`-tabellen for at se den generelle spredning af datapunkterne.

### Øvelse - datafordeling

Er disse tre genrer markant forskellige i opfattelsen af deres dansbarhed, baseret på deres popularitet?

1. Undersøg datafordelingen for vores tre øverste genrer for popularitet og dansbarhed langs en given x- og y-akse.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Du kan opdage koncentriske cirkler omkring et generelt konvergenspunkt, der viser fordelingen af punkter.

    > 🎓 Bemærk, at dette eksempel bruger en KDE (Kernel Density Estimate)-graf, der repræsenterer dataene ved hjælp af en kontinuerlig sandsynlighedstæthedskurve. Dette giver os mulighed for at fortolke data, når vi arbejder med flere fordelinger.

    Generelt stemmer de tre genrer løst overens med hensyn til deres popularitet og dansbarhed. At bestemme klynger i disse løst tilpassede data vil være en udfordring:

    ![fordeling](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Lav et scatterplot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Et scatterplot af de samme akser viser et lignende mønster af konvergens

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Generelt kan du bruge scatterplots til at vise klynger af data, så det er meget nyttigt at mestre denne type visualisering. I næste lektion vil vi tage disse filtrerede data og bruge k-means clustering til at opdage grupper i disse data, der ser ud til at overlappe på interessante måder.

---

## 🚀Udfordring

Som forberedelse til næste lektion, lav et diagram over de forskellige clustering-algoritmer, du måske opdager og bruger i et produktionsmiljø. Hvilke slags problemer forsøger clustering at løse?

## [Quiz efter lektion](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Før du anvender clustering-algoritmer, som vi har lært, er det en god idé at forstå naturen af dit datasæt. Læs mere om dette emne [her](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Denne nyttige artikel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) guider dig gennem de forskellige måder, som forskellige clustering-algoritmer opfører sig på, givet forskellige datatyper.

## Opgave

[Undersøg andre visualiseringer for clustering](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på at sikre nøjagtighed, skal det bemærkes, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.