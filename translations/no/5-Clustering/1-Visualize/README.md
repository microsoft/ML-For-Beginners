<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T21:27:24+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "no"
}
-->
# Introduksjon til klynging

Klynging er en type [Usupervisert læring](https://wikipedia.org/wiki/Unsupervised_learning) som forutsetter at et datasett er umerket eller at dets input ikke er koblet til forhåndsdefinerte output. Det bruker ulike algoritmer for å sortere gjennom umerket data og gi grupperinger basert på mønstre det oppdager i dataen.

[![No One Like You av PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You av PSquare")

> 🎥 Klikk på bildet over for en video. Mens du studerer maskinlæring med klynging, nyt noen nigerianske Dance Hall-låter – dette er en høyt rangert sang fra 2014 av PSquare.

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

### Introduksjon

[Klynging](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) er svært nyttig for datautforskning. La oss se om det kan hjelpe med å oppdage trender og mønstre i hvordan nigerianske publikum konsumerer musikk.

✅ Ta et øyeblikk til å tenke på bruksområdene for klynging. I hverdagen skjer klynging når du har en haug med klesvask og må sortere familiemedlemmenes klær 🧦👕👖🩲. I dataanalyse skjer klynging når man prøver å analysere en brukers preferanser eller bestemme egenskapene til et umerket datasett. Klynging hjelper på en måte med å skape orden i kaos, som en sokkeskuff.

[![Introduksjon til ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduksjon til klynging")

> 🎥 Klikk på bildet over for en video: MITs John Guttag introduserer klynging

I en profesjonell setting kan klynging brukes til å bestemme ting som markedssegmentering, for eksempel å finne ut hvilke aldersgrupper som kjøper hvilke varer. Et annet bruksområde kan være å oppdage avvik, kanskje for å oppdage svindel i et datasett med kredittkorttransaksjoner. Eller du kan bruke klynging til å identifisere svulster i en samling medisinske skanninger.

✅ Tenk et øyeblikk på hvordan du kan ha støtt på klynging 'i det virkelige liv', i en bank-, e-handels- eller forretningssetting.

> 🎓 Interessant nok oppsto klyngeanalyse innenfor feltene antropologi og psykologi på 1930-tallet. Kan du forestille deg hvordan det kan ha blitt brukt?

Alternativt kan du bruke det til å gruppere søkeresultater – for eksempel etter shoppinglenker, bilder eller anmeldelser. Klynging er nyttig når du har et stort datasett som du vil redusere og utføre mer detaljert analyse på, så teknikken kan brukes til å lære om data før andre modeller bygges.

✅ Når dataen din er organisert i klynger, tildeler du den en klynge-ID, og denne teknikken kan være nyttig for å bevare et datasets personvern; du kan i stedet referere til et datapunkt med klynge-ID-en, i stedet for med mer avslørende identifiserbare data. Kan du tenke på andre grunner til hvorfor du ville referere til en klynge-ID i stedet for andre elementer i klyngen for å identifisere den?

Fordyp deg i klyngingsteknikker i denne [Learn-modulen](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Komme i gang med klynging

[Scikit-learn tilbyr et stort utvalg](https://scikit-learn.org/stable/modules/clustering.html) av metoder for å utføre klynging. Typen du velger vil avhenge av bruksområdet ditt. Ifølge dokumentasjonen har hver metode ulike fordeler. Her er en forenklet tabell over metodene som støttes av Scikit-learn og deres passende bruksområder:

| Metodenavn                  | Bruksområde                                                             |
| :--------------------------- | :---------------------------------------------------------------------- |
| K-Means                      | generell bruk, induktiv                                                 |
| Affinity propagation         | mange, ujevne klynger, induktiv                                         |
| Mean-shift                   | mange, ujevne klynger, induktiv                                         |
| Spectral clustering          | få, jevne klynger, transduktiv                                          |
| Ward hierarchical clustering | mange, begrensede klynger, transduktiv                                  |
| Agglomerative clustering     | mange, begrensede, ikke-Euklidiske avstander, transduktiv               |
| DBSCAN                       | ikke-flat geometri, ujevne klynger, transduktiv                         |
| OPTICS                       | ikke-flat geometri, ujevne klynger med variabel tetthet, transduktiv    |
| Gaussian mixtures            | flat geometri, induktiv                                                 |
| BIRCH                        | stort datasett med uteliggere, induktiv                                 |

> 🎓 Hvordan vi lager klynger har mye å gjøre med hvordan vi samler datapunktene i grupper. La oss pakke ut litt vokabular:
>
> 🎓 ['Transduktiv' vs. 'induktiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiv inferens er avledet fra observerte treningscaser som kartlegges til spesifikke testcaser. Induktiv inferens er avledet fra treningscaser som kartlegges til generelle regler som deretter brukes på testcaser.
> 
> Et eksempel: Tenk deg at du har et datasett som bare delvis er merket. Noen ting er 'plater', noen 'CD-er', og noen er blanke. Din oppgave er å gi etiketter til de blanke. Hvis du velger en induktiv tilnærming, vil du trene en modell som ser etter 'plater' og 'CD-er', og bruke disse etikettene på din umerkede data. Denne tilnærmingen vil ha problemer med å klassifisere ting som faktisk er 'kassetter'. En transduktiv tilnærming, derimot, håndterer denne ukjente dataen mer effektivt ved å jobbe for å gruppere lignende elementer sammen og deretter bruke en etikett på en gruppe. I dette tilfellet kan klynger reflektere 'runde musikalske ting' og 'firkantede musikalske ting'.
> 
> 🎓 ['Ikke-flat' vs. 'flat' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Avledet fra matematisk terminologi, refererer ikke-flat vs. flat geometri til målingen av avstander mellom punkter ved enten 'flat' ([Euklidisk](https://wikipedia.org/wiki/Euclidean_geometry)) eller 'ikke-flat' (ikke-Euklidisk) geometriske metoder.
>
>'Flat' i denne sammenhengen refererer til Euklidisk geometri (deler av dette læres som 'plan' geometri), og ikke-flat refererer til ikke-Euklidisk geometri. Hva har geometri med maskinlæring å gjøre? Vel, som to felt som er forankret i matematikk, må det være en felles måte å måle avstander mellom punkter i klynger, og det kan gjøres på en 'flat' eller 'ikke-flat' måte, avhengig av dataens natur. [Euklidiske avstander](https://wikipedia.org/wiki/Euclidean_distance) måles som lengden på en linjesegment mellom to punkter. [Ikke-Euklidiske avstander](https://wikipedia.org/wiki/Non-Euclidean_geometry) måles langs en kurve. Hvis dataen din, visualisert, ser ut til å ikke eksistere på et plan, kan det hende du må bruke en spesialisert algoritme for å håndtere det.
>
![Flat vs Ikke-flat Geometri Infografikk](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Avstander'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klynger defineres av deres avstandsmatrise, f.eks. avstandene mellom punkter. Denne avstanden kan måles på flere måter. Euklidiske klynger defineres av gjennomsnittet av punktverdiene, og inneholder et 'sentroid' eller midtpunkt. Avstander måles dermed ved avstanden til dette sentroidet. Ikke-Euklidiske avstander refererer til 'clustroids', punktet nærmest andre punkter. Clustroids kan på sin side defineres på ulike måter.
> 
> 🎓 ['Begrenset'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Begrenset klynging](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduserer 'semi-supervisert' læring i denne usuperviserte metoden. Forholdene mellom punkter flagges som 'kan ikke kobles' eller 'må kobles', slik at noen regler tvinges på datasettet.
>
>Et eksempel: Hvis en algoritme slippes løs på en batch med umerket eller semi-merket data, kan klyngene den produserer være av dårlig kvalitet. I eksemplet ovenfor kan klyngene gruppere 'runde musikalske ting' og 'firkantede musikalske ting' og 'trekantede ting' og 'kjeks'. Hvis algoritmen får noen begrensninger, eller regler å følge ("elementet må være laget av plast", "elementet må kunne produsere musikk"), kan dette hjelpe med å 'begrense' algoritmen til å ta bedre valg.
> 
> 🎓 'Tetthet'
> 
> Data som er 'støyete' anses å være 'tett'. Avstandene mellom punkter i hver av klyngene kan vise seg, ved undersøkelse, å være mer eller mindre tette, eller 'trange', og dermed må denne dataen analyseres med den passende klyngemetoden. [Denne artikkelen](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrerer forskjellen mellom å bruke K-Means klynging vs. HDBSCAN-algoritmer for å utforske et støyete datasett med ujevn klyngetetthet.

## Klyngealgoritmer

Det finnes over 100 klyngealgoritmer, og deres bruk avhenger av naturen til dataen som skal analyseres. La oss diskutere noen av de viktigste:

- **Hierarkisk klynging**. Hvis et objekt klassifiseres basert på dets nærhet til et nærliggende objekt, snarere enn til et som er lenger unna, dannes klynger basert på avstanden mellom medlemmene. Scikit-learns agglomerative klynging er hierarkisk.

   ![Hierarkisk klynging Infografikk](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Sentroid klynging**. Denne populære algoritmen krever valg av 'k', eller antall klynger som skal dannes, hvoretter algoritmen bestemmer midtpunktet for en klynge og samler data rundt dette punktet. [K-means klynging](https://wikipedia.org/wiki/K-means_clustering) er en populær versjon av sentroid klynging. Midtpunktet bestemmes av nærmeste gjennomsnitt, derav navnet. Den kvadrerte avstanden fra klyngen minimeres.

   ![Sentroid klynging Infografikk](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribusjonsbasert klynging**. Basert på statistisk modellering, fokuserer distribusjonsbasert klynging på å bestemme sannsynligheten for at et datapunkt tilhører en klynge, og tilordner det deretter. Gaussian mixture-metoder tilhører denne typen.

- **Tetthetsbasert klynging**. Datapunkter tilordnes klynger basert på deres tetthet, eller deres gruppering rundt hverandre. Datapunkter langt fra gruppen anses som uteliggere eller støy. DBSCAN, Mean-shift og OPTICS tilhører denne typen klynging.

- **Grid-basert klynging**. For multidimensjonale datasett opprettes et rutenett, og dataen deles mellom rutenettets celler, og skaper dermed klynger.

## Øvelse – klyng dataen din

Klynging som teknikk støttes sterkt av god visualisering, så la oss komme i gang med å visualisere musikkdataen vår. Denne øvelsen vil hjelpe oss med å avgjøre hvilken av klyngemetodene vi mest effektivt bør bruke for naturen til denne dataen.

1. Åpne filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) i denne mappen.

1. Importer pakken `Seaborn` for god visualisering av data.

    ```python
    !pip install seaborn
    ```

1. Legg til musikkdataen fra [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Last opp en dataframe med noen data om sangene. Gjør deg klar til å utforske denne dataen ved å importere bibliotekene og skrive ut dataen:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Sjekk de første linjene med data:

    |     | navn                     | album                        | artist              | artist_top_genre | release_date | lengde | popularitet | dansbarhet | akustisitet | energi | instrumentalisme | livlighet | lydstyrke | talbarhet | tempo   | taktart         |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ---------- | ------------ | ------ | ---------------- | -------- | -------- | --------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternativ r&b   | 2019         | 144000 | 48         | 0.666      | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829    | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71       | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36      | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Få litt informasjon om dataframe, ved å kalle `info()`:

    ```python
    df.info()
    ```

   Utdata ser slik ut:

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

1. Dobbeltsjekk for nullverdier, ved å kalle `isnull()` og verifisere at summen er 0:

    ```python
    df.isnull().sum()
    ```

    Ser bra ut:

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

1. Beskriv dataen:

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

> 🤔 Hvis vi jobber med clustering, en usupervisert metode som ikke krever merket data, hvorfor viser vi denne dataen med etiketter? I utforskningsfasen av dataen er de nyttige, men de er ikke nødvendige for at clustering-algoritmer skal fungere. Du kan like gjerne fjerne kolonneoverskriftene og referere til dataen med kolonnenummer.

Se på de generelle verdiene i dataen. Merk at popularitet kan være '0', som viser sanger som ikke har noen rangering. La oss fjerne disse snart.

1. Bruk et stolpediagram for å finne de mest populære sjangrene:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![mest populære](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Hvis du vil se flere toppverdier, endre toppen `[:5]` til en større verdi, eller fjern den for å se alt.

Merk, når toppsjangeren er beskrevet som 'Missing', betyr det at Spotify ikke klassifiserte den, så la oss fjerne den.

1. Fjern manglende data ved å filtrere det ut

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Nå sjekk sjangrene på nytt:

    ![mest populære](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. De tre toppsjangrene dominerer datasetet. La oss konsentrere oss om `afro dancehall`, `afropop` og `nigerian pop`, og i tillegg filtrere datasetet for å fjerne alt med en popularitetsverdi på 0 (som betyr at det ikke ble klassifisert med en popularitet i datasetet og kan betraktes som støy for våre formål):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Gjør en rask test for å se om dataen korrelerer på noen spesielt sterk måte:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korrelasjoner](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Den eneste sterke korrelasjonen er mellom `energy` og `loudness`, noe som ikke er så overraskende, gitt at høy musikk vanligvis er ganske energisk. Ellers er korrelasjonene relativt svake. Det vil være interessant å se hva en clustering-algoritme kan gjøre med denne dataen.

    > 🎓 Merk at korrelasjon ikke innebærer årsakssammenheng! Vi har bevis på korrelasjon, men ingen bevis på årsakssammenheng. En [morsom nettside](https://tylervigen.com/spurious-correlations) har noen visuelle eksempler som understreker dette poenget.

Er det noen konvergens i dette datasetet rundt en sangs opplevde popularitet og dansbarhet? En FacetGrid viser at det er konsentriske sirkler som stemmer overens, uavhengig av sjanger. Kan det være at nigerianske smaker konvergerer på et visst nivå av dansbarhet for denne sjangeren?  

✅ Prøv forskjellige datapunkter (energy, loudness, speechiness) og flere eller forskjellige musikksjangre. Hva kan du oppdage? Ta en titt på `df.describe()`-tabellen for å se den generelle spredningen av datapunktene.

### Øvelse - datafordeling

Er disse tre sjangrene betydelig forskjellige i oppfatningen av deres dansbarhet, basert på deres popularitet?

1. Undersøk datafordelingen for våre tre toppsjangre for popularitet og dansbarhet langs en gitt x- og y-akse.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Du kan oppdage konsentriske sirkler rundt et generelt konvergenspunkt, som viser fordelingen av punkter.

    > 🎓 Merk at dette eksemplet bruker en KDE (Kernel Density Estimate)-graf som representerer dataen ved hjelp av en kontinuerlig sannsynlighetstetthetskurve. Dette lar oss tolke data når vi jobber med flere fordelinger.

    Generelt sett er de tre sjangrene løst tilpasset når det gjelder deres popularitet og dansbarhet. Å bestemme klynger i denne løst tilpassede dataen vil være en utfordring:

    ![fordeling](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Lag et spredningsdiagram:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Et spredningsdiagram av de samme aksene viser et lignende mønster av konvergens

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Generelt, for clustering, kan du bruke spredningsdiagrammer for å vise klynger av data, så det å mestre denne typen visualisering er veldig nyttig. I neste leksjon vil vi ta denne filtrerte dataen og bruke k-means clustering for å oppdage grupper i denne dataen som ser ut til å overlappe på interessante måter.

---

## 🚀Utfordring

Som forberedelse til neste leksjon, lag et diagram over de forskjellige clustering-algoritmene du kan oppdage og bruke i et produksjonsmiljø. Hvilke typer problemer prøver clustering å adressere?

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Før du bruker clustering-algoritmer, som vi har lært, er det en god idé å forstå naturen til datasetet ditt. Les mer om dette emnet [her](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Denne nyttige artikkelen](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) gir deg en oversikt over hvordan forskjellige clustering-algoritmer oppfører seg, gitt forskjellige dataformer.

## Oppgave

[Undersøk andre visualiseringer for clustering](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.