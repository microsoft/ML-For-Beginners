<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T19:10:56+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "nl"
}
-->
# Introductie tot clustering

Clustering is een type [Ongecontroleerd Leren](https://wikipedia.org/wiki/Unsupervised_learning) dat ervan uitgaat dat een dataset niet gelabeld is of dat de invoer niet gekoppeld is aan vooraf gedefinieerde uitkomsten. Het gebruikt verschillende algoritmen om door niet-gelabelde gegevens te sorteren en groepen te vormen op basis van patronen die het in de gegevens herkent.

[![No One Like You van PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You van PSquare")

> 🎥 Klik op de afbeelding hierboven voor een video. Terwijl je machine learning met clustering bestudeert, geniet van wat Nigeriaanse Dance Hall-nummers - dit is een hoog gewaardeerd nummer uit 2014 van PSquare.

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

### Introductie

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) is erg nuttig voor data-exploratie. Laten we kijken of het kan helpen trends en patronen te ontdekken in de manier waarop Nigeriaanse luisteraars muziek consumeren.

✅ Neem een minuut de tijd om na te denken over de toepassingen van clustering. In het dagelijks leven gebeurt clustering bijvoorbeeld wanneer je een stapel wasgoed hebt en de kleding van je gezinsleden moet sorteren 🧦👕👖🩲. In data science gebeurt clustering wanneer je probeert de voorkeuren van een gebruiker te analyseren of de kenmerken van een niet-gelabelde dataset te bepalen. Clustering helpt op een bepaalde manier om chaos te begrijpen, zoals een sokkenla.

[![Introductie tot ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introductie tot Clustering")

> 🎥 Klik op de afbeelding hierboven voor een video: MIT's John Guttag introduceert clustering.

In een professionele omgeving kan clustering worden gebruikt om zaken zoals marktsegmentatie te bepalen, bijvoorbeeld om te achterhalen welke leeftijdsgroepen welke producten kopen. Een andere toepassing zou anomaliedetectie kunnen zijn, bijvoorbeeld om fraude op te sporen in een dataset van creditcardtransacties. Of je kunt clustering gebruiken om tumoren te identificeren in een reeks medische scans.

✅ Denk een minuut na over hoe je clustering 'in het wild' bent tegengekomen, bijvoorbeeld in een bank-, e-commerce- of zakelijke omgeving.

> 🎓 Interessant genoeg is clusteranalyse ontstaan in de vakgebieden antropologie en psychologie in de jaren 1930. Kun je je voorstellen hoe het toen werd gebruikt?

Een andere toepassing zou kunnen zijn het groeperen van zoekresultaten - bijvoorbeeld door winkellinks, afbeeldingen of recensies. Clustering is nuttig wanneer je een grote dataset hebt die je wilt verkleinen en waarop je meer gedetailleerde analyses wilt uitvoeren. De techniek kan worden gebruikt om meer te leren over gegevens voordat andere modellen worden gebouwd.

✅ Zodra je gegevens zijn georganiseerd in clusters, wijs je ze een cluster-ID toe. Deze techniek kan nuttig zijn om de privacy van een dataset te behouden; je kunt in plaats daarvan verwijzen naar een datapunt via zijn cluster-ID, in plaats van via meer onthullende identificeerbare gegevens. Kun je andere redenen bedenken waarom je een cluster-ID zou gebruiken in plaats van andere elementen van het cluster om het te identificeren?

Verdiep je kennis van clusteringtechnieken in deze [Learn-module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Aan de slag met clustering

[Scikit-learn biedt een breed scala](https://scikit-learn.org/stable/modules/clustering.html) aan methoden om clustering uit te voeren. Het type dat je kiest, hangt af van je gebruikssituatie. Volgens de documentatie heeft elke methode verschillende voordelen. Hier is een vereenvoudigde tabel van de methoden die door Scikit-learn worden ondersteund en hun geschikte gebruikssituaties:

| Methode                      | Gebruikssituatie                                                      |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | algemeen gebruik, inductief                                           |
| Affinity propagation         | veel, ongelijke clusters, inductief                                   |
| Mean-shift                   | veel, ongelijke clusters, inductief                                   |
| Spectral clustering          | weinig, gelijke clusters, transductief                                |
| Ward hiërarchische clustering| veel, beperkte clusters, transductief                                 |
| Agglomerative clustering     | veel, beperkt, niet-Euclidische afstanden, transductief               |
| DBSCAN                       | niet-vlakke geometrie, ongelijke clusters, transductief               |
| OPTICS                       | niet-vlakke geometrie, ongelijke clusters met variabele dichtheid, transductief |
| Gaussian mixtures            | vlakke geometrie, inductief                                           |
| BIRCH                        | grote dataset met uitschieters, inductief                             |

> 🎓 Hoe we clusters creëren heeft veel te maken met hoe we de datapunten in groepen verzamelen. Laten we wat terminologie uitpakken:
>
> 🎓 ['Transductief' vs. 'inductief'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transductieve inferentie is afgeleid van waargenomen trainingsgevallen die worden gekoppeld aan specifieke testgevallen. Inductieve inferentie is afgeleid van trainingsgevallen die worden gekoppeld aan algemene regels die pas daarna worden toegepast op testgevallen.
> 
> Een voorbeeld: Stel je hebt een dataset die slechts gedeeltelijk gelabeld is. Sommige dingen zijn 'platen', sommige 'cd's', en sommige zijn blanco. Jouw taak is om labels toe te voegen aan de blanco items. Als je een inductieve aanpak kiest, train je een model dat zoekt naar 'platen' en 'cd's', en pas je die labels toe op je niet-gelabelde gegevens. Deze aanpak zal moeite hebben om dingen te classificeren die eigenlijk 'cassettebandjes' zijn. Een transductieve aanpak daarentegen behandelt deze onbekende gegevens effectiever omdat het werkt om vergelijkbare items samen te groeperen en vervolgens een label toe te passen op een groep. In dit geval kunnen clusters 'ronde muzikale dingen' en 'vierkante muzikale dingen' weerspiegelen.
> 
> 🎓 ['Niet-vlakke' vs. 'vlakke' geometrie](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Afgeleid van wiskundige terminologie, verwijst niet-vlakke vs. vlakke geometrie naar de manier waarop afstanden tussen punten worden gemeten, ofwel 'vlak' ([Euclidisch](https://wikipedia.org/wiki/Euclidean_geometry)) of 'niet-vlak' (niet-Euclidisch).
>
>'Vlak' in deze context verwijst naar Euclidische geometrie (delen hiervan worden onderwezen als 'platte' geometrie), en niet-vlak verwijst naar niet-Euclidische geometrie. Wat heeft geometrie te maken met machine learning? Nou, als twee velden die geworteld zijn in wiskunde, moet er een gemeenschappelijke manier zijn om afstanden tussen punten in clusters te meten, en dat kan op een 'vlakke' of 'niet-vlakke' manier, afhankelijk van de aard van de gegevens. [Euclidische afstanden](https://wikipedia.org/wiki/Euclidean_distance) worden gemeten als de lengte van een lijnsegment tussen twee punten. [Niet-Euclidische afstanden](https://wikipedia.org/wiki/Non-Euclidean_geometry) worden gemeten langs een curve. Als je gegevens, gevisualiseerd, niet op een vlak lijken te bestaan, moet je mogelijk een gespecialiseerd algoritme gebruiken om ze te verwerken.
>
![Vlakke vs Niet-vlakke Geometrie Infographic](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Afstanden'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusters worden gedefinieerd door hun afstandsmatrix, bijvoorbeeld de afstanden tussen punten. Deze afstand kan op verschillende manieren worden gemeten. Euclidische clusters worden gedefinieerd door het gemiddelde van de puntwaarden en bevatten een 'centroid' of middelpunt. Afstanden worden dus gemeten door de afstand tot dat middelpunt. Niet-Euclidische afstanden verwijzen naar 'clustroids', het punt dat het dichtst bij andere punten ligt. Clustroids kunnen op verschillende manieren worden gedefinieerd.
> 
> 🎓 ['Beperkt'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Beperkte Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduceert 'semi-gecontroleerd' leren in deze ongecontroleerde methode. De relaties tussen punten worden gemarkeerd als 'kan niet koppelen' of 'moet koppelen', zodat enkele regels worden opgelegd aan de dataset.
>
>Een voorbeeld: Als een algoritme vrij wordt gelaten op een batch niet-gelabelde of semi-gelabelde gegevens, kunnen de clusters die het produceert van slechte kwaliteit zijn. In het bovenstaande voorbeeld kunnen de clusters 'ronde muzikale dingen', 'vierkante muzikale dingen', 'driehoekige dingen' en 'koekjes' groeperen. Als er enkele beperkingen of regels worden opgelegd ("het item moet van plastic zijn", "het item moet muziek kunnen produceren"), kan dit helpen om het algoritme te 'beperken' om betere keuzes te maken.
> 
> 🎓 'Dichtheid'
> 
> Gegevens die 'ruis' bevatten, worden beschouwd als 'dicht'. De afstanden tussen punten in elk van zijn clusters kunnen bij nader onderzoek meer of minder dicht, of 'geconcentreerd' blijken te zijn, en daarom moeten deze gegevens worden geanalyseerd met de juiste clusteringmethode. [Dit artikel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) laat het verschil zien tussen het gebruik van K-Means clustering versus HDBSCAN-algoritmen om een dataset met ongelijke clusterdichtheid te verkennen.

## Clustering-algoritmen

Er zijn meer dan 100 clustering-algoritmen, en hun gebruik hangt af van de aard van de gegevens. Laten we enkele van de belangrijkste bespreken:

- **Hiërarchische clustering**. Als een object wordt geclassificeerd op basis van zijn nabijheid tot een nabijgelegen object, in plaats van tot een verder verwijderd object, worden clusters gevormd op basis van de afstand van hun leden tot en van andere objecten. Scikit-learn's agglomeratieve clustering is hiërarchisch.

   ![Hiërarchische clustering Infographic](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. Dit populaire algoritme vereist de keuze van 'k', of het aantal clusters dat moet worden gevormd, waarna het algoritme het middelpunt van een cluster bepaalt en gegevens rond dat punt verzamelt. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) is een populaire versie van centroid clustering. Het middelpunt wordt bepaald door het dichtstbijzijnde gemiddelde, vandaar de naam. De kwadratische afstand tot het cluster wordt geminimaliseerd.

   ![Centroid clustering Infographic](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distributie-gebaseerde clustering**. Gebaseerd op statistische modellering, richt distributie-gebaseerde clustering zich op het bepalen van de waarschijnlijkheid dat een datapunt tot een cluster behoort, en wijst het dienovereenkomstig toe. Gaussian mixture-methoden behoren tot dit type.

- **Dichtheid-gebaseerde clustering**. Datapunten worden toegewezen aan clusters op basis van hun dichtheid, of hun groepering rond elkaar. Datapunten ver van de groep worden beschouwd als uitschieters of ruis. DBSCAN, Mean-shift en OPTICS behoren tot dit type clustering.

- **Raster-gebaseerde clustering**. Voor multidimensionale datasets wordt een raster gemaakt en worden de gegevens verdeeld over de cellen van het raster, waardoor clusters ontstaan.

## Oefening - cluster je gegevens

Clustering als techniek wordt sterk ondersteund door goede visualisatie, dus laten we beginnen met het visualiseren van onze muziekgegevens. Deze oefening helpt ons te beslissen welke van de clusteringmethoden we het meest effectief kunnen gebruiken voor de aard van deze gegevens.

1. Open het [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb)-bestand in deze map.

1. Importeer het `Seaborn`-pakket voor goede datavisualisatie.

    ```python
    !pip install seaborn
    ```

1. Voeg de muziekinformatie toe uit [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Laad een dataframe met enkele gegevens over de nummers. Bereid je voor om deze gegevens te verkennen door de bibliotheken te importeren en de gegevens uit te lezen:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Bekijk de eerste paar regels van de gegevens:

    |     | naam                     | album                        | artiest             | artiest_top_genre | release_date | lengte | populariteit | dansbaarheid | akoestiek   | energie | instrumentaliteit | levendigheid | luidheid | spraakzaamheid | tempo   | maatsoort      |
    | --- | ------------------------ | ---------------------------- | ------------------- | ----------------- | ------------ | ------ | ------------ | ------------ | ----------- | ------ | ----------------- | ------------ | -------- | -------------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternatieve r&b  | 2019         | 144000 | 48           | 0.666        | 0.851       | 0.42   | 0.534             | 0.11         | -6.699   | 0.0829         | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop           | 2020         | 89488  | 30           | 0.71         | 0.0822      | 0.683  | 0.000169          | 0.101        | -5.64    | 0.36           | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Verkrijg wat informatie over de dataframe door `info()` aan te roepen:

    ```python
    df.info()
    ```

   De output ziet er ongeveer zo uit:

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

1. Controleer dubbel op null-waarden door `isnull()` aan te roepen en te verifiëren dat de som 0 is:

    ```python
    df.isnull().sum()
    ```

    Ziet er goed uit:

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

1. Beschrijf de data:

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

> 🤔 Als we werken met clustering, een ongesuperviseerde methode die geen gelabelde data vereist, waarom tonen we deze data dan met labels? In de data-exploratiefase zijn ze handig, maar ze zijn niet noodzakelijk voor de clustering-algoritmes om te werken. Je zou net zo goed de kolomkoppen kunnen verwijderen en naar de data kunnen verwijzen via kolomnummer.

Bekijk de algemene waarden van de data. Merk op dat populariteit '0' kan zijn, wat aangeeft dat nummers geen ranking hebben. Laten we die binnenkort verwijderen.

1. Gebruik een staafdiagram om de meest populaire genres te vinden:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Als je meer topwaarden wilt zien, verander dan de top `[:5]` naar een grotere waarde, of verwijder het om alles te zien.

Let op, wanneer het topgenre wordt beschreven als 'Missing', betekent dit dat Spotify het niet heeft geclassificeerd, dus laten we het verwijderen.

1. Verwijder ontbrekende data door deze eruit te filteren:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Controleer nu de genres opnieuw:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. De top drie genres domineren deze dataset verreweg. Laten we ons concentreren op `afro dancehall`, `afropop` en `nigerian pop`, en daarnaast de dataset filteren om alles met een populariteitswaarde van 0 te verwijderen (wat betekent dat het niet geclassificeerd is met een populariteit in de dataset en kan worden beschouwd als ruis voor onze doeleinden):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Doe een snelle test om te zien of de data op een bijzonder sterke manier correleert:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    De enige sterke correlatie is tussen `energy` en `loudness`, wat niet al te verrassend is, aangezien luide muziek meestal behoorlijk energiek is. Verder zijn de correlaties relatief zwak. Het zal interessant zijn om te zien wat een clustering-algoritme van deze data kan maken.

    > 🎓 Merk op dat correlatie geen oorzaak-gevolg impliceert! We hebben bewijs van correlatie, maar geen bewijs van oorzaak-gevolg. Een [grappige website](https://tylervigen.com/spurious-correlations) heeft enkele visuals die dit punt benadrukken.

Is er enige convergentie in deze dataset rond de waargenomen populariteit en dansbaarheid van een nummer? Een FacetGrid laat zien dat er concentrische cirkels zijn die op één lijn liggen, ongeacht het genre. Zou het kunnen dat Nigeriaanse smaken convergeren op een bepaald niveau van dansbaarheid voor dit genre?

✅ Probeer verschillende datapunten (energie, luidheid, spraakzaamheid) en meer of andere muziekgenres. Wat kun je ontdekken? Bekijk de `df.describe()`-tabel om de algemene spreiding van de datapunten te zien.

### Oefening - data distributie

Zijn deze drie genres significant verschillend in de perceptie van hun dansbaarheid, gebaseerd op hun populariteit?

1. Onderzoek de data distributie van onze top drie genres voor populariteit en dansbaarheid langs een gegeven x- en y-as.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Je kunt concentrische cirkels ontdekken rond een algemeen convergentiepunt, wat de verdeling van punten laat zien.

    > 🎓 Merk op dat dit voorbeeld een KDE (Kernel Density Estimate) grafiek gebruikt die de data vertegenwoordigt met een continue waarschijnlijkheidsdichtheidscurve. Dit stelt ons in staat om data te interpreteren bij het werken met meerdere distributies.

    Over het algemeen komen de drie genres losjes overeen in termen van hun populariteit en dansbaarheid. Het bepalen van clusters in deze losjes uitgelijnde data zal een uitdaging zijn:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Maak een scatterplot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Een scatterplot van dezelfde assen toont een vergelijkbaar patroon van convergentie.

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Over het algemeen kun je voor clustering scatterplots gebruiken om clusters van data te tonen, dus het beheersen van dit type visualisatie is erg nuttig. In de volgende les zullen we deze gefilterde data gebruiken en k-means clustering toepassen om groepen in deze data te ontdekken die op interessante manieren lijken te overlappen.

---

## 🚀Uitdaging

Ter voorbereiding op de volgende les, maak een diagram over de verschillende clustering-algoritmes die je zou kunnen ontdekken en gebruiken in een productieomgeving. Welke soorten problemen probeert de clustering op te lossen?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Voordat je clustering-algoritmes toepast, zoals we hebben geleerd, is het een goed idee om de aard van je dataset te begrijpen. Lees meer over dit onderwerp [hier](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Dit nuttige artikel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) leidt je door de verschillende manieren waarop clustering-algoritmes zich gedragen, gegeven verschillende datavormen.

## Opdracht

[Onderzoek andere visualisaties voor clustering](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.