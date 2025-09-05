<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T21:26:29+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "sv"
}
-->
# Introduktion till klustring

Klustring är en typ av [Oövervakad inlärning](https://wikipedia.org/wiki/Unsupervised_learning) som förutsätter att en dataset är oetiketterad eller att dess indata inte är kopplade till fördefinierade utdata. Den använder olika algoritmer för att sortera igenom oetiketterad data och skapa grupper baserat på mönster som den identifierar i datan.

[![No One Like You av PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You av PSquare")

> 🎥 Klicka på bilden ovan för en video. Medan du studerar maskininlärning med klustring, njut av några nigerianska Dance Hall-låtar - detta är en högt rankad låt från 2014 av PSquare.

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

### Introduktion

[Klustring](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) är mycket användbart för datautforskning. Låt oss se om det kan hjälpa till att upptäcka trender och mönster i hur nigerianska publikgrupper konsumerar musik.

✅ Ta en minut och fundera på användningsområden för klustring. I verkliga livet sker klustring när du har en hög med tvätt och behöver sortera ut familjemedlemmarnas kläder 🧦👕👖🩲. Inom datavetenskap sker klustring när man försöker analysera en användares preferenser eller bestämma egenskaperna hos en oetiketterad dataset. Klustring hjälper på sätt och vis att skapa ordning i kaos, som en strumplåda.

[![Introduktion till ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduktion till klustring")

> 🎥 Klicka på bilden ovan för en video: MIT:s John Guttag introducerar klustring

I en professionell miljö kan klustring användas för att bestämma saker som marknadssegmentering, till exempel vilka åldersgrupper som köper vilka produkter. Ett annat användningsområde kan vara att upptäcka avvikelser, kanske för att identifiera bedrägerier i en dataset med kreditkortstransaktioner. Eller så kan du använda klustring för att identifiera tumörer i en samling medicinska skanningar.

✅ Fundera en minut på hur du kan ha stött på klustring 'i det vilda', inom bank, e-handel eller affärssammanhang.

> 🎓 Intressant nog har klusteranalys sitt ursprung inom antropologi och psykologi på 1930-talet. Kan du föreställa dig hur det kan ha använts?

Alternativt kan du använda det för att gruppera sökresultat - till exempel shoppinglänkar, bilder eller recensioner. Klustring är användbart när du har en stor dataset som du vill reducera och analysera mer detaljerat, så tekniken kan användas för att lära sig om data innan andra modeller konstrueras.

✅ När din data är organiserad i kluster tilldelar du den ett kluster-ID, och denna teknik kan vara användbar för att bevara en datasets integritet; du kan istället referera till en datapunkt med dess kluster-ID, snarare än med mer avslöjande identifierbar data. Kan du tänka dig andra anledningar till varför du skulle referera till ett kluster-ID istället för andra element i klustret för att identifiera det?

Fördjupa din förståelse av klustringstekniker i denna [Learn-modul](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Kom igång med klustring

[Scikit-learn erbjuder ett stort utbud](https://scikit-learn.org/stable/modules/clustering.html) av metoder för att utföra klustring. Vilken typ du väljer beror på ditt användningsområde. Enligt dokumentationen har varje metod olika fördelar. Här är en förenklad tabell över metoderna som stöds av Scikit-learn och deras lämpliga användningsområden:

| Metodnamn                     | Användningsområde                                                    |
| :---------------------------- | :------------------------------------------------------------------- |
| K-Means                       | allmänt syfte, induktiv                                              |
| Affinity propagation          | många, ojämna kluster, induktiv                                      |
| Mean-shift                    | många, ojämna kluster, induktiv                                      |
| Spectral clustering           | få, jämna kluster, transduktiv                                       |
| Ward hierarchical clustering  | många, begränsade kluster, transduktiv                               |
| Agglomerative clustering      | många, begränsade, icke-Euklidiska avstånd, transduktiv              |
| DBSCAN                        | icke-platt geometri, ojämna kluster, transduktiv                     |
| OPTICS                        | icke-platt geometri, ojämna kluster med variabel densitet, transduktiv |
| Gaussian mixtures             | platt geometri, induktiv                                             |
| BIRCH                         | stor dataset med avvikelser, induktiv                                |

> 🎓 Hur vi skapar kluster har mycket att göra med hur vi samlar datapunkter i grupper. Låt oss packa upp lite terminologi:
>
> 🎓 ['Transduktiv' vs. 'induktiv'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiv inferens härleds från observerade träningsfall som kartläggs till specifika testfall. Induktiv inferens härleds från träningsfall som kartläggs till generella regler som sedan tillämpas på testfall.
> 
> Ett exempel: Föreställ dig att du har en dataset som bara delvis är etiketterad. Vissa saker är 'skivor', vissa 'cd-skivor', och vissa är tomma. Din uppgift är att tilldela etiketter till de tomma. Om du väljer en induktiv metod skulle du träna en modell som letar efter 'skivor' och 'cd-skivor' och tillämpa dessa etiketter på din oetiketterade data. Denna metod kommer ha svårt att klassificera saker som faktiskt är 'kassetter'. En transduktiv metod, å andra sidan, hanterar denna okända data mer effektivt eftersom den arbetar för att gruppera liknande objekt och sedan tilldelar en etikett till en grupp. I detta fall kan kluster reflektera 'runda musikaliska saker' och 'fyrkantiga musikaliska saker'.
> 
> 🎓 ['Icke-platt' vs. 'platt' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Härstammar från matematisk terminologi, icke-platt vs. platt geometri hänvisar till mätningen av avstånd mellan punkter antingen med 'platt' ([Euklidisk](https://wikipedia.org/wiki/Euclidean_geometry)) eller 'icke-platt' (icke-Euklidisk) geometriska metoder.
>
>'Platt' i detta sammanhang hänvisar till Euklidisk geometri (delar av vilken lärs ut som 'plan' geometri), och icke-platt hänvisar till icke-Euklidisk geometri. Vad har geometri med maskininlärning att göra? Som två fält som är rotade i matematik måste det finnas ett gemensamt sätt att mäta avstånd mellan punkter i kluster, och det kan göras på ett 'platt' eller 'icke-platt' sätt, beroende på datans natur. [Euklidiska avstånd](https://wikipedia.org/wiki/Euclidean_distance) mäts som längden på en linjesegment mellan två punkter. [Icke-Euklidiska avstånd](https://wikipedia.org/wiki/Non-Euclidean_geometry) mäts längs en kurva. Om din data, visualiserad, verkar inte existera på en plan, kan du behöva använda en specialiserad algoritm för att hantera den.
>
![Platt vs Icke-platt Geometri Infografik](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Avstånd'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Kluster definieras av deras avståndsmatris, t.ex. avstånden mellan punkter. Detta avstånd kan mätas på några sätt. Euklidiska kluster definieras av genomsnittet av punktvärdena och innehåller en 'centroid' eller mittpunkt. Avstånd mäts således genom avståndet till den centroiden. Icke-Euklidiska avstånd hänvisar till 'clustroids', punkten närmast andra punkter. Clustroids kan i sin tur definieras på olika sätt.
> 
> 🎓 ['Begränsad'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Begränsad klustring](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introducerar 'semi-övervakad' inlärning i denna oövervakade metod. Relationerna mellan punkter flaggas som 'kan inte länka' eller 'måste länka' så vissa regler tvingas på datasetet.
>
>Ett exempel: Om en algoritm släpps fri på en samling oetiketterad eller semi-etiketterad data kan klustren den producerar vara av dålig kvalitet. I exemplet ovan kan klustren gruppera 'runda musikaliska saker' och 'fyrkantiga musikaliska saker' och 'triangulära saker' och 'kakor'. Om algoritmen ges vissa begränsningar, eller regler att följa ("objektet måste vara gjort av plast", "objektet måste kunna producera musik") kan detta hjälpa till att 'begränsa' algoritmen att göra bättre val.
> 
> 🎓 'Densitet'
> 
> Data som är 'brusig' anses vara 'tät'. Avstånden mellan punkter i varje av dess kluster kan vid undersökning visa sig vara mer eller mindre täta, eller 'trånga', och denna data behöver analyseras med lämplig klustringsmetod. [Denna artikel](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrerar skillnaden mellan att använda K-Means klustring vs. HDBSCAN-algoritmer för att utforska en brusig dataset med ojämn klusterdensitet.

## Klustringsalgoritmer

Det finns över 100 klustringsalgoritmer, och deras användning beror på datans natur. Låt oss diskutera några av de viktigaste:

- **Hierarkisk klustring**. Om ett objekt klassificeras baserat på dess närhet till ett närliggande objekt, snarare än till ett längre bort, bildas kluster baserat på medlemmarnas avstånd till och från andra objekt. Scikit-learns agglomerativa klustring är hierarkisk.

   ![Hierarkisk klustring Infografik](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid-klustring**. Denna populära algoritm kräver valet av 'k', eller antalet kluster som ska bildas, varefter algoritmen bestämmer mittpunkten för ett kluster och samlar data runt den punkten. [K-means klustring](https://wikipedia.org/wiki/K-means_clustering) är en populär version av centroid-klustring. Centroiden bestäms av det närmaste medelvärdet, därav namnet. Det kvadrerade avståndet från klustret minimeras.

   ![Centroid-klustring Infografik](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Fördelningsbaserad klustring**. Baserad på statistisk modellering fokuserar fördelningsbaserad klustring på att bestämma sannolikheten att en datapunkt tillhör ett kluster och tilldelar den därefter. Gaussian mixture-metoder tillhör denna typ.

- **Densitetsbaserad klustring**. Datapunkter tilldelas kluster baserat på deras densitet, eller deras gruppering runt varandra. Datapunkter långt från gruppen anses vara avvikelser eller brus. DBSCAN, Mean-shift och OPTICS tillhör denna typ av klustring.

- **Rutbaserad klustring**. För multidimensionella datasets skapas ett rutnät och datan delas upp mellan rutnätets celler, vilket skapar kluster.

## Övning - klustra din data

Klustring som teknik underlättas mycket av korrekt visualisering, så låt oss börja med att visualisera vår musikdata. Denna övning kommer att hjälpa oss att avgöra vilken av klustringsmetoderna vi mest effektivt bör använda för datans natur.

1. Öppna filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) i denna mapp.

1. Importera paketet `Seaborn` för bra datavisualisering.

    ```python
    !pip install seaborn
    ```

1. Lägg till musikdatan från [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Ladda upp en dataframe med lite data om låtarna. Förbered dig på att utforska denna data genom att importera biblioteken och dumpa ut datan:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Kontrollera de första raderna av data:

    |     | namn                     | album                        | artist              | artist_top_genre | release_date | längd | popularitet | dansbarhet | akustiskhet | energi | instrumentellhet | livlighet | ljudstyrka | talighet | tempo   | taktart         |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ---------- | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternativ r&b   | 2019         | 144000 | 48         | 0.666      | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71       | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Få information om dataframe genom att kalla på `info()`:

    ```python
    df.info()
    ```

   Utdata ser ut så här:

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

1. Kontrollera om det finns några null-värden genom att kalla på `isnull()` och verifiera att summan är 0:

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

1. Beskriv datan:

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

> 🤔 Om vi arbetar med klustring, en osuperviserad metod som inte kräver etiketterad data, varför visar vi denna data med etiketter? Under datautforskningsfasen är de användbara, men de är inte nödvändiga för att klustringsalgoritmer ska fungera. Du kan lika gärna ta bort kolumnrubrikerna och hänvisa till datan med kolumnnummer.

Titta på de generella värdena i datan. Notera att popularitet kan vara '0', vilket visar låtar som inte har någon ranking. Låt oss ta bort dessa snart.

1. Använd ett stapeldiagram för att ta reda på de mest populära genrerna:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![mest populära](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Om du vill se fler toppvärden, ändra topp `[:5]` till ett större värde, eller ta bort det för att se allt.

Notera, när toppgenren beskrivs som 'Missing', betyder det att Spotify inte klassificerade den, så låt oss ta bort den.

1. Ta bort saknade data genom att filtrera bort dem

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Kontrollera nu genrerna igen:

    ![mest populära](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. De tre toppgenrerna dominerar datasetet. Låt oss koncentrera oss på `afro dancehall`, `afropop` och `nigerian pop`, och dessutom filtrera datasetet för att ta bort allt med ett popularitetsvärde på 0 (vilket betyder att det inte klassificerades med en popularitet i datasetet och kan betraktas som brus för våra syften):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Gör ett snabbt test för att se om datan korrelerar på något särskilt starkt sätt:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korrelationer](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Den enda starka korrelationen är mellan `energy` och `loudness`, vilket inte är särskilt förvånande, eftersom hög musik vanligtvis är ganska energisk. Annars är korrelationerna relativt svaga. Det kommer att bli intressant att se vad en klustringsalgoritm kan göra med denna data.

    > 🎓 Notera att korrelation inte innebär kausalitet! Vi har bevis på korrelation men inget bevis på kausalitet. En [underhållande webbplats](https://tylervigen.com/spurious-correlations) har några visuella exempel som betonar denna punkt.

Finns det någon konvergens i detta dataset kring en låts upplevda popularitet och dansbarhet? En FacetGrid visar att det finns koncentriska cirklar som stämmer överens, oavsett genre. Kan det vara så att nigerianska smaker konvergerar vid en viss nivå av dansbarhet för denna genre?  

✅ Prova olika datapunkter (energy, loudness, speechiness) och fler eller olika musikgenrer. Vad kan du upptäcka? Titta på `df.describe()`-tabellen för att se den generella spridningen av datapunkterna.

### Övning - dataspridning

Är dessa tre genrer signifikant olika i uppfattningen av deras dansbarhet, baserat på deras popularitet?

1. Undersök dataspridningen för våra tre toppgenrer för popularitet och dansbarhet längs en given x- och y-axel.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Du kan upptäcka koncentriska cirklar runt en generell konvergenspunkt, som visar spridningen av punkter.

    > 🎓 Notera att detta exempel använder ett KDE (Kernel Density Estimate)-diagram som representerar datan med en kontinuerlig sannolikhetstäthetskurva. Detta gör det möjligt att tolka data när man arbetar med flera distributioner.

    Generellt sett är de tre genrerna löst anpassade när det gäller deras popularitet och dansbarhet. Att bestämma kluster i denna löst anpassade data kommer att vara en utmaning:

    ![spridning](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Skapa ett spridningsdiagram:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Ett spridningsdiagram av samma axlar visar ett liknande mönster av konvergens

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Generellt sett kan du använda spridningsdiagram för att visa kluster av data, så att bemästra denna typ av visualisering är mycket användbart. I nästa lektion kommer vi att ta denna filtrerade data och använda k-means-klustring för att upptäcka grupper i denna data som verkar överlappa på intressanta sätt.

---

## 🚀Utmaning

Som förberedelse inför nästa lektion, skapa ett diagram över de olika klustringsalgoritmer du kan upptäcka och använda i en produktionsmiljö. Vilka typer av problem försöker klustringen lösa?

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Innan du tillämpar klustringsalgoritmer, som vi har lärt oss, är det en bra idé att förstå naturen av ditt dataset. Läs mer om detta ämne [här](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Denna hjälpsamma artikel](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) går igenom de olika sätt som olika klustringsalgoritmer beter sig, givet olika datamönster.

## Uppgift

[Undersök andra visualiseringar för klustring](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.