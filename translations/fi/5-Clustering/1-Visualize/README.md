<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T00:02:39+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "fi"
}
-->
# Johdanto klusterointiin

Klusterointi on eräänlainen [valvomaton oppiminen](https://wikipedia.org/wiki/Unsupervised_learning), joka olettaa, että datasetti on merkitsemätön tai että sen syötteet eivät ole yhdistetty ennalta määriteltyihin tuloksiin. Se käyttää erilaisia algoritmeja käydäkseen läpi merkitsemätöntä dataa ja luodakseen ryhmiä datasta havaittujen kuvioiden perusteella.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi videon. Opiskellessasi koneoppimista klusteroinnin avulla, nauti samalla nigerialaisista Dance Hall -kappaleista – tämä on PSquaren erittäin arvostettu kappale vuodelta 2014.

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

### Johdanto

[Klusterointi](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) on erittäin hyödyllinen datan tutkimiseen. Katsotaan, voiko se auttaa löytämään trendejä ja kuvioita nigerialaisten yleisöjen musiikinkulutustavoista.

✅ Mieti hetki klusteroinnin käyttötarkoituksia. Arkielämässä klusterointi tapahtuu aina, kun sinulla on kasa pyykkiä ja sinun täytyy lajitella perheenjäsenten vaatteet 🧦👕👖🩲. Data-analytiikassa klusterointi tapahtuu, kun yritetään analysoida käyttäjän mieltymyksiä tai määrittää minkä tahansa merkitsemättömän datasetin ominaisuuksia. Klusterointi auttaa tavallaan tuomaan järjestystä kaaokseen, kuten sukkalaatikkoon.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi videon: MIT:n John Guttag esittelee klusterointia.

Ammatillisessa ympäristössä klusterointia voidaan käyttää esimerkiksi markkinasegmentoinnin määrittämiseen, kuten sen selvittämiseen, mitkä ikäryhmät ostavat mitäkin tuotteita. Toinen käyttötarkoitus voisi olla poikkeavuuksien havaitseminen, esimerkiksi luottokorttitapahtumien datasetistä petosten tunnistamiseen. Tai klusterointia voisi käyttää kasvainten tunnistamiseen lääketieteellisten skannauksien joukosta.

✅ Mieti hetki, miten olet saattanut kohdata klusterointia "luonnossa", esimerkiksi pankkitoiminnassa, verkkokaupassa tai liiketoiminnassa.

> 🎓 Mielenkiintoista on, että klusterianalyysi sai alkunsa antropologian ja psykologian aloilla 1930-luvulla. Voitko kuvitella, miten sitä saatettiin käyttää?

Vaihtoehtoisesti sitä voisi käyttää hakutulosten ryhmittelyyn – esimerkiksi ostoslinkkien, kuvien tai arvostelujen mukaan. Klusterointi on hyödyllistä, kun sinulla on suuri datasetti, jonka haluat pienentää ja josta haluat tehdä tarkempaa analyysiä, joten tekniikkaa voidaan käyttää datan tutkimiseen ennen muiden mallien rakentamista.

✅ Kun datasi on järjestetty klustereihin, sille annetaan klusteri-ID, ja tämä tekniikka voi olla hyödyllinen datasetin yksityisyyden säilyttämisessä; voit viitata datapisteeseen sen klusteri-ID:n avulla sen sijaan, että käyttäisit paljastavampia tunnistettavia tietoja. Voitko keksiä muita syitä, miksi käyttäisit klusteri-ID:tä klusterin muiden elementtien sijaan sen tunnistamiseen?

Syvennä ymmärrystäsi klusterointitekniikoista tässä [Learn-moduulissa](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Klusteroinnin aloittaminen

[Scikit-learn tarjoaa laajan valikoiman](https://scikit-learn.org/stable/modules/clustering.html) menetelmiä klusteroinnin suorittamiseen. Valitsemasi tyyppi riippuu käyttötapauksestasi. Dokumentaation mukaan jokaisella menetelmällä on erilaisia etuja. Tässä on yksinkertaistettu taulukko Scikit-learnin tukemista menetelmistä ja niiden sopivista käyttötapauksista:

| Menetelmän nimi               | Käyttötapaus                                                          |
| :---------------------------- | :-------------------------------------------------------------------- |
| K-Means                       | yleiskäyttö, induktiivinen                                            |
| Affinity propagation          | monet, epätasaiset klusterit, induktiivinen                          |
| Mean-shift                    | monet, epätasaiset klusterit, induktiivinen                          |
| Spectral clustering           | harvat, tasaiset klusterit, transduktiivinen                         |
| Ward hierarchical clustering  | monet, rajoitetut klusterit, transduktiivinen                        |
| Agglomerative clustering      | monet, rajoitetut, ei-euklidiset etäisyydet, transduktiivinen         |
| DBSCAN                        | ei-tasainen geometria, epätasaiset klusterit, transduktiivinen       |
| OPTICS                        | ei-tasainen geometria, epätasaiset klusterit, vaihteleva tiheys, transduktiivinen |
| Gaussian mixtures             | tasainen geometria, induktiivinen                                    |
| BIRCH                         | suuri datasetti, jossa poikkeavuuksia, induktiivinen                 |

> 🎓 Klusterien luominen liittyy vahvasti siihen, miten datan pisteet ryhmitellään ryhmiin. Puretaanpa hieman sanastoa:
>
> 🎓 ['Transduktiivinen' vs. 'induktiivinen'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transduktiivinen päättely perustuu havaittuihin harjoitustapauksiin, jotka liittyvät tiettyihin testitapauksiin. Induktiivinen päättely perustuu harjoitustapauksiin, jotka johtavat yleisiin sääntöihin, joita sovelletaan vasta testitapauksiin.
> 
> Esimerkki: Kuvittele, että sinulla on datasetti, joka on vain osittain merkitty. Jotkut asiat ovat 'levyjä', jotkut 'CD-levyjä', ja jotkut ovat tyhjiä. Tehtäväsi on antaa tyhjille merkinnät. Jos valitset induktiivisen lähestymistavan, kouluttaisit mallin etsimään 'levyjä' ja 'CD-levyjä' ja soveltaisit näitä merkintöjä merkitsemättömään dataan. Tämä lähestymistapa kohtaa vaikeuksia luokitellessaan asioita, jotka ovat itse asiassa 'kasetteja'. Transduktiivinen lähestymistapa sen sijaan käsittelee tätä tuntematonta dataa tehokkaammin, koska se pyrkii ryhmittelemään samanlaiset kohteet yhteen ja soveltamaan ryhmään merkintää. Tässä tapauksessa klusterit saattavat heijastaa 'pyöreitä musiikkiesineitä' ja 'neliömäisiä musiikkiesineitä'.
> 
> 🎓 ['Ei-tasainen' vs. 'tasainen' geometria](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Matematiikan terminologiasta johdettuna ei-tasainen vs. tasainen geometria viittaa pisteiden välisten etäisyyksien mittaamiseen joko 'tasaisilla' ([euklidisilla](https://wikipedia.org/wiki/Euclidean_geometry)) tai 'ei-tasaisilla' (ei-euklidisilla) geometrisilla menetelmillä.
>
>'Tasainen' tässä yhteydessä viittaa euklidiseen geometriaan (josta osia opetetaan 'tasogeometriana'), ja ei-tasainen viittaa ei-euklidiseen geometriaan. Mitä geometrialla on tekemistä koneoppimisen kanssa? No, koska molemmat alat perustuvat matematiikkaan, pisteiden välisten etäisyyksien mittaamiseen klustereissa täytyy olla yhteinen tapa, ja se voidaan tehdä 'tasaisella' tai 'ei-tasaisella' tavalla datan luonteen mukaan. [Euklidiset etäisyydet](https://wikipedia.org/wiki/Euclidean_distance) mitataan viivan pituutena kahden pisteen välillä. [Ei-euklidiset etäisyydet](https://wikipedia.org/wiki/Non-Euclidean_geometry) mitataan käyrän pitkin. Jos datasi, visualisoituna, ei näytä olevan tasossa, saatat tarvita erikoistuneen algoritmin sen käsittelemiseen.
>
![Tasainen vs Ei-tasainen geometria Infografiikka](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Etäisyydet'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klusterit määritellään niiden etäisyysmatriisin perusteella, eli pisteiden välisillä etäisyyksillä. Tämä etäisyys voidaan mitata muutamalla tavalla. Euklidiset klusterit määritellään pistearvojen keskiarvon perusteella, ja niillä on 'centroidi' eli keskipiste. Etäisyydet mitataan siis etäisyytenä centroidiin. Ei-euklidiset etäisyydet viittaavat 'clustroideihin', pisteeseen, joka on lähimpänä muita pisteitä. Clustroidit voidaan puolestaan määritellä eri tavoin.
> 
> 🎓 ['Rajoitettu'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Rajoitettu klusterointi](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) tuo 'puolivalvotun' oppimisen tähän valvomattomaan menetelmään. Pisteiden väliset suhteet merkitään 'ei voi yhdistää' tai 'täytyy yhdistää', joten datasettiin pakotetaan joitakin sääntöjä.
>
>Esimerkki: Jos algoritmi päästetään vapaaksi joukkoon merkitsemätöntä tai osittain merkittyä dataa, sen tuottamat klusterit voivat olla huonolaatuisia. Esimerkissä yllä klusterit saattavat ryhmitellä 'pyöreät musiikkiesineet', 'neliömäiset musiikkiesineet', 'kolmiomaiset esineet' ja 'keksit'. Jos algoritmille annetaan joitakin rajoituksia tai sääntöjä ("esineen täytyy olla muovista tehty", "esineen täytyy pystyä tuottamaan musiikkia"), tämä voi auttaa 'rajoittamaan' algoritmia tekemään parempia valintoja.
> 
> 🎓 'Tiheys'
> 
> Data, joka on 'meluisaa', katsotaan olevan 'tiheää'. Etäisyydet pisteiden välillä kussakin klusterissa voivat osoittautua tarkastelussa tiheämmiksi tai harvemmiksi, ja näin ollen tämä data täytyy analysoida sopivalla klusterointimenetelmällä. [Tämä artikkeli](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) havainnollistaa eroa K-Means-klusteroinnin ja HDBSCAN-algoritmien käytössä meluisan datasetin tutkimiseen, jossa klusterien tiheys on epätasainen.

## Klusterointialgoritmit

Klusterointialgoritmeja on yli 100, ja niiden käyttö riippuu käsillä olevan datan luonteesta. Keskustellaan joistakin tärkeimmistä:

- **Hierarkkinen klusterointi**. Jos objekti luokitellaan sen läheisyyden perusteella lähellä olevaan objektiin, eikä kauempana olevaan, klusterit muodostuvat jäsenten etäisyyden perusteella muihin objekteihin. Scikit-learnin agglomeratiivinen klusterointi on hierarkkista.

   ![Hierarkkinen klusterointi Infografiikka](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid-klusterointi**. Tämä suosittu algoritmi vaatii 'k':n eli muodostettavien klusterien määrän valinnan, minkä jälkeen algoritmi määrittää klusterin keskipisteen ja kerää dataa sen ympärille. [K-means-klusterointi](https://wikipedia.org/wiki/K-means_clustering) on suosittu versio centroid-klusteroinnista. Keskipiste määritetään lähimmän keskiarvon perusteella, mistä nimi johtuu. Klusterin neliöetäisyys minimoidaan.

   ![Centroid-klusterointi Infografiikka](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Jakautumispohjainen klusterointi**. Tilastolliseen mallinnukseen perustuva jakautumispohjainen klusterointi keskittyy määrittämään todennäköisyyden, että datapiste kuuluu klusteriin, ja liittää sen sen mukaisesti. Gaussian-sekoitusmenetelmät kuuluvat tähän tyyppiin.

- **Tiheysperusteinen klusterointi**. Datapisteet liitetään klustereihin niiden tiheyden perusteella, eli niiden ryhmittymisen perusteella toistensa ympärille. Datapisteet, jotka ovat kaukana ryhmästä, katsotaan poikkeavuuksiksi tai meluksi. DBSCAN, Mean-shift ja OPTICS kuuluvat tähän klusterointityyppiin.

- **Ruudukkoon perustuva klusterointi**. Moniulotteisille dataseteille luodaan ruudukko, ja data jaetaan ruudukon soluihin, jolloin muodostuu klustereita.

## Harjoitus – klusteroi datasi

Klusterointitekniikkaa tukee suuresti asianmukainen visualisointi, joten aloitetaan visualisoimalla musiikkidatamme. Tämä harjoitus auttaa meitä päättämään, mitä klusterointimenetelmää tulisi käyttää tehokkaimmin tämän datan luonteen perusteella.

1. Avaa [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) tiedosto tässä kansiossa.

1. Tuo `Seaborn`-paketti hyvää datan visualisointia varten.

    ```python
    !pip install seaborn
    ```

1. Lisää kappaledata tiedostosta [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Lataa dataframe, jossa on tietoa kappaleista. Valmistaudu tutkimaan tätä dataa tuomalla kirjastot ja tulostamalla data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Tarkista datan ensimmäiset rivit:

    |     | nimi                     | albumi                       | artisti             | artistin_genre     | julkaisupäivä | pituus | suosio     | tanssittavuus | akustisuus   | energia | instrumentaalisuus | elävyyys | äänenvoimakkuus | puheisuus | tempo   | aika-allekirjoitus |
    | --- | ------------------------ | ---------------------------- | ------------------- | ------------------ | ------------- | ------ | ---------- | ------------- | ------------ | ------ | ------------------ | -------- | --------------- | --------- | ------- | ------------------ |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b    | 2019          | 144000 | 48         | 0.666         | 0.851        | 0.42   | 0.534              | 0.11     | -6.699          | 0.0829    | 133.015 | 5                 |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop            | 2020          | 89488  | 30         | 0.71          | 0.0822       | 0.683  | 0.000169           | 0.101    | -5.64           | 0.36      | 129.993 | 3                 |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Hanki tietoa tietokehystä kutsumalla `info()`:

    ```python
    df.info()
    ```

   Tuloste näyttää tältä:

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

1. Tarkista null-arvot kutsumalla `isnull()` ja varmista, että summa on 0:

    ```python
    df.isnull().sum()
    ```

    Näyttää hyvältä:

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

1. Kuvaile data:

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

> 🤔 Jos työskentelemme klusteroinnin parissa, valvomatonta menetelmää, joka ei vaadi merkittyä dataa, miksi näytämme tämän datan etiketeillä? Tutkimusvaiheessa ne ovat hyödyllisiä, mutta klusterointialgoritmit eivät tarvitse niitä toimiakseen. Voisit yhtä hyvin poistaa sarakeotsikot ja viitata dataan sarakenumeron perusteella.

Katso datan yleisiä arvoja. Huomaa, että suosio voi olla '0', mikä tarkoittaa kappaleita, joilla ei ole sijoitusta. Poistetaan ne pian.

1. Käytä pylväsdiagrammia selvittääksesi suosituimmat genret:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Jos haluat nähdä enemmän huippuarvoja, muuta top `[:5]` suuremmaksi arvoksi tai poista se nähdäksesi kaikki.

Huomaa, kun huippugenre on kuvattu 'Missing', se tarkoittaa, että Spotify ei luokitellut sitä, joten poistetaan se.

1. Poista puuttuvat tiedot suodattamalla ne pois

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Tarkista genret uudelleen:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Selvästi kolme suosituinta genreä hallitsevat tätä datasettiä. Keskitytään `afro dancehall`, `afropop` ja `nigerian pop`, ja lisäksi suodatetaan datasetti poistamalla kaikki, joiden suosioarvo on 0 (mikä tarkoittaa, että niitä ei luokiteltu datasetissä ja niitä voidaan pitää meluna tarkoituksiimme):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Tee nopea testi nähdäksesi, korreloiko data erityisen vahvasti:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Ainoa vahva korrelaatio on `energy` ja `loudness` välillä, mikä ei ole kovin yllättävää, koska äänekäs musiikki on yleensä melko energistä. Muuten korrelaatiot ovat suhteellisen heikkoja. On mielenkiintoista nähdä, mitä klusterointialgoritmi voi tehdä tämän datan kanssa.

    > 🎓 Huomaa, että korrelaatio ei tarkoita kausaatiota! Meillä on todiste korrelaatiosta, mutta ei todiste kausaatiosta. [Hauska verkkosivusto](https://tylervigen.com/spurious-correlations) sisältää visuaaleja, jotka korostavat tätä asiaa.

Onko tässä datasetissä yhteneväisyyttä kappaleen koetun suosion ja tanssittavuuden välillä? FacetGrid näyttää, että on keskittyneitä ympyröitä, jotka asettuvat linjaan genrestä riippumatta. Voisiko olla, että nigerialaiset mieltymykset keskittyvät tiettyyn tanssittavuuden tasoon tässä genressä?  

✅ Kokeile eri datapisteitä (energy, loudness, speechiness) ja lisää tai eri musiikkigenrejä. Mitä voit löytää? Katso `df.describe()`-taulukkoa nähdäksesi datan yleisen jakauman.

### Harjoitus - datan jakauma

Ovatko nämä kolme genreä merkittävästi erilaisia tanssittavuuden suhteen niiden suosion perusteella?

1. Tutki kolmen suosituimman genren datan jakaumaa suosion ja tanssittavuuden osalta annetulla x- ja y-akselilla.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Voit löytää keskittyneitä ympyröitä yleisen yhtymäkohdan ympärillä, jotka näyttävät pisteiden jakauman.

    > 🎓 Huomaa, että tämä esimerkki käyttää KDE (Kernel Density Estimate) -kaaviota, joka esittää datan jatkuvan todennäköisyystiheyskäyrän avulla. Tämä mahdollistaa datan tulkinnan, kun työskennellään useiden jakaumien kanssa.

    Yleisesti ottaen kolme genreä asettuvat löyhästi linjaan suosion ja tanssittavuuden suhteen. Klusterien määrittäminen tässä löyhästi linjautuvassa datassa tulee olemaan haastavaa:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Luo hajontakaavio:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Hajontakaavio samoilla akseleilla näyttää samanlaisen yhtymäkohdan kuvion

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Yleisesti ottaen klusterointia varten voit käyttää hajontakaavioita datan klustereiden näyttämiseen, joten tämän tyyppisen visualisoinnin hallitseminen on erittäin hyödyllistä. Seuraavassa oppitunnissa otamme tämän suodatetun datan ja käytämme k-means-klusterointia löytääksemme ryhmiä tästä datasta, jotka näyttävät olevan päällekkäisiä mielenkiintoisilla tavoilla.

---

## 🚀Haaste

Valmistaudu seuraavaan oppituntiin tekemällä kaavio eri klusterointialgoritmeista, joita voit löytää ja käyttää tuotantoympäristössä. Minkälaisia ongelmia klusterointi yrittää ratkaista?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Ennen kuin sovellat klusterointialgoritmeja, kuten olemme oppineet, on hyvä idea ymmärtää datasetin luonne. Lue lisää tästä aiheesta [täältä](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Tämä hyödyllinen artikkeli](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) opastaa sinut läpi eri tapoja, joilla klusterointialgoritmit käyttäytyvät eri datamuotojen kanssa.

## Tehtävä

[Tutki muita visualisointeja klusterointia varten](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.