<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T01:46:18+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "fi"
}
-->
# Mielipiteiden analysointi hotelliarvosteluista

Nyt kun olet tutkinut datasettiä yksityiskohtaisesti, on aika suodattaa sarakkeita ja käyttää NLP-tekniikoita datasetissä saadaksesi uusia näkemyksiä hotelleista.

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

### Suodatus- ja mielipiteiden analysointitoiminnot

Kuten olet ehkä huomannut, datasetissä on joitakin ongelmia. Jotkut sarakkeet sisältävät turhaa tietoa, toiset vaikuttavat virheellisiltä. Vaikka ne olisivat oikein, on epäselvää, miten ne on laskettu, eikä vastauksia voi itsenäisesti varmistaa omilla laskelmilla.

## Harjoitus: hieman lisää datan käsittelyä

Puhdista dataa hieman lisää. Lisää sarakkeita, jotka ovat hyödyllisiä myöhemmin, muuta arvoja muissa sarakkeissa ja poista joitakin sarakkeita kokonaan.

1. Alkuperäinen sarakkeiden käsittely

   1. Poista `lat` ja `lng`

   2. Korvaa `Hotel_Address`-arvot seuraavilla arvoilla (jos osoite sisältää kaupungin ja maan nimen, muuta se pelkästään kaupungiksi ja maaksi).

      Nämä ovat datasetin ainoat kaupungit ja maat:

      Amsterdam, Alankomaat

      Barcelona, Espanja

      Lontoo, Yhdistynyt kuningaskunta

      Milano, Italia

      Pariisi, Ranska

      Wien, Itävalta 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Nyt voit tehdä kyselyitä maan tasolla:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Alankomaat  |    105     |
      | Barcelona, Espanja     |    211     |
      | Lontoo, Yhdistynyt kuningaskunta |    400     |
      | Milano, Italia         |    162     |
      | Pariisi, Ranska        |    458     |
      | Wien, Itävalta         |    158     |

2. Hotellin meta-arvostelusarakkeiden käsittely

  1. Poista `Additional_Number_of_Scoring`

  1. Korvaa `Total_Number_of_Reviews` hotellin todellisella datasetissä olevien arvostelujen määrällä 

  1. Korvaa `Average_Score` omalla lasketulla keskiarvolla

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Arvostelusarakkeiden käsittely

   1. Poista `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ja `days_since_review`

   2. Säilytä `Reviewer_Score`, `Negative_Review` ja `Positive_Review` sellaisenaan
     
   3. Säilytä `Tags` toistaiseksi

     - Teemme lisäsuodatuksia tageille seuraavassa osiossa, ja sen jälkeen tagit poistetaan

4. Arvostelijan sarakkeiden käsittely

  1. Poista `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Säilytä `Reviewer_Nationality`

### Tag-sarakkeet

`Tag`-sarake on ongelmallinen, koska se sisältää listan (tekstimuodossa), joka on tallennettu sarakkeeseen. Valitettavasti tämän sarakkeen osioiden järjestys ja määrä eivät ole aina samat. Ihmisen on vaikea tunnistaa oikeita kiinnostavia fraaseja, koska datasetissä on 515 000 riviä ja 1427 hotellia, ja jokaisella on hieman erilaisia vaihtoehtoja, joita arvostelija voi valita. Tässä NLP on hyödyllinen. Voit skannata tekstiä ja löytää yleisimmät fraasit sekä laskea niiden esiintymät.

Valitettavasti emme ole kiinnostuneita yksittäisistä sanoista, vaan monisanaisista fraaseista (esim. *Liikematka*). Monisanaisen frekvenssijakauma-algoritmin suorittaminen näin suurelle datalle (6762646 sanaa) voisi viedä huomattavan paljon aikaa, mutta ilman datan tarkastelua vaikuttaa siltä, että se on välttämätön kustannus. Tässä tutkimuksellinen datan analyysi on hyödyllinen, koska olet nähnyt otoksen tageista, kuten `[' Liikematka  ', ' Yksin matkustava ', ' Yhden hengen huone ', ' Viipyi 5 yötä ', ' Lähetetty mobiililaitteesta ']`, voit alkaa kysyä, onko mahdollista vähentää merkittävästi käsittelyä, jota sinun täytyy tehdä. Onneksi se on mahdollista - mutta ensin sinun täytyy seurata muutamia vaiheita kiinnostavien tagien selvittämiseksi.

### Tagien suodatus

Muista, että datasetin tavoitteena on lisätä mielipiteitä ja sarakkeita, jotka auttavat sinua valitsemaan parhaan hotellin (itsellesi tai ehkä asiakkaalle, joka pyytää sinua tekemään hotellisuositusbotin). Sinun täytyy kysyä itseltäsi, ovatko tagit hyödyllisiä lopullisessa datasetissä. Tässä yksi tulkinta (jos tarvitsisit datasettiä muihin tarkoituksiin, eri tagit saattaisivat jäädä mukaan/pois valinnasta):

1. Matkan tyyppi on olennainen, ja sen pitäisi jäädä
2. Vierasryhmän tyyppi on tärkeä, ja sen pitäisi jäädä
3. Huoneen, sviitin tai studion tyyppi, jossa vieras yöpyi, on merkityksetön (kaikilla hotelleilla on käytännössä samat huoneet)
4. Laite, jolla arvostelu lähetettiin, on merkityksetön
5. Yöpyneiden öiden määrä *voisi* olla merkityksellinen, jos pidempiä oleskeluja yhdistettäisiin hotellin pitämiseen enemmän, mutta se on kaukaa haettua ja todennäköisesti merkityksetön

Yhteenvetona, **säilytä 2 tagityyppiä ja poista muut**.

Ensiksi, et halua laskea tageja ennen kuin ne ovat paremmassa muodossa, joten se tarkoittaa hakasulkeiden ja lainausmerkkien poistamista. Voit tehdä tämän useilla tavoilla, mutta haluat nopeimman, koska suuren datan käsittely voi kestää kauan. Onneksi pandas tarjoaa helpon tavan tehdä jokainen näistä vaiheista.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Jokainen tagi muuttuu muotoon: `Liikematka, Yksin matkustava, Yhden hengen huone, Viipyi 5 yötä, Lähetetty mobiililaitteesta`. 

Seuraavaksi kohtaamme ongelman. Joissakin arvosteluissa tai riveissä on 5 saraketta, joissakin 3, joissakin 6. Tämä johtuu siitä, miten datasetti on luotu, ja sitä on vaikea korjata. Haluat saada frekvenssilaskennan jokaisesta fraasista, mutta ne ovat eri järjestyksessä jokaisessa arvostelussa, joten laskenta voi olla väärä, eikä hotelli saa sille kuuluvaa tagia.

Sen sijaan käytät eri järjestystä hyödyksi, koska jokainen tagi on monisanainen mutta myös erotettu pilkulla! Yksinkertaisin tapa tehdä tämä on luoda 6 väliaikaista saraketta, joihin jokainen tagi lisätään sarakkeeseen, joka vastaa sen järjestystä tagissa. Voit sitten yhdistää 6 saraketta yhdeksi suureksi sarakkeeksi ja suorittaa `value_counts()`-menetelmän tuloksena olevassa sarakkeessa. Tulostamalla sen näet, että oli 2428 uniikkia tagia. Tässä pieni otos:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Lomamatka                     | 417778 |
| Lähetetty mobiililaitteesta   | 307640 |
| Pariskunta                    | 252294 |
| Viipyi 1 yön                  | 193645 |
| Viipyi 2 yötä                 | 133937 |
| Yksin matkustava              | 108545 |
| Viipyi 3 yötä                 | 95821  |
| Liikematka                    | 82939  |
| Ryhmä                         | 65392  |
| Perhe pienten lasten kanssa   | 61015  |
| Viipyi 4 yötä                 | 47817  |
| Kahden hengen huone           | 35207  |
| Standard kahden hengen huone  | 32248  |
| Superior kahden hengen huone  | 31393  |
| Perhe vanhempien lasten kanssa| 26349  |
| Deluxe kahden hengen huone    | 24823  |
| Kahden hengen tai kahden erillisen sängyn huone | 22393  |
| Viipyi 5 yötä                 | 20845  |
| Standard kahden hengen tai kahden erillisen sängyn huone | 17483  |
| Klassinen kahden hengen huone | 16989  |
| Superior kahden hengen tai kahden erillisen sängyn huone | 13570 |
| 2 huonetta                    | 12393  |

Jotkut yleisistä tageista, kuten `Lähetetty mobiililaitteesta`, eivät ole hyödyllisiä meille, joten voi olla järkevää poistaa ne ennen fraasien esiintymien laskemista, mutta se on niin nopea operaatio, että voit jättää ne mukaan ja vain ohittaa ne.

### Yöpymisen keston tagien poistaminen

Näiden tagien poistaminen on ensimmäinen askel, se vähentää hieman tarkasteltavien tagien kokonaismäärää. Huomaa, että et poista niitä datasetistä, vaan valitset poistaa ne tarkastelusta arvostelujen datasetissä.

| Yöpymisen kesto | Count  |
| ---------------- | ------ |
| Viipyi 1 yön     | 193645 |
| Viipyi 2 yötä    | 133937 |
| Viipyi 3 yötä    | 95821  |
| Viipyi 4 yötä    | 47817  |
| Viipyi 5 yötä    | 20845  |
| Viipyi 6 yötä    | 9776   |
| Viipyi 7 yötä    | 7399   |
| Viipyi 8 yötä    | 2502   |
| Viipyi 9 yötä    | 1293   |
| ...              | ...    |

Huoneita, sviittejä, studioita, huoneistoja ja niin edelleen on valtava määrä. Ne kaikki tarkoittavat suunnilleen samaa asiaa eivätkä ole merkityksellisiä sinulle, joten poista ne tarkastelusta.

| Huonetyyppi                  | Count |
| ----------------------------- | ----- |
| Kahden hengen huone          | 35207 |
| Standard kahden hengen huone | 32248 |
| Superior kahden hengen huone | 31393 |
| Deluxe kahden hengen huone   | 24823 |
| Kahden hengen tai kahden erillisen sängyn huone | 22393 |
| Standard kahden hengen tai kahden erillisen sängyn huone | 17483 |
| Klassinen kahden hengen huone | 16989 |
| Superior kahden hengen tai kahden erillisen sängyn huone | 13570 |

Lopuksi, ja tämä on ilahduttavaa (koska se ei vaatinut paljon käsittelyä), sinulle jää seuraavat *hyödylliset* tagit:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Lomamatka                                     | 417778 |
| Pariskunta                                    | 252294 |
| Yksin matkustava                              | 108545 |
| Liikematka                                    | 82939  |
| Ryhmä (yhdistetty Matkustajat ystävien kanssa)| 67535  |
| Perhe pienten lasten kanssa                   | 61015  |
| Perhe vanhempien lasten kanssa                | 26349  |
| Lemmikin kanssa                               | 1405   |

Voit väittää, että `Matkustajat ystävien kanssa` on suunnilleen sama kuin `Ryhmä`, ja olisi perusteltua yhdistää nämä kaksi kuten yllä. Koodi oikeiden tagien tunnistamiseksi löytyy [Tags notebookista](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Viimeinen vaihe on luoda uudet sarakkeet jokaiselle näistä tageista. Sitten, jokaiselle arvosteluriville, jos `Tag`-sarake vastaa yhtä uusista sarakkeista, lisää 1, jos ei, lisää 0. Lopputuloksena on laskenta siitä, kuinka moni arvostelija valitsi tämän hotellin (yhteensä) esimerkiksi liikematkalle vs lomamatkalle tai lemmikin kanssa, ja tämä on hyödyllistä tietoa hotellia suositellessa.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Tallenna tiedostosi

Lopuksi, tallenna dataset nykyisessä muodossaan uudella nimellä.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Mielipiteiden analysointitoiminnot

Tässä viimeisessä osiossa sovellat mielipiteiden analysointia arvostelusarakkeisiin ja tallennat tulokset datasettiin.

## Harjoitus: lataa ja tallenna suodatettu data

Huomaa, että nyt lataat suodatetun datasetin, joka tallennettiin edellisessä osiossa, **ei** alkuperäistä datasettiä.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Stop-sanojen poistaminen

Jos suorittaisit mielipiteiden analysoinnin negatiivisille ja positiivisille arvostelusarakkeille, se voisi kestää kauan. Testattu tehokkaalla testiläppärillä, jossa on nopea prosessori, se kesti 12–14 minuuttia riippuen siitä, mitä mielipiteiden analysointikirjastoa käytettiin. Se on (suhteellisen) pitkä aika, joten kannattaa tutkia, voiko sitä nopeuttaa. 

Stop-sanojen, eli yleisten englanninkielisten sanojen, jotka eivät muuta lauseen mielipidettä, poistaminen on ensimmäinen askel. Poistamalla ne mielipiteiden analysoinnin pitäisi toimia nopeammin, mutta ei olla vähemmän tarkka (koska stop-sanat eivät vaikuta mielipiteeseen, mutta ne hidastavat analyysiä). 

Pisimmän negatiivisen arvostelun pituus oli 395 sanaa, mutta stop-sanojen poistamisen jälkeen se on 195 sanaa.

Stop-sanojen poistaminen on myös nopea operaatio, stop-sanojen poistaminen kahdesta arvostelusarakkeesta yli 515 000 riviltä kesti 3,3 sekuntia testilaitteella. Se voi kestää hieman enemmän tai vähemmän aikaa riippuen laitteen prosessorin nopeudesta, RAM-muistista, SSD:n olemassaolosta ja joistakin muista tekijöistä. Operaation suhteellinen lyhyys tarkoittaa, että jos se parantaa mielipiteiden analysoinnin aikaa, se on sen arvoista.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Mielipiteiden analysoinnin suorittaminen

Nyt sinun pitäisi laskea mielipiteiden analysointi sekä negatiivisille että positiivisille arvostelusarakkeille ja tallentaa tulos kahteen uuteen sarakkeeseen. Mielipiteiden testaus tapahtuu vertaamalla sitä arvostelijan antamaan pisteytykseen samasta arvostelusta. Esimerkiksi, jos mielipiteiden analyysi arvioi negatiivisen arvostelun mielipiteeksi 1 (äärimmäisen positiivinen mielipide) ja positiivisen arvostelun mielipiteeksi 1, mutta arvostelija antoi hotellille alhaisimman mahdollisen pisteen, silloin joko arvosteluteksti ei vastaa pisteytystä tai mielipiteiden analysoija ei pystynyt tunnistamaan mielipidettä oikein. Sinun pitäisi odottaa, että jotkut mielipiteiden pisteet ovat täysin vääriä, ja usein se on selitettävissä, esim. arvostelu voi olla erittäin sarkastinen "Tietenkin RAKASTIN nukkumista huoneessa ilman lämmitystä" ja mielipiteiden analysoija ajattelee, että se on positiivinen mielipide, vaikka ihminen lukisi sen tietäen, että se oli sarkasmia.
NLTK tarjoaa erilaisia sentimenttianalysaattoreita, joita voit käyttää, ja voit vaihtaa niitä nähdäksesi, onko sentimentti tarkempi vai ei. Tässä käytetään VADER-sentimenttianalyysiä.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, kesäkuu 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Myöhemmin ohjelmassasi, kun olet valmis laskemaan sentimentin, voit soveltaa sitä jokaiseen arvosteluun seuraavasti:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Tämä vie noin 120 sekuntia tietokoneellani, mutta aika vaihtelee tietokoneittain. Jos haluat tulostaa tulokset ja nähdä, vastaako sentimentti arvostelua:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Viimeinen asia, joka tiedostolle täytyy tehdä ennen kuin käytät sitä haasteessa, on tallentaa se! Kannattaa myös harkita kaikkien uusien sarakkeiden uudelleenjärjestämistä, jotta niiden kanssa olisi helpompi työskennellä (ihmisen näkökulmasta tämä on kosmeettinen muutos).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Sinun tulisi ajaa koko koodi [analyysivihkosta](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (sen jälkeen kun olet ajanut [suodatusvihkon](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) luodaksesi Hotel_Reviews_Filtered.csv-tiedoston).

Kertauksena, vaiheet ovat:

1. Alkuperäinen dataset-tiedosto **Hotel_Reviews.csv** tutkitaan edellisessä oppitunnissa [tutkimusvihkon](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) avulla
2. Hotel_Reviews.csv suodatetaan [suodatusvihkolla](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), jolloin syntyy **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv käsitellään [sentimenttianalyysivihkolla](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), jolloin syntyy **Hotel_Reviews_NLP.csv**
4. Käytä Hotel_Reviews_NLP.csv-tiedostoa alla olevassa NLP-haasteessa

### Yhteenveto

Kun aloitit, sinulla oli datasetti, jossa oli sarakkeita ja dataa, mutta kaikkea ei voitu vahvistaa tai käyttää. Olet tutkinut dataa, suodattanut tarpeettoman pois, muuntanut tageja hyödyllisiksi, laskenut omia keskiarvoja, lisännyt sentimenttisarakkeita ja toivottavasti oppinut mielenkiintoisia asioita luonnollisen tekstin käsittelystä.

## [Oppitunnin jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Haaste

Nyt kun datasetti on analysoitu sentimentin osalta, kokeile käyttää oppimiasi strategioita (ehkä klusterointia?) tunnistaaksesi sentimenttiin liittyviä kuvioita.

## Kertaus & Itseopiskelu

Käy läpi [tämä Learn-moduuli](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) oppiaksesi lisää ja käyttämään erilaisia työkaluja sentimentin tutkimiseen tekstissä.

## Tehtävä

[Kokeile eri datasettiä](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.