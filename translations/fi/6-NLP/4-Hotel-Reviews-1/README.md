<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T01:30:53+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "fi"
}
-->
# Sentimenttianalyysi hotelliarvosteluilla - datan käsittely

Tässä osiossa käytät aiemmissa oppitunneissa opittuja tekniikoita suuren datasetin tutkimiseen. Kun ymmärrät eri sarakkeiden hyödyllisyyden, opit:

- kuinka poistaa tarpeettomat sarakkeet
- kuinka laskea uutta dataa olemassa olevien sarakkeiden perusteella
- kuinka tallentaa tuloksena syntynyt datasetti lopullista haastetta varten

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

### Johdanto

Tähän mennessä olet oppinut, kuinka tekstidata eroaa numeerisesta datasta. Jos teksti on ihmisen kirjoittamaa tai puhumaa, sitä voidaan analysoida löytääkseen kaavoja ja frekvenssejä, tunteita ja merkityksiä. Tämä oppitunti vie sinut todellisen datasetin ja todellisen haasteen pariin: **[515K hotelliarvostelut Euroopassa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, joka sisältää [CC0: Public Domain -lisenssin](https://creativecommons.org/publicdomain/zero/1.0/). Datasetti on kerätty Booking.comista julkisista lähteistä. Datasetin luoja on Jiashen Liu.

### Valmistelu

Tarvitset:

* Mahdollisuuden ajaa .ipynb-tiedostoja Python 3:lla
* pandas
* NLTK, [joka sinun tulisi asentaa paikallisesti](https://www.nltk.org/install.html)
* Datasetti, joka on saatavilla Kagglesta [515K hotelliarvostelut Euroopassa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Se on noin 230 MB purettuna. Lataa se NLP-oppituntien `/data`-kansioon.

## Tutkiva data-analyysi

Tässä haasteessa oletetaan, että rakennat hotellisuositusbotin käyttäen sentimenttianalyysiä ja vieraiden arvostelupisteitä. Datasetti, jota käytät, sisältää arvosteluja 1493 eri hotellista kuudessa kaupungissa.

Pythonin, hotelliarvosteludatan ja NLTK:n sentimenttianalyysin avulla voit selvittää:

* Mitkä ovat yleisimmin käytetyt sanat ja fraasit arvosteluissa?
* Korreloivatko hotellia kuvaavat *tagit* arvostelupisteiden kanssa (esim. ovatko negatiivisemmat arvostelut tietylle hotellille *Perhe nuorten lasten kanssa* -tagilla kuin *Yksin matkustava*, mikä voisi viitata siihen, että hotelli sopii paremmin *Yksin matkustaville*?)
* Ovatko NLTK:n sentimenttipisteet "samaa mieltä" hotelliarvostelijan numeerisen pisteen kanssa?

#### Datasetti

Tutkitaan datasetti, jonka olet ladannut ja tallentanut paikallisesti. Avaa tiedosto editorissa, kuten VS Code tai jopa Excel.

Datasetin otsikot ovat seuraavat:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Tässä ne on ryhmitelty helpommin tarkasteltavaksi: 
##### Hotellin sarakkeet

* `Hotel_Name`, `Hotel_Address`, `lat` (leveysaste), `lng` (pituusaste)
  * Käyttäen *lat* ja *lng* voit piirtää kartan Pythonilla, joka näyttää hotellien sijainnit (ehkä värikoodattuna negatiivisten ja positiivisten arvostelujen mukaan)
  * Hotel_Address ei vaikuta ilmeisen hyödylliseltä, ja todennäköisesti korvaamme sen maalla helpompaa lajittelua ja hakua varten

**Hotellin meta-arvostelusarakkeet**

* `Average_Score`
  * Datasetin luojan mukaan tämä sarake on *Hotellin keskiarvo, laskettu viimeisen vuoden aikana annettujen kommenttien perusteella*. Tämä vaikuttaa epätavalliselta tavalta laskea pisteet, mutta koska data on kerätty, otamme sen toistaiseksi sellaisenaan. 
  
  ✅ Voitko keksiä toisen tavan laskea keskiarvo datasetin muiden sarakkeiden perusteella?

* `Total_Number_of_Reviews`
  * Hotellin saamien arvostelujen kokonaismäärä - ei ole selvää (ilman koodin kirjoittamista), viittaako tämä datasetin arvosteluihin.
* `Additional_Number_of_Scoring`
  * Tämä tarkoittaa, että arvostelupiste annettiin, mutta arvostelija ei kirjoittanut positiivista tai negatiivista arvostelua

**Arvostelusarakkeet**

- `Reviewer_Score`
  - Tämä on numeerinen arvo, jossa on korkeintaan yksi desimaali, välillä 2.5 ja 10
  - Ei ole selitetty, miksi alin mahdollinen piste on 2.5
- `Negative_Review`
  - Jos arvostelija ei kirjoittanut mitään, tämä kenttä sisältää "**No Negative**"
  - Huomaa, että arvostelija voi kirjoittaa positiivisen arvostelun negatiiviseen arvostelukenttään (esim. "ei ole mitään huonoa tässä hotellissa")
- `Review_Total_Negative_Word_Counts`
  - Korkeampi negatiivisten sanojen määrä viittaa matalampaan pisteeseen (ilman sentimenttianalyysiä)
- `Positive_Review`
  - Jos arvostelija ei kirjoittanut mitään, tämä kenttä sisältää "**No Positive**"
  - Huomaa, että arvostelija voi kirjoittaa negatiivisen arvostelun positiiviseen arvostelukenttään (esim. "tässä hotellissa ei ole mitään hyvää")
- `Review_Total_Positive_Word_Counts`
  - Korkeampi positiivisten sanojen määrä viittaa korkeampaan pisteeseen (ilman sentimenttianalyysiä)
- `Review_Date` ja `days_since_review`
  - Tuoreuden tai vanhentuneisuuden mittari voidaan soveltaa arvosteluun (vanhemmat arvostelut eivät välttämättä ole yhtä tarkkoja kuin uudemmat, koska hotellin johto on voinut muuttua, remontteja on voitu tehdä, tai uima-allas on lisätty jne.)
- `Tags`
  - Nämä ovat lyhyitä kuvauksia, joita arvostelija voi valita kuvaamaan vierailunsa tyyppiä (esim. yksin tai perhe), huoneen tyyppiä, oleskelun pituutta ja tapaa, jolla arvostelu lähetettiin.
  - Valitettavasti näiden tagien käyttö on ongelmallista, katso alla oleva osio, joka käsittelee niiden hyödyllisyyttä

**Arvostelijan sarakkeet**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Tämä saattaa olla tekijä suositusmallissa, esimerkiksi jos voisit määrittää, että tuotteliaammat arvostelijat, joilla on satoja arvosteluja, ovat todennäköisemmin negatiivisia kuin positiivisia. Kuitenkin minkään tietyn arvostelun arvostelijaa ei ole tunnistettu yksilöllisellä koodilla, eikä häntä siksi voida yhdistää arvostelujen joukkoon. Datasetissä on 30 arvostelijaa, joilla on 100 tai enemmän arvosteluja, mutta on vaikea nähdä, kuinka tämä voisi auttaa suositusmallia.
- `Reviewer_Nationality`
  - Jotkut saattavat ajatella, että tietyt kansallisuudet ovat todennäköisemmin antamassa positiivisia tai negatiivisia arvosteluja kansallisen taipumuksen vuoksi. Ole varovainen rakentaessasi tällaisia anekdoottisia näkemyksiä malleihisi. Nämä ovat kansallisia (ja joskus rodullisia) stereotypioita, ja jokainen arvostelija oli yksilö, joka kirjoitti arvostelun kokemuksensa perusteella. Se saattoi olla suodatettu monien linssien läpi, kuten heidän aiemmat hotellivierailunsa, matkustettu etäisyys ja heidän henkilökohtainen temperamenttinsa. Ajatus siitä, että heidän kansallisuutensa oli syy arvostelupisteeseen, on vaikea perustella.

##### Esimerkit

| Keskiarvo   Piste | Arvostelujen Kokonaismäärä | Arvostelijan   Piste | Negatiivinen <br />Arvostelu                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positiivinen   Arvostelu                 | Tagit                                                                                      |
| ----------------- | -------------------------- | -------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
> 🚨 Huomio varovaisuudesta
>
> Työskennellessäsi tämän datasetin parissa kirjoitat koodia, joka laskee jotain tekstistä ilman, että sinun tarvitsee itse lukea tai analysoida tekstiä. Tämä on NLP:n ydin: tulkita merkitystä tai tunnetta ilman, että ihminen tekee sen. On kuitenkin mahdollista, että luet joitakin negatiivisia arvosteluja. Kehottaisin sinua välttämään sitä, koska se ei ole tarpeen. Jotkut niistä ovat typeriä tai epäolennaisia negatiivisia hotelliarvosteluja, kuten "Sää ei ollut hyvä", asia, joka on hotellin tai kenenkään muun hallinnan ulkopuolella. Mutta joissakin arvosteluissa on myös synkempi puoli. Joskus negatiiviset arvostelut ovat rasistisia, seksistisiä tai ikäsyrjiviä. Tämä on valitettavaa, mutta odotettavissa datasetissä, joka on kerätty julkiselta verkkosivustolta. Jotkut arvostelijat jättävät arvosteluja, jotka saattavat olla vastenmielisiä, epämukavia tai järkyttäviä. On parempi antaa koodin mitata tunne kuin lukea ne itse ja järkyttyä. Tästä huolimatta vain vähemmistö kirjoittaa tällaisia asioita, mutta heitä on silti olemassa.
## Harjoitus - Datan tutkiminen
### Lataa data

Visuaalinen tarkastelu riittää, nyt on aika kirjoittaa koodia ja saada vastauksia! Tässä osiossa käytetään pandas-kirjastoa. Ensimmäinen tehtäväsi on varmistaa, että pystyt lataamaan ja lukemaan CSV-datan. Pandas-kirjastossa on nopea CSV-lataustyökalu, ja tulos sijoitetaan dataframeen, kuten aiemmissa oppitunneissa. Lataamassamme CSV-tiedostossa on yli puoli miljoonaa riviä, mutta vain 17 saraketta. Pandas tarjoaa monia tehokkaita tapoja käsitellä dataframea, mukaan lukien mahdollisuuden suorittaa operaatioita jokaiselle riville.

Tästä eteenpäin tässä oppitunnissa on koodiesimerkkejä, selityksiä koodista ja keskustelua tulosten merkityksestä. Käytä mukana olevaa _notebook.ipynb_-tiedostoa koodiasi varten.

Aloitetaan lataamalla datatiedosto, jota käytät:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Kun data on ladattu, voimme suorittaa siihen operaatioita. Pidä tämä koodi ohjelmasi yläosassa seuraavaa osaa varten.

## Tutki dataa

Tässä tapauksessa data on jo *puhdasta*, mikä tarkoittaa, että se on valmis käsiteltäväksi eikä sisällä muiden kielten merkkejä, jotka saattaisivat häiritä algoritmeja, jotka odottavat vain englanninkielisiä merkkejä.

✅ Saatat joutua työskentelemään datan kanssa, joka vaatii alkuvaiheen käsittelyä ennen NLP-tekniikoiden soveltamista, mutta tällä kertaa ei tarvitse. Jos joutuisit, miten käsittelisit ei-englanninkielisiä merkkejä?

Varmista, että kun data on ladattu, voit tutkia sitä koodilla. On helppoa keskittyä `Negative_Review`- ja `Positive_Review`-sarakkeisiin. Ne sisältävät luonnollista tekstiä NLP-algoritmejasi varten. Mutta odota! Ennen kuin siirryt NLP:hen ja sentimenttianalyysiin, seuraa alla olevaa koodia varmistaaksesi, että datasetissä annetut arvot vastaavat pandasilla laskettuja arvoja.

## Dataframen operaatioita

Ensimmäinen tehtävä tässä oppitunnissa on tarkistaa, ovatko seuraavat väittämät oikein kirjoittamalla koodia, joka tutkii dataframea (ilman sen muuttamista).

> Kuten monissa ohjelmointitehtävissä, on useita tapoja suorittaa tämä, mutta hyvä neuvo on tehdä se yksinkertaisimmalla ja helpoimmalla tavalla, erityisesti jos se on helpompi ymmärtää, kun palaat tähän koodiin tulevaisuudessa. Dataframeissa on kattava API, joka usein tarjoaa tehokkaan tavan tehdä haluamasi.

Käsittele seuraavia kysymyksiä kooditehtävinä ja yritä vastata niihin katsomatta ratkaisua.

1. Tulosta juuri lataamasi dataframen *shape* (muoto eli rivien ja sarakkeiden määrä).
2. Laske arvot `Reviewer_Nationality`-sarakkeelle:
   1. Kuinka monta erillistä arvoa `Reviewer_Nationality`-sarakkeessa on ja mitkä ne ovat?
   2. Mikä arvostelijan kansallisuus on datasetissä yleisin (tulosta maa ja arvostelujen määrä)?
   3. Mitkä ovat seuraavat 10 yleisintä kansallisuutta ja niiden lukumäärät?
3. Mikä hotelli sai eniten arvosteluja kunkin 10 yleisimmän arvostelijan kansallisuuden osalta?
4. Kuinka monta arvostelua datasetissä on per hotelli (hotellin arvostelujen lukumäärä)?
5. Vaikka datasetissä on `Average_Score`-sarake jokaiselle hotellille, voit myös laskea keskiarvon (laskemalla kaikkien arvostelijoiden pisteiden keskiarvon datasetissä jokaiselle hotellille). Lisää dataframeen uusi sarake otsikolla `Calc_Average_Score`, joka sisältää lasketun keskiarvon.
6. Onko hotelleja, joilla on sama (pyöristetty yhteen desimaaliin) `Average_Score` ja `Calc_Average_Score`?
   1. Kokeile kirjoittaa Python-funktio, joka ottaa argumenttina Seriesin (rivin) ja vertaa arvoja, tulostaen viestin, kun arvot eivät ole yhtäläiset. Käytä sitten `.apply()`-metodia käsitelläksesi jokaisen rivin funktiolla.
7. Laske ja tulosta, kuinka monella rivillä `Negative_Review`-sarake sisältää arvon "No Negative".
8. Laske ja tulosta, kuinka monella rivillä `Positive_Review`-sarake sisältää arvon "No Positive".
9. Laske ja tulosta, kuinka monella rivillä `Positive_Review`-sarake sisältää arvon "No Positive" **ja** `Negative_Review`-sarake arvon "No Negative".

### Koodivastaukset

1. Tulosta juuri lataamasi dataframen *shape* (muoto eli rivien ja sarakkeiden määrä).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Laske arvot `Reviewer_Nationality`-sarakkeelle:

   1. Kuinka monta erillistä arvoa `Reviewer_Nationality`-sarakkeessa on ja mitkä ne ovat?
   2. Mikä arvostelijan kansallisuus on datasetissä yleisin (tulosta maa ja arvostelujen määrä)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Mitkä ovat seuraavat 10 yleisintä kansallisuutta ja niiden lukumäärät?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Mikä hotelli sai eniten arvosteluja kunkin 10 yleisimmän arvostelijan kansallisuuden osalta?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Kuinka monta arvostelua datasetissä on per hotelli (hotellin arvostelujen lukumäärä)?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   Saatat huomata, että datasetistä laskettu tulos ei vastaa `Total_Number_of_Reviews`-arvoa. On epäselvää, edustiko datasetin arvo hotellin kokonaisarvostelujen määrää, mutta kaikkia ei ehkä ole kerätty, tai kyseessä on jokin muu laskelma. `Total_Number_of_Reviews`-arvoa ei käytetä mallissa tämän epäselvyyden vuoksi.

5. Vaikka datasetissä on `Average_Score`-sarake jokaiselle hotellille, voit myös laskea keskiarvon (laskemalla kaikkien arvostelijoiden pisteiden keskiarvon datasetissä jokaiselle hotellille). Lisää dataframeen uusi sarake otsikolla `Calc_Average_Score`, joka sisältää lasketun keskiarvon. Tulosta sarakkeet `Hotel_Name`, `Average_Score` ja `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Saatat myös ihmetellä `Average_Score`-arvoa ja miksi se joskus eroaa lasketusta keskiarvosta. Koska emme voi tietää, miksi jotkut arvot täsmäävät, mutta toiset eroavat, on turvallisinta tässä tapauksessa käyttää arvostelupisteitä, jotka meillä on, ja laskea keskiarvo itse. Ero on yleensä hyvin pieni, tässä ovat hotellit, joilla on suurin poikkeama datasetin keskiarvon ja lasketun keskiarvon välillä:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Koska vain yhdellä hotellilla on ero, joka on suurempi kuin 1, voimme todennäköisesti jättää eron huomiotta ja käyttää laskettua keskiarvoa.

6. Laske ja tulosta, kuinka monella rivillä `Negative_Review`-sarake sisältää arvon "No Negative".

7. Laske ja tulosta, kuinka monella rivillä `Positive_Review`-sarake sisältää arvon "No Positive".

8. Laske ja tulosta, kuinka monella rivillä `Positive_Review`-sarake sisältää arvon "No Positive" **ja** `Negative_Review`-sarake arvon "No Negative".

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Toinen tapa

Toinen tapa laskea rivejä ilman Lambdaa ja käyttää summaa rivien laskemiseen:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Saatat huomata, että 127 rivillä on sekä "No Negative" että "No Positive" arvot sarakkeissa `Negative_Review` ja `Positive_Review`. Tämä tarkoittaa, että arvostelija antoi hotellille numeerisen pisteen, mutta jätti kirjoittamatta sekä positiivisen että negatiivisen arvostelun. Onneksi tämä on pieni määrä rivejä (127/515738, eli 0,02 %), joten se ei todennäköisesti vääristä malliamme tai tuloksia mihinkään suuntaan. Saatat kuitenkin yllättyä, että arvosteludatasetissä on rivejä ilman arvosteluja, joten datan tutkiminen on tärkeää tällaisten rivien löytämiseksi.

Nyt kun olet tutkinut datasetin, seuraavassa oppitunnissa suodatat dataa ja lisäät sentimenttianalyysin.

---
## 🚀Haaste

Tämä oppitunti osoittaa, kuten aiemmissa oppitunneissa nähtiin, kuinka tärkeää on ymmärtää data ja sen erityispiirteet ennen operaatioiden suorittamista. Tekstipohjainen data vaatii erityistä tarkastelua. Tutki erilaisia tekstipainotteisia datasettejä ja katso, voitko löytää alueita, jotka voisivat tuoda malliin vinoumaa tai vääristynyttä sentimenttiä.

## [Oppitunnin jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Ota [tämä NLP-oppimispolku](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) ja tutustu työkaluihin, joita voit kokeilla puhe- ja tekstipainotteisten mallien rakentamisessa.

## Tehtävä

[NLTK](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.