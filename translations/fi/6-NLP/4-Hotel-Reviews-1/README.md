# Sentimenttianalyysi hotelliarvosteluilla – datan käsittely

Tässä osassa käytät edellisten oppituntien menetelmiä tutkiaksesi laajaa aineistoa. Kun ymmärrät eri sarakkeiden hyödyllisyyden, opit:

- miten poistaa tarpeettomat sarakkeet
- miten laskea uutta tietoa olemassa olevien sarakkeiden perusteella
- miten tallentaa lopullista haastetta varten muokattu aineisto

## [Esiluentokoe](https://ff-quizzes.netlify.app/en/ml/)

### Johdanto

Olet oppinut, että tekstuaalinen data poikkeaa merkittävästi numeerisesta datasta. Jos teksti on ihmisen kirjoittamaa tai puhuma, sitä voidaan analysoida löytääkseen kuvioita, esiintymistiheyksiä, tunteita ja merkityksiä. Tämä oppitunti vie sinut oikeaan tietoaineistoon ja aitoon haasteeseen: **[515 000 hotelliarvostelua Euroopasta](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, ja aineisto on julkaistu [CC0: Public Domain -lisenssillä](https://creativecommons.org/publicdomain/zero/1.0/). Se on kerätty Booking.comin julkisista lähteistä. Aineiston tekijä on Jiashen Liu.

### Valmistelut

Tarvitset seuraavaa:

* Kyvyn suorittaa .ipynb-muotoisia muistikirjoja käyttäen Python 3:a
* pandas-kirjaston
* NLTK:n, [jonka tulee asentaa paikallisesti](https://www.nltk.org/install.html)
* Aineiston, joka on saatavilla Kagglesta: [515 000 hotelliarvostelua Euroopasta](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Pakkaamattomana noin 230 MB. Lataa se `/data`-kansioon, joka liittyy näihin NLP-oppitunteihin.

## Tutkiva datan analyysi

Tässä haasteessa rakennat hotellisuositusbottia käyttäen sentimenttianalyysiä ja asiakkaiden arvosteluja. Käyttämäsi aineisto sisältää arvosteluja 1493 eri hotellista kuudessa kaupungissa.

Pythonin, hotelliarvosteluaineiston ja NLTK:n sentimenttianalyysin avulla voit selvittää:

* Mitkä sanat ja ilmaisut ovat eniten käytettyjä arvosteluissa?
* Korreloivatko hotellin viralliset *tags*-tägit arvostelupisteiden kanssa (esim. onko enemmän negatiivisia arvosteluja *Perhe lapsineen* -tägillä kuin *Yksin matkustava*, mikä saattaisi viitata siihen, että hotelli sopii paremmin *Yksin matkustaville*)?
* Vastaavatko NLTK:n sentimenttipisteet hotelliarvostelijan numeerisia pisteitä?

#### Aineisto

Tutustutaan ladattuun ja paikallisesti tallennettuun aineistoon. Avaa tiedosto esimerkiksi VS Codessa tai Excelissä.

Aineiston sarakkeet ovat seuraavat:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Tässä sarakkeet ryhmiteltyinä helpompaan tarkasteluun:
##### Hotelli-sarakkeet

* `Hotel_Name`, `Hotel_Address`, `lat` (leveysaste), `lng` (pituusaste)
  * Näillä latitude- ja longitude-tiedoilla voisi piirtää Pythonilla kartan hotellien sijainneista (esim. väritettynä negatiivisten ja positiivisten arvostelujen mukaan)
  * Hotel_Address ei vaikuta kovin hyödylliseltä, ja saatamme korvata sen maalla helpompaa lajittelua ja hakua varten

**Hotellin meta-arvostelusarakkeet**

* `Average_Score`
  * Aineiston luojan mukaan tämä sarake kertoo *hotellin keskimääräisen pisteen, laskettuna viime vuoden uusimman kommentin perusteella*. Tämä tapa vaikuttaa erikoiselta, mutta se on kerätty data, joten voimme ottaa sen väliaikaisena arvona.
  
  ✅ Voisitko keksiä toisen tavan laskea keskipiste tämän aineiston muista sarakkeista?

* `Total_Number_of_Reviews`
  * Hotellin saamien arvostelujen kokonaismäärä – ei ole selvää (ilman koodin kirjoittamista), viittaako tämä tämän aineiston arvosteluihin.
* `Additional_Number_of_Scoring`
  * Tämä tarkoittaa, että arvostelupiste annettiin, mutta positiivista tai negatiivista arvostelua ei kirjoitettu

**Arvostelusarakkeet**

- `Reviewer_Score`
  - Numeerinen arvo korkeintaan yhdellä desimaalilla, vaihtelu 2.5 ja 10 välillä
  - Ei selitetä, miksi alin piste on 2.5
- `Negative_Review`
  - Jos arvostelija ei kirjoittanut mitään, tässä lukee "**No Negative**" (ei negatiivista)
  - Huomaa, että arvostelija voi kirjoittaa positiivisen arvostelun tähän kenttään (esim. "tässä hotellissa ei ole mitään huonoa")
- `Review_Total_Negative_Word_Counts`
  - Korkeampi negatiivisten sanojen määrä usein tarkoittaa matalampaa pistettä (ilman sentimentin tarkistusta)
- `Positive_Review`
  - Jos arvostelija ei kirjoittanut mitään, tässä lukee "**No Positive**" (ei positiivista)
  - Huomaa, että arvostelija voi kirjoittaa negatiivisen arvostelun tähän kenttään (esim. "tässä hotellissa ei ole lainkaan mitään hyvää")
- `Review_Total_Positive_Word_Counts`
  - Korkeampi positiivisten sanojen määrä usein tarkoittaa korkeampaa pistettä (ilman sentimentin tarkistusta)
- `Review_Date` ja `days_since_review`
  - Arvostelun tuoreudelle tai vanhentuneisuudelle voisi antaa painoarvon (vanhemmat arvostelut eivät ehkä ole enää yhtä oikeita, koska hotellin johto on vaihtunut, remontteja on tehty tai uima-allas lisätty jne.)
- `Tags`
  - Nämä ovat lyhyitä kuvauksia, joita arvostelija on voinut valita kuvaamaan vierailijan tyyppiä (esim. yksin matkustava tai perhe), huonetyyppiä, oleskelun kestoa ja miten arvostelu on jätetty.
  - Valitettavasti näiden tägien käyttö on hankalaa – katso alla oleva osio, jossa niiden hyödyllisyyttä käsitellään

**Arvostelijan sarakkeet**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Tämä voisi olla tekijä suositusmallissa, jos pystyisit toteamaan, että aktiivisemmat arvostelijat kirjoittavat todennäköisemmin negatiivisia kuin positiivisia arvioita. Kuitenkaan yksittäistä arvostelijaa ei tunnisteta uniikilla koodilla, joten arvioita ei voi yhdistää tiettyihin arvostelijoihin. 30 arvostelijalla on 100 tai useampia arvosteluja, mutta vaikea nähdä, miten tämä auttaisi suositusmallia.
- `Reviewer_Nationality`
  - Jotkut saattavat ajatella, että kansallisuus vaikuttaa positiivisten tai negatiivisten arvostelujen todennäköisyyteen. Ole varovainen sisällyttäessäsi tällaisia ennakko-olettamuksia malleihisi. Ne ovat kansalliseen (ja joskus rotuun liittyvään) stereotypiaan perustuvia, vaikka jokainen arvostelija on yksilö, joka kirjoitti arvostelunsa kokemuksensa perusteella. Kokemus voi olla suodatettu monien asioiden, kuten aiempien hotelliyöpymisten, matkustetun matkan ja henkilökohtaisen temperamentin, kautta. On vaikea perustella, että kansallisuus olisi arvostelupisteen syy.

##### Esimerkkejä

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Tämä on tällä hetkellä rakennustyömaa, jossa minut terrorisoitiin varhain aamulla ja koko päivän sietämättömällä rakennusmelulla, kun lepäsin pitkän matkan jälkeen ja työskentelin huoneessa. Ihmiset työskentelivät koko päivän, eli käytettiin piikkaajia viereisissä huoneissa. Pyysin huoneenvaihtoa, mutta hiljaista huonetta ei ollut tarjolla. Pahentaakseni tilannetta minulta veloitettiin liikaa. Check-out illalla, koska minun piti lähteä hyvin aikaisella lennolla, sain asianmukaisen laskun. Päivää myöhemmin hotelli veloitti toisen maksun ilman suostumustani, yli varatun hinnan. Tämä paikka on kamala. Älä rankaise itseäsi varaamalla tänne | Ei mitään Kamala paikka, pysy poissa | Työmatka Pariskunta Standard Double Room Majoitus 2 yötä |

Kuten näet, tämä asiakas ei viihtynyt hotellissa. Hotellilla on hyvä keskimääräinen piste 7.8 ja 1945 arvostelua, mutta tämä arvostelija antoi sille 2.5 ja kirjoitti 115 sanaa siitä, kuinka negatiivinen heidän oleskelunsa oli. Jos he eivät olisi kirjoittaneet mitään Positive_Review-kenttään, voisi kuvitella, ettei mitään positiivista ollut, mutta he kirjoittivat kuitenkin 7 varoittavaa sanaa. Jos vain laskemme sanat tunteiden tai merkityksen sijaan, arvostelun sisältö voi vääristyä. Heidän pisteensä 2.5 on outo, koska jos hotelli oli niin huono, miksi antaa pisteitä lainkaan? Tutkimalla aineistoa tarkemmin huomaat, että alin mahdollinen pistemäärä on 2.5, ei 0. Korkein mahdollinen on 10.

##### Tags-tägit

Kuten aiemmin mainittu, ensimmäiseltä silmäykseltä idee käyttää `Tags`-kenttää luokitukseen vaikuttaa järkevältä. Valitettavasti nämä tägät eivät ole standardoituja, mikä tarkoittaa, että toisessa hotellissa vaihtoehdot voivat olla *Yhden hengen huone*, *Kahden hengen huone* ja *Parihuone*, mutta toisessa hotellissa ne voivat olla *Deluxe yhden hengen huone*, *Classic Queen -huone* ja *Executive King -huone*. Nämä voivat tarkoittaa samaa asiaa, mutta kun vaihteluita on niin paljon, vaihtoehdot ovat:

1. Yrittää muuttaa kaikki termit yhdeksi standardiksi, mikä on hyvin vaikeaa, koska ei ole selvää, mikä muunto olisi missäkin tapauksessa (esimerkiksi *Classic yhden hengen huone* vastaa *Yhden hengen huonetta*, mutta *Superior Queen Room with Courtyard Garden or City View* on paljon vaikeampi muuntaa)

1. Voidaan käyttää NLP-menetelmää ja mitata tiettyjen termien, kuten *Solo*, *Business Traveller* tai *Family with young kids*, esiintymistiheyttä hotellikohtaisesti ja ottaa se mukaan suositukseen

Tägit ovat yleensä (mutta eivät aina) yksi kenttä, jossa on 5–6 pilkuilla eroteltua arvoa, jotka vastaavat matkustustyyppiä, asiakastyyppiä, huonetyyppiä, yöpymisten määrää ja laitetta, jolla arvostelu lähetettiin. Jotkut arvostelijat eivät kuitenkaan täytä kaikkia kenttiä, joten arvot eivät ole aina samassa järjestyksessä.

Esimerkiksi *Ryhmän tyyppi* -kentässä on 1025 erilaista arvoa `Tags`-sarakkeessa, mutta valitettavasti vain osa viittaa ryhmään (osa liittyy huonetyyppiin jne.). Jos suodatat vain perhettä koskevat, tuloksissa on paljon *Perhehuone* -tyyppisiä tuloksia. Jos mukaan otetaan termi *with*, eli lasketaan *Family with* -arvot, tulokset ovat parempia: yli 80 000 tuloksesta 515 000:sta sisältää ilmaisun "Family with young children" tai "Family with older children".

Tämä tarkoittaa, että tägisarakkeen avulla voi olla hyödyllistä, mutta sen tekeminen käyttökelpoiseksi vaatii työtä.

##### Keskimääräinen hotellin piste

Aineistossa on joitain outouksia tai ristiriitaisuuksia, joita en pysty selittämään, mutta jotka esitetään tässä, jotta tiedät niistä mallia rakentaessasi. Jos ratkaiset asian, ilmoitathan siitä keskusteluosiossa!

Aineistossa on seuraavat sarakkeet, jotka liittyvät keskipisteeseen ja arvostelujen määrään:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Yhden hotellin suurin arvostelujen määrä tässä aineistossa on *Britannia International Hotel Canary Wharf* 4789 arvostelua 515 000:sta. Mutta jos katsot `Total_Number_of_Reviews` -arvoa tässä hotellissa, se on 9086. Voisit päätellä, että on paljon pisteitä ilman arvostelua, joten lisätään mukaan myös `Additional_Number_of_Scoring` -arvo. Tämä on 2682, ja sen lisääminen 4789:ään antaa 7 471, mikä on silti 1615 vähemmän kuin `Total_Number_of_Reviews`.

Jos katsoo `Average_Score` -saraketta, voisi päätellä, että se on aineiston arvostelujen keskiarvo, mutta Kagglessa kuvaillaan sitä seuraavasti: "*Eniten ajankohtaisen viime vuoden kommentin mukaan laskettu hotellin keskipiste*". Tämä ei vaikuta kovin hyödylliseltä, mutta voimme laskea oman keskiarvon arvostelupisteiden perusteella aineistosta. Käyttäen samaa hotellia esimerkkinä, keskimääräinen hotellin piste on annettu arvoksi 7.1, mutta laskettu (aineistossa olevien arvostelijoiden keskiarvo) on 6.8. Se on lähellä, mutta ei sama, ja voimme vain arvata, että `Additional_Number_of_Scoring` -arvostelut nostivat keskiarvon 7.1:een. Valitettavasti, kun tätä ei voi testata tai todistaa, on vaikea luottaa tai käyttää `Average_Score`, `Additional_Number_of_Scoring` ja `Total_Number_of_Reviews` arvoja, kun ne perustuvat tai viittaavat dataan, jota meillä ei ole.

Tilannetta monimutkaistaa se, että toiseksi suurimman arvostelumäärän hotelli saa laskettua keskipisteeksi 8.12 ja aineiston `Average_Score` on 8.1. Onko tämä oikea piste sattumaa vai onko ensimmäinen hotelli poikkeama?


On mahdollista, että nämä hotellit saattavat olla poikkeuksia ja että ehkä suurin osa arvoista täsmää (mutta jotkut eivät jostain syystä), kirjoitamme seuraavaksi lyhyen ohjelman tutkiaksemme datasetin arvoja ja määrittääksemme arvojen oikean käytön (tai käyttämättömyyden).

> 🚨 Varoituksen sana
>
> Kun työskentelet tämän datasetin kanssa, kirjoitat koodia, joka laskee jotain tekstistä ilman, että sinun tarvitsee itse lukea tai analysoida tekstiä. Tämä on NLP:n ydin, tulkita merkitys tai tunnelma ilman ihmisen osallistumista. On kuitenkin mahdollista, että luet joitakin negatiivisia arvosteluja. Kehotan sinua olemaan tekemättä niin, koska sinun ei tarvitse. Jotkut niistä ovat typeriä tai merkityksettömiä negatiivisia hotelliarvosteluja, kuten "Sää ei ollut hyvä", jotain hotellin hallinnan ulkopuolella, tai oikeastaan kenenkään hallinnan ulkopuolella. Mutta joissakin arvosteluissa on myös pimeä puoli. Joskus negatiiviset arvostelut ovat rasistisia, seksistisiä tai ikään perustuvia. Tämä on valitettavaa mutta odotettavissa julkisen verkkosivuston keräämässä datasetissä. Jotkut arvostelijat jättävät arvosteluja, jotka koet epämiellyttävinä, epämukavina tai loukkaavina. On parempi antaa koodin mitata tunnelmaa kuin lukea niitä itse ja loukkaantua. Tosin tällaisia kirjoittavia on vähemmistönä, mutta heitä on kuitenkin olemassa.

## Harjoitus - Datan tutkiminen
### Lataa data

Dataa on tutkittu visuaalisesti tarpeeksi, nyt kirjoitat koodia ja saat vastauksia! Tämä osio käyttää pandas-kirjastoa. Ensimmäinen tehtäväsi on varmistaa, että pystyt lataamaan ja lukemaan CSV-dataa. Pandas-kirjastolla on nopea CSV-lataaja, ja tulos sijoitetaan dataframeen, kuten aiemmissa oppitunneissa. Lataamamme CSV:ssä on yli puoli miljoonaa riviä, mutta vain 17 saraketta. Pandas tarjoaa monia tehokkaita tapoja käsitellä dataa, mukaan lukien mahdollisuuden suorittaa operaatioita jokaiselle riville.

Tästä oppitunnista eteenpäin on mukana koodipätkiä ja joitain selityksiä sekä keskustelua tuloksista. Käytä mukana tullutta _notebook.ipynb_-tiedostoa koodillesi.

Aloitetaan lataamalla käyttämäsi datatiedosto:

```python
# Lataa hotelliarvostelut CSV-tiedostosta
import pandas as pd
import time
# tuodaan time, jotta aloitus- ja lopetusaikaa voidaan käyttää tiedoston latausajan laskemiseen
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df on 'DataFrame' - varmista, että latasit tiedoston data-kansioon
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Nyt kun data on ladattu, voimme tehdä siihen operaatioita. Pidä tämä koodi ohjelmasi yläosassa seuraavaa osaa varten.

## Tutki dataa

Tässä tapauksessa data on jo *puhdasta*, eli se on valmis käsiteltäväksi eikä sisällä muita kuin englantilaisia merkkejä, jotka voisivat sotkea algoritmit.

✅ Saatat joutua työskentelemään datan kanssa, joka vaatii jonkinlaista esikäsittelyä ennen NLP-teknisten menetelmien soveltamista, mutta ei tällä kertaa. Mutta jos joutuisit, kuinka käsittelisit ei-englanninkielisiä merkkejä?

Varmista hetki, että datan latauduttua voit tutkia sitä koodilla. On hyvin helppo keskittyä sarakkeisiin `Negative_Review` ja `Positive_Review`. Niissä on luonnollista tekstiä, jota NLP-algoritmit voivat käsitellä. Mutta odota! Ennen kuin hyppäät NLP:n ja tunnelman analysointiin, sinun pitäisi seurata alla olevaa koodia tarkistaaksesi, vastaavatko datasetissä annetut arvot arvoja, jotka lasket pandasilla.

## Dataframen operoinnit

Ensimmäinen tehtävä tässä oppitunnissa on tarkistaa, ovatko seuraavat väittämät oikein kirjoittamalla koodi, joka tutkii dataframeta (ilman sen muuttamista).

> Kuten monissa ohjelmointitehtävissä, on useita tapoja suorittaa tämä, mutta hyvä neuvo on tehdä se yksinkertaisimmalla, helpoimmalla tavalla, erityisesti jos on helpompaa ymmärtää tätä koodia myöhemmin. Dataframella on laaja API, joka usein mahdollistaa halutun tekemisen tehokkaasti.

Kohtele seuraavia kysymyksiä kooditehtävinä ja yritä vastata niihin ilman ratkaisun katsomista.

1. Tulosta juuri lataamasi dataframen *muoto* (muoto on rivien ja sarakkeiden lukumäärä)
2. Laske arvostelijoiden kansallisuuksien esiintymistiheys:
   1. Kuinka monta eri arvoa sarakkeessa `Reviewer_Nationality` on ja mitkä ne ovat?
   2. Mikä arvostelijan kansallisuus on yleisin datasetissä (tulosta maa ja arvostelujen määrä)?
   3. Mitkä ovat seuraavat 10 yleisintä kansallisuutta ja niiden esiintymistiheys?
3. Mikä oli yleisimmin arvosteltu hotelli kullekin kymmenelle yleiselle arvostelijakansallisuudelle?
4. Kuinka monta arvostelua hotellilla on datasetissä (hotellin esiintymistiheys)?
5. Vaikka datasetissä on sarake `Average_Score` jokaiselle hotellille, voit myös laskea keskiarvon (otsikoiden kaikkien arvostelijoiden pisteiden keskiarvon jokaiselle hotellille). Lisää dataframeesi uusi sarake otsikolla `Calc_Average_Score`, joka sisältää lasketun keskiarvon.
6. Onko joillakin hotelleilla sama (pyöristettynä 1 desimaaliin) `Average_Score` ja `Calc_Average_Score`?
   1. Yritä kirjoittaa Python-funktio, joka ottaa sarjan (rivin) argumentiksi ja vertaa arvoja, tulostaen viestin, kun arvot eivät ole yhtä suuret. Käytä sitten `.apply()`-metodia käsittelemään jokainen rivi funktiolla.
7. Laske ja tulosta, kuinka monessa rivissä sarakkeen `Negative_Review` arvo on "No Negative" 
8. Laske ja tulosta, kuinka monessa rivissä sarakkeen `Positive_Review` arvo on "No Positive"
9. Laske ja tulosta, kuinka monessa rivissä sarakkeen `Positive_Review` arvo on "No Positive" **ja** sarakkeen `Negative_Review` arvo on "No Negative"
### Koodivastaukset

1. Tulosta juuri lataamasi dataframen *muoto* (muoto on rivien ja sarakkeiden lukumäärä)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Laske arvostelijoiden kansallisuuksien esiintymistiheys:

   1. Kuinka monta eri arvoa sarakkeessa `Reviewer_Nationality` on ja mitkä ne ovat?
   2. Mikä arvostelijan kansallisuus on yleisin datasetissä (tulosta maa ja arvostelujen määrä)?

   ```python
   # value_counts() luo Series-olion, jolla on tässä tapauksessa indeksi ja arvot, eli maa ja kuinka usein ne esiintyvät arvioijan kansalaisuuksissa
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # tulosta Seriesin ensimmäiset ja viimeiset rivit. Vaihda nationality_freq.to_string() tulostaaksesi kaikki tiedot
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

   3. Mitkä ovat seuraavat 10 yleisintä kansallisuutta ja niiden esiintymistiheys?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Huomaa, että arvoissa on alussa välilyönti, strip() poistaa sen tulostusta varten
      # Mitkä ovat 10 yleisintä kansallisuutta ja niiden esiintymistiheydet?
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

3. Mikä oli yleisimmin arvosteltu hotelli kullekin kymmenelle yleiselle arvostelijakansallisuudelle?

   ```python
   # Mikä oli eniten arvosteltu hotelli kymmenen suosituimman kansallisuuden joukossa
   # Yleensä Pandasissa vältät eksplisiittisiä silmukoita, mutta halusin näyttää uuden dataframen luomisen kriteerien avulla (älä tee tätä suurilla tietomäärillä, koska se voi olla hyvin hidasta)
   for nat in nationality_freq[:10].index:
      # Ensin poimi kaikki rivit, jotka täyttävät kriteerit, uuteen dataframeen
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Nyt haetaan hotellin esiintymistiheys
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

4. Kuinka monta arvostelua hotellilla on datasetissä (hotellin esiintymistiheys)?

   ```python
   # Luo ensin uusi dataframe vanhan pohjalta poistamalla tarpeettomat sarakkeet
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Ryhmittele rivit Hotel_Name mukaan, laske ne ja laita tulos uuteen sarakkeeseen Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Poista kaikki päällekkäiset rivit
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
   
   Saatat huomata, että datasetissä *lasketut* tulokset eivät vastaa arvoa `Total_Number_of_Reviews`. Ei ole selvää, kuvaako datasetin arvo hotellin kokonaistarvostelumäärää, mutta kaikki arvostelut eivät ole mukana, vai onko kyseessä muu laskelma. Arvoa `Total_Number_of_Reviews` ei käytetä mallissa tämän epäselvyyden vuoksi.

5. Vaikka datasetissä on sarake `Average_Score` jokaiselle hotellille, voit myös laskea keskiarvon (kaikkien arvostelijoiden pisteiden keskiarvon jokaiselle hotellille). Lisää dataframeresultsiisi uusi sarake otsikolla `Calc_Average_Score`, joka sisältää tämän lasketun keskiarvon. Tulosta sarakkeet `Hotel_Name`, `Average_Score` ja `Calc_Average_Score`.

   ```python
   # määrittele funktio, joka ottaa rivin ja suorittaa sillä jonkin laskelman
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' on matemaattinen sana 'keskiarvolle'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Lisää uusi sarake, jossa on kahden keskiarvopistemäärän välinen ero
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Luo df ilman kaikkia Hotel_Name-kentän päällekkäisyyksiä (joten vain 1 rivi per hotelli)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Järjestä dataframe löytääksesi pienimmän ja suurimman keskiarvopistemäärän eron
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Saatat myös pohtia `Average_Score`-arvoa ja miksi se joskus eroaa lasketusta keskiarvosta. Koska emme voi tietää, miksi jotkut arvot täsmäävät, mutta toiset eroavat, on tässä tapauksessa turvallisinta käyttää laskettuja arvostelupisteitä keskiarvon laskemiseen itse. Tosin erot ovat yleensä hyvin pieniä, tässä ovat hotellit, joilla on suurimmat poikkeamat datasetin keskiarvosta ja lasketusta keskiarvosta:

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

   Kun vain yhdellä hotellilla ero pisteissä on yli 1, voimme todennäköisesti jättää erot huomioimatta ja käyttää laskettua keskiarvosummaa.

6. Laske ja tulosta, kuinka monessa rivissä sarakkeen `Negative_Review` arvo on "No Negative" 

7. Laske ja tulosta, kuinka monessa rivissä sarakkeen `Positive_Review` arvo on "No Positive"

8. Laske ja tulosta, kuinka monessa rivissä sarakkeen `Positive_Review` arvo on "No Positive" **ja** sarakkeen `Negative_Review` arvo on "No Negative"

   ```python
   # lambda-funktioiden kanssa:
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

Toinen tapa laskea kohteet ilman lambda-funktioita ja käyttää summaa rivien laskemiseen:

   ```python
   # ilman lambdoja (käyttämällä sekoitusta merkintöjä näyttämään, että voit käyttää molempia)
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

   Saatat huomata, että 127 rivillä molempien sarakkeiden `Negative_Review` ja `Positive_Review` arvot ovat vastaavasti "No Negative" ja "No Positive". Tämä tarkoittaa, että arvostelija antoi hotellille numeerisen arvion, mutta ei kirjoittanut positiivista eikä negatiivista arvostelua. Onneksi tämä on pieni osuus riveistä (127 / 515738, eli 0,02 %), joten se ei todennäköisesti vinouta mallia tai tuloksia, mutta et ehkä odottanut arvosteludatan sisältävän rivejä ilman varsinaisia arvosteluja, joten datan tutkiminen tällaisten rivien löytämiseksi on tärkeää.

Nyt kun olet tutkinut datasetin, voit seuraavassa oppitunnissa suodattaa dataa ja lisätä tunnelman analysoinnin.

---
## 🚀Haaste

Tämä oppitunti osoittaa, kuten aiemmissa oppitunneissa, kuinka kriittistä on ymmärtää oma data ja sen ominaisuudet ennen operaatioiden tekemistä. Erityisesti tekstipohjainen data vaatii huolellista tarkastelua. Kaiva eri tekstipainotteisia datasettejä ja katso, löydätkö alueita, jotka voisivat aiheuttaa vinoutumia tai vääristynyttä tunnelmaa mallissa.

## [Luenton jälkeinen tietovisa](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itsenäinen opiskelu

Suosittelemme [tätä oppimispolkua NLP:stä](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) löytääksesi työkaluja puhe- ja tekstipainotteisten mallien rakentamiseen.

## Tehtävä

[NLTK](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, otathan huomioon, että automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä on virallinen lähde. Tärkeissä asioissa suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä aiheutuvista väärinymmärryksistä tai tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->