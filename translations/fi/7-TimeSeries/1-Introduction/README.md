<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-04T23:52:06+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "fi"
}
-->
# Johdanto aikasarjojen ennustamiseen

![Yhteenveto aikasarjoista sketchnotessa](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Tässä ja seuraavassa oppitunnissa opit hieman aikasarjojen ennustamisesta, joka on mielenkiintoinen ja arvokas osa koneoppimistutkijan työkalupakkia, mutta vähemmän tunnettu kuin monet muut aiheet. Aikasarjojen ennustaminen on eräänlainen "kristallipallo": aiempien suorituskykytietojen, kuten hinnan, perusteella voit ennustaa sen tulevaa potentiaalista arvoa.

[![Johdanto aikasarjojen ennustamiseen](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Johdanto aikasarjojen ennustamiseen")

> 🎥 Klikkaa yllä olevaa kuvaa katsoaksesi videon aikasarjojen ennustamisesta

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

Tämä on hyödyllinen ja mielenkiintoinen ala, jolla on todellista arvoa liiketoiminnalle, koska sillä on suora sovellus hinnoitteluun, varastonhallintaan ja toimitusketjun ongelmiin. Vaikka syväoppimistekniikoita on alettu käyttää tulevan suorituskyvyn ennustamisessa, aikasarjojen ennustaminen on edelleen ala, jossa klassiset koneoppimismenetelmät ovat keskeisessä roolissa.

> Penn Staten hyödyllinen aikasarjojen opetusmateriaali löytyy [täältä](https://online.stat.psu.edu/stat510/lesson/1)

## Johdanto

Oletetaan, että ylläpidät älykkäiden parkkimittareiden verkostoa, joka tuottaa dataa siitä, kuinka usein niitä käytetään ja kuinka pitkään ajan kuluessa.

> Mitä jos voisit ennustaa mittarin tulevan arvon sen aiemman suorituskyvyn perusteella kysynnän ja tarjonnan lakien mukaisesti?

Tarkka ennustaminen siitä, milloin toimia tavoitteen saavuttamiseksi, on haaste, jota voidaan käsitellä aikasarjojen ennustamisen avulla. Vaikka korkeammat hinnat ruuhka-aikoina eivät ilahduttaisi parkkipaikkaa etsiviä ihmisiä, se olisi varma tapa kerätä tuloja katujen siivoamiseen!

Tutustutaan joihinkin aikasarjojen algoritmeihin ja aloitetaan muistikirja datan puhdistamiseksi ja valmistamiseksi. Analysoitava data on peräisin GEFCom2014-ennustuskilpailusta. Se sisältää kolmen vuoden tuntikohtaiset sähkönkulutus- ja lämpötilatiedot vuosilta 2012–2014. Historiallisten sähkönkulutuksen ja lämpötilan mallien perusteella voit ennustaa sähkönkulutuksen tulevia arvoja.

Tässä esimerkissä opit ennustamaan yhden aikavälin eteenpäin käyttäen vain historiallista kulutusdataa. Ennen aloittamista on kuitenkin hyödyllistä ymmärtää, mitä kulissien takana tapahtuu.

## Joitakin määritelmiä

Kun kohtaat termin "aikasarja", sinun täytyy ymmärtää sen käyttö useissa eri yhteyksissä.

🎓 **Aikasarja**

Matematiikassa "aikasarja on datapisteiden sarja, joka on indeksoitu (tai listattu tai piirretty) aikajärjestyksessä. Yleisimmin aikasarja on sekvenssi, joka on otettu peräkkäisinä yhtä välein olevina ajankohtina." Esimerkki aikasarjasta on [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series) -indeksin päivittäinen päätösarvo. Aikasarjojen kuvaajien ja tilastollisen mallinnuksen käyttö on yleistä signaalinkäsittelyssä, säätiedotuksessa, maanjäristysten ennustamisessa ja muilla aloilla, joissa tapahtumat tapahtuvat ja datapisteet voidaan piirtää ajan yli.

🎓 **Aikasarjojen analyysi**

Aikasarjojen analyysi tarkoittaa edellä mainitun aikasarjadatan analysointia. Aikasarjadata voi olla eri muodoissa, mukaan lukien "keskeytetyt aikasarjat", jotka havaitsevat kuvioita aikasarjan kehityksessä ennen ja jälkeen keskeyttävän tapahtuman. Tarvittava analyysityyppi riippuu datan luonteesta. Aikasarjadata voi olla numeromuotoista tai merkkimuotoista.

Analyysi käyttää monenlaisia menetelmiä, mukaan lukien taajuusalueen ja aika-alueen menetelmät, lineaariset ja epälineaariset menetelmät ja paljon muuta. [Lue lisää](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) tavoista analysoida tämän tyyppistä dataa.

🎓 **Aikasarjojen ennustaminen**

Aikasarjojen ennustaminen tarkoittaa mallin käyttöä tulevien arvojen ennustamiseen aiemmin kerätyn datan mallien perusteella. Vaikka regressiomalleja voidaan käyttää aikasarjadatan tutkimiseen, jossa ajan indeksit ovat x-muuttujia kuvaajassa, tällaista dataa on parasta analysoida erityyppisillä malleilla.

Aikasarjadata on järjestettyjen havaintojen lista, toisin kuin data, jota voidaan analysoida lineaarisella regressiolla. Yleisin malli on ARIMA, joka on lyhenne sanoista "Autoregressive Integrated Moving Average".

[ARIMA-mallit](https://online.stat.psu.edu/stat510/lesson/1/1.1) "liittävät sarjan nykyisen arvon aiempiin arvoihin ja aiempiin ennustusvirheisiin." Ne sopivat parhaiten aika-alueen datan analysointiin, jossa data on järjestetty ajan mukaan.

> ARIMA-malleja on useita tyyppejä, joista voit oppia lisää [täältä](https://people.duke.edu/~rnau/411arim.htm), ja joita käsitellään seuraavassa oppitunnissa.

Seuraavassa oppitunnissa rakennat ARIMA-mallin käyttäen [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm) -dataa, joka keskittyy yhteen muuttujaan, joka muuttaa arvoaan ajan kuluessa. Esimerkki tällaisesta datasta on [tämä datasetti](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm), joka tallentaa kuukausittaisen CO2-pitoisuuden Mauna Loa -observatoriossa:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Tunnista muuttuja, joka muuttuu ajan kuluessa tässä datasetissä.

## Aikasarjadatan ominaisuudet, jotka kannattaa huomioida

Kun tarkastelet aikasarjadataa, saatat huomata, että sillä on [tiettyjä ominaisuuksia](https://online.stat.psu.edu/stat510/lesson/1/1.1), jotka sinun täytyy ottaa huomioon ja lieventää, jotta ymmärrät sen kuvioita paremmin. Jos pidät aikasarjadataa potentiaalisena "signaalina", jota haluat analysoida, nämä ominaisuudet voidaan ajatella "kohinana". Usein sinun täytyy vähentää tätä "kohinaa" käyttämällä tilastollisia tekniikoita.

Tässä on joitakin käsitteitä, jotka sinun tulisi tuntea voidaksesi työskennellä aikasarjojen kanssa:

🎓 **Trendit**

Trendit määritellään mitattaviksi nousuiksi ja laskuiksi ajan kuluessa. [Lue lisää](https://machinelearningmastery.com/time-series-trends-in-python). Aikasarjojen yhteydessä kyse on siitä, miten trendejä käytetään ja tarvittaessa poistetaan aikasarjoista.

🎓 **[Kausivaihtelu](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Kausivaihtelu määritellään säännöllisiksi vaihteluiksi, kuten esimerkiksi lomasesonkien vaikutuksiksi myyntiin. [Tutustu](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) siihen, miten erilaiset kuvaajat esittävät kausivaihtelua datassa.

🎓 **Poikkeamat**

Poikkeamat ovat kaukana normaalista datan vaihtelusta.

🎓 **Pitkän aikavälin syklit**

Kausivaihtelusta riippumatta data voi osoittaa pitkän aikavälin syklejä, kuten talouden laskusuhdanteita, jotka kestävät yli vuoden.

🎓 **Vakio vaihtelu**

Ajan kuluessa jotkut datat osoittavat vakioita vaihteluita, kuten energiankulutus päivällä ja yöllä.

🎓 **Äkilliset muutokset**

Data voi osoittaa äkillisiä muutoksia, jotka vaativat lisäanalyysiä. Esimerkiksi COVID-pandemian aiheuttama yritysten sulkeminen aiheutti muutoksia datassa.

✅ Tässä on [esimerkkikuvaaja aikasarjasta](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python), joka näyttää päivittäisen pelin sisäisen valuutan käytön muutaman vuoden ajalta. Voitko tunnistaa mitään yllä mainituista ominaisuuksista tässä datassa?

![Pelivaluutan käyttö](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Harjoitus - sähkönkulutusdatan käsittely

Aloitetaan aikasarjamallin luominen, joka ennustaa tulevaa sähkönkulutusta aiemman kulutuksen perusteella.

> Tämän esimerkin data on peräisin GEFCom2014-ennustuskilpailusta. Se sisältää kolmen vuoden tuntikohtaiset sähkönkulutus- ja lämpötilatiedot vuosilta 2012–2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli ja Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, heinä-syyskuu, 2016.

1. Avaa tämän oppitunnin `working`-kansiossa _notebook.ipynb_-tiedosto. Aloita lisäämällä kirjastot, jotka auttavat sinua lataamaan ja visualisoimaan dataa:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Huomaa, että käytät mukana tulevan `common`-kansion tiedostoja, jotka asettavat ympäristön ja käsittelevät datan lataamisen.

2. Tarkastele seuraavaksi dataa dataframe-muodossa kutsumalla `load_data()` ja `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Näet, että datassa on kaksi saraketta, jotka edustavat päivämäärää ja kulutusta:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Piirrä nyt data kutsumalla `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![energiakuvaaja](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Piirrä nyt vuoden 2014 heinäkuun ensimmäinen viikko antamalla se syötteenä `energy`-muuttujalle `[alkupäivä]:[loppupäivä]`-muodossa:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![heinäkuu](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Upea kuvaaja! Tarkastele näitä kuvaajia ja katso, voitko tunnistaa mitään yllä mainituista ominaisuuksista. Mitä voimme päätellä datan visualisoinnista?

Seuraavassa oppitunnissa luot ARIMA-mallin ennusteiden tekemiseksi.

---

## 🚀Haaste

Tee lista kaikista teollisuudenaloista ja tutkimusalueista, jotka hyötyisivät aikasarjojen ennustamisesta. Voitko keksiä sovelluksen näille tekniikoille taiteessa? Taloustieteessä? Ekologiassa? Vähittäiskaupassa? Teollisuudessa? Rahoituksessa? Missä muualla?

## [Jälkiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Vaikka emme käsittele niitä tässä, neuroverkkoja käytetään joskus parantamaan klassisia aikasarjojen ennustusmenetelmiä. Lue lisää niistä [tässä artikkelissa](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Tehtävä

[Visualisoi lisää aikasarjoja](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.