<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-04T23:42:48+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "fi"
}
-->
# Rakenna regressiomalli Scikit-learnilla: valmistele ja visualisoi data

![Datavisualisoinnin infografiikka](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tämä oppitunti on saatavilla myös R-kielellä!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Johdanto

Nyt kun sinulla on tarvittavat työkalut koneoppimismallien rakentamiseen Scikit-learnilla, olet valmis aloittamaan datan analysoinnin ja kysymysten esittämisen. Kun työskentelet datan parissa ja sovellat koneoppimisratkaisuja, on erittäin tärkeää osata esittää oikeat kysymykset, jotta datan potentiaali saadaan kunnolla hyödynnettyä.

Tässä oppitunnissa opit:

- Kuinka valmistella data mallin rakentamista varten.
- Kuinka käyttää Matplotlibia datan visualisointiin.

## Oikean kysymyksen esittäminen datalle

Kysymys, johon haluat vastauksen, määrittää sen, millaisia koneoppimisalgoritmeja käytät. Ja vastausten laatu riippuu suuresti datan luonteesta.

Tutustu tämän oppitunnin [dataan](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv). Voit avata tämän .csv-tiedoston VS Codessa. Nopealla silmäyksellä huomaat, että datassa on tyhjiä kohtia sekä sekoitus merkkijonoja ja numeerista dataa. Lisäksi on outo sarake nimeltä 'Package', jossa data vaihtelee 'sacks', 'bins' ja muiden arvojen välillä. Data on itse asiassa melko sekavaa.

[![ML aloittelijoille - Kuinka analysoida ja siivota datasetti](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML aloittelijoille - Kuinka analysoida ja siivota datasetti")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videon datan valmistelusta tätä oppituntia varten.

On hyvin harvinaista saada datasetti, joka on täysin valmis koneoppimismallin luomiseen sellaisenaan. Tässä oppitunnissa opit, kuinka valmistella raakadataa käyttämällä Pythonin standardikirjastoja. Opit myös erilaisia tekniikoita datan visualisointiin.

## Tapaustutkimus: 'kurpitsamarkkinat'

Tässä kansiossa löydät .csv-tiedoston juurihakemistosta `data`-kansiosta nimeltä [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), joka sisältää 1757 riviä dataa kurpitsamarkkinoista, ryhmiteltynä kaupungeittain. Tämä on raakadataa, joka on peräisin Yhdysvaltain maatalousministeriön [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) -raporteista.

### Datan valmistelu

Tämä data on julkista. Se voidaan ladata useina erillisinä tiedostoina, kaupungeittain, USDA:n verkkosivustolta. Välttääksemme liian monta erillistä tiedostoa, olemme yhdistäneet kaikki kaupunkien datat yhteen taulukkoon, joten dataa on jo _valmisteltu_ hieman. Seuraavaksi tarkastellaan dataa tarkemmin.

### Kurpitsadata - ensimmäiset havainnot

Mitä huomaat tästä datasta? Näet jo, että siinä on sekoitus merkkijonoja, numeroita, tyhjiä kohtia ja outoja arvoja, jotka täytyy ymmärtää.

Mitä kysymystä voisit esittää tästä datasta käyttäen regressiotekniikkaa? Entä "Ennusta kurpitsan hinta myyntikuukauden perusteella". Kun tarkastelet dataa uudelleen, huomaat, että sinun täytyy tehdä joitakin muutoksia luodaksesi tarvittavan datastruktuurin tätä tehtävää varten.

## Harjoitus - analysoi kurpitsadata

Käytetään [Pandas](https://pandas.pydata.org/) -kirjastoa (nimi tulee sanoista `Python Data Analysis`), joka on erittäin hyödyllinen datan muokkaamiseen, kurpitsadatan analysointiin ja valmisteluun.

### Ensiksi, tarkista puuttuvat päivämäärät

Ensimmäinen askel on tarkistaa puuttuvat päivämäärät:

1. Muunna päivämäärät kuukausimuotoon (nämä ovat Yhdysvaltain päivämääriä, joten muoto on `MM/DD/YYYY`).
2. Tallenna kuukausi uuteen sarakkeeseen.

Avaa _notebook.ipynb_-tiedosto Visual Studio Codessa ja tuo taulukko uuteen Pandas-dataframeen.

1. Käytä `head()`-funktiota nähdäksesi ensimmäiset viisi riviä.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Mitä funktiota käyttäisit nähdäksesi viimeiset viisi riviä?

1. Tarkista, onko nykyisessä dataframessa puuttuvaa dataa:

    ```python
    pumpkins.isnull().sum()
    ```

    Dataa puuttuu, mutta ehkä se ei ole merkityksellistä tämän tehtävän kannalta.

1. Jotta dataframe olisi helpompi käsitellä, valitse vain tarvittavat sarakkeet käyttämällä `loc`-funktiota, joka poimii alkuperäisestä dataframesta rivien (ensimmäinen parametri) ja sarakkeiden (toinen parametri) ryhmän. Ilmaisu `:` alla olevassa esimerkissä tarkoittaa "kaikki rivit".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Toiseksi, määritä kurpitsan keskihinta

Mieti, kuinka määrittäisit kurpitsan keskihinnan tiettynä kuukautena. Mitä sarakkeita valitsisit tähän tehtävään? Vinkki: tarvitset kolme saraketta.

Ratkaisu: laske keskiarvo `Low Price`- ja `High Price`-sarakkeista täyttääksesi uuden Price-sarakkeen, ja muunna Date-sarake näyttämään vain kuukauden. Onneksi yllä olevan tarkistuksen mukaan päivämääristä tai hinnoista ei puutu dataa.

1. Laske keskiarvo lisäämällä seuraava koodi:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Voit halutessasi tulostaa mitä tahansa dataa tarkistaaksesi sen käyttämällä `print(month)`.

2. Kopioi nyt muunnettu data uuteen Pandas-dataframeen:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Kun tulostat dataframen, näet siistin ja järjestetyn datasetin, jonka pohjalta voit rakentaa uuden regressiomallin.

### Mutta hetkinen! Tässä on jotain outoa

Jos tarkastelet `Package`-saraketta, kurpitsat myydään monissa eri kokoonpanoissa. Jotkut myydään '1 1/9 bushel' -mittauksina, jotkut '1/2 bushel' -mittauksina, jotkut per kurpitsa, jotkut per pauna, ja jotkut suurissa laatikoissa, joiden leveydet vaihtelevat.

> Kurpitsojen punnitseminen näyttää olevan erittäin haastavaa

Kun tarkastellaan alkuperäistä dataa, on mielenkiintoista, että kaikki, joiden `Unit of Sale` on 'EACH' tai 'PER BIN', sisältävät myös `Package`-tyypin per tuuma, per bin tai 'each'. Kurpitsat näyttävät olevan erittäin vaikeita punnita johdonmukaisesti, joten suodatetaan ne valitsemalla vain kurpitsat, joiden `Package`-sarake sisältää merkkijonon 'bushel'.

1. Lisää suodatin tiedoston alkuun, alkuperäisen .csv-tuonnin alle:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Jos tulostat datan nyt, näet, että saat vain noin 415 riviä dataa, jotka sisältävät kurpitsat bushel-mittauksina.

### Mutta hetkinen! Vielä yksi asia täytyy tehdä

Huomasitko, että bushel-määrä vaihtelee rivikohtaisesti? Sinun täytyy normalisoida hinnoittelu niin, että näytät hinnan per bushel, joten tee hieman laskutoimituksia standardoidaksesi sen.

1. Lisää nämä rivit uuden_pumpkins-dataframen luovan lohkon jälkeen:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) -sivuston mukaan bushelin paino riippuu tuotteen tyypistä, koska se on tilavuusmittaus. "Esimerkiksi tomaattien bushelin painon pitäisi olla 56 paunaa... Lehdet ja vihreät vievät enemmän tilaa vähemmällä painolla, joten pinaatin bushel painaa vain 20 paunaa." Tämä on melko monimutkaista! Emme vaivaudu tekemään bushelin ja paunan välistä muunnosta, vaan hinnoittelemme bushelin mukaan. Kaikki tämä kurpitsabushelien tutkiminen kuitenkin osoittaa, kuinka tärkeää on ymmärtää datan luonne!

Nyt voit analysoida hinnoittelua yksikköä kohden bushel-mittauksen perusteella. Jos tulostat datan vielä kerran, näet, kuinka se on standardoitu.

✅ Huomasitko, että kurpitsat, jotka myydään puolibushelina, ovat erittäin kalliita? Voitko selvittää miksi? Vinkki: pienet kurpitsat ovat paljon kalliimpia kuin isot, luultavasti siksi, että niitä on paljon enemmän per bushel, kun otetaan huomioon yhden suuren ontelon piirakkakurpitsan käyttämätön tila.

## Visualisointistrategiat

Osa datatieteilijän roolia on osoittaa datan laatu ja luonne, jonka parissa hän työskentelee. Tätä varten he usein luovat mielenkiintoisia visualisointeja, kuten kaavioita, graafeja ja diagrammeja, jotka näyttävät datan eri näkökulmia. Näin he voivat visuaalisesti osoittaa suhteita ja aukkoja, jotka muuten olisivat vaikeasti havaittavissa.

[![ML aloittelijoille - Kuinka visualisoida dataa Matplotlibilla](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML aloittelijoille - Kuinka visualisoida dataa Matplotlibilla")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videon datan visualisoinnista tätä oppituntia varten.

Visualisoinnit voivat myös auttaa määrittämään koneoppimistekniikan, joka sopii parhaiten datalle. Esimerkiksi hajontakaavio, joka näyttää seuraavan linjaa, viittaa siihen, että data sopii hyvin lineaariseen regressioharjoitukseen.

Yksi datavisualisointikirjasto, joka toimii hyvin Jupyter-notebookeissa, on [Matplotlib](https://matplotlib.org/) (jota näit myös edellisessä oppitunnissa).

> Saat lisää kokemusta datan visualisoinnista [näissä tutoriaaleissa](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Harjoitus - kokeile Matplotlibia

Yritä luoda joitakin peruskaavioita näyttämään juuri luomasi uusi dataframe. Mitä perusviivakaavio näyttäisi?

1. Tuo Matplotlib tiedoston alkuun Pandas-tuonnin alle:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Suorita koko notebook uudelleen päivittääksesi.
1. Lisää notebookin loppuun solu, joka piirtää datan laatikkona:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Hajontakaavio, joka näyttää hinnan ja kuukauden välisen suhteen](../../../../2-Regression/2-Data/images/scatterplot.png)

    Onko tämä hyödyllinen kaavio? Yllättääkö siinä jokin sinua?

    Se ei ole erityisen hyödyllinen, sillä se vain näyttää datan pisteiden levityksen tiettynä kuukautena.

### Tee siitä hyödyllinen

Jotta kaaviot näyttäisivät hyödyllistä dataa, sinun täytyy yleensä ryhmitellä data jotenkin. Yritetään luoda kaavio, jossa y-akseli näyttää kuukaudet ja data osoittaa datan jakauman.

1. Lisää solu luodaksesi ryhmitellyn pylväskaavion:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Pylväskaavio, joka näyttää hinnan ja kuukauden välisen suhteen](../../../../2-Regression/2-Data/images/barchart.png)

    Tämä on hyödyllisempi datavisualisointi! Se näyttää, että kurpitsojen korkein hinta esiintyy syys- ja lokakuussa. Vastaako tämä odotuksiasi? Miksi tai miksi ei?

---

## 🚀Haaste

Tutki Matplotlibin tarjoamia erilaisia visualisointityyppejä. Mitkä tyypit sopivat parhaiten regressio-ongelmiin?

## [Jälkikysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itseopiskelu

Tutustu datan visualisoinnin eri tapoihin. Tee lista saatavilla olevista kirjastoista ja merkitse, mitkä sopivat parhaiten tiettyihin tehtäviin, esimerkiksi 2D-visualisointeihin vs. 3D-visualisointeihin. Mitä huomaat?

## Tehtävä

[Visualisoinnin tutkiminen](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.