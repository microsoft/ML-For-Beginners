<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-04T23:38:50+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "fi"
}
-->
# Aloita Pythonin ja Scikit-learnin käyttö regressiomallien kanssa

![Yhteenveto regressioista luonnosmuistiinpanossa](../../../../sketchnotes/ml-regression.png)

> Luonnosmuistiinpanon tekijä [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tämä oppitunti on saatavilla myös R-kielellä!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Johdanto

Näissä neljässä oppitunnissa opit rakentamaan regressiomalleja. Keskustelemme pian siitä, mihin niitä käytetään. Mutta ennen kuin aloitat, varmista, että sinulla on oikeat työkalut valmiina prosessin aloittamiseen!

Tässä oppitunnissa opit:

- Konfiguroimaan tietokoneesi paikallisia koneoppimistehtäviä varten.
- Työskentelemään Jupyter-notebookien kanssa.
- Käyttämään Scikit-learnia, mukaan lukien asennus.
- Tutustumaan lineaariseen regressioon käytännön harjoituksen avulla.

## Asennukset ja konfiguroinnit

[![ML aloittelijoille - Valmista työkalusi koneoppimismallien rakentamiseen](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML aloittelijoille - Valmista työkalusi koneoppimismallien rakentamiseen")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyen videon katsomiseksi, jossa käydään läpi tietokoneen konfigurointi ML:ää varten.

1. **Asenna Python**. Varmista, että [Python](https://www.python.org/downloads/) on asennettu tietokoneellesi. Pythonia käytetään monissa data-analytiikan ja koneoppimisen tehtävissä. Useimmissa tietokonejärjestelmissä Python on jo valmiiksi asennettuna. Käytettävissä on myös hyödyllisiä [Python Coding Packeja](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), jotka helpottavat asennusta joillekin käyttäjille.

   Joissakin Pythonin käyttötapauksissa tarvitaan yksi versio ohjelmistosta, kun taas toisissa tarvitaan eri versio. Tämän vuoksi on hyödyllistä työskennellä [virtuaaliympäristössä](https://docs.python.org/3/library/venv.html).

2. **Asenna Visual Studio Code**. Varmista, että Visual Studio Code on asennettu tietokoneellesi. Seuraa näitä ohjeita [Visual Studio Coden asentamiseksi](https://code.visualstudio.com/) perusasennusta varten. Tässä kurssissa käytät Pythonia Visual Studio Codessa, joten kannattaa tutustua siihen, miten [Visual Studio Code konfiguroidaan](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) Python-kehitystä varten.

   > Tutustu Pythonin käyttöön käymällä läpi tämä kokoelma [Learn-moduuleja](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Pythonin konfigurointi Visual Studio Codessa](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Pythonin konfigurointi Visual Studio Codessa")
   >
   > 🎥 Klikkaa yllä olevaa kuvaa videon katsomiseksi: Pythonin käyttö VS Codessa.

3. **Asenna Scikit-learn** seuraamalla [näitä ohjeita](https://scikit-learn.org/stable/install.html). Koska sinun täytyy varmistaa, että käytät Python 3:a, on suositeltavaa käyttää virtuaaliympäristöä. Huomaa, että jos asennat tämän kirjaston M1 Macille, sivulla on erityisohjeita.

4. **Asenna Jupyter Notebook**. Sinun täytyy [asentaa Jupyter-paketti](https://pypi.org/project/jupyter/).

## ML-kehitysympäristösi

Käytät **notebookeja** Python-koodin kehittämiseen ja koneoppimismallien luomiseen. Tällainen tiedostotyyppi on yleinen työkalu data-analyytikoille, ja ne tunnistaa niiden päätteestä `.ipynb`.

Notebookit ovat interaktiivinen ympäristö, joka mahdollistaa sekä koodauksen että dokumentaation lisäämisen koodin ympärille, mikä on erittäin hyödyllistä kokeellisiin tai tutkimusprojekteihin.

[![ML aloittelijoille - Jupyter Notebookien asennus regressiomallien rakentamisen aloittamiseksi](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML aloittelijoille - Jupyter Notebookien asennus regressiomallien rakentamisen aloittamiseksi")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyen videon katsomiseksi, jossa käydään läpi tämä harjoitus.

### Harjoitus - työskentele notebookin kanssa

Tässä kansiossa löydät tiedoston _notebook.ipynb_.

1. Avaa _notebook.ipynb_ Visual Studio Codessa.

   Jupyter-palvelin käynnistyy Python 3+:lla. Löydät notebookista alueita, jotka voidaan `ajaa`, eli koodinpätkiä. Voit ajaa koodilohkon valitsemalla kuvakkeen, joka näyttää toistopainikkeelta.

2. Valitse `md`-kuvake ja lisää hieman markdownia sekä seuraava teksti **# Tervetuloa notebookiisi**.

   Lisää seuraavaksi Python-koodia.

3. Kirjoita **print('hello notebook')** koodilohkoon.
4. Valitse nuoli ajaaksesi koodin.

   Näet tulostetun lauseen:

    ```output
    hello notebook
    ```

![VS Code avoinna notebookin kanssa](../../../../2-Regression/1-Tools/images/notebook.jpg)

Voit yhdistää koodisi kommentteihin dokumentoidaksesi notebookin itse.

✅ Mieti hetki, kuinka erilainen web-kehittäjän työympäristö on verrattuna data-analyytikon työympäristöön.

## Scikit-learnin käyttöönotto

Nyt kun Python on asennettu paikalliseen ympäristöösi ja olet mukautunut Jupyter-notebookeihin, tutustutaan yhtä mukavasti Scikit-learniin (lausutaan `sci` kuten `science`). Scikit-learn tarjoaa [laajan API:n](https://scikit-learn.org/stable/modules/classes.html#api-ref), joka auttaa sinua suorittamaan ML-tehtäviä.

Heidän [verkkosivustonsa](https://scikit-learn.org/stable/getting_started.html) mukaan "Scikit-learn on avoimen lähdekoodin koneoppimiskirjasto, joka tukee ohjattua ja ohjaamatonta oppimista. Se tarjoaa myös erilaisia työkaluja mallien sovittamiseen, datan esikäsittelyyn, mallien valintaan ja arviointiin sekä moniin muihin hyödyllisiin toimintoihin."

Tässä kurssissa käytät Scikit-learnia ja muita työkaluja koneoppimismallien rakentamiseen suorittaaksesi niin sanottuja 'perinteisiä koneoppimistehtäviä'. Olemme tarkoituksella välttäneet neuroverkkoja ja syväoppimista, sillä ne käsitellään paremmin tulevassa 'AI aloittelijoille' -opetussuunnitelmassamme.

Scikit-learn tekee mallien rakentamisesta ja niiden arvioinnista helppoa. Se keskittyy pääasiassa numeerisen datan käyttöön ja sisältää useita valmiita datasettiä oppimistyökaluiksi. Se sisältää myös valmiiksi rakennettuja malleja, joita opiskelijat voivat kokeilla. Tutustutaan prosessiin, jossa ladataan valmiiksi pakattua dataa ja käytetään sisäänrakennettua estimaattoria ensimmäisen ML-mallin luomiseen Scikit-learnilla perusdatan avulla.

## Harjoitus - ensimmäinen Scikit-learn notebookisi

> Tämä opetus on saanut inspiraationsa [lineaarisen regression esimerkistä](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) Scikit-learnin verkkosivustolla.

[![ML aloittelijoille - Ensimmäinen lineaarisen regression projekti Pythonilla](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML aloittelijoille - Ensimmäinen lineaarisen regression projekti Pythonilla")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyen videon katsomiseksi, jossa käydään läpi tämä harjoitus.

Poista _notebook.ipynb_-tiedostosta kaikki solut painamalla 'roskakori'-kuvaketta.

Tässä osiossa työskentelet pienen datasetin kanssa, joka liittyy diabetekseen ja joka on sisäänrakennettu Scikit-learniin oppimistarkoituksia varten. Kuvittele, että haluaisit testata hoitoa diabeetikoille. Koneoppimismallit voivat auttaa sinua määrittämään, mitkä potilaat reagoisivat hoitoon paremmin, perustuen muuttujien yhdistelmiin. Jopa hyvin yksinkertainen regressiomalli, kun se visualisoidaan, voi paljastaa tietoa muuttujista, jotka auttaisivat sinua järjestämään teoreettisia kliinisiä kokeita.

✅ Regressiomenetelmiä on monenlaisia, ja valinta riippuu siitä, mitä haluat selvittää. Jos haluat ennustaa todennäköistä pituutta tietyn ikäiselle henkilölle, käyttäisit lineaarista regressiota, koska etsit **numeerista arvoa**. Jos haluat selvittää, pitäisikö tiettyä ruokakulttuuria pitää vegaanisena vai ei, etsit **kategoriaa**, joten käyttäisit logistista regressiota. Opit lisää logistisesta regressiosta myöhemmin. Mieti hetki, mitä kysymyksiä voisit esittää datasta ja mikä näistä menetelmistä olisi sopivampi.

Aloitetaan tehtävä.

### Kirjastojen tuonti

Tätä tehtävää varten tuomme joitakin kirjastoja:

- **matplotlib**. Se on hyödyllinen [graafinen työkalu](https://matplotlib.org/), ja käytämme sitä viivakaavion luomiseen.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) on hyödyllinen kirjasto numeerisen datan käsittelyyn Pythonissa.
- **sklearn**. Tämä on [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) -kirjasto.

Tuo joitakin kirjastoja auttamaan tehtävissäsi.

1. Lisää tuonnit kirjoittamalla seuraava koodi:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Yllä tuodaan `matplotlib`, `numpy` sekä `datasets`, `linear_model` ja `model_selection` `sklearn`-kirjastosta. `model_selection` käytetään datan jakamiseen harjoitus- ja testijoukkoihin.

### Diabetes-datasetti

Sisäänrakennettu [diabetes-datasetti](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) sisältää 442 näytettä diabetekseen liittyvästä datasta, jossa on 10 ominaisuusmuuttujaa, joista osa sisältää:

- age: ikä vuosina
- bmi: kehon massan indeksi
- bp: keskimääräinen verenpaine
- s1 tc: T-solut (eräänlainen valkosolu)

✅ Tämä datasetti sisältää 'sukupuolen' käsitteen tärkeänä ominaisuusmuuttujana diabetekseen liittyvässä tutkimuksessa. Monet lääketieteelliset datasetit sisältävät tällaisen binääriluokituksen. Mieti hetki, miten tällaiset luokitukset saattavat sulkea pois tiettyjä väestönosia hoidoista.

Lataa nyt X- ja y-data.

> 🎓 Muista, että tämä on ohjattua oppimista, ja tarvitsemme nimetyn 'y'-kohteen.

Uudessa koodisolussa lataa diabetes-datasetti kutsumalla `load_diabetes()`. Syöte `return_X_y=True` ilmoittaa, että `X` on datamatriisi ja `y` on regressiotavoite.

1. Lisää joitakin tulostuskäskyjä näyttämään datamatriisin muoto ja sen ensimmäinen elementti:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Saat vastauksena tuplen. Teet niin, että määrität tuplen kaksi ensimmäistä arvoa `X`:lle ja `y`:lle. Lue lisää [tuplista](https://wikipedia.org/wiki/Tuple).

    Näet, että tämä data sisältää 442 kohdetta, jotka on muotoiltu 10 elementin taulukoiksi:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Mieti hetki datan ja regressiotavoitteen välistä suhdetta. Lineaarinen regressio ennustaa suhteita ominaisuuden X ja tavoitemuuttujan y välillä. Voitko löytää [tavoitteen](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) diabetes-datasetille dokumentaatiosta? Mitä tämä datasetti havainnollistaa, kun otetaan huomioon tavoite?

2. Valitse seuraavaksi osa tästä datasetistä piirtämistä varten valitsemalla datasetin 3. sarake. Voit tehdä tämän käyttämällä `:`-operaattoria valitaksesi kaikki rivit ja sitten valitsemalla 3. sarakkeen indeksillä (2). Voit myös muotoilla datan 2D-taulukoksi - kuten vaaditaan piirtämistä varten - käyttämällä `reshape(n_rows, n_columns)`. Jos yksi parametreista on -1, vastaava ulottuvuus lasketaan automaattisesti.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Tulosta data milloin tahansa tarkistaaksesi sen muodon.

3. Nyt kun sinulla on data valmiina piirtämistä varten, voit nähdä, voiko kone auttaa määrittämään loogisen jaon numeroiden välillä tässä datasetissä. Tätä varten sinun täytyy jakaa sekä data (X) että tavoite (y) testaus- ja harjoitusjoukkoihin. Scikit-learn tarjoaa yksinkertaisen tavan tehdä tämä; voit jakaa testidatasi tietyssä pisteessä.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nyt olet valmis kouluttamaan mallisi! Lataa lineaarinen regressiomalli ja kouluta sitä X- ja y-harjoitusjoukoilla käyttämällä `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` on funktio, jonka näet monissa ML-kirjastoissa, kuten TensorFlowssa.

5. Luo sitten ennuste testidatan avulla käyttämällä funktiota `predict()`. Tätä käytetään piirtämään viiva dataryhmien välille.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nyt on aika näyttää data kaaviossa. Matplotlib on erittäin hyödyllinen työkalu tähän tehtävään. Luo scatterplot kaikesta X- ja y-testidatasta ja käytä ennustetta piirtääksesi viiva sopivimpaan kohtaan dataryhmien välillä.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![scatterplot, joka näyttää datapisteet diabetekseen liittyen](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Mieti hetki, mitä tässä tapahtuu. Suora viiva kulkee monien pienten datapisteiden läpi, mutta mitä se oikeastaan tekee? Voitko nähdä, miten tämän viivan avulla pitäisi pystyä ennustamaan, mihin uusi, ennennäkemätön datapiste sijoittuu suhteessa kuvaajan y-akseliin? Yritä pukea sanoiksi tämän mallin käytännön hyöty.

Onnittelut, loit ensimmäisen lineaarisen regressiomallisi, teit ennusteen sen avulla ja esittelit sen kuvaajassa!

---
## 🚀Haaste

Piirrä kuvaaja, jossa käytetään eri muuttujaa tästä datasetistä. Vinkki: muokkaa tätä riviä: `X = X[:,2]`. Tämän datasetin tavoitteen perusteella, mitä pystyt päättelemään diabeteksen etenemisestä sairautena?

## [Luennon jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itseopiskelu

Tässä opetusmateriaalissa työskentelit yksinkertaisen lineaarisen regression parissa, etkä univariaatin tai monimuuttujaisen regression kanssa. Lue hieman näiden menetelmien eroista tai katso [tämä video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Lue lisää regressiokonseptista ja pohdi, millaisiin kysymyksiin tällä tekniikalla voidaan vastata. Syvennä ymmärrystäsi ottamalla [tämä opetusohjelma](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott).

## Tehtävä

[Eri datasetti](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.