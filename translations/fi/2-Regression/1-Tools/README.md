# Aloita Pythonilla ja Scikit-learnilla regressiomallien kanssa

![Yhteenveto regressioista sketchnotessa](../../../../translated_images/fi/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote tekijältä [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Ennakkokysely ennen luentoa](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tämä oppitunti on saatavilla myös R:llä!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Johdanto

Näissä neljässä oppitunnissa tutustut regressiomallien rakentamiseen. Keskustelemme pian, mihin niitä käytetään. Mutta ennen kuin aloitat, varmista, että sinulla on oikeat työkalut prosessin aloittamista varten!

Tässä oppitunnissa opit:

- Konfiguroimaan tietokoneesi paikallisia koneoppimistehtäviä varten.
- Työskentelemään Jupyter Notebookien kanssa.
- Käyttämään Scikit-learnia, mukaan lukien asennuksen.
- Tutustumaan lineaariseen regressioon käytännön harjoituksen avulla.

## Asennukset ja asetukset

[![Koneoppiminen aloittelijoille - Valmistele työkalut koneoppimismallien rakentamiseen](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "Koneoppiminen aloittelijoille - Valmistele työkalut koneoppimismallien rakentamiseen")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videon tietokoneesi konfiguroimisesta koneoppimiseen.

1. **Asenna Python**. Varmista, että [Python](https://www.python.org/downloads/) on asennettu tietokoneellesi. Käytät Pythonia monissa datatieteen ja koneoppimisen tehtävissä. Useimmissa tietokonejärjestelmissä on Python-asennus valmiina. Saatavilla on myös hyödyllisiä [Python-koodipaketteja](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), jotka helpottavat asennusta joillekin käyttäjille.

   Joissain tapauksissa Pythonin eri versioita tarvitaan eri tehtäviin, joten on hyödyllistä työskennellä [virtuaaliympäristössä](https://docs.python.org/3/library/venv.html).

2. **Asenna Visual Studio Code**. Varmista, että sinulla on Visual Studio Code asennettuna. Noudata näitä ohjeita [Visual Studio Coden asentamiseen](https://code.visualstudio.com/) perustason asennukseen. Käytät Pythonia tässä kurssissa Visual Studio Codessa, joten saatat haluta kerrata, kuinka [konfiguroida Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) Python-kehitykseen.

   > Harjoittele Pythonin käyttöä tämän [Learn-moduulikokoelman](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) avulla
   >
   > [![Pythonin asentaminen Visual Studio Codeen](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Pythonin asentaminen Visual Studio Codeen")
   >
   > 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi videon Pythonin käytöstä VS Codessa.

3. **Asenna Scikit-learn** seuraamalla [näitä ohjeita](https://scikit-learn.org/stable/install.html). Koska tarvitset Python 3:n, on suositeltavaa käyttää virtuaaliympäristöä. Huomaa, että jos asennat kirjastoa M1 Macille, yllä olevan linkin sivulla on erikoisohjeita.

1. **Asenna Jupyter Notebook**. Sinun tulee [asenna Jupyter-paketti](https://pypi.org/project/jupyter/).

## Koneoppimisen kirjoitusalustasi

Käytät **notebookeja** kirjoittaaksesi Python-koodia ja luodaksesi koneoppimismalleja. Tämän tyyppinen tiedosto on yleinen työkalu datatieteilijöille, ja sen tunnistaa tiedostopäätteestä `.ipynb`.

Notebookit ovat interaktiivinen ympäristö, jossa kehittäjä voi sekä koodata että lisätä muistiinpanoja ja kirjoittaa dokumentaatiota koodin ympärille, mikä on erittäin hyödyllistä kokeellisissa tai tutkimuspainotteisissa projekteissa.

[![Koneoppiminen aloittelijoille - Jupyter Notebookien perustaminen regressiomallien aloittamiseksi](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "Koneoppiminen aloittelijoille - Jupyter Notebookien perustaminen regressiomallien aloittamiseksi")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videon tämän harjoituksen tekemisestä.

### Harjoitus – työskentely notebookin kanssa

Tässä kansiossa löydät tiedoston _notebook.ipynb_.

1. Avaa _notebook.ipynb_ Visual Studio Codessa.

   Jupyter-palvelin käynnistyy Python 3+:lla. Löydät notebookista alueita, joita voi `ajaa`, eli koodinpätkiä. Voit ajaa koodilohkon valitsemalla toistopainikkeen näköisen kuvakkeen.

1. Valitse `md`-kuvake ja lisää hieman markdownia sekä seuraava teksti **# Tervetuloa notebookiisi**.

   Lisää seuraavaksi hieman Python-koodia.

1. Kirjoita koodilohkoon **print('hello notebook')**.
1. Paina nuolta ajaaksesi koodin.

   Näet tulostetun lauseen:

    ```output
    hello notebook
    ```

![VS Code avattuna notebookin kanssa](../../../../translated_images/fi/notebook.4a3ee31f396b8832.webp)

Voit yhdistellä koodiasi kommenttien kanssa dokumentoidaksesi notebookisi itse.

✅ Mieti hetki, kuinka erilainen web-kehittäjän työympäristö on verrattuna datatieteilijän työympäristöön.

## Käyttövalmis Scikit-learnin kanssa

Nyt kun Python on asetettu paikalliseen ympäristöösi ja olet tottunut Jupyter Notebookeihin, tutustutaan yhtä mukavasti Scikit-learniin (lausutaan `sci` kuten `science`). Scikit-learn tarjoaa [laajan API:n](https://scikit-learn.org/stable/modules/classes.html#api-ref) ML-tehtävien suorittamiseen.

Heidän [verkkosivustonsa](https://scikit-learn.org/stable/getting_started.html) mukaan "Scikit-learn on avoimen lähdekoodin koneoppimiskirjasto, joka tukee valvottua ja valvomatonta oppimista. Se tarjoaa myös erilaisia työkaluja mallien sovittamiseen, datan esikäsittelyyn, mallin valintaan ja arviointiin sekä monia muita hyödyllisiä toimintoja."

Tässä kurssissa käytät Scikit-learnia ja muita työkaluja koneoppimismallien rakentamiseen suorittamaan niin sanottuja perinteisiä koneoppimistehtäviä. Olemme tarkoituksella jättäneet neuroverkot ja syväoppimisen pois, sillä ne käsitellään paremmin tulevassa 'AI for Beginners' -opetussuunnitelmassamme.

Scikit-learnilla mallien rakentaminen ja arviointi on vaivatonta. Se keskittyy pääasiassa numeerisen datan käyttöön ja sisältää valmiita dataset-paketteja oppimistyökaluksi. Mukana on myös valmiita malleja opiskelijoiden kokeiltavaksi. Tutustutaan prosessiin, jossa ladataan valmiiksi pakattua dataa ja käytetään sisäänrakennettua estimator-mallia luomaan ensimmäinen koneoppimismalli Scikit-learnilla perusdatan avulla.

## Harjoitus – ensimmäinen Scikit-learn notebookisi

> Tämä opastus perustuu [lineaarisen regression esimerkkiin](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) Scikit-learnin sivustolla.


[![Koneoppiminen aloittelijoille - Ensimmäinen lineaarisen regression projektisi Pythonilla](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "Koneoppiminen aloittelijoille - Ensimmäinen lineaarisen regression projektisi Pythonilla")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videon tämän harjoituksen suorittamisesta.

_notebook.ipynb_-tiedostossa, joka liittyy tähän oppituntiin, tyhjennä kaikki solut painamalla roskakorikuvaketta.

Tässä osiossa työskentelet pienen diabetestadatasetin kanssa, joka on rakennettu oppimistarkoitukseen Scikit-learniin. Kuvittele, että haluat testata hoitoa diabeetikoille. Koneoppimismallit voivat auttaa sinua määrittämään, mitkä potilaat reagoisivat hoitoon paremmin muuttujien yhdistelmien perusteella. Jopa hyvin yksinkertainen regressiomalli, kun se visualisoidaan, voi näyttää tietoa muuttujista, jotka auttaisivat järjestämään teoreettisia kliinisiä kokeita.

✅ Regressiomenetelmiä on monenlaisia, ja valinta riippuu siitä, mitä haluat ennustaa. Jos haluat ennustaa henkilön todennäköisen pituuden tietyn iän perusteella, käytät lineaarista regressiota, koska etsit **numeerista arvoa**. Jos taas haluat selvittää, onko ruokavalio vegaaninen vai ei, etsinnässä on **luokkajako**, jolloin käytät logistista regressiota. Opit lisää logistisesta regressiosta myöhemmin. Mieti hetki, millaisia kysymyksiä voit esittää datalle, ja kumpi näistä menetelmistä olisi sopivampi.

Aloitetaan tehtävästä.

### Kirjastojen tuonti

Tätä tehtävää varten tuomme muutamia kirjastoja:

- **matplotlib**. Se on hyödyllinen [kuvaustyökalu](https://matplotlib.org/), ja käytämme sitä viivadiagrammin luomiseen.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) on hyödyllinen kirjasto numeerisen datan käsittelyyn Pythonilla.
- **sklearn**. Tämä on [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) kirjasto.

Tuo joitakin kirjastoja tehtäviesi tueksi.

1. Lisää importit kirjoittamalla seuraava koodi:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Yllä tuot `matplotlibin`, `numpyn` sekä tuot `datasets`, `linear_model` ja `model_selection` `sklearnista`. `model_selection` käytetään datan jakamiseen harjoitus- ja testijoukkoihin.

### Diabetestietoja sisältävä dataset

Sisäänrakennettu [diabetes-dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) sisältää 442 näytettä diabetesta koskevista tiedoista 10 ominaisuusmuuttujalla, joista osa on:

- age: ikä vuosina
- bmi: painoindeksi
- bp: keskimääräinen verenpaine
- s1 tc: T-soluja (eräs valkosolu)

✅ Tämä dataset sisältää sukupuolen 'sex' ominaisuutena, joka on tärkeä muuttuja diabetes-tutkimuksessa. Monissa lääketieteellisissä datasetissä on tällainen binääriluokitus. Mieti hetki, miten tällaiset luokittelut voivat sulkea osan väestöstä hoitojen ulkopuolelle.

Lataa nyt X- ja y-data.

> 🎓 Muista, että kyseessä on valvottu oppiminen, ja siksi tarvitsemme nimetyn 'y'-kohteen.

Uudessa koodisolussa lataa diabetes-dataset kutsumalla `load_diabetes()`. Parametri `return_X_y=True` kertoo, että `X` on datamatriisi ja `y` regressiotavoite.

1. Lisää muutama print-komento näyttämään datamatriisin muoto ja ensimmäinen alkio:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Palautettavana on tuple. Teet sen, että kaksi ensimmäistä tuple-arvoa annetaan `X` ja `y`:lle vastaavasti. Lue lisää [tuplista](https://wikipedia.org/wiki/Tuple).

    Näet, että tässä datassa on 442 kohdetta, jotka on muotoiltu kymmenen elementin taulukoiksi:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Mieti hetki, mikä suhde datalla ja regressiotavoitteella on. Lineaarinen regressio ennustaa suhdetta ominaisuus X:n ja tavoitemuuttujan y:n välillä. Löydätkö [tavoitteen](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) diabetes-datasetin dokumentaatiosta? Mitä tämä dataset havainnollistaa tavoitteen perusteella?

2. Valitse datasetista osa piirtämistä varten valitsemalla datasetin 3. sarake. Voit tehdä sen käyttämällä `:` operaattoria valitaksesi kaikki rivit, ja sitten sarakkeen valintaa indeksillä (2). Voit myös muotoilla datan 2D-taulukoksi (vaaditaan piirtämiseen) käyttämällä `reshape(n_rows, n_columns)`. Jos parametri on -1, kyseinen ulottuvuus lasketaan automaattisesti.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Tulosta data milloin tahansa tarkistaaksesi sen muodon.

3. Nyt kun sinulla on data valmiina piirtämistä varten, voit kokeilla, voisiko kone auttaa erottamaan järkevän jakokohdan datasetin arvoille. Tätä varten sinun täytyy jakaa data (X) ja tavoite (y) testaus- ja koulutusjoukkoihin. Scikit-learnissa tämä onnistuu helposti; voit jakaa testidatan tietystä pisteestä.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nyt olet valmis kouluttamaan mallin! Lataa lineaarisen regression malli ja kouluta se X- ja y-harjoitusjoukoilla käyttäen `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` on funktio, jota näet monissa ML-kirjastoissa, kuten TensorFlow.

5. Tee sitten ennuste testidatalla käyttämällä `predict()`-funktiota. Tätä käytetään piirtämään viiva dataryhmien välille.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nyt on aika näyttää data kaaviossa. Matplotlib on tätä varten erittäin käyttökelpoinen työkalu. Luo hajontakaavio kaikista X:n ja y:n testidatoista, ja käytä ennustetta piirtämään viiva mallin datajoukkojen väliin sopivimpaan kohtaan.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![hajontakaavio diabetesa koskevista datapisteistä](../../../../translated_images/fi/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Mieti hetki, mitä tässä tapahtuu. Suora viiva kulkee monien pienten pisteiden läpi, mutta mitä se tarkalleen tekee? Näetkö, miten tällä viivalla voi ennustaa, mihin uusi, näkemätön datapiste sijoittuu kuvaajan y-akselin suhteen? Yritä pukea käytännön merkitys tälle mallille sanoiksi.

Onnittelut, olet rakentanut ensimmäisen lineaarisen regressiomallisi, luonut sillä ennusteen ja näyttänyt sen kaaviossa!

---
## 🚀Haaste

Piirrä jokin toinen muuttuja tästä datasetista. Vihje: muokkaa tätä riviä: `X = X[:,2]`. Mitä tämän datasetin tavoite kertoo sinulle diabeteksen etenemisestä sairautena?
## [Jälkikysely luennon jälkeen](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itsenäinen opiskelu

Tässä opastuksessa työskentelit yksinkertaisen lineaarisen regression kanssa, etkä yksimuuttujaisen tai monimuuttujaisen regressioon kanssa. Lue hieman näiden menetelmien eroista tai katso [tämä video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Lue lisää regressiokäsitteestä ja pohdi, millaisiin kysymyksiin tällä tekniikalla voidaan vastata. Ota tämä [opas](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) syventääksesi ymmärrystäsi.

## Tehtävä

[Eri aineisto](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, otathan huomioon, että automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä on virallinen lähde. Tärkeissä asioissa suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä aiheutuvista väärinymmärryksistä tai tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->