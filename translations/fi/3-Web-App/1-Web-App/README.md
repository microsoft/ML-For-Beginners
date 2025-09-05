<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T00:39:06+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "fi"
}
-->
# Rakenna verkkosovellus ML-mallin käyttöön

Tässä oppitunnissa koulutat ML-mallin datajoukolla, joka on kirjaimellisesti "maailman ulkopuolelta": _UFO-havainnot viimeisen vuosisadan ajalta_, jotka on kerätty NUFORC:n tietokannasta.

Opit:

- Kuinka 'pickle' koulutettu malli
- Kuinka käyttää mallia Flask-sovelluksessa

Jatkamme muistikirjojen käyttöä datan puhdistamiseen ja mallin kouluttamiseen, mutta voit viedä prosessin askeleen pidemmälle tutkimalla mallin käyttöä "luonnossa", eli verkkosovelluksessa.

Tätä varten sinun täytyy rakentaa verkkosovellus Flaskin avulla.

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

## Sovelluksen rakentaminen

On olemassa useita tapoja rakentaa verkkosovelluksia, jotka hyödyntävät koneoppimismalleja. Verkkosovelluksen arkkitehtuuri voi vaikuttaa siihen, miten mallisi koulutetaan. Kuvittele, että työskentelet yrityksessä, jossa data-analytiikkatiimi on kouluttanut mallin, jota he haluavat sinun käyttävän sovelluksessa.

### Huomioitavat asiat

On monia kysymyksiä, joita sinun täytyy esittää:

- **Onko kyseessä verkkosovellus vai mobiilisovellus?** Jos rakennat mobiilisovellusta tai tarvitset mallin IoT-kontekstissa, voit käyttää [TensorFlow Lite](https://www.tensorflow.org/lite/) ja hyödyntää mallia Android- tai iOS-sovelluksessa.
- **Missä malli sijaitsee?** Pilvessä vai paikallisesti?
- **Offline-tuki.** Pitääkö sovelluksen toimia offline-tilassa?
- **Mitä teknologiaa käytettiin mallin kouluttamiseen?** Valittu teknologia voi vaikuttaa tarvittaviin työkaluihin.
    - **TensorFlowin käyttö.** Jos koulutat mallin TensorFlowilla, kyseinen ekosysteemi tarjoaa mahdollisuuden muuntaa TensorFlow-malli verkkosovelluksessa käytettäväksi [TensorFlow.js](https://www.tensorflow.org/js/) avulla.
    - **PyTorchin käyttö.** Jos rakennat mallin kirjastolla, kuten [PyTorch](https://pytorch.org/), voit viedä sen [ONNX](https://onnx.ai/) (Open Neural Network Exchange) -muodossa JavaScript-verkkosovelluksiin, jotka voivat käyttää [Onnx Runtime](https://www.onnxruntime.ai/). Tätä vaihtoehtoa tutkitaan tulevassa oppitunnissa Scikit-learnilla koulutetulle mallille.
    - **Lobe.ai:n tai Azure Custom Visionin käyttö.** Jos käytät ML SaaS (Software as a Service) -järjestelmää, kuten [Lobe.ai](https://lobe.ai/) tai [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott), tämäntyyppinen ohjelmisto tarjoaa tapoja viedä malli monille alustoille, mukaan lukien räätälöidyn API:n rakentaminen, jota verkkosovelluksesi voi kysyä pilvessä.

Sinulla on myös mahdollisuus rakentaa kokonainen Flask-verkkosovellus, joka pystyy kouluttamaan mallin itse verkkoselaimessa. Tämä voidaan tehdä myös TensorFlow.js:n avulla JavaScript-kontekstissa.

Meidän tarkoituksiimme, koska olemme työskennelleet Python-pohjaisten muistikirjojen kanssa, tutkitaan vaiheita, joita tarvitaan koulutetun mallin viemiseksi muistikirjasta Pythonilla rakennetun verkkosovelluksen luettavaksi muodoksi.

## Työkalut

Tätä tehtävää varten tarvitset kaksi työkalua: Flaskin ja Picklen, jotka molemmat toimivat Pythonilla.

✅ Mikä on [Flask](https://palletsprojects.com/p/flask/)? Flaskin luojat määrittelevät sen "mikro-kehykseksi", joka tarjoaa verkkokehysten perusominaisuudet Pythonilla ja mallinnusmoottorin verkkosivujen rakentamiseen. Tutustu [tähän oppimismoduuliin](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) harjoitellaksesi Flaskin käyttöä.

✅ Mikä on [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 on Python-moduuli, joka sarjoittaa ja desarjoittaa Python-objektirakenteita. Kun "picklaat" mallin, sarjoitat tai litistät sen rakenteen verkkokäyttöä varten. Ole varovainen: pickle ei ole luontaisesti turvallinen, joten ole varovainen, jos sinua pyydetään "un-picklaamaan" tiedosto. Picklattu tiedosto päättyy `.pkl`.

## Harjoitus - puhdista datasi

Tässä oppitunnissa käytät dataa 80 000 UFO-havainnosta, jotka on kerätty [NUFORC](https://nuforc.org) (The National UFO Reporting Center) -organisaation toimesta. Tämä data sisältää mielenkiintoisia kuvauksia UFO-havainnoista, esimerkiksi:

- **Pitkä esimerkkikuvaus.** "Mies astuu valonsäteestä, joka loistaa ruohokentälle yöllä, ja juoksee kohti Texas Instrumentsin parkkipaikkaa."
- **Lyhyt esimerkkikuvaus.** "valot ajoivat meitä takaa."

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) -taulukko sisältää sarakkeita, jotka kertovat `kaupungin`, `osavaltion` ja `maan`, jossa havainto tapahtui, objektin `muodon` sekä sen `leveysasteen` ja `pituusasteen`.

Tyhjään [muistikirjaan](../../../../3-Web-App/1-Web-App/notebook.ipynb), joka sisältyy tähän oppituntiin:

1. tuo `pandas`, `matplotlib` ja `numpy` kuten teit aiemmissa oppitunneissa ja tuo UFO-taulukko. Voit tarkastella näyte datajoukkoa:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Muunna UFO-data pieneksi datafreimiksi uusilla otsikoilla. Tarkista `Country`-kentän uniikit arvot.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Nyt voit vähentää käsiteltävän datan määrää pudottamalla pois kaikki tyhjät arvot ja tuomalla vain havainnot, jotka kestivät 1-60 sekuntia:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Tuo Scikit-learnin `LabelEncoder`-kirjasto muuntaaksesi maiden tekstiarvot numeroiksi:

    ✅ LabelEncoder koodaa datan aakkosjärjestyksessä

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Datasi pitäisi näyttää tältä:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Harjoitus - rakenna mallisi

Nyt voit valmistautua kouluttamaan mallin jakamalla datan koulutus- ja testiryhmään.

1. Valitse kolme ominaisuutta, joilla haluat kouluttaa mallisi X-vektoriksi, ja y-vektori on `Country`. Haluat pystyä syöttämään `Seconds`, `Latitude` ja `Longitude` ja saada maa-ID:n palautettavaksi.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Kouluta mallisi logistisella regressiolla:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

Tarkkuus ei ole huono **(noin 95%)**, mikä ei ole yllättävää, koska `Country` ja `Latitude/Longitude` korreloivat.

Luomasi malli ei ole kovin vallankumouksellinen, koska `Country` voidaan päätellä sen `Latitude` ja `Longitude` perusteella, mutta tämä on hyvä harjoitus yrittää kouluttaa raakadataa, jonka puhdistit, viet ja sitten käytät tätä mallia verkkosovelluksessa.

## Harjoitus - 'picklaa' mallisi

Nyt on aika _picklata_ mallisi! Voit tehdä sen muutamalla koodirivillä. Kun malli on _picklattu_, lataa picklattu malli ja testaa sitä näyte datajoukolla, joka sisältää arvot sekunneille, leveysasteelle ja pituusasteelle.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Malli palauttaa **'3'**, joka on Iso-Britannian maa-ID. Uskomatonta! 👽

## Harjoitus - rakenna Flask-sovellus

Nyt voit rakentaa Flask-sovelluksen, joka kutsuu malliasi ja palauttaa samankaltaisia tuloksia, mutta visuaalisesti miellyttävämmällä tavalla.

1. Aloita luomalla kansio nimeltä **web-app** _notebook.ipynb_-tiedoston viereen, jossa _ufo-model.pkl_-tiedosto sijaitsee.

1. Luo kyseiseen kansioon kolme muuta kansiota: **static**, jonka sisällä on kansio **css**, ja **templates**. Sinulla pitäisi nyt olla seuraavat tiedostot ja hakemistot:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Katso ratkaisukansiota nähdäksesi valmiin sovelluksen näkymän

1. Ensimmäinen tiedosto, joka luodaan _web-app_-kansioon, on **requirements.txt**-tiedosto. Kuten _package.json_ JavaScript-sovelluksessa, tämä tiedosto listaa sovelluksen tarvitsemat riippuvuudet. Lisää **requirements.txt**-tiedostoon rivit:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Nyt suorita tämä tiedosto siirtymällä _web-app_-kansioon:

    ```bash
    cd web-app
    ```

1. Kirjoita terminaaliin `pip install`, jotta asennat _requirements.txt_-tiedostossa listatut kirjastot:

    ```bash
    pip install -r requirements.txt
    ```

1. Nyt olet valmis luomaan kolme muuta tiedostoa sovelluksen viimeistelyä varten:

    1. Luo **app.py** juureen.
    2. Luo **index.html** _templates_-hakemistoon.
    3. Luo **styles.css** _static/css_-hakemistoon.

1. Täydennä _styles.css_-tiedosto muutamalla tyylillä:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. Seuraavaksi täydennä _index.html_-tiedosto:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Tarkastele tämän tiedoston mallinnusta. Huomaa 'viiksisyntaksi' muuttujien ympärillä, jotka sovellus tarjoaa, kuten ennusteteksti: `{{}}`. Siellä on myös lomake, joka lähettää ennusteen `/predict`-reitille.

    Lopuksi olet valmis rakentamaan Python-tiedoston, joka ohjaa mallin käyttöä ja ennusteiden näyttämistä:

1. Lisää `app.py`-tiedostoon:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Vinkki: kun lisäät [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) Flask-verkkosovellusta suorittaessasi, kaikki sovellukseen tekemäsi muutokset näkyvät välittömästi ilman, että palvelinta tarvitsee käynnistää uudelleen. Ole varovainen! Älä ota tätä tilaa käyttöön tuotantosovelluksessa.

Jos suoritat `python app.py` tai `python3 app.py` - verkkopalvelimesi käynnistyy paikallisesti, ja voit täyttää lyhyen lomakkeen saadaksesi vastauksen polttavaan kysymykseesi siitä, missä UFO-havaintoja on tehty!

Ennen kuin teet sen, tarkastele `app.py`-tiedoston osia:

1. Ensin riippuvuudet ladataan ja sovellus käynnistyy.
1. Sitten malli tuodaan.
1. Sitten index.html renderöidään kotireitillä.

`/predict`-reitillä tapahtuu useita asioita, kun lomake lähetetään:

1. Lomakkeen muuttujat kerätään ja muunnetaan numpy-taulukoksi. Ne lähetetään mallille, ja ennuste palautetaan.
2. Maa, jonka haluamme näyttää, renderöidään uudelleen luettavana tekstinä ennustetusta maa-ID:stä, ja tämä arvo lähetetään takaisin index.html-tiedostoon, jotta se voidaan renderöidä mallissa.

Mallin käyttö tällä tavalla, Flaskin ja picklatun mallin avulla, on suhteellisen suoraviivaista. Vaikeinta on ymmärtää, millaisessa muodossa datan täytyy olla, jotta se voidaan lähettää mallille ennusteen saamiseksi. Tämä riippuu täysin siitä, miten malli on koulutettu. Tässä mallissa tarvitaan kolme datakohtaa syötettäväksi ennusteen saamiseksi.

Ammatillisessa ympäristössä näet, kuinka hyvä viestintä on välttämätöntä niiden ihmisten välillä, jotka kouluttavat mallin, ja niiden, jotka käyttävät sitä verkkosovelluksessa tai mobiilisovelluksessa. Meidän tapauksessamme kyseessä on vain yksi henkilö, sinä!

---

## 🚀 Haaste

Sen sijaan, että työskentelisit muistikirjassa ja toisit mallin Flask-sovellukseen, voisit kouluttaa mallin suoraan Flask-sovelluksessa! Kokeile muuntaa muistikirjan Python-koodi, ehkä datan puhdistamisen jälkeen, kouluttaaksesi mallin sovelluksen sisällä reitillä nimeltä `train`. Mitkä ovat tämän menetelmän hyvät ja huonot puolet?

## [Jälkikysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

On monia tapoja rakentaa verkkosovellus, joka hyödyntää ML-malleja. Tee lista tavoista, joilla voisit käyttää JavaScriptiä tai Pythonia rakentaaksesi verkkosovelluksen koneoppimisen hyödyntämiseen. Mieti arkkitehtuuria: pitäisikö mallin pysyä sovelluksessa vai sijaita pilvessä? Jos jälkimmäinen, miten pääsisit siihen käsiksi? Piirrä arkkitehtuurimalli sovelletulle ML-verkkoratkaisulle.

## Tehtävä

[Kokeile erilaista mallia](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.