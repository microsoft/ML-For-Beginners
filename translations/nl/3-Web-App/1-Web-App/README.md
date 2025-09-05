<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T19:44:37+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "nl"
}
-->
# Bouw een webapp om een ML-model te gebruiken

In deze les ga je een ML-model trainen op een dataset die letterlijk buitenaards is: _UFO-waarnemingen van de afgelopen eeuw_, afkomstig uit de NUFORC-database.

Je leert:

- Hoe je een getraind model kunt 'pickle'
- Hoe je dat model kunt gebruiken in een Flask-app

We blijven notebooks gebruiken om data schoon te maken en ons model te trainen, maar je kunt het proces een stap verder brengen door te verkennen hoe je een model 'in het wild' kunt gebruiken, bijvoorbeeld in een webapp.

Om dit te doen, moet je een webapp bouwen met Flask.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Een app bouwen

Er zijn verschillende manieren om webapps te bouwen die machine learning-modellen gebruiken. Je webarchitectuur kan invloed hebben op de manier waarop je model wordt getraind. Stel je voor dat je werkt in een bedrijf waar de data science-afdeling een model heeft getraind dat ze willen dat jij gebruikt in een app.

### Overwegingen

Er zijn veel vragen die je moet stellen:

- **Is het een webapp of een mobiele app?** Als je een mobiele app bouwt of het model in een IoT-context wilt gebruiken, kun je [TensorFlow Lite](https://www.tensorflow.org/lite/) gebruiken en het model toepassen in een Android- of iOS-app.
- **Waar zal het model zich bevinden?** In de cloud of lokaal?
- **Offline ondersteuning.** Moet de app offline werken?
- **Welke technologie is gebruikt om het model te trainen?** De gekozen technologie kan invloed hebben op de tools die je moet gebruiken.
    - **Gebruik van TensorFlow.** Als je een model traint met TensorFlow, biedt dat ecosysteem de mogelijkheid om een TensorFlow-model te converteren voor gebruik in een webapp met [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Gebruik van PyTorch.** Als je een model bouwt met een bibliotheek zoals [PyTorch](https://pytorch.org/), kun je het exporteren in [ONNX](https://onnx.ai/) (Open Neural Network Exchange)-formaat voor gebruik in JavaScript-webapps die de [Onnx Runtime](https://www.onnxruntime.ai/) kunnen gebruiken. Deze optie wordt in een toekomstige les verkend voor een Scikit-learn-getraind model.
    - **Gebruik van Lobe.ai of Azure Custom Vision.** Als je een ML SaaS (Software as a Service)-systeem zoals [Lobe.ai](https://lobe.ai/) of [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) gebruikt om een model te trainen, biedt dit type software manieren om het model te exporteren voor veel platforms, inclusief het bouwen van een op maat gemaakte API die in de cloud kan worden geraadpleegd door je online applicatie.

Je hebt ook de mogelijkheid om een volledige Flask-webapp te bouwen die het model zelf kan trainen in een webbrowser. Dit kan ook worden gedaan met TensorFlow.js in een JavaScript-context.

Voor onze doeleinden, aangezien we hebben gewerkt met Python-gebaseerde notebooks, laten we de stappen verkennen die je moet nemen om een getraind model uit zo'n notebook te exporteren naar een formaat dat leesbaar is door een Python-gebouwde webapp.

## Tool

Voor deze taak heb je twee tools nodig: Flask en Pickle, beide draaien op Python.

✅ Wat is [Flask](https://palletsprojects.com/p/flask/)? Door de makers gedefinieerd als een 'micro-framework', biedt Flask de basisfuncties van webframeworks met Python en een template-engine om webpagina's te bouwen. Bekijk [deze Learn-module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) om te oefenen met het bouwen met Flask.

✅ Wat is [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 is een Python-module die een Python-objectstructuur serialiseert en deserialiseert. Wanneer je een model 'pickled', serialiseer of 'plat' je de structuur voor gebruik op het web. Let op: Pickle is niet intrinsiek veilig, dus wees voorzichtig als je wordt gevraagd een bestand te 'un-picklen'. Een pickled bestand heeft de extensie `.pkl`.

## Oefening - maak je data schoon

In deze les gebruik je data van 80.000 UFO-waarnemingen, verzameld door [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Deze data bevat enkele interessante beschrijvingen van UFO-waarnemingen, bijvoorbeeld:

- **Lange voorbeeldbeschrijving.** "Een man komt tevoorschijn uit een lichtstraal die 's nachts op een grasveld schijnt en rent naar de parkeerplaats van Texas Instruments".
- **Korte voorbeeldbeschrijving.** "de lichten achtervolgden ons".

De [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) spreadsheet bevat kolommen over de `stad`, `staat` en `land` waar de waarneming plaatsvond, de `vorm` van het object en de `latitude` en `longitude`.

In het lege [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) dat bij deze les is inbegrepen:

1. Importeer `pandas`, `matplotlib` en `numpy` zoals je in eerdere lessen deed en importeer de ufos-spreadsheet. Je kunt een voorbeelddataset bekijken:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Converteer de ufos-data naar een kleine dataframe met nieuwe titels. Controleer de unieke waarden in het `Land`-veld.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Nu kun je de hoeveelheid data die we moeten verwerken verminderen door null-waarden te verwijderen en alleen waarnemingen tussen 1-60 seconden te importeren:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importeer Scikit-learn's `LabelEncoder`-bibliotheek om de tekstwaarden voor landen om te zetten naar een nummer:

    ✅ LabelEncoder codeert data alfabetisch

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Je data zou er nu zo uit moeten zien:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Oefening - bouw je model

Nu kun je je voorbereiden om een model te trainen door de data te verdelen in een trainings- en testgroep.

1. Selecteer de drie kenmerken waarop je wilt trainen als je X-vector, en de y-vector zal het `Land` zijn. Je wilt in staat zijn om `Seconden`, `Latitude` en `Longitude` in te voeren en een land-id te krijgen als resultaat.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Train je model met behulp van logistieke regressie:

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

De nauwkeurigheid is niet slecht **(ongeveer 95%)**, wat niet verrassend is, aangezien `Land` en `Latitude/Longitude` correleren.

Het model dat je hebt gemaakt is niet erg revolutionair, omdat je een `Land` zou moeten kunnen afleiden uit de `Latitude` en `Longitude`, maar het is een goede oefening om te proberen te trainen met ruwe data die je hebt schoongemaakt, geëxporteerd en vervolgens dit model te gebruiken in een webapp.

## Oefening - 'pickle' je model

Nu is het tijd om je model te _picklen_! Je kunt dit doen in een paar regels code. Zodra het is _gepickled_, laad je je gepickled model en test je het tegen een voorbeelddata-array met waarden voor seconden, latitude en longitude.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Het model retourneert **'3'**, wat de landcode is voor het VK. Wauw! 👽

## Oefening - bouw een Flask-app

Nu kun je een Flask-app bouwen om je model aan te roepen en vergelijkbare resultaten te retourneren, maar op een visueel aantrekkelijkere manier.

1. Begin met het maken van een map genaamd **web-app** naast het _notebook.ipynb_-bestand waar je _ufo-model.pkl_-bestand zich bevindt.

1. Maak in die map drie extra mappen: **static**, met een map **css** erin, en **templates**. Je zou nu de volgende bestanden en mappen moeten hebben:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Raadpleeg de oplossingmap voor een weergave van de voltooide app

1. Het eerste bestand dat je moet maken in de _web-app_-map is het **requirements.txt**-bestand. Net zoals _package.json_ in een JavaScript-app, bevat dit bestand de afhankelijkheden die nodig zijn voor de app. Voeg in **requirements.txt** de regels toe:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Voer nu dit bestand uit door naar _web-app_ te navigeren:

    ```bash
    cd web-app
    ```

1. Typ in je terminal `pip install` om de bibliotheken te installeren die in _requirements.txt_ worden vermeld:

    ```bash
    pip install -r requirements.txt
    ```

1. Nu ben je klaar om drie extra bestanden te maken om de app af te maken:

    1. Maak **app.py** in de root.
    2. Maak **index.html** in de _templates_-map.
    3. Maak **styles.css** in de _static/css_-map.

1. Bouw het _styles.css_-bestand uit met een paar stijlen:

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

1. Bouw vervolgens het _index.html_-bestand uit:

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

    Bekijk de templating in dit bestand. Let op de 'mustache'-syntax rond variabelen die door de app worden geleverd, zoals de voorspellingstekst: `{{}}`. Er is ook een formulier dat een voorspelling post naar de `/predict`-route.

    Ten slotte ben je klaar om het Python-bestand te bouwen dat het model gebruikt en de voorspellingen weergeeft:

1. Voeg in `app.py` toe:

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

    > 💡 Tip: wanneer je [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) toevoegt tijdens het uitvoeren van de webapp met Flask, worden wijzigingen die je aanbrengt in je applicatie onmiddellijk weergegeven zonder dat je de server opnieuw hoeft te starten. Let op! Schakel deze modus niet in bij een productie-app.

Als je `python app.py` of `python3 app.py` uitvoert, start je webserver lokaal op en kun je een kort formulier invullen om een antwoord te krijgen op je brandende vraag over waar UFO's zijn waargenomen!

Voordat je dat doet, bekijk de onderdelen van `app.py`:

1. Eerst worden afhankelijkheden geladen en de app gestart.
1. Vervolgens wordt het model geïmporteerd.
1. Daarna wordt index.html weergegeven op de home-route.

Op de `/predict`-route gebeuren er verschillende dingen wanneer het formulier wordt gepost:

1. De formulier-variabelen worden verzameld en geconverteerd naar een numpy-array. Ze worden vervolgens naar het model gestuurd en een voorspelling wordt geretourneerd.
2. De landen die we willen weergeven worden opnieuw weergegeven als leesbare tekst van hun voorspelde landcode, en die waarde wordt teruggestuurd naar index.html om in de template te worden weergegeven.

Een model op deze manier gebruiken, met Flask en een gepickled model, is relatief eenvoudig. Het moeilijkste is om te begrijpen in welke vorm de data moet zijn die naar het model moet worden gestuurd om een voorspelling te krijgen. Dat hangt allemaal af van hoe het model is getraind. Dit model heeft drie datapunten nodig om te worden ingevoerd om een voorspelling te krijgen.

In een professionele omgeving kun je zien hoe goede communicatie noodzakelijk is tussen de mensen die het model trainen en degenen die het gebruiken in een web- of mobiele app. In ons geval is het slechts één persoon, jij!

---

## 🚀 Uitdaging

In plaats van te werken in een notebook en het model te importeren in de Flask-app, kun je het model direct binnen de Flask-app trainen! Probeer je Python-code in het notebook te converteren, misschien nadat je data is schoongemaakt, om het model te trainen vanuit de app op een route genaamd `train`. Wat zijn de voor- en nadelen van deze methode?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Er zijn veel manieren om een webapp te bouwen die ML-modellen gebruikt. Maak een lijst van de manieren waarop je JavaScript of Python kunt gebruiken om een webapp te bouwen die machine learning benut. Denk na over architectuur: moet het model in de app blijven of in de cloud leven? Als het laatste, hoe zou je er toegang toe krijgen? Teken een architecturaal model voor een toegepaste ML-weboplossing.

## Opdracht

[Probeer een ander model](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.