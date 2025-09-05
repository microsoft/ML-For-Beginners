<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T00:37:58+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "da"
}
-->
# Byg en webapp til at bruge en ML-model

I denne lektion vil du træne en ML-model på et datasæt, der er helt ude af denne verden: _UFO-observationer over det sidste århundrede_, hentet fra NUFORC's database.

Du vil lære:

- Hvordan man 'pickler' en trænet model
- Hvordan man bruger den model i en Flask-app

Vi fortsætter med at bruge notebooks til at rense data og træne vores model, men du kan tage processen et skridt videre ved at udforske brugen af en model 'i det fri', så at sige: i en webapp.

For at gøre dette skal du bygge en webapp ved hjælp af Flask.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Bygning af en app

Der er flere måder at bygge webapps på, der kan bruge machine learning-modeller. Din webarkitektur kan påvirke, hvordan din model trænes. Forestil dig, at du arbejder i en virksomhed, hvor data science-gruppen har trænet en model, som de ønsker, at du skal bruge i en app.

### Overvejelser

Der er mange spørgsmål, du skal stille:

- **Er det en webapp eller en mobilapp?** Hvis du bygger en mobilapp eller skal bruge modellen i en IoT-sammenhæng, kan du bruge [TensorFlow Lite](https://www.tensorflow.org/lite/) og bruge modellen i en Android- eller iOS-app.
- **Hvor skal modellen placeres?** I skyen eller lokalt?
- **Offline support.** Skal appen fungere offline?
- **Hvilken teknologi blev brugt til at træne modellen?** Den valgte teknologi kan påvirke de værktøjer, du skal bruge.
    - **Brug af TensorFlow.** Hvis du træner en model med TensorFlow, giver det økosystem mulighed for at konvertere en TensorFlow-model til brug i en webapp ved hjælp af [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Brug af PyTorch.** Hvis du bygger en model med et bibliotek som [PyTorch](https://pytorch.org/), har du mulighed for at eksportere den i [ONNX](https://onnx.ai/) (Open Neural Network Exchange)-format til brug i JavaScript-webapps, der kan bruge [Onnx Runtime](https://www.onnxruntime.ai/). Denne mulighed vil blive udforsket i en fremtidig lektion for en Scikit-learn-trænet model.
    - **Brug af Lobe.ai eller Azure Custom Vision.** Hvis du bruger et ML SaaS-system (Software as a Service) som [Lobe.ai](https://lobe.ai/) eller [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) til at træne en model, giver denne type software måder at eksportere modellen til mange platforme, herunder opbygning af en skræddersyet API, der kan forespørges i skyen af din online-applikation.

Du har også mulighed for at bygge en hel Flask-webapp, der kan træne modellen direkte i en webbrowser. Dette kan også gøres ved hjælp af TensorFlow.js i en JavaScript-sammenhæng.

For vores formål, da vi har arbejdet med Python-baserede notebooks, lad os udforske de trin, du skal tage for at eksportere en trænet model fra en sådan notebook til et format, der kan læses af en Python-bygget webapp.

## Værktøj

Til denne opgave skal du bruge to værktøjer: Flask og Pickle, som begge kører på Python.

✅ Hvad er [Flask](https://palletsprojects.com/p/flask/)? Defineret som et 'mikro-framework' af sine skabere, giver Flask de grundlæggende funktioner i webframeworks ved hjælp af Python og en templating-motor til at bygge websider. Tag et kig på [dette Learn-modul](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) for at øve dig i at bygge med Flask.

✅ Hvad er [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 er et Python-modul, der serialiserer og de-serialiserer en Python-objektstruktur. Når du 'pickler' en model, serialiserer eller flader du dens struktur ud til brug på nettet. Vær forsigtig: pickle er ikke i sig selv sikker, så vær forsigtig, hvis du bliver bedt om at 'un-pickle' en fil. En pickled fil har suffikset `.pkl`.

## Øvelse - rens dine data

I denne lektion vil du bruge data fra 80.000 UFO-observationer, indsamlet af [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Disse data har nogle interessante beskrivelser af UFO-observationer, for eksempel:

- **Lang eksempelbeskrivelse.** "En mand kommer ud af en lysstråle, der skinner på en græsmark om natten, og han løber mod Texas Instruments' parkeringsplads".
- **Kort eksempelbeskrivelse.** "lysene jagtede os".

Regnearket [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) inkluderer kolonner om `city`, `state` og `country`, hvor observationen fandt sted, objektets `shape` og dets `latitude` og `longitude`.

I den tomme [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb), der er inkluderet i denne lektion:

1. Importer `pandas`, `matplotlib` og `numpy`, som du gjorde i tidligere lektioner, og importer UFO-regnearket. Du kan tage et kig på et eksempel-datasæt:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konverter UFO-dataene til en lille dataframe med friske titler. Tjek de unikke værdier i `Country`-feltet.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Nu kan du reducere mængden af data, vi skal håndtere, ved at droppe eventuelle null-værdier og kun importere observationer mellem 1-60 sekunder:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importer Scikit-learns `LabelEncoder`-bibliotek for at konvertere tekstværdier for lande til et nummer:

    ✅ LabelEncoder koder data alfabetisk

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Dine data bør se sådan ud:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Øvelse - byg din model

Nu kan du gøre dig klar til at træne en model ved at opdele dataene i trænings- og testgrupper.

1. Vælg de tre funktioner, du vil træne på som din X-vektor, og y-vektoren vil være `Country`. Du vil kunne indtaste `Seconds`, `Latitude` og `Longitude` og få et land-id som resultat.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Træn din model ved hjælp af logistisk regression:

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

Nøjagtigheden er ikke dårlig **(omkring 95%)**, hvilket ikke er overraskende, da `Country` og `Latitude/Longitude` korrelerer.

Den model, du har oprettet, er ikke særlig revolutionerende, da du burde kunne udlede et `Country` fra dets `Latitude` og `Longitude`, men det er en god øvelse at prøve at træne fra rå data, som du har renset, eksporteret og derefter bruge denne model i en webapp.

## Øvelse - 'pickle' din model

Nu er det tid til at _pickle_ din model! Du kan gøre det med få linjer kode. Når den er _pickled_, skal du indlæse din pickled model og teste den mod en eksempel-datarray, der indeholder værdier for sekunder, breddegrad og længdegrad.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Modellen returnerer **'3'**, hvilket er landekoden for Storbritannien. Vildt! 👽

## Øvelse - byg en Flask-app

Nu kan du bygge en Flask-app til at kalde din model og returnere lignende resultater, men på en mere visuelt tiltalende måde.

1. Start med at oprette en mappe kaldet **web-app** ved siden af _notebook.ipynb_-filen, hvor din _ufo-model.pkl_-fil ligger.

1. I den mappe skal du oprette tre flere mapper: **static**, med en mappe **css** indeni, og **templates**. Du bør nu have følgende filer og mapper:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Se løsningsmappen for at få et overblik over den færdige app

1. Den første fil, der skal oprettes i _web-app_-mappen, er **requirements.txt**-filen. Ligesom _package.json_ i en JavaScript-app, lister denne fil afhængigheder, der kræves af appen. I **requirements.txt** tilføj linjerne:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Kør nu denne fil ved at navigere til _web-app_:

    ```bash
    cd web-app
    ```

1. I din terminal skal du skrive `pip install` for at installere de biblioteker, der er angivet i _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Nu er du klar til at oprette tre flere filer for at færdiggøre appen:

    1. Opret **app.py** i roden.
    2. Opret **index.html** i _templates_-mappen.
    3. Opret **styles.css** i _static/css_-mappen.

1. Udfyld _styles.css_-filen med nogle få stilarter:

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

1. Dernæst skal du udfylde _index.html_-filen:

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

    Tag et kig på templating i denne fil. Bemærk 'mustache'-syntaksen omkring variabler, der vil blive leveret af appen, som for eksempel prediction-teksten: `{{}}`. Der er også en formular, der sender en prediction til `/predict`-ruten.

    Endelig er du klar til at bygge Python-filen, der driver forbruget af modellen og visningen af forudsigelser:

1. I `app.py` tilføj:

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

    > 💡 Tip: Når du tilføjer [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode), mens du kører webappen ved hjælp af Flask, vil eventuelle ændringer, du foretager i din applikation, straks blive afspejlet uden behov for at genstarte serveren. Pas på! Aktiver ikke denne tilstand i en produktionsapp.

Hvis du kører `python app.py` eller `python3 app.py` - starter din webserver lokalt, og du kan udfylde en kort formular for at få svar på dit brændende spørgsmål om, hvor UFO'er er blevet observeret!

Før du gør det, skal du tage et kig på delene af `app.py`:

1. Først indlæses afhængigheder, og appen starter.
1. Derefter importeres modellen.
1. Derefter renderes index.html på hjemmeruten.

På `/predict`-ruten sker der flere ting, når formularen sendes:

1. Formularvariablerne indsamles og konverteres til en numpy-array. De sendes derefter til modellen, og en prediction returneres.
2. De lande, vi ønsker vist, genrenderes som læsbar tekst fra deres forudsagte landekode, og den værdi sendes tilbage til index.html for at blive renderet i templaten.

At bruge en model på denne måde, med Flask og en pickled model, er relativt ligetil. Det sværeste er at forstå, hvilken form dataene skal have for at blive sendt til modellen for at få en prediction. Det afhænger helt af, hvordan modellen blev trænet. Denne har tre datapunkter, der skal indtastes for at få en prediction.

I en professionel sammenhæng kan du se, hvor vigtig god kommunikation er mellem dem, der træner modellen, og dem, der bruger den i en web- eller mobilapp. I vores tilfælde er det kun én person, dig!

---

## 🚀 Udfordring

I stedet for at arbejde i en notebook og importere modellen til Flask-appen, kunne du træne modellen direkte i Flask-appen! Prøv at konvertere din Python-kode i notebooken, måske efter dine data er renset, til at træne modellen fra appen på en rute kaldet `train`. Hvad er fordele og ulemper ved at forfølge denne metode?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Der er mange måder at bygge en webapp på, der kan bruge ML-modeller. Lav en liste over måder, du kunne bruge JavaScript eller Python til at bygge en webapp, der udnytter machine learning. Overvej arkitektur: Skal modellen blive i appen eller leve i skyen? Hvis det sidste, hvordan ville du få adgang til den? Tegn en arkitekturmodel for en anvendt ML-webløsning.

## Opgave

[Prøv en anden model](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.