<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T21:47:31+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "no"
}
-->
# Bygg en webapplikasjon for å bruke en ML-modell

I denne leksjonen skal du trene en ML-modell på et datasett som er helt utenomjordisk: _UFO-observasjoner fra det siste århundret_, hentet fra NUFORCs database.

Du vil lære:

- Hvordan 'pickle' en trent modell
- Hvordan bruke den modellen i en Flask-applikasjon

Vi fortsetter å bruke notebooks for å rense data og trene modellen vår, men du kan ta prosessen et steg videre ved å utforske hvordan man bruker en modell "ute i det fri", så å si: i en webapplikasjon.

For å gjøre dette må du bygge en webapplikasjon ved hjelp av Flask.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Bygge en applikasjon

Det finnes flere måter å bygge webapplikasjoner som kan bruke maskinlæringsmodeller. Din webarkitektur kan påvirke hvordan modellen din blir trent. Tenk deg at du jobber i en bedrift der data science-gruppen har trent en modell som de vil at du skal bruke i en applikasjon.

### Vurderinger

Det er mange spørsmål du må stille:

- **Er det en webapplikasjon eller en mobilapplikasjon?** Hvis du bygger en mobilapplikasjon eller trenger å bruke modellen i en IoT-sammenheng, kan du bruke [TensorFlow Lite](https://www.tensorflow.org/lite/) og bruke modellen i en Android- eller iOS-applikasjon.
- **Hvor skal modellen ligge?** I skyen eller lokalt?
- **Støtte for offline bruk.** Må applikasjonen fungere offline?
- **Hvilken teknologi ble brukt til å trene modellen?** Den valgte teknologien kan påvirke verktøyene du må bruke.
    - **Bruke TensorFlow.** Hvis du trener en modell med TensorFlow, for eksempel, gir det økosystemet muligheten til å konvertere en TensorFlow-modell for bruk i en webapplikasjon ved hjelp av [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Bruke PyTorch.** Hvis du bygger en modell med et bibliotek som [PyTorch](https://pytorch.org/), har du muligheten til å eksportere den i [ONNX](https://onnx.ai/) (Open Neural Network Exchange)-format for bruk i JavaScript-webapplikasjoner som kan bruke [Onnx Runtime](https://www.onnxruntime.ai/). Denne muligheten vil bli utforsket i en fremtidig leksjon for en Scikit-learn-trent modell.
    - **Bruke Lobe.ai eller Azure Custom Vision.** Hvis du bruker et ML SaaS (Software as a Service)-system som [Lobe.ai](https://lobe.ai/) eller [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) for å trene en modell, gir denne typen programvare måter å eksportere modellen for mange plattformer, inkludert å bygge en skreddersydd API som kan forespørres i skyen av din online-applikasjon.

Du har også muligheten til å bygge en hel Flask-webapplikasjon som kan trene modellen selv i en nettleser. Dette kan også gjøres ved hjelp av TensorFlow.js i en JavaScript-sammenheng.

For vårt formål, siden vi har jobbet med Python-baserte notebooks, la oss utforske trinnene du må ta for å eksportere en trent modell fra en slik notebook til et format som kan leses av en Python-bygget webapplikasjon.

## Verktøy

For denne oppgaven trenger du to verktøy: Flask og Pickle, begge som kjører på Python.

✅ Hva er [Flask](https://palletsprojects.com/p/flask/)? Definert som et 'mikro-rammeverk' av sine skapere, gir Flask de grunnleggende funksjonene til webrammeverk ved bruk av Python og en templatemotor for å bygge nettsider. Ta en titt på [denne læringsmodulen](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) for å øve på å bygge med Flask.

✅ Hva er [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 er et Python-modul som serialiserer og de-serialiserer en Python-objektstruktur. Når du 'pickler' en modell, serialiserer eller flater du ut strukturen dens for bruk på nettet. Vær forsiktig: pickle er ikke iboende sikkert, så vær forsiktig hvis du blir bedt om å 'un-pickle' en fil. En picklet fil har suffikset `.pkl`.

## Øvelse - rense dataene dine

I denne leksjonen skal du bruke data fra 80,000 UFO-observasjoner, samlet av [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Disse dataene har noen interessante beskrivelser av UFO-observasjoner, for eksempel:

- **Lang beskrivelse.** "En mann kommer ut fra en lysstråle som skinner på en gresslette om natten, og han løper mot Texas Instruments parkeringsplass".
- **Kort beskrivelse.** "lysene jaget oss".

Regnearket [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) inkluderer kolonner om `city`, `state` og `country` der observasjonen fant sted, objektets `shape` og dets `latitude` og `longitude`.

I den tomme [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) som er inkludert i denne leksjonen:

1. importer `pandas`, `matplotlib` og `numpy` som du gjorde i tidligere leksjoner, og importer UFO-regnearket. Du kan ta en titt på et eksempel på datasettet:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konverter UFO-dataene til en liten dataframe med nye titler. Sjekk de unike verdiene i `Country`-feltet.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Nå kan du redusere mengden data vi trenger å håndtere ved å fjerne eventuelle nullverdier og kun importere observasjoner mellom 1-60 sekunder:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importer Scikit-learns `LabelEncoder`-bibliotek for å konvertere tekstverdier for land til et tall:

    ✅ LabelEncoder koder data alfabetisk

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Dataene dine bør se slik ut:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Øvelse - bygg modellen din

Nå kan du gjøre deg klar til å trene en modell ved å dele dataene inn i trenings- og testgrupper.

1. Velg de tre funksjonene du vil trene på som din X-vektor, og y-vektoren vil være `Country`. Du vil kunne legge inn `Seconds`, `Latitude` og `Longitude` og få en land-ID tilbake.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Tren modellen din ved hjelp av logistisk regresjon:

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

Nøyaktigheten er ikke dårlig **(rundt 95%)**, ikke overraskende, siden `Country` og `Latitude/Longitude` korrelerer.

Modellen du opprettet er ikke veldig revolusjonerende, siden du burde kunne utlede et `Country` fra dets `Latitude` og `Longitude`, men det er en god øvelse å prøve å trene fra rådata som du renset, eksporterte, og deretter bruke denne modellen i en webapplikasjon.

## Øvelse - 'pickle' modellen din

Nå er det på tide å _pickle_ modellen din! Du kan gjøre det med noen få linjer kode. Når den er _picklet_, last inn den picklete modellen og test den mot et eksempeldataarray som inneholder verdier for sekunder, breddegrad og lengdegrad.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Modellen returnerer **'3'**, som er landkoden for Storbritannia. Utrolig! 👽

## Øvelse - bygg en Flask-applikasjon

Nå kan du bygge en Flask-applikasjon for å kalle modellen din og returnere lignende resultater, men på en mer visuelt tiltalende måte.

1. Start med å opprette en mappe kalt **web-app** ved siden av _notebook.ipynb_-filen der _ufo-model.pkl_-filen ligger.

1. I den mappen opprett tre flere mapper: **static**, med en mappe **css** inni, og **templates**. Du bør nå ha følgende filer og kataloger:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Se løsningsmappen for en visning av den ferdige applikasjonen

1. Den første filen du oppretter i _web-app_-mappen er **requirements.txt**-filen. Som _package.json_ i en JavaScript-applikasjon, lister denne filen opp avhengigheter som kreves av applikasjonen. I **requirements.txt** legg til linjene:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Nå, kjør denne filen ved å navigere til _web-app_:

    ```bash
    cd web-app
    ```

1. I terminalen din, skriv `pip install` for å installere bibliotekene som er oppført i _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Nå er du klar til å opprette tre flere filer for å fullføre applikasjonen:

    1. Opprett **app.py** i roten.
    2. Opprett **index.html** i _templates_-katalogen.
    3. Opprett **styles.css** i _static/css_-katalogen.

1. Bygg ut _styles.css_-filen med noen få stiler:

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

1. Deretter bygger du ut _index.html_-filen:

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

    Ta en titt på templatingen i denne filen. Legg merke til 'mustache'-syntaksen rundt variabler som vil bli levert av applikasjonen, som prediksjonsteksten: `{{}}`. Det er også et skjema som sender en prediksjon til `/predict`-ruten.

    Til slutt er du klar til å bygge Python-filen som driver forbruket av modellen og visningen av prediksjoner:

1. I `app.py` legg til:

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

    > 💡 Tips: når du legger til [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) mens du kjører webapplikasjonen ved hjelp av Flask, vil eventuelle endringer du gjør i applikasjonen bli reflektert umiddelbart uten behov for å starte serveren på nytt. Vær oppmerksom! Ikke aktiver denne modusen i en produksjonsapplikasjon.

Hvis du kjører `python app.py` eller `python3 app.py` - starter webserveren din opp lokalt, og du kan fylle ut et kort skjema for å få svar på ditt brennende spørsmål om hvor UFO-er har blitt observert!

Før du gjør det, ta en titt på delene av `app.py`:

1. Først lastes avhengighetene og applikasjonen starter.
1. Deretter importeres modellen.
1. Deretter rendres index.html på hjemmeruten.

På `/predict`-ruten skjer flere ting når skjemaet sendes inn:

1. Skjemavariablene samles og konverteres til et numpy-array. De sendes deretter til modellen, og en prediksjon returneres.
2. Landene som vi ønsker skal vises, rendres på nytt som lesbar tekst fra deres predikerte landkode, og den verdien sendes tilbake til index.html for å bli rendret i templaten.

Å bruke en modell på denne måten, med Flask og en picklet modell, er relativt enkelt. Det vanskeligste er å forstå hvilken form dataene må ha for å bli sendt til modellen for å få en prediksjon. Det avhenger helt av hvordan modellen ble trent. Denne har tre datapunkter som må legges inn for å få en prediksjon.

I en profesjonell setting kan du se hvor viktig god kommunikasjon er mellom de som trener modellen og de som bruker den i en web- eller mobilapplikasjon. I vårt tilfelle er det bare én person, deg!

---

## 🚀 Utfordring

I stedet for å jobbe i en notebook og importere modellen til Flask-applikasjonen, kan du trene modellen direkte i Flask-applikasjonen! Prøv å konvertere Python-koden din i notebooken, kanskje etter at dataene dine er renset, for å trene modellen fra applikasjonen på en rute kalt `train`. Hva er fordeler og ulemper med å bruke denne metoden?

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium

Det finnes mange måter å bygge en webapplikasjon for å bruke ML-modeller. Lag en liste over måtene du kan bruke JavaScript eller Python til å bygge en webapplikasjon som utnytter maskinlæring. Tenk på arkitektur: bør modellen bli værende i applikasjonen eller ligge i skyen? Hvis det siste, hvordan ville du få tilgang til den? Tegn opp en arkitekturmodell for en anvendt ML-webløsning.

## Oppgave

[Prøv en annen modell](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.