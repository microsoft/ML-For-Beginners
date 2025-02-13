# Erstellen einer Webanwendung zur Nutzung eines ML-Modells

In dieser Lektion werden Sie ein ML-Modell auf einem Datensatz trainieren, der wirklich außergewöhnlich ist: _UFO-Sichtungen im letzten Jahrhundert_, bezogen aus der Datenbank von NUFORC.

Sie werden lernen:

- Wie man ein trainiertes Modell 'pickelt'
- Wie man dieses Modell in einer Flask-App verwendet

Wir werden weiterhin Notebooks verwenden, um Daten zu bereinigen und unser Modell zu trainieren, aber Sie können den Prozess einen Schritt weiter gehen, indem Sie das Modell 'in der Wildnis' erkunden, sozusagen: in einer Webanwendung.

Um dies zu tun, müssen Sie eine Webanwendung mit Flask erstellen.

## [Vorlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Eine App erstellen

Es gibt mehrere Möglichkeiten, Webanwendungen zu erstellen, die maschinelles Lernen Modelle konsumieren. Ihre Webarchitektur kann die Art und Weise beeinflussen, wie Ihr Modell trainiert wird. Stellen Sie sich vor, Sie arbeiten in einem Unternehmen, in dem die Datenwissenschaftsgruppe ein Modell trainiert hat, das Sie in einer App verwenden sollen.

### Überlegungen

Es gibt viele Fragen, die Sie stellen müssen:

- **Ist es eine Web-App oder eine mobile App?** Wenn Sie eine mobile App erstellen oder das Modell in einem IoT-Kontext verwenden müssen, könnten Sie [TensorFlow Lite](https://www.tensorflow.org/lite/) verwenden und das Modell in einer Android- oder iOS-App nutzen.
- **Wo wird das Modell gehostet?** In der Cloud oder lokal?
- **Offline-Unterstützung.** Muss die App offline funktionieren?
- **Welche Technologie wurde verwendet, um das Modell zu trainieren?** Die gewählte Technologie kann die Werkzeuge beeinflussen, die Sie verwenden müssen.
    - **Verwendung von TensorFlow.** Wenn Sie ein Modell mit TensorFlow trainieren, bietet dieses Ökosystem die Möglichkeit, ein TensorFlow-Modell für die Verwendung in einer Web-App mit [TensorFlow.js](https://www.tensorflow.org/js/) zu konvertieren.
    - **Verwendung von PyTorch.** Wenn Sie ein Modell mit einer Bibliothek wie [PyTorch](https://pytorch.org/) erstellen, haben Sie die Möglichkeit, es im [ONNX](https://onnx.ai/) (Open Neural Network Exchange) Format für die Verwendung in JavaScript-Web-Apps zu exportieren, die das [Onnx Runtime](https://www.onnxruntime.ai/) nutzen können. Diese Option wird in einer zukünftigen Lektion für ein mit Scikit-learn trainiertes Modell untersucht.
    - **Verwendung von Lobe.ai oder Azure Custom Vision.** Wenn Sie ein ML SaaS (Software as a Service) System wie [Lobe.ai](https://lobe.ai/) oder [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) verwenden, um ein Modell zu trainieren, bietet diese Art von Software Möglichkeiten, das Modell für viele Plattformen zu exportieren, einschließlich der Erstellung einer maßgeschneiderten API, die von Ihrer Online-Anwendung in der Cloud abgefragt werden kann.

Sie haben auch die Möglichkeit, eine vollständige Flask-Webanwendung zu erstellen, die in der Lage wäre, das Modell selbst in einem Webbrowser zu trainieren. Dies kann auch mit TensorFlow.js in einem JavaScript-Kontext erfolgen.

Für unsere Zwecke, da wir mit Python-basierten Notebooks gearbeitet haben, lassen Sie uns die Schritte erkunden, die erforderlich sind, um ein trainiertes Modell aus einem solchen Notebook in ein von einer Python-basierten Web-App lesbares Format zu exportieren.

## Werkzeug

Für diese Aufgabe benötigen Sie zwei Werkzeuge: Flask und Pickle, die beide in Python laufen.

✅ Was ist [Flask](https://palletsprojects.com/p/flask/)? Flask wird von seinen Schöpfern als 'Micro-Framework' definiert und bietet die grundlegenden Funktionen von Web-Frameworks mit Python und einer Template-Engine zum Erstellen von Webseiten. Werfen Sie einen Blick auf [dieses Lernmodul](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), um das Erstellen mit Flask zu üben.

✅ Was ist [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 ist ein Python-Modul, das eine Python-Objektstruktur serialisiert und deserialisiert. Wenn Sie ein Modell 'pickeln', serialisieren oder flatten Sie seine Struktur zur Verwendung im Web. Seien Sie vorsichtig: Pickle ist nicht von Natur aus sicher, also seien Sie vorsichtig, wenn Sie aufgefordert werden, eine Datei 'un-pickeln'. Eine pickled Datei hat die Endung `.pkl`.

## Übung - Bereinigen Sie Ihre Daten

In dieser Lektion verwenden Sie Daten von 80.000 UFO-Sichtungen, die von [NUFORC](https://nuforc.org) (Das Nationale UFO-Meldungszentrum) gesammelt wurden. Diese Daten enthalten einige interessante Beschreibungen von UFO-Sichtungen, zum Beispiel:

- **Lange Beispieldarstellung.** "Ein Mann erscheint aus einem Lichtstrahl, der auf ein Grasfeld in der Nacht scheint, und läuft auf den Parkplatz von Texas Instruments zu."
- **Kurze Beispieldarstellung.** "Die Lichter verfolgten uns."

Die [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) Tabelle enthält Spalten über die `city`, `state` und `country`, wo die Sichtung stattfand, das `shape` des Objekts und dessen `latitude` und `longitude`.

In dem leeren [Notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb), das in dieser Lektion enthalten ist:

1. Importieren Sie `pandas`, `matplotlib` und `numpy`, wie Sie es in den vorherigen Lektionen getan haben, und importieren Sie die ufos-Tabelle. Sie können sich eine Beispiel-Datenmenge ansehen:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konvertieren Sie die UFO-Daten in ein kleines DataFrame mit neuen Titeln. Überprüfen Sie die eindeutigen Werte im Feld `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Jetzt können Sie die Menge der Daten, mit denen wir arbeiten müssen, reduzieren, indem Sie alle Nullwerte entfernen und nur Sichtungen zwischen 1-60 Sekunden importieren:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importieren Sie die `LabelEncoder`-Bibliothek von Scikit-learn, um die Textwerte für Länder in eine Zahl zu konvertieren:

    ✅ LabelEncoder kodiert Daten alphabetisch

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Ihre Daten sollten so aussehen:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Übung - Erstellen Sie Ihr Modell

Jetzt können Sie sich darauf vorbereiten, ein Modell zu trainieren, indem Sie die Daten in die Trainings- und Testgruppe aufteilen.

1. Wählen Sie die drei Merkmale aus, auf denen Sie trainieren möchten, als Ihren X-Vektor, und der y-Vektor wird `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` sein und eine Länder-ID zurückgeben.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Trainieren Sie Ihr Modell mit logistischer Regression:

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

Die Genauigkeit ist nicht schlecht **(ungefähr 95%)**, was nicht überraschend ist, da `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, aber es ist eine gute Übung, zu versuchen, aus Rohdaten zu trainieren, die Sie bereinigt, exportiert und dann dieses Modell in einer Web-App verwendet haben.

## Übung - 'pickeln' Sie Ihr Modell

Jetzt ist es an der Zeit, Ihr Modell _zu pickeln_! Sie können dies in wenigen Codezeilen tun. Sobald es _pickled_ ist, laden Sie Ihr pickled Modell und testen Sie es mit einem Beispieldatenarray, das Werte für Sekunden, Breite und Länge enthält,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Das Modell gibt **'3'** zurück, was der Ländercode für das Vereinigte Königreich ist. Wild! 👽

## Übung - Erstellen Sie eine Flask-App

Jetzt können Sie eine Flask-App erstellen, um Ihr Modell aufzurufen und ähnliche Ergebnisse zurückzugeben, jedoch auf eine visuell ansprechendere Weise.

1. Beginnen Sie damit, einen Ordner namens **web-app** neben der _notebook.ipynb_-Datei zu erstellen, in der sich Ihre _ufo-model.pkl_-Datei befindet.

1. Erstellen Sie in diesem Ordner drei weitere Ordner: **static**, mit einem Ordner **css** darin, und **templates**. Sie sollten jetzt die folgenden Dateien und Verzeichnisse haben:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Verweisen Sie auf den Lösungsordner, um eine Ansicht der fertigen App zu sehen.

1. Die erste Datei, die Sie im _web-app_-Ordner erstellen müssen, ist die **requirements.txt**-Datei. Wie _package.json_ in einer JavaScript-App listet diese Datei die Abhängigkeiten auf, die von der App benötigt werden. Fügen Sie in **requirements.txt** die Zeilen hinzu:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Führen Sie diese Datei jetzt aus, indem Sie in den _web-app_-Ordner navigieren:

    ```bash
    cd web-app
    ```

1. Geben Sie in Ihrem Terminal `pip install` ein, um die in _requirements.txt_ aufgelisteten Bibliotheken zu installieren:

    ```bash
    pip install -r requirements.txt
    ```

1. Jetzt sind Sie bereit, drei weitere Dateien zu erstellen, um die App abzuschließen:

    1. Erstellen Sie **app.py** im Stammverzeichnis.
    2. Erstellen Sie **index.html** im _templates_-Verzeichnis.
    3. Erstellen Sie **styles.css** im _static/css_-Verzeichnis.

1. Gestalten Sie die _styles.css_-Datei mit einigen Stilen:

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

1. Als Nächstes gestalten Sie die _index.html_-Datei:

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

    Werfen Sie einen Blick auf das Template in dieser Datei. Beachten Sie die 'Mustache'-Syntax um Variablen, die von der App bereitgestellt werden, wie den Vorhersagetext: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` fügen Sie hinzu:

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

    > 💡 Tipp: Wenn Sie [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train` hinzufügen. Was sind die Vor- und Nachteile, diesen Ansatz zu verfolgen?

## [Nachlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Überprüfung & Selbststudium

Es gibt viele Möglichkeiten, eine Web-App zu erstellen, um ML-Modelle zu konsumieren. Machen Sie eine Liste der Möglichkeiten, wie Sie JavaScript oder Python verwenden könnten, um eine Web-App zu erstellen, die maschinelles Lernen nutzt. Berücksichtigen Sie die Architektur: Sollte das Modell in der App bleiben oder in der Cloud leben? Wenn Letzteres, wie würden Sie darauf zugreifen? Zeichnen Sie ein architektonisches Modell für eine angewandte ML-Weblösung.

## Aufgabe

[Versuchen Sie ein anderes Modell](assignment.md)

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als die maßgebliche Quelle betrachtet werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.