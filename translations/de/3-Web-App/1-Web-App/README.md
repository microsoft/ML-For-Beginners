<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-04T22:01:30+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "de"
}
-->
# Erstellen einer Web-App zur Nutzung eines ML-Modells

In dieser Lektion wirst du ein ML-Modell mit einem außergewöhnlichen Datensatz trainieren: _UFO-Sichtungen des letzten Jahrhunderts_, basierend auf der Datenbank von NUFORC.

Du wirst lernen:

- Wie man ein trainiertes Modell "pickelt"
- Wie man dieses Modell in einer Flask-App verwendet

Wir werden weiterhin Notebooks verwenden, um Daten zu bereinigen und unser Modell zu trainieren. Du kannst den Prozess jedoch einen Schritt weiterführen, indem du das Modell "in freier Wildbahn" einsetzt, also in einer Web-App.

Dafür musst du eine Web-App mit Flask erstellen.

## [Quiz vor der Lektion](https://ff-quizzes.netlify.app/en/ml/)

## Eine App erstellen

Es gibt verschiedene Möglichkeiten, Web-Apps zu erstellen, die Machine-Learning-Modelle nutzen. Deine Web-Architektur kann beeinflussen, wie dein Modell trainiert wird. Stell dir vor, du arbeitest in einem Unternehmen, in dem die Data-Science-Abteilung ein Modell trainiert hat, das du in einer App verwenden sollst.

### Überlegungen

Es gibt viele Fragen, die du dir stellen musst:

- **Ist es eine Web-App oder eine Mobile-App?** Wenn du eine Mobile-App erstellst oder das Modell in einem IoT-Kontext verwenden möchtest, könntest du [TensorFlow Lite](https://www.tensorflow.org/lite/) nutzen, um das Modell in einer Android- oder iOS-App einzusetzen.
- **Wo wird das Modell gespeichert?** In der Cloud oder lokal?
- **Offline-Unterstützung.** Muss die App offline funktionieren?
- **Welche Technologie wurde verwendet, um das Modell zu trainieren?** Die gewählte Technologie kann die benötigten Tools beeinflussen.
    - **Verwendung von TensorFlow.** Wenn du ein Modell mit TensorFlow trainierst, bietet dieses Ökosystem die Möglichkeit, ein TensorFlow-Modell für die Verwendung in einer Web-App mit [TensorFlow.js](https://www.tensorflow.org/js/) zu konvertieren.
    - **Verwendung von PyTorch.** Wenn du ein Modell mit einer Bibliothek wie [PyTorch](https://pytorch.org/) erstellst, kannst du es im [ONNX](https://onnx.ai/) (Open Neural Network Exchange)-Format exportieren, um es in JavaScript-Web-Apps mit [Onnx Runtime](https://www.onnxruntime.ai/) zu verwenden. Diese Option wird in einer zukünftigen Lektion für ein mit Scikit-learn trainiertes Modell untersucht.
    - **Verwendung von Lobe.ai oder Azure Custom Vision.** Wenn du ein ML-SaaS (Software as a Service)-System wie [Lobe.ai](https://lobe.ai/) oder [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) verwendest, um ein Modell zu trainieren, bietet diese Software Möglichkeiten, das Modell für viele Plattformen zu exportieren, einschließlich der Erstellung einer maßgeschneiderten API, die von deiner Online-Anwendung in der Cloud abgefragt werden kann.

Du hast auch die Möglichkeit, eine komplette Flask-Web-App zu erstellen, die das Modell direkt im Webbrowser trainieren kann. Dies kann auch mit TensorFlow.js in einem JavaScript-Kontext erfolgen.

Für unsere Zwecke, da wir mit Python-basierten Notebooks arbeiten, schauen wir uns die Schritte an, die erforderlich sind, um ein trainiertes Modell aus einem solchen Notebook in ein Format zu exportieren, das von einer Python-basierten Web-App gelesen werden kann.

## Werkzeug

Für diese Aufgabe benötigst du zwei Tools: Flask und Pickle, die beide auf Python laufen.

✅ Was ist [Flask](https://palletsprojects.com/p/flask/)? Flask wird von seinen Entwicklern als "Micro-Framework" definiert und bietet die grundlegenden Funktionen von Web-Frameworks mit Python sowie eine Template-Engine zur Erstellung von Webseiten. Schau dir [dieses Lernmodul](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) an, um das Arbeiten mit Flask zu üben.

✅ Was ist [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 ist ein Python-Modul, das die Struktur eines Python-Objekts serialisiert und deserialisiert. Wenn du ein Modell "pickelst", serialisierst oder flachst du dessen Struktur ab, um es im Web zu verwenden. Vorsicht: Pickle ist nicht von Natur aus sicher, also sei vorsichtig, wenn du aufgefordert wirst, eine Datei zu "ent-pickeln". Eine gepickelte Datei hat die Endung `.pkl`.

## Übung - Daten bereinigen

In dieser Lektion wirst du Daten von 80.000 UFO-Sichtungen verwenden, die vom [NUFORC](https://nuforc.org) (National UFO Reporting Center) gesammelt wurden. Diese Daten enthalten einige interessante Beschreibungen von UFO-Sichtungen, zum Beispiel:

- **Lange Beispielbeschreibung.** "Ein Mann tritt aus einem Lichtstrahl, der nachts auf ein Grasfeld scheint, und rennt in Richtung des Parkplatzes von Texas Instruments."
- **Kurze Beispielbeschreibung.** "Die Lichter haben uns verfolgt."

Die Tabelle [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) enthält Spalten über die `city`, `state` und `country`, in denen die Sichtung stattfand, die `shape` des Objekts sowie dessen `latitude` und `longitude`.

Im leeren [Notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb), das in dieser Lektion enthalten ist:

1. Importiere `pandas`, `matplotlib` und `numpy`, wie du es in den vorherigen Lektionen getan hast, und importiere die UFO-Tabelle. Du kannst dir einen Beispieldatensatz ansehen:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konvertiere die UFO-Daten in ein kleines DataFrame mit neuen Titeln. Überprüfe die eindeutigen Werte im Feld `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Reduziere nun die Datenmenge, indem du alle Nullwerte entfernst und nur Sichtungen zwischen 1-60 Sekunden importierst:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importiere die `LabelEncoder`-Bibliothek von Scikit-learn, um die Textwerte für Länder in Zahlen umzuwandeln:

    ✅ LabelEncoder kodiert Daten alphabetisch

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Deine Daten sollten so aussehen:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Übung - Modell erstellen

Jetzt kannst du dich darauf vorbereiten, ein Modell zu trainieren, indem du die Daten in Trainings- und Testgruppen aufteilst.

1. Wähle die drei Merkmale aus, auf denen du trainieren möchtest, als deinen X-Vektor, und der y-Vektor wird das `Country` sein. Du möchtest in der Lage sein, `Seconds`, `Latitude` und `Longitude` einzugeben und eine Länder-ID zurückzubekommen.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Trainiere dein Modell mit logistischer Regression:

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

Die Genauigkeit ist nicht schlecht **(etwa 95%)**, was nicht überraschend ist, da `Country` und `Latitude/Longitude` korrelieren.

Das Modell, das du erstellt hast, ist nicht besonders revolutionär, da du ein `Country` aus dessen `Latitude` und `Longitude` ableiten können solltest. Es ist jedoch eine gute Übung, aus Rohdaten zu trainieren, die du bereinigt, exportiert und dann in einer Web-App verwendet hast.

## Übung - Modell "pickeln"

Jetzt ist es an der Zeit, dein Modell zu _pickeln_! Das kannst du in wenigen Codezeilen tun. Sobald es _gepickelt_ ist, lade dein gepickeltes Modell und teste es mit einem Beispieldatenarray, das Werte für Sekunden, Breitengrad und Längengrad enthält:

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Das Modell gibt **'3'** zurück, was der Ländercode für das Vereinigte Königreich ist. Verrückt! 👽

## Übung - Flask-App erstellen

Jetzt kannst du eine Flask-App erstellen, um dein Modell aufzurufen und ähnliche Ergebnisse auf eine visuell ansprechendere Weise zurückzugeben.

1. Erstelle zunächst einen Ordner namens **web-app** neben der Datei _notebook.ipynb_, in der sich auch deine Datei _ufo-model.pkl_ befindet.

1. Erstelle in diesem Ordner drei weitere Ordner: **static**, mit einem Ordner **css** darin, und **templates**. Du solltest nun die folgenden Dateien und Verzeichnisse haben:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Sieh dir den Lösungsordner an, um die fertige App zu sehen

1. Die erste Datei, die du im Ordner _web-app_ erstellst, ist die Datei **requirements.txt**. Wie _package.json_ in einer JavaScript-App listet diese Datei die Abhängigkeiten auf, die die App benötigt. Füge in **requirements.txt** die Zeilen hinzu:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Führe diese Datei nun aus, indem du zu _web-app_ navigierst:

    ```bash
    cd web-app
    ```

1. Gib in deinem Terminal `pip install` ein, um die in _requirements.txt_ aufgeführten Bibliotheken zu installieren:

    ```bash
    pip install -r requirements.txt
    ```

1. Jetzt bist du bereit, drei weitere Dateien zu erstellen, um die App fertigzustellen:

    1. Erstelle **app.py** im Root-Verzeichnis.
    2. Erstelle **index.html** im Verzeichnis _templates_.
    3. Erstelle **styles.css** im Verzeichnis _static/css_.

1. Baue die Datei _styles.css_ mit ein paar Stilen aus:

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

1. Baue als Nächstes die Datei _index.html_ aus:

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

    Schau dir das Templating in dieser Datei an. Beachte die "Mustache"-Syntax um Variablen, die von der App bereitgestellt werden, wie den Vorhersagetext: `{{}}`. Es gibt auch ein Formular, das eine Vorhersage an die Route `/predict` sendet.

    Schließlich bist du bereit, die Python-Datei zu erstellen, die den Konsum des Modells und die Anzeige der Vorhersagen steuert:

1. Füge in `app.py` Folgendes hinzu:

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

    > 💡 Tipp: Wenn du [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) hinzufügst, während du die Web-App mit Flask ausführst, werden alle Änderungen, die du an deiner Anwendung vornimmst, sofort reflektiert, ohne dass der Server neu gestartet werden muss. Vorsicht! Aktiviere diesen Modus nicht in einer Produktions-App.

Wenn du `python app.py` oder `python3 app.py` ausführst, startet dein Webserver lokal, und du kannst ein kurzes Formular ausfüllen, um eine Antwort auf deine brennende Frage zu erhalten, wo UFOs gesichtet wurden!

Bevor du das tust, wirf einen Blick auf die Teile von `app.py`:

1. Zuerst werden Abhängigkeiten geladen und die App gestartet.
1. Dann wird das Modell importiert.
1. Anschließend wird index.html auf der Home-Route gerendert.

Auf der Route `/predict` passiert Folgendes, wenn das Formular gesendet wird:

1. Die Formularvariablen werden gesammelt und in ein Numpy-Array konvertiert. Sie werden dann an das Modell gesendet, und eine Vorhersage wird zurückgegeben.
2. Die Länder, die angezeigt werden sollen, werden aus ihrem vorhergesagten Ländercode in lesbaren Text umgewandelt, und dieser Wert wird zurück an index.html gesendet, um im Template gerendert zu werden.

Ein Modell auf diese Weise mit Flask und einem gepickelten Modell zu verwenden, ist relativ einfach. Das Schwierigste ist, zu verstehen, in welcher Form die Daten an das Modell gesendet werden müssen, um eine Vorhersage zu erhalten. Das hängt davon ab, wie das Modell trainiert wurde. Dieses Modell benötigt drei Datenpunkte, um eine Vorhersage zu treffen.

In einem professionellen Umfeld kannst du sehen, wie wichtig eine gute Kommunikation zwischen den Personen ist, die das Modell trainieren, und denen, die es in einer Web- oder Mobile-App verwenden. In unserem Fall bist du es selbst!

---

## 🚀 Herausforderung

Anstatt in einem Notebook zu arbeiten und das Modell in die Flask-App zu importieren, könntest du das Modell direkt in der Flask-App trainieren! Versuche, deinen Python-Code aus dem Notebook zu konvertieren, vielleicht nachdem deine Daten bereinigt wurden, um das Modell innerhalb der App auf einer Route namens `train` zu trainieren. Was sind die Vor- und Nachteile dieser Methode?

## [Quiz nach der Lektion](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Es gibt viele Möglichkeiten, eine Web-App zu erstellen, die ML-Modelle nutzt. Erstelle eine Liste der Möglichkeiten, wie du JavaScript oder Python verwenden könntest, um eine Web-App zu erstellen, die Machine Learning nutzt. Überlege dir die Architektur: Sollte das Modell in der App bleiben oder in der Cloud leben? Wenn Letzteres, wie würdest du darauf zugreifen? Zeichne ein Architekturmodell für eine angewandte ML-Weblösung.

## Aufgabe

[Probiere ein anderes Modell aus](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe des KI-Übersetzungsdienstes [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.