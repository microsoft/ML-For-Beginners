<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ad2cf19d7490247558d20a6a59650d13",
  "translation_date": "2025-09-03T21:54:51+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "de"
}
-->
# Erstellen Sie eine Web-App zur Empfehlung von Küchen

In dieser Lektion erstellen Sie ein Klassifikationsmodell mit einigen der Techniken, die Sie in den vorherigen Lektionen gelernt haben, und verwenden dabei den köstlichen Küchen-Datensatz, der in dieser Serie verwendet wurde. Außerdem erstellen Sie eine kleine Web-App, um ein gespeichertes Modell zu nutzen, indem Sie die Web-Laufzeit von Onnx verwenden.

Eine der nützlichsten praktischen Anwendungen des maschinellen Lernens ist der Aufbau von Empfehlungssystemen, und Sie können heute den ersten Schritt in diese Richtung machen!

[![Präsentation dieser Web-App](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Angewandtes ML")

> 🎥 Klicken Sie auf das Bild oben für ein Video: Jen Looper erstellt eine Web-App mit klassifizierten Küchendaten

## [Quiz vor der Lektion](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/)

In dieser Lektion lernen Sie:

- Wie man ein Modell erstellt und es als Onnx-Modell speichert
- Wie man Netron verwendet, um das Modell zu inspizieren
- Wie man das Modell in einer Web-App für Inferenz verwendet

## Erstellen Sie Ihr Modell

Der Aufbau angewandter ML-Systeme ist ein wichtiger Bestandteil der Nutzung dieser Technologien für Ihre Geschäftssysteme. Sie können Modelle in Ihre Webanwendungen integrieren (und sie somit bei Bedarf offline verwenden), indem Sie Onnx verwenden.

In einer [vorherigen Lektion](../../3-Web-App/1-Web-App/README.md) haben Sie ein Regressionsmodell zu UFO-Sichtungen erstellt, es "eingepickelt" und in einer Flask-App verwendet. Obwohl diese Architektur sehr nützlich ist, handelt es sich um eine vollständige Python-App, und Ihre Anforderungen könnten die Verwendung einer JavaScript-Anwendung umfassen.

In dieser Lektion können Sie ein einfaches JavaScript-basiertes System für Inferenz erstellen. Zunächst müssen Sie jedoch ein Modell trainieren und es für die Verwendung mit Onnx konvertieren.

## Übung - Klassifikationsmodell trainieren

Trainieren Sie zunächst ein Klassifikationsmodell mit dem bereinigten Küchen-Datensatz, den wir verwendet haben.

1. Beginnen Sie mit dem Import nützlicher Bibliotheken:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Sie benötigen '[skl2onnx](https://onnx.ai/sklearn-onnx/)', um Ihr Scikit-learn-Modell in das Onnx-Format zu konvertieren.

1. Arbeiten Sie dann mit Ihren Daten wie in den vorherigen Lektionen, indem Sie eine CSV-Datei mit `read_csv()` lesen:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Entfernen Sie die ersten beiden unnötigen Spalten und speichern Sie die verbleibenden Daten als 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Speichern Sie die Labels als 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Beginnen Sie mit dem Trainingsprozess

Wir verwenden die 'SVC'-Bibliothek, die eine gute Genauigkeit bietet.

1. Importieren Sie die entsprechenden Bibliotheken aus Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Trennen Sie Trainings- und Testdaten:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Erstellen Sie ein SVC-Klassifikationsmodell wie in der vorherigen Lektion:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Testen Sie nun Ihr Modell, indem Sie `predict()` aufrufen:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Drucken Sie einen Klassifikationsbericht aus, um die Qualität des Modells zu überprüfen:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Wie wir zuvor gesehen haben, ist die Genauigkeit gut:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Konvertieren Sie Ihr Modell in Onnx

Stellen Sie sicher, dass die Konvertierung mit der richtigen Tensor-Anzahl erfolgt. Dieser Datensatz enthält 380 aufgelistete Zutaten, daher müssen Sie diese Zahl in `FloatTensorType` angeben:

1. Konvertieren Sie mit einer Tensor-Anzahl von 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Erstellen Sie die Onnx-Datei und speichern Sie sie als **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Hinweis: Sie können [Optionen](https://onnx.ai/sklearn-onnx/parameterized.html) in Ihrem Konvertierungsskript übergeben. In diesem Fall haben wir 'nocl' auf True und 'zipmap' auf False gesetzt. Da es sich um ein Klassifikationsmodell handelt, haben Sie die Möglichkeit, ZipMap zu entfernen, das eine Liste von Wörterbüchern erzeugt (nicht erforderlich). `nocl` bezieht sich darauf, ob Klasseninformationen im Modell enthalten sind. Reduzieren Sie die Größe Ihres Modells, indem Sie `nocl` auf 'True' setzen.

Wenn Sie das gesamte Notebook ausführen, wird ein Onnx-Modell erstellt und in diesem Ordner gespeichert.

## Betrachten Sie Ihr Modell

Onnx-Modelle sind in Visual Studio Code nicht sehr sichtbar, aber es gibt eine sehr gute kostenlose Software, die viele Forscher verwenden, um das Modell zu visualisieren und sicherzustellen, dass es korrekt erstellt wurde. Laden Sie [Netron](https://github.com/lutzroeder/Netron) herunter und öffnen Sie Ihre model.onnx-Datei. Sie können Ihr einfaches Modell visualisiert sehen, mit seinen 380 Eingaben und dem Klassifikator:

![Netron-Visualisierung](../../../../translated_images/netron.a05f39410211915e0f95e2c0e8b88f41e7d13d725faf660188f3802ba5c9e831.de.png)

Netron ist ein hilfreiches Tool, um Ihre Modelle zu betrachten.

Jetzt sind Sie bereit, dieses praktische Modell in einer Web-App zu verwenden. Lassen Sie uns eine App erstellen, die nützlich ist, wenn Sie in Ihren Kühlschrank schauen und herausfinden möchten, welche Kombination Ihrer übrig gebliebenen Zutaten Sie verwenden können, um eine bestimmte Küche zuzubereiten, wie von Ihrem Modell bestimmt.

## Erstellen Sie eine Empfehlungs-Webanwendung

Sie können Ihr Modell direkt in einer Web-App verwenden. Diese Architektur ermöglicht es Ihnen auch, sie lokal und sogar offline auszuführen, falls erforderlich. Beginnen Sie mit der Erstellung einer `index.html`-Datei im selben Ordner, in dem Sie Ihre `model.onnx`-Datei gespeichert haben.

1. Fügen Sie in dieser Datei _index.html_ das folgende Markup hinzu:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Fügen Sie nun innerhalb der `body`-Tags ein wenig Markup hinzu, um eine Liste von Kontrollkästchen anzuzeigen, die einige Zutaten widerspiegeln:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    Beachten Sie, dass jedes Kontrollkästchen einen Wert erhält. Dieser spiegelt den Index wider, an dem die Zutat gemäß dem Datensatz gefunden wird. Apfel, zum Beispiel, in dieser alphabetischen Liste, belegt die fünfte Spalte, daher ist sein Wert '4', da wir bei 0 zu zählen beginnen. Sie können die [Zutaten-Tabelle](../../../../4-Classification/data/ingredient_indexes.csv) konsultieren, um den Index einer bestimmten Zutat zu finden.

    Fahren Sie mit Ihrer Arbeit in der index.html-Datei fort und fügen Sie einen Skriptblock hinzu, in dem das Modell nach dem abschließenden `</div>`-Tag aufgerufen wird.

1. Importieren Sie zunächst die [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime wird verwendet, um die Ausführung Ihrer Onnx-Modelle auf einer Vielzahl von Hardwareplattformen zu ermöglichen, einschließlich Optimierungen und einer API zur Nutzung.

1. Sobald die Runtime eingerichtet ist, können Sie sie aufrufen:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

In diesem Code passieren mehrere Dinge:

1. Sie haben ein Array von 380 möglichen Werten (1 oder 0) erstellt, das je nach Auswahl eines Zutaten-Kontrollkästchens gesetzt und an das Modell zur Inferenz gesendet wird.
2. Sie haben ein Array von Kontrollkästchen erstellt und eine Möglichkeit, zu bestimmen, ob sie im `init`-Funktion aktiviert sind, die beim Start der Anwendung aufgerufen wird. Wenn ein Kontrollkästchen aktiviert ist, wird das `ingredients`-Array geändert, um die ausgewählte Zutat widerzuspiegeln.
3. Sie haben eine `testCheckboxes`-Funktion erstellt, die überprüft, ob ein Kontrollkästchen aktiviert wurde.
4. Sie verwenden die `startInference`-Funktion, wenn die Schaltfläche gedrückt wird, und starten die Inferenz, wenn ein Kontrollkästchen aktiviert ist.
5. Die Inferenzroutine umfasst:
   1. Einrichten eines asynchronen Ladevorgangs des Modells
   2. Erstellen einer Tensor-Struktur, die an das Modell gesendet wird
   3. Erstellen von 'feeds', die den `float_input`-Eingang widerspiegeln, den Sie beim Training Ihres Modells erstellt haben (Sie können Netron verwenden, um diesen Namen zu überprüfen)
   4. Senden dieser 'feeds' an das Modell und Warten auf eine Antwort

## Testen Sie Ihre Anwendung

Öffnen Sie eine Terminal-Sitzung in Visual Studio Code in dem Ordner, in dem sich Ihre index.html-Datei befindet. Stellen Sie sicher, dass Sie [http-server](https://www.npmjs.com/package/http-server) global installiert haben, und geben Sie `http-server` an der Eingabeaufforderung ein. Ein localhost sollte geöffnet werden, und Sie können Ihre Web-App anzeigen. Überprüfen Sie, welche Küche basierend auf verschiedenen Zutaten empfohlen wird:

![Zutaten-Web-App](../../../../translated_images/web-app.4c76450cabe20036f8ec6d5e05ccc0c1c064f0d8f2fe3304d3bcc0198f7dc139.de.png)

Herzlichen Glückwunsch, Sie haben eine Empfehlungs-Web-App mit einigen Feldern erstellt. Nehmen Sie sich Zeit, um dieses System weiter auszubauen!

## 🚀Herausforderung

Ihre Web-App ist sehr minimal, erweitern Sie sie daher weiter, indem Sie Zutaten und ihre Indizes aus den [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv)-Daten verwenden. Welche Geschmacksrichtungen passen zusammen, um ein bestimmtes Nationalgericht zu kreieren?

## [Quiz nach der Lektion](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/)

## Überprüfung & Selbststudium

Während diese Lektion nur kurz die Nützlichkeit der Erstellung eines Empfehlungssystems für Lebensmittelzutaten berührt hat, ist dieser Bereich der ML-Anwendungen sehr reich an Beispielen. Lesen Sie mehr darüber, wie diese Systeme aufgebaut werden:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Aufgabe 

[Erstellen Sie einen neuen Empfehlungsalgorithmus](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.