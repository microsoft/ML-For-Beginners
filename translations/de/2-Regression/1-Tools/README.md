<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6b1cb0e46d4c5b747eff6e3607642760",
  "translation_date": "2025-09-03T21:41:17+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "de"
}
-->
# Erste Schritte mit Python und Scikit-learn für Regressionsmodelle

![Zusammenfassung von Regressionen in einer Sketchnote](../../../../translated_images/ml-regression.4e4f70e3b3ed446e3ace348dec973e133fa5d3680fbc8412b61879507369b98d.de.png)

> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz vor der Lektion](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/9/)

> ### [Diese Lektion ist auch in R verfügbar!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Einführung

In diesen vier Lektionen lernen Sie, wie man Regressionsmodelle erstellt. Wir werden gleich besprechen, wofür diese verwendet werden. Aber bevor Sie irgendetwas tun, stellen Sie sicher, dass Sie die richtigen Werkzeuge haben, um den Prozess zu starten!

In dieser Lektion lernen Sie:

- Ihren Computer für lokale Machine-Learning-Aufgaben zu konfigurieren.
- Mit Jupyter-Notebooks zu arbeiten.
- Scikit-learn zu verwenden, einschließlich der Installation.
- Lineare Regression mit einer praktischen Übung zu erkunden.

## Installationen und Konfigurationen

[![ML für Anfänger - Richten Sie Ihre Werkzeuge ein, um Machine-Learning-Modelle zu erstellen](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML für Anfänger - Richten Sie Ihre Werkzeuge ein, um Machine-Learning-Modelle zu erstellen")

> 🎥 Klicken Sie auf das Bild oben für ein kurzes Video zur Konfiguration Ihres Computers für ML.

1. **Python installieren**. Stellen Sie sicher, dass [Python](https://www.python.org/downloads/) auf Ihrem Computer installiert ist. Sie werden Python für viele Aufgaben in der Datenwissenschaft und im maschinellen Lernen verwenden. Die meisten Computersysteme haben bereits eine Python-Installation. Es gibt auch nützliche [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), die die Einrichtung für einige Benutzer erleichtern.

   Einige Anwendungen von Python erfordern jedoch eine bestimmte Version der Software, während andere eine andere Version benötigen. Aus diesem Grund ist es sinnvoll, in einer [virtuellen Umgebung](https://docs.python.org/3/library/venv.html) zu arbeiten.

2. **Visual Studio Code installieren**. Stellen Sie sicher, dass Visual Studio Code auf Ihrem Computer installiert ist. Folgen Sie diesen Anweisungen, um [Visual Studio Code zu installieren](https://code.visualstudio.com/) für die grundlegende Installation. Sie werden Python in Visual Studio Code in diesem Kurs verwenden, daher sollten Sie sich mit der [Konfiguration von Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) für die Python-Entwicklung vertraut machen.

   > Machen Sie sich mit Python vertraut, indem Sie diese Sammlung von [Learn-Modulen](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) durcharbeiten.
   >
   > [![Python mit Visual Studio Code einrichten](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Python mit Visual Studio Code einrichten")
   >
   > 🎥 Klicken Sie auf das Bild oben für ein Video: Python in VS Code verwenden.

3. **Scikit-learn installieren**, indem Sie [diesen Anweisungen](https://scikit-learn.org/stable/install.html) folgen. Da Sie sicherstellen müssen, dass Sie Python 3 verwenden, wird empfohlen, eine virtuelle Umgebung zu verwenden. Beachten Sie, dass es spezielle Anweisungen gibt, wenn Sie diese Bibliothek auf einem M1 Mac installieren.

4. **Jupyter Notebook installieren**. Sie müssen das [Jupyter-Paket installieren](https://pypi.org/project/jupyter/).

## Ihre ML-Entwicklungsumgebung

Sie werden **Notebooks** verwenden, um Ihren Python-Code zu entwickeln und Machine-Learning-Modelle zu erstellen. Diese Art von Datei ist ein häufig verwendetes Werkzeug für Datenwissenschaftler und kann an ihrer Endung `.ipynb` erkannt werden.

Notebooks sind eine interaktive Umgebung, die es Entwicklern ermöglicht, sowohl Code zu schreiben als auch Notizen und Dokumentation rund um den Code hinzuzufügen, was besonders hilfreich für experimentelle oder forschungsorientierte Projekte ist.

[![ML für Anfänger - Jupyter Notebooks einrichten, um mit der Erstellung von Regressionsmodellen zu beginnen](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML für Anfänger - Jupyter Notebooks einrichten, um mit der Erstellung von Regressionsmodellen zu beginnen")

> 🎥 Klicken Sie auf das Bild oben für ein kurzes Video, das diese Übung durchgeht.

### Übung - Mit einem Notebook arbeiten

In diesem Ordner finden Sie die Datei _notebook.ipynb_.

1. Öffnen Sie _notebook.ipynb_ in Visual Studio Code.

   Ein Jupyter-Server wird mit Python 3+ gestartet. Sie finden Bereiche des Notebooks, die `ausgeführt` werden können, also Codeabschnitte. Sie können einen Codeblock ausführen, indem Sie das Symbol auswählen, das wie eine Wiedergabetaste aussieht.

2. Wählen Sie das `md`-Symbol und fügen Sie ein wenig Markdown sowie den folgenden Text hinzu: **# Willkommen in Ihrem Notebook**.

   Fügen Sie anschließend etwas Python-Code hinzu.

3. Geben Sie **print('hello notebook')** in den Codeblock ein.
4. Wählen Sie den Pfeil aus, um den Code auszuführen.

   Sie sollten die gedruckte Aussage sehen:

    ```output
    hello notebook
    ```

![VS Code mit einem geöffneten Notebook](../../../../translated_images/notebook.4a3ee31f396b88325607afda33cadcc6368de98040ff33942424260aa84d75f2.de.jpg)

Sie können Ihren Code mit Kommentaren durchsetzen, um das Notebook selbst zu dokumentieren.

✅ Denken Sie einen Moment darüber nach, wie unterschiedlich die Arbeitsumgebung eines Webentwicklers im Vergleich zu der eines Datenwissenschaftlers ist.

## Einführung in Scikit-learn

Jetzt, da Python in Ihrer lokalen Umgebung eingerichtet ist und Sie sich mit Jupyter-Notebooks vertraut gemacht haben, machen wir uns ebenso vertraut mit Scikit-learn (ausgesprochen `sci` wie in `science`). Scikit-learn bietet eine [umfangreiche API](https://scikit-learn.org/stable/modules/classes.html#api-ref), die Ihnen bei der Durchführung von ML-Aufgaben hilft.

Laut ihrer [Website](https://scikit-learn.org/stable/getting_started.html) ist "Scikit-learn eine Open-Source-Machine-Learning-Bibliothek, die sowohl überwachte als auch unüberwachte Lernmethoden unterstützt. Sie bietet auch verschiedene Werkzeuge für Modellanpassung, Datenvorverarbeitung, Modellauswahl und -bewertung sowie viele andere Hilfsmittel."

In diesem Kurs werden Sie Scikit-learn und andere Werkzeuge verwenden, um Machine-Learning-Modelle zu erstellen, die sogenannte 'traditionelle Machine-Learning'-Aufgaben ausführen. Wir haben bewusst auf neuronale Netzwerke und Deep Learning verzichtet, da diese besser in unserem kommenden 'AI for Beginners'-Lehrplan behandelt werden.

Scikit-learn macht es einfach, Modelle zu erstellen und zu bewerten. Es konzentriert sich hauptsächlich auf die Verwendung numerischer Daten und enthält mehrere vorgefertigte Datensätze, die als Lernwerkzeuge verwendet werden können. Es enthält auch vorgefertigte Modelle, die Studenten ausprobieren können. Lassen Sie uns den Prozess des Ladens vorgefertigter Daten und der Verwendung eines eingebauten Schätzers für das erste ML-Modell mit Scikit-learn erkunden.

## Übung - Ihr erstes Scikit-learn-Notebook

> Dieses Tutorial wurde inspiriert von dem [Beispiel zur linearen Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) auf der Scikit-learn-Website.

[![ML für Anfänger - Ihr erstes Projekt zur linearen Regression in Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML für Anfänger - Ihr erstes Projekt zur linearen Regression in Python")

> 🎥 Klicken Sie auf das Bild oben für ein kurzes Video, das diese Übung durchgeht.

In der Datei _notebook.ipynb_, die mit dieser Lektion verbunden ist, löschen Sie alle Zellen, indem Sie auf das Symbol 'Mülleimer' klicken.

In diesem Abschnitt arbeiten Sie mit einem kleinen Datensatz über Diabetes, der in Scikit-learn für Lernzwecke integriert ist. Stellen Sie sich vor, Sie wollten eine Behandlung für Diabetespatienten testen. Machine-Learning-Modelle könnten Ihnen helfen zu bestimmen, welche Patienten besser auf die Behandlung ansprechen würden, basierend auf Kombinationen von Variablen. Selbst ein sehr einfaches Regressionsmodell könnte, wenn es visualisiert wird, Informationen über Variablen zeigen, die Ihnen helfen könnten, Ihre theoretischen klinischen Studien zu organisieren.

✅ Es gibt viele Arten von Regressionsmethoden, und welche Sie wählen, hängt von der Frage ab, die Sie beantworten möchten. Wenn Sie die wahrscheinliche Größe einer Person in einem bestimmten Alter vorhersagen möchten, würden Sie lineare Regression verwenden, da Sie einen **numerischen Wert** suchen. Wenn Sie herausfinden möchten, ob eine Art von Küche als vegan betrachtet werden sollte oder nicht, suchen Sie nach einer **Kategoriezuweisung**, sodass Sie logistische Regression verwenden würden. Sie werden später mehr über logistische Regression lernen. Denken Sie ein wenig über einige Fragen nach, die Sie an Daten stellen können, und welche dieser Methoden dafür besser geeignet wären.

Lassen Sie uns mit dieser Aufgabe beginnen.

### Bibliotheken importieren

Für diese Aufgabe importieren wir einige Bibliotheken:

- **matplotlib**. Es ist ein nützliches [Grafikwerkzeug](https://matplotlib.org/) und wir werden es verwenden, um ein Liniendiagramm zu erstellen.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) ist eine nützliche Bibliothek für die Verarbeitung numerischer Daten in Python.
- **sklearn**. Dies ist die [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)-Bibliothek.

Importieren Sie einige Bibliotheken, die Ihnen bei Ihren Aufgaben helfen.

1. Fügen Sie die Importe hinzu, indem Sie den folgenden Code eingeben:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Oben importieren Sie `matplotlib`, `numpy` und Sie importieren `datasets`, `linear_model` und `model_selection` aus `sklearn`. `model_selection` wird verwendet, um Daten in Trainings- und Testsets aufzuteilen.

### Der Diabetes-Datensatz

Der integrierte [Diabetes-Datensatz](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) enthält 442 Datenproben zu Diabetes mit 10 Feature-Variablen, darunter:

- age: Alter in Jahren
- bmi: Body-Mass-Index
- bp: Durchschnittlicher Blutdruck
- s1 tc: T-Zellen (eine Art von weißen Blutkörperchen)

✅ Dieser Datensatz enthält das Konzept von 'Geschlecht' als Feature-Variable, die für die Forschung zu Diabetes wichtig ist. Viele medizinische Datensätze enthalten diese Art von binärer Klassifikation. Denken Sie ein wenig darüber nach, wie solche Kategorisierungen bestimmte Teile der Bevölkerung von Behandlungen ausschließen könnten.

Laden Sie nun die X- und y-Daten.

> 🎓 Denken Sie daran, dass dies überwachtes Lernen ist und wir ein benanntes 'y'-Ziel benötigen.

In einer neuen Codezelle laden Sie den Diabetes-Datensatz, indem Sie `load_diabetes()` aufrufen. Der Input `return_X_y=True` signalisiert, dass `X` eine Datenmatrix und `y` das Regressionsziel sein wird.

1. Fügen Sie einige Print-Befehle hinzu, um die Form der Datenmatrix und ihr erstes Element anzuzeigen:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Was Sie als Antwort erhalten, ist ein Tupel. Sie weisen die beiden ersten Werte des Tupels `X` und `y` zu. Erfahren Sie mehr [über Tupel](https://wikipedia.org/wiki/Tuple).

    Sie können sehen, dass diese Daten 442 Elemente enthalten, die in Arrays mit 10 Elementen geformt sind:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Denken Sie ein wenig über die Beziehung zwischen den Daten und dem Regressionsziel nach. Lineare Regression sagt Beziehungen zwischen Feature X und Zielvariable y voraus. Können Sie das [Ziel](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) für den Diabetes-Datensatz in der Dokumentation finden? Was zeigt dieser Datensatz, wenn man das Ziel betrachtet?

2. Wählen Sie als Nächstes einen Teil dieses Datensatzes aus, um ihn zu plotten, indem Sie die dritte Spalte des Datensatzes auswählen. Sie können dies tun, indem Sie den `:`-Operator verwenden, um alle Zeilen auszuwählen, und dann die dritte Spalte mit dem Index (2) auswählen. Sie können die Daten auch in ein 2D-Array umformen - wie für das Plotten erforderlich - indem Sie `reshape(n_rows, n_columns)` verwenden. Wenn einer der Parameter -1 ist, wird die entsprechende Dimension automatisch berechnet.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Drucken Sie die Daten jederzeit aus, um ihre Form zu überprüfen.

3. Jetzt, da Sie Daten bereit zum Plotten haben, können Sie sehen, ob eine Maschine helfen kann, eine logische Trennung zwischen den Zahlen in diesem Datensatz zu bestimmen. Dazu müssen Sie sowohl die Daten (X) als auch das Ziel (y) in Test- und Trainingssets aufteilen. Scikit-learn bietet eine einfache Möglichkeit, dies zu tun; Sie können Ihre Testdaten an einem bestimmten Punkt aufteilen.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Jetzt sind Sie bereit, Ihr Modell zu trainieren! Laden Sie das lineare Regressionsmodell und trainieren Sie es mit Ihren X- und y-Trainingssets, indem Sie `model.fit()` verwenden:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` ist eine Funktion, die Sie in vielen ML-Bibliotheken wie TensorFlow sehen werden.

5. Erstellen Sie anschließend eine Vorhersage mit Testdaten, indem Sie die Funktion `predict()` verwenden. Diese wird verwendet, um die Linie zwischen den Datengruppen zu zeichnen.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Jetzt ist es Zeit, die Daten in einem Diagramm anzuzeigen. Matplotlib ist ein sehr nützliches Werkzeug für diese Aufgabe. Erstellen Sie ein Streudiagramm aller X- und y-Testdaten und verwenden Sie die Vorhersage, um eine Linie an der passendsten Stelle zwischen den Datengruppierungen des Modells zu zeichnen.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![Ein Streudiagramm, das Datenpunkte zu Diabetes zeigt](../../../../translated_images/scatterplot.ad8b356bcbb33be68d54050e09b9b7bfc03e94fde7371f2609ae43f4c563b2d7.de.png)
✅ Denk ein wenig darüber nach, was hier passiert. Eine gerade Linie verläuft durch viele kleine Datenpunkte, aber was genau macht sie? Kannst du erkennen, wie du diese Linie nutzen könntest, um vorherzusagen, wo ein neuer, unbekannter Datenpunkt in Bezug auf die y-Achse des Plots liegen sollte? Versuche, den praktischen Nutzen dieses Modells in Worte zu fassen.

Herzlichen Glückwunsch, du hast dein erstes lineares Regressionsmodell erstellt, eine Vorhersage damit gemacht und sie in einem Plot dargestellt!

---
## 🚀 Herausforderung

Zeichne eine andere Variable aus diesem Datensatz. Hinweis: Bearbeite diese Zeile: `X = X[:,2]`. Angesichts des Ziels dieses Datensatzes, was kannst du über den Verlauf von Diabetes als Krankheit herausfinden?

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/10/)

## Rückblick & Selbststudium

In diesem Tutorial hast du mit einfacher linearer Regression gearbeitet, anstatt mit univariater oder multipler linearer Regression. Lies ein wenig über die Unterschiede zwischen diesen Methoden oder sieh dir [dieses Video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef) an.

Lies mehr über das Konzept der Regression und denke darüber nach, welche Arten von Fragen mit dieser Technik beantwortet werden können. Nimm an [diesem Tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) teil, um dein Verständnis zu vertiefen.

## Aufgabe

[Ein anderer Datensatz](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.