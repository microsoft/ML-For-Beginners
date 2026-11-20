# Einstieg in Python und Scikit-learn für Regressionsmodelle

![Zusammenfassung der Regressionen in einer Sketchnote](../../../../translated_images/de/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz vor der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

> ### [Diese Lektion ist auch in R verfügbar!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Einführung

In diesen vier Lektionen werden Sie entdecken, wie man Regressionsmodelle erstellt. Wir werden gleich besprechen, wozu diese dienen. Aber bevor Sie irgendetwas tun, stellen Sie sicher, dass Sie die richtigen Werkzeuge haben, um zu starten!

In dieser Lektion lernen Sie:

- Wie Sie Ihren Computer für lokale Machine Learning Aufgaben konfigurieren.
- Wie Sie mit Jupyter Notebooks arbeiten.
- Wie Sie Scikit-learn verwenden, einschließlich Installation.
- Wie Sie lineare Regression mit einer praktischen Übung erkunden.

## Installationen und Konfigurationen

[![ML für Anfänger - Richten Sie Ihre Tools für den Aufbau von Machine Learning Modellen ein](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML für Anfänger - Richten Sie Ihre Tools für den Aufbau von Machine Learning Modellen ein")

> 🎥 Klicken Sie auf das obige Bild für ein kurzes Video, das die Konfiguration Ihres Computers für ML erklärt.

1. **Installieren Sie Python**. Stellen Sie sicher, dass [Python](https://www.python.org/downloads/) auf Ihrem Computer installiert ist. Sie werden Python für viele Data Science und Machine Learning Aufgaben nutzen. Die meisten Computersysteme haben bereits eine Python-Installation. Es gibt auch nützliche [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), die einigen Nutzern die Einrichtung erleichtern.

Einige Anwendungen von Python erfordern jedoch unterschiedliche Softwareversionen. Daher ist es sinnvoll, innerhalb einer [virtuellen Umgebung](https://docs.python.org/3/library/venv.html) zu arbeiten.

2. **Installieren Sie Visual Studio Code**. Stellen Sie sicher, dass Visual Studio Code auf Ihrem Computer installiert ist. Folgen Sie diesen Anweisungen, um [Visual Studio Code zu installieren](https://code.visualstudio.com/) für die grundlegende Installation. Sie werden in diesem Kurs Python in Visual Studio Code verwenden, daher könnte es hilfreich sein, sich anzuschauen, wie man [Visual Studio Code für die Python-Entwicklung konfiguriert](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott).

> Machen Sie sich mit Python vertraut, indem Sie diese Sammlung von [Lernmodulen](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) durcharbeiten.
>
> [![Python mit Visual Studio Code einrichten](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Python mit Visual Studio Code einrichten")
>
> 🎥 Klicken Sie auf das obige Bild für ein Video: Python innerhalb von VS Code verwenden.

3. **Installieren Sie Scikit-learn**, indem Sie [diesen Anweisungen](https://scikit-learn.org/stable/install.html) folgen. Da Sie sicherstellen müssen, dass Sie Python 3 verwenden, wird empfohlen, eine virtuelle Umgebung zu nutzen. Beachten Sie, dass es spezielle Anweisungen für die Installation auf einem M1 Mac gibt, auf die oben verlinkte Seite verweist.

1. **Installieren Sie Jupyter Notebook**. Sie müssen das [Jupyter-Paket installieren](https://pypi.org/project/jupyter/).

## Ihre ML-Entwicklungsumgebung

Sie werden **Notebooks** verwenden, um Ihren Python-Code zu entwickeln und Machine Learning Modelle zu erstellen. Diese Dateitypen sind ein übliches Werkzeug für Data Scientists und sind an ihrer Endung `.ipynb` zu erkennen.

Notebooks sind eine interaktive Umgebung, die es dem Entwickler erlauben, sowohl Code zu schreiben als auch Notizen und Dokumentationen rund um den Code hinzuzufügen, was besonders hilfreich für experimentelle oder forschungsorientierte Projekte ist.

[![ML für Anfänger - Richten Sie Jupyter Notebooks ein, um mit Regressionsmodellen zu starten](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML für Anfänger - Richten Sie Jupyter Notebooks ein, um mit Regressionsmodellen zu starten")

> 🎥 Klicken Sie auf das obige Bild für ein kurzes Video, das Sie durch diese Übung führt.

### Übung - Arbeiten mit einem Notebook

In diesem Ordner finden Sie die Datei _notebook.ipynb_.

1. Öffnen Sie _notebook.ipynb_ in Visual Studio Code.

   Ein Jupyter-Server startet mit Python 3+. In dem Notebook finden Sie Bereiche, die Sie `ausführen` können, Codeabschnitte. Um einen Codeblock auszuführen, wählen Sie das Symbol, das wie ein Play-Button aussieht.

1. Wählen Sie das `md`-Icon und fügen Sie etwas Markdown hinzu, und folgenden Text **# Willkommen in Ihrem Notebook**.

   Fügen Sie anschließend etwas Python-Code ein.

1. Tippen Sie **print('hello notebook')** in den Codeblock.
1. Wählen Sie den Pfeil, um den Code auszuführen.

   Sie sollten die gedruckte Anweisung sehen:

    ```output
    hello notebook
    ```

![VS Code mit geöffnetem Notebook](../../../../translated_images/de/notebook.4a3ee31f396b8832.webp)

Sie können Ihren Code mit Kommentaren durchziehen, um das Notebook selbst zu dokumentieren.

✅ Denken Sie einen Moment darüber nach, wie unterschiedlich die Arbeitsumgebung eines Webentwicklers im Vergleich zu der eines Data Scientists ist.

## Startklar mit Scikit-learn

Jetzt, da Python in Ihrer lokalen Umgebung eingerichtet ist und Sie mit Jupyter Notebooks vertraut sind, lassen Sie uns die gleiche Sicherheit mit Scikit-learn gewinnen (ausgesprochen `sci` wie in `science`). Scikit-learn bietet eine [umfangreiche API](https://scikit-learn.org/stable/modules/classes.html#api-ref), um Ihnen bei ML-Aufgaben zu helfen.

Laut ihrer [Webseite](https://scikit-learn.org/stable/getting_started.html) ist "Scikit-learn eine Open-Source-Machine-Learning-Bibliothek, die überwachte und unüberwachte Lernmethoden unterstützt. Sie bietet auch verschiedene Werkzeuge zur Modellanpassung, Datenvorverarbeitung, Modellauswahl und -bewertung sowie viele weitere Hilfsmittel."

In diesem Kurs verwenden Sie Scikit-learn und andere Werkzeuge, um Machine Learning Modelle zu erstellen, die sogenannte „traditionelle Machine Learning“-Aufgaben durchführen. Wir vermeiden bewusst neuronale Netze und Deep Learning, da diese besser in unserem bald erscheinenden „KI für Anfänger“-Lehrplan behandelt werden.

Scikit-learn macht es einfach, Modelle zu bauen und zu evaluieren. Es konzentriert sich hauptsächlich auf die Nutzung numerischer Daten und enthält mehrere fertige Datensätze als Lernwerkzeuge. Außerdem beinhaltet es vorgefertigte Modelle, die die Studierenden ausprobieren können. Lassen Sie uns den Prozess erkunden, vorverpackte Daten zu laden und einen eingebauten Schätzer zu verwenden, um mit einigen Basisdaten Ihr erstes ML-Modell in Scikit-learn zu erstellen.

## Übung - Ihr erstes Scikit-learn Notebook

> Dieses Tutorial wurde inspiriert von dem [Lineare Regression Beispiel](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) auf der Scikit-learn-Webseite.


[![ML für Anfänger - Ihr erstes lineares Regressionsprojekt in Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML für Anfänger - Ihr erstes lineares Regressionsprojekt in Python")

> 🎥 Klicken Sie auf das obige Bild für ein kurzes Video, das Sie durch diese Übung führt.

Im _notebook.ipynb_-Datei, die zu dieser Lektion gehört, löschen Sie alle Zellen, indem Sie das 'Mülleimer'-Symbol drücken.

In diesem Abschnitt arbeiten Sie mit einem kleinen Datensatz über Diabetes, der zu Lernzwecken in Scikit-learn eingebaut ist. Stellen Sie sich vor, Sie wollten eine Behandlung für Diabetiker testen. Machine Learning Modelle könnten Ihnen helfen zu bestimmen, welche Patienten besser auf die Behandlung reagieren würden, basierend auf Kombinationen von Variablen. Selbst ein sehr einfaches Regressionsmodell kann, wenn es visualisiert wird, Informationen über Variablen zeigen, die Ihnen helfen könnten, Ihre theoretischen klinischen Studien zu organisieren.

✅ Es gibt viele Arten von Regressionsmethoden, und welche Sie wählen, hängt von der Antwort ab, die Sie suchen. Wenn Sie die wahrscheinliche Körpergröße einer Person eines gegebenen Alters vorhersagen wollen, verwenden Sie lineare Regression, da Sie einen **numerischen Wert** suchen. Wenn Sie herausfinden wollen, ob eine Art von Küche als vegan angesehen werden sollte oder nicht, suchen Sie eine **Kategoriezuteilung**, dann würden Sie logistische Regression verwenden. Später lernen Sie mehr über logistische Regression. Denken Sie ein bisschen über Fragen nach, die Sie an Daten stellen können, und welche dieser Methoden dafür besser geeignet wären.

Lassen Sie uns mit dieser Aufgabe beginnen.

### Bibliotheken importieren

Für diese Aufgabe importieren wir einige Bibliotheken:

- **matplotlib**. Es ist ein nützliches [Graphing-Tool](https://matplotlib.org/) und wir werden es verwenden, um ein Liniendiagramm zu erstellen.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) ist eine nützliche Bibliothek für die Verarbeitung numerischer Daten in Python.
- **sklearn**. Das ist die [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) Bibliothek.

Importieren Sie einige Bibliotheken, die Ihnen bei Ihren Aufgaben helfen.

1. Fügen Sie Importe hinzu, indem Sie den folgenden Code eintippen:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Oben importieren Sie `matplotlib`, `numpy` und aus `sklearn` importieren Sie `datasets`, `linear_model` und `model_selection`. `model_selection` wird verwendet, um Daten in Trainings- und Testsets aufzuteilen.

### Der Diabetes-Datensatz

Der eingebaute [Diabetes-Datensatz](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) enthält 442 Stichproben zu Diabetes, mit 10 Merkmalvariablen, darunter:

- Alter: Alter in Jahren
- bmi: Body-Mass-Index
- bp: durchschnittlicher Blutdruck
- s1 tc: T-Zellen (eine Art von weißen Blutkörperchen)

✅ Dieser Datensatz enthält das Konzept „Geschlecht“ als wichtige Merkmalsvariable für die Diabetesforschung. Viele medizinische Datensätze beinhalten diese Art binärer Klassifikation. Denken Sie darüber nach, wie solche Kategorisierungen bestimmte Teile der Bevölkerung möglicherweise von Behandlungen ausschließen.

Laden Sie nun die Daten X und y.

> 🎓 Denken Sie daran, dies ist überwachtes Lernen, und wir benötigen eine benannte Zielvariable „y“.

Laden Sie in einer neuen Code-Zelle den Diabetes-Datensatz durch den Aufruf von `load_diabetes()`. Der Parameter `return_X_y=True` signalisiert, dass `X` eine Datenmatrix ist und `y` das Regressionsziel.

1. Fügen Sie einige print-Befehle hinzu, um die Form der Datenmatrix und ihr erstes Element anzuzeigen:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Was Sie als Antwort zurückbekommen, ist ein Tupel. Sie weisen die beiden ersten Werte des Tupels `X` und `y` zu. Lernen Sie mehr [über Tupel](https://wikipedia.org/wiki/Tuple).

    Sie sehen, dass diese Daten 442 Elemente enthalten, die in Arrays mit 10 Elementen geformt sind:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Denken Sie ein wenig über die Beziehung zwischen den Daten und dem Regressionsziel nach. Lineare Regression sagt Beziehungen zwischen dem Merkmal X und der Zielvariable y voraus. Können Sie das [Ziel](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) für den Diabetes-Datensatz in der Dokumentation finden? Was zeigt dieser Datensatz, basierend auf diesem Ziel?

2. Wählen Sie nun einen Teil dieses Datensatzes für das Plotten aus, indem Sie die 3. Spalte des Datensatzes auswählen. Sie können dies tun, indem Sie `:` verwenden, um alle Zeilen zu wählen, und dann die 3. Spalte mit dem Index (2) auswählen. Sie können die Daten auch in ein 2D-Array umformen – wie für das Plotten erforderlich – mit `reshape(n_rows, n_columns)`. Wenn einer der Parameter -1 ist, wird die entsprechende Dimension automatisch berechnet.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Geben Sie jederzeit die Daten aus, um deren Form zu überprüfen.

3. Jetzt, da Sie Daten zum Plotten bereit haben, sehen Sie, ob eine Maschine helfen kann, eine logische Aufteilung zwischen den Zahlen in diesem Datensatz zu bestimmen. Dafür müssen Sie die Daten (X) und das Ziel (y) in Test- und Trainingssets aufteilen. Scikit-learn bietet einen einfachen Weg dazu; Sie können Ihre Testdaten an einem gegebenen Punkt aufteilen.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Jetzt sind Sie bereit, Ihr Modell zu trainieren! Laden Sie das lineare Regressionsmodell und trainieren Sie es mit Ihren X- und y-Trainingssets mit `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` ist eine Funktion, die Sie in vielen ML-Bibliotheken wie TensorFlow sehen werden.

5. Erstellen Sie dann eine Vorhersage mit Testdaten mittels der Funktion `predict()`. Diese wird verwendet, um die Linie zwischen den Datengruppen zu zeichnen.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nun ist es Zeit, die Daten in einem Diagramm anzuzeigen. Matplotlib ist hierfür ein sehr nützliches Werkzeug. Erstellen Sie ein Streudiagramm aller X- und y-Testdaten und verwenden Sie die Vorhersage, um eine Linie an der passendsten Stelle zwischen den Datencluster des Modells zu zeichnen.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![Ein Streudiagramm mit Datenpunkten zum Thema Diabetes](../../../../translated_images/de/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Denken Sie ein wenig darüber nach, was hier passiert. Eine gerade Linie läuft durch viele kleine Datenpunkte, aber was tut sie genau? Können Sie erkennen, wie Sie diese Linie verwenden können, um vorherzusagen, wo ein neuer, ungesehener Datenpunkt in Bezug auf die y-Achse des Plots passen sollte? Versuchen Sie, den praktischen Nutzen dieses Modells in Worte zu fassen.

Herzlichen Glückwunsch, Sie haben Ihr erstes lineares Regressionsmodell gebaut, eine Vorhersage erstellt und diese in einem Diagramm dargestellt!

---
## 🚀Herausforderung

Plotten Sie eine andere Variable aus diesem Datensatz. Tipp: Bearbeiten Sie diese Zeile: `X = X[:,2]`. Was können Sie über den Verlauf von Diabetes als Krankheit herausfinden, basierend auf dem Ziel dieses Datensatzes?
## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

In diesem Tutorial haben Sie mit einfacher linearer Regression gearbeitet, nicht mit univariater oder multipler linearer Regression. Lesen Sie ein wenig über die Unterschiede zwischen diesen Methoden oder sehen Sie sich [dieses Video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef) an.

Lesen Sie mehr über das Konzept der Regression und überlegen Sie, welche Arten von Fragen mit dieser Technik beantwortet werden können. Nehmen Sie an diesem [Tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) teil, um Ihr Verständnis zu vertiefen.

## Aufgabe

[Ein anderer Datensatz](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Haftungsausschluss**:
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache gilt als maßgebliche Quelle. Bei kritischen Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->