<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-04T22:03:22+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "de"
}
-->
# Einführung in die Klassifikation

In diesen vier Lektionen wirst du einen grundlegenden Schwerpunkt des klassischen maschinellen Lernens erkunden – _Klassifikation_. Wir werden verschiedene Klassifikationsalgorithmen anhand eines Datensatzes über die großartigen Küchen Asiens und Indiens durchgehen. Hoffentlich hast du Appetit!

![nur eine Prise!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Feiere pan-asiatische Küchen in diesen Lektionen! Bild von [Jen Looper](https://twitter.com/jenlooper)

Klassifikation ist eine Form des [überwachten Lernens](https://wikipedia.org/wiki/Supervised_learning), die viele Gemeinsamkeiten mit Regressionsmethoden hat. Wenn maschinelles Lernen darum geht, Werte oder Namen für Dinge anhand von Datensätzen vorherzusagen, dann fällt die Klassifikation im Allgemeinen in zwei Gruppen: _binäre Klassifikation_ und _Mehrklassenklassifikation_.

[![Einführung in die Klassifikation](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Einführung in die Klassifikation")

> 🎥 Klicke auf das Bild oben für ein Video: John Guttag vom MIT stellt die Klassifikation vor

Erinnere dich:

- **Lineare Regression** half dir, Beziehungen zwischen Variablen vorherzusagen und genaue Vorhersagen darüber zu treffen, wo ein neuer Datenpunkt in Bezug auf diese Linie liegen würde. So konntest du beispielsweise vorhersagen, _wie viel ein Kürbis im September im Vergleich zu Dezember kosten würde_.
- **Logistische Regression** half dir, "binäre Kategorien" zu entdecken: Bei diesem Preisniveau, _ist dieser Kürbis orange oder nicht-orange_?

Die Klassifikation verwendet verschiedene Algorithmen, um andere Möglichkeiten zu finden, das Label oder die Klasse eines Datenpunkts zu bestimmen. Lass uns mit diesen Küchendaten arbeiten, um zu sehen, ob wir anhand einer Gruppe von Zutaten die Herkunftsküche bestimmen können.

## [Quiz vor der Lektion](https://ff-quizzes.netlify.app/en/ml/)

> ### [Diese Lektion ist auch in R verfügbar!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Einführung

Die Klassifikation ist eine der grundlegenden Tätigkeiten von Forschern und Datenwissenschaftlern im Bereich des maschinellen Lernens. Von der einfachen Klassifikation eines binären Wertes ("Ist diese E-Mail Spam oder nicht?") bis hin zur komplexen Bildklassifikation und -segmentierung mithilfe von Computer Vision ist es immer nützlich, Daten in Klassen einzuteilen und Fragen dazu zu stellen.

Wissenschaftlich ausgedrückt erstellt deine Klassifikationsmethode ein prädiktives Modell, das es dir ermöglicht, die Beziehung zwischen Eingabevariablen und Ausgabevariablen abzubilden.

![binäre vs. Mehrklassenklassifikation](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binäre vs. Mehrklassenprobleme, die Klassifikationsalgorithmen bewältigen müssen. Infografik von [Jen Looper](https://twitter.com/jenlooper)

Bevor wir mit der Bereinigung unserer Daten, ihrer Visualisierung und der Vorbereitung für unsere ML-Aufgaben beginnen, lass uns ein wenig über die verschiedenen Möglichkeiten lernen, wie maschinelles Lernen zur Klassifikation von Daten genutzt werden kann.

Abgeleitet aus der [Statistik](https://wikipedia.org/wiki/Statistical_classification) verwendet die Klassifikation im klassischen maschinellen Lernen Merkmale wie `smoker`, `weight` und `age`, um die _Wahrscheinlichkeit der Entwicklung von Krankheit X_ zu bestimmen. Als eine Technik des überwachten Lernens, ähnlich den Regressionsübungen, die du zuvor durchgeführt hast, sind deine Daten beschriftet, und die ML-Algorithmen verwenden diese Beschriftungen, um Klassen (oder 'Merkmale') eines Datensatzes zu klassifizieren und vorherzusagen und sie einer Gruppe oder einem Ergebnis zuzuordnen.

✅ Nimm dir einen Moment Zeit, um dir einen Datensatz über Küchen vorzustellen. Welche Fragen könnte ein Mehrklassenmodell beantworten? Welche Fragen könnte ein binäres Modell beantworten? Was wäre, wenn du herausfinden möchtest, ob eine bestimmte Küche wahrscheinlich Bockshornklee verwendet? Oder was wäre, wenn du sehen möchtest, ob du mit einer Tüte voller Sternanis, Artischocken, Blumenkohl und Meerrettich ein typisches indisches Gericht zubereiten könntest?

[![Verrückte Mystery-Körbe](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Verrückte Mystery-Körbe")

> 🎥 Klicke auf das Bild oben für ein Video. Die ganze Prämisse der Show 'Chopped' ist der 'Mystery-Korb', bei dem Köche aus einer zufälligen Auswahl an Zutaten ein Gericht zaubern müssen. Sicherlich hätte ein ML-Modell geholfen!

## Hallo 'Classifier'

Die Frage, die wir an diesen Küchendatensatz stellen möchten, ist tatsächlich eine **Mehrklassenfrage**, da wir mit mehreren potenziellen Nationalküchen arbeiten. Angesichts einer Reihe von Zutaten, zu welcher dieser vielen Klassen passt die Daten?

Scikit-learn bietet mehrere verschiedene Algorithmen zur Klassifikation von Daten, je nachdem, welche Art von Problem du lösen möchtest. In den nächsten zwei Lektionen wirst du einige dieser Algorithmen kennenlernen.

## Übung – Daten bereinigen und ausbalancieren

Die erste Aufgabe, bevor wir mit diesem Projekt beginnen, besteht darin, die Daten zu bereinigen und **auszubalancieren**, um bessere Ergebnisse zu erzielen. Beginne mit der leeren Datei _notebook.ipynb_ im Stammverzeichnis dieses Ordners.

Das erste, was du installieren musst, ist [imblearn](https://imbalanced-learn.org/stable/). Dies ist ein Scikit-learn-Paket, das dir hilft, die Daten besser auszubalancieren (du wirst gleich mehr über diese Aufgabe erfahren).

1. Um `imblearn` zu installieren, führe `pip install` aus, wie folgt:

    ```python
    pip install imblearn
    ```

1. Importiere die Pakete, die du benötigst, um deine Daten zu importieren und zu visualisieren, und importiere auch `SMOTE` aus `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Jetzt bist du bereit, die Daten zu importieren.

1. Die nächste Aufgabe besteht darin, die Daten zu importieren:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Mit `read_csv()` liest du den Inhalt der CSV-Datei _cusines.csv_ und speicherst ihn in der Variablen `df`.

1. Überprüfe die Form der Daten:

    ```python
    df.head()
    ```

   Die ersten fünf Zeilen sehen so aus:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Hole dir Informationen über diese Daten, indem du `info()` aufrufst:

    ```python
    df.info()
    ```

    Deine Ausgabe sieht so aus:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Übung – Küchen entdecken

Jetzt wird die Arbeit interessanter. Lass uns die Verteilung der Daten pro Küche entdecken.

1. Stelle die Daten als Balken dar, indem du `barh()` aufrufst:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![Verteilung der Küchendaten](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Es gibt eine begrenzte Anzahl von Küchen, aber die Verteilung der Daten ist ungleichmäßig. Das kannst du beheben! Bevor du das tust, erkunde noch ein wenig mehr.

1. Finde heraus, wie viele Daten pro Küche verfügbar sind, und gib sie aus:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    Die Ausgabe sieht so aus:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Zutaten entdecken

Jetzt kannst du tiefer in die Daten eintauchen und herausfinden, welche typischen Zutaten pro Küche verwendet werden. Du solltest wiederkehrende Daten bereinigen, die Verwirrung zwischen den Küchen stiften. Lass uns mehr über dieses Problem erfahren.

1. Erstelle eine Funktion `create_ingredient()` in Python, um ein Zutaten-Datenframe zu erstellen. Diese Funktion beginnt damit, eine nicht hilfreiche Spalte zu entfernen, und sortiert die Zutaten nach ihrer Häufigkeit:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Jetzt kannst du diese Funktion verwenden, um eine Vorstellung von den zehn beliebtesten Zutaten pro Küche zu bekommen.

1. Rufe `create_ingredient()` auf und stelle die Daten mit `barh()` dar:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Mache dasselbe für die japanischen Daten:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanisch](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Nun für die chinesischen Zutaten:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinesisch](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Stelle die indischen Zutaten dar:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indisch](../../../../4-Classification/1-Introduction/images/indian.png)

1. Schließlich stelle die koreanischen Zutaten dar:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![koreanisch](../../../../4-Classification/1-Introduction/images/korean.png)

1. Entferne nun die häufigsten Zutaten, die Verwirrung zwischen verschiedenen Küchen stiften, indem du `drop()` aufrufst:

   Jeder liebt Reis, Knoblauch und Ingwer!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Den Datensatz ausbalancieren

Nachdem du die Daten bereinigt hast, verwende [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) – "Synthetic Minority Over-sampling Technique" – um sie auszugleichen.

1. Rufe `fit_resample()` auf. Diese Strategie generiert neue Stichproben durch Interpolation.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Durch das Ausbalancieren deiner Daten erzielst du bessere Ergebnisse bei der Klassifikation. Denke an eine binäre Klassifikation. Wenn die meisten deiner Daten einer Klasse angehören, wird ein ML-Modell diese Klasse häufiger vorhersagen, einfach weil es mehr Daten dafür gibt. Das Ausbalancieren der Daten nimmt verzerrte Daten und hilft, dieses Ungleichgewicht zu beseitigen.

1. Jetzt kannst du die Anzahl der Labels pro Zutat überprüfen:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Deine Ausgabe sieht so aus:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Die Daten sind schön sauber, ausgeglichen und sehr lecker!

1. Der letzte Schritt besteht darin, deine ausgeglichenen Daten, einschließlich Labels und Features, in ein neues Datenframe zu speichern, das in eine Datei exportiert werden kann:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Du kannst einen letzten Blick auf die Daten werfen, indem du `transformed_df.head()` und `transformed_df.info()` aufrufst. Speichere eine Kopie dieser Daten für die Verwendung in zukünftigen Lektionen:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Diese frische CSV-Datei befindet sich nun im Stamm-Datenordner.

---

## 🚀 Herausforderung

Dieses Curriculum enthält mehrere interessante Datensätze. Durchsuche die `data`-Ordner und sieh nach, ob einer Datensätze enthält, die sich für binäre oder Mehrklassenklassifikation eignen. Welche Fragen würdest du an diesen Datensatz stellen?

## [Quiz nach der Lektion](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Erkunde die API von SMOTE. Für welche Anwendungsfälle ist sie am besten geeignet? Welche Probleme löst sie?

## Aufgabe 

[Erkunde Klassifikationsmethoden](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe des KI-Übersetzungsdienstes [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.