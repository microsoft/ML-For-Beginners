<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2f88fbc741d792890ff2f1430fe0dae0",
  "translation_date": "2025-09-03T21:38:33+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "de"
}
-->
# Erstellen eines Regressionsmodells mit Scikit-learn: Regression auf vier Arten

![Infografik zu linearer vs. polynomialer Regression](../../../../translated_images/linear-polynomial.5523c7cb6576ccab0fecbd0e3505986eb2d191d9378e785f82befcf3a578a6e7.de.png)
> Infografik von [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/13/)

> ### [Diese Lektion ist auch in R verfügbar!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Einführung 

Bisher hast du untersucht, was Regression ist, anhand von Beispieldaten aus dem Kürbispreis-Datensatz, den wir in dieser Lektion verwenden werden. Du hast ihn auch mit Matplotlib visualisiert.

Jetzt bist du bereit, tiefer in die Regression für maschinelles Lernen einzutauchen. Während die Visualisierung hilft, Daten zu verstehen, liegt die wahre Stärke des maschinellen Lernens im _Trainieren von Modellen_. Modelle werden mit historischen Daten trainiert, um automatisch Datenabhängigkeiten zu erfassen, und sie ermöglichen es, Ergebnisse für neue Daten vorherzusagen, die das Modell zuvor nicht gesehen hat.

In dieser Lektion wirst du mehr über zwei Arten von Regression lernen: _einfache lineare Regression_ und _polynomiale Regression_, zusammen mit einigen mathematischen Grundlagen dieser Techniken. Diese Modelle ermöglichen es uns, Kürbispreise basierend auf verschiedenen Eingabedaten vorherzusagen.

[![ML für Anfänger - Verständnis der linearen Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML für Anfänger - Verständnis der linearen Regression")

> 🎥 Klicke auf das Bild oben für eine kurze Videoübersicht zur linearen Regression.

> Im gesamten Lehrplan gehen wir von minimalen Mathematikkenntnissen aus und versuchen, das Thema für Studierende aus anderen Bereichen zugänglich zu machen. Achte auf Notizen, 🧮 Hinweise, Diagramme und andere Lernhilfen, die das Verständnis erleichtern.

### Voraussetzungen

Du solltest inzwischen mit der Struktur der Kürbisdaten vertraut sein, die wir untersuchen. Du findest sie vorab geladen und bereinigt in der Datei _notebook.ipynb_ dieser Lektion. In der Datei wird der Kürbispreis pro Scheffel in einem neuen DataFrame angezeigt. Stelle sicher, dass du diese Notebooks in Visual Studio Code ausführen kannst.

### Vorbereitung

Zur Erinnerung: Du lädst diese Daten, um Fragen dazu zu stellen.

- Wann ist die beste Zeit, Kürbisse zu kaufen? 
- Welchen Preis kann ich für eine Kiste Miniaturkürbisse erwarten?
- Sollte ich sie in halben Scheffelkörben oder in 1 1/9 Scheffelkisten kaufen?
Lass uns weiter in diese Daten eintauchen.

In der vorherigen Lektion hast du einen Pandas-DataFrame erstellt und ihn mit einem Teil des ursprünglichen Datensatzes gefüllt, wobei du die Preise pro Scheffel standardisiert hast. Dadurch konntest du jedoch nur etwa 400 Datenpunkte sammeln, und das nur für die Herbstmonate.

Schau dir die Daten an, die wir in dem begleitenden Notebook dieser Lektion vorab geladen haben. Die Daten sind vorab geladen, und ein erster Streudiagramm wurde erstellt, um Monatsdaten zu zeigen. Vielleicht können wir durch eine gründlichere Bereinigung der Daten noch mehr Details über die Natur der Daten erhalten.

## Eine lineare Regressionslinie

Wie du in Lektion 1 gelernt hast, besteht das Ziel einer linearen Regression darin, eine Linie zu zeichnen, um:

- **Beziehungen zwischen Variablen zu zeigen**. Die Beziehung zwischen Variablen darzustellen.
- **Vorhersagen zu treffen**. Genaue Vorhersagen darüber zu machen, wo ein neuer Datenpunkt im Verhältnis zu dieser Linie liegen würde.

Typisch für die **Methode der kleinsten Quadrate** ist es, diese Art von Linie zu zeichnen. Der Begriff "kleinste Quadrate" bedeutet, dass alle Datenpunkte um die Regressionslinie quadriert und dann addiert werden. Idealerweise ist diese Summe so klein wie möglich, da wir eine geringe Fehleranzahl oder `kleinste Quadrate` anstreben.

Wir tun dies, da wir eine Linie modellieren möchten, die die geringste kumulative Entfernung von allen unseren Datenpunkten hat. Wir quadrieren die Terme vor dem Addieren, da uns die Größe der Abweichung wichtiger ist als ihre Richtung.

> **🧮 Zeig mir die Mathematik** 
> 
> Diese Linie, die als _Best-Fit-Linie_ bezeichnet wird, kann durch [eine Gleichung](https://en.wikipedia.org/wiki/Simple_linear_regression) ausgedrückt werden: 
> 
> ```
> Y = a + bX
> ```
>
> `X` ist die 'erklärende Variable'. `Y` ist die 'abhängige Variable'. Die Steigung der Linie ist `b`, und `a` ist der y-Achsenabschnitt, der den Wert von `Y` angibt, wenn `X = 0`. 
>
>![Berechnung der Steigung](../../../../translated_images/slope.f3c9d5910ddbfcf9096eb5564254ba22c9a32d7acd7694cab905d29ad8261db3.de.png)
>
> Zuerst berechnen wir die Steigung `b`. Infografik von [Jen Looper](https://twitter.com/jenlooper)
>
> Mit Bezug auf die ursprüngliche Frage zu den Kürbisdaten: "Vorhersage des Preises eines Kürbisses pro Scheffel nach Monat", würde `X` den Preis und `Y` den Verkaufsmonat darstellen. 
>
>![Vervollständigung der Gleichung](../../../../translated_images/calculation.a209813050a1ddb141cdc4bc56f3af31e67157ed499e16a2ecf9837542704c94.de.png)
>
> Berechnung des Wertes von Y. Wenn du etwa 4 $ zahlst, muss es April sein! Infografik von [Jen Looper](https://twitter.com/jenlooper)
>
> Die Mathematik, die die Linie berechnet, muss die Steigung der Linie zeigen, die auch vom Achsenabschnitt abhängt, oder wo `Y` liegt, wenn `X = 0`.
>
> Du kannst die Methode zur Berechnung dieser Werte auf der Website [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) beobachten. Besuche auch [diesen Rechner für die Methode der kleinsten Quadrate](https://www.mathsisfun.com/data/least-squares-calculator.html), um zu sehen, wie die Werte die Linie beeinflussen.

## Korrelation

Ein weiterer Begriff, den du verstehen solltest, ist der **Korrelationskoeffizient** zwischen den gegebenen X- und Y-Variablen. Mit einem Streudiagramm kannst du diesen Koeffizienten schnell visualisieren. Ein Diagramm mit Datenpunkten, die in einer ordentlichen Linie verstreut sind, hat eine hohe Korrelation, während ein Diagramm mit Datenpunkten, die überall zwischen X und Y verstreut sind, eine niedrige Korrelation hat.

Ein gutes lineares Regressionsmodell ist eines, das eine hohe (näher an 1 als an 0) Korrelation aufweist, basierend auf der Methode der kleinsten Quadrate mit einer Regressionslinie.

✅ Führe das begleitende Notebook dieser Lektion aus und sieh dir das Streudiagramm von Monat zu Preis an. Scheint die Datenassoziation zwischen Monat und Preis für Kürbisverkäufe laut deiner visuellen Interpretation des Streudiagramms eine hohe oder niedrige Korrelation zu haben? Ändert sich das, wenn du eine feinere Messung anstelle von `Monat` verwendest, z. B. *Tag des Jahres* (d. h. Anzahl der Tage seit Jahresbeginn)?

Im folgenden Code nehmen wir an, dass wir die Daten bereinigt haben und einen DataFrame namens `new_pumpkins` erhalten haben, ähnlich dem folgenden:

ID | Monat | TagDesJahres | Sorte | Stadt | Verpackung | Niedriger Preis | Hoher Preis | Preis
---|-------|--------------|-------|-------|------------|-----------------|-------------|------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Der Code zur Bereinigung der Daten ist verfügbar in [`notebook.ipynb`](notebook.ipynb). Wir haben die gleichen Bereinigungsschritte wie in der vorherigen Lektion durchgeführt und die Spalte `TagDesJahres` mit folgendem Ausdruck berechnet: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Jetzt, da du die Mathematik hinter der linearen Regression verstehst, lass uns ein Regressionsmodell erstellen, um zu sehen, ob wir vorhersagen können, welches Kürbispaket die besten Kürbispreise hat. Jemand, der Kürbisse für einen Feiertags-Kürbisstand kauft, könnte diese Informationen benötigen, um seine Einkäufe von Kürbispaketen für den Stand zu optimieren.

## Suche nach Korrelation

[![ML für Anfänger - Suche nach Korrelation: Der Schlüssel zur linearen Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML für Anfänger - Suche nach Korrelation: Der Schlüssel zur linearen Regression")

> 🎥 Klicke auf das Bild oben für eine kurze Videoübersicht zur Korrelation.

Aus der vorherigen Lektion hast du wahrscheinlich gesehen, dass der Durchschnittspreis für verschiedene Monate wie folgt aussieht:

<img alt="Durchschnittspreis nach Monat" src="../2-Data/images/barchart.png" width="50%"/>

Dies deutet darauf hin, dass es eine gewisse Korrelation geben sollte, und wir können versuchen, ein lineares Regressionsmodell zu trainieren, um die Beziehung zwischen `Monat` und `Preis` oder zwischen `TagDesJahres` und `Preis` vorherzusagen. Hier ist das Streudiagramm, das die letztere Beziehung zeigt:

<img alt="Streudiagramm von Preis vs. Tag des Jahres" src="images/scatter-dayofyear.png" width="50%" /> 

Lass uns sehen, ob es eine Korrelation gibt, indem wir die Funktion `corr` verwenden:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Es sieht so aus, als ob die Korrelation ziemlich gering ist, -0.15 für `Monat` und -0.17 für `TagDesJahres`, aber es könnte eine andere wichtige Beziehung geben. Es scheint, dass es verschiedene Preiscluster gibt, die mit verschiedenen Kürbissorten korrespondieren. Um diese Hypothese zu bestätigen, lass uns jede Kürbiskategorie mit einer anderen Farbe darstellen. Indem wir einen `ax`-Parameter an die `scatter`-Plot-Funktion übergeben, können wir alle Punkte im selben Diagramm darstellen:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Streudiagramm von Preis vs. Tag des Jahres" src="images/scatter-dayofyear-color.png" width="50%" /> 

Unsere Untersuchung legt nahe, dass die Sorte einen größeren Einfluss auf den Gesamtpreis hat als das tatsächliche Verkaufsdatum. Wir können dies mit einem Balkendiagramm sehen:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Balkendiagramm von Preis vs. Sorte" src="images/price-by-variety.png" width="50%" /> 

Lass uns für den Moment nur auf eine Kürbissorte, den 'Pie Type', fokussieren und sehen, welchen Einfluss das Datum auf den Preis hat:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Streudiagramm von Preis vs. Tag des Jahres" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Wenn wir jetzt die Korrelation zwischen `Preis` und `TagDesJahres` mit der Funktion `corr` berechnen, erhalten wir etwa `-0.27` - was bedeutet, dass das Trainieren eines Vorhersagemodells sinnvoll ist.

> Bevor wir ein lineares Regressionsmodell trainieren, ist es wichtig sicherzustellen, dass unsere Daten sauber sind. Lineare Regression funktioniert nicht gut mit fehlenden Werten, daher macht es Sinn, alle leeren Zellen zu entfernen:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Eine andere Herangehensweise wäre, diese leeren Werte mit den Mittelwerten der entsprechenden Spalte zu füllen.

## Einfache lineare Regression

[![ML für Anfänger - Lineare und polynomiale Regression mit Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML für Anfänger - Lineare und polynomiale Regression mit Scikit-learn")

> 🎥 Klicke auf das Bild oben für eine kurze Videoübersicht zur linearen und polynomialen Regression.

Um unser lineares Regressionsmodell zu trainieren, verwenden wir die **Scikit-learn**-Bibliothek.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Wir beginnen damit, Eingabewerte (Features) und die erwartete Ausgabe (Label) in separate numpy-Arrays zu trennen:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Beachte, dass wir `reshape` auf die Eingabedaten anwenden mussten, damit das Paket für lineare Regression sie korrekt versteht. Lineare Regression erwartet ein 2D-Array als Eingabe, wobei jede Zeile des Arrays einem Vektor von Eingabefeatures entspricht. In unserem Fall, da wir nur eine Eingabe haben, benötigen wir ein Array mit der Form N×1, wobei N die Datensatzgröße ist.

Dann müssen wir die Daten in Trainings- und Testdatensätze aufteilen, damit wir unser Modell nach dem Training validieren können:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Das eigentliche Training des linearen Regressionsmodells dauert nur zwei Codezeilen. Wir definieren das `LinearRegression`-Objekt und passen es mit der Methode `fit` an unsere Daten an:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Das `LinearRegression`-Objekt enthält nach dem `fit`-Vorgang alle Koeffizienten der Regression, die über die Eigenschaft `.coef_` abgerufen werden können. In unserem Fall gibt es nur einen Koeffizienten, der etwa `-0.017` sein sollte. Das bedeutet, dass die Preise mit der Zeit leicht sinken, aber nicht zu stark, etwa um 2 Cent pro Tag. Wir können auch den Schnittpunkt der Regression mit der Y-Achse über `lin_reg.intercept_` abrufen - er wird in unserem Fall etwa `21` sein, was den Preis zu Jahresbeginn angibt.

Um zu sehen, wie genau unser Modell ist, können wir die Preise auf einem Testdatensatz vorhersagen und dann messen, wie nah unsere Vorhersagen an den erwarteten Werten liegen. Dies kann mit der Mean-Square-Error (MSE)-Metrik erfolgen, die den Mittelwert aller quadrierten Unterschiede zwischen erwartetem und vorhergesagtem Wert darstellt.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Unser Fehler scheint sich auf zwei Punkte zu konzentrieren, was etwa 17 % entspricht. Nicht besonders gut. Ein weiterer Indikator für die Modellqualität ist der **Bestimmtheitskoeffizient**, der wie folgt berechnet werden kann:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Wenn der Wert 0 ist, bedeutet das, dass das Modell die Eingabedaten nicht berücksichtigt und als *schlechtester linearer Prädiktor* agiert, der einfach den Mittelwert des Ergebnisses darstellt. Der Wert 1 bedeutet, dass wir alle erwarteten Ausgaben perfekt vorhersagen können. In unserem Fall liegt der Koeffizient bei etwa 0,06, was ziemlich niedrig ist.

Wir können auch die Testdaten zusammen mit der Regressionslinie darstellen, um besser zu sehen, wie die Regression in unserem Fall funktioniert:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Lineare Regression" src="images/linear-results.png" width="50%" />

## Polynomiale Regression

Eine andere Art der linearen Regression ist die polynomiale Regression. Während es manchmal eine lineare Beziehung zwischen Variablen gibt – je größer der Kürbis im Volumen, desto höher der Preis – können diese Beziehungen manchmal nicht als Ebene oder gerade Linie dargestellt werden.

✅ Hier sind [einige weitere Beispiele](https://online.stat.psu.edu/stat501/lesson/9/9.8) für Daten, die polynomiale Regression verwenden könnten.

Betrachten Sie erneut die Beziehung zwischen Datum und Preis. Sieht dieses Streudiagramm so aus, als sollte es unbedingt durch eine gerade Linie analysiert werden? Können Preise nicht schwanken? In diesem Fall können Sie polynomiale Regression ausprobieren.

✅ Polynome sind mathematische Ausdrücke, die aus einer oder mehreren Variablen und Koeffizienten bestehen können.

Die polynomiale Regression erstellt eine gekrümmte Linie, um nichtlineare Daten besser anzupassen. In unserem Fall sollten wir, wenn wir eine quadrierte `DayOfYear`-Variable in die Eingabedaten aufnehmen, unsere Daten mit einer parabolischen Kurve anpassen können, die zu einem bestimmten Zeitpunkt im Jahr ein Minimum erreicht.

Scikit-learn enthält eine hilfreiche [Pipeline-API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), um verschiedene Schritte der Datenverarbeitung zu kombinieren. Eine **Pipeline** ist eine Kette von **Schätzern**. In unserem Fall erstellen wir eine Pipeline, die zuerst polynomiale Merkmale zu unserem Modell hinzufügt und dann die Regression trainiert:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Die Verwendung von `PolynomialFeatures(2)` bedeutet, dass wir alle Polynome zweiten Grades aus den Eingabedaten einbeziehen. In unserem Fall bedeutet das einfach `DayOfYear`<sup>2</sup>, aber bei zwei Eingabevariablen X und Y fügt dies X<sup>2</sup>, XY und Y<sup>2</sup> hinzu. Wir können auch Polynome höheren Grades verwenden, wenn wir möchten.

Pipelines können genauso verwendet werden wie das ursprüngliche `LinearRegression`-Objekt, d. h. wir können die Pipeline `fit`ten und dann `predict` verwenden, um die Vorhersageergebnisse zu erhalten. Hier ist das Diagramm, das Testdaten und die Annäherungskurve zeigt:

<img alt="Polynomiale Regression" src="images/poly-results.png" width="50%" />

Mit der polynomialen Regression können wir einen etwas niedrigeren MSE und einen höheren Bestimmtheitskoeffizienten erzielen, aber nicht signifikant. Wir müssen andere Merkmale berücksichtigen!

> Sie können sehen, dass die minimalen Kürbispreise irgendwo um Halloween beobachtet werden. Wie können Sie das erklären?

🎃 Herzlichen Glückwunsch, Sie haben gerade ein Modell erstellt, das helfen kann, den Preis von Kürbissen für Kuchen vorherzusagen. Sie können wahrscheinlich dasselbe Verfahren für alle Kürbissorten wiederholen, aber das wäre mühsam. Lernen wir jetzt, wie man Kürbissorten in unser Modell einbezieht!

## Kategorische Merkmale

In der idealen Welt möchten wir in der Lage sein, Preise für verschiedene Kürbissorten mit demselben Modell vorherzusagen. Die Spalte `Variety` unterscheidet sich jedoch etwas von Spalten wie `Month`, da sie nicht-numerische Werte enthält. Solche Spalten werden als **kategorisch** bezeichnet.

[![ML für Anfänger - Kategorische Merkmalsvorhersagen mit linearer Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML für Anfänger - Kategorische Merkmalsvorhersagen mit linearer Regression")

> 🎥 Klicken Sie auf das Bild oben für eine kurze Videoübersicht zur Verwendung kategorischer Merkmale.

Hier können Sie sehen, wie der Durchschnittspreis von der Sorte abhängt:

<img alt="Durchschnittspreis nach Sorte" src="images/price-by-variety.png" width="50%" />

Um die Sorte zu berücksichtigen, müssen wir sie zuerst in numerische Form umwandeln, oder **codieren**. Es gibt mehrere Möglichkeiten, dies zu tun:

* Einfache **numerische Codierung** erstellt eine Tabelle mit verschiedenen Sorten und ersetzt dann den Sortennamen durch einen Index in dieser Tabelle. Dies ist keine gute Idee für die lineare Regression, da die lineare Regression den tatsächlichen numerischen Wert des Index nimmt und ihn mit einem Koeffizienten multipliziert, um ihn zum Ergebnis hinzuzufügen. In unserem Fall ist die Beziehung zwischen der Indexnummer und dem Preis eindeutig nicht linear, selbst wenn wir sicherstellen, dass die Indizes in einer bestimmten Reihenfolge angeordnet sind.
* **One-hot-Codierung** ersetzt die Spalte `Variety` durch 4 verschiedene Spalten, eine für jede Sorte. Jede Spalte enthält `1`, wenn die entsprechende Zeile einer bestimmten Sorte entspricht, und `0` andernfalls. Das bedeutet, dass es in der linearen Regression vier Koeffizienten gibt, einen für jede Kürbissorte, die für den "Startpreis" (oder eher "Zusatzpreis") für diese bestimmte Sorte verantwortlich sind.

Der folgende Code zeigt, wie wir eine Sorte one-hot codieren können:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

Um die lineare Regression mit einer one-hot codierten Sorte als Eingabe zu trainieren, müssen wir nur die `X`- und `y`-Daten korrekt initialisieren:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Der Rest des Codes ist derselbe wie der, den wir oben verwendet haben, um die lineare Regression zu trainieren. Wenn Sie es ausprobieren, werden Sie sehen, dass der mittlere quadratische Fehler ungefähr gleich bleibt, aber wir erhalten einen viel höheren Bestimmtheitskoeffizienten (~77 %). Um noch genauere Vorhersagen zu erhalten, können wir mehr kategorische Merkmale sowie numerische Merkmale wie `Month` oder `DayOfYear` berücksichtigen. Um ein großes Array von Merkmalen zu erhalten, können wir `join` verwenden:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Hier berücksichtigen wir auch `City` und `Package`-Typ, was uns einen MSE von 2,84 (10 %) und einen Bestimmtheitskoeffizienten von 0,94 gibt!

## Alles zusammenführen

Um das beste Modell zu erstellen, können wir kombinierte (one-hot codierte kategorische + numerische) Daten aus dem obigen Beispiel zusammen mit polynomialer Regression verwenden. Hier ist der vollständige Code zu Ihrer Bequemlichkeit:

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Dies sollte uns den besten Bestimmtheitskoeffizienten von fast 97 % und MSE=2,23 (~8 % Vorhersagefehler) geben.

| Modell | MSE | Bestimmtheitskoeffizient |
|-------|-----|---------------------------|
| `DayOfYear` Linear | 2,77 (17,2 %) | 0,07 |
| `DayOfYear` Polynomial | 2,73 (17,0 %) | 0,08 |
| `Variety` Linear | 5,24 (19,7 %) | 0,77 |
| Alle Merkmale Linear | 2,84 (10,5 %) | 0,94 |
| Alle Merkmale Polynomial | 2,23 (8,25 %) | 0,97 |

🏆 Gut gemacht! Sie haben in einer Lektion vier Regressionsmodelle erstellt und die Modellqualität auf 97 % verbessert. Im letzten Abschnitt zur Regression lernen Sie die logistische Regression kennen, um Kategorien zu bestimmen.

---
## 🚀 Herausforderung

Testen Sie mehrere verschiedene Variablen in diesem Notebook, um zu sehen, wie die Korrelation mit der Modellgenauigkeit zusammenhängt.

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/14/)

## Überprüfung & Selbststudium

In dieser Lektion haben wir über lineare Regression gelernt. Es gibt andere wichtige Arten der Regression. Lesen Sie über Stepwise-, Ridge-, Lasso- und Elasticnet-Techniken. Ein guter Kurs, um mehr zu lernen, ist der [Stanford Statistical Learning Kurs](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Aufgabe

[Erstellen Sie ein Modell](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.