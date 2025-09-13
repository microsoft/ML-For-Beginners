<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T18:50:01+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "nl"
}
-->
# Aan de slag met Python en Scikit-learn voor regressiemodellen

![Samenvatting van regressies in een sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote door [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-quiz voor de les](https://ff-quizzes.netlify.app/en/ml/)

> ### [Deze les is beschikbaar in R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introductie

In deze vier lessen ontdek je hoe je regressiemodellen kunt bouwen. We zullen binnenkort bespreken waarvoor deze modellen worden gebruikt. Maar voordat je iets doet, zorg ervoor dat je de juiste tools hebt om aan de slag te gaan!

In deze les leer je:

- Je computer configureren voor lokale machine learning-taken.
- Werken met Jupyter-notebooks.
- Scikit-learn gebruiken, inclusief installatie.
- Lineaire regressie verkennen met een praktische oefening.

## Installaties en configuraties

[![ML voor beginners - Stel je tools in om Machine Learning-modellen te bouwen](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML voor beginners - Stel je tools in om Machine Learning-modellen te bouwen")

> 🎥 Klik op de afbeelding hierboven voor een korte video over het configureren van je computer voor ML.

1. **Installeer Python**. Zorg ervoor dat [Python](https://www.python.org/downloads/) op je computer is geïnstalleerd. Je zult Python gebruiken voor veel data science- en machine learning-taken. De meeste computersystemen hebben al een Python-installatie. Er zijn ook handige [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) beschikbaar om de installatie voor sommige gebruikers te vergemakkelijken.

   Sommige toepassingen van Python vereisen echter een specifieke versie van de software, terwijl andere een andere versie vereisen. Om deze reden is het handig om te werken binnen een [virtuele omgeving](https://docs.python.org/3/library/venv.html).

2. **Installeer Visual Studio Code**. Zorg ervoor dat Visual Studio Code op je computer is geïnstalleerd. Volg deze instructies om [Visual Studio Code te installeren](https://code.visualstudio.com/) voor de basisinstallatie. Je gaat Python gebruiken in Visual Studio Code tijdens deze cursus, dus het kan handig zijn om je kennis op te frissen over hoe je [Visual Studio Code configureert](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) voor Python-ontwikkeling.

   > Maak jezelf vertrouwd met Python door deze verzameling [Learn-modules](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) door te nemen.
   >
   > [![Python instellen met Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Python instellen met Visual Studio Code")
   >
   > 🎥 Klik op de afbeelding hierboven voor een video: Python gebruiken binnen VS Code.

3. **Installeer Scikit-learn** door de [instructies hier](https://scikit-learn.org/stable/install.html) te volgen. Omdat je Python 3 moet gebruiken, wordt aanbevolen om een virtuele omgeving te gebruiken. Let op, als je deze bibliotheek installeert op een M1 Mac, zijn er speciale instructies op de pagina hierboven.

4. **Installeer Jupyter Notebook**. Je moet het [Jupyter-pakket installeren](https://pypi.org/project/jupyter/).

## Je ML-ontwikkelomgeving

Je gaat **notebooks** gebruiken om je Python-code te ontwikkelen en machine learning-modellen te maken. Dit type bestand is een veelgebruikt hulpmiddel voor datawetenschappers en kan worden herkend aan hun extensie `.ipynb`.

Notebooks zijn een interactieve omgeving waarmee de ontwikkelaar zowel kan coderen als notities en documentatie rond de code kan toevoegen, wat erg handig is voor experimentele of onderzoeksgerichte projecten.

[![ML voor beginners - Stel Jupyter Notebooks in om regressiemodellen te bouwen](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML voor beginners - Stel Jupyter Notebooks in om regressiemodellen te bouwen")

> 🎥 Klik op de afbeelding hierboven voor een korte video over deze oefening.

### Oefening - werken met een notebook

In deze map vind je het bestand _notebook.ipynb_.

1. Open _notebook.ipynb_ in Visual Studio Code.

   Een Jupyter-server zal starten met Python 3+. Je zult gebieden in de notebook vinden die kunnen worden `uitgevoerd`, stukjes code. Je kunt een codeblok uitvoeren door het pictogram te selecteren dat eruitziet als een afspeelknop.

2. Selecteer het `md`-pictogram en voeg een beetje markdown toe, en de volgende tekst **# Welkom bij je notebook**.

   Voeg vervolgens wat Python-code toe.

3. Typ **print('hello notebook')** in het codeblok.
4. Selecteer de pijl om de code uit te voeren.

   Je zou de geprinte verklaring moeten zien:

    ```output
    hello notebook
    ```

![VS Code met een notebook geopend](../../../../2-Regression/1-Tools/images/notebook.jpg)

Je kunt je code afwisselen met opmerkingen om de notebook zelf te documenteren.

✅ Denk even na over hoe anders de werkomgeving van een webontwikkelaar is in vergelijking met die van een datawetenschapper.

## Aan de slag met Scikit-learn

Nu Python is ingesteld in je lokale omgeving en je vertrouwd bent met Jupyter-notebooks, laten we even vertrouwd raken met Scikit-learn (uitgesproken als `sci` zoals in `science`). Scikit-learn biedt een [uitgebreide API](https://scikit-learn.org/stable/modules/classes.html#api-ref) om je te helpen ML-taken uit te voeren.

Volgens hun [website](https://scikit-learn.org/stable/getting_started.html): "Scikit-learn is een open source machine learning-bibliotheek die zowel supervised als unsupervised learning ondersteunt. Het biedt ook verschillende tools voor modelaanpassing, gegevensvoorverwerking, modelselectie en evaluatie, en vele andere hulpmiddelen."

In deze cursus gebruik je Scikit-learn en andere tools om machine learning-modellen te bouwen voor wat we 'traditionele machine learning'-taken noemen. We hebben bewust neurale netwerken en deep learning vermeden, omdat deze beter worden behandeld in ons komende 'AI voor Beginners'-curriculum.

Scikit-learn maakt het eenvoudig om modellen te bouwen en te evalueren voor gebruik. Het richt zich voornamelijk op het gebruik van numerieke gegevens en bevat verschillende kant-en-klare datasets die kunnen worden gebruikt als leermiddelen. Het bevat ook vooraf gebouwde modellen die studenten kunnen proberen. Laten we het proces verkennen van het laden van vooraf verpakte gegevens en het gebruik van een ingebouwde estimator om een eerste ML-model te maken met Scikit-learn met enkele basisgegevens.

## Oefening - je eerste Scikit-learn notebook

> Deze tutorial is geïnspireerd door het [lineaire regressievoorbeeld](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) op de website van Scikit-learn.

[![ML voor beginners - Je eerste lineaire regressieproject in Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML voor beginners - Je eerste lineaire regressieproject in Python")

> 🎥 Klik op de afbeelding hierboven voor een korte video over deze oefening.

In het bestand _notebook.ipynb_ dat bij deze les hoort, verwijder je alle cellen door op het 'prullenbak'-pictogram te drukken.

In deze sectie werk je met een kleine dataset over diabetes die is ingebouwd in Scikit-learn voor leermiddelen. Stel je voor dat je een behandeling voor diabetespatiënten wilde testen. Machine learning-modellen kunnen je helpen bepalen welke patiënten beter op de behandeling zouden reageren, op basis van combinaties van variabelen. Zelfs een heel eenvoudig regressiemodel, wanneer gevisualiseerd, kan informatie tonen over variabelen die je zouden helpen je theoretische klinische proeven te organiseren.

✅ Er zijn veel soorten regressiemethoden, en welke je kiest hangt af van het antwoord dat je zoekt. Als je de waarschijnlijke lengte van een persoon van een bepaalde leeftijd wilt voorspellen, gebruik je lineaire regressie, omdat je op zoek bent naar een **numerieke waarde**. Als je wilt ontdekken of een type keuken als veganistisch moet worden beschouwd, zoek je naar een **categorie-indeling**, dus gebruik je logistische regressie. Je leert later meer over logistische regressie. Denk eens na over enkele vragen die je aan gegevens kunt stellen, en welke van deze methoden meer geschikt zou zijn.

Laten we aan de slag gaan met deze taak.

### Bibliotheken importeren

Voor deze taak importeren we enkele bibliotheken:

- **matplotlib**. Dit is een handige [grafiektool](https://matplotlib.org/) en we zullen het gebruiken om een lijnplot te maken.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) is een handige bibliotheek voor het omgaan met numerieke gegevens in Python.
- **sklearn**. Dit is de [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) bibliotheek.

Importeer enkele bibliotheken om je taken te ondersteunen.

1. Voeg imports toe door de volgende code te typen:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Hierboven importeer je `matplotlib`, `numpy` en je importeert `datasets`, `linear_model` en `model_selection` van `sklearn`. `model_selection` wordt gebruikt voor het splitsen van gegevens in trainings- en testsets.

### De diabetes-dataset

De ingebouwde [diabetes-dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) bevat 442 gegevensmonsters over diabetes, met 10 kenmerkvariabelen, waaronder:

- leeftijd: leeftijd in jaren
- bmi: body mass index
- bp: gemiddelde bloeddruk
- s1 tc: T-cellen (een type witte bloedcellen)

✅ Deze dataset bevat het concept 'geslacht' als een kenmerkvariabele die belangrijk is voor onderzoek naar diabetes. Veel medische datasets bevatten dit type binaire classificatie. Denk eens na over hoe categorisaties zoals deze bepaalde delen van de bevolking kunnen uitsluiten van behandelingen.

Laad nu de X- en y-gegevens.

> 🎓 Onthoud, dit is supervised learning, en we hebben een benoemde 'y'-doelvariabele nodig.

In een nieuwe codecel laad je de diabetes-dataset door `load_diabetes()` aan te roepen. De input `return_X_y=True` geeft aan dat `X` een gegevensmatrix zal zijn en `y` de regressiedoelvariabele.

1. Voeg enkele printopdrachten toe om de vorm van de gegevensmatrix en het eerste element ervan te tonen:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Wat je terugkrijgt als antwoord, is een tuple. Wat je doet, is de twee eerste waarden van de tuple toewijzen aan respectievelijk `X` en `y`. Leer meer [over tuples](https://wikipedia.org/wiki/Tuple).

    Je kunt zien dat deze gegevens 442 items bevatten, gevormd in arrays van 10 elementen:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Denk eens na over de relatie tussen de gegevens en de regressiedoelvariabele. Lineaire regressie voorspelt relaties tussen kenmerk X en doelvariabele y. Kun je de [doelvariabele](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) voor de diabetes-dataset vinden in de documentatie? Wat demonstreert deze dataset, gezien die doelvariabele?

2. Selecteer vervolgens een deel van deze dataset om te plotten door de 3e kolom van de dataset te selecteren. Je kunt dit doen door de `:`-operator te gebruiken om alle rijen te selecteren en vervolgens de 3e kolom te selecteren met behulp van de index (2). Je kunt de gegevens ook opnieuw vormgeven tot een 2D-array - zoals vereist voor het plotten - door `reshape(n_rows, n_columns)` te gebruiken. Als een van de parameters -1 is, wordt de overeenkomstige dimensie automatisch berekend.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Print op elk moment de gegevens om de vorm ervan te controleren.

3. Nu je gegevens klaar hebt om te plotten, kun je zien of een machine kan helpen een logische splitsing te bepalen tussen de cijfers in deze dataset. Om dit te doen, moet je zowel de gegevens (X) als de doelvariabele (y) splitsen in test- en trainingssets. Scikit-learn heeft een eenvoudige manier om dit te doen; je kunt je testgegevens op een bepaald punt splitsen.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nu ben je klaar om je model te trainen! Laad het lineaire regressiemodel en train het met je X- en y-trainingssets met behulp van `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` is een functie die je vaak zult zien in ML-bibliotheken zoals TensorFlow.

5. Maak vervolgens een voorspelling met behulp van testgegevens, met behulp van de functie `predict()`. Dit zal worden gebruikt om de lijn te tekenen tussen de gegevensgroepen.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nu is het tijd om de gegevens in een plot te tonen. Matplotlib is een zeer handig hulpmiddel voor deze taak. Maak een scatterplot van alle X- en y-testgegevens en gebruik de voorspelling om een lijn te tekenen op de meest geschikte plaats, tussen de gegevensgroepen van het model.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![een scatterplot die datapunten rond diabetes toont](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Denk eens na over wat hier gebeurt. Een rechte lijn loopt door veel kleine datapunten, maar wat doet die precies? Kun je zien hoe je deze lijn zou moeten kunnen gebruiken om te voorspellen waar een nieuw, onbekend datapunt zou moeten passen in relatie tot de y-as van de plot? Probeer in woorden uit te leggen wat het praktische nut van dit model is.

Gefeliciteerd, je hebt je eerste lineaire regressiemodel gebouwd, er een voorspelling mee gemaakt en deze weergegeven in een plot!

---
## 🚀Uitdaging

Plot een andere variabele uit deze dataset. Tip: bewerk deze regel: `X = X[:,2]`. Gezien de target van deze dataset, wat kun je ontdekken over de progressie van diabetes als ziekte?
## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

In deze tutorial heb je gewerkt met eenvoudige lineaire regressie, in plaats van univariate of multivariate lineaire regressie. Lees wat meer over de verschillen tussen deze methoden, of bekijk [deze video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Lees meer over het concept van regressie en denk na over wat voor soort vragen met deze techniek beantwoord kunnen worden. Volg [deze tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) om je begrip te verdiepen.

## Opdracht

[Een andere dataset](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.