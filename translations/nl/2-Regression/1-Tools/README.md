# Aan de slag met Python en Scikit-learn voor regressiemodellen

![Samenvatting van regressies in een sketchnote](../../../../translated_images/nl/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote door [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-college quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Deze les is ook beschikbaar in R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introductie

In deze vier lessen ontdek je hoe je regressiemodellen bouwt. We zullen zo bespreken waar die voor zijn. Maar voordat je iets doet, zorg dat je de juiste tools klaar hebt staan om het proces te starten!

In deze les leer je hoe je:

- Je computer configureert voor lokale machine learning taken.
- Werkt met Jupyter Notebooks.
- Scikit-learn gebruikt, inclusief installatie.
- Lineaire regressie verkent met een praktische oefening.

## Installaties en configuraties

[![ML voor beginners - Configureer je tools klaar om Machine Learning modellen te bouwen](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML voor beginners - Configureer je tools klaar om Machine Learning modellen te bouwen")

> 🎥 Klik op de afbeelding hierboven voor een korte video waarin je door de configuratie van je computer voor ML wordt geleid.

1. **Installeer Python**. Zorg dat [Python](https://www.python.org/downloads/) op je computer is geïnstalleerd. Je gebruikt Python voor veel data science en machine learning taken. De meeste computersystemen hebben al een Python-installatie. Er zijn ook handige [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) beschikbaar om het opzetten voor sommige gebruikers makkelijker te maken.

   Sommige toepassingen van Python vereisen echter één versie van de software, terwijl andere een andere versie behoeven. Daarom is het handig om binnen een [virtuele omgeving](https://docs.python.org/3/library/venv.html) te werken.

2. **Installeer Visual Studio Code**. Zorg dat Visual Studio Code op je computer is geïnstalleerd. Volg deze instructies om [Visual Studio Code te installeren](https://code.visualstudio.com/) voor de basisinstallatie. Je gaat Python in Visual Studio Code gebruiken in deze cursus, dus je kunt ook de tijd nemen om te leren hoe je [Visual Studio Code configureert](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) voor Python-ontwikkeling.

   > Raken vertrouwd met Python door deze verzameling [Learn-modules](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) door te werken.
   >
   > [![Python instellen met Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Python instellen met Visual Studio Code")
   >
   > 🎥 Klik op de afbeelding hierboven voor een video: Python gebruiken binnen VS Code.

3. **Installeer Scikit-learn**, door de [instructies hier te volgen](https://scikit-learn.org/stable/install.html). Omdat je moet zorgen dat je Python 3 gebruikt, is het aan te raden om een virtuele omgeving te gebruiken. Let op: als je deze bibliotheek op een M1 Mac installeert, zijn er speciale instructies op de bovenstaande pagina.

1. **Installeer Jupyter Notebook**. Je moet het [Jupyter pakket installeren](https://pypi.org/project/jupyter/).

## Je ML ontwikkelomgeving

Je gaat **notebooks** gebruiken om je Python-code te ontwikkelen en machine learning modellen te maken. Dit type bestand is een veelgebruikt hulpmiddel voor datawetenschappers en ze zijn te herkennen aan hun suffix of extensie `.ipynb`.

Notebooks zijn een interactieve omgeving die de ontwikkelaar in staat stelt om zowel code te schrijven als notities en documentatie toe te voegen rondom de code, wat erg nuttig is voor experimentele of onderzoeksgerichte projecten.

[![ML voor beginners - Stel Jupyter Notebooks in om regressiemodellen te bouwen](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML voor beginners - Stel Jupyter Notebooks in om regressiemodellen te bouwen")

> 🎥 Klik op de afbeelding hierboven voor een korte video die deze oefening doorloopt.

### Oefening - werk met een notebook

In deze map vind je het bestand _notebook.ipynb_.

1. Open _notebook.ipynb_ in Visual Studio Code.

   Er zal een Jupyter server starten met Python 3+ actief. Je vindt gedeeltes van het notebook die je kunt `runnen`, stukjes code. Je kunt een codeblok uitvoeren door te klikken op het icoontje dat eruitziet als een afspeelknop.

1. Selecteer het `md` icoon en voeg wat markdown toe, en de volgende tekst **# Welkom in je notebook**.

   Voeg daarna wat Python code toe.

1. Typ **print('hello notebook')** in het codeblok.
1. Selecteer de pijl om de code te runnen.

   Je zou de volgende afgedrukte tekst moeten zien:

    ```output
    hello notebook
    ```

![VS Code met een geopend notebook](../../../../translated_images/nl/notebook.4a3ee31f396b8832.webp)

Je kunt je code afwisselen met commentaren om je notebook zelf te documenteren.

✅ Denk even na over hoe anders de werkomgeving van een webontwikkelaar is in vergelijking met die van een datawetenschapper.

## Aan de slag met Scikit-learn

Nu Python is ingesteld op je lokale omgeving en je vertrouwd bent met Jupyter Notebooks, laten we ook vertrouwd raken met Scikit-learn (spreek uit als `sci` zoals in `science`). Scikit-learn biedt een [uitgebreide API](https://scikit-learn.org/stable/modules/classes.html#api-ref) om je te helpen ML taken uit te voeren.

Volgens hun [website](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn is een open source machine learning bibliotheek die zowel supervised als unsupervised learning ondersteunt. Het biedt ook diverse tools voor model fitting, datapreprocessing, modelselectie en evaluatie, en vele andere hulpmiddelen."

In deze cursus gebruik je Scikit-learn en andere tools om machine learning modellen te bouwen waarmee we 'traditionele machine learning' taken uitvoeren. We hebben bewust neurale netwerken en deep learning vermeden, omdat die beter behandeld worden in ons aankomende curriculum 'AI voor Beginners'.

Scikit-learn maakt het gemakkelijk om modellen te bouwen en te evalueren. Het is vooral gericht op het gebruik van numerieke data en bevat verschillende kant-en-klare datasets om te gebruiken als leermiddelen. Het bevat ook vooraf gebouwde modellen voor studenten om te proberen. Laten we het proces verkennen van het laden van voorverpakte data en het gebruiken van een ingebouwde estimator om je eerste ML-model te maken met Scikit-learn met wat basisdata.

## Oefening - je eerste Scikit-learn notebook

> Deze tutorial is geïnspireerd door het [lineaire regressie voorbeeld](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) op de Scikit-learn website.


[![ML voor beginners - Je Eerste Lineaire Regressie Project in Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML voor beginners - Je Eerste Lineaire Regressie Project in Python")

> 🎥 Klik op de afbeelding hierboven voor een korte video waarin deze oefening wordt doorlopen.

In het bestand _notebook.ipynb_ dat bij deze les hoort, maak alle cellen leeg door op het 'prullenbak' icoon te klikken.

In deze sectie werk je met een kleine dataset over diabetes die ingebouwd is in Scikit-learn voor leermiddelen. Stel je voor dat je een behandeling voor diabetici wilde testen. Machine learning modellen kunnen je helpen bepalen welke patiënten beter op de behandeling reageren op basis van combinaties van variabelen. Zelfs een heel eenvoudig regressiemodel kan, wanneer gevisualiseerd, informatie tonen over variabelen die je zouden helpen je theoretische klinische onderzoeken te organiseren.

✅ Er zijn veel soorten regressiemethoden, en welke je kiest hangt af van het antwoord dat je zoekt. Wil je de waarschijnlijke lengte voorspellen voor een persoon van een bepaalde leeftijd, gebruik je lineaire regressie, omdat je een **numerieke waarde** zoekt. Ben je geïnteresseerd in te ontdekken of een keuken als veganistisch beschouwd moet worden, dan zoek je een **categorietoewijzing** en gebruik je logistische regressie. Je leert later meer over logistische regressie. Denk even na over de vragen die je aan data kunt stellen en welke van deze methoden het meest geschikt is.

Laten we aan deze taak beginnen.

### Bibliotheken importeren

Voor deze taak importeren we een paar bibliotheken:

- **matplotlib**. Het is een handige [grafiektool](https://matplotlib.org/) en we gebruiken het om een lijngrafiek te maken.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) is een handige bibliotheek voor het omgaan met numerieke data in Python.
- **sklearn**. Dit is de [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) bibliotheek.

Importeer enkele bibliotheken om je te helpen met je taken.

1. Voeg de imports toe door de volgende code te typen:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Boven importeer je `matplotlib`, `numpy` en je importeert `datasets`, `linear_model` en `model_selection` van `sklearn`. `model_selection` wordt gebruikt om data in trainings- en testsets te splitsen.

### De diabetes dataset

De ingebouwde [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) bevat 442 datasets over diabetes, met 10 feature-variabelen, waaronder:

- leeftijd: leeftijd in jaren
- bmi: body mass index
- bp: gemiddelde bloeddruk
- s1 tc: T-cellen (een type witte bloedcellen)

✅ Deze dataset bevat het concept 'geslacht' als een belangrijke featurevariabele in het onderzoek rondom diabetes. Veel medische datasets bevatten dit type binaire classificatie. Denk na over hoe zulke categoriseringen bepaalde delen van de bevolking kunnen uitsluiten van behandelingen.

Laad nu de X- en y-data.

> 🎓 Denk eraan, dit is supervised learning en we hebben een benoemde 'y' target nodig.

Laad in een nieuwe codecel de diabetes dataset door `load_diabetes()` aan te roepen. De invoer `return_X_y=True` betekent dat `X` een datamatrijs zal zijn, en `y` het regressiedoel.

1. Voeg printopdrachten toe om de vorm van de datamatrijs en het eerste element te tonen:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Wat je terugkrijgt als respons is een tuple. Wat je doet is de eerste twee waarden van de tuple toewijzen aan respectievelijk `X` en `y`. Lees meer [over tuples](https://wikipedia.org/wiki/Tuple).

    Je ziet dat deze data 442 items bevat, geordend in arrays van 10 elementen:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Denk even na over de relatie tussen de data en het regressiedoel. Lineaire regressie voorspelt relaties tussen feature X en targetvariabele y. Kun je het [target](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) voor de diabetes dataset in de documentatie vinden? Wat toont deze dataset, gegeven dat target?

2. Selecteer vervolgens een deel van deze dataset om te plotten door de 3e kolom van de dataset te nemen. Dit doe je door de `:` operator te gebruiken om alle rijen te selecteren en de 3e kolom te selecteren met index (2). Je kunt de data ook opnieuw vormgeven tot een 2D-array - zoals nodig is om te plotten - met `reshape(n_rows, n_columns)`. Als een van de parameters -1 is, wordt de overeenkomstige dimensie automatisch berekend.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Print de data op elk moment om de vorm te checken.

3. Nu je data klaar is om te plotten, kun je zien of een machine kan helpen bij het bepalen van een logische scheiding tussen de getallen in deze dataset. Hiervoor moet je zowel de data (X) als het target (y) splitsen in test- en trainingssets. Scikit-learn heeft hier een eenvoudige manier voor; je kunt je testdata op een punt splitsen.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nu ben je klaar om je model te trainen! Laad het lineaire regressiemodel en train het met je X- en y-trainingssets via `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` is een functie die je in veel ML-bibliotheken ziet, zoals TensorFlow

5. Maak vervolgens een voorspelling met de testdata, met de functie `predict()`. Dit wordt gebruikt om de lijn te trekken tussen de gegevensgroepen.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Tijd om de data weer te geven in een plot. Matplotlib is erg handig voor deze taak. Maak een scatterplot van alle X en y testdata en gebruik de voorspelling om een lijn te tekenen op de meest logische plek tussen de modelgegevensgroepen.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![een scatterplot die datapunten laat zien rondom diabetes](../../../../translated_images/nl/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Denk na over wat hier gebeurt. Een rechte lijn gaat door veel kleine stippen data heen, maar wat doet hij precies? Zie je hoe je deze lijn zou kunnen gebruiken om te voorspellen waar een nieuwe, ongeziene datapunt zou moeten passen ten opzichte van de y-as van de plot? Probeer in woorden te vatten wat het praktische nut is van dit model.

Gefeliciteerd, je hebt je eerste lineaire regressiemodel gebouwd, een voorspelling gemaakt en die geplot!

---
## 🚀Uitdaging

Plot een andere variabele uit deze dataset. Tip: bewerk deze regel: `X = X[:,2]`. Gegeven het target van deze dataset, wat kun je ontdekken over de progressie van diabetes als ziekte?
## [Post-college quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

In deze tutorial werkte je met eenvoudige lineaire regressie, in plaats van univariate of multivariate lineaire regressie. Lees wat over de verschillen tussen deze methoden, of bekijk [deze video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Lees meer over het concept regressie en denk na over welke soorten vragen met deze techniek beantwoord kunnen worden. Volg deze [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) om je begrip te verdiepen.

## Opdracht

[Een andere dataset](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Dit document is vertaald met behulp van de AI vertaaldienst [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u er rekening mee te houden dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->