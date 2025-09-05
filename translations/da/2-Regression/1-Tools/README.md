<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-04T23:36:28+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "da"
}
-->
# Kom godt i gang med Python og Scikit-learn til regressionsmodeller

![Oversigt over regressioner i en sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote af [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne lektion er også tilgængelig i R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introduktion

I disse fire lektioner vil du lære, hvordan man bygger regressionsmodeller. Vi vil snart diskutere, hvad de bruges til. Men før du gør noget, skal du sørge for, at du har de rigtige værktøjer klar til at starte processen!

I denne lektion vil du lære at:

- Konfigurere din computer til lokale machine learning-opgaver.
- Arbejde med Jupyter-notebooks.
- Bruge Scikit-learn, herunder installation.
- Udforske lineær regression med en praktisk øvelse.

## Installationer og konfigurationer

[![ML for begyndere - Konfigurer dine værktøjer til at bygge Machine Learning-modeller](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML for begyndere - Konfigurer dine værktøjer til at bygge Machine Learning-modeller")

> 🎥 Klik på billedet ovenfor for en kort video om konfiguration af din computer til ML.

1. **Installer Python**. Sørg for, at [Python](https://www.python.org/downloads/) er installeret på din computer. Du vil bruge Python til mange data science- og machine learning-opgaver. De fleste computersystemer har allerede en Python-installation. Der findes også nyttige [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), som kan gøre opsætningen lettere for nogle brugere.

   Nogle anvendelser af Python kræver én version af softwaren, mens andre kræver en anden version. Derfor er det nyttigt at arbejde i et [virtuelt miljø](https://docs.python.org/3/library/venv.html).

2. **Installer Visual Studio Code**. Sørg for, at du har Visual Studio Code installeret på din computer. Følg disse instruktioner for at [installere Visual Studio Code](https://code.visualstudio.com/) til den grundlæggende installation. Du vil bruge Python i Visual Studio Code i dette kursus, så det kan være en god idé at opdatere din viden om, hvordan man [konfigurerer Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) til Python-udvikling.

   > Bliv fortrolig med Python ved at gennemgå denne samling af [Learn-moduler](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Opsæt Python med Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Opsæt Python med Visual Studio Code")
   >
   > 🎥 Klik på billedet ovenfor for en video: Brug Python i VS Code.

3. **Installer Scikit-learn** ved at følge [disse instruktioner](https://scikit-learn.org/stable/install.html). Da du skal sikre dig, at du bruger Python 3, anbefales det, at du bruger et virtuelt miljø. Bemærk, at hvis du installerer dette bibliotek på en M1 Mac, er der særlige instruktioner på den side, der er linket til ovenfor.

4. **Installer Jupyter Notebook**. Du skal [installere Jupyter-pakken](https://pypi.org/project/jupyter/).

## Dit ML-udviklingsmiljø

Du vil bruge **notebooks** til at udvikle din Python-kode og oprette machine learning-modeller. Denne type fil er et almindeligt værktøj for dataforskere, og de kan identificeres ved deres suffix eller filtype `.ipynb`.

Notebooks er et interaktivt miljø, der giver udvikleren mulighed for både at kode og tilføje noter samt skrive dokumentation omkring koden, hvilket er meget nyttigt for eksperimentelle eller forskningsorienterede projekter.

[![ML for begyndere - Opsæt Jupyter Notebooks for at begynde at bygge regressionsmodeller](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML for begyndere - Opsæt Jupyter Notebooks for at begynde at bygge regressionsmodeller")

> 🎥 Klik på billedet ovenfor for en kort video om denne øvelse.

### Øvelse - arbejde med en notebook

I denne mappe finder du filen _notebook.ipynb_.

1. Åbn _notebook.ipynb_ i Visual Studio Code.

   En Jupyter-server vil starte med Python 3+. Du vil finde områder i notebooken, der kan `køres`, stykker af kode. Du kan køre en kodeblok ved at vælge ikonet, der ligner en afspilningsknap.

2. Vælg `md`-ikonet og tilføj lidt markdown med følgende tekst **# Velkommen til din notebook**.

   Tilføj derefter noget Python-kode.

3. Skriv **print('hello notebook')** i kodeblokken.
4. Vælg pilen for at køre koden.

   Du bør se den udskrevne erklæring:

    ```output
    hello notebook
    ```

![VS Code med en notebook åben](../../../../2-Regression/1-Tools/images/notebook.jpg)

Du kan blande din kode med kommentarer for at selv-dokumentere notebooken.

✅ Tænk et øjeblik over, hvor forskelligt en webudviklers arbejdsmiljø er i forhold til en dataforskers.

## Kom godt i gang med Scikit-learn

Nu hvor Python er sat op i dit lokale miljø, og du er fortrolig med Jupyter-notebooks, lad os blive lige så fortrolige med Scikit-learn (udtales `sci` som i `science`). Scikit-learn tilbyder en [omfattende API](https://scikit-learn.org/stable/modules/classes.html#api-ref) til at hjælpe dig med at udføre ML-opgaver.

Ifølge deres [websted](https://scikit-learn.org/stable/getting_started.html) er "Scikit-learn et open source machine learning-bibliotek, der understøtter superviseret og usuperviseret læring. Det tilbyder også forskellige værktøjer til modeltilpasning, databehandling, modelvalg og evaluering samt mange andre funktioner."

I dette kursus vil du bruge Scikit-learn og andre værktøjer til at bygge machine learning-modeller til at udføre det, vi kalder 'traditionelle machine learning'-opgaver. Vi har bevidst undgået neurale netværk og deep learning, da de er bedre dækket i vores kommende 'AI for Beginners'-curriculum.

Scikit-learn gør det nemt at bygge modeller og evaluere dem til brug. Det fokuserer primært på brugen af numeriske data og indeholder flere færdiglavede datasæt til brug som læringsværktøjer. Det inkluderer også forudbyggede modeller, som studerende kan prøve. Lad os udforske processen med at indlæse forpakkede data og bruge en indbygget estimator til den første ML-model med Scikit-learn med nogle grundlæggende data.

## Øvelse - din første Scikit-learn notebook

> Denne tutorial er inspireret af [eksemplet på lineær regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) på Scikit-learns websted.

[![ML for begyndere - Dit første projekt med lineær regression i Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML for begyndere - Dit første projekt med lineær regression i Python")

> 🎥 Klik på billedet ovenfor for en kort video om denne øvelse.

I filen _notebook.ipynb_ tilknyttet denne lektion skal du rydde alle celler ved at trykke på 'skraldespand'-ikonet.

I denne sektion vil du arbejde med et lille datasæt om diabetes, som er indbygget i Scikit-learn til læringsformål. Forestil dig, at du ville teste en behandling for diabetiske patienter. Machine Learning-modeller kan hjælpe dig med at afgøre, hvilke patienter der vil reagere bedre på behandlingen baseret på kombinationer af variabler. Selv en meget grundlæggende regressionsmodel, når den visualiseres, kan vise information om variabler, der kan hjælpe dig med at organisere dine teoretiske kliniske forsøg.

✅ Der findes mange typer regressionsmetoder, og hvilken du vælger afhænger af det svar, du leder efter. Hvis du vil forudsige den sandsynlige højde for en person i en given alder, vil du bruge lineær regression, da du søger en **numerisk værdi**. Hvis du er interesseret i at finde ud af, om en type køkken skal betragtes som vegansk eller ej, leder du efter en **kategoriinddeling**, så du vil bruge logistisk regression. Du vil lære mere om logistisk regression senere. Tænk lidt over nogle spørgsmål, du kan stille til data, og hvilken af disse metoder der ville være mest passende.

Lad os komme i gang med denne opgave.

### Importer biblioteker

Til denne opgave vil vi importere nogle biblioteker:

- **matplotlib**. Det er et nyttigt [grafværktøj](https://matplotlib.org/), og vi vil bruge det til at oprette en linjeplot.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) er et nyttigt bibliotek til håndtering af numeriske data i Python.
- **sklearn**. Dette er [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)-biblioteket.

Importer nogle biblioteker til at hjælpe med dine opgaver.

1. Tilføj imports ved at skrive følgende kode:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Ovenfor importerer du `matplotlib`, `numpy`, og du importerer `datasets`, `linear_model` og `model_selection` fra `sklearn`. `model_selection` bruges til at opdele data i trænings- og test-sæt.

### Diabetes-datasættet

Det indbyggede [diabetes-datasæt](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) inkluderer 442 prøver af data omkring diabetes med 10 feature-variabler, nogle af dem inkluderer:

- age: alder i år
- bmi: body mass index
- bp: gennemsnitligt blodtryk
- s1 tc: T-celler (en type hvide blodceller)

✅ Dette datasæt inkluderer begrebet 'køn' som en feature-variabel, der er vigtig for forskning omkring diabetes. Mange medicinske datasæt inkluderer denne type binær klassifikation. Tænk lidt over, hvordan kategoriseringer som denne kan udelukke visse dele af befolkningen fra behandlinger.

Nu skal du indlæse X- og y-data.

> 🎓 Husk, dette er superviseret læring, og vi har brug for et navngivet 'y'-mål.

I en ny kodecelle skal du indlæse diabetes-datasættet ved at kalde `load_diabetes()`. Inputtet `return_X_y=True` signalerer, at `X` vil være en datamatrix, og `y` vil være regressionsmålet.

1. Tilføj nogle print-kommandoer for at vise formen på datamatricen og dens første element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Det, du får tilbage som svar, er en tuple. Det, du gør, er at tildele de to første værdier af tuplen til henholdsvis `X` og `y`. Læs mere [om tuples](https://wikipedia.org/wiki/Tuple).

    Du kan se, at disse data har 442 elementer formet i arrays med 10 elementer:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Tænk lidt over forholdet mellem dataene og regressionsmålet. Lineær regression forudsiger forholdet mellem feature X og målvariabel y. Kan du finde [målet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) for diabetes-datasættet i dokumentationen? Hvad demonstrerer dette datasæt, givet målet?

2. Vælg derefter en del af dette datasæt til at plotte ved at vælge den 3. kolonne i datasættet. Du kan gøre dette ved at bruge `:`-operatoren til at vælge alle rækker og derefter vælge den 3. kolonne ved hjælp af indekset (2). Du kan også omforme dataene til at være et 2D-array - som krævet for plotning - ved at bruge `reshape(n_rows, n_columns)`. Hvis en af parametrene er -1, beregnes den tilsvarende dimension automatisk.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Print dataene når som helst for at kontrollere deres form.

3. Nu hvor du har data klar til at blive plottet, kan du se, om en maskine kan hjælpe med at bestemme en logisk opdeling mellem tallene i dette datasæt. For at gøre dette skal du opdele både dataene (X) og målet (y) i test- og træningssæt. Scikit-learn har en ligetil måde at gøre dette på; du kan opdele dine testdata på et givet punkt.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nu er du klar til at træne din model! Indlæs den lineære regressionsmodel og træn den med dine X- og y-træningssæt ved hjælp af `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` er en funktion, du vil se i mange ML-biblioteker som TensorFlow.

5. Derefter skal du oprette en forudsigelse ved hjælp af testdata ved hjælp af funktionen `predict()`. Dette vil blive brugt til at tegne linjen mellem data-grupperne.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nu er det tid til at vise dataene i et plot. Matplotlib er et meget nyttigt værktøj til denne opgave. Opret et scatterplot af alle X- og y-testdata, og brug forudsigelsen til at tegne en linje på det mest passende sted mellem modellens data-grupperinger.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![et scatterplot, der viser datapunkter omkring diabetes](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Tænk lidt over, hvad der foregår her. En lige linje løber gennem mange små datapunkter, men hvad gør den egentlig? Kan du se, hvordan du burde kunne bruge denne linje til at forudsige, hvor et nyt, ukendt datapunkt skulle passe i forhold til plottets y-akse? Prøv at sætte ord på den praktiske anvendelse af denne model.

Tillykke, du har bygget din første lineære regressionsmodel, lavet en forudsigelse med den og vist den i et plot!

---
## 🚀Udfordring

Plot en anden variabel fra dette datasæt. Tip: rediger denne linje: `X = X[:,2]`. Givet målet for dette datasæt, hvad kan du opdage om udviklingen af diabetes som en sygdom?
## [Quiz efter forelæsning](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

I denne tutorial arbejdede du med simpel lineær regression, frem for univariat eller multivariat regression. Læs lidt om forskellene mellem disse metoder, eller se [denne video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Læs mere om begrebet regression og tænk over, hvilke slags spørgsmål der kan besvares med denne teknik. Tag [denne tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) for at uddybe din forståelse.

## Opgave

[Et andet datasæt](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på at opnå nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.