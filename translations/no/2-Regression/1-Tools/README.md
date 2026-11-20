# Kom i gang med Python og Scikit-learn for regresjonsmodeller

![Oppsummering av regresjoner i en sketchnote](../../../../translated_images/no/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-forelesningsquiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne leksjonen er tilgjengelig i R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introduksjon

I disse fire leksjonene vil du oppdage hvordan du bygger regresjonsmodeller. Vi skal snart diskutere hva de brukes til. Men før du starter, sørg for at du har de riktige verktøyene på plass for å starte prosessen!

I denne leksjonen vil du lære hvordan du:

- Konfigurerer datamaskinen din for lokale maskinlæringsoppgaver.
- Arbeider med Jupyter Notebooks.
- Bruker Scikit-learn, inkludert installasjon.
- Utforsker lineær regresjon med en praktisk øvelse.

## Installasjoner og konfigurasjoner

[![ML for nybegynnere - Sett opp verktøyene dine klare til å bygge maskinlæringsmodeller](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML for nybegynnere -Sett opp verktøyene dine klare til å bygge Maskinlæringsmodeller")

> 🎥 Klikk på bildet over for en kort video som viser konfigurasjon av datamaskinen for ML.

1. **Installer Python**. Sørg for at [Python](https://www.python.org/downloads/) er installert på datamaskinen din. Du vil bruke Python til mange data science- og maskinlæringsoppgaver. De fleste datasystemer har allerede en Python-installasjon. Det finnes også nyttige [Python-kodepakker](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) som gjør oppsettet enklere for noen brukere.

   Noen bruksområder av Python krever imidlertid én versjon av programvaren, mens andre krever en annen versjon. Derfor er det nyttig å jobbe innenfor et [virtuelt miljø](https://docs.python.org/3/library/venv.html).

2. **Installer Visual Studio Code**. Sørg for at du har Visual Studio Code installert på datamaskinen din. Følg disse instruksjonene for å [installere Visual Studio Code](https://code.visualstudio.com/) for grunninstallasjonen. Du skal bruke Python i Visual Studio Code i dette kurset, så det kan være lurt å friske opp hvordan du [konfigurerer Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) for Python-utvikling.

   > Bli komfortabel med Python ved å jobbe deg gjennom denne samlingen av [Lær-moduler](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Sett opp Python med Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Sett opp Python med Visual Studio Code")
   >
   > 🎥 Klikk på bildet over for en video: bruk av Python i VS Code.

3. **Installer Scikit-learn**, ved å følge [disse instruksjonene](https://scikit-learn.org/stable/install.html). Siden du må være sikker på at du bruker Python 3, anbefales det å bruke et virtuelt miljø. Merk at hvis du installerer dette biblioteket på en M1 Mac, finnes det spesielle instruksjoner på siden som er lenket ovenfor.

1. **Installer Jupyter Notebook**. Du må [installere Jupyter-pakken](https://pypi.org/project/jupyter/).

## Ditt ML-utviklingsmiljø

Du skal bruke **notebooks** for å utvikle Python-kode og lage maskinlæringsmodeller. Denne filtypen er et vanlig verktøy for dataforskere, og de kan identifiseres ved sin suffiks eller filendelse `.ipynb`.

Notebooks er et interaktivt miljø som lar utvikleren både kode og legge til notater og skrive dokumentasjon rundt koden, noe som er svært nyttig for eksperimentelle eller forskningsorienterte prosjekter.

[![ML for nybegynnere - Sett opp Jupyter Notebooks for å begynne å bygge regresjonsmodeller](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML for nybegynnere - Sett opp Jupyter Notebooks for å begynne å bygge regresjonsmodeller")

> 🎥 Klikk på bildet over for en kort video hvor du jobber gjennom denne øvelsen.

### Øvelse - arbeid med en notebook

I denne mappen finner du filen _notebook.ipynb_.

1. Åpne _notebook.ipynb_ i Visual Studio Code.

   En Jupyter-server vil starte med Python 3+ aktivert. Du vil finne deler av notebooken som kan ` kjøres `, kodebiter. Du kan kjøre en kodeblokk ved å klikke på ikonet som ser ut som en avspillingsknapp.

1. Velg `md`-ikonet og legg til litt markdown med følgende tekst **# Velkommen til din notebook**.

   Legg deretter til litt Python-kode.

1. Skriv **print('hello notebook')** i kodeblokken.
1. Velg pilen for å kjøre koden.

   Du bør se den utskrevne setningen:

    ```output
    hello notebook
    ```

![VS Code med en åpen notebook](../../../../translated_images/no/notebook.4a3ee31f396b8832.webp)

Du kan veksle mellom kode og kommentarer for å selvdokumentere notebooken.

✅ Tenk et øyeblikk på hvor forskjellig en webutviklers arbeidsmiljø er sammenlignet med en dataforskers.

## Kom i gang med Scikit-learn

Nå som Python er satt opp i ditt lokale miljø, og du er komfortabel med Jupyter Notebooks, la oss bli like komfortable med Scikit-learn (uttales `sci` som i `science`). Scikit-learn tilbyr en [omfattende API](https://scikit-learn.org/stable/modules/classes.html#api-ref) som hjelper deg med å utføre ML-oppgaver.

Ifølge deres [nettside](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn er et åpen kildekode maskinlæringsbibliotek som støtter veiledet og ikke-veiledet læring. Det tilbyr også flere verktøy for modelltilpasning, datapreprosessering, modellvalg og evaluering, og mange andre verktøy."

I dette kurset skal du bruke Scikit-learn og andre verktøy for å bygge maskinlæringsmodeller for det vi kaller 'tradisjonelle maskinlæringsoppgaver'. Vi unngår bevisst nevrale nettverk og dyp læring, da disse dekkes bedre i vårt kommende 'AI for nybegynnere'-pensum.

Scikit-learn gjør det enkelt å bygge modeller og evaluere dem til bruk. Det fokuserer primært på numeriske data og inneholder flere ferdiglagde datasett som læringsverktøy. Det inkluderer også ferdigbygde modeller som studenter kan prøve. La oss utforske prosessen med å laste forhåndspakket data og bruke en innebygd estimator for å lage din første ML-modell med Scikit-learn med noen grunnleggende data.

## Øvelse - din første Scikit-learn-notebook

> Denne veiledningen er inspirert av [lineær regresjons-eksemplet](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) på Scikit-learns nettsted.


[![ML for nybegynnere - Ditt første lineære regresjonsprosjekt i Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML for nybegynnere - Ditt første lineære regresjonsprosjekt i Python")

> 🎥 Klikk på bildet over for en kort video hvor du jobber gjennom denne øvelsen.

I filen _notebook.ipynb_ knyttet til denne leksjonen, slett alle cellene ved å trykke på 'papirkurv'-ikonet.

I denne delen skal du arbeide med et lite datasett om diabetes som er innebygd i Scikit-learn for læringsformål. Tenk deg at du ønsker å teste en behandling for diabetikere. Maskinlæringsmodeller kan hjelpe deg å avgjøre hvilke pasienter som vil svare best på behandlingen, basert på kombinasjoner av variabler. Selv en veldig enkel regresjonsmodell, når den visualiseres, kan vise informasjon om variabler som kan hjelpe deg å organisere dine teoretiske kliniske studier.

✅ Det finnes mange typer regresjonsmetoder, og hvilken du velger avhenger av svaret du leter etter. Hvis du vil forutsi sannsynlig høyde for en person i en gitt alder, bruker du lineær regresjon, siden du søker en **numerisk verdi**. Hvis du er interessert i å finne ut om en type mat skal klassifiseres som vegansk eller ikke, søker du en **kategori-tilordning**, så du ville bruke logistisk regresjon. Du vil lære mer om logistisk regresjon senere. Tenk litt på spørsmål du kan stille data, og hvilken av disse metodene som passer best.

La oss komme i gang med denne oppgaven.

### Importer biblioteker

For denne oppgaven vil vi importere noen biblioteker:

- **matplotlib**. Det er et nyttig [grafikkverktøy](https://matplotlib.org/) som vi skal bruke for å lage en linjegraf.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) er et nyttig bibliotek for håndtering av numeriske data i Python.
- **sklearn**. Dette er [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)-biblioteket.

Importer noen biblioteker for å hjelpe med oppgavene dine.

1. Legg til importene ved å skrive følgende kode:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Ovenfor importerer du `matplotlib`, `numpy` og importerer `datasets`, `linear_model` og `model_selection` fra `sklearn`. `model_selection` brukes for å splitte data i trenings- og testsett.

### Diabetes-datasettet

Det innebygde [diabetes-datasettet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) inneholder 442 datapunkter om diabetes, med 10 forklaringsvariabler, noen av dem er:

- alder: alder i år
- bmi: kroppsmasseindeks
- bp: gjennomsnittlig blodtrykk
- s1 tc: T-celler (en type hvite blodceller)

✅ Dette datasettet inkluderer begrepet 'kjønn' som en forklaringsvariabel viktig for diabetesforskning. Mange medisinske datasett inneholder denne typen binær klassifisering. Tenk litt på hvordan slike kategoriseringer kan ekskludere deler av befolkningen fra behandlinger.

Nå, last inn X- og y-dataene.

> 🎓 Husk at dette er veiledet læring, og vi trenger et navngitt 'y'-mål.

I en ny kodecelle, last diabetes-datasettet ved å kalle `load_diabetes()`. Argumentet `return_X_y=True` signaliserer at `X` blir en datamatris, og `y` vil være regresjonsmålet.

1. Legg til noen print-kommandoer for å vise formen på datamatrisen og det første elementet:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Det du får tilbake som svar, er en tuppel. Du tildeler de to første verdiene i tuppelen til `X` og `y` henholdsvis. Lær mer [om tupler](https://wikipedia.org/wiki/Tuple).

    Du kan se at denne dataen har 442 elementer formet som arrayer med 10 elementer:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Tenk litt på forholdet mellom dataen og regresjonsmålet. Lineær regresjon forutsier forholdet mellom egenskap X og målet y. Kan du finne [målet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) for diabetes-datasettet i dokumentasjonen? Hva demonstrerer dette datasettet, gitt målet?

2. Velg så en del av dette datasettet for å lage en graf ved å velge 3. kolonne. Du kan gjøre dette med `:`-operatoren for å velge alle rader, og deretter velge kolonne 3 med indeks (2). Du kan også omforme dataen til en 2D-array - som kreves for plott - ved å bruke `reshape(n_rader, n_kolonner)`. Hvis en av parameterne er -1, beregnes den tilsvarende dimensjonen automatisk.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Skriv ut data når som helst for å sjekke formen på den.

3. Nå som du har data klar for plott, kan du se om en maskin kan hjelpe med å bestemme en logisk inndeling mellom tallene i datasettet. For dette må du dele både data (X) og mål (y) i test- og treningssett. Scikit-learn har en enkel måte å gjøre dette på; du kan dele testdataen på et gitt punkt.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nå er du klar til å trene modellen din! Last inn lineær regresjonsmodell og tren den med treningssettene X og y med `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` er en funksjon du finner i mange ML-biblioteker som TensorFlow

5. Lag deretter en prediksjon ved å bruke testdata med funksjonen `predict()`. Dette skal brukes til å tegne linjen mellom datagruppene.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nå er det tid for å vise dataen i en graf. Matplotlib er et veldig nyttig verktøy for denne oppgaven. Lag et scatterplot av alle X- og y-testdataene, og bruk prediksjonen til å tegne en linje på det mest passende stedet mellom modellens datagrupper.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![et scatterplot som viser datapunkter rundt diabetes](../../../../translated_images/no/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Tenk litt på hva som skjer her. En rett linje går gjennom mange små datapunkter, men hva gjør den egentlig? Kan du se hvordan du skal kunne bruke denne linjen til å forutsi hvor et nytt, ukjent datapunkt bør passe i forhold til plottets y-akse? Prøv å sette ord på den praktiske bruken av denne modellen.

Gratulerer, du har bygget din første lineære regresjonsmodell, laget en prediksjon med den, og vist den i en graf!

---
## 🚀Utfordring

Lag en graf for en annen variabel fra dette datasettet. Tips: endre denne linjen: `X = X[:,2]`. Med tanke på datasetts mål, hva kan du oppdage om utviklingen av diabetes som sykdom?
## [Post-forelesningsquiz](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

I denne veiledningen jobbet du med enkel lineær regresjon, i stedet for univariat eller multippel lineær regresjon. Les litt om forskjellene mellom disse metodene, eller ta en titt på [denne videoen](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Les mer om konseptet regresjon og tenk på hvilke typer spørsmål som kan besvares med denne teknikken. Ta denne [veiledningen](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) for å utdype din forståelse.

## Oppgave

[Et annet datasett](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfraskrivelse**:
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det opprinnelige dokumentet på originalspråket skal betraktes som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->