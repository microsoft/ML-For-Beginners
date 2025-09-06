<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T21:13:58+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "no"
}
-->
# Kom i gang med Python og Scikit-learn for regresjonsmodeller

![Oppsummering av regresjoner i en sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne leksjonen er også tilgjengelig i R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introduksjon

I disse fire leksjonene vil du lære hvordan du bygger regresjonsmodeller. Vi skal snart diskutere hva disse brukes til. Men før du gjør noe som helst, sørg for at du har de riktige verktøyene på plass for å starte prosessen!

I denne leksjonen vil du lære å:

- Konfigurere datamaskinen din for lokale maskinlæringsoppgaver.
- Jobbe med Jupyter-notatbøker.
- Bruke Scikit-learn, inkludert installasjon.
- Utforske lineær regresjon gjennom en praktisk øvelse.

## Installasjoner og konfigurasjoner

[![ML for nybegynnere - Sett opp verktøyene dine for å bygge maskinlæringsmodeller](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML for nybegynnere - Sett opp verktøyene dine for å bygge maskinlæringsmodeller")

> 🎥 Klikk på bildet over for en kort video som viser hvordan du konfigurerer datamaskinen din for maskinlæring.

1. **Installer Python**. Sørg for at [Python](https://www.python.org/downloads/) er installert på datamaskinen din. Du vil bruke Python til mange oppgaver innen datavitenskap og maskinlæring. De fleste datamaskiner har allerede en Python-installasjon. Det finnes også nyttige [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) som kan gjøre oppsettet enklere for noen brukere.

   Noen bruksområder for Python krever én versjon av programvaren, mens andre krever en annen versjon. Derfor er det nyttig å jobbe i et [virtuelt miljø](https://docs.python.org/3/library/venv.html).

2. **Installer Visual Studio Code**. Sørg for at Visual Studio Code er installert på datamaskinen din. Følg disse instruksjonene for å [installere Visual Studio Code](https://code.visualstudio.com/) for grunnleggende installasjon. Du skal bruke Python i Visual Studio Code i dette kurset, så det kan være lurt å friske opp hvordan du [konfigurerer Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) for Python-utvikling.

   > Bli komfortabel med Python ved å jobbe gjennom denne samlingen av [Learn-moduler](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Sett opp Python med Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Sett opp Python med Visual Studio Code")
   >
   > 🎥 Klikk på bildet over for en video: bruk Python i VS Code.

3. **Installer Scikit-learn** ved å følge [disse instruksjonene](https://scikit-learn.org/stable/install.html). Siden du må sørge for at du bruker Python 3, anbefales det at du bruker et virtuelt miljø. Merk at hvis du installerer dette biblioteket på en M1 Mac, finnes det spesielle instruksjoner på siden som er lenket over.

4. **Installer Jupyter Notebook**. Du må [installere Jupyter-pakken](https://pypi.org/project/jupyter/).

## Ditt ML-utviklingsmiljø

Du skal bruke **notatbøker** for å utvikle Python-koden din og lage maskinlæringsmodeller. Denne typen filer er et vanlig verktøy for dataforskere, og de kan identifiseres ved suffikset eller filtypen `.ipynb`.

Notatbøker er et interaktivt miljø som lar utvikleren både kode og legge til notater og skrive dokumentasjon rundt koden, noe som er ganske nyttig for eksperimentelle eller forskningsorienterte prosjekter.

[![ML for nybegynnere - Sett opp Jupyter-notatbøker for å begynne å bygge regresjonsmodeller](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML for nybegynnere - Sett opp Jupyter-notatbøker for å begynne å bygge regresjonsmodeller")

> 🎥 Klikk på bildet over for en kort video som viser denne øvelsen.

### Øvelse - jobb med en notatbok

I denne mappen finner du filen _notebook.ipynb_.

1. Åpne _notebook.ipynb_ i Visual Studio Code.

   En Jupyter-server vil starte med Python 3+. Du vil finne områder i notatboken som kan `kjøres`, altså kodeblokker. Du kan kjøre en kodeblokk ved å velge ikonet som ser ut som en avspillingsknapp.

2. Velg `md`-ikonet og legg til litt markdown, og følgende tekst: **# Velkommen til din notatbok**.

   Deretter legger du til litt Python-kode.

3. Skriv **print('hello notebook')** i kodeblokken.
4. Velg pilen for å kjøre koden.

   Du bør se den utskrevne meldingen:

    ```output
    hello notebook
    ```

![VS Code med en åpen notatbok](../../../../2-Regression/1-Tools/images/notebook.jpg)

Du kan blande koden din med kommentarer for å selv-dokumentere notatboken.

✅ Tenk et øyeblikk på hvor forskjellig arbeidsmiljøet til en webutvikler er sammenlignet med en dataforsker.

## Kom i gang med Scikit-learn

Nå som Python er satt opp i ditt lokale miljø, og du er komfortabel med Jupyter-notatbøker, la oss bli like komfortable med Scikit-learn (uttales `sci` som i `science`). Scikit-learn tilbyr en [omfattende API](https://scikit-learn.org/stable/modules/classes.html#api-ref) for å hjelpe deg med å utføre ML-oppgaver.

Ifølge deres [nettsted](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn er et åpen kildekode-bibliotek for maskinlæring som støtter både overvåket og ikke-overvåket læring. Det tilbyr også ulike verktøy for modelltilpasning, dataprosessering, modellvalg og evaluering, samt mange andre nyttige funksjoner."

I dette kurset vil du bruke Scikit-learn og andre verktøy for å bygge maskinlæringsmodeller for å utføre det vi kaller 'tradisjonelle maskinlæringsoppgaver'. Vi har bevisst unngått nevrale nettverk og dyp læring, da disse dekkes bedre i vårt kommende 'AI for nybegynnere'-pensum.

Scikit-learn gjør det enkelt å bygge modeller og evaluere dem for bruk. Det fokuserer primært på bruk av numeriske data og inneholder flere ferdiglagde datasett som kan brukes som læringsverktøy. Det inkluderer også forhåndsbygde modeller som studenter kan prøve. La oss utforske prosessen med å laste inn forhåndspakkede data og bruke en innebygd estimator for å lage vår første ML-modell med Scikit-learn ved hjelp av noen grunnleggende data.

## Øvelse - din første Scikit-learn-notatbok

> Denne opplæringen er inspirert av [eksempelet på lineær regresjon](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) på Scikit-learns nettsted.

[![ML for nybegynnere - Ditt første lineære regresjonsprosjekt i Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML for nybegynnere - Ditt første lineære regresjonsprosjekt i Python")

> 🎥 Klikk på bildet over for en kort video som viser denne øvelsen.

I _notebook.ipynb_-filen som er tilknyttet denne leksjonen, tøm alle cellene ved å trykke på 'søppelbøtte'-ikonet.

I denne delen skal du jobbe med et lite datasett om diabetes som er innebygd i Scikit-learn for læringsformål. Tenk deg at du ønsket å teste en behandling for diabetikere. Maskinlæringsmodeller kan hjelpe deg med å avgjøre hvilke pasienter som vil respondere bedre på behandlingen, basert på kombinasjoner av variabler. Selv en veldig grunnleggende regresjonsmodell, når den visualiseres, kan vise informasjon om variabler som kan hjelpe deg med å organisere dine teoretiske kliniske studier.

✅ Det finnes mange typer regresjonsmetoder, og hvilken du velger avhenger av spørsmålet du ønsker å besvare. Hvis du vil forutsi sannsynlig høyde for en person med en gitt alder, vil du bruke lineær regresjon, siden du søker en **numerisk verdi**. Hvis du er interessert i å finne ut om en type mat skal anses som vegansk eller ikke, ser du etter en **kategoriinndeling**, og da vil du bruke logistisk regresjon. Du vil lære mer om logistisk regresjon senere. Tenk litt på noen spørsmål du kan stille til data, og hvilken av disse metodene som ville være mest passende.

La oss komme i gang med denne oppgaven.

### Importer biblioteker

For denne oppgaven skal vi importere noen biblioteker:

- **matplotlib**. Et nyttig [verktøy for grafer](https://matplotlib.org/) som vi skal bruke til å lage en linjediagram.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) er et nyttig bibliotek for håndtering av numeriske data i Python.
- **sklearn**. Dette er [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)-biblioteket.

Importer noen biblioteker for å hjelpe med oppgavene dine.

1. Legg til imports ved å skrive følgende kode:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Over importerer du `matplotlib`, `numpy`, og du importerer `datasets`, `linear_model` og `model_selection` fra `sklearn`. `model_selection` brukes til å dele data i trenings- og testsett.

### Diabetes-datasettet

Det innebygde [diabetes-datasettet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) inkluderer 442 datasettprøver om diabetes, med 10 funksjonsvariabler, inkludert:

- age: alder i år
- bmi: kroppsmasseindeks
- bp: gjennomsnittlig blodtrykk
- s1 tc: T-celler (en type hvite blodceller)

✅ Dette datasettet inkluderer konseptet 'kjønn' som en funksjonsvariabel viktig for forskning på diabetes. Mange medisinske datasett inkluderer denne typen binær klassifisering. Tenk litt på hvordan slike kategoriseringer kan ekskludere visse deler av befolkningen fra behandlinger.

Nå, last inn X- og y-dataene.

> 🎓 Husk, dette er overvåket læring, og vi trenger et navngitt 'y'-mål.

I en ny kodecelle, last inn diabetes-datasettet ved å kalle `load_diabetes()`. Inputen `return_X_y=True` signaliserer at `X` vil være en datamatrise, og `y` vil være regresjonsmålet.

1. Legg til noen print-kommandoer for å vise formen på datamatrisen og dens første element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Det du får tilbake som svar, er en tuple. Det du gjør, er å tilordne de to første verdiene i tuplen til henholdsvis `X` og `y`. Lær mer [om tupler](https://wikipedia.org/wiki/Tuple).

    Du kan se at disse dataene har 442 elementer formet i matriser med 10 elementer:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Tenk litt på forholdet mellom dataene og regresjonsmålet. Lineær regresjon forutsier forholdet mellom funksjonen X og målvariabelen y. Kan du finne [målet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) for diabetes-datasettet i dokumentasjonen? Hva demonstrerer dette datasettet, gitt målet?

2. Velg deretter en del av dette datasettet for å plotte ved å velge den tredje kolonnen i datasettet. Du kan gjøre dette ved å bruke `:`-operatoren for å velge alle rader, og deretter velge den tredje kolonnen ved hjelp av indeksen (2). Du kan også omforme dataene til å være en 2D-matrise - som kreves for plotting - ved å bruke `reshape(n_rows, n_columns)`. Hvis en av parameterne er -1, beregnes den tilsvarende dimensjonen automatisk.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Når som helst, skriv ut dataene for å sjekke formen.

3. Nå som du har data klare til å bli plottet, kan du se om en maskin kan hjelpe med å bestemme en logisk inndeling mellom tallene i dette datasettet. For å gjøre dette, må du dele både dataene (X) og målet (y) i test- og treningssett. Scikit-learn har en enkel måte å gjøre dette på; du kan dele testdataene dine på et gitt punkt.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nå er du klar til å trene modellen din! Last inn den lineære regresjonsmodellen og tren den med X- og y-treningssettene dine ved å bruke `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` er en funksjon du vil se i mange ML-biblioteker som TensorFlow.

5. Deretter lager du en prediksjon ved hjelp av testdataene, ved å bruke funksjonen `predict()`. Dette vil brukes til å tegne linjen mellom datagruppene.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nå er det på tide å vise dataene i et diagram. Matplotlib er et veldig nyttig verktøy for denne oppgaven. Lag et spredningsdiagram av alle X- og y-testdataene, og bruk prediksjonen til å tegne en linje på det mest passende stedet mellom modellens datagrupper.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![et spredningsdiagram som viser datapunkter rundt diabetes](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Tenk litt over hva som skjer her. En rett linje går gjennom mange små datapunkter, men hva gjør den egentlig? Kan du se hvordan du bør kunne bruke denne linjen til å forutsi hvor et nytt, ukjent datapunkt bør passe i forhold til y-aksen i plottet? Prøv å sette ord på den praktiske bruken av denne modellen.

Gratulerer, du har bygget din første lineære regresjonsmodell, laget en prediksjon med den, og vist den i et plott!

---
## 🚀Utfordring

Plott en annen variabel fra dette datasettet. Hint: rediger denne linjen: `X = X[:,2]`. Gitt målet for dette datasettet, hva kan du oppdage om utviklingen av diabetes som sykdom?
## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

I denne opplæringen jobbet du med enkel lineær regresjon, i stedet for univariat eller multippel lineær regresjon. Les litt om forskjellene mellom disse metodene, eller ta en titt på [denne videoen](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Les mer om konseptet regresjon og tenk over hvilke typer spørsmål som kan besvares med denne teknikken. Ta denne [opplæringen](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) for å utdype forståelsen din.

## Oppgave

[Et annet datasett](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.