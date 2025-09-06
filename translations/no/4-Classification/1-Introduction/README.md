<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T21:55:45+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "no"
}
-->
# Introduksjon til klassifisering

I disse fire leksjonene skal du utforske et grunnleggende fokusområde innen klassisk maskinlæring - _klassifisering_. Vi skal gå gjennom bruken av ulike klassifiseringsalgoritmer med et datasett om alle de fantastiske kjøkkenene i Asia og India. Håper du er sulten!

![bare en klype!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Feir pan-asiatiske kjøkken i disse leksjonene! Bilde av [Jen Looper](https://twitter.com/jenlooper)

Klassifisering er en form for [supervised learning](https://wikipedia.org/wiki/Supervised_learning) som har mye til felles med regresjonsteknikker. Hvis maskinlæring handler om å forutsi verdier eller navn på ting ved hjelp av datasett, faller klassifisering generelt inn i to grupper: _binær klassifisering_ og _multiklasse klassifisering_.

[![Introduksjon til klassifisering](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introduksjon til klassifisering")

> 🎥 Klikk på bildet over for en video: MITs John Guttag introduserer klassifisering

Husk:

- **Lineær regresjon** hjalp deg med å forutsi forholdet mellom variabler og lage nøyaktige spådommer om hvor et nytt datapunkt ville falle i forhold til den linjen. For eksempel kunne du forutsi _hva prisen på et gresskar ville være i september vs. desember_.
- **Logistisk regresjon** hjalp deg med å oppdage "binære kategorier": ved dette prisnivået, _er dette gresskaret oransje eller ikke-oransje_?

Klassifisering bruker ulike algoritmer for å bestemme andre måter å avgjøre en datapunktopps etikett eller klasse. La oss jobbe med dette matlagingsdatasettet for å se om vi, ved å observere en gruppe ingredienser, kan avgjøre hvilket kjøkken de tilhører.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne leksjonen er tilgjengelig i R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introduksjon

Klassifisering er en av de grunnleggende aktivitetene for maskinlæringsforskere og dataforskere. Fra enkel klassifisering av en binær verdi ("er denne e-posten spam eller ikke?"), til kompleks bildeklassifisering og segmentering ved hjelp av datamaskinsyn, er det alltid nyttig å kunne sortere data i klasser og stille spørsmål om det.

For å formulere prosessen på en mer vitenskapelig måte, skaper klassifiseringsmetoden din en prediktiv modell som gjør det mulig å kartlegge forholdet mellom inputvariabler og outputvariabler.

![binær vs. multiklasse klassifisering](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binære vs. multiklasse problemer for klassifiseringsalgoritmer å håndtere. Infografikk av [Jen Looper](https://twitter.com/jenlooper)

Før vi starter prosessen med å rense dataene våre, visualisere dem og forberede dem for våre ML-oppgaver, la oss lære litt om de ulike måtene maskinlæring kan brukes til å klassifisere data.

Avledet fra [statistikk](https://wikipedia.org/wiki/Statistical_classification), bruker klassifisering med klassisk maskinlæring funksjoner som `smoker`, `weight` og `age` for å avgjøre _sannsynligheten for å utvikle X sykdom_. Som en supervised learning-teknikk, lik de regresjonsøvelsene du utførte tidligere, er dataene dine merket, og ML-algoritmene bruker disse merkene til å klassifisere og forutsi klasser (eller 'funksjoner') i et datasett og tilordne dem til en gruppe eller et utfall.

✅ Ta et øyeblikk til å forestille deg et datasett om matretter. Hva ville en multiklassemodell kunne svare på? Hva ville en binær modell kunne svare på? Hva om du ville avgjøre om en gitt matrett sannsynligvis bruker bukkehornkløver? Hva om du ville se om du, gitt en pose med stjerneanis, artisjokker, blomkål og pepperrot, kunne lage en typisk indisk rett?

[![Gale mysteriekurver](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Gale mysteriekurver")

> 🎥 Klikk på bildet over for en video. Hele premisset for showet 'Chopped' er 'mystery basket', der kokker må lage en rett ut av et tilfeldig utvalg ingredienser. Sikkert en ML-modell ville ha hjulpet!

## Hei 'klassifiserer'

Spørsmålet vi ønsker å stille til dette matlagingsdatasettet er faktisk et **multiklasse-spørsmål**, siden vi har flere potensielle nasjonale kjøkken å jobbe med. Gitt en gruppe ingredienser, hvilken av disse mange klassene passer dataene inn i?

Scikit-learn tilbyr flere forskjellige algoritmer for å klassifisere data, avhengig av hvilken type problem du vil løse. I de neste to leksjonene skal du lære om flere av disse algoritmene.

## Øvelse - rens og balanser dataene dine

Den første oppgaven, før du starter dette prosjektet, er å rense og **balansere** dataene dine for å få bedre resultater. Start med den tomme _notebook.ipynb_-filen i roten av denne mappen.

Det første du må installere er [imblearn](https://imbalanced-learn.org/stable/). Dette er en Scikit-learn-pakke som lar deg bedre balansere dataene (du vil lære mer om denne oppgaven om et øyeblikk).

1. For å installere `imblearn`, kjør `pip install`, slik:

    ```python
    pip install imblearn
    ```

1. Importer pakkene du trenger for å importere dataene dine og visualisere dem, og importer også `SMOTE` fra `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Nå er du klar til å lese og importere dataene.

1. Den neste oppgaven vil være å importere dataene:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Ved å bruke `read_csv()` vil innholdet i csv-filen _cusines.csv_ bli lest og plassert i variabelen `df`.

1. Sjekk formen på dataene:

    ```python
    df.head()
    ```

   De første fem radene ser slik ut:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Få informasjon om disse dataene ved å kalle `info()`:

    ```python
    df.info()
    ```

    Utdataene dine ligner:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Øvelse - lære om matretter

Nå begynner arbeidet å bli mer interessant. La oss oppdage fordelingen av data per kjøkken.

1. Plott dataene som stolper ved å kalle `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![fordeling av matdata](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Det er et begrenset antall kjøkken, men fordelingen av data er ujevn. Du kan fikse det! Før du gjør det, utforsk litt mer.

1. Finn ut hvor mye data som er tilgjengelig per kjøkken og skriv det ut:

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

    Utdataene ser slik ut:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Oppdage ingredienser

Nå kan du grave dypere i dataene og lære hva som er de typiske ingrediensene per kjøkken. Du bør rense ut gjentakende data som skaper forvirring mellom kjøkken, så la oss lære om dette problemet.

1. Lag en funksjon `create_ingredient()` i Python for å lage en ingrediens-datasett. Denne funksjonen vil starte med å fjerne en unyttig kolonne og sortere ingredienser etter antall:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Nå kan du bruke den funksjonen til å få en idé om de ti mest populære ingrediensene per kjøkken.

1. Kall `create_ingredient()` og plott det ved å kalle `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Gjør det samme for de japanske dataene:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japansk](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Nå for de kinesiske ingrediensene:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kinesisk](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Plott de indiske ingrediensene:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indisk](../../../../4-Classification/1-Introduction/images/indian.png)

1. Til slutt, plott de koreanske ingrediensene:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![koreansk](../../../../4-Classification/1-Introduction/images/korean.png)

1. Nå, fjern de vanligste ingrediensene som skaper forvirring mellom ulike kjøkken, ved å kalle `drop()`:

   Alle elsker ris, hvitløk og ingefær!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Balanser datasettet

Nå som du har renset dataene, bruk [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - for å balansere det.

1. Kall `fit_resample()`, denne strategien genererer nye prøver ved interpolasjon.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Ved å balansere dataene dine, vil du få bedre resultater når du klassifiserer dem. Tenk på en binær klassifisering. Hvis mesteparten av dataene dine er én klasse, vil en ML-modell forutsi den klassen oftere, bare fordi det er mer data for den. Å balansere dataene tar skjeve data og hjelper med å fjerne denne ubalansen.

1. Nå kan du sjekke antall etiketter per ingrediens:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Utdataene dine ser slik ut:

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

    Dataene er fine og rene, balanserte og veldig delikate!

1. Det siste trinnet er å lagre de balanserte dataene, inkludert etiketter og funksjoner, i et nytt datasett som kan eksporteres til en fil:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Du kan ta en siste titt på dataene ved å bruke `transformed_df.head()` og `transformed_df.info()`. Lagre en kopi av disse dataene for bruk i fremtidige leksjoner:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Denne ferske CSV-filen kan nå finnes i rotdata-mappen.

---

## 🚀Utfordring

Dette pensumet inneholder flere interessante datasett. Grav gjennom `data`-mappene og se om noen inneholder datasett som ville være passende for binær eller multiklasse klassifisering? Hvilke spørsmål ville du stille til dette datasettet?

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium

Utforsk SMOTEs API. Hvilke bruksområder er det best egnet for? Hvilke problemer løser det?

## Oppgave 

[Utforsk klassifiseringsmetoder](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.