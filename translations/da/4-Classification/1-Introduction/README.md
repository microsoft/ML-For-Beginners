<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T00:52:58+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "da"
}
-->
# Introduktion til klassifikation

I disse fire lektioner vil du udforske et grundlæggende fokusområde inden for klassisk maskinlæring - _klassifikation_. Vi vil gennemgå brugen af forskellige klassifikationsalgoritmer med et datasæt om alle de fantastiske køkkener fra Asien og Indien. Håber du er sulten!

![bare en knivspids!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Fejr pan-asiatiske køkkener i disse lektioner! Billede af [Jen Looper](https://twitter.com/jenlooper)

Klassifikation er en form for [supervised learning](https://wikipedia.org/wiki/Supervised_learning), der har meget til fælles med regressionsteknikker. Hvis maskinlæring handler om at forudsige værdier eller navne på ting ved hjælp af datasæt, falder klassifikation generelt i to grupper: _binær klassifikation_ og _multiklassifikation_.

[![Introduktion til klassifikation](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introduktion til klassifikation")

> 🎥 Klik på billedet ovenfor for en video: MIT's John Guttag introducerer klassifikation

Husk:

- **Lineær regression** hjalp dig med at forudsige forholdet mellem variabler og lave præcise forudsigelser om, hvor et nyt datapunkt ville falde i forhold til den linje. Så du kunne forudsige _hvad prisen på et græskar ville være i september vs. december_, for eksempel.
- **Logistisk regression** hjalp dig med at opdage "binære kategorier": ved dette prisniveau, _er dette græskar orange eller ikke-orange_?

Klassifikation bruger forskellige algoritmer til at bestemme andre måder at identificere en datapunkts label eller klasse. Lad os arbejde med dette køkkendatasæt for at se, om vi ved at observere en gruppe ingredienser kan bestemme dets oprindelseskøkken.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne lektion er tilgængelig i R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introduktion

Klassifikation er en af de grundlæggende aktiviteter for maskinlæringsforskere og dataanalytikere. Fra grundlæggende klassifikation af en binær værdi ("er denne e-mail spam eller ikke?") til kompleks billedklassifikation og segmentering ved hjælp af computer vision, er det altid nyttigt at kunne sortere data i klasser og stille spørgsmål til dem.

For at formulere processen på en mere videnskabelig måde skaber din klassifikationsmetode en forudsigelsesmodel, der gør det muligt for dig at kortlægge forholdet mellem inputvariabler og outputvariabler.

![binær vs. multiklassifikation](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binære vs. multiklasseproblemer, som klassifikationsalgoritmer skal håndtere. Infografik af [Jen Looper](https://twitter.com/jenlooper)

Før vi begynder processen med at rense vores data, visualisere dem og forberede dem til vores ML-opgaver, lad os lære lidt om de forskellige måder, maskinlæring kan bruges til at klassificere data.

Afledt af [statistik](https://wikipedia.org/wiki/Statistical_classification) bruger klassifikation med klassisk maskinlæring funktioner som `smoker`, `weight` og `age` til at bestemme _sandsynligheden for at udvikle X sygdom_. Som en supervised learning-teknik, der ligner de regression-øvelser, du udførte tidligere, er dine data mærket, og ML-algoritmerne bruger disse labels til at klassificere og forudsige klasser (eller 'features') af et datasæt og tildele dem til en gruppe eller et resultat.

✅ Tag et øjeblik til at forestille dig et datasæt om køkkener. Hvad ville en multiklassemodel kunne svare på? Hvad ville en binær model kunne svare på? Hvad hvis du ville afgøre, om et givet køkken sandsynligvis ville bruge bukkehorn? Hvad hvis du ville se, om du, givet en pose med stjerneanis, artiskokker, blomkål og peberrod, kunne lave en typisk indisk ret?

[![Skøre mystiske kurve](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Skøre mystiske kurve")

> 🎥 Klik på billedet ovenfor for en video. Hele præmissen for showet 'Chopped' er den 'mystiske kurv', hvor kokke skal lave en ret ud af et tilfældigt valg af ingredienser. En ML-model ville helt sikkert have hjulpet!

## Hej 'classifier'

Spørgsmålet, vi vil stille til dette køkkendatasæt, er faktisk et **multiklasse-spørgsmål**, da vi har flere potentielle nationale køkkener at arbejde med. Givet en batch af ingredienser, hvilken af disse mange klasser passer dataene til?

Scikit-learn tilbyder flere forskellige algoritmer til at klassificere data, afhængigt af hvilken type problem du vil løse. I de næste to lektioner vil du lære om flere af disse algoritmer.

## Øvelse - rens og balancer dine data

Den første opgave, før vi starter dette projekt, er at rense og **balancere** dine data for at opnå bedre resultater. Start med den tomme _notebook.ipynb_-fil i roden af denne mappe.

Det første, der skal installeres, er [imblearn](https://imbalanced-learn.org/stable/). Dette er en Scikit-learn-pakke, der giver dig mulighed for bedre at balancere dataene (du vil lære mere om denne opgave om et øjeblik).

1. For at installere `imblearn`, kør `pip install`, som vist her:

    ```python
    pip install imblearn
    ```

1. Importér de pakker, du har brug for til at importere dine data og visualisere dem, og importér også `SMOTE` fra `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Nu er du klar til at importere dataene.

1. Den næste opgave er at importere dataene:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Brug af `read_csv()` vil læse indholdet af csv-filen _cusines.csv_ og placere det i variablen `df`.

1. Tjek dataenes form:

    ```python
    df.head()
    ```

   De første fem rækker ser sådan ud:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Få information om disse data ved at kalde `info()`:

    ```python
    df.info()
    ```

    Din output ligner:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Øvelse - lær om køkkener

Nu begynder arbejdet at blive mere interessant. Lad os opdage fordelingen af data pr. køkken.

1. Plot dataene som søjler ved at kalde `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![fordeling af køkkendata](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Der er et begrænset antal køkkener, men fordelingen af data er ujævn. Det kan du rette! Før du gør det, skal du udforske lidt mere.

1. Find ud af, hvor mange data der er tilgængelige pr. køkken, og print det ud:

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

    Outputtet ser sådan ud:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Opdag ingredienser

Nu kan du grave dybere ned i dataene og lære, hvad de typiske ingredienser pr. køkken er. Du bør rense tilbagevendende data, der skaber forvirring mellem køkkener, så lad os lære om dette problem.

1. Opret en funktion `create_ingredient()` i Python for at oprette en ingrediens-datastruktur. Denne funktion starter med at fjerne en ubrugelig kolonne og sortere ingredienser efter deres antal:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Nu kan du bruge den funktion til at få en idé om de ti mest populære ingredienser pr. køkken.

1. Kald `create_ingredient()` og plot det ved at kalde `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Gør det samme for de japanske data:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japansk](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Nu for de kinesiske ingredienser:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kinesisk](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Plot de indiske ingredienser:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indisk](../../../../4-Classification/1-Introduction/images/indian.png)

1. Til sidst, plot de koreanske ingredienser:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![koreansk](../../../../4-Classification/1-Introduction/images/korean.png)

1. Nu skal du fjerne de mest almindelige ingredienser, der skaber forvirring mellem forskellige køkkener, ved at kalde `drop()`:

   Alle elsker ris, hvidløg og ingefær!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Balancer datasættet

Nu hvor du har renset dataene, skal du bruge [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - til at balancere dem.

1. Kald `fit_resample()`, denne strategi genererer nye prøver ved interpolation.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Ved at balancere dine data vil du få bedre resultater, når du klassificerer dem. Tænk på en binær klassifikation. Hvis de fleste af dine data er én klasse, vil en ML-model forudsige den klasse oftere, bare fordi der er flere data for den. Balancering af dataene fjerner denne skævhed.

1. Nu kan du tjekke antallet af labels pr. ingrediens:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Din output ser sådan ud:

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

    Dataene er nu rene, balancerede og meget lækre!

1. Det sidste trin er at gemme dine balancerede data, inklusive labels og features, i en ny datastruktur, der kan eksporteres til en fil:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Du kan tage et sidste kig på dataene ved hjælp af `transformed_df.head()` og `transformed_df.info()`. Gem en kopi af disse data til brug i fremtidige lektioner:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Denne friske CSV kan nu findes i roden af data-mappen.

---

## 🚀Udfordring

Dette pensum indeholder flere interessante datasæt. Gennemse `data`-mapperne og se, om nogle indeholder datasæt, der ville være passende til binær eller multiklasseklassifikation? Hvilke spørgsmål ville du stille til dette datasæt?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Udforsk SMOTEs API. Hvilke anvendelsesscenarier er det bedst egnet til? Hvilke problemer løser det?

## Opgave 

[Udforsk klassifikationsmetoder](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.