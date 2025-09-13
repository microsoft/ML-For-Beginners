<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T19:58:06+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "nl"
}
-->
# Introductie tot classificatie

In deze vier lessen ga je een fundamenteel aspect van klassieke machine learning verkennen: _classificatie_. We zullen verschillende classificatie-algoritmen gebruiken met een dataset over de briljante keukens van Azië en India. Hopelijk heb je trek!

![een snufje!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Vier pan-Aziatische keukens in deze lessen! Afbeelding door [Jen Looper](https://twitter.com/jenlooper)

Classificatie is een vorm van [supervised learning](https://wikipedia.org/wiki/Supervised_learning) die veel gemeen heeft met regressietechnieken. Als machine learning draait om het voorspellen van waarden of namen door datasets te gebruiken, dan valt classificatie meestal in twee groepen: _binaire classificatie_ en _multiclass classificatie_.

[![Introductie tot classificatie](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introductie tot classificatie")

> 🎥 Klik op de afbeelding hierboven voor een video: MIT's John Guttag introduceert classificatie

Onthoud:

- **Lineaire regressie** hielp je relaties tussen variabelen te voorspellen en nauwkeurige voorspellingen te maken over waar een nieuw datapunt zou vallen in relatie tot die lijn. Zo kon je bijvoorbeeld voorspellen _wat de prijs van een pompoen zou zijn in september versus december_.
- **Logistische regressie** hielp je "binaire categorieën" ontdekken: bij dit prijsniveau, _is deze pompoen oranje of niet-oranje_?

Classificatie gebruikt verschillende algoritmen om andere manieren te bepalen waarop een label of klasse van een datapunt kan worden vastgesteld. Laten we met deze keuken-dataset werken om te zien of we, door een groep ingrediënten te observeren, de herkomst van de keuken kunnen bepalen.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Deze les is beschikbaar in R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introductie

Classificatie is een van de fundamentele activiteiten van de machine learning-onderzoeker en datawetenschapper. Van eenvoudige classificatie van een binaire waarde ("is deze e-mail spam of niet?") tot complexe beeldclassificatie en segmentatie met behulp van computer vision, het is altijd nuttig om gegevens in klassen te sorteren en er vragen over te stellen.

Om het proces op een meer wetenschappelijke manier te beschrijven, creëert je classificatiemethode een voorspellend model dat je in staat stelt de relatie tussen invoervariabelen en uitvoervariabelen in kaart te brengen.

![binaire vs. multiclass classificatie](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binaire vs. multiclass problemen voor classificatie-algoritmen. Infographic door [Jen Looper](https://twitter.com/jenlooper)

Voordat we beginnen met het schoonmaken van onze data, het visualiseren ervan en het voorbereiden voor onze ML-taken, laten we eerst wat leren over de verschillende manieren waarop machine learning kan worden gebruikt om data te classificeren.

Afgeleid van [statistiek](https://wikipedia.org/wiki/Statistical_classification), gebruikt classificatie met klassieke machine learning kenmerken zoals `smoker`, `weight` en `age` om de _waarschijnlijkheid van het ontwikkelen van X ziekte_ te bepalen. Als een supervised learning-techniek, vergelijkbaar met de regressie-oefeningen die je eerder hebt uitgevoerd, is je data gelabeld en gebruiken de ML-algoritmen die labels om klassen (of 'kenmerken') van een dataset te classificeren en toe te wijzen aan een groep of uitkomst.

✅ Neem even de tijd om je een dataset over keukens voor te stellen. Welke vragen zou een multiclass-model kunnen beantwoorden? Welke vragen zou een binair model kunnen beantwoorden? Wat als je wilde bepalen of een bepaalde keuken waarschijnlijk fenegriek gebruikt? Wat als je wilde zien of je, gegeven een tas vol steranijs, artisjokken, bloemkool en mierikswortel, een typisch Indiaas gerecht zou kunnen maken?

[![Gekke mystery baskets](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Gekke mystery baskets")

> 🎥 Klik op de afbeelding hierboven voor een video. Het hele concept van de show 'Chopped' draait om de 'mystery basket' waarin chefs een gerecht moeten maken van een willekeurige keuze aan ingrediënten. Een ML-model zou zeker hebben geholpen!

## Hallo 'classifier'

De vraag die we willen stellen over deze keuken-dataset is eigenlijk een **multiclass-vraag**, omdat we met verschillende mogelijke nationale keukens werken. Gegeven een batch ingrediënten, bij welke van deze vele klassen past de data?

Scikit-learn biedt verschillende algoritmen om data te classificeren, afhankelijk van het soort probleem dat je wilt oplossen. In de volgende twee lessen leer je over enkele van deze algoritmen.

## Oefening - maak je data schoon en balanceer het

De eerste taak, voordat je aan dit project begint, is om je data schoon te maken en **te balanceren** om betere resultaten te krijgen. Begin met het lege _notebook.ipynb_-bestand in de root van deze map.

Het eerste wat je moet installeren is [imblearn](https://imbalanced-learn.org/stable/). Dit is een Scikit-learn-pakket waarmee je de data beter kunt balanceren (je leert hier meer over in een minuut).

1. Om `imblearn` te installeren, voer je `pip install` uit, zoals hieronder:

    ```python
    pip install imblearn
    ```

1. Importeer de pakketten die je nodig hebt om je data te importeren en te visualiseren, en importeer ook `SMOTE` van `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Nu ben je klaar om de data te importeren.

1. De volgende taak is om de data te importeren:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Door `read_csv()` te gebruiken, wordt de inhoud van het csv-bestand _cusines.csv_ gelezen en geplaatst in de variabele `df`.

1. Controleer de vorm van de data:

    ```python
    df.head()
    ```

   De eerste vijf rijen zien er zo uit:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Verkrijg informatie over deze data door `info()` aan te roepen:

    ```python
    df.info()
    ```

    Je output lijkt op:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Oefening - leren over keukens

Nu begint het werk interessanter te worden. Laten we de verdeling van data per keuken ontdekken.

1. Plot de data als balken door `barh()` aan te roepen:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![keuken data verdeling](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Er zijn een beperkt aantal keukens, maar de verdeling van data is ongelijk. Je kunt dat oplossen! Voordat je dat doet, verken je nog wat meer.

1. Ontdek hoeveel data er beschikbaar is per keuken en print het uit:

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

    De output ziet er zo uit:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Ingrediënten ontdekken

Nu kun je dieper in de data duiken en leren wat de typische ingrediënten per keuken zijn. Je moet terugkerende data verwijderen die verwarring veroorzaakt tussen keukens, dus laten we meer leren over dit probleem.

1. Maak een functie `create_ingredient()` in Python om een ingrediënten-dataframe te maken. Deze functie begint met het verwijderen van een nutteloze kolom en sorteert ingrediënten op hun aantal:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Nu kun je die functie gebruiken om een idee te krijgen van de tien meest populaire ingrediënten per keuken.

1. Roep `create_ingredient()` aan en plot het door `barh()` aan te roepen:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Doe hetzelfde voor de Japanse data:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Nu voor de Chinese ingrediënten:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Plot de Indiase ingrediënten:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Tot slot, plot de Koreaanse ingrediënten:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Verwijder nu de meest voorkomende ingrediënten die verwarring veroorzaken tussen verschillende keukens door `drop()` aan te roepen:

   Iedereen houdt van rijst, knoflook en gember!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Balanceer de dataset

Nu je de data hebt schoongemaakt, gebruik [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - om het te balanceren.

1. Roep `fit_resample()` aan, deze strategie genereert nieuwe samples door interpolatie.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Door je data te balanceren, krijg je betere resultaten bij het classificeren ervan. Denk aan een binaire classificatie. Als het grootste deel van je data één klasse is, zal een ML-model die klasse vaker voorspellen, simpelweg omdat er meer data voor is. Het balanceren van de data neemt scheve data en helpt deze onbalans te verwijderen.

1. Nu kun je het aantal labels per ingrediënt controleren:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Je output ziet er zo uit:

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

    De data is mooi schoon, gebalanceerd en erg smakelijk!

1. De laatste stap is om je gebalanceerde data, inclusief labels en kenmerken, op te slaan in een nieuw dataframe dat kan worden geëxporteerd naar een bestand:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Je kunt nog een laatste blik werpen op de data met `transformed_df.head()` en `transformed_df.info()`. Sla een kopie van deze data op voor gebruik in toekomstige lessen:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Deze verse CSV is nu te vinden in de root data map.

---

## 🚀Uitdaging

Deze curriculum bevat verschillende interessante datasets. Doorzoek de `data`-mappen en kijk of er datasets zijn die geschikt zouden zijn voor binaire of multiclass-classificatie. Welke vragen zou je stellen over deze dataset?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Verken de API van SMOTE. Voor welke use cases is het het meest geschikt? Welke problemen lost het op?

## Opdracht 

[Verken classificatiemethoden](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.