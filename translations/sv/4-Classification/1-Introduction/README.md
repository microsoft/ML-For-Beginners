<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T21:55:18+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "sv"
}
-->
# Introduktion till klassificering

I dessa fyra lektioner kommer du att utforska ett grundläggande fokus inom klassisk maskininlärning - _klassificering_. Vi kommer att gå igenom användningen av olika klassificeringsalgoritmer med en dataset om alla fantastiska kök från Asien och Indien. Hoppas du är hungrig!

![bara en nypa!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Fira pan-asiatiska kök i dessa lektioner! Bild av [Jen Looper](https://twitter.com/jenlooper)

Klassificering är en form av [övervakad inlärning](https://wikipedia.org/wiki/Supervised_learning) som har mycket gemensamt med regressionstekniker. Om maskininlärning handlar om att förutsäga värden eller namn på saker med hjälp av dataset, så faller klassificering generellt sett in i två grupper: _binär klassificering_ och _multiklassklassificering_.

[![Introduktion till klassificering](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introduktion till klassificering")

> 🎥 Klicka på bilden ovan för en video: MIT:s John Guttag introducerar klassificering

Kom ihåg:

- **Linjär regression** hjälpte dig att förutsäga relationer mellan variabler och göra exakta förutsägelser om var en ny datapunkt skulle hamna i förhållande till den linjen. Så, du kunde förutsäga _vad priset på en pumpa skulle vara i september jämfört med december_, till exempel.
- **Logistisk regression** hjälpte dig att upptäcka "binära kategorier": vid denna prisnivå, _är denna pumpa orange eller inte-orange_?

Klassificering använder olika algoritmer för att bestämma andra sätt att avgöra en datapunkts etikett eller klass. Låt oss arbeta med denna dataset om kök för att se om vi, genom att observera en grupp ingredienser, kan avgöra dess ursprungskök.

## [Quiz före lektionen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denna lektion finns tillgänglig i R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introduktion

Klassificering är en av de grundläggande aktiviteterna för forskare inom maskininlärning och dataanalytiker. Från grundläggande klassificering av ett binärt värde ("är detta e-postmeddelande skräppost eller inte?") till komplex bildklassificering och segmentering med hjälp av datorseende, är det alltid användbart att kunna sortera data i klasser och ställa frågor om den.

För att uttrycka processen på ett mer vetenskapligt sätt skapar din klassificeringsmetod en prediktiv modell som gör det möjligt att kartlägga relationen mellan indata och utdata.

![binär vs. multiklassklassificering](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binära vs. multiklassproblem för klassificeringsalgoritmer att hantera. Infografik av [Jen Looper](https://twitter.com/jenlooper)

Innan vi börjar processen med att rensa vår data, visualisera den och förbereda den för våra ML-uppgifter, låt oss lära oss lite om de olika sätten maskininlärning kan användas för att klassificera data.

Härledd från [statistik](https://wikipedia.org/wiki/Statistical_classification), klassificering med klassisk maskininlärning använder egenskaper, såsom `smoker`, `weight` och `age` för att avgöra _sannolikheten att utveckla X sjukdom_. Som en övervakad inlärningsteknik liknande de regressionövningar du utförde tidigare, är din data märkt och ML-algoritmerna använder dessa etiketter för att klassificera och förutsäga klasser (eller 'egenskaper') i en dataset och tilldela dem till en grupp eller ett resultat.

✅ Ta en stund och föreställ dig en dataset om kök. Vad skulle en multiklassmodell kunna svara på? Vad skulle en binär modell kunna svara på? Vad händer om du ville avgöra om ett visst kök sannolikt använder bockhornsklöver? Vad händer om du ville se om du, med en present av en matkasse full av stjärnanis, kronärtskockor, blomkål och pepparrot, kunde skapa en typisk indisk rätt?

[![Galna mysteriekorgar](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Galna mysteriekorgar")

> 🎥 Klicka på bilden ovan för en video. Hela premissen för programmet 'Chopped' är 'mysteriekorgen' där kockar måste göra en rätt av ett slumpmässigt val av ingredienser. Visst skulle en ML-modell ha hjälpt!

## Hej 'klassificerare'

Frågan vi vill ställa om denna dataset om kök är faktiskt en **multiklassfråga**, eftersom vi har flera potentiella nationella kök att arbeta med. Givet en mängd ingredienser, vilken av dessa många klasser passar datan in i?

Scikit-learn erbjuder flera olika algoritmer att använda för att klassificera data, beroende på vilken typ av problem du vill lösa. I de kommande två lektionerna kommer du att lära dig om flera av dessa algoritmer.

## Övning - rensa och balansera din data

Den första uppgiften, innan vi börjar detta projekt, är att rensa och **balansera** din data för att få bättre resultat. Börja med den tomma filen _notebook.ipynb_ i roten av denna mapp.

Det första du behöver installera är [imblearn](https://imbalanced-learn.org/stable/). Detta är ett Scikit-learn-paket som gör det möjligt att bättre balansera datan (du kommer att lära dig mer om denna uppgift om en stund).

1. För att installera `imblearn`, kör `pip install`, så här:

    ```python
    pip install imblearn
    ```

1. Importera de paket du behöver för att importera din data och visualisera den, importera också `SMOTE` från `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Nu är du redo att importera datan nästa gång.

1. Nästa uppgift är att importera datan:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Genom att använda `read_csv()` kommer innehållet i csv-filen _cusines.csv_ att läsas och placeras i variabeln `df`.

1. Kontrollera datans form:

    ```python
    df.head()
    ```

   De första fem raderna ser ut så här:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Få information om denna data genom att kalla på `info()`:

    ```python
    df.info()
    ```

    Din utdata liknar:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Övning - lära sig om kök

Nu börjar arbetet bli mer intressant. Låt oss upptäcka fördelningen av data, per kök.

1. Plotta datan som staplar genom att kalla på `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![fördelning av köksdata](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Det finns ett begränsat antal kök, men fördelningen av data är ojämn. Du kan fixa det! Innan du gör det, utforska lite mer.

1. Ta reda på hur mycket data som finns tillgängligt per kök och skriv ut det:

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

    Utdata ser ut så här:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Upptäcka ingredienser

Nu kan du gräva djupare i datan och lära dig vilka som är de typiska ingredienserna per kök. Du bör rensa bort återkommande data som skapar förvirring mellan kök, så låt oss lära oss om detta problem.

1. Skapa en funktion `create_ingredient()` i Python för att skapa en ingrediens-dataset. Denna funktion börjar med att ta bort en oanvändbar kolumn och sortera ingredienser efter deras antal:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Nu kan du använda den funktionen för att få en idé om de tio mest populära ingredienserna per kök.

1. Kalla på `create_ingredient()` och plotta det genom att kalla på `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Gör samma sak för den japanska datan:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanska](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Nu för de kinesiska ingredienserna:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kinesiska](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Plotta de indiska ingredienserna:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indiska](../../../../4-Classification/1-Introduction/images/indian.png)

1. Slutligen, plotta de koreanska ingredienserna:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![koreanska](../../../../4-Classification/1-Introduction/images/korean.png)

1. Nu, ta bort de vanligaste ingredienserna som skapar förvirring mellan olika kök, genom att kalla på `drop()`:

   Alla älskar ris, vitlök och ingefära!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Balansera datasetet

Nu när du har rensat datan, använd [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - för att balansera den.

1. Kalla på `fit_resample()`, denna strategi genererar nya prover genom interpolation.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Genom att balansera din data får du bättre resultat när du klassificerar den. Tänk på en binär klassificering. Om det mesta av din data är en klass, kommer en ML-modell att förutsäga den klassen oftare, bara för att det finns mer data för den. Att balansera datan tar bort eventuella snedvridningar och hjälper till att eliminera denna obalans.

1. Nu kan du kontrollera antalet etiketter per ingrediens:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Din utdata ser ut så här:

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

    Datan är fin och ren, balanserad och mycket läcker!

1. Det sista steget är att spara din balanserade data, inklusive etiketter och egenskaper, i en ny dataset som kan exporteras till en fil:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Du kan ta en sista titt på datan med `transformed_df.head()` och `transformed_df.info()`. Spara en kopia av denna data för användning i framtida lektioner:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Denna nya CSV-fil kan nu hittas i rotens datamapp.

---

## 🚀Utmaning

Detta läroplan innehåller flera intressanta dataset. Gräv igenom `data`-mapparna och se om någon innehåller dataset som skulle vara lämpliga för binär eller multiklassklassificering? Vilka frågor skulle du ställa till denna dataset?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Utforska SMOTEs API. Vilka användningsområden är det bäst lämpat för? Vilka problem löser det?

## Uppgift 

[Utforska klassificeringsmetoder](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör du vara medveten om att automatiserade översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.