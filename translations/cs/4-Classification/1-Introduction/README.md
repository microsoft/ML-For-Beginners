<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T00:52:25+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "cs"
}
-->
# Úvod do klasifikace

V těchto čtyřech lekcích se ponoříte do základního zaměření klasického strojového učení - _klasifikace_. Projdeme si použití různých klasifikačních algoritmů na datasetu o všech úžasných kuchyních Asie a Indie. Doufáme, že máte chuť k jídlu!

![jen špetka!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Oslavte pan-asijské kuchyně v těchto lekcích! Obrázek od [Jen Looper](https://twitter.com/jenlooper)

Klasifikace je forma [supervizovaného učení](https://wikipedia.org/wiki/Supervised_learning), která má mnoho společného s regresními technikami. Pokud je strojové učení o předpovídání hodnot nebo názvů věcí pomocí datasetů, pak klasifikace obecně spadá do dvou skupin: _binární klasifikace_ a _multitřídní klasifikace_.

[![Úvod do klasifikace](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Úvod do klasifikace")

> 🎥 Klikněte na obrázek výše pro video: John Guttag z MIT představuje klasifikaci

Pamatujte:

- **Lineární regrese** vám pomohla předpovědět vztahy mezi proměnnými a vytvořit přesné předpovědi, kam nový datový bod spadne ve vztahu k této linii. Například jste mohli předpovědět _jaká bude cena dýně v září vs. prosinci_.
- **Logistická regrese** vám pomohla objevit "binární kategorie": při této cenové hladině, _je tato dýně oranžová nebo neoranžová_?

Klasifikace využívá různé algoritmy k určení dalších způsobů, jak určit štítek nebo třídu datového bodu. Pojďme pracovat s tímto datasetem o kuchyních a zjistit, zda můžeme na základě skupiny ingrediencí určit její původní kuchyni.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tato lekce je dostupná v R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Úvod

Klasifikace je jednou ze základních činností výzkumníka strojového učení a datového vědce. Od základní klasifikace binární hodnoty ("je tento e-mail spam nebo ne?") až po složitou klasifikaci a segmentaci obrázků pomocí počítačového vidění, je vždy užitečné být schopen třídit data do tříd a klást si otázky.

Řečeno vědeckým způsobem, vaše klasifikační metoda vytváří prediktivní model, který vám umožňuje mapovat vztah mezi vstupními proměnnými a výstupními proměnnými.

![binární vs. multitřídní klasifikace](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binární vs. multitřídní problémy, které klasifikační algoritmy řeší. Infografika od [Jen Looper](https://twitter.com/jenlooper)

Než začneme proces čištění dat, jejich vizualizace a přípravy na úkoly strojového učení, pojďme se trochu dozvědět o různých způsobech, jak lze strojové učení využít ke klasifikaci dat.

Odvozeno z [statistiky](https://wikipedia.org/wiki/Statistical_classification), klasifikace pomocí klasického strojového učení využívá vlastnosti, jako `smoker`, `weight` a `age`, k určení _pravděpodobnosti rozvoje X nemoci_. Jako technika supervizovaného učení podobná regresním cvičením, která jste prováděli dříve, jsou vaše data označena a algoritmy strojového učení používají tyto štítky k klasifikaci a předpovídání tříd (nebo 'vlastností') datasetu a jejich přiřazení do skupiny nebo výsledku.

✅ Udělejte si chvíli na představu datasetu o kuchyních. Na jaké otázky by mohl odpovědět multitřídní model? Na jaké otázky by mohl odpovědět binární model? Co kdybyste chtěli zjistit, zda daná kuchyně pravděpodobně používá pískavici? Co kdybyste chtěli zjistit, zda byste mohli vytvořit typické indické jídlo z tašky s potravinami plné badyánu, artyčoků, květáku a křenu?

[![Bláznivé tajemné košíky](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Bláznivé tajemné košíky")

> 🎥 Klikněte na obrázek výše pro video. Celý koncept pořadu 'Chopped' je založen na 'tajemném košíku', kde kuchaři musí vytvořit jídlo z náhodného výběru ingrediencí. Určitě by model strojového učení pomohl!

## Ahoj 'klasifikátore'

Otázka, kterou chceme položit tomuto datasetu o kuchyních, je vlastně **multitřídní otázka**, protože máme několik potenciálních národních kuchyní, se kterými můžeme pracovat. Na základě dávky ingrediencí, do které z těchto mnoha tříd budou data spadat?

Scikit-learn nabízí několik různých algoritmů pro klasifikaci dat, v závislosti na typu problému, který chcete vyřešit. V následujících dvou lekcích se naučíte o několika z těchto algoritmů.

## Cvičení - vyčistěte a vyvažte svá data

Prvním úkolem, než začneme tento projekt, je vyčistit a **vyvážit** svá data, abyste dosáhli lepších výsledků. Začněte s prázdným souborem _notebook.ipynb_ v kořenové složce této složky.

První věc, kterou je třeba nainstalovat, je [imblearn](https://imbalanced-learn.org/stable/). Jedná se o balíček Scikit-learn, který vám umožní lépe vyvážit data (o této úloze se dozvíte více za chvíli).

1. Pro instalaci `imblearn` spusťte `pip install`, takto:

    ```python
    pip install imblearn
    ```

1. Importujte balíčky, které potřebujete k importu svých dat a jejich vizualizaci, také importujte `SMOTE` z `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Nyní jste připraveni importovat data.

1. Dalším úkolem bude import dat:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Použití `read_csv()` načte obsah souboru csv _cusines.csv_ a umístí jej do proměnné `df`.

1. Zkontrolujte tvar dat:

    ```python
    df.head()
    ```

   Prvních pět řádků vypadá takto:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Získejte informace o těchto datech pomocí volání `info()`:

    ```python
    df.info()
    ```

    Vaše výstup vypadá takto:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Cvičení - poznávání kuchyní

Nyní se práce začíná stávat zajímavější. Pojďme objevit rozložení dat podle kuchyně.

1. Vykreslete data jako sloupce pomocí volání `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![rozložení dat o kuchyních](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Existuje konečný počet kuchyní, ale rozložení dat je nerovnoměrné. Můžete to opravit! Než tak učiníte, prozkoumejte trochu více.

1. Zjistěte, kolik dat je k dispozici na kuchyni, a vytiskněte to:

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

    výstup vypadá takto:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Objevování ingrediencí

Nyní se můžete ponořit hlouběji do dat a zjistit, jaké jsou typické ingredience pro jednotlivé kuchyně. Měli byste odstranit opakující se data, která vytvářejí zmatek mezi kuchyněmi, takže se pojďme dozvědět o tomto problému.

1. Vytvořte funkci `create_ingredient()` v Pythonu pro vytvoření dataframe ingrediencí. Tato funkce začne odstraněním neužitečného sloupce a tříděním ingrediencí podle jejich počtu:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Nyní můžete použít tuto funkci k získání představy o deseti nejoblíbenějších ingrediencích podle kuchyně.

1. Zavolejte `create_ingredient()` a vykreslete to pomocí volání `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Udělejte totéž pro japonská data:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Nyní pro čínské ingredience:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Vykreslete indické ingredience:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Nakonec vykreslete korejské ingredience:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Nyní odstraňte nejběžnější ingredience, které vytvářejí zmatek mezi různými kuchyněmi, pomocí volání `drop()`:

   Každý miluje rýži, česnek a zázvor!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Vyvažte dataset

Nyní, když jste data vyčistili, použijte [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - k jejich vyvážení.

1. Zavolejte `fit_resample()`, tato strategie generuje nové vzorky interpolací.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Vyvážením dat dosáhnete lepších výsledků při jejich klasifikaci. Přemýšlejte o binární klasifikaci. Pokud většina vašich dat patří do jedné třídy, model strojového učení bude tuto třídu předpovídat častěji, jen proto, že pro ni existuje více dat. Vyvážení dat odstraní tuto nerovnováhu.

1. Nyní můžete zkontrolovat počet štítků na ingredienci:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Vaše výstup vypadá takto:

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

    Data jsou pěkná, čistá, vyvážená a velmi chutná!

1. Posledním krokem je uložení vyvážených dat, včetně štítků a vlastností, do nového dataframe, který lze exportovat do souboru:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Můžete se ještě jednou podívat na data pomocí `transformed_df.head()` a `transformed_df.info()`. Uložte kopii těchto dat pro použití v budoucích lekcích:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Tento nový CSV soubor nyní najdete v kořenové složce dat.

---

## 🚀Výzva

Tento kurz obsahuje několik zajímavých datasetů. Projděte složky `data` a zjistěte, zda některé obsahují datasety, které by byly vhodné pro binární nebo multitřídní klasifikaci? Jaké otázky byste si mohli položit o tomto datasetu?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Prozkoumejte API SMOTE. Pro jaké případy použití je nejvhodnější? Jaké problémy řeší?

## Zadání 

[Prozkoumejte metody klasifikace](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.