<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T16:26:25+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "sk"
}
-->
# Úvod do klasifikácie

V týchto štyroch lekciách sa budete venovať základnému zameraniu klasického strojového učenia - _klasifikácii_. Prejdeme si používanie rôznych klasifikačných algoritmov s datasetom o všetkých úžasných kuchyniach Ázie a Indie. Dúfam, že máte chuť na jedlo!

![len štipka!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Oslávte panázijské kuchyne v týchto lekciách! Obrázok od [Jen Looper](https://twitter.com/jenlooper)

Klasifikácia je forma [supervised learning](https://wikipedia.org/wiki/Supervised_learning), ktorá má veľa spoločného s regresnými technikami. Ak je strojové učenie o predpovedaní hodnôt alebo názvov vecí pomocou datasetov, potom klasifikácia všeobecne spadá do dvoch skupín: _binárna klasifikácia_ a _multiklasová klasifikácia_.

[![Úvod do klasifikácie](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Úvod do klasifikácie")

> 🎥 Kliknite na obrázok vyššie pre video: MIT's John Guttag predstavuje klasifikáciu

Pamätajte:

- **Lineárna regresia** vám pomohla predpovedať vzťahy medzi premennými a robiť presné predpovede o tom, kde nový dátový bod spadne vo vzťahu k tejto čiare. Napríklad ste mohli predpovedať _aká bude cena tekvice v septembri vs. decembri_.
- **Logistická regresia** vám pomohla objaviť "binárne kategórie": pri tejto cenovej hladine, _je táto tekvica oranžová alebo nie-oranžová_?

Klasifikácia používa rôzne algoritmy na určenie iných spôsobov, ako určiť označenie alebo triedu dátového bodu. Poďme pracovať s týmto datasetom o kuchyniach, aby sme zistili, či na základe skupiny ingrediencií dokážeme určiť jej pôvodnú kuchyňu.

## [Kvíz pred lekciou](https://ff-quizzes.netlify.app/en/ml/)

> ### [Táto lekcia je dostupná v R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Úvod

Klasifikácia je jednou zo základných aktivít výskumníka strojového učenia a dátového vedca. Od základnej klasifikácie binárnej hodnoty ("je tento email spam alebo nie?") až po komplexnú klasifikáciu a segmentáciu obrázkov pomocou počítačového videnia, je vždy užitočné vedieť triediť dáta do tried a klásť im otázky.

Ak to vyjadríme vedeckejšie, vaša klasifikačná metóda vytvára prediktívny model, ktorý vám umožňuje mapovať vzťah medzi vstupnými premennými a výstupnými premennými.

![binárna vs. multiklasová klasifikácia](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binárne vs. multiklasové problémy, ktoré musia klasifikačné algoritmy riešiť. Infografika od [Jen Looper](https://twitter.com/jenlooper)

Predtým, než začneme proces čistenia našich dát, ich vizualizácie a prípravy na úlohy strojového učenia, poďme sa trochu naučiť o rôznych spôsoboch, akými môže byť strojové učenie využité na klasifikáciu dát.

Odvodené zo [štatistiky](https://wikipedia.org/wiki/Statistical_classification), klasifikácia pomocou klasického strojového učenia používa vlastnosti, ako `smoker`, `weight` a `age`, na určenie _pravdepodobnosti rozvoja X choroby_. Ako technika supervised learning podobná regresným cvičeniam, ktoré ste vykonávali skôr, vaše dáta sú označené a algoritmy strojového učenia používajú tieto označenia na klasifikáciu a predpovedanie tried (alebo 'vlastností') datasetu a ich priradenie do skupiny alebo výsledku.

✅ Predstavte si dataset o kuchyniach. Aké otázky by mohol multiklasový model zodpovedať? Aké otázky by mohol zodpovedať binárny model? Čo ak by ste chceli určiť, či daná kuchyňa pravdepodobne používa senovku grécku? Čo ak by ste chceli zistiť, či by ste z darovaného nákupného košíka plného badiánu, artičokov, karfiolu a chrenu mohli vytvoriť typické indické jedlo?

[![Bláznivé tajomné košíky](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Bláznivé tajomné košíky")

> 🎥 Kliknite na obrázok vyššie pre video. Celý koncept relácie 'Chopped' je o 'tajomnom košíku', kde šéfkuchári musia pripraviť jedlo z náhodného výberu ingrediencií. Určite by model strojového učenia pomohol!

## Ahoj 'klasifikátor'

Otázka, ktorú chceme položiť tomuto datasetu o kuchyniach, je vlastne **multiklasová otázka**, pretože máme niekoľko potenciálnych národných kuchýň, s ktorými môžeme pracovať. Na základe dávky ingrediencií, do ktorej z týchto mnohých tried budú dáta patriť?

Scikit-learn ponúka niekoľko rôznych algoritmov na klasifikáciu dát, v závislosti od typu problému, ktorý chcete vyriešiť. V nasledujúcich dvoch lekciách sa naučíte o niekoľkých z týchto algoritmov.

## Cvičenie - vyčistite a vyvážte svoje dáta

Prvým krokom pred začatím projektu je vyčistiť a **vyvážiť** svoje dáta, aby ste dosiahli lepšie výsledky. Začnite s prázdnym súborom _notebook.ipynb_ v koreňovom adresári tejto zložky.

Prvá vec, ktorú je potrebné nainštalovať, je [imblearn](https://imbalanced-learn.org/stable/). Toto je balík Scikit-learn, ktorý vám umožní lepšie vyvážiť dáta (o tejto úlohe sa dozviete viac za chvíľu).

1. Na inštaláciu `imblearn` spustite `pip install`, takto:

    ```python
    pip install imblearn
    ```

1. Importujte balíky, ktoré potrebujete na importovanie a vizualizáciu dát, tiež importujte `SMOTE` z `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Teraz ste pripravení na importovanie dát.

1. Ďalším krokom bude importovanie dát:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Použitie `read_csv()` načíta obsah súboru _cusines.csv_ a uloží ho do premennej `df`.

1. Skontrolujte tvar dát:

    ```python
    df.head()
    ```

   Prvých päť riadkov vyzerá takto:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Získajte informácie o týchto dátach volaním `info()`:

    ```python
    df.info()
    ```

    Vaša výstupná podoba sa podobá:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Cvičenie - učenie o kuchyniach

Teraz sa práca začína stávať zaujímavejšou. Poďme objaviť distribúciu dát podľa kuchyne.

1. Vykreslite dáta ako stĺpce volaním `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![distribúcia dát o kuchyniach](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Existuje konečný počet kuchýň, ale distribúcia dát je nerovnomerná. Môžete to opraviť! Predtým však trochu preskúmajte.

1. Zistite, koľko dát je dostupných na kuchyňu a vytlačte to:

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

    Výstup vyzerá takto:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Objavovanie ingrediencií

Teraz môžete ísť hlbšie do dát a zistiť, aké sú typické ingrediencie pre jednotlivé kuchyne. Mali by ste vyčistiť opakujúce sa dáta, ktoré vytvárajú zmätok medzi kuchyňami, takže sa poďme dozvedieť o tomto probléme.

1. Vytvorte funkciu `create_ingredient()` v Pythone na vytvorenie dataframe ingrediencií. Táto funkcia začne odstránením nepotrebného stĺpca a triedením ingrediencií podľa ich počtu:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Teraz môžete použiť túto funkciu na získanie predstavy o desiatich najpopulárnejších ingredienciách podľa kuchyne.

1. Zavolajte `create_ingredient()` a vykreslite to volaním `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Urobte to isté pre japonské dáta:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Teraz pre čínske ingrediencie:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Vykreslite indické ingrediencie:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Nakoniec vykreslite kórejské ingrediencie:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Teraz odstráňte najbežnejšie ingrediencie, ktoré vytvárajú zmätok medzi rôznymi kuchyňami, volaním `drop()`:

   Každý miluje ryžu, cesnak a zázvor!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Vyváženie datasetu

Teraz, keď ste vyčistili dáta, použite [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - na ich vyváženie.

1. Zavolajte `fit_resample()`, táto stratégia generuje nové vzorky interpoláciou.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Vyvážením dát dosiahnete lepšie výsledky pri ich klasifikácii. Premýšľajte o binárnej klasifikácii. Ak väčšina vašich dát patrí do jednej triedy, model strojového učenia bude predpovedať túto triedu častejšie, len preto, že je pre ňu viac dát. Vyváženie dát odstráni túto nerovnováhu.

1. Teraz môžete skontrolovať počet označení na ingredienciu:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Vaša výstupná podoba vyzerá takto:

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

    Dáta sú pekné, čisté, vyvážené a veľmi chutné!

1. Posledným krokom je uloženie vyvážených dát, vrátane označení a vlastností, do nového dataframe, ktorý môže byť exportovaný do súboru:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Môžete sa ešte raz pozrieť na dáta pomocou `transformed_df.head()` a `transformed_df.info()`. Uložte kópiu týchto dát na použitie v budúcich lekciách:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Tento nový CSV súbor sa teraz nachádza v koreňovom adresári dát.

---

## 🚀Výzva

Tento učebný plán obsahuje niekoľko zaujímavých datasetov. Prezrite si zložky `data` a zistite, či niektoré obsahujú datasety, ktoré by boli vhodné pre binárnu alebo multiklasovú klasifikáciu? Aké otázky by ste mohli položiť tomuto datasetu?

## [Kvíz po lekcii](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Preskúmajte API SMOTE. Pre aké prípady použitia je najlepšie? Aké problémy rieši?

## Zadanie 

[Preskúmajte metódy klasifikácie](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.