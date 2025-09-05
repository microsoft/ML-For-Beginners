<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T13:18:08+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "sl"
}
-->
# Uvod v klasifikacijo

V teh štirih lekcijah boste raziskovali eno temeljnih področij klasičnega strojnega učenja - _klasifikacijo_. Preučili bomo uporabo različnih algoritmov za klasifikacijo z uporabo nabora podatkov o vseh čudovitih azijskih in indijskih kuhinjah. Upam, da ste lačni!

![samo ščepec!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Praznujte panazijske kuhinje v teh lekcijah! Slika: [Jen Looper](https://twitter.com/jenlooper)

Klasifikacija je oblika [nadzorovanega učenja](https://wikipedia.org/wiki/Supervised_learning), ki ima veliko skupnega s tehnikami regresije. Če je strojno učenje namenjeno napovedovanju vrednosti ali imen stvari z uporabo naborov podatkov, potem klasifikacijo običajno delimo v dve skupini: _binarna klasifikacija_ in _večrazredna klasifikacija_.

[![Uvod v klasifikacijo](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Uvod v klasifikacijo")

> 🎥 Kliknite zgornjo sliko za video: MIT-jev John Guttag predstavlja klasifikacijo

Zapomnite si:

- **Linearna regresija** vam je pomagala napovedati odnose med spremenljivkami in natančno predvideti, kam bo padla nova podatkovna točka glede na to črto. Na primer, lahko ste napovedali _kakšna bo cena buče septembra v primerjavi z decembrom_.
- **Logistična regresija** vam je pomagala odkriti "binarne kategorije": pri tej ceni, _ali je ta buča oranžna ali ne-oranžna_?

Klasifikacija uporablja različne algoritme za določanje drugih načinov ugotavljanja oznake ali razreda podatkovne točke. Delali bomo s temi podatki o kuhinjah, da ugotovimo, ali lahko na podlagi skupine sestavin določimo izvorno kuhinjo.

## [Predlekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ta lekcija je na voljo tudi v jeziku R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Uvod

Klasifikacija je ena temeljnih dejavnosti raziskovalcev strojnega učenja in podatkovnih znanstvenikov. Od osnovne klasifikacije binarne vrednosti ("ali je to e-poštno sporočilo vsiljena pošta ali ne?") do zapletene klasifikacije slik in segmentacije s pomočjo računalniškega vida, je vedno koristno znati razvrstiti podatke v razrede in jim zastaviti vprašanja.

Če proces opišemo na bolj znanstven način, vaša metoda klasifikacije ustvari napovedni model, ki vam omogoča, da preslikate odnos med vhodnimi in izhodnimi spremenljivkami.

![binarna vs. večrazredna klasifikacija](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binarni in večrazredni problemi za algoritme klasifikacije. Infografika: [Jen Looper](https://twitter.com/jenlooper)

Preden začnemo s procesom čiščenja podatkov, njihove vizualizacije in priprave za naloge strojnega učenja, se naučimo nekaj več o različnih načinih, kako lahko strojno učenje uporabimo za klasifikacijo podatkov.

Izpeljana iz [statistike](https://wikipedia.org/wiki/Statistical_classification), klasifikacija s klasičnim strojnim učenjem uporablja značilnosti, kot so `kadilec`, `teža` in `starost`, za določanje _verjetnosti razvoja določene bolezni_. Kot tehnika nadzorovanega učenja, podobna regresijskim vajam, ki ste jih izvajali prej, so vaši podatki označeni, algoritmi strojnega učenja pa te oznake uporabljajo za klasifikacijo in napovedovanje razredov (ali 'značilnosti') nabora podatkov ter njihovo dodelitev skupini ali izidu.

✅ Vzemite si trenutek in si zamislite nabor podatkov o kuhinjah. Na katera vprašanja bi lahko odgovoril večrazredni model? Na katera vprašanja bi lahko odgovoril binarni model? Kaj če bi želeli ugotoviti, ali določena kuhinja uporablja sabljiko? Kaj če bi želeli preveriti, ali bi lahko iz vrečke zvezdastega janeža, artičok, cvetače in hrena pripravili tipično indijsko jed?

[![Nore skrivnostne košare](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Nore skrivnostne košare")

> 🎥 Kliknite zgornjo sliko za video. Celoten koncept oddaje 'Chopped' temelji na 'skrivnostni košari', kjer morajo kuharji iz naključnih sestavin pripraviti jed. Zagotovo bi jim model strojnega učenja pomagal!

## Pozdravljen 'klasifikator'

Vprašanje, ki si ga želimo zastaviti o tem naboru podatkov o kuhinjah, je pravzaprav **večrazredno vprašanje**, saj imamo na voljo več potencialnih nacionalnih kuhinj. Glede na skupino sestavin, v kateri od teh številnih razredov bodo podatki ustrezali?

Scikit-learn ponuja več različnih algoritmov za klasifikacijo podatkov, odvisno od vrste problema, ki ga želite rešiti. V naslednjih dveh lekcijah se boste naučili več o teh algoritmih.

## Vaja - očistite in uravnotežite svoje podatke

Prva naloga, preden začnemo s tem projektom, je očistiti in **uravnotežiti** podatke za boljše rezultate. Začnite z prazno datoteko _notebook.ipynb_ v korenski mapi te mape.

Prva stvar, ki jo morate namestiti, je [imblearn](https://imbalanced-learn.org/stable/). To je paket Scikit-learn, ki vam bo omogočil boljše uravnoteženje podatkov (več o tej nalogi boste izvedeli v nadaljevanju).

1. Za namestitev `imblearn` zaženite `pip install`, kot sledi:

    ```python
    pip install imblearn
    ```

1. Uvozite pakete, ki jih potrebujete za uvoz in vizualizacijo podatkov, prav tako uvozite `SMOTE` iz `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Zdaj ste pripravljeni na uvoz podatkov.

1. Naslednja naloga je uvoz podatkov:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Z uporabo `read_csv()` boste prebrali vsebino csv datoteke _cusines.csv_ in jo shranili v spremenljivko `df`.

1. Preverite obliko podatkov:

    ```python
    df.head()
    ```

   Prvih pet vrstic izgleda takole:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Pridobite informacije o teh podatkih z uporabo `info()`:

    ```python
    df.info()
    ```

    Vaš izpis je podoben:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Vaja - spoznavanje kuhinj

Zdaj postane delo bolj zanimivo. Odkrijmo porazdelitev podatkov po kuhinjah.

1. Prikaz podatkov kot stolpce z uporabo `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![porazdelitev podatkov o kuhinjah](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Število kuhinj je omejeno, vendar je porazdelitev podatkov neenakomerna. To lahko popravite! Preden to storite, raziščite še malo.

1. Ugotovite, koliko podatkov je na voljo za vsako kuhinjo, in jih izpišite:

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

    Izpis izgleda takole:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Odkrijte sestavine

Zdaj lahko globlje raziščete podatke in ugotovite, katere so tipične sestavine za posamezno kuhinjo. Odstraniti morate ponavljajoče se podatke, ki povzročajo zmedo med kuhinjami, zato se lotimo tega problema.

1. Ustvarite funkcijo `create_ingredient()` v Pythonu za ustvarjanje podatkovnega okvira sestavin. Ta funkcija bo začela z odstranitvijo neuporabnega stolpca in razvrstila sestavine po njihovem številu:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Zdaj lahko uporabite to funkcijo, da dobite idejo o desetih najbolj priljubljenih sestavinah po kuhinji.

1. Pokličite `create_ingredient()` in jo prikažite z uporabo `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![tajska](../../../../4-Classification/1-Introduction/images/thai.png)

1. Enako storite za japonske podatke:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japonska](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Zdaj za kitajske sestavine:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kitajska](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Prikažite indijske sestavine:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indijska](../../../../4-Classification/1-Introduction/images/indian.png)

1. Na koncu prikažite korejske sestavine:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korejska](../../../../4-Classification/1-Introduction/images/korean.png)

1. Zdaj odstranite najpogostejše sestavine, ki povzročajo zmedo med različnimi kuhinjami, z uporabo `drop()`:

   Vsi imajo radi riž, česen in ingver!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Uravnotežite nabor podatkov

Zdaj, ko ste očistili podatke, uporabite [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - za njihovo uravnoteženje.

1. Pokličite `fit_resample()`, ta strategija ustvari nove vzorce z interpolacijo.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Z uravnoteženjem podatkov boste dosegli boljše rezultate pri klasifikaciji. Pomislite na binarno klasifikacijo. Če je večina vaših podatkov enega razreda, bo model strojnega učenja ta razred pogosteje napovedal, zgolj zato, ker je zanj več podatkov. Uravnoteženje podatkov odstrani to neravnovesje.

1. Zdaj lahko preverite število oznak na sestavino:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Vaš izpis izgleda takole:

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

    Podatki so zdaj čisti, uravnoteženi in zelo okusni!

1. Zadnji korak je shranjevanje uravnoteženih podatkov, vključno z oznakami in značilnostmi, v nov podatkovni okvir, ki ga lahko izvozite v datoteko:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Še enkrat si oglejte podatke z uporabo `transformed_df.head()` in `transformed_df.info()`. Shranite kopijo teh podatkov za uporabo v prihodnjih lekcijah:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Ta svež CSV je zdaj na voljo v korenski mapi podatkov.

---

## 🚀Izziv

Ta učni načrt vsebuje več zanimivih naborov podatkov. Prebrskajte mape `data` in preverite, ali katera vsebuje nabore podatkov, ki bi bili primerni za binarno ali večrazredno klasifikacijo? Na katera vprašanja bi lahko odgovorili s tem naborom podatkov?

## [Po-lekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Raziščite API za SMOTE. Za katere primere uporabe je najbolj primeren? Katere težave rešuje?

## Naloga 

[Raziščite metode klasifikacije](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.