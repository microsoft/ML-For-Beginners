<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T13:17:37+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "hr"
}
-->
# Uvod u klasifikaciju

U ovih četiri lekcije istražit ćete temeljni fokus klasičnog strojnog učenja - _klasifikaciju_. Proći ćemo kroz korištenje različitih algoritama klasifikacije s datasetom o svim briljantnim kuhinjama Azije i Indije. Nadamo se da ste gladni!

![samo prstohvat!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Proslavite pan-azijske kuhinje u ovim lekcijama! Slika: [Jen Looper](https://twitter.com/jenlooper)

Klasifikacija je oblik [nadziranog učenja](https://wikipedia.org/wiki/Supervised_learning) koji ima mnogo zajedničkog s tehnikama regresije. Ako je strojno učenje usmjereno na predviđanje vrijednosti ili naziva stvari pomoću podataka, tada se klasifikacija općenito dijeli u dvije skupine: _binarna klasifikacija_ i _višeklasna klasifikacija_.

[![Uvod u klasifikaciju](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Uvod u klasifikaciju")

> 🎥 Kliknite na sliku iznad za video: John Guttag s MIT-a predstavlja klasifikaciju

Zapamtite:

- **Linearna regresija** pomogla vam je predvidjeti odnose između varijabli i napraviti točna predviđanja o tome gdje će novi podatak pasti u odnosu na tu liniju. Na primjer, mogli ste predvidjeti _koja će cijena bundeve biti u rujnu u usporedbi s prosincem_.
- **Logistička regresija** pomogla vam je otkriti "binarne kategorije": na ovoj cjenovnoj razini, _je li bundeva narančasta ili nije narančasta_?

Klasifikacija koristi različite algoritme za određivanje drugih načina dodjeljivanja oznake ili klase podatkovnoj točki. Radit ćemo s ovim podacima o kuhinjama kako bismo vidjeli možemo li, promatrajući skup sastojaka, odrediti iz koje kuhinje potječu.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ova lekcija dostupna je i na R jeziku!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Uvod

Klasifikacija je jedna od temeljnih aktivnosti istraživača strojnog učenja i podatkovnih znanstvenika. Od osnovne klasifikacije binarne vrijednosti ("je li ovaj email spam ili nije?"), do složene klasifikacije i segmentacije slika pomoću računalnog vida, uvijek je korisno moći razvrstati podatke u klase i postavljati pitanja o njima.

Da bismo proces izrazili na znanstveniji način, vaša metoda klasifikacije stvara prediktivni model koji vam omogućuje mapiranje odnosa između ulaznih varijabli i izlaznih varijabli.

![binarna vs. višeklasna klasifikacija](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Binarni i višeklasni problemi za algoritme klasifikacije. Infografika: [Jen Looper](https://twitter.com/jenlooper)

Prije nego što započnemo proces čišćenja podataka, njihove vizualizacije i pripreme za zadatke strojnog učenja, naučimo malo više o različitim načinima na koje se strojno učenje može koristiti za klasifikaciju podataka.

Izvedena iz [statistike](https://wikipedia.org/wiki/Statistical_classification), klasifikacija pomoću klasičnog strojnog učenja koristi značajke, poput `smoker`, `weight` i `age`, kako bi odredila _vjerojatnost razvoja određene bolesti_. Kao tehnika nadziranog učenja slična regresijskim vježbama koje ste ranije izvodili, vaši podaci su označeni, a algoritmi strojnog učenja koriste te oznake za klasifikaciju i predviđanje klasa (ili 'značajki') skupa podataka te njihovo dodjeljivanje grupi ili ishodu.

✅ Zastanite na trenutak i zamislite skup podataka o kuhinjama. Na što bi višeklasni model mogao odgovoriti? Na što bi binarni model mogao odgovoriti? Što ako želite odrediti koristi li određena kuhinja vjerojatno piskavicu? Što ako želite vidjeti možete li, s obzirom na vrećicu namirnica punu zvjezdastog anisa, artičoka, cvjetače i hrena, pripremiti tipično indijsko jelo?

[![Lude misteriozne košare](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Lude misteriozne košare")

> 🎥 Kliknite na sliku iznad za video. Cijela premisa emisije 'Chopped' je 'misteriozna košara' u kojoj kuhari moraju pripremiti jelo od nasumično odabranih sastojaka. Sigurno bi ML model pomogao!

## Pozdrav 'klasifikatoru'

Pitanje koje želimo postaviti ovom skupu podataka o kuhinjama zapravo je **višeklasno pitanje**, jer imamo nekoliko potencijalnih nacionalnih kuhinja s kojima radimo. S obzirom na skup sastojaka, kojoj od ovih mnogih klasa će podaci pripadati?

Scikit-learn nudi nekoliko različitih algoritama za klasifikaciju podataka, ovisno o vrsti problema koji želite riješiti. U sljedeće dvije lekcije naučit ćete o nekoliko tih algoritama.

## Vježba - očistite i uravnotežite svoje podatke

Prvi zadatak prije početka ovog projekta je očistiti i **uravnotežiti** svoje podatke kako biste dobili bolje rezultate. Počnite s praznom datotekom _notebook.ipynb_ u korijenu ove mape.

Prvo što trebate instalirati je [imblearn](https://imbalanced-learn.org/stable/). Ovo je Scikit-learn paket koji će vam omogućiti bolje uravnoteženje podataka (o ovom zadatku ćete naučiti više za trenutak).

1. Za instalaciju `imblearn`, pokrenite `pip install`, ovako:

    ```python
    pip install imblearn
    ```

1. Uvezite pakete potrebne za uvoz i vizualizaciju podataka, također uvezite `SMOTE` iz `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Sada ste spremni za uvoz podataka.

1. Sljedeći zadatak je uvoz podataka:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Korištenje `read_csv()` učitat će sadržaj csv datoteke _cusines.csv_ i smjestiti ga u varijablu `df`.

1. Provjerite oblik podataka:

    ```python
    df.head()
    ```

   Prvih pet redaka izgleda ovako:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Dobijte informacije o ovim podacima pozivom `info()`:

    ```python
    df.info()
    ```

    Vaš izlaz izgleda ovako:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Vježba - učenje o kuhinjama

Sada rad postaje zanimljiviji. Otkrijmo distribuciju podataka po kuhinji.

1. Prikaz podataka kao stupaca pozivom `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![distribucija podataka o kuhinjama](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Postoji ograničen broj kuhinja, ali distribucija podataka je neujednačena. To možete popraviti! Prije toga, istražite malo više.

1. Saznajte koliko je podataka dostupno po kuhinji i ispišite ih:

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

    izlaz izgleda ovako:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Otkrijte sastojke

Sada možete dublje istražiti podatke i saznati koji su tipični sastojci po kuhinji. Trebali biste očistiti ponavljajuće podatke koji stvaraju konfuziju između kuhinja, pa naučimo o ovom problemu.

1. Napravite funkciju `create_ingredient()` u Pythonu za stvaranje dataframea sastojaka. Ova funkcija će započeti uklanjanjem nevažnog stupca i sortirati sastojke prema njihovom broju:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Sada možete koristiti tu funkciju kako biste dobili ideju o deset najpopularnijih sastojaka po kuhinji.

1. Pozovite `create_ingredient()` i prikažite rezultate pozivom `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![tajlandska](../../../../4-Classification/1-Introduction/images/thai.png)

1. Učinite isto za japanske podatke:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanska](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Sada za kineske sastojke:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kineska](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Prikažite indijske sastojke:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indijska](../../../../4-Classification/1-Introduction/images/indian.png)

1. Na kraju, prikažite korejske sastojke:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korejska](../../../../4-Classification/1-Introduction/images/korean.png)

1. Sada uklonite najčešće sastojke koji stvaraju konfuziju između različitih kuhinja, pozivom `drop()`:

   Svi vole rižu, češnjak i đumbir!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Uravnotežite skup podataka

Sada kada ste očistili podatke, koristite [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Tehnika sintetičkog manjinskog uzorkovanja" - za uravnoteženje podataka.

1. Pozovite `fit_resample()`, ova strategija generira nove uzorke interpolacijom.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Uravnoteženjem podataka, postići ćete bolje rezultate pri klasifikaciji. Razmislite o binarnoj klasifikaciji. Ako većina vaših podataka pripada jednoj klasi, ML model će češće predviđati tu klasu, samo zato što za nju ima više podataka. Uravnoteženje podataka uklanja ovu neravnotežu.

1. Sada možete provjeriti broj oznaka po sastojku:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Vaš izlaz izgleda ovako:

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

    Podaci su sada čisti, uravnoteženi i vrlo ukusni!

1. Posljednji korak je spremanje uravnoteženih podataka, uključujući oznake i značajke, u novi dataframe koji se može izvesti u datoteku:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Možete još jednom pogledati podatke koristeći `transformed_df.head()` i `transformed_df.info()`. Spremite kopiju ovih podataka za korištenje u budućim lekcijama:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Ovaj svježi CSV sada se nalazi u korijenskoj mapi podataka.

---

## 🚀Izazov

Ovaj kurikulum sadrži nekoliko zanimljivih skupova podataka. Pregledajte mape `data` i provjerite sadrže li neki skupovi podataka koji bi bili prikladni za binarnu ili višeklasnu klasifikaciju? Koja biste pitanja postavili ovom skupu podataka?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Istražite SMOTE-ov API. Za koje slučajeve upotrebe je najprikladniji? Koje probleme rješava?

## Zadatak 

[Istrazite metode klasifikacije](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.