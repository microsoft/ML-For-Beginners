<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T16:25:51+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "hu"
}
-->
# Bevezetés az osztályozásba

Ebben a négy leckében a klasszikus gépi tanulás egyik alapvető területét, az _osztályozást_ fogod megismerni. Különböző osztályozási algoritmusokat fogunk alkalmazni egy adatállományon, amely Ázsia és India csodálatos konyháiról szól. Reméljük, éhes vagy!

![csak egy csipet!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Ünnepeld a pán-ázsiai konyhákat ezekben a leckékben! Kép: [Jen Looper](https://twitter.com/jenlooper)

Az osztályozás a [felügyelt tanulás](https://wikipedia.org/wiki/Supervised_learning) egyik formája, amely sok hasonlóságot mutat a regressziós technikákkal. Ha a gépi tanulás lényege az, hogy adatállományok segítségével értékeket vagy neveket jósoljunk meg, akkor az osztályozás általában két csoportba sorolható: _bináris osztályozás_ és _többosztályos osztályozás_.

[![Bevezetés az osztályozásba](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Bevezetés az osztályozásba")

> 🎥 Kattints a fenti képre egy videóért: MIT John Guttag bemutatja az osztályozást

Emlékezz:

- **Lineáris regresszió** segített megjósolni a változók közötti kapcsolatokat, és pontos előrejelzéseket készíteni arról, hogy egy új adatpont hol helyezkedne el a vonalhoz viszonyítva. Például meg tudtad jósolni, _milyen árú lesz egy tök szeptemberben vagy decemberben_.
- **Logisztikus regresszió** segített felfedezni "bináris kategóriákat": ezen az árponton _ez a tök narancssárga vagy nem narancssárga_?

Az osztályozás különböző algoritmusokat használ annak meghatározására, hogy egy adatpont milyen címkét vagy osztályt kapjon. Dolgozzunk ezzel a konyhai adatállománnyal, hogy megállapítsuk, egy összetevőcsoport alapján melyik konyha eredetéhez tartozik.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ez a lecke elérhető R-ben is!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Bevezetés

Az osztályozás a gépi tanulás kutatójának és adatkutatójának egyik alapvető tevékenysége. Az egyszerű bináris értékek osztályozásától ("ez az e-mail spam vagy nem?") a komplex képosztályozásig és szegmentálásig számítógépes látás segítségével, mindig hasznos az adatokat osztályokba rendezni és kérdéseket feltenni róluk.

Tudományosabb megfogalmazásban az osztályozási módszered egy prediktív modellt hoz létre, amely lehetővé teszi, hogy az input változók és az output változók közötti kapcsolatot feltérképezd.

![bináris vs. többosztályos osztályozás](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Bináris vs. többosztályos problémák, amelyeket az osztályozási algoritmusok kezelnek. Infografika: [Jen Looper](https://twitter.com/jenlooper)

Mielőtt elkezdenénk az adatok tisztítását, vizualizálását és előkészítését a gépi tanulási feladatokhoz, ismerjük meg, hogyan lehet a gépi tanulást különböző módokon alkalmazni az adatok osztályozására.

A [statisztikából](https://wikipedia.org/wiki/Statistical_classification) származó klasszikus gépi tanulási osztályozás olyan jellemzőket használ, mint például `dohányos`, `súly` és `életkor`, hogy meghatározza _X betegség kialakulásának valószínűségét_. A korábban végzett regressziós gyakorlatokhoz hasonló felügyelt tanulási technikaként az adataid címkézettek, és a gépi tanulási algoritmusok ezeket a címkéket használják az adatok osztályainak (vagy 'jellemzőinek') osztályozására és előrejelzésére, majd egy csoporthoz vagy eredményhez rendelésére.

✅ Képzelj el egy konyhákról szóló adatállományt. Milyen kérdéseket tudna megválaszolni egy többosztályos modell? Milyen kérdéseket tudna megválaszolni egy bináris modell? Mi lenne, ha meg akarnád határozni, hogy egy adott konyha valószínűleg használ-e görögszénát? Mi lenne, ha azt akarnád kideríteni, hogy egy élelmiszerkosárban található csillagánizs, articsóka, karfiol és torma alapján készíthetsz-e egy tipikus indiai ételt?

[![Őrült rejtélyes kosarak](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Őrült rejtélyes kosarak")

> 🎥 Kattints a fenti képre egy videóért. A 'Chopped' című műsor egész koncepciója a 'rejtélyes kosár', ahol a séfeknek egy véletlenszerű összetevőkből kell ételt készíteniük. Biztosan segített volna egy gépi tanulási modell!

## Helló 'osztályozó'

Az a kérdés, amit a konyhai adatállománytól szeretnénk megkérdezni, valójában egy **többosztályos kérdés**, mivel több lehetséges nemzeti konyhával dolgozunk. Egy adag összetevő alapján, melyik osztályba illik az adat?

A Scikit-learn több különböző algoritmust kínál az adatok osztályozására, attól függően, hogy milyen problémát szeretnél megoldani. A következő két leckében megismerhetsz néhányat ezek közül az algoritmusok közül.

## Gyakorlat - tisztítsd és egyensúlyozd ki az adataidat

Az első feladat, mielőtt elkezdenénk ezt a projektet, az adatok tisztítása és **kiegyensúlyozása**, hogy jobb eredményeket érjünk el. Kezdd a _notebook.ipynb_ üres fájllal a mappa gyökerében.

Az első telepítendő csomag az [imblearn](https://imbalanced-learn.org/stable/). Ez egy Scikit-learn csomag, amely lehetővé teszi az adatok jobb kiegyensúlyozását (erről a feladatról hamarosan többet fogsz tanulni).

1. Az `imblearn` telepítéséhez futtasd a `pip install` parancsot, így:

    ```python
    pip install imblearn
    ```

1. Importáld a szükséges csomagokat az adatok importálásához és vizualizálásához, valamint importáld a `SMOTE`-ot az `imblearn`-ből.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Most készen állsz az adatok importálására.

1. A következő feladat az adatok importálása:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   A `read_csv()` segítségével beolvashatod a _cusines.csv_ fájl tartalmát, és elhelyezheted a `df` változóban.

1. Ellenőrizd az adatok alakját:

    ```python
    df.head()
    ```

   Az első öt sor így néz ki:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Szerezz információt az adatokról az `info()` hívásával:

    ```python
    df.info()
    ```

    Az eredményed hasonló lesz:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Gyakorlat - konyhák megismerése

Most kezd igazán érdekessé válni a munka. Fedezzük fel az adatok eloszlását konyhánként.

1. Ábrázold az adatokat oszlopokként a `barh()` hívásával:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![konyhai adatok eloszlása](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Véges számú konyha van, de az adatok eloszlása egyenetlen. Ezt kijavíthatod! Mielőtt ezt megtennéd, fedezz fel egy kicsit többet.

1. Derítsd ki, mennyi adat áll rendelkezésre konyhánként, és írasd ki:

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

    Az eredmény így néz ki:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Összetevők felfedezése

Most mélyebben belemerülhetsz az adatokba, és megtudhatod, melyek a tipikus összetevők konyhánként. Tisztítsd ki az ismétlődő adatokat, amelyek zavart okoznak a konyhák között, hogy jobban megértsd ezt a problémát.

1. Hozz létre egy `create_ingredient()` nevű függvényt Pythonban, amely egy összetevő adatkeretet hoz létre. Ez a függvény egy haszontalan oszlop elhagyásával kezd, majd az összetevőket azok számossága szerint rendezi:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Most már használhatod ezt a függvényt, hogy képet kapj a tíz legnépszerűbb összetevőről konyhánként.

1. Hívd meg a `create_ingredient()` függvényt, és ábrázold az adatokat a `barh()` hívásával:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Ugyanezt tedd meg a japán adatokkal:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japán](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Most a kínai összetevők:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![kínai](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Ábrázold az indiai összetevőket:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indiai](../../../../4-Classification/1-Introduction/images/indian.png)

1. Végül ábrázold a koreai összetevőket:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![koreai](../../../../4-Classification/1-Introduction/images/korean.png)

1. Most hagyd el a leggyakoribb összetevőket, amelyek zavart okoznak a különböző konyhák között, a `drop()` hívásával: 

   Mindenki szereti a rizst, fokhagymát és gyömbért!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Az adatállomány kiegyensúlyozása

Most, hogy megtisztítottad az adatokat, használd a [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - technikát az adatok kiegyensúlyozására.

1. Hívd meg a `fit_resample()` függvényt, amely interpolációval új mintákat generál.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Az adatok kiegyensúlyozásával jobb eredményeket érhetsz el az osztályozás során. Gondolj egy bináris osztályozásra. Ha az adataid többsége egy osztályba tartozik, a gépi tanulási modell gyakrabban fogja azt az osztályt előre jelezni, egyszerűen azért, mert több adat áll rendelkezésre róla. Az adatok kiegyensúlyozása segít eltávolítani ezt az egyensúlyhiányt.

1. Most ellenőrizheted az összetevők címkéinek számát:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Az eredményed így néz ki:

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

    Az adatok szépek, tiszták, kiegyensúlyozottak és nagyon ínycsiklandóak!

1. Az utolsó lépés az, hogy a kiegyensúlyozott adatokat, beleértve a címkéket és jellemzőket, egy új adatkeretbe mentsd, amelyet fájlba exportálhatsz:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Még egyszer megnézheted az adatokat a `transformed_df.head()` és `transformed_df.info()` hívásával. Ments egy másolatot ezekről az adatokból, hogy a jövőbeli leckékben használhasd:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Ez a friss CSV most megtalálható az adatállomány mappájának gyökerében.

---

## 🚀Kihívás

Ez a tananyag számos érdekes adatállományt tartalmaz. Nézd át a `data` mappákat, és nézd meg, hogy van-e olyan adatállomány, amely bináris vagy többosztályos osztályozásra alkalmas? Milyen kérdéseket tennél fel ennek az adatállománynak?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Fedezd fel a SMOTE API-t. Milyen felhasználási esetekre a legalkalmasabb? Milyen problémákat old meg?

## Feladat 

[Fedezd fel az osztályozási módszereket](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Fontos információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.