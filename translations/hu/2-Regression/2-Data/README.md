<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T15:24:45+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "hu"
}
-->
# Készítsünk regressziós modellt Scikit-learn segítségével: adatok előkészítése és vizualizálása

![Adatvizualizációs infografika](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ez a lecke elérhető R-ben is!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Bevezetés

Most, hogy rendelkezésedre állnak azok az eszközök, amelyekkel elkezdheted a gépi tanulási modellek építését Scikit-learn segítségével, készen állsz arra, hogy kérdéseket tegyél fel az adataiddal kapcsolatban. Amikor adatokkal dolgozol és gépi tanulási megoldásokat alkalmazol, nagyon fontos, hogy megtanuld, hogyan tegyél fel megfelelő kérdéseket, hogy kiaknázhasd az adathalmazodban rejlő lehetőségeket.

Ebben a leckében megtanulod:

- Hogyan készítsd elő az adataidat a modellépítéshez.
- Hogyan használd a Matplotlibet adatvizualizációhoz.

## Hogyan tegyél fel megfelelő kérdést az adataiddal kapcsolatban?

Az a kérdés, amelyre választ szeretnél kapni, meghatározza, hogy milyen típusú gépi tanulási algoritmusokat fogsz használni. A kapott válasz minősége pedig nagymértékben függ az adataid természetétől.

Nézd meg a [leckéhez biztosított adatokat](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv). Ezt a .csv fájlt megnyithatod VS Code-ban. Egy gyors átnézés azonnal megmutatja, hogy vannak hiányzó értékek, valamint szöveges és numerikus adatok keveréke. Van egy furcsa oszlop is, amelyet "Package"-nek hívnak, ahol az adatok között szerepelnek például "sacks", "bins" és más értékek. Az adatok valójában elég zűrösek.

[![ML kezdőknek - Hogyan elemezzünk és tisztítsunk egy adathalmazt](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML kezdőknek - Hogyan elemezzünk és tisztítsunk egy adathalmazt")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja az adatok előkészítését ehhez a leckéhez.

Valójában nem túl gyakori, hogy egy adathalmaz teljesen készen áll arra, hogy gépi tanulási modellt készítsünk belőle. Ebben a leckében megtanulod, hogyan készíts elő egy nyers adathalmazt standard Python könyvtárak segítségével. Emellett különböző technikákat is megismerhetsz az adatok vizualizálására.

## Esettanulmány: "a tökpiac"

Ebben a mappában találsz egy .csv fájlt a gyökér `data` mappában, amelynek neve [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv). Ez a fájl 1757 sor adatot tartalmaz a tökpiacról, városok szerint csoportosítva. Ez nyers adat, amelyet az [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) oldalról származtatott az Egyesült Államok Mezőgazdasági Minisztériuma.

### Adatok előkészítése

Ezek az adatok közkincsnek számítanak. Az USDA weboldaláról külön fájlokban, városonként letölthetők. Az adatok túlzott szétaprózódásának elkerülése érdekében az összes városi adatot egy táblázatba fűztük össze, így már egy kicsit _előkészítettük_ az adatokat. Most nézzük meg közelebbről az adatokat.

### A tökadatok - korai következtetések

Mit veszel észre ezekkel az adatokkal kapcsolatban? Már láttad, hogy van szövegek, számok, hiányzó értékek és furcsa értékek keveréke, amelyeket értelmezni kell.

Milyen kérdést tehetsz fel ezekkel az adatokkal kapcsolatban regressziós technikát alkalmazva? Például: "Előrejelezni egy tök árát egy adott hónapban." Ha újra megnézed az adatokat, láthatod, hogy néhány változtatást kell végrehajtanod, hogy létrehozd a feladathoz szükséges adatstruktúrát.

## Gyakorlat - elemezd a tökadatokat

Használjuk a [Pandas](https://pandas.pydata.org/) könyvtárat (a név a `Python Data Analysis` rövidítése), amely nagyon hasznos az adatok formázásához, hogy elemezzük és előkészítsük a tökadatokat.

### Először ellenőrizd a hiányzó dátumokat

Először lépéseket kell tenned a hiányzó dátumok ellenőrzésére:

1. Konvertáld a dátumokat hónap formátumba (ezek amerikai dátumok, tehát a formátum `MM/DD/YYYY`).
2. Hozz létre egy új oszlopot, amely csak a hónapot tartalmazza.

Nyisd meg a _notebook.ipynb_ fájlt a Visual Studio Code-ban, és importáld a táblázatot egy új Pandas dataframe-be.

1. Használd a `head()` függvényt az első öt sor megtekintéséhez.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Milyen függvényt használnál az utolsó öt sor megtekintéséhez?

1. Ellenőrizd, hogy van-e hiányzó adat az aktuális dataframe-ben:

    ```python
    pumpkins.isnull().sum()
    ```

    Van hiányzó adat, de lehet, hogy ez nem számít a feladat szempontjából.

1. Hogy könnyebben dolgozhass a dataframe-mel, válaszd ki csak azokat az oszlopokat, amelyekre szükséged van, a `loc` függvény segítségével, amely az eredeti dataframe-ből egy sorokból (első paraméter) és oszlopokból (második paraméter) álló csoportot von ki. Az alábbi esetben a `:` kifejezés azt jelenti, hogy "minden sor".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Másodszor, határozd meg a tök átlagárát

Gondold át, hogyan határozhatod meg egy tök átlagárát egy adott hónapban. Mely oszlopokat választanád ehhez a feladathoz? Tipp: három oszlopra lesz szükséged.

Megoldás: vedd az `Low Price` és `High Price` oszlopok átlagát, hogy kitöltsd az új Price oszlopot, és konvertáld a Date oszlopot úgy, hogy csak a hónapot mutassa. Szerencsére az előző ellenőrzés szerint nincs hiányzó adat a dátumok vagy árak esetében.

1. Az átlag kiszámításához add hozzá a következő kódot:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Nyugodtan nyomtass ki bármilyen adatot, amit ellenőrizni szeretnél a `print(month)` segítségével.

2. Most másold át az átalakított adatokat egy új Pandas dataframe-be:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Ha kinyomtatod a dataframe-et, egy tiszta, rendezett adathalmazt fogsz látni, amelyre építheted az új regressziós modelledet.

### De várj! Valami furcsa van itt

Ha megnézed a `Package` oszlopot, a tökök sokféle konfigurációban kerülnek értékesítésre. Néhányat "1 1/9 bushel" mértékegységben, néhányat "1/2 bushel" mértékegységben, néhányat darabonként, néhányat fontonként, és néhányat nagy dobozokban, amelyek szélessége változó.

> Úgy tűnik, hogy a tökök súlyának következetes mérése nagyon nehéz

Ha beleásod magad az eredeti adatokba, érdekes, hogy bármi, aminek `Unit of Sale` értéke 'EACH' vagy 'PER BIN', szintén a `Package` típus szerint van megadva, például hüvelykben, binben vagy darabonként. Úgy tűnik, hogy a tökök súlyának következetes mérése nagyon nehéz, ezért szűrjük őket úgy, hogy csak azokat a tököket válasszuk ki, amelyek `Package` oszlopában szerepel a 'bushel' szó.

1. Adj hozzá egy szűrőt a fájl tetejére, az eredeti .csv importálása alá:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Ha most kinyomtatod az adatokat, láthatod, hogy csak azokat a körülbelül 415 sort kapod, amelyek bushelben mért tököket tartalmaznak.

### De várj! Még egy dolgot meg kell tenni

Észrevetted, hogy a bushel mennyisége soronként változik? Normalizálnod kell az árképzést, hogy bushelre vetítve mutasd az árakat, tehát végezz némi matematikát az árak standardizálásához.

1. Add hozzá ezeket a sorokat a new_pumpkins dataframe létrehozó blokk után:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ A [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) szerint a bushel súlya a termék típusától függ, mivel ez egy térfogatmérés. "Egy bushel paradicsom például 56 fontot kell, hogy nyomjon... A levelek és zöldek több helyet foglalnak kevesebb súllyal, így egy bushel spenót csak 20 font." Ez mind elég bonyolult! Ne foglalkozzunk a bushel-font átváltással, hanem inkább bushelre vetítve árazzunk. Mindez a bushel tökök tanulmányozása azonban megmutatja, mennyire fontos megérteni az adatok természetét!

Most már elemezheted az árképzést egységenként a bushel mértékegység alapján. Ha még egyszer kinyomtatod az adatokat, láthatod, hogyan lett standardizálva.

✅ Észrevetted, hogy a fél bushelben árult tökök nagyon drágák? Ki tudod találni, miért? Tipp: a kis tökök sokkal drágábbak, mint a nagyok, valószínűleg azért, mert sokkal több van belőlük bushelben, tekintve az egy nagy üreges tök által elfoglalt kihasználatlan helyet.

## Vizualizációs stratégiák

Az adatelemzők egyik feladata, hogy bemutassák az adatok minőségét és természetét, amelyekkel dolgoznak. Ehhez gyakran készítenek érdekes vizualizációkat, például diagramokat, grafikonokat és táblázatokat, amelyek az adatok különböző aspektusait mutatják be. Ily módon vizuálisan képesek megmutatni az összefüggéseket és hiányosságokat, amelyeket egyébként nehéz lenne feltárni.

[![ML kezdőknek - Hogyan vizualizáljuk az adatokat Matplotlib segítségével](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML kezdőknek - Hogyan vizualizáljuk az adatokat Matplotlib segítségével")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja az adatok vizualizálását ehhez a leckéhez.

A vizualizációk segíthetnek meghatározni, hogy mely gépi tanulási technika a legmegfelelőbb az adatokhoz. Például egy olyan szórásdiagram, amely látszólag egy vonalat követ, azt jelzi, hogy az adatok jó jelöltek lehetnek egy lineáris regressziós feladathoz.

Egy adatvizualizációs könyvtár, amely jól működik Jupyter notebookokban, a [Matplotlib](https://matplotlib.org/) (amelyet az előző leckében is láttál).

> Szerezz több tapasztalatot az adatvizualizációval [ezekben az oktatóanyagokban](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Gyakorlat - kísérletezz a Matplotlibgel

Próbálj meg néhány alapvető diagramot készíteni, hogy megjelenítsd az új dataframe-et, amelyet éppen létrehoztál. Mit mutatna egy alapvető vonaldiagram?

1. Importáld a Matplotlibet a fájl tetején, a Pandas importálása alatt:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Futtasd újra az egész notebookot a frissítéshez.
1. A notebook alján adj hozzá egy cellát, hogy dobozdiagramot készíts:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Egy szórásdiagram, amely az ár és hónap közötti kapcsolatot mutatja](../../../../2-Regression/2-Data/images/scatterplot.png)

    Hasznos ez a diagram? Meglepett valami vele kapcsolatban?

    Ez nem különösebben hasznos, mivel csak az adataidat mutatja pontok szórásaként egy adott hónapban.

### Tedd hasznossá

Ahhoz, hogy a diagramok hasznos adatokat mutassanak, általában valahogyan csoportosítani kell az adatokat. Próbáljunk meg létrehozni egy diagramot, ahol az y tengely a hónapokat mutatja, és az adatok az eloszlást szemléltetik.

1. Adj hozzá egy cellát, hogy csoportosított oszlopdiagramot készíts:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Egy oszlopdiagram, amely az ár és hónap közötti kapcsolatot mutatja](../../../../2-Regression/2-Data/images/barchart.png)

    Ez egy hasznosabb adatvizualizáció! Úgy tűnik, hogy a tökök legmagasabb ára szeptemberben és októberben van. Ez megfelel az elvárásaidnak? Miért vagy miért nem?

---

## 🚀Kihívás

Fedezd fel a Matplotlib által kínált különböző vizualizációs típusokat. Mely típusok a legmegfelelőbbek regressziós problémákhoz?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Nézd meg az adatvizualizáció különböző módjait. Készíts listát a rendelkezésre álló könyvtárakról, és jegyezd fel, hogy melyek a legjobbak adott típusú feladatokhoz, például 2D vizualizációkhoz vagy 3D vizualizációkhoz. Mit fedezel fel?

## Feladat

[Adatvizualizáció felfedezése](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Fontos információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.