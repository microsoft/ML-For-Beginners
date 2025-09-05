<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T15:21:00+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "hu"
}
-->
# Kezdjük el a Python és a Scikit-learn használatát regressziós modellekhez

![Vázlat a regressziókról](../../../../sketchnotes/ml-regression.png)

> Vázlatrajz: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ez a lecke R nyelven is elérhető!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Bevezetés

Ebben a négy leckében megtanulhatod, hogyan építs regressziós modelleket. Rövidesen megbeszéljük, hogy mire használhatók ezek. De mielőtt bármibe belekezdenél, győződj meg róla, hogy a megfelelő eszközök rendelkezésre állnak a folyamat elindításához!

Ebben a leckében megtanulod:

- Hogyan konfiguráld a számítógéped helyi gépi tanulási feladatokhoz.
- Hogyan dolgozz Jupyter notebookokkal.
- Hogyan használd a Scikit-learn könyvtárat, beleértve annak telepítését.
- Hogyan fedezd fel a lineáris regressziót egy gyakorlati feladaton keresztül.

## Telepítések és konfigurációk

[![ML kezdőknek - Eszközök beállítása gépi tanulási modellek építéséhez](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML kezdőknek - Eszközök beállítása gépi tanulási modellek építéséhez")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja, hogyan konfiguráld a számítógéped a gépi tanuláshoz.

1. **Telepítsd a Python-t**. Győződj meg róla, hogy a [Python](https://www.python.org/downloads/) telepítve van a számítógépeden. A Python-t számos adatfeldolgozási és gépi tanulási feladathoz fogod használni. A legtöbb számítógépes rendszer már tartalmaz Python telepítést. Hasznosak lehetnek a [Python Coding Pack-ek](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) is, amelyek megkönnyítik a beállítást néhány felhasználó számára.

   A Python bizonyos használati módjai azonban eltérő verziókat igényelhetnek. Ezért hasznos lehet egy [virtuális környezetben](https://docs.python.org/3/library/venv.html) dolgozni.

2. **Telepítsd a Visual Studio Code-ot**. Győződj meg róla, hogy a Visual Studio Code telepítve van a számítógépedre. Kövesd ezeket az utasításokat a [Visual Studio Code telepítéséhez](https://code.visualstudio.com/). Ebben a kurzusban a Python-t a Visual Studio Code-ban fogod használni, ezért érdemes lehet felfrissíteni a tudásodat arról, hogyan [konfiguráld a Visual Studio Code-ot](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) Python fejlesztéshez.

   > Ismerkedj meg a Python-nal ezeknek a [Learn moduloknak](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) a segítségével.
   >
   > [![Python beállítása a Visual Studio Code-ban](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Python beállítása a Visual Studio Code-ban")
   >
   > 🎥 Kattints a fenti képre egy videóért: Python használata a VS Code-ban.

3. **Telepítsd a Scikit-learn-t**, a [következő utasítások](https://scikit-learn.org/stable/install.html) alapján. Mivel Python 3-at kell használnod, ajánlott egy virtuális környezet használata. Ha M1 Mac-en telepíted ezt a könyvtárat, különleges utasításokat találsz a fenti oldalon.

4. **Telepítsd a Jupyter Notebook-ot**. Telepítsd a [Jupyter csomagot](https://pypi.org/project/jupyter/).

## A gépi tanulási fejlesztési környezeted

A Python kód fejlesztéséhez és gépi tanulási modellek létrehozásához **notebookokat** fogsz használni. Ez a fájltípus az adatkutatók körében gyakori eszköz, és `.ipynb` kiterjesztéssel azonosítható.

A notebookok interaktív környezetet biztosítanak, amely lehetővé teszi a fejlesztő számára, hogy kódot írjon, jegyzeteket készítsen, és dokumentációt írjon a kód köré, ami különösen hasznos kísérleti vagy kutatási projektek esetén.

[![ML kezdőknek - Jupyter Notebookok beállítása regressziós modellek építéséhez](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML kezdőknek - Jupyter Notebookok beállítása regressziós modellek építéséhez")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja ezt a gyakorlatot.

### Gyakorlat - dolgozz egy notebookkal

Ebben a mappában megtalálod a _notebook.ipynb_ fájlt.

1. Nyisd meg a _notebook.ipynb_ fájlt a Visual Studio Code-ban.

   Egy Jupyter szerver indul el Python 3+ környezettel. A notebookban olyan részeket találsz, amelyek `futtathatók`, azaz kódrészletek. Egy kódrészletet a lejátszás gombra hasonlító ikon kiválasztásával futtathatsz.

2. Válaszd ki az `md` ikont, és adj hozzá egy kis markdown szöveget, például a következőt: **# Üdvözöllek a notebookodban**.

   Ezután adj hozzá egy kis Python kódot.

3. Írd be a következő kódot: **print('hello notebook')**.
4. Kattints a nyílra a kód futtatásához.

   A következő kimenetet kell látnod:

    ```output
    hello notebook
    ```

![VS Code egy megnyitott notebookkal](../../../../2-Regression/1-Tools/images/notebook.jpg)

A kódot megjegyzésekkel egészítheted ki, hogy önmagad számára dokumentáld a notebookot.

✅ Gondolkodj el egy percre azon, hogy mennyire különbözik egy webfejlesztő munkakörnyezete egy adatkutatóétól.

## Scikit-learn használatának elsajátítása

Most, hogy a Python be van állítva a helyi környezetedben, és kényelmesen használod a Jupyter notebookokat, ismerkedj meg a Scikit-learn-nel (ejtsd: `száj` mint a `science`). A Scikit-learn egy [kiterjedt API-t](https://scikit-learn.org/stable/modules/classes.html#api-ref) biztosít, amely segít a gépi tanulási feladatok elvégzésében.

A [weboldaluk](https://scikit-learn.org/stable/getting_started.html) szerint: "A Scikit-learn egy nyílt forráskódú gépi tanulási könyvtár, amely támogatja a felügyelt és felügyelet nélküli tanulást. Emellett különféle eszközöket biztosít a modellillesztéshez, adat-előfeldolgozáshoz, modellkiválasztáshoz és értékeléshez, valamint számos egyéb segédprogramhoz."

Ebben a kurzusban a Scikit-learn-t és más eszközöket fogsz használni gépi tanulási modellek építéséhez, hogy úgynevezett 'hagyományos gépi tanulási' feladatokat végezz. Szándékosan kerültük a neurális hálózatokat és a mélytanulást, mivel ezek jobban lefedhetők a hamarosan megjelenő 'AI kezdőknek' tananyagunkban.

A Scikit-learn egyszerűvé teszi a modellek építését és értékelését. Elsősorban numerikus adatok használatára összpontosít, és számos előre elkészített adathalmazt tartalmaz tanulási célokra. Emellett előre elkészített modelleket is tartalmaz, amelyeket a diákok kipróbálhatnak. Fedezzük fel a folyamatot, amely során előre csomagolt adatokat töltünk be, és egy beépített becslőt használunk az első ML modellünkhöz a Scikit-learn segítségével.

## Gyakorlat - az első Scikit-learn notebookod

> Ez az oktatóanyag a Scikit-learn weboldalán található [lineáris regressziós példa](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) alapján készült.

[![ML kezdőknek - Az első lineáris regressziós projekted Python-ban](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML kezdőknek - Az első lineáris regressziós projekted Python-ban")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja ezt a gyakorlatot.

A leckéhez tartozó _notebook.ipynb_ fájlban töröld ki az összes cellát a 'szemetes' ikonra kattintva.

Ebben a részben egy kis, a Scikit-learn-be beépített diabétesz adathalmazzal fogsz dolgozni tanulási célokra. Képzeld el, hogy egy kezelést szeretnél tesztelni cukorbetegek számára. A gépi tanulási modellek segíthetnek meghatározni, hogy mely betegek reagálnának jobban a kezelésre, a változók kombinációi alapján. Még egy nagyon alapvető regressziós modell is, ha vizualizáljuk, információt nyújthat a változókról, amelyek segíthetnek a klinikai vizsgálatok megszervezésében.

✅ Számos regressziós módszer létezik, és hogy melyiket választod, az attól függ, milyen kérdésre keresel választ. Ha például egy adott korú személy várható magasságát szeretnéd megjósolni, lineáris regressziót használnál, mivel egy **numerikus értéket** keresel. Ha viszont azt szeretnéd megtudni, hogy egy konyha típusa vegánnak tekinthető-e vagy sem, akkor egy **kategória-hozzárendelést** keresel, így logisztikus regressziót használnál. Később többet megtudhatsz a logisztikus regresszióról. Gondolkodj el azon, hogy milyen kérdéseket tehetsz fel az adatokkal kapcsolatban, és melyik módszer lenne megfelelőbb.

Kezdjünk neki ennek a feladatnak.

### Könyvtárak importálása

Ehhez a feladathoz néhány könyvtárat fogunk importálni:

- **matplotlib**. Ez egy hasznos [grafikonkészítő eszköz](https://matplotlib.org/), amelyet vonaldiagramok készítésére fogunk használni.
- **numpy**. A [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) egy hasznos könyvtár numerikus adatok kezelésére Python-ban.
- **sklearn**. Ez a [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) könyvtár.

Importálj néhány könyvtárat a feladatok elvégzéséhez.

1. Add hozzá az importokat az alábbi kód beírásával:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   A fenti kódban importálod a `matplotlib`-et, a `numpy`-t, valamint a `datasets`, `linear_model` és `model_selection` modulokat a `sklearn`-ből. A `model_selection` a teszt- és tanulóhalmazok szétválasztására szolgál.

### A diabétesz adathalmaz

A beépített [diabétesz adathalmaz](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 442 diabéteszhez kapcsolódó mintát tartalmaz, 10 jellemző változóval, amelyek közül néhány:

- age: életkor években
- bmi: testtömegindex
- bp: átlagos vérnyomás
- s1 tc: T-sejtek (egy típusú fehérvérsejtek)

✅ Ez az adathalmaz tartalmazza a 'nem' fogalmát, mint a diabétesz kutatás szempontjából fontos jellemző változót. Számos orvosi adathalmaz tartalmaz ilyen típusú bináris osztályozást. Gondolkodj el azon, hogy az ilyen kategorizálások hogyan zárhatnak ki bizonyos népességcsoportokat a kezelésekből.

Most töltsd be az X és y adatokat.

> 🎓 Ne feledd, hogy ez felügyelt tanulás, és szükségünk van egy megnevezett 'y' célváltozóra.

Egy új kódcellában töltsd be a diabétesz adathalmazt a `load_diabetes()` hívásával. A `return_X_y=True` bemenet jelzi, hogy az `X` egy adatmátrix lesz, az `y` pedig a regressziós cél.

1. Adj hozzá néhány print parancsot, hogy megjelenítsd az adatmátrix alakját és az első elemét:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Amit válaszként kapsz, az egy tuple. Amit csinálsz, az az, hogy a tuple első két értékét hozzárendeled az `X`-hez és az `y`-hoz. Tudj meg többet a [tuple-ökről](https://wikipedia.org/wiki/Tuple).

    Láthatod, hogy ezek az adatok 442 elemet tartalmaznak, amelyek 10 elemből álló tömbökbe vannak rendezve:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Gondolkodj el az adatok és a regressziós cél közötti kapcsolaton. A lineáris regresszió az X jellemző és az y célváltozó közötti kapcsolatot jósolja meg. Megtalálod a [célváltozót](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) a diabétesz adathalmaz dokumentációjában? Mit mutat ez az adathalmaz a célváltozó alapján?

2. Ezután válassz ki egy részt ebből az adathalmazból, hogy ábrázolhasd, például az adathalmaz 3. oszlopát. Ezt a `:` operátorral teheted meg, hogy kiválaszd az összes sort, majd az index (2) segítségével kiválaszd a 3. oszlopot. Az adatokat 2D tömbbé is átalakíthatod - ahogy az ábrázoláshoz szükséges - a `reshape(n_rows, n_columns)` használatával. Ha az egyik paraméter -1, a megfelelő dimenzió automatikusan kiszámításra kerül.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Bármikor nyomtasd ki az adatokat, hogy ellenőrizd az alakjukat.

3. Most, hogy az adatok készen állnak az ábrázolásra, megnézheted, hogy egy gép segíthet-e logikus határvonalat húzni az adathalmaz számai között. Ehhez szét kell választanod az adatokat (X) és a célváltozót (y) teszt- és tanulóhalmazokra. A Scikit-learn egyszerű módot kínál erre; az adataidat egy adott ponton oszthatod szét.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Most készen állsz a modell betanítására! Töltsd be a lineáris regressziós modellt, és tanítsd be az X és y tanulóhalmazokkal a `model.fit()` használatával:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ A `model.fit()` egy olyan függvény, amelyet sok ML könyvtárban, például a TensorFlow-ban is láthatsz.

5. Ezután hozz létre egy előrejelzést a tesztadatok alapján a `predict()` függvény használatával. Ezt fogod használni a vonal meghúzásához az adathalmaz csoportjai között.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Most itt az ideje, hogy megjelenítsd az adatokat egy diagramon. A Matplotlib
✅ Gondolkodj el egy kicsit azon, mi történik itt. Egy egyenes vonal halad át sok apró adatponton, de pontosan mit csinál? Látod, hogyan tudnád ezt a vonalat felhasználni arra, hogy megjósold, hol helyezkedne el egy új, még nem látott adatpont a grafikon y tengelyéhez viszonyítva? Próbáld meg szavakba önteni ennek a modellnek a gyakorlati hasznát.

Gratulálok, elkészítetted az első lineáris regressziós modelledet, készítettél vele egy előrejelzést, és megjelenítetted egy grafikonon!

---
## 🚀Kihívás

Ábrázolj egy másik változót ebből az adatállományból. Tipp: szerkeszd ezt a sort: `X = X[:,2]`. Ennek az adatállománynak a célértéke alapján mit tudsz felfedezni a cukorbetegség betegségként való előrehaladásáról?
## [Utólagos kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Ebben a bemutatóban egyszerű lineáris regresszióval dolgoztál, nem pedig univariáns vagy többszörös lineáris regresszióval. Olvass egy kicsit ezeknek a módszereknek a különbségeiről, vagy nézd meg [ezt a videót](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Olvass többet a regresszió fogalmáról, és gondolkodj el azon, milyen típusú kérdésekre lehet választ adni ezzel a technikával. Vegyél részt [ebben a bemutatóban](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott), hogy elmélyítsd a tudásodat.

## Feladat

[Egy másik adatállomány](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Fontos információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.