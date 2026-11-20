# Kezdjen Python és Scikit-learn használatával regressziós modellekhez

![Összefoglaló regressziókról sketchnote formában](../../../../translated_images/hu/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote készítette [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ez az óra elérhető R-ben is!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Bevezetés

Ezekben a négy leckében felfedezheti, hogyan építhet regressziós modelleket. Röviden megvitatjuk is, mire valók ezek. De mielőtt bármit is csinálna, győződjön meg róla, hogy a megfelelő eszközök rendelkezésre állnak a folyamat elindításához!

Ebben az órában megtanulja, hogyan:

- Konfigurálja számítógépét helyi gépi tanulási feladatokra.
- Dolgozzon Jupyter Notebook-okkal.
- Használja a Scikit-learn-t, beleértve a telepítést is.
- Fedezze fel a lineáris regressziót egy gyakorlati feladaton keresztül.

## Telepítések és beállítások

[![Kezdőknek ML - Állítsa be eszközeit a gépi tanulási modellek készítéséhez](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "Kezdőknek ML - Állítsa be eszközeit a gépi tanulási modellek készítéséhez")

> 🎥 Kattintson a fenti képre egy rövid videóért, amely bemutatja a számítógép konfigurálását ML-hez.

1. **Telepítse a Pythont**. Győződjön meg róla, hogy [Python](https://www.python.org/downloads/) telepítve van a gépén. A Python sok adat tudományi és gépi tanulási feladathoz használatos. A legtöbb számítógépen már van telepített Python. Hasznos [Python kódcsomagok](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) is elérhetők, amelyek megkönnyítik a beállítást néhány felhasználó számára.

   Azonban a Python használati módjaihoz egy verziót igényelhetnek, míg másokhoz más verziót. Ezért hasznos egy [virtuális környezetben](https://docs.python.org/3/library/venv.html) dolgozni.

2. **Telepítse a Visual Studio Code-ot**. Győződjön meg arról, hogy a Visual Studio Code telepítve van a gépén. Kövesse az utasításokat a [Visual Studio Code telepítéséhez](https://code.visualstudio.com/) az alapvető telepítéshez. Ebben a tanfolyamban Python-t fog használni Visual Studio Code-ban, ezért érdemes felfrissíteni az ismereteit a [Visual Studio Code konfigurálásáról](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) Python fejlesztéshez.

   > Szokjon hozzá a Python használatához ennek a [Learn modul gyűjteménynek](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) a segítségével
   >
   > [![Python beállítása Visual Studio Code-bal](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Python beállítása Visual Studio Code-bal")
   >
   > 🎥 Kattintson a fenti képre egy videóért: Python használata VS Code-ban.

3. **Telepítse a Scikit-learn-t**, kövesse [ezeket az utasításokat](https://scikit-learn.org/stable/install.html). Mivel biztosítani kell, hogy Python 3-at használjon, ajánlott virtuális környezetet használni. Fontos megjegyezni, ha ezt a könyvtárat M1 Mac-re telepíti, külön utasítások vannak a fent linkelt oldalon.

1. **Telepítse a Jupyter Notebookot**. Szüksége lesz a [Jupyter csomag telepítésére](https://pypi.org/project/jupyter/).

## Az Ön gépi tanulási fejlesztőkörnyezete

A Python kód fejlesztéséhez és gépi tanulási modellek létrehozásához **notebookokat** fog használni. Ez a fájltípus gyakori eszköz adat tudósok körében, és `.ipynb` kiterjesztéssel azonosítható.

A notebookok interaktív környezetet biztosítanak, amely lehetővé teszi a fejlesztő számára, hogy egyszerre írjon kódot és jegyzeteket, dokumentációt fűzzön a kód mellé, ami nagyon hasznos kísérleti vagy kutatásorientált projektekhez.

[![Kezdőknek ML - Jupyter Notebookok beállítása regressziós modellek építéséhez](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "Kezdőknek ML - Jupyter Notebookok beállítása regressziós modellek építéséhez")

> 🎥 Kattintson a fenti képre egy rövid videóért, amely végigvezet ezen a gyakorlaton.

### Gyakorlat - dolgozzon notebookkal

Ebben a mappában megtalálja a _notebook.ipynb_ fájlt.

1. Nyissa meg a _notebook.ipynb_ fájlt Visual Studio Code-ban.

   Egy Jupyter szerver fog indulni Python 3+ verzióval. Talál majd a notebookban futtatható részeket, kódrészleteket. Egy kódtömböt futtathat az 'play' gomb alakú ikon kiválasztásával.

1. Válassza ki az `md` ikont, és írjon egy kis markdown szöveget, az alábbi szöveggel: **# Üdvözöljük a notebookjában**.

   Ezután adjon hozzá egy kis Python kódot.

1. Gépelje be a **print('hello notebook')** parancsot a kódtömbbe.
1. Válassza ki a futtatáshoz az ívet ábrázoló nyilat.

   A következő kinyomtatott üzenetet kell látnia:

    ```output
    hello notebook
    ```

![VS Code megnyitott notebookkal](../../../../translated_images/hu/notebook.4a3ee31f396b8832.webp)

Kódját kommentekkel is megfűzheti a notebook önmagában való dokumentálása érdekében.

✅ Gondolja át egy percig, milyen különbségek vannak a webfejlesztő és az adat tudós munkakörnyezete között.

## Scikit-learn használatra készen

Most, hogy a Python beállítása megtörtént a helyi környezetében, és már jól ismeri a Jupyter Notebookokat, ismerkedjünk meg alaposabban a Scikit-learn-nel (kiejtése `sai`, mint a `science`). A Scikit-learn egy [kiterjedt API-t](https://scikit-learn.org/stable/modules/classes.html#api-ref) biztosít, amely segít a gépi tanulási feladatok végrehajtásában.

Honlapjuk szerint [website](https://scikit-learn.org/stable/getting_started.html), "A Scikit-learn egy nyílt forráskódú gépi tanulási könyvtár, amely támogatja a felügyelt és felügyelet nélküli tanulást. Továbbá különféle eszközöket biztosít modellillesztéshez, adat előfeldolgozáshoz, modell kiválasztáshoz és értékeléshez, valamint sok más hasznos funkciót."

Ebben a tanfolyamban a Scikit-learn-t és más eszközöket fog használni, hogy gépi tanulási modelleket építsen, amelyek úgynevezett 'hagyományos gépi tanulási' feladatokat végeznek. Tudatosan kerültük a neurális hálózatokat és mély tanulást, mivel ezek jobban lefedve lesznek a közelgő 'AI kezdőknek' tananyagunkban.

A Scikit-learn megkönnyíti a modellek építését és értékelését használatra. Elsősorban numerikus adatokat használ, és számos előre elkészített adatkészletet tartalmaz tanulási célokra. Továbbá előre elkészített modelleket is tartalmaz a diákoknak. Nézzük meg az előre csomagolt adatok betöltésének és egy beépített becslő használatának folyamatát, hogy létrehozzuk első gépi tanulási modellünket alapvető adatokkal a Scikit-learn segítségével.

## Gyakorlat - az első Scikit-learn notebookja

> Ez a bemutató a Scikit-learn honlapján található [lineáris regresszió példán](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) alapult.


[![ML kezdőknek - Az első lineáris regressziós projektje Pythonban](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML kezdőknek - Az első lineáris regressziós projektje Pythonban")

> 🎥 Kattintson a fenti képre egy rövid videóért, amely bemutatja ezt a gyakorlatot.

A leckéhez tartozó _notebook.ipynb_ fájlban törölje az összes cellát a "kukás" ikon megnyomásával.

Ebben a szakaszban egy kis diabéteszről szóló adatsoron fog dolgozni, amely a Scikit-learnbe van beépítve tanulási célokra. Tegyük fel, hogy kezelést szeretne tesztelni cukorbetegeken. A gépi tanulási modellek segíthetik annak meghatározását, hogy mely betegek reagálnának jobban a kezelésre a változók kombinációja alapján. Még egy nagyon alap regressziós modell is, ha megjelenítik, információt adhat a változókról, amelyek segíthetnek klinikai kísérletek megtervezésében.

✅ Sokféle regressziós módszer létezik, és az, hogy Ön melyiket választja, attól függ, milyen választ szeretne kapni. Ha egy adott korú személy valószínű magasságát szeretné megjósolni, lineáris regressziót használ, mert **numerikus értéket** keres. Ha meg szeretné állapítani, hogy egy étkezési típus vegánnak minősül-e vagy sem, akkor kategória besorolást keres, így logisztikus regressziót használna. Erről később még többet tanul. Gondolja át, milyen kérdéseket tehet fel az adatoknak, és melyik módszer lenne megfelelőbb.

Kezdjük el ezt a feladatot.

### Könyvtárak importálása

Ehhez a feladathoz néhány könyvtárat fogunk importálni:

- **matplotlib**. Hasznos [grafikus eszköz](https://matplotlib.org/), vonaldiagram készítéséhez használjuk.
- **numpy**. A [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) hasznos könyvtár a numerikus adatok kezeléséhez Pythonban.
- **sklearn**. Ez a [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) könyvtár.

Importáljuk ezeket a könyvtárakat, hogy segítsenek a feladataink elvégzésében.

1. Add hozzá az importokat az alábbi kód begépelésével:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Fentebb importálja a `matplotlib`, `numpy` könyvtárakat, valamint a `datasets`, `linear_model` és `model_selection` modulokat a `sklearn`-ből. A `model_selection` az adatok tanító- és teszthalmazra bontásához használatos.

### A diabétesz adatsor

A beépített [diabétesz adatsor](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 442 mintát tartalmaz diabéteszről, 10 jellemző változóval, néhány ezek közül:

- kor: életkor években
- bmi: testtömeg-index
- vérnyomás: átlagos vérnyomás
- s1 tc: T-sejtek (a fehérvérsejtek egy típusa)

✅ Ez az adatsor tartalmazza a 'nem' változót is, amely fontos a diabétesz kutatásában. Sok orvosi adatsor tartalmaz ilyen bináris osztályozást. Gondolja át, hogyan zárhat ki az ilyen kategorizálás bizonyos csoportokat a kezelésekből.

Most töltsük be az X és y adatokat.

> 🎓 Ne feledje, ez felügyelt tanulás, ezért szükségünk van egy 'y' célváltozóra.

Egy új kódcella megnyitásával töltse be a diabétesz adatsort a `load_diabetes()` hívásával. A bemenet `return_X_y=True` jelzi, hogy az `X` adatmátrix lesz, az `y` pedig a regressziós cél.

1. Adjon hozzá néhány print parancsot az adatmátrix alakjának és első elemének megjelenítéséhez:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Amit visszakap, az egy tuple. Ön azt teszi, hogy a tuple két első értékét az `X` és `y` változóknak rendeli. Tudjon meg többet [a tuple-ökről](https://wikipedia.org/wiki/Tuple).

    Láthatja, hogy ez az adat 442 elemből áll, mindegyik 10 elemből álló tömbökben:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Gondolja át az adat és a regressziós cél kapcsolatát. A lineáris regresszió az X jellemző és az y célváltozó közti kapcsolatot jósolja meg. Meg tudja találni a [célt](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) a diabétesz adatsor dokumentációjában? Mit mutat ez az adatsor, a cél alapján?

2. Ezután válasszon ki egy részét az adatsornak az ábrázoláshoz, válassza ki az adatsor 3. oszlopát. Ezt úgy teheti meg, hogy az `:` operátorral az összes sort kiválasztja, majd az index (`2`) segítségével a 3. oszlopot. Az adatot kétdimenziós tömbbé is alakíthatja az ábrázoláshoz, a `reshape(n_rows, n_columns)` használatával. Ha az egyik paraméter -1, a megfelelő dimenzió automatikusan kiszámításra kerül.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Bármikor nyomtassa ki az adatot az alak ellenőrzéséhez.

3. Most, hogy az adatok ábrázolásra megfelelőek, nézze meg, tud-e a gép segíteni logikus felosztás meghatározásában az adatok között. Ehhez az adatok (X) és a cél (y) is fel kell, hogy legyenek osztva teszt- és tanulókészletekre. A Scikit-learn egy egyszerű módot kínál erre, megadhat egy pontot, ahol a teszt adatokat szétválasztja.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Most készen áll a modell betanítására! Töltse be a lineáris regressziós modellt, és tanítsa a X és y tanulókészletekkel a `model.fit()` használatával:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ A `model.fit()` egy olyan függvény, amelyet számos ML könyvtárban, például TensorFlow-ban is látni fog.

5. Ezután készítsen előrejelzést a teszt adatokon a `predict()` függvénnyel. Ezt használja majd arra, hogy vonalat húzzon az adatcsoportok közé.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Most itt az ideje megjeleníteni az adatokat egy ábrában. A Matplotlib nagyon hasznos eszköz erre a feladatra. Készítsen szórási diagramot az X és y teszt adataihoz, és használja az előrejelzést, hogy a vonalat a modell adatcsoportjai közé húzza.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![egy szórási diagram diabétesz adatokkal](../../../../translated_images/hu/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Gondolja át, mi is történik itt. Egy egyenes vonal húzódik sok kis adatpont között, de pontosan mit csinál? Látja, hogyan használhatná ezt a vonalat arra, hogy előre jelezze, hol illeszkedik egy új, még nem látott adatpont az ábra y tengelyéhez képest? Próbálja meg megfogalmazni ennek a modellnek a gyakorlati hasznát.

Gratulálunk, megépítette első lineáris regressziós modelljét, elkészítette annak előrejelzését, és megjelenítette egy ábrán!

---
## 🚀Kihívás

Ábrázoljon egy másik változót ebből az adatsorból. Tipp: szerkessze ezt a sort: `X = X[:,2]`. Tekintettel az adathalmaz céljára, mit tud felfedezni a diabétesz fejlődésével kapcsolatban?
## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés & Önálló tanulás

Ebben a bemutatóban egyszerű lineáris regresszióval dolgozott, nem pedig egyváltozós vagy többváltozós lineáris regresszióval. Olvasson kicsit a módszerek közötti különbségekről, vagy nézze meg [ezt a videót](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Olvasson többet a regresszió fogalmáról, és gondolkodjon el azon, hogy milyen kérdésekre adhat választ ez a technika. Vegye igénybe ezt a [gyakorlati útmutatót](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) a mélyebb megértés érdekében.

## Feladat

[Egy másik adatállomány](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Fontos információk esetén professzionális emberi fordítást javasolunk. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely ebből a fordításból ered.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->