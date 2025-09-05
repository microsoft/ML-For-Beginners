<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T15:33:09+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "hu"
}
-->
# Bevezetés az idősoros előrejelzésbe

![Idősorok összefoglalása egy vázlatrajzban](../../../../sketchnotes/ml-timeseries.png)

> Vázlatrajz: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Ebben és a következő leckében megismerkedhetsz az idősoros előrejelzéssel, amely a gépi tanulás tudományának egy érdekes és értékes, bár kevésbé ismert területe. Az idősoros előrejelzés olyan, mint egy „varázsgömb”: egy változó, például ár múltbeli teljesítménye alapján megjósolhatod annak jövőbeli potenciális értékét.

[![Bevezetés az idősoros előrejelzésbe](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Bevezetés az idősoros előrejelzésbe")

> 🎥 Kattints a fenti képre az idősoros előrejelzésről szóló videóért

## [Előzetes kvíz](https://ff-quizzes.netlify.app/en/ml/)

Ez egy hasznos és érdekes terület, amely valódi értéket képvisel az üzleti életben, mivel közvetlenül alkalmazható árképzési, készletgazdálkodási és ellátási lánc problémákra. Bár a mélytanulási technikák egyre inkább használatosak a jövőbeli teljesítmény jobb előrejelzésére, az idősoros előrejelzés továbbra is nagyrészt a klasszikus gépi tanulási technikákra támaszkodik.

> A Penn State hasznos idősoros tananyaga [itt található](https://online.stat.psu.edu/stat510/lesson/1)

## Bevezetés

Tegyük fel, hogy egy sor okos parkolóórát üzemeltetsz, amelyek adatokat szolgáltatnak arról, hogy milyen gyakran és mennyi ideig használják őket az idő múlásával.

> Mi lenne, ha meg tudnád jósolni a parkolóóra jövőbeli értékét a kereslet és kínálat törvényei alapján, a múltbeli teljesítményére alapozva?

Pontosan megjósolni, mikor kell cselekedni a cél elérése érdekében, egy olyan kihívás, amelyet az idősoros előrejelzés segítségével lehet megoldani. Bár nem örülnének az emberek, ha forgalmas időszakokban többet kellene fizetniük parkolóhelyért, ez biztos módja lenne a bevétel növelésének, például az utcák tisztítására.

Nézzük meg néhány idősoros algoritmus típusát, és kezdjünk el egy notebookot az adatok tisztítására és előkészítésére. Az elemzendő adatok a GEFCom2014 előrejelzési versenyből származnak. Ez 3 évnyi óránkénti villamosenergia-fogyasztási és hőmérsékleti adatokat tartalmaz 2012 és 2014 között. A villamosenergia-fogyasztás és a hőmérséklet történelmi mintái alapján megjósolhatod a villamosenergia-fogyasztás jövőbeli értékeit.

Ebben a példában megtanulod, hogyan lehet egy időlépést előre jelezni, kizárólag a történelmi fogyasztási adatok alapján. Mielőtt azonban elkezdenéd, hasznos megérteni, mi zajlik a háttérben.

## Néhány definíció

Amikor az „idősor” kifejezéssel találkozol, fontos megérteni annak használatát különböző kontextusokban.

🎓 **Idősor**

A matematikában az „idősor egy időrendben indexelt (vagy listázott vagy grafikonon ábrázolt) adatpontok sorozata. Leggyakrabban az idősor egy sorozat, amelyet egymást követő, egyenlő időközönként vesznek fel.” Az idősor egyik példája a [Dow Jones ipari átlag](https://wikipedia.org/wiki/Time_series) napi záróértéke. Az idősorok grafikonjainak és statisztikai modellezésének használata gyakran előfordul jelanalízisben, időjárás-előrejelzésben, földrengés-előrejelzésben és más olyan területeken, ahol események történnek, és adatpontokat lehet időben ábrázolni.

🎓 **Idősoros elemzés**

Az idősoros elemzés az előbb említett idősoros adatok elemzése. Az idősoros adatok különböző formákat ölthetnek, beleértve az „megszakított idősorokat”, amelyek mintákat észlelnek egy idősor fejlődésében egy megszakító esemény előtt és után. Az idősorhoz szükséges elemzés típusa az adatok természetétől függ. Az idősoros adatok maguk is lehetnek számok vagy karakterek sorozatai.

Az elvégzendő elemzés különféle módszereket használ, beleértve a frekvenciatartományt és az időtartományt, lineáris és nemlineáris módszereket, és még sok mást. [Tudj meg többet](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) az ilyen típusú adatok elemzésének számos módjáról.

🎓 **Idősoros előrejelzés**

Az idősoros előrejelzés egy modell használata a jövőbeli értékek megjóslására, a korábban gyűjtött adatok által mutatott minták alapján. Bár regressziós modellekkel is lehet idősoros adatokat vizsgálni, ahol az időindexek x változóként jelennek meg egy grafikonon, az ilyen adatokat leginkább speciális típusú modellekkel lehet elemezni.

Az idősoros adatok egy rendezett megfigyelések listája, szemben a lineáris regresszióval elemezhető adatokkal. A leggyakoribb modell az ARIMA, amely az „Autoregresszív Integrált Mozgó Átlag” rövidítése.

[ARIMA modellek](https://online.stat.psu.edu/stat510/lesson/1/1.1) „kapcsolatot teremtenek egy sorozat jelenlegi értéke és a múltbeli értékek, valamint a múltbeli előrejelzési hibák között.” Ezek leginkább az időtartományban rendezett adatok elemzésére alkalmasak.

> Az ARIMA modelleknek több típusa van, amelyekről [itt](https://people.duke.edu/~rnau/411arim.htm) tudhatsz meg többet, és amelyeket a következő leckében érinteni fogsz.

A következő leckében egy ARIMA modellt fogsz építeni [Univariáns Idősorok](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm) használatával, amely egyetlen változóra összpontosít, amely idővel változtatja értékét. Az ilyen típusú adatok egyik példája [ez az adatállomány](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm), amely a Mauna Loa Obszervatóriumban mért havi CO2 koncentrációt rögzíti:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Azonosítsd a változót, amely idővel változik ebben az adatállományban.

## Az idősoros adatok jellemzői, amelyeket figyelembe kell venni

Amikor idősoros adatokat vizsgálsz, észreveheted, hogy [bizonyos jellemzőkkel](https://online.stat.psu.edu/stat510/lesson/1/1.1) rendelkeznek, amelyeket figyelembe kell venni és csökkenteni kell, hogy jobban megértsd a mintáikat. Ha az idősoros adatokat potenciálisan egy „jelként” tekinted, amelyet elemezni szeretnél, ezek a jellemzők „zajként” is felfoghatók. Gyakran szükséges csökkenteni ezt a „zajt” bizonyos statisztikai technikák alkalmazásával.

Íme néhány fogalom, amelyet ismerned kell ahhoz, hogy idősoros adatokkal dolgozhass:

🎓 **Trendek**

A trendek idővel mérhető növekedéseket és csökkenéseket jelentenek. [Olvass többet](https://machinelearningmastery.com/time-series-trends-in-python). Az idősorok kontextusában arról van szó, hogyan lehet használni, és ha szükséges, eltávolítani a trendeket az idősorokból.

🎓 **[Szezonális hatások](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

A szezonális hatások olyan időszakos ingadozások, mint például az ünnepi rohamok, amelyek befolyásolhatják az értékesítést. [Nézd meg](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm), hogyan jelennek meg a szezonális hatások különböző típusú grafikonokon.

🎓 **Szélsőséges értékek**

A szélsőséges értékek messze esnek az adatok szokásos szórásától.

🎓 **Hosszú távú ciklus**

A szezonális hatásoktól függetlenül az adatok hosszú távú ciklust is mutathatnak, például egy gazdasági visszaesést, amely egy évnél tovább tart.

🎓 **Állandó szórás**

Idővel néhány adat állandó ingadozásokat mutat, például napi és éjszakai energiafogyasztás.

🎓 **Hirtelen változások**

Az adatok hirtelen változást mutathatnak, amely további elemzést igényelhet. Például a COVID miatt hirtelen bezáró üzletek változásokat okoztak az adatokban.

✅ Itt van egy [példa idősoros grafikon](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python), amely néhány év alatt napi játékon belüli pénzköltést mutat. Felismered az adatokban a fent felsorolt jellemzők bármelyikét?

![Játékon belüli pénzköltés](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Gyakorlat - kezdjük az energiafogyasztási adatokkal

Kezdjünk el létrehozni egy idősoros modellt, amely a múltbeli fogyasztás alapján megjósolja a jövőbeli energiafogyasztást.

> Az adatok ebben a példában a GEFCom2014 előrejelzési versenyből származnak. Ez 3 évnyi óránkénti villamosenergia-fogyasztási és hőmérsékleti adatokat tartalmaz 2012 és 2014 között.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli és Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. Nyisd meg a `working` mappában található _notebook.ipynb_ fájlt. Kezdd azzal, hogy hozzáadod azokat a könyvtárakat, amelyek segítenek az adatok betöltésében és vizualizálásában:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Figyelj arra, hogy a `common` mappában található fájlokat használod, amelyek beállítják a környezetet és kezelik az adatok letöltését.

2. Ezután vizsgáld meg az adatokat egy dataframe-ként, a `load_data()` és `head()` hívásával:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Láthatod, hogy két oszlop van, amelyek az időpontot és a fogyasztást képviselik:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Most ábrázold az adatokat a `plot()` hívásával:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![energia grafikon](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Ezután ábrázold 2014 júliusának első hetét, az `energy` bemenetként való megadásával `[kezdő dátum]:[záró dátum]` mintában:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![július](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Gyönyörű grafikon! Nézd meg ezeket a grafikonokat, és próbáld meg meghatározni a fent felsorolt jellemzők bármelyikét. Mit tudunk megállapítani az adatok vizualizálásával?

A következő leckében egy ARIMA modellt fogsz létrehozni, hogy előrejelzéseket készíts.

---

## 🚀Kihívás

Készíts listát az összes olyan iparágról és kutatási területről, amely szerinted hasznot húzhat az idősoros előrejelzésből. Eszedbe jut olyan alkalmazás ezekre a technikákra a művészetekben? Az ökonometriában? Az ökológiában? A kiskereskedelemben? Az iparban? A pénzügyekben? Hol máshol?

## [Utólagos kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Bár itt nem foglalkozunk velük, a neurális hálózatokat néha használják az idősoros előrejelzés klasszikus módszereinek kiegészítésére. Olvass róluk többet [ebben a cikkben](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Feladat

[Vizualizálj további idősorokat](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az [Co-op Translator](https://github.com/Azure/co-op-translator) AI fordítási szolgáltatás segítségével került lefordításra. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.