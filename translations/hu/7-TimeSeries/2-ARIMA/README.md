<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T15:29:04+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "hu"
}
-->
# Idősorok előrejelzése ARIMA-val

Az előző leckében megismerkedtél az idősorok előrejelzésének alapjaival, és betöltöttél egy adatállományt, amely az elektromos terhelés ingadozásait mutatja egy időszak alatt.

[![Bevezetés az ARIMA-ba](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Bevezetés az ARIMA-ba")

> 🎥 Kattints a fenti képre egy videóért: Rövid bevezetés az ARIMA modellekbe. Az példa R-ben készült, de a koncepciók univerzálisak.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Bevezetés

Ebben a leckében megismerkedsz az [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) modellek építésének egy konkrét módjával. Az ARIMA modellek különösen alkalmasak olyan adatok illesztésére, amelyek [nem állomásosak](https://wikipedia.org/wiki/Stationary_process).

## Általános fogalmak

Ahhoz, hogy ARIMA-val dolgozhass, néhány alapfogalmat ismerned kell:

- 🎓 **Állomásosság**. Statisztikai értelemben az állomásosság olyan adatokra utal, amelyek eloszlása nem változik időbeli eltolás esetén. A nem állomásos adatok ingadozásokat mutatnak trendek miatt, amelyeket át kell alakítani az elemzéshez. Például a szezonális hatások ingadozásokat okozhatnak az adatokban, amelyeket 'szezonális differenciálás' révén lehet eltávolítani.

- 🎓 **[Differenciálás](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. A differenciálás statisztikai értelemben az a folyamat, amely során a nem állomásos adatokat állomásossá alakítjuk az állandó trend eltávolításával. "A differenciálás eltávolítja az idősor szintjének változásait, megszünteti a trendet és a szezonális hatásokat, ezáltal stabilizálja az idősor átlagát." [Shixiong et al tanulmánya](https://arxiv.org/abs/1904.07632)

## ARIMA az idősorok kontextusában

Nézzük meg az ARIMA részeit, hogy jobban megértsük, hogyan segít az idősorok modellezésében és előrejelzések készítésében.

- **AR - AutoRegresszív**. Az autoregresszív modellek, ahogy a nevük is sugallja, visszatekintenek az időben, hogy elemezzék az adatok korábbi értékeit, és feltételezéseket tegyenek róluk. Ezeket a korábbi értékeket 'késéseknek' nevezzük. Példa lehet a havi ceruzaeladások adatai. Minden hónap eladási összesítése az adathalmazban egy 'változó' lenne. Ez a modell úgy épül fel, hogy "az érdeklődésre számot tartó változót saját késleltetett (azaz korábbi) értékeire regresszálják." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrált**. Az ARIMA modellekben az 'I' az *[integrált](https://wikipedia.org/wiki/Order_of_integration)* aspektusra utal. Az adatok 'integrálása' a differenciálási lépések alkalmazásával történik, hogy megszüntessük a nem állomásosságot.

- **MA - Mozgó Átlag**. A [mozgó átlag](https://wikipedia.org/wiki/Moving-average_model) aspektus az output változóra utal, amelyet a késések aktuális és korábbi értékeinek megfigyelésével határozunk meg.

Összefoglalva: Az ARIMA-t arra használjuk, hogy a modell minél jobban illeszkedjen az idősorok speciális formájához.

## Gyakorlat - ARIMA modell építése

Nyisd meg a [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) mappát ebben a leckében, és keresd meg a [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) fájlt.

1. Futtasd a notebookot, hogy betöltsd a `statsmodels` Python könyvtárat; erre szükséged lesz az ARIMA modellekhez.

1. Töltsd be a szükséges könyvtárakat.

1. Most tölts be néhány további könyvtárat, amelyek hasznosak az adatok ábrázolásához:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. Töltsd be az adatokat a `/data/energy.csv` fájlból egy Pandas dataframe-be, és nézd meg:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Ábrázold az összes elérhető energiaadatot 2012 januárjától 2014 decemberéig. Nem lesz meglepetés, hiszen ezt az adatot láttuk az előző leckében:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Most építsünk egy modellt!

### Képzési és tesztelési adathalmazok létrehozása

Most, hogy betöltötted az adatokat, szétválaszthatod őket képzési és tesztelési halmazokra. A modell képzését a képzési halmazon végzed. Szokás szerint, miután a modell befejezte a képzést, a tesztelési halmazzal értékeled annak pontosságát. Biztosítanod kell, hogy a tesztelési halmaz egy későbbi időszakot fed le, mint a képzési halmaz, hogy a modell ne szerezzen információt a jövőbeli időszakokról.

1. Jelölj ki egy két hónapos időszakot 2014. szeptember 1-től október 31-ig a képzési halmaz számára. A tesztelési halmaz a 2014. november 1-től december 31-ig tartó két hónapos időszakot foglalja magában:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Mivel ezek az adatok az energia napi fogyasztását tükrözik, erős szezonális mintázat figyelhető meg, de a fogyasztás leginkább a legutóbbi napok fogyasztásához hasonló.

1. Vizualizáld a különbségeket:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![képzési és tesztelési adatok](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Ezért viszonylag kis időablak használata elegendő lehet az adatok képzéséhez.

    > Megjegyzés: Mivel az ARIMA modell illesztéséhez használt függvény a fitting során mintán belüli validációt alkalmaz, kihagyjuk a validációs adatokat.

### Az adatok előkészítése a képzéshez

Most elő kell készítened az adatokat a képzéshez, szűréssel és skálázással. Szűrd az adathalmazt, hogy csak a szükséges időszakokat és oszlopokat tartalmazza, és skálázd az adatokat, hogy az értékek a 0 és 1 közötti intervallumba essenek.

1. Szűrd az eredeti adathalmazt, hogy csak az említett időszakokat és a szükséges 'load' oszlopot, valamint a dátumot tartalmazza:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Megnézheted az adatok alakját:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skálázd az adatokat, hogy a (0, 1) tartományba essenek.

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Vizualizáld az eredeti és a skálázott adatokat:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![eredeti](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Az eredeti adatok

    ![skálázott](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > A skálázott adatok

1. Most, hogy kalibráltad a skálázott adatokat, skálázd a tesztadatokat is:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA megvalósítása

Elérkezett az idő az ARIMA megvalósítására! Most használni fogod a korábban telepített `statsmodels` könyvtárat.

Kövesd az alábbi lépéseket:

   1. Határozd meg a modellt a `SARIMAX()` meghívásával, és add meg a modell paramétereit: p, d, és q paraméterek, valamint P, D, és Q paraméterek.
   2. Készítsd elő a modellt a képzési adatokhoz a fit() függvény meghívásával.
   3. Készíts előrejelzéseket a `forecast()` függvény meghívásával, és add meg az előrejelzés lépéseinek számát (a `horizontot`).

> 🎓 Mire valók ezek a paraméterek? Az ARIMA modellben három paramétert használunk, amelyek segítenek az idősorok főbb aspektusainak modellezésében: szezonális hatások, trendek és zaj. Ezek a paraméterek:

`p`: az autoregresszív aspektushoz kapcsolódó paraméter, amely a *múltbeli* értékeket veszi figyelembe.
`d`: az integrált részhez kapcsolódó paraméter, amely meghatározza, hogy mennyi *differenciálást* (🎓 emlékszel a differenciálásra 👆?) kell alkalmazni az idősorra.
`q`: a mozgó átlag részhez kapcsolódó paraméter.

> Megjegyzés: Ha az adatok szezonális aspektussal rendelkeznek - mint ezek -, akkor szezonális ARIMA modellt (SARIMA) használunk. Ebben az esetben egy másik paraméterkészletet kell használni: `P`, `D`, és `Q`, amelyek ugyanazokat az összefüggéseket írják le, mint `p`, `d`, és `q`, de a modell szezonális komponenseire vonatkoznak.

1. Kezdd azzal, hogy beállítod a preferált horizontértéket. Próbáljunk ki 3 órát:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Az ARIMA modell paramétereinek legjobb értékeinek kiválasztása kihívást jelenthet, mivel ez némileg szubjektív és időigényes. Érdemes lehet használni az `auto_arima()` függvényt a [`pyramid` könyvtárból](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Egyelőre próbálj ki néhány manuális beállítást, hogy jó modellt találj.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Egy eredménytábla jelenik meg.

Elkészítetted az első modelledet! Most meg kell találnunk egy módot annak értékelésére.

### A modell értékelése

A modell értékeléséhez alkalmazhatod az úgynevezett `lépésről lépésre` validációt. A gyakorlatban az idősor modelleket minden alkalommal újra kell tanítani, amikor új adatok válnak elérhetővé. Ez lehetővé teszi, hogy a modell minden időlépésnél a legjobb előrejelzést készítse.

Ezzel a technikával az idősor elején kezdve tanítsd a modellt a képzési adathalmazon. Ezután készíts előrejelzést a következő időlépésre. Az előrejelzést összehasonlítjuk az ismert értékkel. A képzési halmazt ezután kibővítjük az ismert értékkel, és a folyamatot megismételjük.

> Megjegyzés: A képzési halmaz ablakát érdemes fixen tartani a hatékonyabb képzés érdekében, így minden alkalommal, amikor új megfigyelést adsz hozzá a képzési halmazhoz, eltávolítod az ablak elejéről a megfigyelést.

Ez a folyamat robusztusabb becslést nyújt arról, hogy a modell hogyan fog teljesíteni a gyakorlatban. Ugyanakkor számítási költséggel jár, mivel sok modellt kell létrehozni. Ez elfogadható, ha az adatok kicsik vagy a modell egyszerű, de problémát jelenthet nagyobb léptékben.

A lépésről lépésre validáció az idősor modellek értékelésének arany standardja, és ajánlott saját projektjeidhez.

1. Először hozz létre egy tesztadatpontot minden HORIZONT lépéshez.

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    Az adatok vízszintesen eltolódnak a horizont pontja szerint.

1. Készíts előrejelzéseket a tesztadatokon ezzel a csúszó ablak megközelítéssel, a tesztadatok hosszának megfelelő méretű ciklusban:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    Nézheted a képzés folyamatát:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Hasonlítsd össze az előrejelzéseket a tényleges terheléssel:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Kimenet
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Figyeld meg az óránkénti adatok előrejelzését, összehasonlítva a tényleges terheléssel. Mennyire pontos ez?

### A modell pontosságának ellenőrzése

Ellenőrizd a modell pontosságát az összes előrejelzés átlagos abszolút százalékos hibájának (MAPE) tesztelésével.
> **🧮 Mutasd a matematikát**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> A [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) a predikció pontosságát mutatja egy arányként, amelyet a fenti képlet határoz meg. A tényleges és az előrejelzett érték közötti különbséget elosztjuk a tényleges értékkel.  
> "Ennek a számításnak az abszolút értékét minden előrejelzett időpontra összegezzük, majd elosztjuk az illesztett pontok számával, n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Fejezd ki az egyenletet kódban:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Számítsd ki az egy lépésre vonatkozó MAPE értéket:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    Egy lépés előrejelzés MAPE:  0.5570581332313952 %

1. Nyomtasd ki a több lépésre vonatkozó előrejelzés MAPE értékét:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Egy alacsony szám a legjobb: gondolj arra, hogy egy előrejelzés, amelynek MAPE értéke 10, 10%-kal tér el.

1. De mint mindig, az ilyen pontosságmérést vizuálisan könnyebb megérteni, ezért ábrázoljuk:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![idősor modell](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Egy nagyon szép ábra, amely egy jó pontosságú modellt mutat. Szép munka!

---

## 🚀Kihívás

Merülj el az idősor modellek pontosságának tesztelési módjaiban. Ebben a leckében érintjük a MAPE-t, de vannak más módszerek, amelyeket használhatnál? Kutass utána, és jegyzeteld le őket. Egy hasznos dokumentumot itt találhatsz: [itt](https://otexts.com/fpp2/accuracy.html)

## [Utó-leckekvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Ez a lecke csak az ARIMA-val történő idősor előrejelzés alapjait érinti. Szánj időt arra, hogy elmélyítsd tudásodat, és nézd meg [ezt a repót](https://microsoft.github.io/forecasting/) és annak különböző modell típusait, hogy megtanuld, hogyan lehet más módokon idősor modelleket építeni.

## Feladat

[Egy új ARIMA modell](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.