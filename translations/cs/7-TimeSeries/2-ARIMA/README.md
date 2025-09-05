<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T23:45:36+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "cs"
}
-->
# Prognóza časových řad pomocí ARIMA

V předchozí lekci jste se seznámili se základy prognózování časových řad a načetli dataset zobrazující výkyvy elektrického zatížení v průběhu času.

[![Úvod do ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Úvod do ARIMA")

> 🎥 Klikněte na obrázek výše pro video: Stručný úvod do modelů ARIMA. Příklad je proveden v R, ale koncepty jsou univerzální.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

V této lekci objevíte konkrétní způsob, jak vytvářet modely pomocí [ARIMA: *A*uto*R*egresivní *I*ntegrální *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Modely ARIMA jsou obzvláště vhodné pro data, která vykazují [nestacionaritu](https://wikipedia.org/wiki/Stationary_process).

## Obecné koncepty

Abyste mohli pracovat s ARIMA, je třeba znát některé základní pojmy:

- 🎓 **Stacionarita**. Z pohledu statistiky stacionarita označuje data, jejichž rozdělení se nemění při posunu v čase. Nestacionární data naopak vykazují výkyvy způsobené trendy, které je nutné transformovat, aby mohla být analyzována. Sezónnost například může způsobit výkyvy v datech, které lze odstranit procesem „sezónního diferenciace“.

- 🎓 **[Diferenciace](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Diferenciace dat, opět z pohledu statistiky, označuje proces transformace nestacionárních dat na stacionární odstraněním jejich nekonstantního trendu. „Diferenciace odstraňuje změny úrovně časové řady, eliminuje trend a sezónnost a následně stabilizuje průměr časové řady.“ [Studie od Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA v kontextu časových řad

Pojďme rozebrat jednotlivé části ARIMA, abychom lépe pochopili, jak nám pomáhá modelovat časové řady a provádět prognózy.

- **AR - AutoRegresivní**. Autoregresivní modely, jak název napovídá, se „dívají zpět“ v čase, aby analyzovaly předchozí hodnoty ve vašich datech a vytvořily na jejich základě předpoklady. Tyto předchozí hodnoty se nazývají „zpoždění“ (lags). Příkladem mohou být data zobrazující měsíční prodeje tužek. Celkový prodej za každý měsíc by byl považován za „vyvíjející se proměnnou“ v datasetu. Tento model je vytvořen tak, že „vyvíjející se proměnná zájmu je regrese na své vlastní zpožděné (tj. předchozí) hodnoty.“ [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrované**. Na rozdíl od podobných modelů 'ARMA' se 'I' v ARIMA vztahuje na jeho *[integrovaný](https://wikipedia.org/wiki/Order_of_integration)* aspekt. Data jsou „integrovaná“, když jsou aplikovány kroky diferenciace, aby se odstranila nestacionarita.

- **MA - Klouzavý průměr**. [Klouzavý průměr](https://wikipedia.org/wiki/Moving-average_model) v tomto modelu označuje výstupní proměnnou, která je určena pozorováním aktuálních a minulých hodnot zpoždění.

Shrnutí: ARIMA se používá k tomu, aby model co nejlépe odpovídal specifické formě dat časových řad.

## Cvičení - vytvoření modelu ARIMA

Otevřete složku [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) v této lekci a najděte soubor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Spusťte notebook a načtěte knihovnu Pythonu `statsmodels`; budete ji potřebovat pro modely ARIMA.

1. Načtěte potřebné knihovny.

1. Nyní načtěte několik dalších knihoven užitečných pro vykreslování dat:

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

1. Načtěte data ze souboru `/data/energy.csv` do Pandas dataframe a podívejte se na ně:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Vykreslete všechna dostupná data o energii od ledna 2012 do prosince 2014. Nemělo by vás nic překvapit, protože tato data jsme viděli v minulé lekci:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Nyní vytvoříme model!

### Vytvoření trénovacích a testovacích datasetů

Nyní máte data načtená, takže je můžete rozdělit na trénovací a testovací sady. Model budete trénovat na trénovací sadě. Jako obvykle, po dokončení trénování modelu vyhodnotíte jeho přesnost pomocí testovací sady. Musíte zajistit, aby testovací sada pokrývala pozdější období než trénovací sada, aby model nezískal informace z budoucích časových období.

1. Vyčleňte dvouměsíční období od 1. září do 31. října 2014 pro trénovací sadu. Testovací sada bude zahrnovat dvouměsíční období od 1. listopadu do 31. prosince 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Protože tato data odrážejí denní spotřebu energie, existuje silný sezónní vzorec, ale spotřeba je nejvíce podobná spotřebě v nedávných dnech.

1. Vizualizujte rozdíly:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![trénovací a testovací data](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Proto by mělo být dostačující použít relativně malé časové okno pro trénování dat.

    > Poznámka: Protože funkce, kterou používáme k přizpůsobení modelu ARIMA, používá validaci na vzorku během přizpůsobení, vynecháme validační data.

### Příprava dat pro trénování

Nyní je třeba připravit data pro trénování filtrováním a škálováním dat. Filtrovat dataset tak, aby zahrnoval pouze potřebná časová období a sloupce, a škálovat data, aby byla zobrazena v intervalu 0,1.

1. Filtrovat původní dataset tak, aby zahrnoval pouze výše uvedená časová období na sadu a pouze potřebný sloupec 'load' plus datum:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Můžete vidět tvar dat:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Škálovat data do rozsahu (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Vizualizujte původní vs. škálovaná data:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![původní](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Původní data

    ![škálovaná](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Škálovaná data

1. Nyní, když jste kalibrovali škálovaná data, můžete škálovat testovací data:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementace ARIMA

Je čas implementovat ARIMA! Nyní použijete knihovnu `statsmodels`, kterou jste nainstalovali dříve.

Nyní je třeba postupovat podle několika kroků:

   1. Definujte model voláním `SARIMAX()` a předáním parametrů modelu: parametry p, d a q, a parametry P, D a Q.
   2. Připravte model pro trénovací data voláním funkce `fit()`.
   3. Proveďte prognózy voláním funkce `forecast()` a specifikujte počet kroků (tzv. „horizont“) pro prognózu.

> 🎓 Co znamenají všechny tyto parametry? V modelu ARIMA existují 3 parametry, které pomáhají modelovat hlavní aspekty časové řady: sezónnost, trend a šum. Tyto parametry jsou:

`p`: parametr spojený s autoregresivním aspektem modelu, který zahrnuje *minulé* hodnoty.
`d`: parametr spojený s integrovanou částí modelu, který ovlivňuje množství *diferenciace* (🎓 pamatujete si diferenciaci 👆?) aplikované na časovou řadu.
`q`: parametr spojený s částí modelu klouzavého průměru.

> Poznámka: Pokud vaše data mají sezónní aspekt - což tato data mají -, použijeme sezónní model ARIMA (SARIMA). V takovém případě je třeba použít další sadu parametrů: `P`, `D` a `Q`, které popisují stejné asociace jako `p`, `d` a `q`, ale odpovídají sezónním komponentám modelu.

1. Začněte nastavením preferované hodnoty horizontu. Zkusme 3 hodiny:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Výběr nejlepších hodnot pro parametry modelu ARIMA může být náročný, protože je do jisté míry subjektivní a časově náročný. Můžete zvážit použití funkce `auto_arima()` z knihovny [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Prozatím zkuste ručně vybrat některé hodnoty pro nalezení dobrého modelu.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Vytiskne se tabulka výsledků.

Vytvořili jste svůj první model! Nyní musíme najít způsob, jak jej vyhodnotit.

### Vyhodnocení modelu

Pro vyhodnocení modelu můžete provést tzv. validaci „walk forward“. V praxi se modely časových řad znovu trénují pokaždé, když jsou k dispozici nová data. To umožňuje modelu provést nejlepší prognózu v každém časovém kroku.

Začněte na začátku časové řady pomocí této techniky, trénujte model na trénovací datové sadě. Poté proveďte prognózu na další časový krok. Prognóza je vyhodnocena oproti známé hodnotě. Trénovací sada je poté rozšířena o známou hodnotu a proces se opakuje.

> Poznámka: Měli byste udržovat okno trénovací sady pevné pro efektivnější trénování, takže pokaždé, když přidáte novou pozorování do trénovací sady, odstraníte pozorování ze začátku sady.

Tento proces poskytuje robustnější odhad toho, jak bude model fungovat v praxi. Přichází však s výpočetními náklady na vytvoření tolika modelů. To je přijatelné, pokud jsou data malá nebo pokud je model jednoduchý, ale může to být problém ve větším měřítku.

Validace „walk forward“ je zlatým standardem pro vyhodnocení modelů časových řad a je doporučena pro vaše vlastní projekty.

1. Nejprve vytvořte testovací datový bod pro každý krok HORIZONTU.

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

    Data jsou horizontálně posunuta podle bodu horizontu.

1. Proveďte prognózy na testovacích datech pomocí tohoto přístupu posuvného okna v cyklu o velikosti délky testovacích dat:

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

    Můžete sledovat probíhající trénování:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Porovnejte prognózy se skutečným zatížením:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Výstup
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Sledujte prognózu hodinových dat ve srovnání se skutečným zatížením. Jak přesná je?

### Kontrola přesnosti modelu

Zkontrolujte přesnost svého modelu testováním jeho střední absolutní procentuální chyby (MAPE) u všech prognóz.
> **🧮 Ukázka výpočtu**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) se používá k vyjádření přesnosti predikce jako poměru definovaného výše uvedeným vzorcem. Rozdíl mezi skutečnou hodnotou a predikovanou hodnotou je dělen skutečnou hodnotou.  
"Absolutní hodnota v tomto výpočtu se sečte pro každý předpovězený bod v čase a vydělí se počtem přizpůsobených bodů n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Vyjádřete rovnici v kódu:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Vypočítejte MAPE pro jeden krok:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE předpovědi pro jeden krok:  0.5570581332313952 %

1. Vytiskněte MAPE pro více kroků:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Nízké číslo je nejlepší: vezměte v úvahu, že předpověď s MAPE 10 je o 10 % mimo.

1. Ale jak vždy, je snazší tento typ měření přesnosti vidět vizuálně, takže si to vykreslíme:

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

    ![model časové řady](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Velmi pěkný graf, ukazující model s dobrou přesností. Skvělá práce!

---

## 🚀Výzva

Prozkoumejte způsoby testování přesnosti modelu časové řady. V této lekci se dotýkáme MAPE, ale existují i jiné metody, které byste mohli použít? Prozkoumejte je a okomentujte. Užitečný dokument najdete [zde](https://otexts.com/fpp2/accuracy.html)

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Tato lekce se dotýká pouze základů předpovědi časové řady pomocí ARIMA. Věnujte čas prohloubení svých znalostí tím, že prozkoumáte [toto úložiště](https://microsoft.github.io/forecasting/) a jeho různé typy modelů, abyste se naučili další způsoby, jak vytvářet modely časové řady.

## Zadání

[Nový model ARIMA](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.