<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T15:30:01+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "sk"
}
-->
# Predpovedanie časových radov pomocou ARIMA

V predchádzajúcej lekcii ste sa dozvedeli niečo o predpovedaní časových radov a načítali ste dataset zobrazujúci výkyvy elektrického zaťaženia v priebehu času.

[![Úvod do ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Úvod do ARIMA")

> 🎥 Kliknite na obrázok vyššie pre video: Stručný úvod do modelov ARIMA. Príklad je spracovaný v R, ale koncepty sú univerzálne.

## [Kvíz pred lekciou](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

V tejto lekcii objavíte konkrétny spôsob vytvárania modelov pomocou [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Modely ARIMA sú obzvlášť vhodné na prispôsobenie údajov, ktoré vykazujú [ne-stacionárnosť](https://wikipedia.org/wiki/Stationary_process).

## Všeobecné koncepty

Aby ste mohli pracovať s ARIMA, musíte poznať niektoré základné pojmy:

- 🎓 **Stacionárnosť**. Z pohľadu štatistiky stacionárnosť označuje údaje, ktorých distribúcia sa nemení pri posune v čase. Ne-stacionárne údaje vykazujú výkyvy spôsobené trendmi, ktoré je potrebné transformovať na analýzu. Sezónnosť, napríklad, môže spôsobovať výkyvy v údajoch a môže byť eliminovaná procesom 'sezónneho diferenciovania'.

- 🎓 **[Diferencovanie](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Diferencovanie údajov, opäť z pohľadu štatistiky, označuje proces transformácie ne-stacionárnych údajov na stacionárne odstránením ich ne-konštantného trendu. "Diferencovanie odstraňuje zmeny v úrovni časového radu, eliminuje trend a sezónnosť a následne stabilizuje priemer časového radu." [Štúdia od Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA v kontexte časových radov

Rozložme časti ARIMA, aby sme lepšie pochopili, ako nám pomáha modelovať časové rady a robiť predpovede.

- **AR - AutoRegresívne**. Autoregresívne modely, ako naznačuje názov, sa pozerajú 'späť' v čase, aby analyzovali predchádzajúce hodnoty vo vašich údajoch a robili o nich predpoklady. Tieto predchádzajúce hodnoty sa nazývajú 'oneskorenia'. Príkladom by mohli byť údaje zobrazujúce mesačný predaj ceruziek. Každý mesačný súčet predaja by sa považoval za 'vyvíjajúcu sa premennú' v datasete. Tento model je postavený tak, že "vyvíjajúca sa premenná záujmu je regresovaná na svoje oneskorené (t.j. predchádzajúce) hodnoty." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrované**. Na rozdiel od podobných modelov 'ARMA', 'I' v ARIMA označuje jeho *[integrovaný](https://wikipedia.org/wiki/Order_of_integration)* aspekt. Údaje sú 'integrované', keď sa aplikujú kroky diferenciovania na elimináciu ne-stacionárnosti.

- **MA - Pohyblivý priemer**. [Pohyblivý priemer](https://wikipedia.org/wiki/Moving-average_model) v tomto modeli označuje výstupnú premennú, ktorá je určená pozorovaním aktuálnych a minulých hodnôt oneskorení.

Zhrnutie: ARIMA sa používa na prispôsobenie modelu špeciálnej forme údajov časových radov čo najpresnejšie.

## Cvičenie - vytvorte model ARIMA

Otvorte [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) priečinok v tejto lekcii a nájdite súbor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Spustite notebook na načítanie knižnice `statsmodels` pre Python; túto budete potrebovať pre modely ARIMA.

1. Načítajte potrebné knižnice.

1. Teraz načítajte niekoľko ďalších knižníc užitočných na vykresľovanie údajov:

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

1. Načítajte údaje zo súboru `/data/energy.csv` do Pandas dataframe a pozrite sa na ne:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Vykreslite všetky dostupné údaje o energii od januára 2012 do decembra 2014. Nemalo by vás nič prekvapiť, keďže sme tieto údaje videli v predchádzajúcej lekcii:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Teraz vytvorme model!

### Vytvorte tréningové a testovacie datasety

Keď sú vaše údaje načítané, môžete ich rozdeliť na tréningovú a testovaciu množinu. Model budete trénovať na tréningovej množine. Ako obvykle, po dokončení tréningu modelu vyhodnotíte jeho presnosť pomocou testovacej množiny. Musíte zabezpečiť, aby testovacia množina pokrývala neskoršie obdobie v čase ako tréningová množina, aby ste zabezpečili, že model nezíska informácie z budúcich časových období.

1. Priraďte dvojmesačné obdobie od 1. septembra do 31. októbra 2014 tréningovej množine. Testovacia množina bude zahŕňať dvojmesačné obdobie od 1. novembra do 31. decembra 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Keďže tieto údaje odrážajú dennú spotrebu energie, existuje silný sezónny vzor, ale spotreba je najviac podobná spotrebe v nedávnych dňoch.

1. Vizualizujte rozdiely:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![tréningové a testovacie údaje](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Preto by malo byť dostatočné použiť relatívne malé časové okno na tréning údajov.

    > Poznámka: Keďže funkcia, ktorú používame na prispôsobenie modelu ARIMA, používa validáciu v rámci vzorky počas prispôsobovania, vynecháme validačné údaje.

### Pripravte údaje na tréning

Teraz musíte pripraviť údaje na tréning filtrovaním a škálovaním údajov. Filtrovanie datasetu zahŕňa zahrnutie iba potrebných časových období a stĺpcov, a škálovanie zabezpečuje, že údaje sú projektované v intervale 0,1.

1. Filtrovanie pôvodného datasetu na zahrnutie iba vyššie uvedených časových období na množinu a zahrnutie iba potrebného stĺpca 'load' plus dátum:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Môžete vidieť tvar údajov:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Škálovanie údajov do rozsahu (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Vizualizujte pôvodné vs. škálované údaje:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![pôvodné](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Pôvodné údaje

    ![škálované](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Škálované údaje

1. Teraz, keď ste kalibrovali škálované údaje, môžete škálovať testovacie údaje:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementujte ARIMA

Je čas implementovať ARIMA! Teraz použijete knižnicu `statsmodels`, ktorú ste nainštalovali skôr.

Teraz musíte postupovať podľa niekoľkých krokov:

   1. Definujte model volaním `SARIMAX()` a zadaním parametrov modelu: parametre p, d a q, a parametre P, D a Q.
   2. Pripravte model na tréningové údaje volaním funkcie fit().
   3. Vytvorte predpovede volaním funkcie `forecast()` a špecifikovaním počtu krokov (tzv. `horizont`) na predpovedanie.

> 🎓 Na čo slúžia všetky tieto parametre? V modeli ARIMA existujú 3 parametre, ktoré sa používajú na modelovanie hlavných aspektov časového radu: sezónnosť, trend a šum. Tieto parametre sú:

`p`: parameter spojený s autoregresívnym aspektom modelu, ktorý zahŕňa *minulé* hodnoty.
`d`: parameter spojený s integrovanou časťou modelu, ktorý ovplyvňuje množstvo *diferencovania* (🎓 pamätáte si diferenciovanie 👆?) aplikovaného na časový rad.
`q`: parameter spojený s aspektom pohyblivého priemeru modelu.

> Poznámka: Ak vaše údaje majú sezónny aspekt - čo tieto majú -, používame sezónny model ARIMA (SARIMA). V takom prípade musíte použiť ďalšiu sadu parametrov: `P`, `D` a `Q`, ktoré opisujú rovnaké asociácie ako `p`, `d` a `q`, ale zodpovedajú sezónnym komponentom modelu.

1. Začnite nastavením preferovanej hodnoty horizontu. Skúsme 3 hodiny:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Výber najlepších hodnôt pre parametre modelu ARIMA môže byť náročný, pretože je čiastočne subjektívny a časovo náročný. Môžete zvážiť použitie funkcie `auto_arima()` z knižnice [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Zatiaľ skúste manuálne výbery na nájdenie dobrého modelu.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Tlačí sa tabuľka výsledkov.

Vytvorili ste svoj prvý model! Teraz musíme nájsť spôsob, ako ho vyhodnotiť.

### Vyhodnoťte svoj model

Na vyhodnotenie modelu môžete použiť tzv. `walk forward` validáciu. V praxi sa modely časových radov pretrénujú vždy, keď sú k dispozícii nové údaje. To umožňuje modelu robiť najlepšiu predpoveď v každom časovom kroku.

Začnite na začiatku časového radu pomocou tejto techniky, trénujte model na tréningovej množine údajov. Potom urobte predpoveď na ďalší časový krok. Predpoveď sa vyhodnotí oproti známej hodnote. Tréningová množina sa potom rozšíri o známu hodnotu a proces sa opakuje.

> Poznámka: Mali by ste udržiavať pevné okno tréningovej množiny pre efektívnejší tréning, takže vždy, keď pridáte novú pozorovanie do tréningovej množiny, odstránite pozorovanie zo začiatku množiny.

Tento proces poskytuje robustnejšie odhady toho, ako bude model fungovať v praxi. Avšak, prichádza s výpočtovými nákladmi na vytvorenie toľkých modelov. To je prijateľné, ak sú údaje malé alebo ak je model jednoduchý, ale môže byť problémom pri veľkom rozsahu.

Walk-forward validácia je zlatým štandardom hodnotenia modelov časových radov a odporúča sa pre vaše vlastné projekty.

1. Najskôr vytvorte testovací dátový bod pre každý krok HORIZONTU.

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

    Údaje sa horizontálne posúvajú podľa bodu horizontu.

1. Urobte predpovede na testovacích údajoch pomocou tohto prístupu posuvného okna v slučke veľkosti dĺžky testovacích údajov:

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

    Môžete sledovať, ako prebieha tréning:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Porovnajte predpovede so skutočným zaťažením:

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

    Pozorujte predpoveď hodinových údajov v porovnaní so skutočným zaťažením. Aká presná je táto predpoveď?

### Skontrolujte presnosť modelu

Skontrolujte presnosť svojho modelu testovaním jeho strednej absolútnej percentuálnej chyby (MAPE) na všetkých predpovediach.
> **🧮 Ukáž mi matematiku**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) sa používa na zobrazenie presnosti predpovede ako pomeru definovaného vyššie uvedeným vzorcom. Rozdiel medzi skutočnou hodnotou a predpovedanou hodnotou je vydelený skutočnou hodnotou. 
>
> "Absolútna hodnota v tomto výpočte sa sčíta pre každý predpovedaný bod v čase a vydelí sa počtom prispôsobených bodov n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Vyjadrite rovnicu v kóde:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Vypočítajte MAPE pre jeden krok:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE predpovede pre jeden krok:  0.5570581332313952 %

1. Vytlačte MAPE pre viac krokov:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Nízke číslo je najlepšie: zvážte, že predpoveď s MAPE 10 je odchýlená o 10 %.

1. Ale ako vždy, je jednoduchšie vidieť tento typ merania presnosti vizuálne, takže si to zobrazme na grafe:

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

    ![model časovej rady](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Veľmi pekný graf, ktorý ukazuje model s dobrou presnosťou. Skvelá práca!

---

## 🚀Výzva

Preskúmajte spôsoby testovania presnosti modelu časovej rady. V tejto lekcii sa dotýkame MAPE, ale existujú aj iné metódy, ktoré by ste mohli použiť? Preskúmajte ich a pridajte poznámky. Užitočný dokument nájdete [tu](https://otexts.com/fpp2/accuracy.html)

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad & Samoštúdium

Táto lekcia sa dotýka iba základov predpovedania časových radov pomocou ARIMA. Venujte čas prehĺbeniu svojich znalostí preskúmaním [tohto repozitára](https://microsoft.github.io/forecasting/) a jeho rôznych typov modelov, aby ste sa naučili ďalšie spôsoby vytvárania modelov časových radov.

## Zadanie

[Nový ARIMA model](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.