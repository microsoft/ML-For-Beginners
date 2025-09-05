<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T21:18:44+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "sv"
}
-->
# Tidsserieprognoser med ARIMA

I den föregående lektionen lärde du dig lite om tidsserieprognoser och laddade en dataset som visar variationer i elektrisk belastning över en tidsperiod.

[![Introduktion till ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduktion till ARIMA")

> 🎥 Klicka på bilden ovan för en video: En kort introduktion till ARIMA-modeller. Exemplet görs i R, men koncepten är universella.

## [Quiz före lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

I denna lektion kommer du att upptäcka ett specifikt sätt att bygga modeller med [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). ARIMA-modeller är särskilt lämpade för att passa data som visar [icke-stationaritet](https://wikipedia.org/wiki/Stationary_process).

## Grundläggande koncept

För att kunna arbeta med ARIMA finns det några begrepp du behöver känna till:

- 🎓 **Stationaritet**. Ur ett statistiskt perspektiv hänvisar stationaritet till data vars fördelning inte förändras när den förskjuts i tid. Icke-stationär data visar däremot variationer på grund av trender som måste transformeras för att kunna analyseras. Säsongsvariationer, till exempel, kan introducera fluktuationer i data och kan elimineras genom en process som kallas 'säsongsdifferensiering'.

- 🎓 **[Differensiering](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Differensiering av data, återigen ur ett statistiskt perspektiv, hänvisar till processen att transformera icke-stationär data för att göra den stationär genom att ta bort dess icke-konstanta trend. "Differensiering tar bort förändringar i nivån på en tidsserie, eliminerar trend och säsongsvariationer och stabiliserar därmed medelvärdet för tidsserien." [Artikel av Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA i kontexten av tidsserier

Låt oss bryta ner delarna av ARIMA för att bättre förstå hur det hjälper oss att modellera tidsserier och göra prognoser baserat på dem.

- **AR - för AutoRegressiv**. Autoregressiva modeller, som namnet antyder, tittar 'bakåt' i tiden för att analysera tidigare värden i din data och göra antaganden om dem. Dessa tidigare värden kallas 'lags'. Ett exempel skulle vara data som visar månatlig försäljning av pennor. Varje månads försäljningssiffra skulle betraktas som en 'utvecklande variabel' i datasetet. Denna modell byggs som "den utvecklande variabeln av intresse regresseras på sina egna fördröjda (dvs. tidigare) värden." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - för Integrerad**. Till skillnad från liknande 'ARMA'-modeller hänvisar 'I' i ARIMA till dess *[integrerade](https://wikipedia.org/wiki/Order_of_integration)* aspekt. Data är 'integrerad' när differensieringssteg tillämpas för att eliminera icke-stationaritet.

- **MA - för Glidande Medelvärde**. Den [glidande medelvärdes](https://wikipedia.org/wiki/Moving-average_model)-aspekten av denna modell hänvisar till utgångsvariabeln som bestäms genom att observera de aktuella och tidigare värdena av lags.

Slutsats: ARIMA används för att skapa en modell som passar den speciella formen av tidsseriedata så nära som möjligt.

## Övning - bygg en ARIMA-modell

Öppna mappen [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) i denna lektion och hitta filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Kör notebooken för att ladda Python-biblioteket `statsmodels`; du kommer att behöva detta för ARIMA-modeller.

1. Ladda nödvändiga bibliotek.

1. Ladda nu upp flera fler bibliotek som är användbara för att plotta data:

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

1. Ladda data från filen `/data/energy.csv` till en Pandas-dataram och ta en titt:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Plotta all tillgänglig energidata från januari 2012 till december 2014. Det borde inte vara några överraskningar eftersom vi såg denna data i den senaste lektionen:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Nu ska vi bygga en modell!

### Skapa tränings- och testdatamängder

Nu är din data laddad, så du kan dela upp den i tränings- och testmängder. Du kommer att träna din modell på träningsmängden. Som vanligt, efter att modellen har avslutat träningen, kommer du att utvärdera dess noggrannhet med hjälp av testmängden. Du måste säkerställa att testmängden täcker en senare tidsperiod än träningsmängden för att säkerställa att modellen inte får information från framtida tidsperioder.

1. Tilldela en tvåmånadersperiod från 1 september till 31 oktober 2014 till träningsmängden. Testmängden kommer att inkludera tvåmånadersperioden från 1 november till 31 december 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Eftersom denna data reflekterar den dagliga energiförbrukningen finns det ett starkt säsongsmönster, men förbrukningen är mest lik förbrukningen under mer nyliga dagar.

1. Visualisera skillnaderna:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![tränings- och testdata](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Därför bör det vara tillräckligt att använda ett relativt litet tidsfönster för att träna datan.

    > Obs: Eftersom funktionen vi använder för att passa ARIMA-modellen använder in-sample validering under passningen, kommer vi att utelämna valideringsdata.

### Förbered datan för träning

Nu behöver du förbereda datan för träning genom att filtrera och skala din data. Filtrera din dataset för att endast inkludera de tidsperioder och kolumner du behöver, och skala för att säkerställa att datan projiceras inom intervallet 0,1.

1. Filtrera den ursprungliga datasetet för att endast inkludera de nämnda tidsperioderna per mängd och endast inkludera den behövda kolumnen 'load' plus datumet:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Du kan se formen på datan:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skala datan till att vara inom intervallet (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualisera den ursprungliga vs. skalade datan:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![ursprunglig](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Den ursprungliga datan

    ![skalad](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Den skalade datan

1. Nu när du har kalibrerat den skalade datan kan du skala testdatan:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementera ARIMA

Det är dags att implementera ARIMA! Du kommer nu att använda `statsmodels`-biblioteket som du installerade tidigare.

Nu behöver du följa flera steg:

   1. Definiera modellen genom att kalla på `SARIMAX()` och skicka in modellparametrarna: p, d och q-parametrar, samt P, D och Q-parametrar.
   2. Förbered modellen för träningsdatan genom att kalla på funktionen `fit()`.
   3. Gör prognoser genom att kalla på funktionen `forecast()` och specificera antalet steg (prognoshorisonten) att förutsäga.

> 🎓 Vad är alla dessa parametrar till för? I en ARIMA-modell finns det 3 parametrar som används för att hjälpa till att modellera de huvudsakliga aspekterna av en tidsserie: säsongsvariation, trend och brus. Dessa parametrar är:

`p`: parametern som är associerad med den autoregressiva aspekten av modellen, som inkorporerar *tidigare* värden.  
`d`: parametern som är associerad med den integrerade delen av modellen, som påverkar mängden *differensiering* (🎓 kom ihåg differensiering 👆?) som ska tillämpas på en tidsserie.  
`q`: parametern som är associerad med den glidande medelvärdesdelen av modellen.

> Obs: Om din data har en säsongsaspekt - vilket denna har - använder vi en säsongs-ARIMA-modell (SARIMA). I så fall behöver du använda en annan uppsättning parametrar: `P`, `D` och `Q` som beskriver samma associationer som `p`, `d` och `q`, men motsvarar de säsongsbetonade komponenterna i modellen.

1. Börja med att ställa in din föredragna horisontvärde. Låt oss prova 3 timmar:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Att välja de bästa värdena för en ARIMA-modells parametrar kan vara utmanande eftersom det är något subjektivt och tidskrävande. Du kan överväga att använda en `auto_arima()`-funktion från [`pyramid`-biblioteket](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. För tillfället, prova några manuella val för att hitta en bra modell.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    En tabell med resultat skrivs ut.

Du har byggt din första modell! Nu behöver vi hitta ett sätt att utvärdera den.

### Utvärdera din modell

För att utvärdera din modell kan du utföra den så kallade `walk forward`-valideringen. I praktiken tränas tidsseriemodeller om varje gång ny data blir tillgänglig. Detta gör att modellen kan göra den bästa prognosen vid varje tidssteg.

Börja vid början av tidsserien med denna teknik, träna modellen på träningsdatamängden. Gör sedan en prognos för nästa tidssteg. Prognosen utvärderas mot det kända värdet. Träningsmängden utökas sedan för att inkludera det kända värdet och processen upprepas.

> Obs: Du bör hålla träningsmängdens fönster fast för mer effektiv träning så att varje gång du lägger till en ny observation till träningsmängden, tar du bort observationen från början av mängden.

Denna process ger en mer robust uppskattning av hur modellen kommer att prestera i praktiken. Dock kommer det till kostnaden av att skapa så många modeller. Detta är acceptabelt om datan är liten eller om modellen är enkel, men kan vara ett problem i större skala.

Walk-forward-validering är guldstandarden för utvärdering av tidsseriemodeller och rekommenderas för dina egna projekt.

1. Först, skapa en testdatapunkt för varje HORIZON-steg.

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

    Datan förskjuts horisontellt enligt dess horisontpunkt.

1. Gör prognoser på din testdata med denna glidande fönsteransats i en loop av testdatans längd:

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

    Du kan se träningen ske:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Jämför prognoserna med den faktiska belastningen:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Output  
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Observera prognosen för timdata jämfört med den faktiska belastningen. Hur noggrann är den?

### Kontrollera modellens noggrannhet

Kontrollera noggrannheten för din modell genom att testa dess medelabsoluta procentuella fel (MAPE) över alla prognoser.
> **🧮 Visa mig matematiken**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) används för att visa prognosnoggrannhet som en kvot definierad av formeln ovan. Skillnaden mellan verkligt och förutspått delas med det verkliga.
>
> "Det absoluta värdet i denna beräkning summeras för varje prognostiserad tidpunkt och delas med antalet anpassade punkter n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Uttryck ekvationen i kod:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Beräkna MAPE för ett steg:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE för ett steg:  0.5570581332313952 %

1. Skriv ut MAPE för fler steg:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Ett lågt värde är bäst: tänk på att en prognos med en MAPE på 10 innebär att den avviker med 10%.

1. Men som alltid är det enklare att se denna typ av noggrannhetsmätning visuellt, så låt oss plotta det:

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

    ![en tidsseriemodell](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 En mycket fin graf som visar en modell med god noggrannhet. Bra jobbat!

---

## 🚀Utmaning

Utforska olika sätt att testa noggrannheten hos en tidsseriemodell. Vi berör MAPE i denna lektion, men finns det andra metoder du kan använda? Undersök dem och kommentera dem. Ett användbart dokument finns [här](https://otexts.com/fpp2/accuracy.html)

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Repetition & Självstudier

Denna lektion täcker endast grunderna i tidsserieprognoser med ARIMA. Ta dig tid att fördjupa dina kunskaper genom att utforska [detta repository](https://microsoft.github.io/forecasting/) och dess olika modelltyper för att lära dig andra sätt att bygga tidsseriemodeller.

## Uppgift

[En ny ARIMA-modell](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.