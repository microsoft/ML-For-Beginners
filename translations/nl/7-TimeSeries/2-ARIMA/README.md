<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T18:58:20+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "nl"
}
-->
# Tijdreeksvoorspelling met ARIMA

In de vorige les heb je wat geleerd over tijdreeksvoorspelling en een dataset geladen die de fluctuaties van het elektriciteitsverbruik over een bepaalde periode laat zien.

[![Introductie tot ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introductie tot ARIMA")

> 🎥 Klik op de afbeelding hierboven voor een video: Een korte introductie tot ARIMA-modellen. Het voorbeeld is gedaan in R, maar de concepten zijn universeel.

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

## Introductie

In deze les ontdek je een specifieke manier om modellen te bouwen met [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). ARIMA-modellen zijn bijzonder geschikt om data te modelleren die [niet-stationair](https://wikipedia.org/wiki/Stationary_process) is.

## Algemene concepten

Om met ARIMA te kunnen werken, zijn er enkele concepten die je moet begrijpen:

- 🎓 **Stationariteit**. Vanuit een statistisch perspectief verwijst stationariteit naar data waarvan de verdeling niet verandert wanneer deze in de tijd wordt verschoven. Niet-stationaire data vertoont daarentegen fluctuaties door trends die moeten worden getransformeerd om te kunnen worden geanalyseerd. Seizoensgebondenheid kan bijvoorbeeld fluctuaties in data veroorzaken en kan worden geëlimineerd door een proces van 'seizoensverschillen'.

- 🎓 **[Verschillen nemen](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Het nemen van verschillen in data, wederom vanuit een statistisch perspectief, verwijst naar het proces van het transformeren van niet-stationaire data om deze stationair te maken door de niet-constante trend te verwijderen. "Het nemen van verschillen verwijdert de veranderingen in het niveau van een tijdreeks, elimineert trends en seizoensgebondenheid en stabiliseert daardoor het gemiddelde van de tijdreeks." [Paper van Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA in de context van tijdreeksen

Laten we de onderdelen van ARIMA uitpakken om beter te begrijpen hoe het ons helpt tijdreeksen te modelleren en voorspellingen te maken.

- **AR - voor AutoRegressief**. Autoregressieve modellen, zoals de naam al aangeeft, kijken 'terug' in de tijd om eerdere waarden in je data te analyseren en aannames over deze waarden te maken. Deze eerdere waarden worden 'lags' genoemd. Een voorbeeld zou data zijn die de maandelijkse verkoop van potloden toont. De verkoopcijfers van elke maand worden beschouwd als een 'evoluerende variabele' in de dataset. Dit model wordt gebouwd door de "evoluerende variabele van interesse te regresseren op zijn eigen vertraagde (d.w.z. eerdere) waarden." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - voor Geïntegreerd**. In tegenstelling tot de vergelijkbare 'ARMA'-modellen verwijst de 'I' in ARIMA naar het *[geïntegreerde](https://wikipedia.org/wiki/Order_of_integration)* aspect. De data wordt 'geïntegreerd' wanneer stappen van verschillen nemen worden toegepast om niet-stationariteit te elimineren.

- **MA - voor Moving Average**. Het [moving-average](https://wikipedia.org/wiki/Moving-average_model)-aspect van dit model verwijst naar de outputvariabele die wordt bepaald door de huidige en eerdere waarden van lags te observeren.

Kort gezegd: ARIMA wordt gebruikt om een model zo nauwkeurig mogelijk te laten aansluiten bij de speciale vorm van tijdreeksdata.

## Oefening - bouw een ARIMA-model

Open de [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working)-map in deze les en vind het bestand [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Voer de notebook uit om de `statsmodels` Python-bibliotheek te laden; je hebt deze nodig voor ARIMA-modellen.

1. Laad de benodigde bibliotheken.

1. Laad nu nog enkele andere bibliotheken die nuttig zijn voor het plotten van data:

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

1. Laad de data uit het bestand `/data/energy.csv` in een Pandas dataframe en bekijk het:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Plot alle beschikbare energiedata van januari 2012 tot december 2014. Er zouden geen verrassingen moeten zijn, aangezien we deze data in de vorige les hebben gezien:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Nu gaan we een model bouwen!

### Maak trainings- en testdatasets

Nu je data is geladen, kun je deze scheiden in trainings- en testsets. Je traint je model op de trainingsset. Zoals gebruikelijk evalueer je na het trainen de nauwkeurigheid van het model met behulp van de testset. Je moet ervoor zorgen dat de testset een latere periode in de tijd beslaat dan de trainingsset, zodat het model geen informatie uit toekomstige tijdsperioden verkrijgt.

1. Wijs een periode van twee maanden toe van 1 september tot 31 oktober 2014 aan de trainingsset. De testset omvat de periode van twee maanden van 1 november tot 31 december 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Aangezien deze data het dagelijkse energieverbruik weerspiegelt, is er een sterk seizoensgebonden patroon, maar het verbruik is het meest vergelijkbaar met het verbruik in meer recente dagen.

1. Visualiseer de verschillen:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![trainings- en testdata](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Daarom zou het gebruik van een relatief klein tijdsvenster voor het trainen van de data voldoende moeten zijn.

    > Opmerking: Aangezien de functie die we gebruiken om het ARIMA-model te passen in-sample validatie gebruikt tijdens het passen, zullen we validatiedata weglaten.

### Bereid de data voor op training

Nu moet je de data voorbereiden op training door filtering en schaling van je data uit te voeren. Filter je dataset om alleen de benodigde tijdsperioden en kolommen op te nemen, en schaal de data zodat deze wordt geprojecteerd in het interval 0,1.

1. Filter de originele dataset om alleen de eerder genoemde tijdsperioden per set en alleen de benodigde kolom 'load' plus de datum op te nemen:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Je kunt de vorm van de data zien:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Schaal de data naar het bereik (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualiseer de originele versus geschaalde data:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![origineel](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > De originele data

    ![geschaald](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > De geschaalde data

1. Nu je de geschaalde data hebt gekalibreerd, kun je de testdata schalen:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementeer ARIMA

Het is tijd om ARIMA te implementeren! Je gaat nu de `statsmodels`-bibliotheek gebruiken die je eerder hebt geïnstalleerd.

Nu moet je een aantal stappen volgen:

   1. Definieer het model door `SARIMAX()` aan te roepen en de modelparameters door te geven: p-, d- en q-parameters, en P-, D- en Q-parameters.
   2. Bereid het model voor op de trainingsdata door de functie `fit()` aan te roepen.
   3. Maak voorspellingen door de functie `forecast()` aan te roepen en het aantal stappen (de `horizon`) te specificeren om te voorspellen.

> 🎓 Waar zijn al deze parameters voor? In een ARIMA-model zijn er 3 parameters die worden gebruikt om de belangrijkste aspecten van een tijdreeks te modelleren: seizoensgebondenheid, trend en ruis. Deze parameters zijn:

`p`: de parameter die verband houdt met het autoregressieve aspect van het model, dat *verleden* waarden incorporeert.
`d`: de parameter die verband houdt met het geïntegreerde deel van het model, dat de hoeveelheid *verschillen nemen* (🎓 herinner je verschillen nemen 👆?) beïnvloedt die op een tijdreeks wordt toegepast.
`q`: de parameter die verband houdt met het moving-average deel van het model.

> Opmerking: Als je data een seizoensgebonden aspect heeft - wat bij deze data het geval is - , gebruiken we een seizoensgebonden ARIMA-model (SARIMA). In dat geval moet je een andere set parameters gebruiken: `P`, `D` en `Q`, die dezelfde associaties beschrijven als `p`, `d` en `q`, maar betrekking hebben op de seizoensgebonden componenten van het model.

1. Begin met het instellen van je gewenste horizonwaarde. Laten we 3 uur proberen:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Het selecteren van de beste waarden voor de parameters van een ARIMA-model kan uitdagend zijn, omdat het enigszins subjectief en tijdrovend is. Je kunt overwegen een `auto_arima()`-functie te gebruiken uit de [`pyramid`-bibliotheek](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Probeer voorlopig enkele handmatige selecties om een goed model te vinden.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Een tabel met resultaten wordt afgedrukt.

Je hebt je eerste model gebouwd! Nu moeten we een manier vinden om het te evalueren.

### Evalueer je model

Om je model te evalueren, kun je de zogenaamde `walk forward`-validatie uitvoeren. In de praktijk worden tijdreeksmodellen elke keer opnieuw getraind wanneer nieuwe data beschikbaar komt. Dit stelt het model in staat om de beste voorspelling te maken op elk tijdstip.

Begin aan het begin van de tijdreeks met deze techniek, train het model op de trainingsdataset. Maak vervolgens een voorspelling voor de volgende tijdstap. De voorspelling wordt geëvalueerd aan de hand van de bekende waarde. De trainingsset wordt vervolgens uitgebreid met de bekende waarde en het proces wordt herhaald.

> Opmerking: Je moet het venster van de trainingsset vast houden voor efficiënter trainen, zodat elke keer dat je een nieuwe observatie aan de trainingsset toevoegt, je de observatie uit het begin van de set verwijdert.

Dit proces biedt een robuustere schatting van hoe het model in de praktijk zal presteren. Het komt echter met de rekenkundige kosten van het maken van zoveel modellen. Dit is acceptabel als de data klein is of als het model eenvoudig is, maar kan een probleem zijn op grotere schaal.

Walk-forward validatie is de gouden standaard voor tijdreeksmodel-evaluatie en wordt aanbevolen voor je eigen projecten.

1. Maak eerst een testdatapunt voor elke HORIZON-stap.

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

    De data wordt horizontaal verschoven volgens het horizonpunt.

1. Maak voorspellingen op je testdata met behulp van deze schuifvensterbenadering in een lus ter grootte van de testdatalengte:

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

    Je kunt het trainen zien gebeuren:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Vergelijk de voorspellingen met de daadwerkelijke load:

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

    Observeer de voorspelling van de uurlijkse data, vergeleken met de daadwerkelijke load. Hoe nauwkeurig is dit?

### Controleer de nauwkeurigheid van je model

Controleer de nauwkeurigheid van je model door de gemiddelde absolute procentuele fout (MAPE) over alle voorspellingen te testen.
> **🧮 Laat me de wiskunde zien**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) wordt gebruikt om de nauwkeurigheid van voorspellingen weer te geven als een verhouding, gedefinieerd door de bovenstaande formule. Het verschil tussen de werkelijke waarde en de voorspelde waarde wordt gedeeld door de werkelijke waarde. 
>
> "De absolute waarde in deze berekening wordt opgeteld voor elk voorspeld tijdstip en gedeeld door het aantal passende punten n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Druk de vergelijking uit in code:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Bereken de MAPE van één stap:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE van één stap voorspelling:  0.5570581332313952 %

1. Print de MAPE van de meerstapsvoorspelling:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Een mooi laag getal is het beste: bedenk dat een voorspelling met een MAPE van 10 een afwijking van 10% heeft.

1. Maar zoals altijd is het makkelijker om dit soort nauwkeurigheidsmetingen visueel te zien, dus laten we het plotten:

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

    ![een tijdreeksmodel](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Een zeer mooie grafiek, die een model met goede nauwkeurigheid laat zien. Goed gedaan!

---

## 🚀Uitdaging

Verdiep je in de manieren om de nauwkeurigheid van een tijdreeksmodel te testen. We behandelen MAPE in deze les, maar zijn er andere methoden die je kunt gebruiken? Onderzoek ze en annoteer ze. Een nuttig document is te vinden [hier](https://otexts.com/fpp2/accuracy.html)

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Deze les behandelt alleen de basis van tijdreeksvoorspelling met ARIMA. Neem de tijd om je kennis te verdiepen door [deze repository](https://microsoft.github.io/forecasting/) en de verschillende modeltypes te verkennen om andere manieren te leren om tijdreeksmodellen te bouwen.

## Opdracht

[Een nieuw ARIMA-model](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.