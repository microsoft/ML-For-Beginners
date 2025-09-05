<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T23:46:21+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "da"
}
-->
# Tidsserieprognoser med ARIMA

I den forrige lektion lærte du lidt om tidsserieprognoser og indlæste et datasæt, der viser udsving i elektrisk belastning over en tidsperiode.

[![Introduktion til ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduktion til ARIMA")

> 🎥 Klik på billedet ovenfor for en video: En kort introduktion til ARIMA-modeller. Eksemplet er lavet i R, men konceptet er universelt.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

I denne lektion vil du lære en specifik metode til at bygge modeller med [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). ARIMA-modeller er særligt velegnede til at tilpasse data, der viser [ikke-stationaritet](https://wikipedia.org/wiki/Stationary_process).

## Generelle begreber

For at kunne arbejde med ARIMA er der nogle begreber, du skal kende til:

- 🎓 **Stationaritet**. Fra et statistisk perspektiv refererer stationaritet til data, hvis fordeling ikke ændrer sig, når den forskydes i tid. Ikke-stationære data viser derimod udsving på grund af tendenser, som skal transformeres for at kunne analyseres. Sæsonvariationer kan for eksempel introducere udsving i data og kan fjernes ved en proces kaldet 'sæsonmæssig differencering'.

- 🎓 **[Differencering](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Differencering af data, igen fra et statistisk perspektiv, refererer til processen med at transformere ikke-stationære data for at gøre dem stationære ved at fjerne deres ikke-konstante tendens. "Differencering fjerner ændringer i niveauet af en tidsserie, eliminerer tendens og sæsonvariation og stabiliserer dermed gennemsnittet af tidsserien." [Papir af Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA i konteksten af tidsserier

Lad os pakke ARIMA's dele ud for bedre at forstå, hvordan det hjælper os med at modellere tidsserier og lave forudsigelser baseret på dem.

- **AR - for AutoRegressiv**. Autoregressive modeller ser, som navnet antyder, 'tilbage' i tiden for at analysere tidligere værdier i dine data og lave antagelser om dem. Disse tidligere værdier kaldes 'lags'. Et eksempel kunne være data, der viser månedlige salg af blyanter. Hver måneds salgstal ville blive betragtet som en 'udviklende variabel' i datasættet. Denne model bygges ved, at "den udviklende variabel af interesse regresseres på sine egne forsinkede (dvs. tidligere) værdier." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - for Integreret**. I modsætning til de lignende 'ARMA'-modeller refererer 'I' i ARIMA til dens *[integrerede](https://wikipedia.org/wiki/Order_of_integration)* aspekt. Dataene bliver 'integreret', når differenceringstrin anvendes for at eliminere ikke-stationaritet.

- **MA - for Glidende Gennemsnit**. Det [glidende gennemsnit](https://wikipedia.org/wiki/Moving-average_model)-aspekt af denne model refererer til outputvariablen, der bestemmes ved at observere de aktuelle og tidligere værdier af lags.

Kort sagt: ARIMA bruges til at få en model til at passe så tæt som muligt til den særlige form for tidsseriedata.

## Øvelse - byg en ARIMA-model

Åbn [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working)-mappen i denne lektion og find filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Kør notebooken for at indlæse Python-biblioteket `statsmodels`; du skal bruge dette til ARIMA-modeller.

1. Indlæs nødvendige biblioteker.

1. Indlæs nu flere biblioteker, der er nyttige til at plotte data:

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

1. Indlæs data fra filen `/data/energy.csv` i en Pandas dataframe og kig på det:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Plot alle tilgængelige energidata fra januar 2012 til december 2014. Der bør ikke være nogen overraskelser, da vi så disse data i den sidste lektion:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Nu skal vi bygge en model!

### Opret trænings- og testdatasæt

Nu er dine data indlæst, så du kan opdele dem i trænings- og testdatasæt. Du vil træne din model på træningssættet. Som sædvanligt, efter at modellen er færdig med at træne, vil du evaluere dens nøjagtighed ved hjælp af testdatasættet. Du skal sikre dig, at testdatasættet dækker en senere periode end træningssættet for at sikre, at modellen ikke får information fra fremtidige tidsperioder.

1. Tildel en to-måneders periode fra 1. september til 31. oktober 2014 til træningssættet. Testdatasættet vil inkludere to-måneders perioden fra 1. november til 31. december 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Da disse data afspejler det daglige energiforbrug, er der et stærkt sæsonmønster, men forbruget ligner mest forbruget i de mere nylige dage.

1. Visualiser forskellene:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![trænings- og testdata](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Derfor bør det være tilstrækkeligt at bruge et relativt lille tidsvindue til at træne dataene.

    > Bemærk: Da den funktion, vi bruger til at tilpasse ARIMA-modellen, bruger in-sample validering under tilpasning, vil vi udelade valideringsdata.

### Forbered dataene til træning

Nu skal du forberede dataene til træning ved at udføre filtrering og skalering af dine data. Filtrer dit datasæt, så det kun inkluderer de nødvendige tidsperioder og kolonner, og skaler dataene for at sikre, at de projiceres i intervallet 0,1.

1. Filtrer det originale datasæt, så det kun inkluderer de nævnte tidsperioder pr. sæt og kun inkluderer den nødvendige kolonne 'load' plus datoen:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Du kan se formen af dataene:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skaler dataene til at være i intervallet (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualiser de originale vs. skalerede data:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > De originale data

    ![skaleret](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > De skalerede data

1. Nu hvor du har kalibreret de skalerede data, kan du skalere testdataene:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementer ARIMA

Det er tid til at implementere ARIMA! Du vil nu bruge `statsmodels`-biblioteket, som du installerede tidligere.

Nu skal du følge flere trin:

   1. Definer modellen ved at kalde `SARIMAX()` og angive modelparametrene: p-, d- og q-parametre samt P-, D- og Q-parametre.
   2. Forbered modellen til træningsdataene ved at kalde funktionen `fit()`.
   3. Lav forudsigelser ved at kalde funktionen `forecast()` og angive antallet af trin (horisonten), der skal forudsiges.

> 🎓 Hvad er alle disse parametre til? I en ARIMA-model er der 3 parametre, der bruges til at hjælpe med at modellere de vigtigste aspekter af en tidsserie: sæsonvariation, tendens og støj. Disse parametre er:

`p`: parameteren, der er forbundet med den autoregressive del af modellen, som inkorporerer *tidligere* værdier.  
`d`: parameteren, der er forbundet med den integrerede del af modellen, som påvirker mængden af *differencering* (🎓 husk differencering 👆?) der skal anvendes på en tidsserie.  
`q`: parameteren, der er forbundet med den glidende gennemsnitsdel af modellen.

> Bemærk: Hvis dine data har en sæsonmæssig komponent - hvilket disse data har - bruger vi en sæsonmæssig ARIMA-model (SARIMA). I så fald skal du bruge et andet sæt parametre: `P`, `D` og `Q`, som beskriver de samme associationer som `p`, `d` og `q`, men svarer til de sæsonmæssige komponenter i modellen.

1. Start med at indstille din foretrukne horisontværdi. Lad os prøve 3 timer:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Det kan være udfordrende at vælge de bedste værdier for en ARIMA-models parametre, da det er noget subjektivt og tidskrævende. Du kan overveje at bruge en `auto_arima()`-funktion fra [`pyramid`-biblioteket](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Prøv for nu nogle manuelle valg for at finde en god model.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    En tabel med resultater udskrives.

Du har bygget din første model! Nu skal vi finde en måde at evaluere den på.

### Evaluer din model

For at evaluere din model kan du udføre den såkaldte `walk forward`-validering. I praksis bliver tidsseriemodeller gen-trænet hver gang nye data bliver tilgængelige. Dette giver modellen mulighed for at lave den bedste forudsigelse på hvert tidspunkt.

Startende fra begyndelsen af tidsserien, træner du modellen på træningsdatasættet. Derefter laver du en forudsigelse for det næste tidspunkt. Forudsigelsen evalueres mod den kendte værdi. Træningssættet udvides derefter til at inkludere den kendte værdi, og processen gentages.

> Bemærk: Du bør holde træningssættets vindue fast for mere effektiv træning, så hver gang du tilføjer en ny observation til træningssættet, fjerner du observationen fra begyndelsen af sættet.

Denne proces giver en mere robust estimering af, hvordan modellen vil præstere i praksis. Dog kommer det med den beregningsmæssige omkostning ved at skabe så mange modeller. Dette er acceptabelt, hvis dataene er små eller modellen er simpel, men kan være et problem i stor skala.

Walk-forward validering er guldstandarden for evaluering af tidsseriemodeller og anbefales til dine egne projekter.

1. Først, opret et testdatapunkt for hvert HORIZON-trin.

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

    Dataene forskydes horisontalt i henhold til deres horisontpunkt.

1. Lav forudsigelser på dine testdata ved hjælp af denne glidende vinduesmetode i en løkke på størrelse med testdatasættets længde:

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

    Du kan se træningen foregå:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Sammenlign forudsigelserne med den faktiske belastning:

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

    Observer forudsigelsen af den timelige data sammenlignet med den faktiske belastning. Hvor nøjagtig er den?

### Tjek modellens nøjagtighed

Tjek nøjagtigheden af din model ved at teste dens gennemsnitlige absolutte procentfejl (MAPE) over alle forudsigelserne.
> **🧮 Vis mig matematikken**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) bruges til at vise forudsigelsesnøjagtighed som et forhold defineret af ovenstående formel. Forskellen mellem faktisk og forudsagt deles med den faktiske værdi. "Den absolutte værdi i denne beregning summeres for hvert forudsagt tidspunkt og deles med antallet af tilpassede punkter n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Udtryk ligningen i kode:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Beregn MAPE for ét trin:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE for ét trin:  0.5570581332313952 %

1. Udskriv MAPE for multi-trins prognose:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Et lavt tal er bedst: husk, at en prognose med en MAPE på 10 er 10% ved siden af.

1. Men som altid er det nemmere at se denne type nøjagtighedsmåling visuelt, så lad os plotte det:

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

    ![en tidsseriemodel](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Et rigtig flot plot, der viser en model med god nøjagtighed. Godt arbejde!

---

## 🚀Udfordring

Undersøg metoder til at teste nøjagtigheden af en tidsseriemodel. Vi berører MAPE i denne lektion, men er der andre metoder, du kunne bruge? Undersøg dem og annotér dem. Et nyttigt dokument kan findes [her](https://otexts.com/fpp2/accuracy.html)

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Denne lektion berører kun det grundlæggende i tidsserieprognoser med ARIMA. Brug lidt tid på at uddybe din viden ved at undersøge [dette repository](https://microsoft.github.io/forecasting/) og dets forskellige modeltyper for at lære andre måder at bygge tidsseriemodeller på.

## Opgave

[En ny ARIMA-model](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.