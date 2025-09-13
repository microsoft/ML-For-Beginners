<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T21:19:25+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "no"
}
-->
# Tidsserieprognoser med ARIMA

I forrige leksjon lærte du litt om tidsserieprognoser og lastet inn et datasett som viser svingninger i elektrisk belastning over en tidsperiode.

[![Introduksjon til ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduksjon til ARIMA")

> 🎥 Klikk på bildet ovenfor for en video: En kort introduksjon til ARIMA-modeller. Eksempelet er gjort i R, men konseptene er universelle.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Introduksjon

I denne leksjonen vil du oppdage en spesifikk måte å bygge modeller på med [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). ARIMA-modeller er spesielt godt egnet til å tilpasse data som viser [ikke-stasjonaritet](https://wikipedia.org/wiki/Stationary_process).

## Generelle konsepter

For å kunne jobbe med ARIMA, er det noen konsepter du må kjenne til:

- 🎓 **Stasjonaritet**. Fra et statistisk perspektiv refererer stasjonaritet til data der distribusjonen ikke endres når den forskyves i tid. Ikke-stasjonære data viser derimot svingninger på grunn av trender som må transformeres for å kunne analyseres. Sesongvariasjoner, for eksempel, kan introdusere svingninger i data og kan elimineres ved en prosess kalt 'sesongdifferensiering'.

- 🎓 **[Differensiering](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Differensiering av data, igjen fra et statistisk perspektiv, refererer til prosessen med å transformere ikke-stasjonære data for å gjøre dem stasjonære ved å fjerne deres ikke-konstante trend. "Differensiering fjerner endringene i nivået til en tidsserie, eliminerer trend og sesongvariasjoner og stabiliserer dermed gjennomsnittet av tidsserien." [Paper av Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA i konteksten av tidsserier

La oss pakke ut delene av ARIMA for bedre å forstå hvordan det hjelper oss med å modellere tidsserier og lage prognoser basert på dem.

- **AR - for AutoRegressiv**. Autoregressive modeller, som navnet antyder, ser 'tilbake' i tid for å analysere tidligere verdier i dataene dine og gjøre antakelser om dem. Disse tidligere verdiene kalles 'lags'. Et eksempel kan være data som viser månedlige salg av blyanter. Hver måneds salgssum vil bli betraktet som en 'utviklende variabel' i datasettet. Denne modellen bygges ved at "den utviklende variabelen av interesse regresseres på sine egne forsinkede (dvs. tidligere) verdier." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - for Integrert**. I motsetning til de lignende 'ARMA'-modellene, refererer 'I' i ARIMA til dens *[integrerte](https://wikipedia.org/wiki/Order_of_integration)* aspekt. Dataene blir 'integrert' når differensieringstrinn brukes for å eliminere ikke-stasjonaritet.

- **MA - for Glidende Gjennomsnitt**. Det [glidende gjennomsnittet](https://wikipedia.org/wiki/Moving-average_model) i denne modellen refererer til utgangsvariabelen som bestemmes ved å observere nåværende og tidligere verdier av lags.

Kort oppsummert: ARIMA brukes til å lage en modell som passer så tett som mulig til den spesielle formen for tidsseriedata.

## Øvelse - bygg en ARIMA-modell

Åpne [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working)-mappen i denne leksjonen og finn filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Kjør notebooken for å laste inn Python-biblioteket `statsmodels`; du trenger dette for ARIMA-modeller.

1. Last inn nødvendige biblioteker.

1. Deretter laster du inn flere biblioteker som er nyttige for å plotte data:

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

1. Last inn data fra `/data/energy.csv`-filen til en Pandas dataframe og ta en titt:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Plot all tilgjengelig energidata fra januar 2012 til desember 2014. Det bør ikke være noen overraskelser, da vi så disse dataene i forrige leksjon:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Nå, la oss bygge en modell!

### Opprett trenings- og testdatasett

Nå er dataene dine lastet inn, så du kan dele dem opp i trenings- og testsett. Du vil trene modellen din på treningssettet. Som vanlig, etter at modellen er ferdig trent, vil du evaluere nøyaktigheten ved hjelp av testsettet. Du må sørge for at testsettet dekker en senere tidsperiode enn treningssettet for å sikre at modellen ikke får informasjon fra fremtidige tidsperioder.

1. Tildel en to-måneders periode fra 1. september til 31. oktober 2014 til treningssettet. Testsettet vil inkludere to-måneders perioden fra 1. november til 31. desember 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Siden disse dataene reflekterer det daglige energiforbruket, er det et sterkt sesongmønster, men forbruket er mest likt forbruket i mer nylige dager.

1. Visualiser forskjellene:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![trenings- og testdata](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Derfor bør det være tilstrekkelig å bruke et relativt lite tidsvindu for å trene dataene.

    > Merk: Siden funksjonen vi bruker for å tilpasse ARIMA-modellen bruker in-sample validering under tilpasning, vil vi utelate valideringsdata.

### Forbered dataene for trening

Nå må du forberede dataene for trening ved å filtrere og skalere dem. Filtrer datasettet ditt for kun å inkludere de nødvendige tidsperiodene og kolonnene, og skaler dataene for å sikre at de projiseres i intervallet 0,1.

1. Filtrer det originale datasettet for kun å inkludere de nevnte tidsperiodene per sett og kun inkludere den nødvendige kolonnen 'load' pluss datoen:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Du kan se formen på dataene:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skaler dataene til å være i området (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualiser de originale vs. skalerte dataene:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > De originale dataene

    ![skalert](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > De skalerte dataene

1. Nå som du har kalibrert de skalerte dataene, kan du skalere testdataene:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementer ARIMA

Det er på tide å implementere ARIMA! Du vil nå bruke `statsmodels`-biblioteket som du installerte tidligere.

Nå må du følge flere trinn:

   1. Definer modellen ved å kalle `SARIMAX()` og sende inn modellparametrene: p, d og q-parametere, samt P, D og Q-parametere.
   2. Forbered modellen for treningsdataene ved å kalle fit()-funksjonen.
   3. Lag prognoser ved å kalle `forecast()`-funksjonen og spesifisere antall steg (horisonten) som skal prognoseres.

> 🎓 Hva er alle disse parameterne til? I en ARIMA-modell er det 3 parametere som brukes for å modellere de viktigste aspektene ved en tidsserie: sesongvariasjon, trend og støy. Disse parameterne er:

`p`: parameteren knyttet til den autoregressive delen av modellen, som inkorporerer *tidligere* verdier.  
`d`: parameteren knyttet til den integrerte delen av modellen, som påvirker mengden *differensiering* (🎓 husk differensiering 👆?) som skal brukes på en tidsserie.  
`q`: parameteren knyttet til den glidende gjennomsnittsdelen av modellen.

> Merk: Hvis dataene dine har en sesongmessig komponent - som disse dataene har - bruker vi en sesongmessig ARIMA-modell (SARIMA). I så fall må du bruke et annet sett med parametere: `P`, `D` og `Q`, som beskriver de samme assosiasjonene som `p`, `d` og `q`, men tilsvarer de sesongmessige komponentene i modellen.

1. Start med å sette din foretrukne horisontverdi. La oss prøve 3 timer:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Å velge de beste verdiene for en ARIMA-modells parametere kan være utfordrende, da det er noe subjektivt og tidkrevende. Du kan vurdere å bruke en `auto_arima()`-funksjon fra [`pyramid`-biblioteket](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. For nå, prøv noen manuelle valg for å finne en god modell.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    En tabell med resultater blir skrevet ut.

Du har bygget din første modell! Nå må vi finne en måte å evaluere den på.

### Evaluer modellen din

For å evaluere modellen din kan du utføre den såkalte `walk forward`-valideringen. I praksis blir tidsseriemodeller re-trent hver gang nye data blir tilgjengelige. Dette lar modellen lage den beste prognosen ved hvert tidssteg.

Start ved begynnelsen av tidsserien med denne teknikken, tren modellen på treningsdatasettet. Deretter lager du en prognose for neste tidssteg. Prognosen evalueres mot den kjente verdien. Treningssettet utvides deretter til å inkludere den kjente verdien, og prosessen gjentas.

> Merk: Du bør holde treningssettets vindu fast for mer effektiv trening, slik at hver gang du legger til en ny observasjon i treningssettet, fjerner du observasjonen fra begynnelsen av settet.

Denne prosessen gir en mer robust estimering av hvordan modellen vil prestere i praksis. Imidlertid kommer det med beregningskostnaden ved å lage så mange modeller. Dette er akseptabelt hvis dataene er små eller hvis modellen er enkel, men kan være et problem i stor skala.

Walk-forward validering er gullstandarden for evaluering av tidsseriemodeller og anbefales for dine egne prosjekter.

1. Først, opprett et testdatapunkt for hvert HORIZON-steg.

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

    Dataene forskyves horisontalt i henhold til horisontpunktet.

1. Lag prognoser for testdataene dine ved hjelp av denne glidende vindustilnærmingen i en løkke på størrelse med testdatasettets lengde:

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

    Du kan se treningen foregå:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Sammenlign prognosene med den faktiske belastningen:

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

    Observer prognosen for timebaserte data sammenlignet med den faktiske belastningen. Hvor nøyaktig er dette?

### Sjekk modellens nøyaktighet

Sjekk nøyaktigheten til modellen din ved å teste dens gjennomsnittlige absolutte prosentvise feil (MAPE) over alle prognosene.
> **🧮 Vis meg matematikken**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) brukes for å vise prediksjonsnøyaktighet som et forhold definert av formelen ovenfor. Forskjellen mellom faktisk og forutsagt deles på det faktiske.
>
> "Den absolutte verdien i denne beregningen summeres for hvert prognosert tidspunkt og deles på antall tilpassede punkter n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Uttrykk ligningen i kode:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Beregn MAPE for ett steg:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE for ett steg:  0.5570581332313952 %

1. Skriv ut MAPE for flertrinnsprognosen:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Et lavt tall er best: husk at en prognose med en MAPE på 10 er feil med 10%.

1. Men som alltid, det er enklere å se denne typen nøyaktighetsmåling visuelt, så la oss plotte det:

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

🏆 Et veldig fint plot som viser en modell med god nøyaktighet. Bra jobbet!

---

## 🚀Utfordring

Utforsk måter å teste nøyaktigheten til en tidsseriemodell. Vi berører MAPE i denne leksjonen, men finnes det andre metoder du kan bruke? Undersøk dem og kommenter dem. Et nyttig dokument kan finnes [her](https://otexts.com/fpp2/accuracy.html)

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Denne leksjonen berører kun det grunnleggende innen tidsserieprognoser med ARIMA. Ta deg tid til å utdype kunnskapen din ved å utforske [dette repositoriet](https://microsoft.github.io/forecasting/) og dets ulike modelltyper for å lære andre måter å bygge tidsseriemodeller på.

## Oppgave

[En ny ARIMA-modell](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.