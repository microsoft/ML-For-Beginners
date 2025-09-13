<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T15:30:48+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "ro"
}
-->
# Prognoza seriilor temporale cu ARIMA

În lecția anterioară, ai învățat câte ceva despre prognoza seriilor temporale și ai încărcat un set de date care arată fluctuațiile consumului de energie electrică pe o anumită perioadă de timp.

[![Introducere în ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introducere în ARIMA")

> 🎥 Fă clic pe imaginea de mai sus pentru un videoclip: O scurtă introducere în modelele ARIMA. Exemplul este realizat în R, dar conceptele sunt universale.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

## Introducere

În această lecție, vei descoperi o metodă specifică de a construi modele folosind [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Modelele ARIMA sunt deosebit de potrivite pentru a se adapta datelor care prezintă [non-staționaritate](https://wikipedia.org/wiki/Stationary_process).

## Concepte generale

Pentru a putea lucra cu ARIMA, există câteva concepte pe care trebuie să le cunoști:

- 🎓 **Staționaritate**. Dintr-un context statistic, staționaritatea se referă la datele a căror distribuție nu se schimbă atunci când sunt deplasate în timp. Datele non-staționare, în schimb, prezintă fluctuații datorate tendințelor care trebuie transformate pentru a fi analizate. Sezonalitatea, de exemplu, poate introduce fluctuații în date și poate fi eliminată printr-un proces de 'diferențiere sezonieră'.

- 🎓 **[Diferențiere](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Diferențierea datelor, din nou dintr-un context statistic, se referă la procesul de transformare a datelor non-staționare pentru a le face staționare prin eliminarea tendinței lor non-constante. "Diferențierea elimină schimbările în nivelul unei serii temporale, eliminând tendințele și sezonalitatea și, în consecință, stabilizând media seriei temporale." [Lucrare de Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA în contextul seriilor temporale

Să analizăm componentele ARIMA pentru a înțelege mai bine cum ne ajută să modelăm seriile temporale și să facem predicții pe baza acestora.

- **AR - pentru AutoRegresiv**. Modelele autoregresive, așa cum sugerează numele, privesc 'înapoi' în timp pentru a analiza valorile anterioare din datele tale și pentru a face presupuneri despre acestea. Aceste valori anterioare sunt numite 'lags'. Un exemplu ar fi datele care arată vânzările lunare de creioane. Totalul vânzărilor din fiecare lună ar fi considerat o 'variabilă în evoluție' în setul de date. Acest model este construit astfel încât "variabila de interes în evoluție este regresată pe valorile sale întârziate (adică, anterioare)." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - pentru Integrat**. Spre deosebire de modelele similare 'ARMA', 'I' din ARIMA se referă la aspectul său *[integrat](https://wikipedia.org/wiki/Order_of_integration)*. Datele sunt 'integrate' atunci când sunt aplicate pași de diferențiere pentru a elimina non-staționaritatea.

- **MA - pentru Medie Mobilă**. Aspectul de [medie mobilă](https://wikipedia.org/wiki/Moving-average_model) al acestui model se referă la variabila de ieșire care este determinată prin observarea valorilor curente și anterioare ale lagurilor.

Pe scurt: ARIMA este utilizat pentru a face ca un model să se potrivească cât mai bine cu forma specială a datelor din seriile temporale.

## Exercițiu - construiește un model ARIMA

Deschide folderul [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) din această lecție și găsește fișierul [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Rulează notebook-ul pentru a încărca biblioteca Python `statsmodels`; vei avea nevoie de aceasta pentru modelele ARIMA.

1. Încarcă bibliotecile necesare.

1. Acum, încarcă mai multe biblioteci utile pentru a reprezenta grafic datele:

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

1. Încarcă datele din fișierul `/data/energy.csv` într-un dataframe Pandas și analizează-le:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Reprezintă grafic toate datele disponibile despre consumul de energie din ianuarie 2012 până în decembrie 2014. Nu ar trebui să existe surprize, deoarece am văzut aceste date în lecția anterioară:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Acum, să construim un model!

### Creează seturi de date pentru antrenare și testare

Acum că datele tale sunt încărcate, le poți separa în seturi de antrenare și testare. Vei antrena modelul pe setul de antrenare. Ca de obicei, după ce modelul a terminat antrenarea, îi vei evalua acuratețea folosind setul de testare. Trebuie să te asiguri că setul de testare acoperă o perioadă ulterioară în timp față de setul de antrenare pentru a te asigura că modelul nu obține informații din perioadele viitoare.

1. Alocă o perioadă de două luni, de la 1 septembrie până la 31 octombrie 2014, pentru setul de antrenare. Setul de testare va include perioada de două luni de la 1 noiembrie până la 31 decembrie 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Deoarece aceste date reflectă consumul zilnic de energie, există un model sezonier puternic, dar consumul este cel mai asemănător cu consumul din zilele mai recente.

1. Vizualizează diferențele:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![date de antrenare și testare](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Prin urmare, utilizarea unei ferestre relativ mici de timp pentru antrenarea datelor ar trebui să fie suficientă.

    > Notă: Deoarece funcția pe care o folosim pentru a ajusta modelul ARIMA utilizează validare în eșantion în timpul ajustării, vom omite datele de validare.

### Pregătește datele pentru antrenare

Acum, trebuie să pregătești datele pentru antrenare prin filtrarea și scalarea acestora. Filtrează setul de date pentru a include doar perioadele de timp și coloanele necesare și scalează datele pentru a te asigura că sunt proiectate în intervalul 0,1.

1. Filtrează setul de date original pentru a include doar perioadele de timp menționate anterior pentru fiecare set și doar coloana necesară 'load' plus data:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Poți vedea forma datelor:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Scalează datele pentru a fi în intervalul (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Vizualizează datele originale vs. datele scalate:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Datele originale

    ![scaled](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Datele scalate

1. Acum că ai calibrat datele scalate, poți scala datele de testare:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementează ARIMA

Este timpul să implementezi ARIMA! Acum vei folosi biblioteca `statsmodels` pe care ai instalat-o mai devreme.

Acum trebuie să urmezi câțiva pași:

   1. Definește modelul apelând `SARIMAX()` și trecând parametrii modelului: parametrii p, d și q, și parametrii P, D și Q.
   2. Pregătește modelul pentru datele de antrenare apelând funcția `fit()`.
   3. Fă predicții apelând funcția `forecast()` și specificând numărul de pași (orizontul) pentru prognoză.

> 🎓 Ce reprezintă toți acești parametri? Într-un model ARIMA există 3 parametri care sunt utilizați pentru a ajuta la modelarea principalelor aspecte ale unei serii temporale: sezonalitatea, tendința și zgomotul. Acești parametri sunt:

`p`: parametrul asociat aspectului autoregresiv al modelului, care încorporează valorile *anterioare*.
`d`: parametrul asociat părții integrate a modelului, care afectează cantitatea de *diferențiere* (🎓 amintește-ți diferențierea 👆?) aplicată unei serii temporale.
`q`: parametrul asociat părții de medie mobilă a modelului.

> Notă: Dacă datele tale au un aspect sezonier - ceea ce este cazul aici -, folosim un model ARIMA sezonier (SARIMA). În acest caz, trebuie să folosești un alt set de parametri: `P`, `D` și `Q`, care descriu aceleași asocieri ca `p`, `d` și `q`, dar corespund componentelor sezoniere ale modelului.

1. Începe prin a seta valoarea preferată pentru orizont. Să încercăm 3 ore:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Selectarea celor mai bune valori pentru parametrii unui model ARIMA poate fi provocatoare, deoarece este oarecum subiectivă și consumatoare de timp. Ai putea lua în considerare utilizarea unei funcții `auto_arima()` din biblioteca [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Deocamdată, încearcă câteva selecții manuale pentru a găsi un model bun.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Se afișează un tabel cu rezultatele.

Ai construit primul tău model! Acum trebuie să găsim o modalitate de a-l evalua.

### Evaluează modelul tău

Pentru a-ți evalua modelul, poți efectua așa-numita validare `walk forward`. În practică, modelele de serii temporale sunt re-antrenate de fiecare dată când devin disponibile date noi. Acest lucru permite modelului să facă cea mai bună prognoză la fiecare pas de timp.

Începând de la începutul seriei temporale, folosind această tehnică, antrenează modelul pe setul de date de antrenare. Apoi, fă o predicție pentru următorul pas de timp. Predicția este evaluată în raport cu valoarea cunoscută. Setul de antrenare este apoi extins pentru a include valoarea cunoscută, iar procesul se repetă.

> Notă: Ar trebui să menții fereastra setului de antrenare fixă pentru o antrenare mai eficientă, astfel încât de fiecare dată când adaugi o nouă observație la setul de antrenare, să elimini observația de la începutul setului.

Acest proces oferă o estimare mai robustă a modului în care modelul va performa în practică. Totuși, vine cu costul computațional de a crea atât de multe modele. Acest lucru este acceptabil dacă datele sunt mici sau dacă modelul este simplu, dar ar putea fi o problemă la scară mare.

Validarea walk-forward este standardul de aur pentru evaluarea modelelor de serii temporale și este recomandată pentru proiectele tale.

1. Mai întâi, creează un punct de date de testare pentru fiecare pas al ORIZONTULUI.

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

    Datele sunt deplasate orizontal în funcție de punctul lor de orizont.

1. Fă predicții pe datele tale de testare folosind această abordare cu fereastră glisantă într-un buclă de dimensiunea lungimii datelor de testare:

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

    Poți urmări procesul de antrenare:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Compară predicțiile cu sarcina reală:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Rezultate
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Observă predicția datelor orare, comparativ cu sarcina reală. Cât de precis este acest lucru?

### Verifică acuratețea modelului

Verifică acuratețea modelului tău testând eroarea procentuală medie absolută (MAPE) pe toate predicțiile.
> **🧮 Arată-mi matematica**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) este utilizat pentru a arăta acuratețea predicției ca un raport definit de formula de mai sus. Diferența dintre valoarea reală și cea prezisă este împărțită la valoarea reală. "Valoarea absolută în acest calcul este însumată pentru fiecare punct de prognoză în timp și împărțită la numărul de puncte ajustate n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Exprimă ecuația în cod:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Calculează MAPE pentru un pas:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE pentru prognoza unui pas:  0.5570581332313952 %

1. Afișează MAPE pentru prognoza multi-pas:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Un număr mic este ideal: consideră că o prognoză cu un MAPE de 10 este cu 10% în afara valorii reale.

1. Dar, ca întotdeauna, este mai ușor să vezi acest tip de măsurare a acurateței vizual, așa că hai să o reprezentăm grafic:

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

    ![un model de serie temporală](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Un grafic foarte reușit, care arată un model cu o acuratețe bună. Bravo!

---

## 🚀Provocare

Explorează metodele de testare a acurateței unui model de serie temporală. Am discutat despre MAPE în această lecție, dar există și alte metode pe care le-ai putea folosi? Cercetează-le și notează-le. Un document util poate fi găsit [aici](https://otexts.com/fpp2/accuracy.html)

## [Quiz după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Studiu Individual

Această lecție abordează doar elementele de bază ale prognozei seriilor temporale cu ARIMA. Ia-ți timp să îți aprofundezi cunoștințele explorând [acest depozit](https://microsoft.github.io/forecasting/) și diferitele tipuri de modele pentru a învăța alte modalități de a construi modele de serie temporală.

## Temă

[Un nou model ARIMA](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.