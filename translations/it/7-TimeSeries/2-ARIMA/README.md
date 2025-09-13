<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-06T07:28:07+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "it"
}
-->
# Previsione delle serie temporali con ARIMA

Nella lezione precedente, hai imparato qualcosa sulla previsione delle serie temporali e hai caricato un dataset che mostra le fluttuazioni del carico elettrico nel corso di un periodo di tempo.

[![Introduzione ad ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduzione ad ARIMA")

> 🎥 Clicca sull'immagine sopra per un video: Una breve introduzione ai modelli ARIMA. L'esempio è fatto in R, ma i concetti sono universali.

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Introduzione

In questa lezione, scoprirai un modo specifico per costruire modelli con [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). I modelli ARIMA sono particolarmente adatti per adattarsi ai dati che mostrano [non-stazionarietà](https://wikipedia.org/wiki/Stationary_process).

## Concetti generali

Per lavorare con ARIMA, ci sono alcuni concetti che devi conoscere:

- 🎓 **Stazionarietà**. Dal punto di vista statistico, la stazionarietà si riferisce a dati la cui distribuzione non cambia quando vengono spostati nel tempo. I dati non stazionari, invece, mostrano fluttuazioni dovute a tendenze che devono essere trasformate per essere analizzate. La stagionalità, ad esempio, può introdurre fluttuazioni nei dati e può essere eliminata attraverso un processo di 'differenziazione stagionale'.

- 🎓 **[Differenziazione](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. La differenziazione dei dati, sempre dal punto di vista statistico, si riferisce al processo di trasformazione dei dati non stazionari per renderli stazionari rimuovendo la loro tendenza non costante. "La differenziazione rimuove i cambiamenti nel livello di una serie temporale, eliminando tendenza e stagionalità e stabilizzando di conseguenza la media della serie temporale." [Articolo di Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA nel contesto delle serie temporali

Analizziamo le parti di ARIMA per comprendere meglio come ci aiuta a modellare le serie temporali e a fare previsioni su di esse.

- **AR - per AutoRegressivo**. I modelli autoregressivi, come suggerisce il nome, guardano 'indietro' nel tempo per analizzare i valori precedenti nei tuoi dati e fare ipotesi su di essi. Questi valori precedenti sono chiamati 'lag'. Un esempio potrebbe essere un dataset che mostra le vendite mensili di matite. Il totale delle vendite di ogni mese sarebbe considerato una 'variabile evolutiva' nel dataset. Questo modello è costruito come "la variabile evolutiva di interesse viene regressa sui suoi valori laggati (cioè, precedenti)." [Wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - per Integrato**. A differenza dei modelli simili 'ARMA', la 'I' in ARIMA si riferisce al suo aspetto *[integrato](https://wikipedia.org/wiki/Order_of_integration)*. I dati vengono 'integrati' quando vengono applicati passaggi di differenziazione per eliminare la non-stazionarietà.

- **MA - per Media Mobile**. L'aspetto [media mobile](https://wikipedia.org/wiki/Moving-average_model) di questo modello si riferisce alla variabile di output che viene determinata osservando i valori attuali e passati dei lag.

In sintesi: ARIMA viene utilizzato per adattare un modello alla forma speciale dei dati delle serie temporali nel modo più accurato possibile.

## Esercizio - costruire un modello ARIMA

Apri la cartella [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) in questa lezione e trova il file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Esegui il notebook per caricare la libreria Python `statsmodels`; ti servirà per i modelli ARIMA.

1. Carica le librerie necessarie.

1. Ora, carica altre librerie utili per la visualizzazione dei dati:

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

1. Carica i dati dal file `/data/energy.csv` in un dataframe Pandas e dai un'occhiata:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Traccia tutti i dati energetici disponibili da gennaio 2012 a dicembre 2014. Non ci dovrebbero essere sorprese, poiché abbiamo visto questi dati nella lezione precedente:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Ora, costruiamo un modello!

### Creare dataset di addestramento e test

Ora che i tuoi dati sono stati caricati, puoi separarli in set di addestramento e test. Addestrerai il tuo modello sul set di addestramento. Come di consueto, dopo che il modello ha terminato l'addestramento, valuterai la sua accuratezza utilizzando il set di test. Devi assicurarti che il set di test copra un periodo successivo rispetto al set di addestramento per garantire che il modello non ottenga informazioni da periodi futuri.

1. Assegna un periodo di due mesi dal 1° settembre al 31 ottobre 2014 al set di addestramento. Il set di test includerà il periodo di due mesi dal 1° novembre al 31 dicembre 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Poiché questi dati riflettono il consumo giornaliero di energia, c'è un forte pattern stagionale, ma il consumo è più simile al consumo dei giorni più recenti.

1. Visualizza le differenze:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![dati di addestramento e test](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Pertanto, utilizzare una finestra temporale relativamente piccola per addestrare i dati dovrebbe essere sufficiente.

    > Nota: Poiché la funzione che utilizziamo per adattare il modello ARIMA utilizza la validazione in-sample durante l'adattamento, ometteremo i dati di validazione.

### Preparare i dati per l'addestramento

Ora devi preparare i dati per l'addestramento eseguendo il filtraggio e la scalatura dei tuoi dati. Filtra il tuo dataset per includere solo i periodi di tempo e le colonne necessarie, e scala i dati per garantire che siano proiettati nell'intervallo 0,1.

1. Filtra il dataset originale per includere solo i periodi di tempo sopra menzionati per ogni set e includendo solo la colonna 'load' necessaria più la data:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Puoi vedere la forma dei dati:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Scala i dati per essere nell'intervallo (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualizza i dati originali rispetto a quelli scalati:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![originale](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > I dati originali

    ![scalato](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > I dati scalati

1. Ora che hai calibrato i dati scalati, puoi scalare i dati di test:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Implementare ARIMA

È il momento di implementare ARIMA! Ora utilizzerai la libreria `statsmodels` che hai installato in precedenza.

Ora devi seguire diversi passaggi:

   1. Definisci il modello chiamando `SARIMAX()` e passando i parametri del modello: parametri p, d e q, e parametri P, D e Q.
   2. Prepara il modello per i dati di addestramento chiamando la funzione fit().
   3. Fai previsioni chiamando la funzione `forecast()` e specificando il numero di passi (l'orizzonte) da prevedere.

> 🎓 A cosa servono tutti questi parametri? In un modello ARIMA ci sono 3 parametri che vengono utilizzati per aiutare a modellare gli aspetti principali di una serie temporale: stagionalità, tendenza e rumore. Questi parametri sono:

`p`: il parametro associato all'aspetto autoregressivo del modello, che incorpora i valori *passati*.
`d`: il parametro associato alla parte integrata del modello, che influenza la quantità di *differenziazione* (🎓 ricorda la differenziazione 👆?) da applicare a una serie temporale.
`q`: il parametro associato alla parte della media mobile del modello.

> Nota: Se i tuoi dati hanno un aspetto stagionale - come in questo caso -, utilizziamo un modello ARIMA stagionale (SARIMA). In tal caso, devi utilizzare un altro set di parametri: `P`, `D` e `Q` che descrivono le stesse associazioni di `p`, `d` e `q`, ma corrispondono ai componenti stagionali del modello.

1. Inizia impostando il valore preferito per l'orizzonte. Proviamo con 3 ore:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Selezionare i migliori valori per i parametri di un modello ARIMA può essere impegnativo, poiché è in parte soggettivo e richiede tempo. Potresti considerare di utilizzare una funzione `auto_arima()` dalla libreria [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Per ora prova alcune selezioni manuali per trovare un buon modello.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Viene stampata una tabella di risultati.

Hai costruito il tuo primo modello! Ora dobbiamo trovare un modo per valutarlo.

### Valutare il tuo modello

Per valutare il tuo modello, puoi eseguire la cosiddetta validazione `walk forward`. In pratica, i modelli di serie temporali vengono riaddestrati ogni volta che sono disponibili nuovi dati. Questo consente al modello di fare la migliore previsione a ogni passo temporale.

Partendo dall'inizio della serie temporale utilizzando questa tecnica, addestra il modello sul set di dati di addestramento. Quindi fai una previsione sul passo temporale successivo. La previsione viene valutata rispetto al valore noto. Il set di addestramento viene quindi ampliato per includere il valore noto e il processo viene ripetuto.

> Nota: Dovresti mantenere la finestra del set di addestramento fissa per un addestramento più efficiente, in modo che ogni volta che aggiungi una nuova osservazione al set di addestramento, rimuovi l'osservazione dall'inizio del set.

Questo processo fornisce una stima più robusta di come il modello si comporterà nella pratica. Tuttavia, comporta un costo computazionale per la creazione di così tanti modelli. Questo è accettabile se i dati sono piccoli o se il modello è semplice, ma potrebbe essere un problema su larga scala.

La validazione walk-forward è lo standard d'oro per la valutazione dei modelli di serie temporali ed è raccomandata per i tuoi progetti.

1. Per prima cosa, crea un punto dati di test per ogni passo HORIZON.

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

    I dati vengono spostati orizzontalmente in base al loro punto di orizzonte.

1. Fai previsioni sui tuoi dati di test utilizzando questo approccio a finestra scorrevole in un ciclo della lunghezza dei dati di test:

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

    Puoi osservare l'addestramento in corso:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Confronta le previsioni con il carico effettivo:

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

    Osserva la previsione dei dati orari rispetto al carico effettivo. Quanto è accurata?

### Controlla l'accuratezza del modello

Controlla l'accuratezza del tuo modello testando il suo errore percentuale assoluto medio (MAPE) su tutte le previsioni.
> **🧮 Mostrami la matematica**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) viene utilizzato per mostrare l'accuratezza delle previsioni come rapporto definito dalla formula sopra. La differenza tra il valore reale e quello previsto viene divisa per il valore reale. 
>
> "Il valore assoluto in questo calcolo viene sommato per ogni punto previsto nel tempo e diviso per il numero di punti adattati n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Espressione dell'equazione in codice:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Calcolo del MAPE per un singolo passo:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE per previsione a un passo:  0.5570581332313952 %

1. Stampa del MAPE per previsione multi-step:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Un numero basso è preferibile: considera che una previsione con un MAPE di 10 è errata del 10%.

1. Ma come sempre, è più facile visualizzare questo tipo di misurazione di accuratezza, quindi tracciamolo:

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

    ![un modello di serie temporale](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Un grafico molto bello, che mostra un modello con buona accuratezza. Ben fatto!

---

## 🚀Sfida

Esplora i modi per testare l'accuratezza di un modello di serie temporale. In questa lezione abbiamo accennato al MAPE, ma ci sono altri metodi che potresti utilizzare? Ricercali e annotali. Un documento utile può essere trovato [qui](https://otexts.com/fpp2/accuracy.html)

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione & Studio Autonomo

Questa lezione tratta solo le basi della previsione di serie temporali con ARIMA. Dedica del tempo ad approfondire la tua conoscenza esplorando [questo repository](https://microsoft.github.io/forecasting/) e i suoi vari tipi di modelli per imparare altri modi di costruire modelli di serie temporali.

## Compito

[Un nuovo modello ARIMA](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.