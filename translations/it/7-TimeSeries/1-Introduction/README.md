<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-06T07:28:48+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "it"
}
-->
# Introduzione alla previsione delle serie temporali

![Riepilogo delle serie temporali in uno sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

In questa lezione e nella successiva, imparerai qualcosa sulla previsione delle serie temporali, una parte interessante e preziosa del repertorio di uno scienziato di ML, che è un po' meno conosciuta rispetto ad altri argomenti. La previsione delle serie temporali è una sorta di "sfera di cristallo": basandosi sulle prestazioni passate di una variabile, come il prezzo, è possibile prevederne il potenziale valore futuro.

[![Introduzione alla previsione delle serie temporali](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introduzione alla previsione delle serie temporali")

> 🎥 Clicca sull'immagine sopra per un video sulla previsione delle serie temporali

## [Quiz preliminare alla lezione](https://ff-quizzes.netlify.app/en/ml/)

È un campo utile e interessante con un reale valore per il business, dato il suo diretto utilizzo per problemi di pricing, inventario e gestione della catena di approvvigionamento. Sebbene le tecniche di deep learning abbiano iniziato a essere utilizzate per ottenere maggiori approfondimenti e prevedere meglio le prestazioni future, la previsione delle serie temporali rimane un campo fortemente influenzato dalle tecniche classiche di ML.

> Il curriculum utile sulle serie temporali della Penn State è disponibile [qui](https://online.stat.psu.edu/stat510/lesson/1)

## Introduzione

Supponiamo che tu gestisca una rete di parchimetri intelligenti che forniscono dati su quanto spesso vengono utilizzati e per quanto tempo nel tempo.

> E se potessi prevedere, basandoti sulle prestazioni passate del parchimetro, il suo valore futuro secondo le leggi della domanda e dell'offerta?

Prevedere accuratamente quando agire per raggiungere il tuo obiettivo è una sfida che potrebbe essere affrontata con la previsione delle serie temporali. Non renderebbe felici le persone essere addebitate di più nei momenti di maggiore affluenza mentre cercano un parcheggio, ma sarebbe sicuramente un modo per generare entrate per pulire le strade!

Esploriamo alcuni tipi di algoritmi per le serie temporali e iniziamo un notebook per pulire e preparare alcuni dati. I dati che analizzerai provengono dalla competizione di previsione GEFCom2014. Consistono in 3 anni di valori orari di carico elettrico e temperatura tra il 2012 e il 2014. Dati i modelli storici di carico elettrico e temperatura, puoi prevedere i valori futuri del carico elettrico.

In questo esempio, imparerai a prevedere un passo temporale in avanti, utilizzando solo i dati storici del carico. Prima di iniziare, tuttavia, è utile capire cosa succede dietro le quinte.

## Alcune definizioni

Quando incontri il termine "serie temporali", devi comprenderne l'uso in diversi contesti.

🎓 **Serie temporali**

In matematica, "una serie temporale è una serie di punti dati indicizzati (o elencati o rappresentati graficamente) in ordine temporale. Più comunemente, una serie temporale è una sequenza presa a intervalli di tempo successivi e ugualmente distanziati." Un esempio di serie temporale è il valore di chiusura giornaliero del [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). L'uso di grafici delle serie temporali e della modellazione statistica è frequentemente incontrato nell'elaborazione dei segnali, nella previsione meteorologica, nella previsione dei terremoti e in altri campi in cui si verificano eventi e i punti dati possono essere tracciati nel tempo.

🎓 **Analisi delle serie temporali**

L'analisi delle serie temporali è l'analisi dei dati delle serie temporali sopra menzionati. I dati delle serie temporali possono assumere forme distinte, inclusi i "time series interrotti", che rilevano modelli nell'evoluzione di una serie temporale prima e dopo un evento di interruzione. Il tipo di analisi necessaria per la serie temporale dipende dalla natura dei dati. I dati delle serie temporali stessi possono assumere la forma di serie di numeri o caratteri.

L'analisi da eseguire utilizza una varietà di metodi, inclusi dominio delle frequenze e dominio del tempo, lineari e non lineari, e altro ancora. [Scopri di più](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) sui molti modi per analizzare questo tipo di dati.

🎓 **Previsione delle serie temporali**

La previsione delle serie temporali è l'uso di un modello per prevedere valori futuri basandosi sui modelli mostrati dai dati raccolti in passato. Sebbene sia possibile utilizzare modelli di regressione per esplorare i dati delle serie temporali, con indici temporali come variabili x su un grafico, tali dati sono meglio analizzati utilizzando tipi speciali di modelli.

I dati delle serie temporali sono un elenco di osservazioni ordinate, a differenza dei dati che possono essere analizzati tramite regressione lineare. Il più comune è ARIMA, un acronimo che sta per "Autoregressive Integrated Moving Average".

[I modelli ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "collegano il valore presente di una serie ai valori passati e agli errori di previsione passati." Sono più appropriati per analizzare i dati nel dominio del tempo, dove i dati sono ordinati nel tempo.

> Esistono diversi tipi di modelli ARIMA, che puoi imparare [qui](https://people.duke.edu/~rnau/411arim.htm) e che affronterai nella prossima lezione.

Nella prossima lezione, costruirai un modello ARIMA utilizzando [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), che si concentra su una variabile che cambia il suo valore nel tempo. Un esempio di questo tipo di dati è [questo dataset](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) che registra la concentrazione mensile di CO2 presso l'Osservatorio di Mauna Loa:

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

✅ Identifica la variabile che cambia nel tempo in questo dataset

## Caratteristiche dei dati delle serie temporali da considerare

Quando osservi i dati delle serie temporali, potresti notare che presentano [certe caratteristiche](https://online.stat.psu.edu/stat510/lesson/1/1.1) che devi prendere in considerazione e mitigare per comprendere meglio i loro modelli. Se consideri i dati delle serie temporali come un potenziale "segnale" che vuoi analizzare, queste caratteristiche possono essere considerate "rumore". Spesso sarà necessario ridurre questo "rumore" compensando alcune di queste caratteristiche utilizzando tecniche statistiche.

Ecco alcuni concetti che dovresti conoscere per lavorare con le serie temporali:

🎓 **Trend**

I trend sono definiti come aumenti e diminuzioni misurabili nel tempo. [Leggi di più](https://machinelearningmastery.com/time-series-trends-in-python). Nel contesto delle serie temporali, si tratta di come utilizzare e, se necessario, rimuovere i trend dalla tua serie temporale.

🎓 **[Stagionalità](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

La stagionalità è definita come fluttuazioni periodiche, come i picchi di vendita durante le festività, ad esempio. [Dai un'occhiata](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) a come diversi tipi di grafici mostrano la stagionalità nei dati.

🎓 **Outlier**

Gli outlier sono valori lontani dalla varianza standard dei dati.

🎓 **Ciclo a lungo termine**

Indipendentemente dalla stagionalità, i dati potrebbero mostrare un ciclo a lungo termine, come una recessione economica che dura più di un anno.

🎓 **Varianza costante**

Nel tempo, alcuni dati mostrano fluttuazioni costanti, come il consumo di energia tra giorno e notte.

🎓 **Cambiamenti improvvisi**

I dati potrebbero mostrare un cambiamento improvviso che potrebbe richiedere ulteriori analisi. La chiusura improvvisa delle attività a causa del COVID, ad esempio, ha causato cambiamenti nei dati.

✅ Ecco un [esempio di grafico delle serie temporali](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) che mostra la spesa giornaliera in valuta di gioco su alcuni anni. Riesci a identificare qualcuna delle caratteristiche sopra elencate in questi dati?

![Spesa in valuta di gioco](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Esercizio - iniziare con i dati sull'uso dell'energia

Iniziamo a creare un modello di serie temporali per prevedere il consumo futuro di energia dato il consumo passato.

> I dati in questo esempio provengono dalla competizione di previsione GEFCom2014. Consistono in 3 anni di valori orari di carico elettrico e temperatura tra il 2012 e il 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli e Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, luglio-settembre, 2016.

1. Nella cartella `working` di questa lezione, apri il file _notebook.ipynb_. Inizia aggiungendo le librerie che ti aiuteranno a caricare e visualizzare i dati:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Nota, stai utilizzando i file dalla cartella `common` inclusa, che configurano il tuo ambiente e gestiscono il download dei dati.

2. Successivamente, esamina i dati come dataframe chiamando `load_data()` e `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Puoi vedere che ci sono due colonne che rappresentano la data e il carico:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Ora, traccia i dati chiamando `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![grafico energia](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Ora, traccia la prima settimana di luglio 2014, fornendola come input a `energy` nel pattern `[from date]: [to date]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![luglio](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Un grafico bellissimo! Dai un'occhiata a questi grafici e vedi se riesci a determinare qualcuna delle caratteristiche sopra elencate. Cosa possiamo dedurre visualizzando i dati?

Nella prossima lezione, creerai un modello ARIMA per fare alcune previsioni.

---

## 🚀Sfida

Fai un elenco di tutte le industrie e aree di ricerca che ti vengono in mente che potrebbero beneficiare della previsione delle serie temporali. Riesci a pensare a un'applicazione di queste tecniche nelle arti? In econometria? In ecologia? Nel commercio al dettaglio? Nell'industria? Nella finanza? Dove altro?

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e studio autonomo

Sebbene non li affronteremo qui, le reti neurali sono talvolta utilizzate per migliorare i metodi classici di previsione delle serie temporali. Leggi di più su di esse [in questo articolo](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Compito

[Visualizza altre serie temporali](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.