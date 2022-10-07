# Iniziare con Python e Scikit-learn per i modelli di regressione

![Sommario delle regressioni in uno sketchnote](../../../sketchnotes/ml-regression.png)

> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Qui Pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/9/?loc=it)

## Introduzione

In queste quattro lezioni, si scoprirà come costruire modelli di regressione. Si discuterà di cosa siano fra breve.
Prima di tutto, ci si deve assicurare di avere a disposizione gli strumenti adatti per far partire il processo!

In questa lezione, si imparerà come:

- Configurare il proprio computer per attività locali di machine learning.
- Lavorare con i Jupyter notebook.
- Usare Scikit-learn, compresa l'installazione.
- Esplorare la regressione lineare con un esercizio pratico.

## Installazioni e configurazioni

[![Usare Python con Visual Studio Code](https://img.youtube.com/vi/7EXd4_ttIuw/0.jpg)](https://youtu.be/7EXd4_ttIuw "Using Python with Visual Studio Code")

> 🎥 Fare click sull'immagine qui sopra per un video: usare Python all'interno di VS Code.

1. **Installare Python**. Assicurarsi che [Python](https://www.python.org/downloads/) sia installato nel proprio computer. Si userà Python for per molte attività di data science e machine learning. La maggior parte dei sistemi già include una installazione di Python. Ci sono anche utili [Pacchetti di Codice Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) disponbili, per facilitare l'installazione per alcuni utenti.

   Alcuni utilizzi di Python, tuttavia, richiedono una versione del software, laddove altri ne richiedono un'altra differente. Per questa ragione, è utile lavorare con un [ambiente virtuale](https://docs.python.org/3/library/venv.html).

2. **Installare Visual Studio Code**. Assicurarsi di avere installato Visual Studio Code sul proprio computer. Si seguano queste istruzioni per [installare Visual Studio Code](https://code.visualstudio.com/) per l'installazione basica. Si userà Python in Visual Studio Code in questo corso, quindi meglio rinfrescarsi le idee su come [configurare Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) per lo sviluppo in Python.

   > Si prenda confidenza con Python tramite questa collezione di [moduli di apprendimento](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)

3. **Installare Scikit-learn**, seguendo [queste istruzioni](https://scikit-learn.org/stable/install.html). Visto che ci si deve assicurare di usare Python 3, ci si raccomanda di usare un ambiente virtuale. Si noti che se si installa questa libreria in un M1 Mac, ci sono istruzioni speciali nella pagina di cui al riferimento qui sopra.

1. **Installare Jupyter Notebook**. Servirà [installare il pacchetto Jupyter](https://pypi.org/project/jupyter/). 

## Ambiente di creazione ML

Si useranno **notebook** per sviluppare il codice Python e creare modelli di machine learning. Questo tipo di file è uno strumento comune per i data scientist, e viene identificato dal suffisso o estensione `.ipynb`.

I notebook sono un ambiente interattivo che consente allo sviluppatore di scrivere codice, aggiungere note e scrivere documentazione attorno al codice il che è particolarmente utile per progetti sperimentali o orientati alla ricerca.

### Esercizio - lavorare con un notebook

In questa cartella, si troverà il file _notebook.ipynb_. 

1. Aprire _notebook.ipynb_ in Visual Studio Code.

   Un server Jupyter verrà lanciato con Python 3+. Si troveranno aree del notebook che possono essere `eseguite`, pezzi di codice. Si può eseguire un blocco di codice selezionando l'icona che assomiglia a un bottone di riproduzione.

1. Selezionare l'icona `md` e aggiungere un  poco di markdown, e il seguente testo **# Benvenuto nel tuo notebook**.

   Poi, aggiungere un blocco di codice Python. 

1. Digitare **print('hello notebook')** nell'area riservata al codice.
1. Selezionare la freccia per eseguire il codice.

   Si dovrebbe vedere stampata la seguente frase:

    ```output
    hello notebook
    ```

![VS Code con un notebook aperto](../images/notebook.jpg)

Si può inframezzare il codice con commenti per auto documentare il notebook.

✅ Si pensi per un minuto all'ambiente di lavoro di uno sviluppatore web rispetto a quello di un data scientist.

## Scikit-learn installato e funzionante

Adesso che Python è impostato nel proprio ambiente locale, e si è familiari con i notebook Jupyter, si acquisterà ora confidenza con Scikit-learn (si pronuncia con la `si` della parola inglese `science`). Scikit-learn fornisce una [API estensiva](https://scikit-learn.org/stable/modules/classes.html#api-ref) che aiuta a eseguire attività ML.

Stando al loro [sito web](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn è una libreria di machine learning open source che supporta l'apprendimento assistito (supervised learning) e non assistito (unsuperivised learnin). Fornisce anche strumenti vari per l'adattamento del modello, la pre-elaborazione dei dati, la selezione e la valutazione dei modelli e molte altre utilità."

In questo corso, si userà Scikit-learn e altri strumenti per costruire modelli di machine learning per eseguire quelle che vengono chiamate attività di 'machine learning tradizionale'. Si sono deliberamente evitate le reti neurali e il deep learning visto che saranno meglio trattati nel prossimo programma di studi 'AI per Principianti'.

Scikit-learn rende semplice costruire modelli e valutarli per l'uso. Si concentra principalmente sull'utilizzo di dati numerici e contiene diversi insiemi di dati già pronti per l'uso come strumenti di apprendimento. Include anche modelli pre-costruiti per gli studenti da provare. Si esplora ora il processo di caricamento dei dati preconfezionati, e, utilizzando un modello di stimatore incorporato, un primo modello ML con Scikit-Learn con alcuni dati di base.

## Esercizio - Il Primo notebook Scikit-learn

> Questo tutorial è stato ispirato dall'[esempio di regressione lineare](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) nel sito web di Scikit-learn.

Nel file _notebook.ipynb_ associato a questa lezione, svuotare tutte le celle usando l'icona cestino ('trash can').

In questa sezione, di lavorerà con un piccolo insieme di dati sul diabete che è incorporato in Scikit-learn per scopi di apprendimento. Si immagini di voler testare un trattamento per i pazienti diabetici. I modelli di machine learning potrebbero essere di aiuto nel determinare quali pazienti risponderebbero meglio al trattamento, in base a combinazioni di variabili. Anche un modello di regressione molto semplice, quando visualizzato, potrebbe mostrare informazioni sulle variabili che aiuteranno a organizzare le sperimentazioni cliniche teoriche.

✅ Esistono molti tipi di metodi di regressione e quale scegliere dipende dalla risposta che si sta cercando. Se si vuole prevedere l'altezza probabile per una persona di una data età, si dovrebbe usare la regressione lineare, visto che si sta cercando un **valore numerico**. Se si è interessati a scoprire se un tipo di cucina dovrebbe essere considerato vegano o no, si sta cercando un'**assegnazione di categoria** quindi si dovrebbe usare la regressione logistica. Si imparerà di più sulla regressione logistica in seguito. Si pensi ad alcune domande che si possono chiedere ai dati e quale di questi metodi sarebbe più appropriato.

Si inizia con questa attività.

### Importare le librerie

Per questo compito verranno importate alcune librerie:

- **matplotlib**. E' un utile [strumento grafico](https://matplotlib.org/) e verrà usato per creare una trama a linee.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) è una libreira utile per gestire i dati numerici in Python.
- **sklearn**. Questa è la libreria Scikit-learn.

Importare alcune librerie che saranno di aiuto per le proprie attività.

1. Con il seguente codice si aggiungono le importazioni:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Qui sopra vengono importati `matplottlib`, e `numpy`, da `sklearn` si importa `datasets`, `linear_model` e `model_selection`. `model_selection` viene usato per dividere i dati negli insiemi di addestramento e test.

### L'insieme di dati riguardante il diabete

L'[insieme dei dati sul diabete](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) include 442 campioni di dati sul diabete, con 10 variabili caratteristiche, alcune delle quali includono:

- age (età): età in anni
- bmi: indice di massa corporea (body mass index)
- bp: media pressione sanguinea
- s1 tc: Cellule T (un tipo di leucocito)

✅ Questo insieme di dati include il concetto di "sesso" come caratteristica variabile importante per la ricerca sul diabete. Molti insiemi di dati medici includono questo tipo di classificazione binaria. Si rifletta su come categorizzazioni come questa potrebbe escludere alcune parti di una popolazione dai trattamenti.

Ora si caricano i dati di X e y.

> 🎓 Si ricordi, questo è apprendimento supervisionato (supervised learning), e serve dare un nome all'obiettivo 'y'.

In una nuova cella di codice, caricare l'insieme di dati sul diabete chiamando `load_diabetes()`. Il parametro `return_X_y=True` segnala che `X` sarà una matrice di dati e `y` sarà l'obiettivo della regressione. 

1. Si aggiungono alcuni comandi di stampa per msotrare la forma della matrice di dati e i suoi primi elementi:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Quella che viene ritornata è una tuple. Quello che si sta facento è assegnare i primi due valori della tupla a  `X` e `y` rispettivamente. Per saperne di più sulle [tuples](https://wikipedia.org/wiki/Tuple).

    Si può vedere che questi dati hanno 442 elementi divisi in array di 10 elementi:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Si rifletta sulla relazione tra i dati e l'obiettivo di regressione. La regressione lineare prevede le relazioni tra la caratteristica X e la variabile di destinazione y. Si può trovare l'[obiettivo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) per l'insieme di dati sul diabete nella documentazione? Cosa dimostra questo insieme di dati, dato quell'obiettivo?

2. Successivamente, selezionare una porzione di questo insieme di dati da tracciare sistemandola in un nuovo array usando la funzione di numpy's `newaxis`. Verrà usata la regressione lineare per generare una linea tra i valori in questi dati secondo il modello che determina.

   ```python
   X = X[:, np.newaxis, 2]
   ```

   ✅ A piacere, stampare i dati per verificarne la forma.

3. Ora che si hanno dei dati pronti per essere tracciati, è possibile vedere se una macchina può aiutare a determinare una divisione logica tra i numeri in questo insieme di dati. Per fare ciò, è necessario dividere sia i dati (X) che l'obiettivo (y) in insiemi di test e addestamento. Scikit-learn ha un modo semplice per farlo; si possono dividere i  dati di prova in un determinato punto.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Ora si è pronti ad addestare il modello! Caricare il modello di regressione lineare e addestrarlo con i propri insiemi di addestramento X e y usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` è una funzione che si vedrà in molte librerie ML tipo TensorFlow

5. Successivamente creare una previsione usando i dati di test, con la funzione `predict()`. Questo servirà per tracciare la linea tra i gruppi di dati

    ```python
    y_pred = model.predict(X_test)
    ```

6. Ora è il momento di mostrare i dati in un tracciato. Matplotlib è uno strumento molto utile per questo compito. Si crei un grafico a dispersione (scatterplot) di tutti i dati del test X e y e si utilizzi la previsione per disegnare una linea nel luogo più appropriato, tra i raggruppamenti dei dati del modello.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.show()
    ```

   ![un grafico a dispersione che mostra i punti dati sul diabete](../images/scatterplot.png)

   ✅ Si pensi a cosa sta succedendo qui. Una linea retta scorre attraverso molti piccoli punti dati, ma cosa sta facendo esattamente? Si può capire come si dovrebbe utilizzare questa linea per prevedere dove un nuovo punto di dati non noto dovrebbe adattarsi alla relazione con l'asse y del tracciato? Si cerchi di mettere in parole l'uso pratico di questo modello.

Congratulazioni, si è costruito il primo modello di regressione lineare, creato una previsione con esso, e visualizzata in una tracciato!

---

## 🚀Sfida

Tracciare una variabile diversa da questo insieme di dati. Suggerimento: modificare questa riga: `X = X[:, np.newaxis, 2]`. Dato l'obiettivo di questo insieme di dati, cosa si potrebbe riuscire a scoprire circa la progressione del diabete come matattia?

## [Qui post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/10/?loc=it)

## Riepilogo e Auto Apprendimento

In questo tutorial, si è lavorato con una semplice regressione lineare, piuttosto che una regressione univariata o multipla. Ci so informi circa le differenze tra questi metodi oppure si dia uno sguardo a [questo video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Si legga di più sul concetto di regressione e si pensi a quale tipo di domande potrebbero trovare risposta con questa tecnica. Seguire questo [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) per approfondire la propria conoscenza.

## Compito

[Un insieme di dati diverso](assignment.it.md)

