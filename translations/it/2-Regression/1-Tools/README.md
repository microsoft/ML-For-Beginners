<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6b1cb0e46d4c5b747eff6e3607642760",
  "translation_date": "2025-08-29T20:32:19+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "it"
}
-->
# Inizia con Python e Scikit-learn per modelli di regressione

![Riepilogo delle regressioni in uno sketchnote](../../../../translated_images/ml-regression.4e4f70e3b3ed446e3ace348dec973e133fa5d3680fbc8412b61879507369b98d.it.png)

> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/9/)

> ### [Questa lezione è disponibile in R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introduzione

In queste quattro lezioni, scoprirai come costruire modelli di regressione. Discuteremo a cosa servono tra poco. Ma prima di fare qualsiasi cosa, assicurati di avere gli strumenti giusti per iniziare il processo!

In questa lezione, imparerai a:

- Configurare il tuo computer per attività di machine learning in locale.
- Lavorare con i notebook Jupyter.
- Usare Scikit-learn, inclusa l'installazione.
- Esplorare la regressione lineare con un esercizio pratico.

## Installazioni e configurazioni

[![ML per principianti - Configura i tuoi strumenti per costruire modelli di Machine Learning](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML per principianti - Configura i tuoi strumenti per costruire modelli di Machine Learning")

> 🎥 Clicca sull'immagine sopra per un breve video su come configurare il tuo computer per il ML.

1. **Installa Python**. Assicurati che [Python](https://www.python.org/downloads/) sia installato sul tuo computer. Userai Python per molte attività di data science e machine learning. La maggior parte dei sistemi operativi include già un'installazione di Python. Sono disponibili anche utili [Pacchetti di Codifica Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) per semplificare la configurazione per alcuni utenti.

   Tuttavia, alcuni utilizzi di Python richiedono una versione specifica del software, mentre altri ne richiedono una diversa. Per questo motivo, è utile lavorare all'interno di un [ambiente virtuale](https://docs.python.org/3/library/venv.html).

2. **Installa Visual Studio Code**. Assicurati di avere Visual Studio Code installato sul tuo computer. Segui queste istruzioni per [installare Visual Studio Code](https://code.visualstudio.com/) per l'installazione di base. Userai Python in Visual Studio Code in questo corso, quindi potresti voler ripassare come [configurare Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) per lo sviluppo in Python.

   > Familiarizza con Python lavorando attraverso questa raccolta di [moduli Learn](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configura Python con Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configura Python con Visual Studio Code")
   >
   > 🎥 Clicca sull'immagine sopra per un video: usare Python all'interno di VS Code.

3. **Installa Scikit-learn**, seguendo [queste istruzioni](https://scikit-learn.org/stable/install.html). Poiché è necessario utilizzare Python 3, si consiglia di utilizzare un ambiente virtuale. Nota che, se stai installando questa libreria su un Mac con chip M1, ci sono istruzioni speciali nella pagina sopra linkata.

4. **Installa Jupyter Notebook**. Dovrai [installare il pacchetto Jupyter](https://pypi.org/project/jupyter/).

## Il tuo ambiente di sviluppo ML

Userai **notebook** per sviluppare il tuo codice Python e creare modelli di machine learning. Questo tipo di file è uno strumento comune per i data scientist e può essere identificato dalla sua estensione `.ipynb`.

I notebook sono un ambiente interattivo che consente al programmatore di scrivere codice e aggiungere note e documentazione intorno al codice, il che è molto utile per progetti sperimentali o orientati alla ricerca.

[![ML per principianti - Configura Jupyter Notebook per iniziare a costruire modelli di regressione](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML per principianti - Configura Jupyter Notebook per iniziare a costruire modelli di regressione")

> 🎥 Clicca sull'immagine sopra per un breve video su questo esercizio.

### Esercizio - lavora con un notebook

In questa cartella, troverai il file _notebook.ipynb_.

1. Apri _notebook.ipynb_ in Visual Studio Code.

   Un server Jupyter si avvierà con Python 3+. Troverai aree del notebook che possono essere `eseguite`, ovvero blocchi di codice. Puoi eseguire un blocco di codice selezionando l'icona che sembra un pulsante di riproduzione.

2. Seleziona l'icona `md` e aggiungi un po' di markdown, con il seguente testo **# Benvenuto nel tuo notebook**.

   Successivamente, aggiungi del codice Python.

3. Scrivi **print('hello notebook')** nel blocco di codice.
4. Seleziona la freccia per eseguire il codice.

   Dovresti vedere la dichiarazione stampata:

    ```output
    hello notebook
    ```

![VS Code con un notebook aperto](../../../../translated_images/notebook.4a3ee31f396b88325607afda33cadcc6368de98040ff33942424260aa84d75f2.it.jpg)

Puoi alternare il tuo codice con commenti per auto-documentare il notebook.

✅ Pensa per un momento a quanto è diverso l'ambiente di lavoro di uno sviluppatore web rispetto a quello di un data scientist.

## Iniziare con Scikit-learn

Ora che Python è configurato nel tuo ambiente locale e ti senti a tuo agio con i notebook Jupyter, familiarizziamo con Scikit-learn (si pronuncia `sci` come in `scienza`). Scikit-learn fornisce un' [API estesa](https://scikit-learn.org/stable/modules/classes.html#api-ref) per aiutarti a svolgere attività di ML.

Secondo il loro [sito web](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn è una libreria open source di machine learning che supporta l'apprendimento supervisionato e non supervisionato. Fornisce anche vari strumenti per il fitting dei modelli, la pre-elaborazione dei dati, la selezione e la valutazione dei modelli, e molte altre utilità."

In questo corso, utilizzerai Scikit-learn e altri strumenti per costruire modelli di machine learning per svolgere quelle che chiamiamo attività di 'machine learning tradizionale'. Abbiamo deliberatamente evitato reti neurali e deep learning, poiché sono meglio trattati nel nostro prossimo curriculum 'AI for Beginners'.

Scikit-learn rende semplice costruire modelli e valutarli per l'uso. Si concentra principalmente sull'uso di dati numerici e contiene diversi dataset preconfezionati da utilizzare come strumenti di apprendimento. Include anche modelli predefiniti per gli studenti da provare. Esploriamo il processo di caricamento di dati preconfezionati e l'uso di un stimatore integrato per il primo modello ML con Scikit-learn con alcuni dati di base.

## Esercizio - il tuo primo notebook con Scikit-learn

> Questo tutorial è stato ispirato dall'[esempio di regressione lineare](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) sul sito web di Scikit-learn.

[![ML per principianti - Il tuo primo progetto di regressione lineare in Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML per principianti - Il tuo primo progetto di regressione lineare in Python")

> 🎥 Clicca sull'immagine sopra per un breve video su questo esercizio.

Nel file _notebook.ipynb_ associato a questa lezione, cancella tutte le celle premendo l'icona del 'cestino'.

In questa sezione, lavorerai con un piccolo dataset sul diabete integrato in Scikit-learn per scopi di apprendimento. Immagina di voler testare un trattamento per pazienti diabetici. I modelli di Machine Learning potrebbero aiutarti a determinare quali pazienti risponderebbero meglio al trattamento, in base a combinazioni di variabili. Anche un modello di regressione molto semplice, quando visualizzato, potrebbe mostrare informazioni sulle variabili che ti aiuterebbero a organizzare i tuoi ipotetici studi clinici.

✅ Esistono molti tipi di metodi di regressione, e quale scegliere dipende dalla risposta che stai cercando. Se vuoi prevedere l'altezza probabile di una persona di una certa età, useresti la regressione lineare, poiché stai cercando un **valore numerico**. Se invece vuoi scoprire se un tipo di cucina dovrebbe essere considerato vegano o meno, stai cercando un **assegnamento di categoria**, quindi useresti la regressione logistica. Immagina alcune domande che puoi porre ai dati e quale di questi metodi sarebbe più appropriato.

Iniziamo con questo compito.

### Importa le librerie

Per questo compito importeremo alcune librerie:

- **matplotlib**. È uno strumento utile per [creare grafici](https://matplotlib.org/) e lo useremo per creare un grafico a linee.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) è una libreria utile per gestire dati numerici in Python.
- **sklearn**. Questa è la libreria [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importa alcune librerie per aiutarti nei tuoi compiti.

1. Aggiungi gli import digitando il seguente codice:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Sopra stai importando `matplotlib`, `numpy` e stai importando `datasets`, `linear_model` e `model_selection` da `sklearn`. `model_selection` è usato per dividere i dati in set di addestramento e test.

### Il dataset sul diabete

Il [dataset sul diabete](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) integrato include 442 campioni di dati relativi al diabete, con 10 variabili caratteristiche, alcune delle quali includono:

- age: età in anni
- bmi: indice di massa corporea
- bp: pressione sanguigna media
- s1 tc: T-Cells (un tipo di globuli bianchi)

✅ Questo dataset include il concetto di 'sesso' come variabile caratteristica importante per la ricerca sul diabete. Molti dataset medici includono questo tipo di classificazione binaria. Rifletti su come categorizzazioni come questa potrebbero escludere alcune parti della popolazione dai trattamenti.

Ora, carica i dati X e y.

> 🎓 Ricorda, questo è apprendimento supervisionato, e abbiamo bisogno di un target 'y' nominato.

In una nuova cella di codice, carica il dataset sul diabete chiamando `load_diabetes()`. L'input `return_X_y=True` segnala che `X` sarà una matrice di dati e `y` sarà il target di regressione.

1. Aggiungi alcuni comandi print per mostrare la forma della matrice di dati e il suo primo elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Quello che ottieni come risposta è una tupla. Quello che stai facendo è assegnare i due primi valori della tupla rispettivamente a `X` e `y`. Scopri di più [sulle tuple](https://wikipedia.org/wiki/Tuple).

    Puoi vedere che questi dati hanno 442 elementi organizzati in array di 10 elementi:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Rifletti sulla relazione tra i dati e il target di regressione. La regressione lineare prevede relazioni tra la caratteristica X e la variabile target y. Riesci a trovare il [target](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) per il dataset sul diabete nella documentazione? Cosa sta dimostrando questo dataset, dato quel target?

2. Successivamente, seleziona una porzione di questo dataset da tracciare selezionando la terza colonna del dataset. Puoi farlo usando l'operatore `:` per selezionare tutte le righe, e poi selezionando la terza colonna usando l'indice (2). Puoi anche rimodellare i dati in un array 2D - come richiesto per il tracciamento - usando `reshape(n_rows, n_columns)`. Se uno dei parametri è -1, la dimensione corrispondente viene calcolata automaticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ In qualsiasi momento, stampa i dati per controllarne la forma.

3. Ora che hai i dati pronti per essere tracciati, puoi vedere se una macchina può aiutarti a determinare una divisione logica tra i numeri in questo dataset. Per fare ciò, devi dividere sia i dati (X) che il target (y) in set di test e di addestramento. Scikit-learn ha un modo semplice per farlo; puoi dividere i tuoi dati di test in un punto specifico.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Ora sei pronto per addestrare il tuo modello! Carica il modello di regressione lineare e addestralo con i tuoi set di addestramento X e y usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` è una funzione che vedrai in molte librerie di ML come TensorFlow.

5. Poi, crea una previsione usando i dati di test, utilizzando la funzione `predict()`. Questo sarà usato per tracciare la linea tra i gruppi di dati.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Ora è il momento di mostrare i dati in un grafico. Matplotlib è uno strumento molto utile per questo compito. Crea un grafico a dispersione di tutti i dati di test X e y, e usa la previsione per tracciare una linea nel punto più appropriato, tra i gruppi di dati del modello.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![un grafico a dispersione che mostra punti dati relativi al diabete](../../../../translated_images/scatterplot.ad8b356bcbb33be68d54050e09b9b7bfc03e94fde7371f2609ae43f4c563b2d7.it.png)
✅ Pensa un po' a cosa sta succedendo qui. Una linea retta attraversa molti piccoli punti di dati, ma cosa sta facendo esattamente? Riesci a capire come questa linea dovrebbe permetterti di prevedere dove un nuovo punto dati, mai visto prima, dovrebbe posizionarsi in relazione all'asse y del grafico? Prova a mettere in parole l'utilità pratica di questo modello.

Congratulazioni, hai costruito il tuo primo modello di regressione lineare, creato una previsione con esso e l'hai visualizzata in un grafico!

---
## 🚀Sfida

Traccia un grafico con una variabile diversa da questo dataset. Suggerimento: modifica questa riga: `X = X[:,2]`. Considerando il target di questo dataset, cosa riesci a scoprire sulla progressione del diabete come malattia?
## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/10/)

## Revisione & Studio Autonomo

In questo tutorial, hai lavorato con la regressione lineare semplice, piuttosto che con la regressione univariata o multipla. Leggi un po' sulle differenze tra questi metodi, oppure dai un'occhiata a [questo video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Leggi di più sul concetto di regressione e pensa a quali tipi di domande possono essere risolte con questa tecnica. Segui [questo tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) per approfondire la tua comprensione.

## Compito

[Un dataset diverso](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.