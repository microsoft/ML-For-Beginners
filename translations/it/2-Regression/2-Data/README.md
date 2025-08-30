<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a683e1fe430bb0d4a10b68f6ca15e0a6",
  "translation_date": "2025-08-29T20:36:35+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "it"
}
-->
# Costruire un modello di regressione con Scikit-learn: preparare e visualizzare i dati

![Infografica sulla visualizzazione dei dati](../../../../translated_images/data-visualization.54e56dded7c1a804d00d027543f2881cb32da73aeadda2d4a4f10f3497526114.it.png)

Infografica di [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/11/)

> ### [Questa lezione è disponibile in R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introduzione

Ora che hai configurato gli strumenti necessari per iniziare a costruire modelli di machine learning con Scikit-learn, sei pronto per iniziare a porre domande ai tuoi dati. Quando lavori con i dati e applichi soluzioni di ML, è molto importante sapere come formulare la domanda giusta per sfruttare al meglio il potenziale del tuo dataset.

In questa lezione imparerai:

- Come preparare i dati per la costruzione di un modello.
- Come utilizzare Matplotlib per la visualizzazione dei dati.

## Porre la domanda giusta ai tuoi dati

La domanda a cui vuoi rispondere determinerà il tipo di algoritmi di ML che utilizzerai. E la qualità della risposta che otterrai dipenderà fortemente dalla natura dei tuoi dati.

Dai un'occhiata ai [dati](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) forniti per questa lezione. Puoi aprire questo file .csv in VS Code. Una rapida occhiata mostra immediatamente che ci sono spazi vuoti e un mix di stringhe e dati numerici. C'è anche una colonna strana chiamata 'Package' dove i dati sono un mix tra 'sacks', 'bins' e altri valori. I dati, in effetti, sono un po' disordinati.

[![ML per principianti - Come analizzare e pulire un dataset](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML per principianti - Come analizzare e pulire un dataset")

> 🎥 Clicca sull'immagine sopra per un breve video che mostra come preparare i dati per questa lezione.

In effetti, non è molto comune ricevere un dataset completamente pronto per essere utilizzato per creare un modello di ML. In questa lezione, imparerai come preparare un dataset grezzo utilizzando librerie standard di Python. Imparerai anche varie tecniche per visualizzare i dati.

## Caso di studio: 'il mercato delle zucche'

In questa cartella troverai un file .csv nella cartella principale `data` chiamato [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) che include 1757 righe di dati sul mercato delle zucche, raggruppati per città. Questi sono dati grezzi estratti dai [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuiti dal Dipartimento dell'Agricoltura degli Stati Uniti.

### Preparare i dati

Questi dati sono di dominio pubblico. Possono essere scaricati in molti file separati, uno per città, dal sito web dell'USDA. Per evitare troppi file separati, abbiamo concatenato tutti i dati delle città in un unico foglio di calcolo, quindi abbiamo già _preparato_ un po' i dati. Ora, diamo un'occhiata più da vicino ai dati.

### I dati sulle zucche - prime conclusioni

Cosa noti su questi dati? Hai già visto che c'è un mix di stringhe, numeri, spazi vuoti e valori strani che devi interpretare.

Quale domanda puoi porre a questi dati utilizzando una tecnica di regressione? Che ne dici di "Prevedere il prezzo di una zucca in vendita durante un determinato mese". Guardando di nuovo i dati, ci sono alcune modifiche che devi fare per creare la struttura dei dati necessaria per questo compito.

## Esercizio - analizzare i dati sulle zucche

Utilizziamo [Pandas](https://pandas.pydata.org/), (il nome sta per `Python Data Analysis`) uno strumento molto utile per modellare i dati, per analizzare e preparare questi dati sulle zucche.

### Primo, controlla le date mancanti

Per prima cosa, dovrai controllare se ci sono date mancanti:

1. Converti le date in un formato mensile (queste sono date statunitensi, quindi il formato è `MM/DD/YYYY`).
2. Estrai il mese in una nuova colonna.

Apri il file _notebook.ipynb_ in Visual Studio Code e importa il foglio di calcolo in un nuovo dataframe Pandas.

1. Usa la funzione `head()` per visualizzare le prime cinque righe.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Quale funzione useresti per visualizzare le ultime cinque righe?

1. Controlla se ci sono dati mancanti nel dataframe attuale:

    ```python
    pumpkins.isnull().sum()
    ```

    Ci sono dati mancanti, ma forse non saranno rilevanti per il compito da svolgere.

1. Per rendere il tuo dataframe più facile da gestire, seleziona solo le colonne necessarie utilizzando la funzione `loc`, che estrae dal dataframe originale un gruppo di righe (passate come primo parametro) e colonne (passate come secondo parametro). L'espressione `:` nel caso seguente significa "tutte le righe".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Secondo, determina il prezzo medio delle zucche

Pensa a come determinare il prezzo medio di una zucca in un determinato mese. Quali colonne sceglieresti per questo compito? Suggerimento: avrai bisogno di 3 colonne.

Soluzione: calcola la media delle colonne `Low Price` e `High Price` per popolare la nuova colonna Price, e converti la colonna Date per mostrare solo il mese. Fortunatamente, secondo il controllo precedente, non ci sono dati mancanti per date o prezzi.

1. Per calcolare la media, aggiungi il seguente codice:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Sentiti libero di stampare qualsiasi dato che desideri controllare utilizzando `print(month)`.

2. Ora, copia i tuoi dati convertiti in un nuovo dataframe Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Stampando il tuo dataframe, vedrai un dataset pulito e ordinato su cui puoi costruire il tuo nuovo modello di regressione.

### Ma aspetta! C'è qualcosa di strano qui

Se guardi la colonna `Package`, le zucche sono vendute in molte configurazioni diverse. Alcune sono vendute in misure di '1 1/9 bushel', altre in '1/2 bushel', alcune per zucca, altre per libbra, e altre ancora in grandi scatole di larghezze variabili.

> Le zucche sembrano molto difficili da pesare in modo coerente

Esaminando i dati originali, è interessante notare che tutto ciò che ha `Unit of Sale` uguale a 'EACH' o 'PER BIN' ha anche il tipo `Package` per pollice, per bin, o 'each'. Le zucche sembrano essere molto difficili da pesare in modo coerente, quindi filtriamole selezionando solo le zucche con la stringa 'bushel' nella colonna `Package`.

1. Aggiungi un filtro all'inizio del file, sotto l'importazione iniziale del .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Se stampi i dati ora, puoi vedere che stai ottenendo solo le circa 415 righe di dati contenenti zucche per bushel.

### Ma aspetta! C'è ancora una cosa da fare

Hai notato che la quantità di bushel varia per riga? Devi normalizzare i prezzi in modo da mostrare il prezzo per bushel, quindi fai qualche calcolo per standardizzarlo.

1. Aggiungi queste righe dopo il blocco che crea il dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Secondo [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), il peso di un bushel dipende dal tipo di prodotto, poiché è una misura di volume. "Un bushel di pomodori, ad esempio, dovrebbe pesare 56 libbre... Foglie e verdure occupano più spazio con meno peso, quindi un bushel di spinaci pesa solo 20 libbre." È tutto piuttosto complicato! Non preoccupiamoci di fare una conversione bushel-libbra, e invece calcoliamo il prezzo per bushel. Tutto questo studio sui bushel di zucche, tuttavia, dimostra quanto sia importante comprendere la natura dei tuoi dati!

Ora puoi analizzare i prezzi per unità in base alla loro misura in bushel. Se stampi i dati un'altra volta, puoi vedere come sono stati standardizzati.

✅ Hai notato che le zucche vendute a mezzo bushel sono molto costose? Riesci a capire perché? Suggerimento: le zucche piccole sono molto più costose di quelle grandi, probabilmente perché ce ne sono molte di più per bushel, dato lo spazio inutilizzato occupato da una grande zucca vuota per torte.

## Strategie di visualizzazione

Parte del ruolo del data scientist è dimostrare la qualità e la natura dei dati con cui sta lavorando. Per fare ciò, spesso creano visualizzazioni interessanti, come grafici, diagrammi e tabelle, che mostrano diversi aspetti dei dati. In questo modo, possono mostrare visivamente relazioni e lacune che altrimenti sarebbero difficili da individuare.

[![ML per principianti - Come visualizzare i dati con Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML per principianti - Come visualizzare i dati con Matplotlib")

> 🎥 Clicca sull'immagine sopra per un breve video che mostra come visualizzare i dati per questa lezione.

Le visualizzazioni possono anche aiutare a determinare la tecnica di machine learning più appropriata per i dati. Un diagramma a dispersione che sembra seguire una linea, ad esempio, indica che i dati sono un buon candidato per un esercizio di regressione lineare.

Una libreria di visualizzazione dei dati che funziona bene nei notebook Jupyter è [Matplotlib](https://matplotlib.org/) (che hai visto anche nella lezione precedente).

> Ottieni più esperienza con la visualizzazione dei dati in [questi tutorial](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Esercizio - sperimenta con Matplotlib

Prova a creare alcuni grafici di base per visualizzare il nuovo dataframe che hai appena creato. Cosa mostrerebbe un grafico a linee di base?

1. Importa Matplotlib all'inizio del file, sotto l'importazione di Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Esegui nuovamente l'intero notebook per aggiornare.
1. Alla fine del notebook, aggiungi una cella per tracciare i dati come un boxplot:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Un diagramma a dispersione che mostra la relazione tra prezzo e mese](../../../../translated_images/scatterplot.b6868f44cbd2051c6680ccdbb1510697d06a3ff6cd4abda656f5009c0ed4e3fc.it.png)

    Questo grafico è utile? Ti sorprende qualcosa?

    Non è particolarmente utile, poiché tutto ciò che fa è mostrare i tuoi dati come una distribuzione di punti in un determinato mese.

### Rendilo utile

Per ottenere grafici che mostrino dati utili, di solito è necessario raggruppare i dati in qualche modo. Proviamo a creare un grafico in cui l'asse y mostra i mesi e i dati dimostrano la distribuzione dei dati.

1. Aggiungi una cella per creare un grafico a barre raggruppato:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Un grafico a barre che mostra la relazione tra prezzo e mese](../../../../translated_images/barchart.a833ea9194346d769c77a3a870f7d8aee51574cd1138ca902e5500830a41cbce.it.png)

    Questo è una visualizzazione dei dati più utile! Sembra indicare che il prezzo più alto per le zucche si verifica a settembre e ottobre. Questo corrisponde alle tue aspettative? Perché o perché no?

---

## 🚀Sfida

Esplora i diversi tipi di visualizzazioni che Matplotlib offre. Quali tipi sono più appropriati per i problemi di regressione?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/12/)

## Revisione e studio autonomo

Dai un'occhiata ai molti modi per visualizzare i dati. Fai un elenco delle varie librerie disponibili e annota quali sono le migliori per determinati tipi di compiti, ad esempio visualizzazioni 2D rispetto a visualizzazioni 3D. Cosa scopri?

## Compito

[Esplorare la visualizzazione](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche potrebbero contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale eseguita da un traduttore umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.