<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-06T07:33:25+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "it"
}
-->
# Tecniche di Machine Learning

Il processo di costruzione, utilizzo e manutenzione dei modelli di machine learning e dei dati che utilizzano è molto diverso rispetto a molti altri flussi di lavoro di sviluppo. In questa lezione, demistificheremo il processo e delineeremo le principali tecniche che devi conoscere. Imparerai a:

- Comprendere i processi alla base del machine learning a un livello generale.
- Esplorare concetti di base come "modelli", "previsioni" e "dati di addestramento".

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

[![ML per principianti - Tecniche di Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML per principianti - Tecniche di Machine Learning")

> 🎥 Clicca sull'immagine sopra per un breve video che illustra questa lezione.

## Introduzione

A un livello generale, l'arte di creare processi di machine learning (ML) si compone di diversi passaggi:

1. **Decidere la domanda**. La maggior parte dei processi di ML inizia ponendo una domanda che non può essere risolta con un semplice programma condizionale o un motore basato su regole. Queste domande spesso riguardano previsioni basate su una raccolta di dati.
2. **Raccogliere e preparare i dati**. Per rispondere alla tua domanda, hai bisogno di dati. La qualità e, a volte, la quantità dei tuoi dati determineranno quanto bene potrai rispondere alla domanda iniziale. La visualizzazione dei dati è un aspetto importante di questa fase. Questa fase include anche la suddivisione dei dati in un gruppo di addestramento e uno di test per costruire un modello.
3. **Scegliere un metodo di addestramento**. A seconda della tua domanda e della natura dei tuoi dati, devi scegliere come addestrare un modello per riflettere al meglio i dati e fare previsioni accurate. Questa parte del processo di ML richiede competenze specifiche e, spesso, una notevole quantità di sperimentazione.
4. **Addestrare il modello**. Utilizzando i tuoi dati di addestramento, userai vari algoritmi per addestrare un modello a riconoscere schemi nei dati. Il modello potrebbe utilizzare pesi interni che possono essere regolati per privilegiare alcune parti dei dati rispetto ad altre, al fine di costruire un modello migliore.
5. **Valutare il modello**. Utilizzi dati mai visti prima (i tuoi dati di test) dal set raccolto per verificare le prestazioni del modello.
6. **Ottimizzazione dei parametri**. In base alle prestazioni del modello, puoi ripetere il processo utilizzando parametri o variabili diversi che controllano il comportamento degli algoritmi utilizzati per addestrare il modello.
7. **Prevedere**. Usa nuovi input per testare l'accuratezza del modello.

## Quale domanda porre

I computer sono particolarmente abili nel scoprire schemi nascosti nei dati. Questa capacità è molto utile per i ricercatori che hanno domande su un determinato dominio che non possono essere facilmente risolte creando un motore basato su regole condizionali. Dato un compito attuariale, ad esempio, un data scientist potrebbe essere in grado di costruire regole artigianali sulla mortalità dei fumatori rispetto ai non fumatori.

Quando però vengono introdotte molte altre variabili, un modello di ML potrebbe dimostrarsi più efficiente nel prevedere i tassi di mortalità futuri basandosi sulla storia sanitaria passata. Un esempio più allegro potrebbe essere fare previsioni meteorologiche per il mese di aprile in una determinata località basandosi su dati che includono latitudine, longitudine, cambiamenti climatici, vicinanza all'oceano, schemi delle correnti a getto e altro.

✅ Questa [presentazione](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sui modelli meteorologici offre una prospettiva storica sull'uso del ML nell'analisi meteorologica.  

## Attività preliminari alla costruzione

Prima di iniziare a costruire il tuo modello, ci sono diverse attività che devi completare. Per testare la tua domanda e formulare un'ipotesi basata sulle previsioni di un modello, devi identificare e configurare diversi elementi.

### Dati

Per rispondere alla tua domanda con un certo grado di certezza, hai bisogno di una buona quantità di dati del tipo giusto. Ci sono due cose che devi fare a questo punto:

- **Raccogliere i dati**. Tenendo a mente la lezione precedente sull'equità nell'analisi dei dati, raccogli i tuoi dati con attenzione. Sii consapevole delle fonti di questi dati, di eventuali bias intrinseci e documenta la loro origine.
- **Preparare i dati**. Ci sono diversi passaggi nel processo di preparazione dei dati. Potresti dover unire i dati e normalizzarli se provengono da fonti diverse. Puoi migliorare la qualità e la quantità dei dati attraverso vari metodi, come convertire stringhe in numeri (come facciamo in [Clustering](../../5-Clustering/1-Visualize/README.md)). Potresti anche generare nuovi dati basandoti sugli originali (come facciamo in [Classificazione](../../4-Classification/1-Introduction/README.md)). Puoi pulire e modificare i dati (come faremo prima della lezione sull'[App Web](../../3-Web-App/README.md)). Infine, potresti anche doverli randomizzare e mescolare, a seconda delle tecniche di addestramento.

✅ Dopo aver raccolto e processato i tuoi dati, prenditi un momento per verificare se la loro struttura ti permetterà di affrontare la domanda che intendi porre. Potrebbe essere che i dati non funzionino bene per il tuo compito, come scopriamo nelle lezioni su [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Caratteristiche e Target

Una [caratteristica](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) è una proprietà misurabile dei tuoi dati. In molti dataset è espressa come intestazione di colonna, ad esempio 'data', 'dimensione' o 'colore'. La tua variabile caratteristica, solitamente rappresentata come `X` nel codice, rappresenta la variabile di input che verrà utilizzata per addestrare il modello.

Un target è ciò che stai cercando di prevedere. Il target, solitamente rappresentato come `y` nel codice, rappresenta la risposta alla domanda che stai cercando di porre ai tuoi dati: a dicembre, quale **colore** di zucche sarà il più economico? A San Francisco, quali quartieri avranno il miglior **prezzo** immobiliare? A volte il target è anche chiamato attributo etichetta.

### Selezionare la variabile caratteristica

🎓 **Selezione delle caratteristiche ed Estrazione delle caratteristiche** Come fai a sapere quale variabile scegliere quando costruisci un modello? Probabilmente attraverserai un processo di selezione delle caratteristiche o estrazione delle caratteristiche per scegliere le variabili giuste per il modello più performante. Tuttavia, non sono la stessa cosa: "L'estrazione delle caratteristiche crea nuove caratteristiche da funzioni delle caratteristiche originali, mentre la selezione delle caratteristiche restituisce un sottoinsieme delle caratteristiche." ([fonte](https://wikipedia.org/wiki/Feature_selection))

### Visualizzare i dati

Un aspetto importante della cassetta degli attrezzi del data scientist è la capacità di visualizzare i dati utilizzando diverse librerie eccellenti come Seaborn o MatPlotLib. Rappresentare i dati visivamente potrebbe permetterti di scoprire correlazioni nascoste che puoi sfruttare. Le tue visualizzazioni potrebbero anche aiutarti a scoprire bias o dati sbilanciati (come scopriamo in [Classificazione](../../4-Classification/2-Classifiers-1/README.md)).

### Suddividere il dataset

Prima dell'addestramento, devi suddividere il tuo dataset in due o più parti di dimensioni disuguali che rappresentino comunque bene i dati.

- **Addestramento**. Questa parte del dataset viene adattata al tuo modello per addestrarlo. Questo set costituisce la maggior parte del dataset originale.
- **Test**. Un dataset di test è un gruppo indipendente di dati, spesso raccolto dai dati originali, che utilizzi per confermare le prestazioni del modello costruito.
- **Validazione**. Un set di validazione è un gruppo indipendente più piccolo di esempi che utilizzi per ottimizzare i parametri del modello o la sua architettura, al fine di migliorarlo. A seconda della dimensione dei tuoi dati e della domanda che stai ponendo, potresti non aver bisogno di costruire questo terzo set (come notiamo in [Previsioni di Serie Temporali](../../7-TimeSeries/1-Introduction/README.md)).

## Costruire un modello

Utilizzando i tuoi dati di addestramento, il tuo obiettivo è costruire un modello, o una rappresentazione statistica dei tuoi dati, utilizzando vari algoritmi per **addestrarlo**. Addestrare un modello lo espone ai dati e gli permette di fare supposizioni sugli schemi percepiti che scopre, valida e accetta o rifiuta.

### Decidere un metodo di addestramento

A seconda della tua domanda e della natura dei tuoi dati, sceglierai un metodo per addestrarlo. Esaminando la [documentazione di Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - che utilizziamo in questo corso - puoi esplorare molti modi per addestrare un modello. A seconda della tua esperienza, potresti dover provare diversi metodi per costruire il miglior modello. È probabile che attraverserai un processo in cui i data scientist valutano le prestazioni di un modello fornendogli dati mai visti prima, verificando l'accuratezza, i bias e altri problemi che degradano la qualità, e selezionando il metodo di addestramento più appropriato per il compito.

### Addestrare un modello

Con i tuoi dati di addestramento, sei pronto per "adattarli" per creare un modello. Noterai che in molte librerie di ML troverai il codice 'model.fit' - è in questo momento che invii la tua variabile caratteristica come un array di valori (di solito 'X') e una variabile target (di solito 'y').

### Valutare il modello

Una volta completato il processo di addestramento (può richiedere molte iterazioni, o 'epoche', per addestrare un modello grande), sarai in grado di valutare la qualità del modello utilizzando dati di test per misurarne le prestazioni. Questi dati sono un sottoinsieme dei dati originali che il modello non ha analizzato in precedenza. Puoi stampare una tabella di metriche sulla qualità del modello.

🎓 **Adattamento del modello**

Nel contesto del machine learning, l'adattamento del modello si riferisce all'accuratezza della funzione sottostante del modello mentre tenta di analizzare dati con cui non ha familiarità.

🎓 **Underfitting** e **overfitting** sono problemi comuni che degradano la qualità del modello, poiché il modello si adatta troppo poco o troppo bene. Questo causa previsioni troppo allineate o troppo poco allineate con i dati di addestramento. Un modello overfit prevede i dati di addestramento troppo bene perché ha imparato troppo bene i dettagli e il rumore dei dati. Un modello underfit non è accurato poiché non riesce ad analizzare accuratamente né i dati di addestramento né quelli che non ha ancora "visto".

![modello overfitting](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografica di [Jen Looper](https://twitter.com/jenlooper)

## Ottimizzazione dei parametri

Una volta completato il tuo addestramento iniziale, osserva la qualità del modello e considera di migliorarlo regolando i suoi 'iperparametri'. Leggi di più sul processo [nella documentazione](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Previsione

Questo è il momento in cui puoi utilizzare dati completamente nuovi per testare l'accuratezza del tuo modello. In un contesto di ML "applicato", dove stai costruendo risorse web per utilizzare il modello in produzione, questo processo potrebbe coinvolgere la raccolta di input dell'utente (ad esempio, la pressione di un pulsante) per impostare una variabile e inviarla al modello per l'inferenza o la valutazione.

In queste lezioni, scoprirai come utilizzare questi passaggi per preparare, costruire, testare, valutare e prevedere - tutti i gesti di un data scientist e altro ancora, mentre progredisci nel tuo percorso per diventare un ingegnere ML "full stack".

---

## 🚀Sfida

Disegna un diagramma di flusso che rifletta i passaggi di un praticante di ML. Dove ti trovi attualmente nel processo? Dove prevedi di incontrare difficoltà? Cosa ti sembra facile?

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e Studio Autonomo

Cerca online interviste con data scientist che discutono del loro lavoro quotidiano. Eccone [una](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Compito

[Intervista un data scientist](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.