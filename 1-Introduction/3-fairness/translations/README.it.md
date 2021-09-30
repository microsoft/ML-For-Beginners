# Equità e machine learning

![Riepilogo dell'equità in machine learning in uno sketchnote](../../../sketchnotes/ml-fairness.png)
> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz pre-lezione](https://white-water-09ec41f0f.azurestaticapps.net/quiz/5/?loc=it)

## Introduzione

In questo programma di studi, si inizierà a scoprire come machine learning può e sta influenzando la  vita quotidiana. Anche ora, sistemi e modelli sono coinvolti nelle attività decisionali quotidiane, come le diagnosi sanitarie o l'individuazione di frodi. Quindi è importante che questi modelli funzionino bene per fornire risultati equi per tutti.

Si immagini cosa può accadere quando i dati che si stanno utilizzando per costruire questi modelli mancano di determinati dati demografici, come razza, genere, visione politica, religione, o rappresentano tali dati demografici in modo sproporzionato. E quando il risultato del modello viene interpretato per favorire alcuni gruppi demografici? Qual è la conseguenza per l'applicazione?

In questa lezione, si dovrà:

- Aumentare la propria consapevolezza sull'importanza dell'equità nel machine learning.
- Informarsi sui danni legati all'equità.
- Apprendere ulteriori informazioni sulla valutazione e la mitigazione dell'ingiustizia.

## Prerequisito

Come prerequisito, si segua il percorso di apprendimento "Principi di AI Responsabile" e si guardi il video qui sotto sull'argomento:

Si scopra di più sull'AI Responsabile seguendo questo [percorso di apprendimento](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-15963-cxa)

[![L'approccio di Microsoft all'AI responsabileL'](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "approccio di Microsoft all'AI Responsabile")

> 🎥 Fare clic sull'immagine sopra per un video: L'approccio di Microsoft all'AI Responsabile

## Iniquità nei dati e negli algoritmi

> "Se si torturano i dati abbastanza a lungo, essi confesseranno qualsiasi cosa" - Ronald Coase

Questa affermazione suona estrema, ma è vero che i dati possono essere manipolati per supportare qualsiasi conclusione. Tale manipolazione a volte può avvenire involontariamente. Come esseri umani, abbiamo tutti dei pregiudizi, ed è spesso difficile sapere consapevolmente quando si introduce un pregiudizio nei dati.

Garantire l'equità nell'intelligenza artificiale e machine learning rimane una sfida socio-tecnica complessa. Ciò significa che non può essere affrontata da prospettive puramente sociali o tecniche.

### Danni legati all'equità

Cosa si intende per ingiustizia? L'"ingiustizia" comprende gli impatti negativi, o "danni", per un gruppo di persone, come quelli definiti in termini di razza, genere, età o stato di disabilità.

I principali danni legati all'equità possono essere classificati come:

- **Allocazione**, se un genere o un'etnia, ad esempio, sono preferiti a un altro.
- **Qualità di servizio** Se si addestrano i dati per uno scenario specifico, ma la realtà è molto più complessa, si ottiene un servizio scadente.
- **Stereotipi**. Associazione di un dato gruppo con attributi preassegnati.
- **Denigrazione**. Criticare ed etichettare ingiustamente qualcosa o qualcuno.
- **Sovra o sotto rappresentazione**. L'idea è che un certo gruppo non è visto in una certa professione, e qualsiasi servizio o funzione che continua a promuovere ciò, contribuisce al danno.

Si dia un'occhiata agli esempi.

### Allocazione

Si consideri un ipotetico sistema per la scrematura delle domande di prestito. Il sistema tende a scegliere gli uomini bianchi come candidati migliori rispetto ad altri gruppi. Di conseguenza, i prestiti vengono negati ad alcuni richiedenti.

Un altro esempio potrebbe essere uno strumento sperimentale di assunzione sviluppato da una grande azienda per selezionare i candidati. Lo strumento discrimina sistematicamente un genere utilizzando i modelli che sono stati addestrati a preferire parole associate con altro. Ha portato a penalizzare i candidati i cui curricula contengono parole come "squadra di rugby femminile".

✅ Si compia una piccola ricerca per trovare un esempio reale di qualcosa del genere

### Qualità di Servizio

I ricercatori hanno scoperto che diversi classificatori di genere commerciali avevano tassi di errore più elevati intorno alle immagini di donne con tonalità della pelle più scura rispetto alle immagini di uomini con tonalità della pelle più chiare. [Riferimento](https://www.media.mit.edu/publications/gender-shades-intersectional-accuracy-disparities-in-commercial-gender-classification/)

Un altro esempio infamante è un distributore di sapone per le mani che sembrava non essere in grado di percepire le persone con la pelle scura. [Riferimento](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)

### Stereotipi

La visione di genere stereotipata è stata trovata nella traduzione automatica. Durante la traduzione in turco "he is a nurse and she is a doctor" (lui è un'infermiere e lei un medico), sono stati riscontrati problemi. Il turco è una lingua senza genere che ha un pronome, "o" per trasmettere una terza persona singolare, ma tradurre la frase dal turco all'inglese produce lo stereotipo e scorretto come "she is a nurse and he is a doctor" (lei è un'infermiera e lui è un medico).

![traduzione in turco](../images/gender-bias-translate-en-tr.png)

![Traduzione in inglese](../images/gender-bias-translate-tr-en.png)

### Denigrazione

Una tecnologia di etichettatura delle immagini ha contrassegnato in modo infamante le immagini di persone dalla pelle scura come gorilla. L'etichettatura errata è dannosa non solo perché il sistema ha commesso un errore, ma anche perché ha applicato specificamente un'etichetta che ha una lunga storia di essere intenzionalmente utilizzata per denigrare i neri.

[![AI: Non sono una donna?](https://img.youtube.com/vi/QxuyfWoVV98/0.jpg)](https://www.youtube.com/watch?v=QxuyfWoVV98 "AI, non sono una donna?")
> 🎥 Cliccare sull'immagine sopra per un video: AI, Ain't I a Woman - una performance che mostra il danno causato dalla denigrazione razzista da parte dell'AI

### Sovra o sotto rappresentazione

I risultati di ricerca di immagini distorti possono essere un buon esempio di questo danno. Quando si cercano immagini di professioni con una percentuale uguale o superiore di uomini rispetto alle donne, come l'ingegneria o CEO, si osserva che i risultati sono più fortemente distorti verso un determinato genere.

![Ricerca CEO di Bing](../images/ceos.png)
> Questa ricerca su Bing per "CEO" produce risultati piuttosto inclusivi

Questi cinque principali tipi di danno non si escludono a vicenda e un singolo sistema può presentare più di un tipo di danno. Inoltre, ogni caso varia nella sua gravità. Ad esempio, etichettare ingiustamente qualcuno come criminale è un danno molto più grave che etichettare erroneamente un'immagine. È importante, tuttavia, ricordare che anche danni relativamente non gravi possono far sentire le persone alienate o emarginate e l'impatto cumulativo può essere estremamente opprimente.

✅ **Discussione**: rivisitare alcuni degli esempi e vedere se mostrano danni diversi.

|                                     | Allocatione | Qualita di servizio | Stereotipo | Denigrazione | Sovra o sotto rappresentazione |
| ----------------------------------- | :---------: | :-----------------: | :--------: | :----------: | :----------------------------: |
| Sistema di assunzione automatizzato |      x      |          x          |     x      |              |               x                |
| Traduzione automatica               |             |                     |            |              |                                |
| Eitchettatura foto                  |             |                     |            |              |                                |

## Rilevare l'ingiustizia

Ci sono molte ragioni per cui un dato sistema si comporta in modo scorretto. I pregiudizi sociali, ad esempio, potrebbero riflettersi nell'insieme di dati utilizzati per addestrarli. Ad esempio, l'ingiustizia delle assunzioni potrebbe essere stata esacerbata dall'eccessivo affidamento sui dati storici. Utilizzando i modelli nei curricula inviati all'azienda per un periodo di 10 anni, il modello ha determinato che gli uomini erano più qualificati perché la maggior parte dei curricula proveniva da uomini, un riflesso del passato dominio maschile nell'industria tecnologica.

Dati inadeguati su un determinato gruppo di persone possono essere motivo di ingiustizia. Ad esempio, i classificatori di immagini hanno un tasso di errore più elevato per le immagini di persone dalla pelle scura perché le tonalità della pelle più scure sono sottorappresentate nei dati.

Anche le ipotesi errate fatte durante lo sviluppo causano iniquità. Ad esempio, un sistema di analisi facciale destinato a prevedere chi commetterà un crimine basato sulle immagini dei volti delle persone può portare a ipotesi dannose. Ciò potrebbe portare a danni sostanziali per le persone classificate erroneamente.

## Si comprendano i propri modelli e si costruiscano in modo onesto

Sebbene molti aspetti dell'equità non vengano catturati nelle metriche di equità quantitativa e non sia possibile rimuovere completamente i pregiudizi da un sistema per garantire l'equità, si è comunque responsabili di rilevare e mitigare il più possibile i problemi di equità.

Quando si lavora con modelli di machine learning, è importante comprendere i propri modelli assicurandone l'interpretabilità e valutando e mitigando l'ingiustizia.

Si utilizza l'esempio di selezione del prestito per isolare il caso e determinare il livello di impatto di ciascun fattore sulla previsione.

## Metodi di valutazione

1. **Identificare i danni (e benefici)**. Il primo passo è identificare danni e benefici. Si pensi a come azioni e decisioni possono influenzare sia i potenziali clienti che un'azienda stessa.

1. **Identificare i gruppi interessati**. Una volta compreso il tipo di danni o benefici che possono verificarsi, identificare i gruppi che potrebbero essere interessati. Questi gruppi sono definiti per genere, etnia o gruppo sociale?

1. **Definire le metriche di equità**. Infine, si definisca una metrica in modo da avere qualcosa su cui misurare il proprio lavoro per migliorare la situazione.

### **Identificare danni (e benefici)**

Quali sono i danni e i benefici associati al prestito? Si pensi agli scenari di falsi negativi e falsi positivi:

**Falsi negativi** (rifiutato, ma Y=1) - in questo caso viene rifiutato un richiedente che sarà in grado di rimborsare un prestito. Questo è un evento avverso perché le risorse dei prestiti non sono erogate a richiedenti qualificati.

**Falsi positivi** (accettato, ma Y=0) - in questo caso, il richiedente ottiene un prestito ma alla fine fallisce. Di conseguenza, il caso del richiedente verrà inviato a un'agenzia di recupero crediti che può influire sulle sue future richieste di prestito.

### **Identificare i gruppi interessati**

Il passo successivo è determinare quali gruppi potrebbero essere interessati. Ad esempio, nel caso di una richiesta di carta di credito, un modello potrebbe stabilire che le donne dovrebbero ricevere limiti di credito molto più bassi rispetto ai loro coniugi che condividono i beni familiari. Un intero gruppo demografico, definito in base al genere, è così interessato.

### **Definire le metriche di equità**

Si sono identificati i danni e un gruppo interessato, in questo caso, delineato per genere. Ora, si usino i fattori quantificati per disaggregare le loro metriche. Ad esempio, utilizzando i dati di seguito, si può vedere che le donne hanno il più alto tasso di falsi positivi e gli uomini il più piccolo, e che è vero il contrario per i falsi negativi.

✅ In una futura lezione sul Clustering, si vedrà come costruire questa 'matrice di confusione' nel codice

|             | percentuale di falsi positivi | Percentuale di falsi negativi | conteggio |
| ----------- | ----------------------------- | ----------------------------- | --------- |
| Donna       | 0,37                          | 0,27                          | 54032     |
| Uomo        | 0,31                          | 0.35                          | 28620     |
| Non binario | 0,33                          | 0,31                          | 1266      |

Questa tabella ci dice diverse cose. Innanzitutto, si nota che ci sono relativamente poche persone non binarie nei dati. I dati sono distorti, quindi si deve fare attenzione a come si interpretano  questi numeri.

In questo caso, ci sono 3 gruppi e 2 metriche. Quando si pensa a come il nostro sistema influisce sul gruppo di clienti con i loro richiedenti di prestito, questo può essere sufficiente, ma quando si desidera definire un numero maggiore di gruppi, è possibile distillare questo in insiemi più piccoli di riepiloghi. Per fare ciò, si possono aggiungere più metriche, come la differenza più grande o il rapporto più piccolo di ogni falso negativo e falso positivo.

✅ Ci si fermi a pensare: quali altri gruppi potrebbero essere interessati dalla richiesta di prestito?

## Mitigare l'ingiustizia

Per mitigare l'ingiustizia, si esplori il modello per generare vari modelli mitigati e si confrontino i compromessi tra accuratezza ed equità per selezionare il modello più equo.

Questa lezione introduttiva non approfondisce i dettagli dell'algoritmo della mitigazione dell'ingiustizia, come l'approccio di post-elaborazione e riduzione, ma ecco uno strumento che si potrebbe voler provare.

### Fairlearn

[Fairlearn](https://fairlearn.github.io/) è un pacchetto Python open source che consente di valutare l'equità dei propri sistemi e mitigare l'ingiustizia.

Lo strumento consente di valutare in che modo le previsioni di un modello influiscono su diversi gruppi, consentendo di confrontare più modelli utilizzando metriche di equità e prestazioni e fornendo una serie di algoritmi per mitigare l'ingiustizia nella classificazione binaria e nella regressione.

- Si scopra come utilizzare i diversi componenti controllando il GitHub di [Fairlearn](https://github.com/fairlearn/fairlearn/)

- Si esplori la [guida per l'utente](https://fairlearn.github.io/main/user_guide/index.html), e gli [esempi](https://fairlearn.github.io/main/auto_examples/index.html)

- Si provino alcuni [notebook di esempio](https://github.com/fairlearn/fairlearn/tree/master/notebooks).

- Si scopra [come abilitare le valutazioni dell'equità](https://docs.microsoft.com/azure/machine-learning/how-to-machine-learning-fairness-aml?WT.mc_id=academic-15963-cxa) dei modelli di Machine Learning in Azure Machine Learning.

- Si dia un'occhiata a questi [notebook di esempio](https://github.com/Azure/MachineLearningNotebooks/tree/master/contrib/fairness) per ulteriori scenari di valutazione dell'equità in Azure Machine Learning.

---

## 🚀 Sfida

Per evitare che vengano introdotti pregiudizi, in primo luogo, si dovrebbe:

- avere una diversità di background e prospettive tra le persone che lavorano sui sistemi
- investire in insiemi di dati che riflettano le diversità della società
- sviluppare metodi migliori per rilevare e correggere i pregiudizi quando si verificano

Si pensi a scenari di vita reale in cui l'ingiustizia è evidente nella creazione e nell'utilizzo del modello. Cos'altro si dovrebbe considerare?

## [Quiz post-lezione](https://white-water-09ec41f0f.azurestaticapps.net/quiz/6/?loc=it)

## Revisione e Auto Apprendimento

In questa lezione si sono apprese alcune nozioni di base sui concetti di equità e ingiustizia in machine learning.

Si guardi questo workshop per approfondire gli argomenti:

- YouTube: Danni correlati all'equità nei sistemi di IA: esempi, valutazione e mitigazione di Hanna Wallach e Miro Dudik [Danni correlati all'equità nei sistemi di IA: esempi, valutazione e mitigazione - YouTube](https://www.youtube.com/watch?v=1RptHwfkx_k)

Si legga anche:

- Centro risorse RAI di Microsoft: [risorse AI responsabili – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Gruppo di ricerca FATE di Microsoft[: FATE: equità, responsabilità, trasparenza ed etica nell'intelligenza artificiale - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

Si esplori il toolkit Fairlearn

[Fairlearn](https://fairlearn.org/)

Si scoprano gli strumenti di Azure Machine Learning per garantire l'equità

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-15963-cxa)

## Compito

[Esplorare Fairlearn](assignment.it.md)
