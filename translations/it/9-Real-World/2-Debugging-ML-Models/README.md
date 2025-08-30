<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ba0f6e1019351351c8ee4c92867b6a0b",
  "translation_date": "2025-08-29T21:17:42+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "it"
}
-->
# Postscript: Debugging dei modelli di Machine Learning utilizzando i componenti della dashboard Responsible AI

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Introduzione

Il machine learning influenza la nostra vita quotidiana. L'IA sta trovando spazio in alcuni dei sistemi più importanti che ci riguardano come individui e come società, dalla sanità, alla finanza, all'istruzione e all'occupazione. Ad esempio, sistemi e modelli sono coinvolti in attività decisionali quotidiane, come diagnosi mediche o rilevamento di frodi. Di conseguenza, i progressi nell'IA, insieme alla sua adozione accelerata, stanno incontrando aspettative sociali in evoluzione e una crescente regolamentazione. Continuamente vediamo aree in cui i sistemi di IA non soddisfano le aspettative; espongono nuove sfide; e i governi stanno iniziando a regolamentare le soluzioni di IA. È quindi importante analizzare questi modelli per garantire risultati equi, affidabili, inclusivi, trasparenti e responsabili per tutti.

In questo curriculum, esamineremo strumenti pratici che possono essere utilizzati per valutare se un modello presenta problemi di IA responsabile. Le tecniche tradizionali di debugging del machine learning tendono a basarsi su calcoli quantitativi come l'accuratezza aggregata o la perdita media di errore. Immagina cosa può accadere quando i dati che utilizzi per costruire questi modelli mancano di determinate demografie, come razza, genere, opinione politica, religione, o rappresentano in modo sproporzionato tali demografie. E se l'output del modello fosse interpretato in modo da favorire alcune demografie? Questo può introdurre una sovra o sotto rappresentazione di questi gruppi sensibili, causando problemi di equità, inclusività o affidabilità nel modello. Un altro fattore è che i modelli di machine learning sono considerati "scatole nere", il che rende difficile comprendere e spiegare cosa guida le previsioni di un modello. Tutte queste sono sfide che i data scientist e gli sviluppatori di IA affrontano quando non dispongono di strumenti adeguati per analizzare e valutare l'equità o l'affidabilità di un modello.

In questa lezione, imparerai a fare debugging dei tuoi modelli utilizzando:

- **Analisi degli errori**: identificare dove nella distribuzione dei dati il modello presenta alti tassi di errore.
- **Panoramica del modello**: eseguire analisi comparative tra diversi gruppi di dati per scoprire disparità nelle metriche di performance del modello.
- **Analisi dei dati**: indagare dove potrebbe esserci una sovra o sotto rappresentazione dei dati che può inclinare il modello a favorire una demografia rispetto a un'altra.
- **Importanza delle caratteristiche**: comprendere quali caratteristiche stanno guidando le previsioni del modello a livello globale o locale.

## Prerequisito

Come prerequisito, ti invitiamo a consultare [Strumenti di IA responsabile per sviluppatori](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif sugli strumenti di IA responsabile](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Analisi degli errori

Le metriche tradizionali di performance dei modelli utilizzate per misurare l'accuratezza sono per lo più calcoli basati su previsioni corrette vs errate. Ad esempio, determinare che un modello è accurato l'89% delle volte con una perdita di errore di 0.001 può essere considerato una buona performance. Gli errori spesso non sono distribuiti uniformemente nel dataset sottostante. Potresti ottenere un punteggio di accuratezza del modello dell'89%, ma scoprire che ci sono regioni dei dati per cui il modello fallisce il 42% delle volte. La conseguenza di questi schemi di fallimento con determinati gruppi di dati può portare a problemi di equità o affidabilità. È essenziale comprendere le aree in cui il modello funziona bene o meno. Le regioni dei dati con un alto numero di imprecisioni nel modello potrebbero rivelarsi un'importante demografia dei dati.

![Analizza e correggi gli errori del modello](../../../../translated_images/ea-error-distribution.117452e1177c1dd84fab2369967a68bcde787c76c6ea7fdb92fcf15d1fce8206.it.png)

Il componente di Analisi degli Errori nella dashboard RAI illustra come i fallimenti del modello sono distribuiti tra vari gruppi con una visualizzazione ad albero. Questo è utile per identificare caratteristiche o aree con un alto tasso di errore nel dataset. Vedendo da dove provengono la maggior parte delle imprecisioni del modello, puoi iniziare a indagare sulla causa principale. Puoi anche creare gruppi di dati per eseguire analisi. Questi gruppi di dati aiutano nel processo di debugging per determinare perché la performance del modello è buona in un gruppo, ma errata in un altro.

![Analisi degli errori](../../../../translated_images/ea-error-cohort.6886209ea5d438c4daa8bfbf5ce3a7042586364dd3eccda4a4e3d05623ac702a.it.png)

Gli indicatori visivi sulla mappa ad albero aiutano a individuare più rapidamente le aree problematiche. Ad esempio, più scuro è il colore rosso di un nodo dell'albero, maggiore è il tasso di errore.

La mappa di calore è un'altra funzionalità di visualizzazione che gli utenti possono utilizzare per indagare il tasso di errore utilizzando una o due caratteristiche per trovare un contributore agli errori del modello in tutto il dataset o nei gruppi.

![Mappa di calore dell'analisi degli errori](../../../../translated_images/ea-heatmap.8d27185e28cee3830c85e1b2e9df9d2d5e5c8c940f41678efdb68753f2f7e56c.it.png)

Usa l'analisi degli errori quando hai bisogno di:

* Comprendere a fondo come i fallimenti del modello sono distribuiti in un dataset e tra diverse dimensioni di input e caratteristiche.
* Scomporre le metriche di performance aggregate per scoprire automaticamente gruppi errati e informare i tuoi passi di mitigazione mirati.

## Panoramica del modello

Valutare la performance di un modello di machine learning richiede una comprensione olistica del suo comportamento. Questo può essere ottenuto esaminando più di una metrica, come tasso di errore, accuratezza, richiamo, precisione o MAE (Errore Assoluto Medio), per trovare disparità tra le metriche di performance. Una metrica di performance può sembrare ottima, ma le imprecisioni possono emergere in un'altra metrica. Inoltre, confrontare le metriche per disparità in tutto il dataset o nei gruppi aiuta a far luce su dove il modello funziona bene o meno. Questo è particolarmente importante per osservare la performance del modello tra caratteristiche sensibili e non sensibili (es. razza, genere o età del paziente) per scoprire potenziali ingiustizie che il modello potrebbe avere. Ad esempio, scoprire che il modello è più errato in un gruppo che ha caratteristiche sensibili può rivelare potenziali ingiustizie.

Il componente Panoramica del Modello della dashboard RAI aiuta non solo ad analizzare le metriche di performance della rappresentazione dei dati in un gruppo, ma offre agli utenti la possibilità di confrontare il comportamento del modello tra diversi gruppi.

![Gruppi di dataset - panoramica del modello nella dashboard RAI](../../../../translated_images/model-overview-dataset-cohorts.dfa463fb527a35a0afc01b7b012fc87bf2cad756763f3652bbd810cac5d6cf33.it.png)

La funzionalità di analisi basata sulle caratteristiche del componente consente agli utenti di restringere sottogruppi di dati all'interno di una particolare caratteristica per identificare anomalie a livello granulare. Ad esempio, la dashboard ha un'intelligenza integrata per generare automaticamente gruppi per una caratteristica selezionata dall'utente (es. *"time_in_hospital < 3"* o *"time_in_hospital >= 7"*). Questo consente a un utente di isolare una particolare caratteristica da un gruppo di dati più ampio per vedere se è un influenzatore chiave degli esiti errati del modello.

![Gruppi di caratteristiche - panoramica del modello nella dashboard RAI](../../../../translated_images/model-overview-feature-cohorts.c5104d575ffd0c80b7ad8ede7703fab6166bfc6f9125dd395dcc4ace2f522f70.it.png)

Il componente Panoramica del Modello supporta due classi di metriche di disparità:

**Disparità nella performance del modello**: Questi set di metriche calcolano la disparità (differenza) nei valori della metrica di performance selezionata tra sottogruppi di dati. Ecco alcuni esempi:

* Disparità nel tasso di accuratezza
* Disparità nel tasso di errore
* Disparità nella precisione
* Disparità nel richiamo
* Disparità nell'errore assoluto medio (MAE)

**Disparità nel tasso di selezione**: Questa metrica contiene la differenza nel tasso di selezione (previsione favorevole) tra sottogruppi. Un esempio di ciò è la disparità nei tassi di approvazione dei prestiti. Il tasso di selezione indica la frazione di punti dati in ogni classe classificati come 1 (nella classificazione binaria) o la distribuzione dei valori di previsione (nella regressione).

## Analisi dei dati

> "Se torturi i dati abbastanza a lungo, confesseranno qualsiasi cosa" - Ronald Coase

Questa affermazione può sembrare estrema, ma è vero che i dati possono essere manipolati per supportare qualsiasi conclusione. Tale manipolazione può talvolta avvenire involontariamente. Come esseri umani, abbiamo tutti dei pregiudizi, ed è spesso difficile sapere consapevolmente quando stiamo introducendo pregiudizi nei dati. Garantire equità nell'IA e nel machine learning rimane una sfida complessa.

I dati rappresentano un enorme punto cieco per le metriche tradizionali di performance dei modelli. Potresti avere punteggi di accuratezza elevati, ma questo non riflette sempre il pregiudizio sottostante che potrebbe essere presente nel tuo dataset. Ad esempio, se un dataset di dipendenti ha il 27% di donne in posizioni dirigenziali in un'azienda e il 73% di uomini nello stesso livello, un modello di IA per la pubblicità di lavoro addestrato su questi dati potrebbe indirizzare principalmente un pubblico maschile per posizioni di alto livello. Questo squilibrio nei dati ha inclinato la previsione del modello a favorire un genere rispetto a un altro. Questo rivela un problema di equità, dove c'è un pregiudizio di genere nel modello di IA.

Il componente Analisi dei Dati nella dashboard RAI aiuta a identificare aree in cui c'è una sovra o sotto rappresentazione nel dataset. Aiuta gli utenti a diagnosticare la causa principale degli errori e dei problemi di equità introdotti da squilibri nei dati o dalla mancanza di rappresentazione di un particolare gruppo di dati. Questo offre agli utenti la possibilità di visualizzare i dataset basati su risultati previsti e reali, gruppi di errori e caratteristiche specifiche. Talvolta scoprire un gruppo di dati sottorappresentato può anche rivelare che il modello non sta imparando bene, da cui derivano le alte imprecisioni. Avere un modello con pregiudizi nei dati non è solo un problema di equità, ma dimostra che il modello non è inclusivo o affidabile.

![Componente Analisi dei Dati nella dashboard RAI](../../../../translated_images/dataanalysis-cover.8d6d0683a70a5c1e274e5a94b27a71137e3d0a3b707761d7170eb340dd07f11d.it.png)

Usa l'analisi dei dati quando hai bisogno di:

* Esplorare le statistiche del tuo dataset selezionando diversi filtri per suddividere i dati in diverse dimensioni (noti anche come gruppi).
* Comprendere la distribuzione del tuo dataset tra diversi gruppi e caratteristiche.
* Determinare se le tue scoperte relative a equità, analisi degli errori e causalità (derivate da altri componenti della dashboard) sono il risultato della distribuzione del tuo dataset.
* Decidere in quali aree raccogliere più dati per mitigare gli errori derivanti da problemi di rappresentazione, rumore delle etichette, rumore delle caratteristiche, pregiudizi delle etichette e fattori simili.

## Interpretabilità del modello

I modelli di machine learning tendono a essere "scatole nere". Comprendere quali caratteristiche chiave dei dati guidano la previsione di un modello può essere una sfida. È importante fornire trasparenza sul motivo per cui un modello fa una certa previsione. Ad esempio, se un sistema di IA prevede che un paziente diabetico è a rischio di essere ricoverato nuovamente in ospedale entro 30 giorni, dovrebbe essere in grado di fornire dati di supporto che hanno portato alla sua previsione. Avere indicatori di supporto porta trasparenza per aiutare i medici o gli ospedali a prendere decisioni ben informate. Inoltre, essere in grado di spiegare perché un modello ha fatto una previsione per un singolo paziente consente responsabilità con le normative sanitarie. Quando utilizzi modelli di machine learning in modi che influenzano la vita delle persone, è cruciale comprendere e spiegare cosa influenza il comportamento di un modello. L'esplicabilità e l'interpretabilità del modello aiutano a rispondere a domande in scenari come:

* Debug del modello: Perché il mio modello ha commesso questo errore? Come posso migliorare il mio modello?
* Collaborazione uomo-IA: Come posso comprendere e fidarmi delle decisioni del modello?
* Conformità normativa: Il mio modello soddisfa i requisiti legali?

Il componente Importanza delle Caratteristiche della dashboard RAI ti aiuta a fare debugging e ottenere una comprensione completa di come un modello fa previsioni. È anche uno strumento utile per i professionisti del machine learning e i decisori per spiegare e mostrare prove delle caratteristiche che influenzano il comportamento del modello per la conformità normativa. Successivamente, gli utenti possono esplorare spiegazioni globali e locali per convalidare quali caratteristiche guidano le previsioni del modello. Le spiegazioni globali elencano le principali caratteristiche che hanno influenzato la previsione complessiva del modello. Le spiegazioni locali mostrano quali caratteristiche hanno portato alla previsione del modello per un caso individuale. La capacità di valutare spiegazioni locali è anche utile nel debugging o nell'auditing di un caso specifico per comprendere meglio e interpretare perché un modello ha fatto una previsione accurata o inaccurata.

![Componente Importanza delle Caratteristiche della dashboard RAI](../../../../translated_images/9-feature-importance.cd3193b4bba3fd4bccd415f566c2437fb3298c4824a3dabbcab15270d783606e.it.png)

* Spiegazioni globali: Ad esempio, quali caratteristiche influenzano il comportamento complessivo di un modello di ricovero ospedaliero per diabete?
* Spiegazioni locali: Ad esempio, perché un paziente diabetico di età superiore ai 60 anni con ricoveri precedenti è stato previsto come ricoverato o non ricoverato entro 30 giorni in ospedale?

Nel processo di debugging per esaminare la performance del modello tra diversi gruppi, Importanza delle Caratteristiche mostra il livello di impatto che una caratteristica ha tra i gruppi. Aiuta a rivelare anomalie quando si confronta il livello di influenza che la caratteristica ha nel guidare le previsioni errate del modello. Il componente Importanza delle Caratteristiche può mostrare quali valori in una caratteristica hanno influenzato positivamente o negativamente l'esito del modello. Ad esempio, se un modello ha fatto una previsione inaccurata, il componente ti dà la possibilità di approfondire e individuare quali caratteristiche o valori delle caratteristiche hanno guidato la previsione. Questo livello di dettaglio aiuta non solo nel debugging ma fornisce trasparenza e responsabilità in situazioni di auditing. Infine, il componente può aiutarti a identificare problemi di equità. Per illustrare, se una caratteristica sensibile come etnia o genere è altamente influente nel guidare la previsione del modello, questo potrebbe essere un segno di pregiudizio razziale o di genere nel modello.

![Importanza delle caratteristiche](../../../../translated_images/9-features-influence.3ead3d3f68a84029f1e40d3eba82107445d3d3b6975d4682b23d8acc905da6d0.it.png)

Usa l'interpretabilità quando hai bisogno di:

* Determinare quanto sono affidabili le previsioni del tuo sistema di IA comprendendo quali caratteristiche sono più importanti per le previsioni.
* Approcciare il debugging del tuo modello comprendendolo prima e identificando se il modello sta utilizzando caratteristiche valide o semplicemente false correlazioni.
* Scoprire potenziali fonti di ingiustizia comprendendo se il modello sta basando le previsioni su caratteristiche sensibili o su caratteristiche altamente correlate con esse.
* Costruire fiducia degli utenti nelle decisioni del modello generando spiegazioni locali per illustrare i loro risultati.
* Completare un audit normativo di un sistema di IA per convalidare i modelli e monitorare l'impatto delle decisioni del modello sugli esseri umani.

## Conclusione

Tutti i componenti della dashboard RAI sono strumenti pratici per aiutarti a costruire modelli di machine learning meno dannosi e più affidabili per la società. Migliorano la prevenzione di minacce ai diritti umani; discriminazione o esclusione di determinati gruppi dalle opportunità di vita; e il rischio di danni fisici o psicologici. Aiutano anche a costruire fiducia nelle decisioni del tuo modello generando spiegazioni locali per illustrare i loro risultati. Alcuni dei potenziali danni possono essere classificati come:

- **Allocazione**, se un genere o un'etnia, ad esempio, è favorito rispetto a un altro.
- **Qualità del servizio**. Se addestri i dati per uno scenario specifico ma la realtà è molto più complessa, ciò porta a un servizio di scarsa qualità.
- **Stereotipizzazione**. Associare un determinato gruppo a attributi preassegnati.
- **Denigrazione**. Criticare ingiustamente e etichettare qualcosa o qualcuno.
- **Sovra- o sotto-rappresentazione**. L'idea è che un determinato gruppo non sia rappresentato in una certa professione, e qualsiasi servizio o funzione che continui a promuovere questa situazione contribuisce a causare danni.

### Dashboard Azure RAI

La [dashboard Azure RAI](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu), basata su strumenti open-source sviluppati da istituzioni accademiche e organizzazioni leader, tra cui Microsoft, è fondamentale per i data scientist e gli sviluppatori di AI per comprendere meglio il comportamento dei modelli, individuare e mitigare problemi indesiderati nei modelli di AI.

- Scopri come utilizzare i diversi componenti consultando la [documentazione della dashboard RAI.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Dai un'occhiata ad alcuni [notebook di esempio della dashboard RAI](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) per il debugging di scenari di AI più responsabile in Azure Machine Learning.

---
## 🚀 Sfida

Per evitare che si introducano pregiudizi statistici o nei dati fin dall'inizio, dovremmo:

- garantire una diversità di background e prospettive tra le persone che lavorano sui sistemi
- investire in dataset che riflettano la diversità della nostra società
- sviluppare metodi migliori per rilevare e correggere i pregiudizi quando si verificano

Pensa a scenari reali in cui l'ingiustizia è evidente nella costruzione e nell'uso dei modelli. Cos'altro dovremmo considerare?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Revisione e Studio Autonomo

In questa lezione, hai appreso alcuni strumenti pratici per integrare l'AI responsabile nel machine learning.

Guarda questo workshop per approfondire gli argomenti:

- Responsible AI Dashboard: Un punto di riferimento per mettere in pratica l'AI responsabile, a cura di Besmira Nushi e Mehrnoosh Sameki

[![Responsible AI Dashboard: Un punto di riferimento per mettere in pratica l'AI responsabile](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Un punto di riferimento per mettere in pratica l'AI responsabile")

> 🎥 Clicca sull'immagine sopra per il video: Responsible AI Dashboard: Un punto di riferimento per mettere in pratica l'AI responsabile, a cura di Besmira Nushi e Mehrnoosh Sameki

Consulta i seguenti materiali per saperne di più sull'AI responsabile e su come costruire modelli più affidabili:

- Strumenti della dashboard RAI di Microsoft per il debugging dei modelli di ML: [Risorse sugli strumenti di AI responsabile](https://aka.ms/rai-dashboard)

- Esplora il toolkit di AI responsabile: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Centro risorse RAI di Microsoft: [Risorse sull'AI responsabile – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Gruppo di ricerca FATE di Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Compito

[Esplora la dashboard RAI](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche potrebbero contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si consiglia una traduzione professionale eseguita da un traduttore umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.