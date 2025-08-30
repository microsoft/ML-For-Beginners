<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20f18ff565638be615df4174858e4a7f",
  "translation_date": "2025-08-29T21:12:50+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "it"
}
-->
# Postscript: Machine learning nel mondo reale

![Riepilogo del machine learning nel mondo reale in uno sketchnote](../../../../translated_images/ml-realworld.26ee2746716155771f8076598b6145e6533fe4a9e2e465ea745f46648cbf1b84.it.png)  
> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

In questo curriculum, hai imparato molti modi per preparare i dati per l'addestramento e creare modelli di machine learning. Hai costruito una serie di modelli classici di regressione, clustering, classificazione, elaborazione del linguaggio naturale e serie temporali. Congratulazioni! Ora potresti chiederti a cosa serva tutto questo... quali sono le applicazioni reali di questi modelli?

Sebbene l'industria sia molto interessata all'IA, che di solito sfrutta il deep learning, ci sono ancora applicazioni preziose per i modelli classici di machine learning. Potresti persino utilizzare alcune di queste applicazioni già oggi! In questa lezione, esplorerai come otto diversi settori e domini di competenza utilizzano questi tipi di modelli per rendere le loro applicazioni più performanti, affidabili, intelligenti e utili per gli utenti.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/)

## 💰 Finanza

Il settore finanziario offre molte opportunità per il machine learning. Molti problemi in questo ambito si prestano a essere modellati e risolti utilizzando il ML.

### Rilevamento delle frodi con carte di credito

Abbiamo imparato a conoscere il [clustering k-means](../../5-Clustering/2-K-Means/README.md) in precedenza nel corso, ma come può essere utilizzato per risolvere problemi legati alle frodi con carte di credito?

Il clustering k-means è utile in una tecnica di rilevamento delle frodi con carte di credito chiamata **rilevamento degli outlier**. Gli outlier, o deviazioni nelle osservazioni su un set di dati, possono indicarci se una carta di credito viene utilizzata normalmente o se sta accadendo qualcosa di insolito. Come mostrato nell'articolo collegato di seguito, è possibile ordinare i dati delle carte di credito utilizzando un algoritmo di clustering k-means e assegnare ogni transazione a un cluster in base a quanto appare come un outlier. Successivamente, è possibile valutare i cluster più rischiosi per distinguere le transazioni fraudolente da quelle legittime.  
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestione patrimoniale

Nella gestione patrimoniale, un individuo o un'azienda gestisce gli investimenti per conto dei propri clienti. Il loro compito è mantenere e far crescere la ricchezza a lungo termine, quindi è essenziale scegliere investimenti che performino bene.

Un modo per valutare le prestazioni di un particolare investimento è attraverso la regressione statistica. La [regressione lineare](../../2-Regression/1-Tools/README.md) è uno strumento prezioso per comprendere come un fondo si comporta rispetto a un benchmark. Possiamo anche dedurre se i risultati della regressione sono statisticamente significativi o quanto influirebbero sugli investimenti di un cliente. È possibile ampliare ulteriormente l'analisi utilizzando la regressione multipla, dove possono essere presi in considerazione ulteriori fattori di rischio. Per un esempio di come ciò funzionerebbe per un fondo specifico, consulta l'articolo di seguito sull'uso della regressione per valutare le prestazioni di un fondo.  
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educazione

Anche il settore educativo è un'area molto interessante in cui il ML può essere applicato. Ci sono problemi intriganti da affrontare, come rilevare imbrogli nei test o nei saggi o gestire i bias, intenzionali o meno, nel processo di correzione.

### Prevedere il comportamento degli studenti

[Coursera](https://coursera.com), un fornitore di corsi online aperti, ha un ottimo blog tecnico in cui discute molte decisioni ingegneristiche. In questo caso di studio, hanno tracciato una linea di regressione per esplorare eventuali correlazioni tra un basso punteggio NPS (Net Promoter Score) e la ritenzione o l'abbandono del corso.  
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigare i bias

[Grammarly](https://grammarly.com), un assistente di scrittura che controlla errori di ortografia e grammatica, utilizza sofisticati [sistemi di elaborazione del linguaggio naturale](../../6-NLP/README.md) nei suoi prodotti. Hanno pubblicato un interessante caso di studio nel loro blog tecnico su come hanno affrontato il bias di genere nel machine learning, di cui hai appreso nella nostra [lezione introduttiva sull'equità](../../1-Introduction/3-fairness/README.md).  
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retail

Il settore retail può sicuramente beneficiare dell'uso del ML, con applicazioni che vanno dal miglioramento del percorso del cliente all'ottimizzazione della gestione dell'inventario.

### Personalizzazione del percorso del cliente

In Wayfair, un'azienda che vende articoli per la casa come mobili, aiutare i clienti a trovare i prodotti giusti per i loro gusti e bisogni è fondamentale. In questo articolo, gli ingegneri dell'azienda descrivono come utilizzano ML e NLP per "mostrare i risultati giusti ai clienti". In particolare, il loro Query Intent Engine è stato costruito per utilizzare l'estrazione di entità, l'addestramento di classificatori, l'estrazione di asset e opinioni e il tagging del sentiment nelle recensioni dei clienti. Questo è un classico caso d'uso di come l'NLP funziona nel retail online.  
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestione dell'inventario

Aziende innovative e agili come [StitchFix](https://stitchfix.com), un servizio di box che spedisce abbigliamento ai consumatori, si affidano fortemente al ML per le raccomandazioni e la gestione dell'inventario. I loro team di stilisti lavorano insieme ai team di merchandising, infatti: "uno dei nostri data scientist ha sperimentato un algoritmo genetico e lo ha applicato all'abbigliamento per prevedere quale sarebbe stato un capo di successo che non esiste oggi. Lo abbiamo presentato al team di merchandising e ora possono usarlo come strumento."  
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Sanità

Il settore sanitario può sfruttare il ML per ottimizzare compiti di ricerca e anche problemi logistici come la riammissione dei pazienti o il contenimento delle malattie.

### Gestione delle sperimentazioni cliniche

La tossicità nelle sperimentazioni cliniche è una grande preoccupazione per i produttori di farmaci. Quanta tossicità è tollerabile? In questo studio, l'analisi di vari metodi di sperimentazione clinica ha portato allo sviluppo di un nuovo approccio per prevedere le probabilità di esiti delle sperimentazioni cliniche. In particolare, sono stati in grado di utilizzare random forest per produrre un [classificatore](../../4-Classification/README.md) in grado di distinguere tra gruppi di farmaci.  
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestione della riammissione ospedaliera

L'assistenza ospedaliera è costosa, soprattutto quando i pazienti devono essere riammessi. Questo articolo discute di un'azienda che utilizza il ML per prevedere il potenziale di riammissione utilizzando algoritmi di [clustering](../../5-Clustering/README.md). Questi cluster aiutano gli analisti a "scoprire gruppi di riammissioni che potrebbero condividere una causa comune".  
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestione delle malattie

La recente pandemia ha messo in evidenza i modi in cui il machine learning può aiutare a fermare la diffusione delle malattie. In questo articolo, riconoscerai l'uso di ARIMA, curve logistiche, regressione lineare e SARIMA. "Questo lavoro è un tentativo di calcolare il tasso di diffusione di questo virus e quindi di prevedere i decessi, le guarigioni e i casi confermati, in modo che possa aiutarci a prepararci meglio e sopravvivere."  
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecologia e Green Tech

La natura e l'ecologia consistono in molti sistemi sensibili in cui l'interazione tra animali e natura è al centro dell'attenzione. È importante essere in grado di misurare accuratamente questi sistemi e agire in modo appropriato se accade qualcosa, come un incendio boschivo o un calo della popolazione animale.

### Gestione forestale

Hai imparato il [Reinforcement Learning](../../8-Reinforcement/README.md) nelle lezioni precedenti. Può essere molto utile quando si cerca di prevedere modelli in natura. In particolare, può essere utilizzato per monitorare problemi ecologici come incendi boschivi e la diffusione di specie invasive. In Canada, un gruppo di ricercatori ha utilizzato il Reinforcement Learning per costruire modelli di dinamiche degli incendi boschivi a partire da immagini satellitari. Utilizzando un innovativo "processo di diffusione spaziale (SSP)", hanno immaginato un incendio boschivo come "l'agente in qualsiasi cella del paesaggio". "Il set di azioni che il fuoco può intraprendere da una posizione in qualsiasi momento include la diffusione a nord, sud, est o ovest o il non diffondersi."  

Questo approccio inverte il solito setup del RL poiché le dinamiche del corrispondente Markov Decision Process (MDP) sono una funzione nota per la diffusione immediata degli incendi boschivi. Leggi di più sugli algoritmi classici utilizzati da questo gruppo al link sottostante.  
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Monitoraggio del movimento degli animali

Sebbene il deep learning abbia creato una rivoluzione nel monitoraggio visivo dei movimenti degli animali (puoi costruire il tuo [tracker per orsi polari](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) qui), il ML classico ha ancora un ruolo in questo compito.

I sensori per monitorare i movimenti degli animali da fattoria e l'IoT fanno uso di questo tipo di elaborazione visiva, ma tecniche di ML più basilari sono utili per pre-elaborare i dati. Ad esempio, in questo articolo, le posture delle pecore sono state monitorate e analizzate utilizzando vari algoritmi di classificazione. Potresti riconoscere la curva ROC a pagina 335.  
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestione dell'energia

Nelle nostre lezioni sulla [previsione delle serie temporali](../../7-TimeSeries/README.md), abbiamo introdotto il concetto di parchimetri intelligenti per generare entrate per una città basandosi sulla comprensione di domanda e offerta. Questo articolo discute in dettaglio come clustering, regressione e previsione delle serie temporali si combinano per aiutare a prevedere il consumo energetico futuro in Irlanda, basandosi sui contatori intelligenti.  
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Assicurazioni

Il settore assicurativo è un altro settore che utilizza il ML per costruire e ottimizzare modelli finanziari e attuariali validi.

### Gestione della volatilità

MetLife, un fornitore di assicurazioni sulla vita, è trasparente nel modo in cui analizza e mitiga la volatilità nei propri modelli finanziari. In questo articolo noterai visualizzazioni di classificazione binaria e ordinale. Scoprirai anche visualizzazioni di previsione.  
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Arte, Cultura e Letteratura

Nelle arti, ad esempio nel giornalismo, ci sono molti problemi interessanti. Rilevare le fake news è un problema enorme poiché è stato dimostrato che influenzano l'opinione delle persone e persino destabilizzano le democrazie. Anche i musei possono beneficiare dell'uso del ML in tutto, dalla ricerca di collegamenti tra manufatti alla pianificazione delle risorse.

### Rilevamento delle fake news

Rilevare le fake news è diventato un gioco del gatto e del topo nei media di oggi. In questo articolo, i ricercatori suggeriscono che un sistema che combina diverse tecniche di ML che abbiamo studiato può essere testato e il miglior modello implementato: "Questo sistema si basa sull'elaborazione del linguaggio naturale per estrarre caratteristiche dai dati e quindi queste caratteristiche vengono utilizzate per l'addestramento di classificatori di machine learning come Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) e Logistic Regression (LR)."  
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Questo articolo mostra come combinare diversi domini del ML possa produrre risultati interessanti che possono aiutare a fermare la diffusione delle fake news e prevenire danni reali; in questo caso, l'impulso è stato la diffusione di voci sui trattamenti per il COVID che hanno incitato alla violenza di massa.

### ML nei musei

I musei sono all'inizio di una rivoluzione dell'IA in cui catalogare e digitalizzare le collezioni e trovare collegamenti tra i manufatti sta diventando più facile con l'avanzare della tecnologia. Progetti come [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) stanno aiutando a svelare i misteri di collezioni inaccessibili come gli Archivi Vaticani. Ma anche l'aspetto commerciale dei musei beneficia dei modelli di ML.

Ad esempio, l'Art Institute of Chicago ha costruito modelli per prevedere cosa interessa al pubblico e quando visiterà le esposizioni. L'obiettivo è creare esperienze di visita individualizzate e ottimizzate ogni volta che l'utente visita il museo. "Durante l'anno fiscale 2017, il modello ha previsto la partecipazione e gli ingressi con un'accuratezza dell'1%, afferma Andrew Simnick, vicepresidente senior dell'Art Institute."  
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentazione dei clienti

Le strategie di marketing più efficaci mirano ai clienti in modi diversi in base a vari raggruppamenti. In questo articolo, vengono discussi gli usi degli algoritmi di clustering per supportare il marketing differenziato. Il marketing differenziato aiuta le aziende a migliorare il riconoscimento del marchio, raggiungere più clienti e guadagnare di più.  
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Sfida
Identifica un altro settore che trae vantaggio da alcune delle tecniche che hai appreso in questo curriculum e scopri come utilizza il ML.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/)

## Revisione e Studio Autonomo

Il team di data science di Wayfair ha diversi video interessanti su come utilizzano il ML nella loro azienda. Vale la pena [dare un'occhiata](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Compito

[Una caccia al tesoro sul ML](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.