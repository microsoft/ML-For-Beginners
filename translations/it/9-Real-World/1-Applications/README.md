<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-06T07:31:21+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "it"
}
-->
# Postscript: Apprendimento automatico nel mondo reale

![Riassunto dell'apprendimento automatico nel mondo reale in uno sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

In questo curriculum, hai imparato molti modi per preparare i dati per l'addestramento e creare modelli di apprendimento automatico. Hai costruito una serie di modelli classici di regressione, clustering, classificazione, elaborazione del linguaggio naturale e serie temporali. Congratulazioni! Ora potresti chiederti a cosa servono... quali sono le applicazioni reali di questi modelli?

Sebbene l'interesse dell'industria si sia concentrato molto sull'IA, che di solito sfrutta il deep learning, ci sono ancora applicazioni preziose per i modelli classici di apprendimento automatico. Potresti persino utilizzare alcune di queste applicazioni oggi! In questa lezione, esplorerai come otto diversi settori e domini di competenza utilizzano questi tipi di modelli per rendere le loro applicazioni più performanti, affidabili, intelligenti e utili per gli utenti.

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanza

Il settore finanziario offre molte opportunità per l'apprendimento automatico. Molti problemi in questo ambito si prestano a essere modellati e risolti utilizzando l'ML.

### Rilevamento delle frodi con carta di credito

Abbiamo imparato [k-means clustering](../../5-Clustering/2-K-Means/README.md) in precedenza nel corso, ma come può essere utilizzato per risolvere problemi legati alle frodi con carta di credito?

Il clustering k-means è utile in una tecnica di rilevamento delle frodi con carta di credito chiamata **rilevamento degli outlier**. Gli outlier, o deviazioni nelle osservazioni su un set di dati, possono dirci se una carta di credito viene utilizzata in modo normale o se sta accadendo qualcosa di insolito. Come mostrato nel documento collegato di seguito, è possibile ordinare i dati delle carte di credito utilizzando un algoritmo di clustering k-means e assegnare ogni transazione a un cluster in base a quanto appare come outlier. Successivamente, è possibile valutare i cluster più rischiosi per distinguere le transazioni fraudolente da quelle legittime.
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestione patrimoniale

Nella gestione patrimoniale, un individuo o un'azienda gestisce investimenti per conto dei propri clienti. Il loro compito è mantenere e far crescere la ricchezza a lungo termine, quindi è essenziale scegliere investimenti che performino bene.

Un modo per valutare le prestazioni di un investimento è attraverso la regressione statistica. [La regressione lineare](../../2-Regression/1-Tools/README.md) è uno strumento prezioso per comprendere come un fondo si comporta rispetto a un benchmark. Possiamo anche dedurre se i risultati della regressione sono statisticamente significativi o quanto influenzerebbero gli investimenti di un cliente. È possibile espandere ulteriormente l'analisi utilizzando la regressione multipla, dove possono essere presi in considerazione ulteriori fattori di rischio. Per un esempio di come questo funzionerebbe per un fondo specifico, consulta il documento di seguito sulla valutazione delle prestazioni dei fondi utilizzando la regressione.
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educazione

Il settore educativo è anche un'area molto interessante dove l'ML può essere applicato. Ci sono problemi interessanti da affrontare, come rilevare imbrogli nei test o nei saggi o gestire i bias, intenzionali o meno, nel processo di correzione.

### Prevedere il comportamento degli studenti

[Coursera](https://coursera.com), un fornitore di corsi online aperti, ha un ottimo blog tecnico dove discutono molte decisioni ingegneristiche. In questo caso di studio, hanno tracciato una linea di regressione per cercare di esplorare una correlazione tra un basso punteggio NPS (Net Promoter Score) e la ritenzione o l'abbandono del corso.
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigare i bias

[Grammarly](https://grammarly.com), un assistente di scrittura che controlla errori di ortografia e grammatica, utilizza sofisticati [sistemi di elaborazione del linguaggio naturale](../../6-NLP/README.md) nei suoi prodotti. Hanno pubblicato un interessante caso di studio nel loro blog tecnico su come hanno affrontato il bias di genere nell'apprendimento automatico, di cui hai appreso nella nostra [lezione introduttiva sulla giustizia](../../1-Introduction/3-fairness/README.md).
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retail

Il settore retail può sicuramente beneficiare dell'uso dell'ML, con tutto, dalla creazione di un percorso cliente migliore alla gestione ottimale dell'inventario.

### Personalizzazione del percorso cliente

Wayfair, un'azienda che vende articoli per la casa come mobili, considera fondamentale aiutare i clienti a trovare i prodotti giusti per i loro gusti e bisogni. In questo articolo, gli ingegneri dell'azienda descrivono come utilizzano ML e NLP per "mostrare i risultati giusti ai clienti". In particolare, il loro motore di intenti di ricerca è stato costruito per utilizzare l'estrazione di entità, l'addestramento di classificatori, l'estrazione di asset e opinioni e il tagging del sentiment nelle recensioni dei clienti. Questo è un classico caso d'uso di come l'NLP funziona nel retail online.
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestione dell'inventario

Aziende innovative e agili come [StitchFix](https://stitchfix.com), un servizio di box che spedisce abbigliamento ai consumatori, si affidano molto all'ML per le raccomandazioni e la gestione dell'inventario. I loro team di stilisti lavorano insieme ai team di merchandising, infatti: "uno dei nostri data scientist ha sperimentato un algoritmo genetico e lo ha applicato all'abbigliamento per prevedere quale sarebbe stato un capo di successo che non esiste oggi. Lo abbiamo portato al team di merchandising e ora possono usarlo come strumento."
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Sanità

Il settore sanitario può sfruttare l'ML per ottimizzare le attività di ricerca e anche problemi logistici come la riammissione dei pazienti o il contenimento delle malattie.

### Gestione delle sperimentazioni cliniche

La tossicità nelle sperimentazioni cliniche è una grande preoccupazione per i produttori di farmaci. Quanta tossicità è tollerabile? In questo studio, l'analisi di vari metodi di sperimentazione clinica ha portato allo sviluppo di un nuovo approccio per prevedere le probabilità di esiti delle sperimentazioni cliniche. In particolare, sono stati in grado di utilizzare random forest per produrre un [classificatore](../../4-Classification/README.md) in grado di distinguere tra gruppi di farmaci.
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestione della riammissione ospedaliera

L'assistenza ospedaliera è costosa, soprattutto quando i pazienti devono essere riammessi. Questo documento discute di un'azienda che utilizza l'ML per prevedere il potenziale di riammissione utilizzando algoritmi di [clustering](../../5-Clustering/README.md). Questi cluster aiutano gli analisti a "scoprire gruppi di riammissioni che potrebbero condividere una causa comune".
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestione delle malattie

La recente pandemia ha messo in evidenza i modi in cui l'apprendimento automatico può aiutare a fermare la diffusione delle malattie. In questo articolo, riconoscerai l'uso di ARIMA, curve logistiche, regressione lineare e SARIMA. "Questo lavoro è un tentativo di calcolare il tasso di diffusione di questo virus e quindi di prevedere i decessi, i recuperi e i casi confermati, in modo che possa aiutarci a prepararci meglio e sopravvivere."
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecologia e tecnologia verde

La natura e l'ecologia consistono in molti sistemi sensibili dove l'interazione tra animali e natura è al centro dell'attenzione. È importante essere in grado di misurare questi sistemi accuratamente e agire in modo appropriato se accade qualcosa, come un incendio boschivo o un calo della popolazione animale.

### Gestione forestale

Hai imparato [Reinforcement Learning](../../8-Reinforcement/README.md) nelle lezioni precedenti. Può essere molto utile quando si cerca di prevedere modelli in natura. In particolare, può essere utilizzato per monitorare problemi ecologici come gli incendi boschivi e la diffusione di specie invasive. In Canada, un gruppo di ricercatori ha utilizzato il Reinforcement Learning per costruire modelli di dinamiche degli incendi boschivi basati su immagini satellitari. Utilizzando un innovativo "processo di diffusione spaziale (SSP)", hanno immaginato un incendio boschivo come "l'agente in qualsiasi cella del paesaggio". "Il set di azioni che il fuoco può intraprendere da una posizione in qualsiasi momento include la diffusione a nord, sud, est o ovest o la non diffusione."

Questo approccio inverte il solito setup RL poiché le dinamiche del corrispondente Markov Decision Process (MDP) sono una funzione nota per la diffusione immediata degli incendi boschivi." Leggi di più sugli algoritmi classici utilizzati da questo gruppo al link di seguito.
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Sensori di movimento degli animali

Sebbene il deep learning abbia creato una rivoluzione nel monitoraggio visivo dei movimenti degli animali (puoi costruire il tuo [tracker per orsi polari](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) qui), l'ML classico ha ancora un ruolo in questo compito.

I sensori per monitorare i movimenti degli animali da fattoria e l'IoT fanno uso di questo tipo di elaborazione visiva, ma tecniche ML più basilari sono utili per pre-elaborare i dati. Ad esempio, in questo documento, le posture delle pecore sono state monitorate e analizzate utilizzando vari algoritmi di classificazione. Potresti riconoscere la curva ROC a pagina 335.
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestione dell'energia

Nelle nostre lezioni su [previsione delle serie temporali](../../7-TimeSeries/README.md), abbiamo invocato il concetto di parchimetri intelligenti per generare entrate per una città basandoci sulla comprensione della domanda e dell'offerta. Questo articolo discute in dettaglio come clustering, regressione e previsione delle serie temporali si combinano per aiutare a prevedere l'uso futuro dell'energia in Irlanda, basandosi sui contatori intelligenti.
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Assicurazioni

Il settore assicurativo è un altro settore che utilizza l'ML per costruire e ottimizzare modelli finanziari e attuariali.

### Gestione della volatilità

MetLife, un fornitore di assicurazioni sulla vita, è trasparente riguardo al modo in cui analizza e mitiga la volatilità nei suoi modelli finanziari. In questo articolo noterai visualizzazioni di classificazione binaria e ordinale. Scoprirai anche visualizzazioni di previsione.
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Arte, cultura e letteratura

Nelle arti, ad esempio nel giornalismo, ci sono molti problemi interessanti. Rilevare le fake news è un problema enorme poiché è stato dimostrato che influenzano l'opinione delle persone e persino destabilizzano le democrazie. Anche i musei possono beneficiare dell'uso dell'ML in tutto, dal trovare collegamenti tra artefatti alla pianificazione delle risorse.

### Rilevamento delle fake news

Rilevare le fake news è diventato un gioco del gatto e del topo nei media di oggi. In questo articolo, i ricercatori suggeriscono che un sistema che combina diverse tecniche ML studiate può essere testato e il miglior modello implementato: "Questo sistema si basa sull'elaborazione del linguaggio naturale per estrarre caratteristiche dai dati e poi queste caratteristiche vengono utilizzate per l'addestramento di classificatori di apprendimento automatico come Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) e Logistic Regression (LR)."
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Questo articolo mostra come combinare diversi domini ML possa produrre risultati interessanti che possono aiutare a fermare la diffusione delle fake news e creare danni reali; in questo caso, l'impulso è stato la diffusione di voci sui trattamenti COVID che hanno incitato alla violenza di massa.

### ML nei musei

I musei sono all'apice di una rivoluzione dell'IA in cui catalogare e digitalizzare collezioni e trovare collegamenti tra artefatti sta diventando più facile con l'avanzare della tecnologia. Progetti come [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) stanno aiutando a svelare i misteri di collezioni inaccessibili come gli Archivi Vaticani. Ma anche l'aspetto commerciale dei musei beneficia dei modelli ML.

Ad esempio, l'Art Institute of Chicago ha costruito modelli per prevedere cosa interessa al pubblico e quando visiterà le esposizioni. L'obiettivo è creare esperienze di visita individualizzate e ottimizzate ogni volta che l'utente visita il museo. "Durante l'anno fiscale 2017, il modello ha previsto la partecipazione e le entrate con un'accuratezza dell'1 percento, afferma Andrew Simnick, vicepresidente senior presso l'Art Institute."
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentazione dei clienti

Le strategie di marketing più efficaci mirano ai clienti in modi diversi basandosi su vari raggruppamenti. In questo articolo, vengono discussi gli usi degli algoritmi di clustering per supportare il marketing differenziato. Il marketing differenziato aiuta le aziende a migliorare il riconoscimento del marchio, raggiungere più clienti e guadagnare di più.
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Sfida

Identifica un altro settore che beneficia di alcune delle tecniche che hai imparato in questo curriculum e scopri come utilizza l'ML.
## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e Studio Autonomo

Il team di data science di Wayfair ha diversi video interessanti su come utilizzano il ML nella loro azienda. Vale la pena [dare un'occhiata](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Compito

[Una caccia al tesoro sul ML](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.