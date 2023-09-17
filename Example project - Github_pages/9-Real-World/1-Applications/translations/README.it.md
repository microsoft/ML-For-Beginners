# Poscritto: Machine learning nel mondo reale

![Riepilogo di machine learning nel mondo reale in uno sketchnote](../../../sketchnotes/ml-realworld.png)
> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

In questo programma di studi si sono appresi molti modi per preparare i dati per l'addestramento e creare modelli di machine learning. Sono stati creati una serie di modelli classici di regressione, clustering, classificazione, elaborazione del linguaggio naturale e serie temporali. Congratulazioni! Ora, se ci si sta chiedendo a cosa serva tutto questo... quali sono le applicazioni del mondo reale per questi modelli?

Sebbene l'intelligenza artificiale abbia suscitato molto interesse nell'industria, che di solito sfrutta il deep learning, esistono ancora preziose applicazioni per i modelli classici di machine learning. Si potrebbero anche usare alcune di queste applicazioni oggi! In questa lezione, si esplorerà come otto diversi settori e campi relativi all'argomento utilizzano questi tipi di modelli per rendere le loro applicazioni più performanti, affidabili, intelligenti e preziose per gli utenti.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/?loc=it)

## Finanza

Il settore finanziario offre molte opportunità per machine learning. Molti problemi in quest'area si prestano ad essere modellati e risolti utilizzando machine learning.

### Rilevamento frodi con carta di credito

Si è appreso del [clustering k-means](../../../5-Clustering/2-K-Means/translations/README.it.md) in precedenza nel corso, ma come può essere utilizzato per risolvere i problemi relativi alle frodi con carta di credito?

Il clustering K-means è utile con una tecnica di rilevamento delle frodi con carta di credito chiamata **rilevamento dei valori anomali**. I valori anomali, o le deviazioni nelle osservazioni su un insieme di dati, possono svelare se una carta di credito viene utilizzata normalmente o se sta succedendo qualcosa di insolito. Come mostrato nel documento collegato di seguito, è possibile ordinare i dati della carta di credito utilizzando un algoritmo di clustering k-means e assegnare ogni transazione a un cluster in base a quanto sembra essere un valore anomalo. Quindi, si possono valutare i cluster più rischiosi per le transazioni fraudolente rispetto a quelle legittime.

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf

### Gestione patrimoniale

Nella gestione patrimoniale, un individuo o un'impresa gestisce gli investimenti per conto dei propri clienti. Il loro compito è sostenere e far crescere la ricchezza a lungo termine, quindi è essenziale scegliere investimenti che funzionino bene.

Un modo per valutare le prestazioni di un particolare investimento è attraverso la regressione statistica. La[regressione lineare](../../../2-Regression/1-Tools/translations/README.it.md) è uno strumento prezioso per capire come si comporta un fondo rispetto a un benchmark. Si può anche dedurre se i risultati della regressione sono statisticamente significativi o quanto influenzerebbero gli investimenti di un cliente. Si potrebbe anche espandere ulteriormente la propria analisi utilizzando la regressione multipla, in cui è possibile prendere in considerazione ulteriori fattori di rischio. Per un esempio di come funzionerebbe per un fondo specifico, consultare il documento di seguito sulla valutazione delle prestazioni del fondo utilizzando la regressione.

http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/

## Istruzione

Anche il settore educativo è un'area molto interessante in cui si può applicare machine learning. Ci sono problemi interessanti da affrontare come rilevare l'imbroglio nei test o nei saggi o gestire i pregiudizi nel processo di correzione, non intenzionali o meno.

### Prevedere il comportamento degli studenti

[Coursera](https://coursera.com), un fornitore di corsi aperti online, ha un fantastico blog di tecnologia in cui discutono molte decisioni ingegneristiche. In questo caso di studio, hanno tracciato una linea di regressione per cercare di esplorare qualsiasi correlazione tra un punteggio NPS (Net Promoter Score) basso e il mantenimento o l'abbandono del corso.

https://medium.com/coursera-engineering/regressione-controllata-quantificare-l'impatto-della-qualità-del-corso-sulla-ritenzione-dell'allievo-31f956bd592a

### Mitigare i pregiudizi

[Grammarly](https://grammarly.com), un assistente di scrittura che controlla gli errori di ortografia e grammatica, utilizza sofisticati [sistemi di elaborazione del linguaggio naturale](../../../6-NLP/translations/README.it.md) in tutti i suoi prodotti. Hanno pubblicato un interessante caso di studio nel loro blog tecnologico su come hanno affrontato il pregiudizio di genere nell'apprendimento automatico, di cui si si è appreso nella [lezione introduttiva sull'equità](../../../1-Introduction/3-fairness/translations/README.it.md).

https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/

## Vendita al dettaglio

Il settore della vendita al dettaglio può sicuramente trarre vantaggio dall'uso di machine learning, dalla creazione di un percorso migliore per il cliente allo stoccaggio dell'inventario in modo ottimale.

### Personalizzare il percorso del cliente

In Wayfair, un'azienda che vende articoli per la casa come i mobili, aiutare i clienti a trovare i prodotti giusti per i loro gusti e le loro esigenze è fondamentale. In questo articolo, gli ingegneri dell'azienda descrivono come utilizzano ML e NLP per "far emergere i risultati giusti per i clienti". In particolare, il loro motore di intento di ricerca è stato creato per utilizzare l'estrazione di entità, l'addestramento di classificatori, l'estrazione di risorse e opinioni e l'etichettatura del sentimento sulle recensioni dei clienti. Questo è un classico caso d'uso di come funziona NLP nella vendita al dettaglio online.

https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search

### Gestione dell’inventario

Aziende innovative e agili come [StitchFix](https://stitchfix.com), un servizio che spedisce abbigliamento ai consumatori, si affidano molto al machine learning per consigli e gestione dell'inventario. I loro team di stilisti lavorano insieme ai loro team di merchandising, infatti: "uno dei nostri data scientist ha armeggiato con un algoritmo genetico e lo ha applicato all'abbigliamento per prevedere quale sarebbe un capo di abbigliamento di successo che oggi non esiste. L'abbiamo portato al team del merchandising e ora possono usarlo come strumento".

https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/

## Assistenza sanitaria

Il settore sanitario può sfruttare il machine learning per ottimizzare le attività di ricerca e anche problemi logistici come la riammissione dei pazienti o l'arresto della diffusione delle malattie.

### Gestione delle sperimentazioni cliniche

La tossicità negli studi clinici è una delle principali preoccupazioni per i produttori di farmaci. Quanta tossicità è tollerabile? In questo studio, l'analisi di vari metodi di sperimentazione clinica ha portato allo sviluppo di un nuovo approccio per prevedere le probabilità dei risultati della sperimentazione clinica. Nello specifico, sono stati in grado di usare random forest per produrre un [classificatore](../../../4-Classification/translations/README.it.md) in grado di distinguere tra gruppi di farmaci.

https://www.sciencedirect.com/science/article/pii/S2451945616302914

### Gestione della riammissione ospedaliera

Le cure ospedaliere sono costose, soprattutto quando i pazienti devono essere ricoverati di nuovo. Questo documento discute un'azienda che utilizza il machine learning per prevedere il potenziale di riammissione utilizzando algoritmi di [clustering](../../../5-Clustering/translations/README.it.md). Questi cluster aiutano gli analisti a "scoprire gruppi di riammissioni che possono condividere una causa comune".

https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning

### Gestione della malattia

La recente pandemia ha messo in luce i modi in cui machine learning può aiutare a fermare la diffusione della malattia. In questo articolo, si riconoscerà l'uso di ARIMA, curve logistiche, regressione lineare e SARIMA. "Questo lavoro è un tentativo di calcolare il tasso di diffusione di questo virus e quindi di prevedere morti, guarigioni e casi confermati, in modo che possa aiutare a prepararci meglio e sopravvivere".

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/

## 🌲 Ecologia e Green Tech

Natura ed ecologia sono costituiti da molti sistemi sensibili in cui l'interazione tra animali e natura viene messa a fuoco. È importante essere in grado di misurare accuratamente questi sistemi e agire in modo appropriato se accade qualcosa, come un incendio boschivo o un calo della popolazione animale.

### Gestione delle foreste

Si è appreso il [Reinforcement Learning](../../../8-Reinforcement/translations/README.it.md) nelle lezioni precedenti. Può essere molto utile quando si cerca di prevedere i modelli in natura. In particolare, può essere utilizzato per monitorare problemi ecologici come gli incendi boschivi e la diffusione di specie invasive. In Canada, un gruppo di ricercatori ha utilizzato Reinforcement Learning per costruire modelli di dinamica degli incendi boschivi da immagini satellitari. Utilizzando un innovativo "processo di diffusione spaziale (SSP)", hanno immaginato un incendio boschivo come "l'agente in qualsiasi cellula del paesaggio". "L'insieme di azioni che l'incendio può intraprendere da un luogo in qualsiasi momento include la diffusione a nord, sud, est o ovest o la mancata diffusione.

Questo approccio inverte la solita configurazione RL poiché la dinamica del corrispondente Processo Decisionale di Markov (MDP) è una funzione nota per la diffusione immediata degli incendi". Maggiori informazioni sugli algoritmi classici utilizzati da questo gruppo al link sottostante.

https://www.frontiersin.org/articles/10.3389/fneur.2018.00006/pieno

### Rilevamento del movimento degli animali

Mentre il deep learning ha creato una rivoluzione nel tracciamento visivo dei movimenti degli animali (qui si può costruire il proprio [localizzatore di orsi polari](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) ), il machine learning classico ha ancora un posto in questo compito.

I sensori per tracciare i movimenti degli animali da fattoria e l'internet delle cose fanno uso di questo tipo di elaborazione visiva, ma tecniche di machine learning di base sono utili per preelaborare i dati. Ad esempio, in questo documento, le posture delle pecore sono state monitorate e analizzate utilizzando vari algoritmi di classificazione. Si potrebbe riconoscere la curva ROC a pagina 335.

https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf

### Gestione energetica

Nelle lezioni sulla [previsione delle serie temporali](../../../7-TimeSeries/translations/README.it.md), si è invocato il concetto di parchimetri intelligenti per generare entrate per una città in base alla comprensione della domanda e dell'offerta. Questo articolo discute in dettaglio come il raggruppamento, la regressione e la previsione delle serie temporali si sono combinati per aiutare a prevedere il futuro uso di energia in Irlanda, sulla base della misurazione intelligente.

https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf

## Assicurazione

Il settore assicurativo è un altro settore che utilizza machine learning per costruire e ottimizzare modelli finanziari e attuariali sostenibili.

### Gestione della volatilità

MetLife, un fornitore di assicurazioni sulla vita, è disponibile con il modo in cui analizzano e mitigano la volatilità nei loro modelli finanziari. In questo articolo si noteranno le visualizzazioni di classificazione binaria e ordinale. Si scopriranno anche visualizzazioni di previsione.

https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf

## 🎨 Arte, cultura e letteratura

Nelle arti, per esempio nel giornalismo, ci sono molti problemi interessanti. Rilevare notizie false è un problema enorme poiché è stato dimostrato che influenza l'opinione delle persone e persino che fa cadere le democrazie. I musei possono anche trarre vantaggio dall'utilizzo di machine learning in tutto, dalla ricerca di collegamenti tra gli artefatti alla pianificazione delle risorse.

### Rilevamento di notizie false

Rilevare notizie false è diventato un gioco del gatto e del topo nei media di oggi. In questo articolo, i ricercatori suggeriscono che un sistema che combina diverse delle tecniche ML qui studiate può essere testato e il miglior modello implementato: "Questo sistema si basa sull'elaborazione del linguaggio naturale per estrarre funzionalità dai dati e quindi queste funzionalità vengono utilizzate per l'addestramento di classificatori di machine learning come Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) e Logistic Regression (LR)."

https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf

Questo articolo mostra come la combinazione di diversi campi ML possa produrre risultati interessanti in grado di aiutare a impedire che le notizie false si diffondano e creino danni reali; in questo caso, l'impulso è stato la diffusione di voci su trattamenti COVID che incitavano alla violenza di massa.

### ML per Musei

I musei sono all'apice di una rivoluzione dell'intelligenza artificiale in cui catalogare e digitalizzare le collezioni e trovare collegamenti tra i manufatti sta diventando più facile con l'avanzare della tecnologia. Progetti come [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) stanno aiutando a svelare i misteri di collezioni inaccessibili come gli Archivi Vaticani. Ma anche l'aspetto commerciale dei musei beneficia dei modelli di machine learning.

Ad esempio, l'Art Institute di Chicago ha costruito modelli per prevedere a cosa è interessato il pubblico e quando parteciperà alle esposizioni. L'obiettivo è creare esperienze di visita personalizzate e ottimizzate ogni volta che l'utente visita il museo. "Durante l'anno fiscale 2017, il modello ha previsto presenze e ammissioni entro l'1% di scostamento, afferma Andrew Simnick, vicepresidente senior dell'Art Institute".

https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices

## Marketing

### Segmentazione della clientela

Le strategie di marketing più efficaci si rivolgono ai clienti in modi diversi in base a vari raggruppamenti. In questo articolo vengono discussi gli usi degli algoritmi di Clustering per supportare il marketing differenziato. Il marketing differenziato aiuta le aziende a migliorare il riconoscimento del marchio, raggiungere più clienti e guadagnare di più.

https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/

## 🚀 Sfida

Identificare un altro settore che beneficia di alcune delle tecniche apprese in questo programma di studi e scoprire come utilizza il machine learning.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/?loc=it)

## Revisione e Auto Apprendimento

Il team di data science di Wayfair ha diversi video interessanti su come usano il machine learning nella loro azienda. Vale la pena [dare un'occhiata](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Compito

[Una caccia al tesoro per ML](assignment.it.md)
