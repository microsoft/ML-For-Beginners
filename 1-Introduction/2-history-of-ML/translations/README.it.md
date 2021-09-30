# Storia di machine learning

![Riepilogo della storia di machine learning in uno sketchnote](../../../sketchnotes/ml-history.png)
> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz pre-lezione](https://white-water-09ec41f0f.azurestaticapps.net/quiz/3/?loc=it)

In questa lezione, si camminerà attraverso le principali pietre miliari nella storia di machine learning e dell'intelligenza artificiale.

La storia dell'intelligenza artificiale, AI, come campo è intrecciata con la storia di machine learning, poiché gli algoritmi e i progressi computazionali alla base di machine learning hanno contribuito allo sviluppo dell'intelligenza artificiale. È utile ricordare che, mentre questi campi come distinte aree di indagine hanno cominciato a cristallizzarsi negli anni '50, importanti [scoperte algoritmiche, statistiche, matematiche, computazionali e tecniche](https://wikipedia.org/wiki/Timeline_of_machine_learning) hanno preceduto e si sono sovrapposte a questa era. In effetti, le persone hanno riflettuto su queste domande per [centinaia di anni](https://wikipedia.org/wiki/History_of_artificial_intelligence); questo articolo discute le basi intellettuali storiche dell'idea di una "macchina pensante".

## Scoperte rilevanti

- 1763, 1812 [Teorema di Bayes](https://it.wikipedia.org/wiki/Teorema_di_Bayes) e suoi predecessori. Questo teorema e le sue applicazioni sono alla base dell'inferenza, descrivendo la probabilità che un evento si verifichi in base alla conoscenza precedente.
- 1805 [Metodo dei Minimi Quadrati](https://it.wikipedia.org/wiki/Metodo_dei_minimi_quadrati) del matematico francese Adrien-Marie Legendre. Questa teoria, che verrà trattata nell'unità Regressione, aiuta nell'adattamento dei dati.
- 1913 [Processo Markoviano](https://it.wikipedia.org/wiki/Processo_markoviano) dal nome del matematico russo Andrey Markov è usato per descrivere una sequenza di possibili eventi basati su uno stato precedente.
- 1957 [Percettrone](https://it.wikipedia.org/wiki/Percettrone) è un tipo di classificatore lineare inventato dallo psicologo americano Frank Rosenblatt che sta alla base dei progressi nel deep learning.
- 1967 [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor) è un algoritmo originariamente progettato per mappare i percorsi. In un contesto ML viene utilizzato per rilevare i modelli.
- 1970 [La Retropropagazione dell'Errore](https://it.wikipedia.org/wiki/Retropropagazione_dell'errore) viene utilizzata per addestrare [le reti neurali feed-forward](https://it.wikipedia.org/wiki/Rete_neurale_feed-forward).
- Le [Reti Neurali Ricorrenti](https://it.wikipedia.org/wiki/Rete_neurale_ricorrente) del 1982 sono reti neurali artificiali derivate da reti neurali feedforward che creano grafici temporali.

✅ Fare una piccola ricerca. Quali altre date si distinguono come fondamentali nella storia del machine learning e dell'intelligenza artificiale?
## 1950: Macchine che pensano

Alan Turing, una persona davvero notevole che è stata votata [dal pubblico nel 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) come il più grande scienziato del XX secolo, è accreditato per aver contribuito a gettare le basi per il concetto di "macchina in grado di pensare". Ha affrontato gli oppositori e il suo stesso bisogno di prove empiriche di questo concetto in parte creando il [Test di Turing](https://www.bbc.com/news/technology-18475646), che verrà esplorato nelle lezioni di NLP (elaborazione del linguaggio naturale).

## 1956: Progetto di Ricerca Estivo Dartmouth

"Il Dartmouth Summer Research Project sull'intelligenza artificiale è stato un evento seminale per l'intelligenza artificiale come campo", qui è stato coniato il termine "intelligenza artificiale" ([fonte](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> In linea di principio, ogni aspetto dell'apprendimento o qualsiasi altra caratteristica dell'intelligenza può essere descritto in modo così preciso che si può costruire una macchina per simularlo.

Il ricercatore capo, il professore di matematica John McCarthy, sperava "di procedere sulla base della congettura che ogni aspetto dell'apprendimento o qualsiasi altra caratteristica dell'intelligenza possa in linea di principio essere descritta in modo così preciso che si possa costruire una macchina per simularlo". I partecipanti includevano un altro luminare nel campo, Marvin Minsky.

Il workshop è accreditato di aver avviato e incoraggiato diverse discussioni tra cui "l'ascesa di metodi simbolici, sistemi focalizzati su domini limitati (primi sistemi esperti) e sistemi deduttivi contro sistemi induttivi". ([fonte](https://wikipedia.org/wiki/Dartmouth_workshop)).

## 1956 - 1974: "Gli anni d'oro"

Dagli anni '50 fino alla metà degli anni '70, l'ottimismo era alto nella speranza che l'AI potesse risolvere molti problemi. Nel 1967, Marvin Minsky dichiarò con sicurezza che "Entro una generazione... il problema della creazione di 'intelligenza artificiale' sarà sostanzialmente risolto". (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

La ricerca sull'elaborazione del linguaggio naturale è fiorita, la ricerca è stata perfezionata e resa più potente ed è stato creato il concetto di "micro-mondi", in cui compiti semplici sono stati completati utilizzando istruzioni in linguaggio semplice.

La ricerca è stata ben finanziata dalle agenzie governative, sono stati fatti progressi nel calcolo e negli algoritmi e sono stati costruiti prototipi di macchine intelligenti. Alcune di queste macchine includono:

* [Shakey il robot](https://wikipedia.org/wiki/Shakey_the_robot), che poteva manovrare e decidere come eseguire i compiti "intelligentemente".

   ![Shakey, un robot intelligente](../images/shakey.jpg)
   > Shakey nel 1972

* Eliza, una delle prime "chatterbot", poteva conversare con le persone e agire come una "terapeuta" primitiva. Si Imparerà di più su Eliza nelle lezioni di NLP.

   ![Eliza, un bot](../images/eliza.png)
   > Una versione di Eliza, un chatbot

* Il "mondo dei blocchi" era un esempio di un micromondo in cui i blocchi potevano essere impilati e ordinati e si potevano testare esperimenti su macchine per insegnare a prendere decisioni. I progressi realizzati con librerie come [SHRDLU](https://it.wikipedia.org/wiki/SHRDLU) hanno contribuito a far progredire l'elaborazione del linguaggio.

    [![Il mondo dei blocchi con SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "Il mondo dei blocchi con SHRDLU")

   > 🎥 Fare clic sull'immagine sopra per un video: Blocks world con SHRDLU

## 1974 - 1980: "L'inverno dell'AI"

Verso la metà degli anni '70, era diventato evidente che la complessità della creazione di "macchine intelligenti" era stata sottovalutata e che la sua promessa, data la potenza di calcolo disponibile, era stata esagerata. I finanziamenti si sono prosciugati e la fiducia nel settore è rallentata. Alcuni problemi che hanno influito sulla fiducia includono:

- **Limitazioni**. La potenza di calcolo era troppo limitata.
- **Esplosione combinatoria**. La quantità di parametri necessari per essere addestrati è cresciuta in modo esponenziale man mano che veniva chiesto di più ai computer, senza un'evoluzione parallela della potenza e delle capacità di calcolo.
- **Scarsità di dati**. C'era una scarsità di dati che ostacolava il processo di test, sviluppo e perfezionamento degli algoritmi.
- **Stiamo facendo le domande giuste?**. Le stesse domande che venivano poste cominciarono ad essere messe in discussione. I ricercatori hanno iniziato a criticare i loro approcci:
   - I test di Turing furono messi in discussione attraverso, tra le altre idee, la "teoria della stanza cinese" che postulava che "la programmazione di un computer digitale può far sembrare che capisca il linguaggio ma non potrebbe produrre una vera comprensione". ([fonte](https://plato.stanford.edu/entries/chinese-room/))
   - L'etica dell'introduzione di intelligenze artificiali come la "terapeuta" ELIZA nella società è stata messa in discussione.

Allo stesso tempo, iniziarono a formarsi varie scuole di pensiero sull'AI. È stata stabilita una dicotomia tra pratiche ["scruffy" contro "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies). I laboratori _scruffy_ ottimizzavano i programmi per ore fino a quando non ottenevano i risultati desiderati. I laboratori _Neat_ "si focalizzavano sulla logica e sulla risoluzione formale dei problemi". ELIZA e SHRDLU erano ben noti _sistemi scruffy_. Negli anni '80, quando è emersa la richiesta di rendere riproducibili i sistemi ML, l'_approccio neat_ ha gradualmente preso il sopravvento in quanto i suoi risultati sono più spiegabili.

## Sistemi esperti degli anni '80

Man mano che il settore cresceva, i suoi vantaggi per le imprese diventavano più chiari e negli anni '80 lo stesso accadeva con la proliferazione di "sistemi esperti". "I sistemi esperti sono stati tra le prime forme di software di intelligenza artificiale (AI) di vero successo". ([fonte](https://wikipedia.org/wiki/Expert_system)).

Questo tipo di sistema è in realtà _ibrido_, costituito in parte da un motore di regole che definisce i requisiti aziendali e un motore di inferenza che sfrutta il sistema di regole per dedurre nuovi fatti.

Questa era ha visto anche una crescente attenzione rivolta alle reti neurali.

## 1987 - 1993: AI 'Chill'

La proliferazione di hardware specializzato per sistemi esperti ha avuto lo sfortunato effetto di diventare troppo specializzato. L'ascesa dei personal computer ha anche gareggiato con questi grandi sistemi centralizzati specializzati. La democratizzazione dell'informatica era iniziata e alla fine ha spianato la strada alla moderna esplosione dei big data.

## 1993 - 2011

Questa epoca ha visto una nuova era per ML e AI per essere in grado di risolvere alcuni dei problemi che erano stati causati in precedenza dalla mancanza di dati e potenza di calcolo. La quantità di dati ha iniziato ad aumentare rapidamente e a diventare più ampiamente disponibile, nel bene e nel male, soprattutto con l'avvento degli smartphone intorno al 2007. La potenza di calcolo si è ampliata in modo esponenziale e gli algoritmi si sono evoluti di pari passo. Il campo ha iniziato a maturare quando i giorni a ruota libera del passato hanno iniziato a cristallizzarsi in una vera disciplina.

## Adesso

Oggi, machine learning e intelligenza artificiale toccano quasi ogni parte della nostra vita. Questa era richiede un'attenta comprensione dei rischi e dei potenziali effetti di questi algoritmi sulle vite umane. Come ha affermato Brad Smith di Microsoft, "La tecnologia dell'informazione solleva questioni che vanno al cuore delle protezioni fondamentali dei diritti umani come la privacy e la libertà di espressione. Questi problemi aumentano la responsabilità delle aziende tecnologiche che creano questi prodotti. A nostro avviso, richiedono anche un'attenta regolamentazione del governo e lo sviluppo di norme sugli usi accettabili" ([fonte](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

Resta da vedere cosa riserva il futuro, ma è importante capire questi sistemi informatici e il software e gli algoritmi che eseguono. Ci si augura che questo programma di studi aiuti ad acquisire una migliore comprensione in modo che si possa decidere in autonomia.

[![La storia del deeplearningLa](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 " storia del deep learning")
> 🎥 Fare clic sull'immagine sopra per un video: Yann LeCun discute la storia del deep learning in questa lezione

---

## 🚀 Sfida

Approfondire uno di questi momenti storici e scoprire
 di più sulle persone che stanno dietro ad essi. Ci sono personaggi affascinanti e nessuna scoperta scientifica è mai stata creata in un vuoto culturale. Cosa si è scoperto?

## [Quiz post-lezione](https://white-water-09ec41f0f.azurestaticapps.net/quiz/4/?loc=it)

## Revisione e Auto Apprendimento

Ecco gli elementi da guardare e ascoltare:

[Questo podcast in cui Amy Boyd discute l'evoluzione dell'AI](http://runasradio.com/Shows/Show/739)

[![La storia dell'AI di Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "La storia dell'AI di Amy Boyd")

## Compito

[Creare una sequenza temporale](assignment.it.md)
