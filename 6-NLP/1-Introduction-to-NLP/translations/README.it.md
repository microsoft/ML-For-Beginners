# Introduzione all'elaborazione del linguaggio naturale

Questa lezione copre una breve storia e concetti importanti dell' *elaborazione del linguaggio naturale*, un sottocampo della *linguistica computazionale*.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/?loc=it)

## Introduzione

NLP, come è comunemente conosciuto, è una delle aree più note in cui machine learning è stato applicato e utilizzato nei software di produzione.

✅ Si riesce a pensare a un software che si usa tutti i giorni che probabilmente ha NLP incorporato? Che dire dei programmi di elaborazione testi o le app mobili che si usano regolarmente?

Si imparerà a conoscere:

- **L'idea delle lingue**. Come si sono sviluppate le lingue e quali sono state le principali aree di studio.
- **Definizione e concetti**. Si impareranno anche definizioni e concetti su come i computer elaborano il testo, inclusa l'analisi, la grammatica e l'identificazione di nomi e verbi. Ci sono alcune attività di codifica in questa lezione e vengono introdotti diversi concetti importanti che si imparerà a codificare più avanti nelle lezioni successive.

## Linguistica computazionale

La linguistica computazionale è un'area di ricerca e sviluppo che da molti decenni studia come i computer possono lavorare e persino capire, tradurre e comunicare con le lingue. L'elaborazione del linguaggio naturale (NLP) è un campo correlato incentrato su come i computer possono elaborare le lingue "naturali" o umane.

### Esempio: dettatura telefonica

Se si è mai dettato al telefono invece di digitare o posto una domanda a un assistente virtuale, il proprio discorso è stato convertito in formato testuale e quindi elaborato o *analizzato* dalla lingua con la quale si è parlato. Le parole chiave rilevate sono state quindi elaborate in un formato che il telefono o l'assistente possono comprendere e utilizzare.

![comprensione](../images/comprehension.png)
> La vera comprensione linguistica è difficile! Immagine di [Jen Looper](https://twitter.com/jenlooper)

### Come è resa possibile questa tecnologia?

Questo è possibile perché qualcuno ha scritto un programma per computer per farlo. Alcuni decenni fa, alcuni scrittori di fantascienza prevedevano che le persone avrebbero parlato principalmente con i loro computer e che i computer avrebbero sempre capito esattamente cosa intendevano. Purtroppo, si è rivelato essere un problema più difficile di quanto molti immaginavano, e sebbene sia un problema molto meglio compreso oggi, ci sono sfide significative nel raggiungere un'elaborazione del linguaggio naturale "perfetta" quando si tratta di comprendere il significato di una frase. Questo è un problema particolarmente difficile quando si tratta di comprendere l'umore o rilevare emozioni come il sarcasmo in una frase.

A questo punto, si potrebbero ricordare le lezioni scolastiche in cui l'insegnante ha coperto le parti della grammatica in una frase. In alcuni paesi, agli studenti viene insegnata la grammatica e la linguistica come materie dedicate, ma in molti questi argomenti sono inclusi nell'apprendimento di una lingua: o la prima lingua nella scuola primaria (imparare a leggere e scrivere) e forse una seconda lingua in post-primario o liceo. Non occorre preoccuparsi se non si è esperti nel distinguere i nomi dai verbi o gli avverbi dagli aggettivi!

Se si fa fatica a comprendere la differenza tra il *presente semplice* e il *presente progressivo*, non si è soli. Questa è una cosa impegnativa per molte persone, anche madrelingua di una lingua. La buona notizia è che i computer sono davvero bravi ad applicare regole formali e si imparerà a scrivere codice in grado di *analizzare* una frase così come un essere umano. La sfida più grande che si esaminerà in seguito è capire il *significato* e il *sentimento* di una frase.

## Prerequisiti

Per questa lezione, il prerequisito principale è essere in grado di leggere e comprendere la lingua di questa lezione. Non ci sono problemi di matematica o equazioni da risolvere. Sebbene l'autore originale abbia scritto questa lezione in inglese, è anche tradotta in altre lingue, quindi si potrebbe leggere una traduzione. Ci sono esempi in cui vengono utilizzati un numero di lingue diverse (per confrontare le diverse regole grammaticali di lingue diverse). Questi *non* sono tradotti, ma il testo esplicativo lo è, quindi il significato dovrebbe essere chiaro.

Per le attività di codifica, si utilizzerà Python e gli esempi utilizzano Python 3.8.

In questa sezione servirà e si utilizzerà:

- **Comprensione del linguaggio Python 3**. Questa lezione utilizza input, loop, lettura di file, array.
- **Visual Studio Code + estensione**. Si userà Visual Studio Code e la sua estensione Python. Si può anche usare un IDE Python a propria scelta.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) è una libreria di elaborazione del testo semplificata per Python. Seguire le istruzioni sul sito TextBlob per installarlo sul proprio sistema (installare anche i corpora, come mostrato di seguito):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Suggerimento: si può eseguire Python direttamente negli ambienti VS Code. Controllare la [documentazione](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) per ulteriori informazioni.

## Parlare con le macchine

La storia del tentativo di far capire ai computer il linguaggio umano risale a decenni fa e uno dei primi scienziati a considerare l'elaborazione del linguaggio naturale è stato *Alan Turing*.

### Il Test di Turing.

Quando Turing stava facendo ricerche sull'*intelligenza artificiale* negli anni '50, considerò se un test di conversazione potesse essere somministrato a un essere umano e a un computer (tramite corrispondenza digitata) in cui l'essere umano nella conversazione non era sicuro se stesse conversando con un altro umano o un computer.

Se, dopo una certa durata di conversazione, l'essere umano non è riuscito a determinare se le risposte provenivano da un computer o meno, allora si potrebbe dire che il computer *sta pensando*?

### L'ispirazione - 'il gioco dell'imitazione'

L'idea è nata da un gioco di società chiamato *The Imitation Game* in cui un interrogatore è da solo in una stanza e ha il compito di determinare quale delle due persone (in un'altra stanza) sono rispettivamente maschio e femmina. L'interrogatore può inviare note e deve cercare di pensare a domande in cui le risposte scritte rivelano il sesso della persona misteriosa. Ovviamente, i giocatori nell'altra stanza stanno cercando di ingannare l'interrogatore rispondendo alle domande in modo tale da fuorviare o confondere l'interrogatore, dando anche l'impressione di rispondere onestamente.

### Lo sviluppo di Eliza

Negli anni '60 uno scienziato del MIT chiamato *Joseph* Weizenbaum sviluppò [*Eliza*](https:/wikipedia.org/wiki/ELIZA), un "terapista" informatico che poneva domande a un umano e dava l'impressione di comprendere le loro risposte. Tuttavia, mentre Eliza poteva analizzare una frase e identificare alcuni costrutti grammaticali e parole chiave in modo da dare una risposta ragionevole, non si poteva dire *che capisse* la frase. Se a Eliza viene presentata una frase che segue il formato "**Sono** _triste_", potrebbe riorganizzare e sostituire le parole nella frase per formare la risposta "Da quanto tempo **sei** _triste_".

Questo dava l'impressione che Eliza avesse capito la frase e stesse facendo una domanda successiva, mentre in realtà stava cambiando il tempo e aggiungendo alcune parole. Se Eliza non fosse stata in grado di identificare una parola chiave per la quale aveva una risposta, avrebbe dato invece una risposta casuale che potrebbe essere applicabile a molte frasi diverse. Eliza avrebbe potuto essere facilmente ingannata, ad esempio se un utente avesse scritto "**Sei** una _bicicletta_" avrebbe potuto rispondere con "Da quanto tempo **sono** una _bicicletta_?", invece di una risposta più ragionata.

[![Chiacchierare conEliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco " Chiaccherare con Eliza")

> 🎥 Fare clic sull'immagine sopra per un video sul programma ELIZA originale

> Nota: si può leggere la descrizione originale di [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) pubblicata nel 1966 se si dispone di un account ACM. In alternativa, leggere di Eliza su [wikipedia](https://it.wikipedia.org/wiki/ELIZA_(chatterbot))

## Esercizio: codificare un bot conversazionale di base

Un bot conversazionale, come Eliza, è un programma che sollecita l'input dell'utente e sembra capire e rispondere in modo intelligente. A differenza di Eliza, questo bot non avrà diverse regole che gli danno l'impressione di avere una conversazione intelligente. Invece, il bot avrà una sola capacità, per mantenere viva la conversazione con risposte casuali che potrebbero funzionare in quasi tutte le conversazioni banali.

### Il piano

I passaggi durante la creazione di un bot conversazionale:

1. Stampare le istruzioni che consigliano all'utente come interagire con il bot
2. Iniziare un ciclo
   1. Accettare l'input dell'utente
   2. Se l'utente ha chiesto di uscire, allora si esce
   3. Elaborare l'input dell'utente e determinare la risposta (in questo caso, la risposta è una scelta casuale da un elenco di possibili risposte generiche)
   4. Stampare la risposta
3. Riprendere il ciclo dal passaggio 2

### Costruire il bot

Si crea il bot. Si inizia definendo alcune frasi.

1. Creare questo bot in Python con le seguenti risposte casuali:

   ```python
   random_responses = ["That is quite interesting, please tell me more.",
                       "I see. Do go on.",
                       "Why do you say that?",
                       "Funny weather we've been having, isn't it?",
                       "Let's change the subject.",
                       "Did you catch the game last night?"]
   ```

   Ecco un esempio di output come guida (l'input dell'utente è sulle righe che iniziano con `>`):

   ```output
   Hello, I am Marvin, the simple robot.
   You can end this conversation at any time by typing 'bye'
   After typing each answer, press 'enter'
   How are you today?
   > I am good thanks
   That is quite interesting, please tell me more.
   > today I went for a walk     
   Did you catch the game last night?
   > I did, but my team lost
   Funny weather we've been having, isn't it?
   > yes but I hope next week is better
   Let's change the subject.
   > ok, lets talk about music
   Why do you say that?
   > because I like music!
   Why do you say that?
   > bye
   It was nice talking to you, goodbye!
   ```

   Una possibile soluzione al compito è [qui](../solution/bot.py)

   ✅ Fermarsi e riflettere

   1. Si ritiene che le risposte casuali "ingannerebbero" qualcuno facendogli pensare che il bot le abbia effettivamente capite?
   2. Di quali caratteristiche avrebbe bisogno il bot per essere più efficace?
   3. Se un bot potesse davvero "capire" il significato di una frase, avrebbe bisogno di "ricordare" anche il significato delle frasi precedenti in una conversazione?

---

## 🚀 Sfida

Scegliere uno degli elementi "fermarsi e riflettere" qui sopra e provare a implementarli nel codice o scrivere una soluzione su carta usando pseudocodice.

Nella prossima lezione si impareranno una serie di altri approcci all'analisi del linguaggio naturale e dell'machine learning.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/?loc=it)

## Revisione e Auto Apprendimento

Dare un'occhiata ai riferimenti di seguito come ulteriori opportunità di lettura.

### Bibliografia

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Edizione primavera 2020), Edward N. Zalta (a cura di), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Università di Princeton "About WordNet". [WordNet](https://wordnet.princeton.edu/). Princeton University 2010.

## Compito

[Cercare un bot](assignment.it.md)
