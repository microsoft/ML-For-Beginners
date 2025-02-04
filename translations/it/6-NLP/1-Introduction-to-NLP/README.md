# Introduzione all'elaborazione del linguaggio naturale

Questa lezione copre una breve storia e i concetti importanti dell'*elaborazione del linguaggio naturale*, un sotto-campo della *linguistica computazionale*.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Introduzione

L'NLP, come è comunemente conosciuto, è una delle aree più note in cui il machine learning è stato applicato e utilizzato nel software di produzione.

✅ Riesci a pensare a un software che usi ogni giorno che probabilmente ha dell'NLP integrato? Che dire dei tuoi programmi di elaborazione testi o delle app mobili che usi regolarmente?

Imparerai a conoscere:

- **L'idea delle lingue**. Come si sono sviluppate le lingue e quali sono state le principali aree di studio.
- **Definizione e concetti**. Imparerai anche definizioni e concetti su come i computer elaborano il testo, inclusi il parsing, la grammatica e l'identificazione di nomi e verbi. Ci sono alcuni compiti di codifica in questa lezione e vengono introdotti diversi concetti importanti che imparerai a programmare nelle prossime lezioni.

## Linguistica computazionale

La linguistica computazionale è un'area di ricerca e sviluppo che studia da decenni come i computer possono lavorare con le lingue, comprenderle, tradurle e comunicare con esse. L'elaborazione del linguaggio naturale (NLP) è un campo correlato che si concentra su come i computer possono elaborare le lingue 'naturali', o umane.

### Esempio - dettatura telefonica

Se hai mai dettato al tuo telefono invece di digitare o hai fatto una domanda a un assistente virtuale, il tuo discorso è stato convertito in una forma di testo e poi elaborato o *analizzato* dalla lingua che hai parlato. Le parole chiave rilevate sono state quindi elaborate in un formato che il telefono o l'assistente poteva comprendere e su cui poteva agire.

![comprensione](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.it.png)
> La vera comprensione linguistica è difficile! Immagine di [Jen Looper](https://twitter.com/jenlooper)

### Come è possibile questa tecnologia?

Questo è possibile perché qualcuno ha scritto un programma per computer per farlo. Alcuni decenni fa, alcuni scrittori di fantascienza prevedevano che le persone avrebbero parlato principalmente ai loro computer, e i computer avrebbero sempre capito esattamente cosa intendevano. Purtroppo, si è rivelato un problema più difficile di quanto molti immaginassero, e sebbene oggi sia un problema molto meglio compreso, ci sono sfide significative nel raggiungere una 'perfetta' elaborazione del linguaggio naturale quando si tratta di comprendere il significato di una frase. Questo è un problema particolarmente difficile quando si tratta di comprendere l'umorismo o di rilevare emozioni come il sarcasmo in una frase.

A questo punto, potresti ricordare le lezioni scolastiche in cui l'insegnante copriva le parti della grammatica in una frase. In alcuni paesi, agli studenti viene insegnata la grammatica e la linguistica come materia dedicata, ma in molti, questi argomenti sono inclusi come parte dell'apprendimento di una lingua: sia la tua prima lingua nella scuola primaria (imparare a leggere e scrivere) e forse una seconda lingua nella scuola post-primaria, o superiore. Non preoccuparti se non sei un esperto nel distinguere i nomi dai verbi o gli avverbi dagli aggettivi!

Se hai difficoltà con la differenza tra il *presente semplice* e il *presente progressivo*, non sei solo. Questo è un problema impegnativo per molte persone, anche per i madrelingua di una lingua. La buona notizia è che i computer sono davvero bravi ad applicare regole formali, e imparerai a scrivere codice che può *analizzare* una frase così bene come un essere umano. La sfida più grande che esaminerai in seguito è comprendere il *significato* e il *sentimento* di una frase.

## Prerequisiti

Per questa lezione, il prerequisito principale è essere in grado di leggere e comprendere la lingua di questa lezione. Non ci sono problemi matematici o equazioni da risolvere. Sebbene l'autore originale abbia scritto questa lezione in inglese, è anche tradotta in altre lingue, quindi potresti leggere una traduzione. Ci sono esempi in cui vengono utilizzate diverse lingue (per confrontare le diverse regole grammaticali delle diverse lingue). Questi non sono tradotti, ma il testo esplicativo sì, quindi il significato dovrebbe essere chiaro.

Per i compiti di codifica, utilizzerai Python e gli esempi utilizzano Python 3.8.

In questa sezione, avrai bisogno e utilizzerai:

- **Comprensione di Python 3**. Comprensione del linguaggio di programmazione in Python 3, questa lezione utilizza input, loop, lettura di file, array.
- **Visual Studio Code + estensione**. Utilizzeremo Visual Studio Code e la sua estensione Python. Puoi anche utilizzare un IDE Python di tua scelta.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) è una libreria semplificata per l'elaborazione del testo in Python. Segui le istruzioni sul sito di TextBlob per installarlo sul tuo sistema (installa anche i corpora, come mostrato di seguito):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Suggerimento: Puoi eseguire Python direttamente negli ambienti VS Code. Consulta i [documenti](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) per maggiori informazioni.

## Parlare con le macchine

La storia del tentativo di far comprendere ai computer il linguaggio umano risale a decenni fa, e uno dei primi scienziati a considerare l'elaborazione del linguaggio naturale è stato *Alan Turing*.

### Il 'test di Turing'

Quando Turing stava ricercando l'*intelligenza artificiale* negli anni '50, considerò se potesse essere somministrato un test conversazionale a un essere umano e a un computer (tramite corrispondenza scritta) in cui l'umano nella conversazione non fosse sicuro se stesse conversando con un altro umano o un computer.

Se, dopo una certa lunghezza della conversazione, l'umano non riusciva a determinare se le risposte provenissero da un computer o meno, allora si poteva dire che il computer stesse *pensando*?

### L'ispirazione - 'il gioco dell'imitazione'

L'idea per questo venne da un gioco di società chiamato *Il gioco dell'imitazione* in cui un interrogatore è solo in una stanza e incaricato di determinare quali delle due persone (in un'altra stanza) sono rispettivamente maschio e femmina. L'interrogatore può inviare note e deve cercare di pensare a domande in cui le risposte scritte rivelino il genere della persona misteriosa. Naturalmente, i giocatori nell'altra stanza cercano di ingannare l'interrogatore rispondendo alle domande in modo tale da fuorviare o confondere l'interrogatore, pur dando l'apparenza di rispondere onestamente.

### Sviluppare Eliza

Negli anni '60, uno scienziato del MIT chiamato *Joseph Weizenbaum* sviluppò [*Eliza*](https://wikipedia.org/wiki/ELIZA), un 'terapeuta' computerizzato che faceva domande all'umano e dava l'impressione di comprendere le loro risposte. Tuttavia, mentre Eliza poteva analizzare una frase e identificare certi costrutti grammaticali e parole chiave in modo da dare una risposta ragionevole, non si poteva dire che *comprendesse* la frase. Se a Eliza veniva presentata una frase seguendo il formato "**Io sono** <u>triste</u>" poteva riorganizzare e sostituire le parole nella frase per formare la risposta "Da quanto tempo **sei** <u>triste</u>".

Questo dava l'impressione che Eliza comprendesse l'affermazione e stesse facendo una domanda di follow-up, mentre in realtà stava cambiando il tempo verbale e aggiungendo alcune parole. Se Eliza non riusciva a identificare una parola chiave per cui aveva una risposta, dava invece una risposta casuale che poteva essere applicabile a molte affermazioni diverse. Eliza poteva essere facilmente ingannata, ad esempio se un utente scriveva "**Tu sei** una <u>bicicletta</u>" poteva rispondere con "Da quanto tempo **sono** una <u>bicicletta</u>?", invece di una risposta più ragionata.

[![Chattare con Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chattare con Eliza")

> 🎥 Clicca sull'immagine sopra per un video sul programma originale ELIZA

> Nota: Puoi leggere la descrizione originale di [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) pubblicata nel 1966 se hai un account ACM. In alternativa, leggi di Eliza su [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Esercizio - codificare un bot conversazionale di base

Un bot conversazionale, come Eliza, è un programma che sollecita input dall'utente e sembra comprendere e rispondere in modo intelligente. A differenza di Eliza, il nostro bot non avrà diverse regole che gli danno l'apparenza di avere una conversazione intelligente. Invece, il nostro bot avrà una sola abilità, quella di mantenere la conversazione con risposte casuali che potrebbero funzionare in quasi qualsiasi conversazione banale.

### Il piano

I tuoi passaggi quando costruisci un bot conversazionale:

1. Stampa istruzioni che avvisano l'utente su come interagire con il bot
2. Avvia un ciclo
   1. Accetta input dall'utente
   2. Se l'utente ha chiesto di uscire, allora esci
   3. Elabora l'input dell'utente e determina la risposta (in questo caso, la risposta è una scelta casuale da un elenco di possibili risposte generiche)
   4. Stampa la risposta
3. torna al passaggio 2

### Costruire il bot

Creiamo il bot. Inizieremo definendo alcune frasi.

1. Crea questo bot in Python con le seguenti risposte casuali:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Ecco un esempio di output per guidarti (l'input dell'utente è sulle righe che iniziano con `>`):

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

    Una possibile soluzione al compito è [qui](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Fermati e rifletti

    1. Pensi che le risposte casuali 'ingannerebbero' qualcuno facendogli credere che il bot li comprenda veramente?
    2. Quali caratteristiche avrebbe bisogno il bot per essere più efficace?
    3. Se un bot potesse davvero 'comprendere' il significato di una frase, avrebbe bisogno di 'ricordare' anche il significato delle frasi precedenti in una conversazione?

---

## 🚀Sfida

Scegli uno degli elementi "fermati e rifletti" sopra e prova a implementarlo nel codice o scrivi una soluzione su carta usando pseudocodice.

Nella prossima lezione, imparerai su numerosi altri approcci per analizzare il linguaggio naturale e il machine learning.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Revisione e studio autonomo

Dai un'occhiata alle referenze qui sotto come ulteriori opportunità di lettura.

### Referenze

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Compito

[Search for a bot](assignment.md)

**Disclaimer**:
Questo documento è stato tradotto utilizzando servizi di traduzione automatizzati basati su intelligenza artificiale. Anche se ci impegniamo per garantire l'accuratezza, si prega di essere consapevoli che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua madre dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione umana professionale. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.