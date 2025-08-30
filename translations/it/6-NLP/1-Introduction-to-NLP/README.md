<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "89e923cf3e8bdff9662536e8bf9516e6",
  "translation_date": "2025-08-29T22:32:33+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "it"
}
-->
# Introduzione all'elaborazione del linguaggio naturale

Questa lezione tratta una breve storia e i concetti fondamentali dell'*elaborazione del linguaggio naturale*, un sottocampo della *linguistica computazionale*.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Introduzione

L'NLP, come è comunemente noto, è uno dei settori più conosciuti in cui il machine learning è stato applicato e utilizzato in software di produzione.

✅ Riesci a pensare a un software che usi ogni giorno e che probabilmente abbia incorporato qualche funzionalità di NLP? Che dire dei tuoi programmi di elaborazione testi o delle app mobili che usi regolarmente?

Imparerai:

- **L'idea delle lingue**. Come si sono sviluppate le lingue e quali sono stati i principali ambiti di studio.
- **Definizioni e concetti**. Imparerai anche definizioni e concetti su come i computer elaborano il testo, inclusi il parsing, la grammatica e l'identificazione di nomi e verbi. In questa lezione ci sono alcuni compiti di programmazione e vengono introdotti diversi concetti importanti che imparerai a programmare nelle lezioni successive.

## Linguistica computazionale

La linguistica computazionale è un'area di ricerca e sviluppo che, per molti decenni, ha studiato come i computer possano lavorare con le lingue, comprenderle, tradurle e persino comunicare con esse. L'elaborazione del linguaggio naturale (NLP) è un campo correlato che si concentra su come i computer possano elaborare le lingue "naturali", ovvero quelle umane.

### Esempio - dettatura al telefono

Se hai mai dettato al tuo telefono invece di digitare o hai fatto una domanda a un assistente virtuale, il tuo discorso è stato convertito in forma testuale e poi elaborato o *analizzato* dalla lingua che hai parlato. Le parole chiave rilevate sono state poi trasformate in un formato che il telefono o l'assistente poteva comprendere e su cui agire.

![comprensione](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.it.png)
> La vera comprensione linguistica è difficile! Immagine di [Jen Looper](https://twitter.com/jenlooper)

### Come è resa possibile questa tecnologia?

Questo è possibile perché qualcuno ha scritto un programma per farlo. Alcuni decenni fa, alcuni scrittori di fantascienza avevano previsto che le persone avrebbero parlato principalmente ai loro computer e che i computer avrebbero sempre capito esattamente cosa intendessero. Purtroppo, si è rivelato un problema più difficile di quanto molti immaginassero e, sebbene oggi sia un problema molto meglio compreso, ci sono sfide significative nel raggiungere una comprensione "perfetta" del linguaggio naturale, soprattutto quando si tratta di comprendere il significato di una frase. Questo è particolarmente difficile quando si tratta di capire l'umorismo o rilevare emozioni come il sarcasmo in una frase.

A questo punto, potresti ricordare le lezioni scolastiche in cui l'insegnante trattava le parti della grammatica in una frase. In alcuni paesi, agli studenti viene insegnata la grammatica e la linguistica come materia dedicata, ma in molti altri questi argomenti sono inclusi nell'apprendimento di una lingua: sia la tua lingua madre nella scuola primaria (imparare a leggere e scrivere) sia forse una seconda lingua nella scuola secondaria. Non preoccuparti se non sei un esperto nel distinguere i nomi dai verbi o gli avverbi dagli aggettivi!

Se hai difficoltà a distinguere tra il *presente semplice* e il *presente progressivo*, non sei solo. Questo è un argomento difficile per molte persone, anche per i madrelingua di una lingua. La buona notizia è che i computer sono davvero bravi ad applicare regole formali, e imparerai a scrivere codice che può *analizzare* una frase bene quanto un essere umano. La sfida più grande che esaminerai in seguito è comprendere il *significato* e il *sentimento* di una frase.

## Prerequisiti

Per questa lezione, il principale prerequisito è essere in grado di leggere e comprendere la lingua di questa lezione. Non ci sono problemi matematici o equazioni da risolvere. Sebbene l'autore originale abbia scritto questa lezione in inglese, è anche tradotta in altre lingue, quindi potresti leggere una traduzione. Ci sono esempi in cui vengono utilizzate diverse lingue (per confrontare le diverse regole grammaticali delle lingue). Questi *non* sono tradotti, ma il testo esplicativo sì, quindi il significato dovrebbe essere chiaro.

Per i compiti di programmazione, utilizzerai Python e gli esempi sono basati su Python 3.8.

In questa sezione, avrai bisogno e utilizzerai:

- **Comprensione di Python 3**. Comprensione del linguaggio di programmazione Python 3, questa lezione utilizza input, cicli, lettura di file, array.
- **Visual Studio Code + estensione**. Utilizzeremo Visual Studio Code e la sua estensione Python. Puoi anche utilizzare un IDE Python a tua scelta.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) è una libreria semplificata per l'elaborazione del testo in Python. Segui le istruzioni sul sito di TextBlob per installarlo sul tuo sistema (installa anche i corpora, come mostrato di seguito):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Suggerimento: Puoi eseguire Python direttamente negli ambienti di VS Code. Consulta i [documenti](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) per ulteriori informazioni.

## Parlare con le macchine

La storia di tentare di far comprendere ai computer il linguaggio umano risale a decenni fa, e uno dei primi scienziati a considerare l'elaborazione del linguaggio naturale fu *Alan Turing*.

### Il 'test di Turing'

Quando Turing stava studiando l'*intelligenza artificiale* negli anni '50, si chiedeva se fosse possibile somministrare un test conversazionale a un essere umano e a un computer (tramite corrispondenza scritta) in cui l'essere umano nella conversazione non fosse sicuro se stesse conversando con un altro essere umano o con un computer.

Se, dopo una certa durata della conversazione, l'essere umano non riusciva a determinare se le risposte provenissero da un computer o meno, allora si poteva dire che il computer stesse *pensando*?

### L'ispirazione - 'il gioco dell'imitazione'

L'idea per questo venne da un gioco di società chiamato *Il gioco dell'imitazione*, in cui un interrogatore è da solo in una stanza e ha il compito di determinare quale delle due persone (in un'altra stanza) sia rispettivamente maschio e femmina. L'interrogatore può inviare note e deve cercare di pensare a domande in cui le risposte scritte rivelino il genere della persona misteriosa. Ovviamente, i giocatori nell'altra stanza cercano di ingannare l'interrogatore rispondendo alle domande in modo tale da fuorviare o confondere l'interrogatore, pur dando l'impressione di rispondere onestamente.

### Sviluppare Eliza

Negli anni '60, uno scienziato del MIT chiamato *Joseph Weizenbaum* sviluppò [*Eliza*](https://wikipedia.org/wiki/ELIZA), un "terapeuta" computerizzato che poneva domande all'essere umano e dava l'impressione di comprendere le sue risposte. Tuttavia, mentre Eliza poteva analizzare una frase e identificare determinati costrutti grammaticali e parole chiave per dare una risposta ragionevole, non si poteva dire che *comprendesse* la frase. Se a Eliza veniva presentata una frase con il formato "**Io sono** <u>triste</u>", poteva riorganizzare e sostituire le parole nella frase per formare la risposta "Da quanto tempo **sei** <u>triste</u>?".

Questo dava l'impressione che Eliza comprendesse l'affermazione e stesse ponendo una domanda di approfondimento, mentre in realtà stava cambiando il tempo verbale e aggiungendo alcune parole. Se Eliza non riusciva a identificare una parola chiave per cui aveva una risposta, dava invece una risposta casuale che poteva essere applicabile a molte affermazioni diverse. Eliza poteva essere facilmente ingannata, ad esempio se un utente scriveva "**Tu sei** una <u>bicicletta</u>", poteva rispondere con "Da quanto tempo **sono** una <u>bicicletta</u>?", invece di una risposta più ragionata.

[![Conversare con Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Conversare con Eliza")

> 🎥 Clicca sull'immagine sopra per un video sul programma originale ELIZA

> Nota: Puoi leggere la descrizione originale di [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) pubblicata nel 1966 se hai un account ACM. In alternativa, leggi di Eliza su [Wikipedia](https://wikipedia.org/wiki/ELIZA).

## Esercizio - programmare un bot conversazionale di base

Un bot conversazionale, come Eliza, è un programma che raccoglie input dall'utente e sembra comprendere e rispondere in modo intelligente. A differenza di Eliza, il nostro bot non avrà diverse regole che gli danno l'apparenza di avere una conversazione intelligente. Invece, il nostro bot avrà una sola abilità: mantenere la conversazione con risposte casuali che potrebbero funzionare in quasi qualsiasi conversazione banale.

### Il piano

I tuoi passaggi per costruire un bot conversazionale:

1. Stampare istruzioni che consigliano all'utente come interagire con il bot
2. Avviare un ciclo
   1. Accettare input dall'utente
   2. Se l'utente ha chiesto di uscire, allora uscire
   3. Elaborare l'input dell'utente e determinare la risposta (in questo caso, la risposta è una scelta casuale da un elenco di possibili risposte generiche)
   4. Stampare la risposta
3. Tornare al passaggio 2

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

    1. Pensi che le risposte casuali potrebbero "ingannare" qualcuno facendogli credere che il bot capisca davvero?
    2. Quali caratteristiche dovrebbe avere il bot per essere più efficace?
    3. Se un bot potesse davvero "comprendere" il significato di una frase, dovrebbe anche "ricordare" il significato delle frasi precedenti in una conversazione?

---

## 🚀Sfida

Scegli uno degli elementi "fermati e rifletti" sopra e prova a implementarlo in codice o scrivi una soluzione su carta usando pseudocodice.

Nella prossima lezione, imparerai diversi approcci per analizzare il linguaggio naturale e il machine learning.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Revisione e studio autonomo

Dai un'occhiata ai riferimenti qui sotto come opportunità di lettura aggiuntiva.

### Riferimenti

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Compito 

[Cerca un bot](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.