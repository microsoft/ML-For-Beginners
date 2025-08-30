<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6396d5d8617572cd2ac1de74fb0deb22",
  "translation_date": "2025-08-29T22:37:15+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "it"
}
-->
# Traduzione e analisi del sentiment con ML

Nelle lezioni precedenti hai imparato a costruire un bot di base utilizzando `TextBlob`, una libreria che incorpora il machine learning dietro le quinte per eseguire compiti NLP di base come l'estrazione di frasi nominali. Un'altra sfida importante nella linguistica computazionale è la _traduzione_ accurata di una frase da una lingua parlata o scritta a un'altra.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

La traduzione è un problema molto complesso, aggravato dal fatto che esistono migliaia di lingue, ognuna con regole grammaticali molto diverse. Un approccio consiste nel convertire le regole grammaticali formali di una lingua, come l'inglese, in una struttura indipendente dalla lingua, per poi tradurla riconvertendola in un'altra lingua. Questo approccio prevede i seguenti passaggi:

1. **Identificazione**. Identificare o etichettare le parole nella lingua di input come sostantivi, verbi, ecc.
2. **Creare la traduzione**. Produrre una traduzione diretta di ogni parola nel formato della lingua di destinazione.

### Esempio di frase, dall'inglese all'irlandese

In 'inglese', la frase _I feel happy_ è composta da tre parole nell'ordine:

- **soggetto** (I)
- **verbo** (feel)
- **aggettivo** (happy)

Tuttavia, nella lingua 'irlandese', la stessa frase ha una struttura grammaticale molto diversa: emozioni come "*felice*" o "*triste*" sono espresse come se fossero *su di te*.

La frase inglese `I feel happy` in irlandese sarebbe `Tá athas orm`. Una traduzione *letterale* sarebbe `La felicità è su di me`.

Un parlante irlandese che traduce in inglese direbbe `I feel happy`, non `Happy is upon me`, perché comprende il significato della frase, anche se le parole e la struttura della frase sono diverse.

L'ordine formale della frase in irlandese è:

- **verbo** (Tá o is)
- **aggettivo** (athas, o happy)
- **soggetto** (orm, o upon me)

## Traduzione

Un programma di traduzione ingenuo potrebbe tradurre solo le parole, ignorando la struttura della frase.

✅ Se hai imparato una seconda (o terza o più) lingua da adulto, potresti aver iniziato pensando nella tua lingua madre, traducendo un concetto parola per parola nella tua testa nella seconda lingua, e poi pronunciando la tua traduzione. Questo è simile a ciò che fanno i programmi di traduzione ingenua. È importante superare questa fase per raggiungere la fluidità!

La traduzione ingenua porta a traduzioni errate (e talvolta esilaranti): `I feel happy` si traduce letteralmente in `Mise bhraitheann athas` in irlandese. Questo significa (letteralmente) `io sento felicità` e non è una frase valida in irlandese. Anche se inglese e irlandese sono lingue parlate su due isole vicine, sono lingue molto diverse con strutture grammaticali differenti.

> Puoi guardare alcuni video sulle tradizioni linguistiche irlandesi come [questo](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Approcci di machine learning

Finora hai imparato l'approccio basato su regole formali per l'elaborazione del linguaggio naturale. Un altro approccio è ignorare il significato delle parole e _usare invece il machine learning per rilevare schemi_. Questo può funzionare nella traduzione se hai molti testi (un *corpus*) o testi (*corpora*) sia nella lingua di origine che in quella di destinazione.

Ad esempio, considera il caso di *Orgoglio e Pregiudizio*, un famoso romanzo inglese scritto da Jane Austen nel 1813. Se consulti il libro in inglese e una traduzione umana del libro in *francese*, potresti rilevare frasi in una lingua che sono tradotte _idiomaticamente_ nell'altra. Lo farai tra poco.

Ad esempio, quando una frase inglese come `I have no money` viene tradotta letteralmente in francese, potrebbe diventare `Je n'ai pas de monnaie`. "Monnaie" è un falso amico francese, poiché 'money' e 'monnaie' non sono sinonimi. Una traduzione migliore che un umano potrebbe fare sarebbe `Je n'ai pas d'argent`, perché trasmette meglio il significato che non hai soldi (piuttosto che 'spiccioli', che è il significato di 'monnaie').

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.it.png)

> Immagine di [Jen Looper](https://twitter.com/jenlooper)

Se un modello ML ha abbastanza traduzioni umane su cui costruire un modello, può migliorare l'accuratezza delle traduzioni identificando schemi comuni nei testi precedentemente tradotti da esperti parlanti umani di entrambe le lingue.

### Esercizio - traduzione

Puoi usare `TextBlob` per tradurre frasi. Prova la famosa prima riga di **Orgoglio e Pregiudizio**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` fa un ottimo lavoro nella traduzione: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Si può sostenere che la traduzione di TextBlob sia molto più precisa, in effetti, rispetto alla traduzione francese del 1932 del libro di V. Leconte e Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

In questo caso, la traduzione informata dal ML fa un lavoro migliore rispetto al traduttore umano che aggiunge inutilmente parole alla bocca dell'autore originale per 'chiarezza'.

> Cosa sta succedendo qui? E perché TextBlob è così bravo nella traduzione? Bene, dietro le quinte, sta usando Google Translate, un'IA sofisticata in grado di analizzare milioni di frasi per prevedere le stringhe migliori per il compito in questione. Non c'è nulla di manuale qui e hai bisogno di una connessione internet per usare `blob.translate`.

✅ Prova altre frasi. Quale è migliore, ML o traduzione umana? In quali casi?

## Analisi del sentiment

Un'altra area in cui il machine learning può funzionare molto bene è l'analisi del sentiment. Un approccio non basato su ML al sentiment è identificare parole e frasi che sono 'positive' e 'negative'. Poi, dato un nuovo testo, calcolare il valore totale delle parole positive, negative e neutre per identificare il sentiment complessivo. 

Questo approccio è facilmente ingannabile, come potresti aver visto nel compito di Marvin: la frase `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` è una frase sarcastica con sentiment negativo, ma l'algoritmo semplice rileva 'great', 'wonderful', 'glad' come positive e 'waste', 'lost' e 'dark' come negative. Il sentiment complessivo è influenzato da queste parole contrastanti.

✅ Fermati un attimo e pensa a come esprimiamo il sarcasmo come parlanti umani. L'inflessione del tono gioca un ruolo importante. Prova a dire la frase "Well, that film was awesome" in modi diversi per scoprire come la tua voce trasmette il significato.

### Approcci ML

L'approccio ML consisterebbe nel raccogliere manualmente testi negativi e positivi - tweet, recensioni di film o qualsiasi cosa in cui l'umano abbia dato un punteggio *e* un'opinione scritta. Poi si possono applicare tecniche NLP alle opinioni e ai punteggi, in modo che emergano schemi (ad esempio, le recensioni positive di film tendono ad avere la frase 'Oscar worthy' più delle recensioni negative, o le recensioni positive di ristoranti dicono 'gourmet' molto più di 'disgusting').

> ⚖️ **Esempio**: Se lavorassi nell'ufficio di un politico e ci fosse una nuova legge in discussione, i cittadini potrebbero scrivere all'ufficio con email a favore o contro la particolare nuova legge. Supponiamo che ti venga assegnato il compito di leggere le email e di classificarle in 2 pile, *a favore* e *contro*. Se ci fossero molte email, potresti sentirti sopraffatto nel tentativo di leggerle tutte. Non sarebbe bello se un bot potesse leggerle tutte per te, capirle e dirti in quale pila appartiene ogni email? 
> 
> Un modo per ottenere ciò è utilizzare il Machine Learning. Addestreresti il modello con una parte delle email *contro* e una parte delle email *a favore*. Il modello tenderebbe ad associare frasi e parole al lato contro e al lato a favore, *ma non comprenderebbe alcun contenuto*, solo che certe parole e schemi erano più probabili in un'email *contro* o *a favore*. Potresti testarlo con alcune email che non avevi usato per addestrare il modello e vedere se giunge alla stessa conclusione a cui sei giunto tu. Poi, una volta soddisfatto dell'accuratezza del modello, potresti elaborare email future senza doverle leggere tutte.

✅ Questo processo ti sembra simile a processi che hai usato in lezioni precedenti?

## Esercizio - frasi sentimentali

Il sentiment è misurato con una *polarità* da -1 a 1, dove -1 è il sentiment più negativo e 1 è il più positivo. Il sentiment è anche misurato con un punteggio da 0 a 1 per oggettività (0) e soggettività (1).

Dai un'altra occhiata a *Orgoglio e Pregiudizio* di Jane Austen. Il testo è disponibile qui su [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). L'esempio seguente mostra un breve programma che analizza il sentiment delle prime e ultime frasi del libro e ne visualizza la polarità del sentiment e il punteggio di soggettività/oggettività.

Dovresti usare la libreria `TextBlob` (descritta sopra) per determinare il `sentiment` (non devi scrivere il tuo calcolatore di sentiment) nel seguente compito.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vedrai il seguente output:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Sfida - controlla la polarità del sentiment

Il tuo compito è determinare, utilizzando la polarità del sentiment, se *Orgoglio e Pregiudizio* ha più frasi assolutamente positive rispetto a quelle assolutamente negative. Per questo compito, puoi assumere che un punteggio di polarità di 1 o -1 sia assolutamente positivo o negativo rispettivamente.

**Passaggi:**

1. Scarica una [copia di Orgoglio e Pregiudizio](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) da Project Gutenberg come file .txt. Rimuovi i metadati all'inizio e alla fine del file, lasciando solo il testo originale.
2. Apri il file in Python ed estrai i contenuti come stringa.
3. Crea un TextBlob utilizzando la stringa del libro.
4. Analizza ogni frase del libro in un ciclo.
   1. Se la polarità è 1 o -1, memorizza la frase in un array o lista di messaggi positivi o negativi.
5. Alla fine, stampa tutte le frasi positive e negative (separatamente) e il numero di ciascuna.

Ecco una [soluzione di esempio](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Verifica delle conoscenze

1. Il sentiment si basa sulle parole usate nella frase, ma il codice *comprende* le parole?
2. Pensi che la polarità del sentiment sia accurata, o in altre parole, sei *d'accordo* con i punteggi?
   1. In particolare, sei d'accordo o in disaccordo con la polarità assolutamente **positiva** delle seguenti frasi?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Le seguenti 3 frasi sono state valutate con un sentiment assolutamente positivo, ma a una lettura attenta, non sono frasi positive. Perché l'analisi del sentiment ha pensato che fossero frasi positive?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Sei d'accordo o in disaccordo con la polarità assolutamente **negativa** delle seguenti frasi?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Qualsiasi appassionato di Jane Austen capirà che spesso usa i suoi libri per criticare gli aspetti più ridicoli della società inglese della Reggenza. Elizabeth Bennett, il personaggio principale in *Orgoglio e Pregiudizio*, è un'acuta osservatrice sociale (come l'autrice) e il suo linguaggio è spesso fortemente sfumato. Anche Mr. Darcy (l'interesse amoroso nella storia) nota l'uso giocoso e ironico del linguaggio da parte di Elizabeth: "Ho avuto il piacere della tua conoscenza abbastanza a lungo da sapere che trovi grande divertimento nel professare occasionalmente opinioni che in realtà non sono le tue."

---

## 🚀Sfida

Puoi migliorare Marvin estraendo altre caratteristiche dall'input dell'utente?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## Revisione e studio autonomo
Ci sono molti modi per estrarre il sentimento da un testo. Pensa alle applicazioni aziendali che potrebbero utilizzare questa tecnica. Rifletti su come potrebbe andare storto. Leggi di più sui sistemi sofisticati e pronti per le imprese che analizzano il sentimento, come [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Prova alcune delle frasi di Orgoglio e Pregiudizio sopra e verifica se riesce a rilevare le sfumature.

## Compito

[Licenza poetica](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.