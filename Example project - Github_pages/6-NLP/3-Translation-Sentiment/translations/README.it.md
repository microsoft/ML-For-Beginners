# Traduzione e analisi del sentiment con ML

Nelle lezioni precedenti si è imparato come creare un bot di base utilizzando `TextBlob`, una libreria che incorpora machine learning dietro le quinte per eseguire attività di base di NPL come l'estrazione di frasi nominali. Un'altra sfida importante nella linguistica computazionale è _la traduzione_ accurata di una frase da una lingua parlata o scritta a un'altra.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/?loc=it)

La traduzione è un problema molto difficile, aggravato dal fatto che ci sono migliaia di lingue e ognuna può avere regole grammaticali molto diverse. Un approccio consiste nel convertire le regole grammaticali formali per una lingua, come l'inglese, in una struttura non dipendente dalla lingua e quindi tradurla convertendola in un'altra lingua. Questo approccio significa che si dovrebbero eseguire i seguenti passaggi:

1. **Identificazione**. Identificare o taggare le parole nella lingua di input in sostantivi, verbi, ecc.
2. **Creare la traduzione**. Produrre una traduzione diretta di ogni parola nel formato della lingua di destinazione.

### Frase di esempio, dall'inglese all'irlandese

In inglese, la frase _I feel happy_ (sono felice) è composta da tre parole nell'ordine:

- **soggetto** (I)
- **verbo** (feel)
- **aggettivo** (happy)

Tuttavia, nella lingua irlandese, la stessa frase ha una struttura grammaticale molto diversa - emozioni come "*felice*" o "*triste*" sono espresse come se fossero *su se stessi*.

La frase inglese `I feel happy` in irlandese sarebbe `Tá athas orm`. Una traduzione *letterale* sarebbe `Happy is upon me` (felicità su di me).

Un oratore irlandese che traduce in inglese direbbe `I feel happy`, non `Happy is upon me`, perché capirebbe il significato della frase, anche se le parole e la struttura della frase sono diverse.

L'ordine formale per la frase in irlandese sono:

- **verbo** (Tá o is)
- **aggettivo** (athas, o happy)
- **soggetto** (orm, o upon me)

## Traduzione

Un programma di traduzione ingenuo potrebbe tradurre solo parole, ignorando la struttura della frase.

✅ Se si è imparato una seconda (o terza o più) lingua da adulto, si potrebbe aver iniziato pensando nella propria lingua madre, traducendo un concetto parola per parola nella propria testa nella seconda lingua, e poi pronunciando la traduzione. Questo è simile a quello che stanno facendo i programmi per computer di traduzione ingenui. È importante superare questa fase per raggiungere la fluidità!

La traduzione ingenua porta a cattive (e talvolta esilaranti) traduzioni errate: `I feel happy` si traduce letteralmente in `Mise bhraitheann athas` in irlandese. Ciò significa (letteralmente) `me feel happy` e non è una frase irlandese valida. Anche se l'inglese e l'irlandese sono lingue parlate su due isole vicine, sono lingue molto diverse con strutture grammaticali diverse.

> E' possibile guardare alcuni video sulle tradizioni linguistiche irlandesi come [questo](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Approcci di machine learning

Finora, si è imparato a conoscere l'approccio delle regole formali all'elaborazione del linguaggio naturale. Un altro approccio consiste nell'ignorare il significato delle parole e _utilizzare invece machine learning per rilevare i modelli_. Questo può funzionare nella traduzione se si ha molto testo (un *corpus*) o testi (*corpora*) sia nella lingua di origine che in quella di destinazione.

Si prenda ad esempio il caso di *Pride and Prejudice (Orgoglio* e pregiudizio),un noto romanzo inglese scritto da Jane Austen nel 1813. Se si consulta il libro in inglese e una traduzione umana del libro in *francese*, si potrebberoi rilevare frasi in uno che sono tradotte *idiomaticamente* nell'altro. Si farà fra un minuto.

Ad esempio, quando una frase inglese come `I have no money` (non ho denaro) viene tradotta letteralmente in francese, potrebbe diventare `Je n'ai pas de monnaie`. "Monnaie" è un complicato "falso affine" francese, poiché "money" e "monnaie" non sono sinonimi. Una traduzione migliore che un essere umano potrebbe fare sarebbe `Je n'ai pas d'argent`, perché trasmette meglio il significato che non si hanno soldi (piuttosto che "moneta spicciola" che è il significato di "monnaie").

![monnaie](../images/monnaie.png)

> Immagine di [Jen Looper](https://twitter.com/jenlooper)

Se un modello ML ha abbastanza traduzioni umane su cui costruire un modello, può migliorare l'accuratezza delle traduzioni identificando modelli comuni in testi che sono stati precedentemente tradotti da umani esperti parlanti di entrambe le lingue.

### Esercizio - traduzione

Si può usare `TextBlob` per tradurre le frasi. Provare la famosa prima riga di **Orgoglio e Pregiudizio**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` fa un buon lavoro con la traduzione: "C'est une vérité universalllement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!".

Si può sostenere che la traduzione di TextBlob è molto più esatta, infatti, della traduzione francese del 1932 del libro di V. Leconte e Ch. Pressoir:

"C'est une vérité universelle qu'un celibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle residence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

In questo caso, la traduzione informata da ML fa un lavoro migliore del traduttore umano che mette inutilmente parole nella bocca dell'autore originale per "chiarezza".

> Cosa sta succedendo qui? e perché TextBlob è così bravo a tradurre? Ebbene, dietro le quinte, utilizza Google translate, una sofisticata intelligenza artificiale in grado di analizzare milioni di frasi per prevedere le migliori stringhe per il compito da svolgere. Non c'è niente di manuale in corso qui e serve una connessione Internet per usare `blob.translate`.

✅ Provare altre frasi. Qual'è migliore, ML o traduzione umana? In quali casi?

## Analisi del sentiment

Un'altra area in cui l'apprendimento automatico può funzionare molto bene è l'analisi del sentiment. Un approccio non ML al sentiment consiste nell'identificare parole e frasi che sono "positive" e "negative". Quindi, dato un nuovo pezzo di testo, calcolare il valore totale delle parole positive, negative e neutre per identificare il sentimento generale.

Questo approccio è facilmente ingannabile come si potrebbe aver visto nel compito di Marvin: la frase `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` (Grande, è stata una meravigliosa perdita di tempo, sono contento che ci siamo persi su questa strada oscura) è una frase sarcastica e negativa, ma il semplice algoritmo rileva 'great' (grande), 'wonderful' (meraviglioso), 'glad' (contento) come positivo e 'waste' (spreco), 'lost' (perso) e 'dark' (oscuro) come negativo. Il sentimento generale è influenzato da queste parole contrastanti.

✅ Si rifletta un momento su come si trasmette il sarcasmo come oratori umani. L'inflessione del tono gioca un ruolo importante. Provare a dire la frase "Beh, quel film è stato fantastico" in modi diversi per scoprire come la propria voce trasmette significato.

### Approcci ML

L'approccio ML sarebbe quello di raccogliere manualmente corpi di testo negativi e positivi: tweet, recensioni di film o qualsiasi cosa in cui l'essere umano abbia assegnato un punteggio *e* un parere scritto. Quindi le tecniche di NPL possono essere applicate alle opinioni e ai punteggi, in modo che emergano modelli (ad esempio, le recensioni positive di film tendono ad avere la frase "degno di un Oscar" più delle recensioni di film negative, o le recensioni positive di ristoranti dicono "gourmet" molto più di "disgustoso").

> ⚖️ **Esempio**: se si è lavorato nell'ufficio di un politico e c'era qualche nuova legge in discussione, gli elettori potrebbero scrivere all'ufficio con e-mail a sostegno o e-mail contro la nuova legge specifica. Si supponga che si abbia il compito di leggere le e-mail e ordinarle in 2 pile, *pro* e *contro*. Se ci fossero molte e-mail, si potrebbe essere sopraffatti dal tentativo di leggerle tutte. Non sarebbe bello se un bot potesse leggerle tutte, capirle e dire a quale pila apparteneva ogni email?
>
> Un modo per raggiungere questo obiettivo è utilizzare machine learning. Si addestrerebbe il modello con una parte delle email *contro* e una parte delle email *per* . Il modello tenderebbe ad associare frasi e parole con il lato contro o il lato per, *ma non capirebbe alcun contenuto*, solo che è più probabile che alcune parole e modelli in una email appaiano in un *contro* o in un *pro*. Si potrebbe fare una prova con alcune e-mail non usate per addestrare il modello e vedere se si arriva alla stessa conclusione tratta da un umano. Quindi, una volta soddisfatti dell'accuratezza del modello, si potrebbero elaborare le email future senza doverle leggere tutte.

✅ Questo processo ricorda processi usati nelle lezioni precedenti?

## Esercizio - frasi sentimentali

Il sentimento viene misurato con una *polarità* da -1 a 1, il che significa che -1 è il sentimento più negativo e 1 è il più positivo. Il sentimento viene anche misurato con un punteggio 0 - 1 per oggettività (0) e soggettività (1).

Si dia un'altra occhiata a *Orgoglio e pregiudizio* di Jane Austen. Il testo è disponibile qui su [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). L'esempio seguente mostra un breve programma che analizza il sentimento della prima e dell'ultima frase del libro e ne mostra la polarità del sentimento e il punteggio di soggettività/oggettività.

Si dovrebbe utilizzare la libreria `TextBlob` (descritta sopra) per determinare il `sentiment` (non si deve scrivere il proprio calcolatore del sentimento) nella seguente attività.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Si dovrebbe ottenere il seguente risultato:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Sfida: controllare la polarità del sentimento

Il compito è determinare, usando la polarità del sentiment, se *Orgoglio e Pregiudizio* ha più frasi assolutamente positive di quelle assolutamente negative. Per questa attività, si può presumere che un punteggio di polarità di 1 o -1 sia rispettivamente assolutamente positivo o negativo.

**Procedura:**

1. Scaricare una [copia di Orgoglio e pregiudizio](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) dal Progetto Gutenberg come file .txt. Rimuovere i metadati all'inizio e alla fine del file, lasciando solo il testo originale
2. Aprire il file in Python ed estrare il contenuto come una stringa
3. Creare un TextBlob usando la stringa del libro
4. Analizzare ogni frase del libro in un ciclo
   1. Se la polarità è 1 o -1, memorizzare la frase in un array o in un elenco di messaggi positivi o negativi
5. Alla fine, stampare tutte le frasi positive e negative (separatamente) e il numero di ciascuna.

Ecco una [soluzione](../solution/notebook.ipynb) di esempio.

✅ Verifica delle conoscenze

1. Il sentimento si basa sulle parole usate nella frase, ma il codice *comprende* le parole?
2. Si ritiene che la polarità del sentimento sia accurata o, in altre parole, si è *d'accordo* con i punteggi?
   1. In particolare, si è d'accordo o in disaccordo con l'assoluta polarità **positiva** delle seguenti frasi?
      * What an excellent father you have, girls! (Che padre eccellente avete, ragazze!) said she, when the door was shut. (disse lei, non appena si chiuse la porta).
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” ("Il vostro esame di Mr. Darcy è finito, presumo", disse Miss Bingley; "e vi prego qual è il risultato?") “I am perfectly convinced by it that Mr. Darcy has no defect. (Sono perfettamente convinta che il signor Darcy non abbia difetti).
      * How wonderfully these sort of things occur! (Come accadono meravigliosamente questo genere di cose!).
      * I have the greatest dislike in the world to that sort of thing. (Ho la più grande antipatia del mondo per quel genere di cose).
      * Charlotte is an excellent manager, I dare say (Charlotte è un'eccellente manager, oserei dire).
      * “This is delightful indeed! (“Questo è davvero delizioso!)
      * I am so happy! (Che gioia!)
      * Your idea of the ponies is delightful. (La vostra idea dei pony è deliziosa).
   2. Le successive 3 frasi sono state valutate con un sentimento assolutamente positivo, ma a una lettura attenta, non sono frasi positive. Perché l'analisi del sentiment ha pensato che fossero frasi positive?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power. (Come sarò felice, quando il suo soggiorno a Netherfield sarà finito!» "Vorrei poter dire qualcosa per consolarti", rispose Elizabeth; “ma proprio non ci riesco).
      * If I could but see you as happy! (Se solo potessi vederti felice!)
      * Our distress, my dear Lizzy, is very great. (La nostra angoscia, mia cara Lizzy, è devvero grande).
   3. Sei d'accordo o in disaccordo con la polarità **negativa** assoluta delle seguenti frasi?
      - Everybody is disgusted with his pride. (Tutti sono disgustati dal suo orgoglio).
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful. ("Vorrei sapere come si comporta in mezzo agli estranei." "Allora sentirete, ma preparatevi a qualcosa di terribile).
      - The pause was to Elizabeth’s feelings dreadful. (La pausa fu terribile per i sentimenti di Elizabeth).
      - It would be dreadful! (Sarebbe terribile!)

✅ Qualsiasi appassionato di Jane Austen capirebbe che usa spesso i suoi libri per criticare gli aspetti più ridicoli della società inglese Regency. Elizabeth Bennett, la protagonista di *Orgoglio e pregiudizio,* è un'attenta osservatrice sociale (come l'autrice) e il suo linguaggio è spesso pesantemente sfumato. Anche Mr. Darcy (l'interesse amoroso della storia) nota l'uso giocoso e canzonatorio del linguaggio di Elizabeth: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own." ("Ho il piacere di conoscervi da abbastanza tempo per sapere quanto vi divertiate a esprimere di tanto in tanto delle opinioni che in realtà non vi appartengono")

---

## 🚀 Sfida

Si può rendere Marvin ancora migliore estraendo altre funzionalità dall'input dell'utente?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/?loc=it)

## Revisione e Auto Apprendimento

Esistono molti modi per estrarre il sentiment dal testo. Si pensi alle applicazioni aziendali che potrebbero utilizzare questa tecnica. Pensare a cosa potrebbe andare storto. Ulteriori informazioni sui sistemi sofisticati pronti per l'azienda che analizzano il sentiment come l'[analisi del testo di Azure](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Provare alcune delle frasi di Orgoglio e Pregiudizio sopra e vedere se può rilevare sfumature.

## Compito

[Licenza poetica](assignment.it.md)
