# Compiti e tecniche comuni di elaborazione del linguaggio naturale

Per la maggior parte delle attività di *elaborazione del linguaggio naturale* , il testo da elaborare deve essere suddiviso, esaminato e i risultati archiviati o incrociati con regole e insiemi di dati. Queste attività consentono al programmatore di derivare il _significato_ o l'_intento_ o solo la _frequenza_ di termini e parole in un testo.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/33/?loc=it)

Si esaminano le comuni tecniche utilizzate nell'elaborazione del testo. Combinate con machine learning, queste tecniche aiutano ad analizzare grandi quantità di testo in modo efficiente. Prima di applicare machine learning a queste attività, tuttavia, occorre cercare di comprendere i problemi incontrati da uno specialista in NLP.

## Compiti comuni per NLP

Esistono diversi modi per analizzare un testo su cui si sta lavorando. Ci sono attività che si possono eseguire e attraverso le quali si è in grado di valutare la comprensione del testo e trarre conclusioni. Di solito si eseguono queste attività in sequenza.

### Tokenizzazione

Probabilmente la prima cosa che la maggior parte degli algoritmi di NLP deve fare è dividere il testo in token o parole. Anche se questo sembra semplice, dover tenere conto della punteggiatura e dei delimitatori di parole e frasi di lingue diverse può renderlo complicato. Potrebbe essere necessario utilizzare vari metodi per determinare le demarcazioni.

![Tokenizzazione](../images/tokenization.png)
> Tokenizzazione di una frase da **Orgoglio e Pregiudizio**. Infografica di [Jen Looper](https://twitter.com/jenlooper)

### Embedding

I [word embeddings](https://it.wikipedia.org/wiki/Word_embedding) sono un modo per convertire numericamente i dati di testo. Gli embedding vengono eseguiti in modo tale che le parole con un significato simile o le parole usate insieme vengano raggruppate insieme.

![word embeddings](../images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Incorporazioni di parole per una frase in **Orgoglio e Pregiudizio**. Infografica di [Jen Looper](https://twitter.com/jenlooper)

✅ Provare [questo interessante strumento](https://projector.tensorflow.org/) per sperimentare i word embedding. Facendo clic su una parola vengono visualizzati gruppi di parole simili: gruppi di "toy" con "disney", "lego", "playstation" e "console".

### Analisi e codifica di parti del discorso

Ogni parola che è stata tokenizzata può essere etichettata come parte del discorso: un sostantivo, un verbo o un aggettivo. La frase `the quick red fox jumped over the lazy brown dog` potrebbe essere etichettata come fox = sostantivo, jumped = verbo.

![elaborazione](../images/parse.png)

> Analisi di una frase da **Orgoglio e Pregiudizio**. Infografica di [Jen Looper](https://twitter.com/jenlooper)

L'analisi consiste nel riconoscere quali parole sono correlate tra loro in una frase - per esempio `the quick red fox jumped` è una sequenza aggettivo-sostantivo-verbo che è separata dalla sequenza `lazy brown dog` .

### Frequenze di parole e frasi

Una procedura utile quando si analizza un corpo di testo di grandi dimensioni è creare un dizionario di ogni parola o frase di interesse e con quale frequenza viene visualizzata. La frase `the quick red fox jumped over the lazy brown dog` ha una frequenza di parole di 2 per the.

Si esamina un testo di esempio in cui si conta la frequenza delle parole. La poesia di Rudyard Kipling The Winners contiene i seguenti versi:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Poiché le frequenze delle frasi possono essere o meno insensibili alle maiuscole o alle maiuscole, a seconda di quanto richiesto, la frase `a friend` ha una frequenza di 2, `the` ha una frequenza di 6 e `travels` è 2.

### N-grammi

Un testo può essere suddiviso in sequenze di parole di una lunghezza prestabilita, una parola singola (unigramma), due parole (bigrammi), tre parole (trigrammi) o un numero qualsiasi di parole (n-grammi).

Ad esempio, `the quick red fox jumped over the lazy brown dog` con un punteggio n-grammo di 2 produce i seguenti n-grammi:

1. the quick
2. quick red
3. red fox
4. fox jumped
5. jumped over
6. over the
7. the lazy
8. lazy brown
9. brown dog

Potrebbe essere più facile visualizzarlo come una casella scorrevole per la frase. Qui è per n-grammi di 3 parole, l'n-grammo è in grassetto in ogni frase:

1. **the quick red** fox jumped over the lazy brown dog
2. the **quick red fox** jumped over the lazy brown dog
3. the quick **red fox jumped** over the lazy brown dog
4. the quick red **fox jumped over** the lazy brown dog
5. the quick red fox **jumped over the** lazy brown dog
6. the quick red fox jumped **over the lazy** brown dog
7. the quick red fox jumped over **the lazy brown** dog
8. the quick red fox jumped over the **lazy brown dog**

![finestra scorrevole n-grammi](../images/n-grams.gif)

> Valore N-gram di 3: Infografica di [Jen Looper](https://twitter.com/jenlooper)

### Estrazione frase nominale

Nella maggior parte delle frasi, c'è un sostantivo che è il soggetto o l'oggetto della frase. In inglese, è spesso identificabile con "a" o "an" o "the" che lo precede. Identificare il soggetto o l'oggetto di una frase "estraendo la frase nominale" è un compito comune in NLP quando si cerca di capire il significato di una frase.

✅ Nella frase "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", si possono identificare i nomi nelle frasi?

Nella frase `the quick red fox jumped over the lazy brown dog` ci sono 2 frasi nominali: **quick red fox** e **lazy brown dog**.

### Analisi del sentiment

Una frase o un testo può essere analizzato per il sentimento, o quanto *positivo* o *negativo* esso sia. Il sentimento si misura in *polarità* e *oggettività/soggettività*. La polarità è misurata da -1,0 a 1,0 (da negativo a positivo) e da 0,0 a 1,0 (dal più oggettivo al più soggettivo).

✅ In seguito si imparerà che ci sono diversi modi per determinare il sentimento usando machine learning ma un modo è avere un elenco di parole e frasi che sono classificate come positive o negative da un esperto umano e applicare quel modello al testo per calcolare un punteggio di polarità. Si riesce a vedere come funzionerebbe in alcune circostanze e meno bene in altre?

### Inflessione

L'inflessione consente di prendere una parola e ottenere il singolare o il plurale della parola.

### Lemmatizzazione

Un *lemma* è la radice o il lemma per un insieme di parole, ad esempio *volò*, *vola*, *volando* ha un lemma del verbo *volare*.

Ci sono anche utili database disponibili per il ricercatore NPL, in particolare:

### WordNet

[WordNet](https://wordnet.princeton.edu/) è un database di parole, sinonimi, contari e molti altri dettagli per ogni parola in molte lingue diverse. È incredibilmente utile quando si tenta di costruire traduzioni, correttori ortografici o strumenti di lingua di qualsiasi tipo.

## Librerie NPL

Fortunatamente, non è necessario creare tutte queste tecniche da soli, poiché sono disponibili eccellenti librerie Python che le rendono molto più accessibili agli sviluppatori che non sono specializzati nell'elaborazione del linguaggio naturale o in machine learning. Le prossime lezioni includono altri esempi di queste, ma qui si impareranno alcuni esempi utili che aiuteranno con il prossimo compito.

### Esercizio: utilizzo della libreria `TextBlob`

Si usa una libreria chiamata TextBlob in quanto contiene API utili per affrontare questi tipi di attività. TextBlob "sta sulle spalle giganti di [NLTK](https://nltk.org) e [pattern](https://github.com/clips/pattern), e si sposa bene con entrambi". Ha una notevole quantità di ML incorporato nella sua API.

> Nota: per TextBlob è disponibile un'utile [guida rapida](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart), consigliata per sviluppatori Python esperti

Quando si tenta di identificare *le frasi nominali*, TextBlob offre diverse opzioni di estrattori per trovarle.

1. Dare un'occhiata a `ConllExtractor`.

   ```python
   from textblob import TextBlob
   from textblob.np_extractors import ConllExtractor
   # importa e crea un extrattore Conll da usare successivamente
   extractor = ConllExtractor()

   # quando serve un estrattore di frasi nominali:
   user_input = input("> ")
   user_input_blob = TextBlob(user_input, np_extractor=extractor)  # notare specificato estrattore non predefinito
   np = user_input_blob.noun_phrases                                    
   ```

   > Cosa sta succedendo qui? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) è "Un estrattore di frasi nominali che utilizza l'analisi dei blocchi addestrata con il corpus di formazione ConLL-2000". ConLL-2000 si riferisce alla Conferenza del 2000 sull'apprendimento computazionale del linguaggio naturale. Ogni anno la conferenza ha ospitato un workshop per affrontare uno spinoso problema della NPL, e nel 2000 è stato lo spezzettamento dei sostantivi. Un modello è stato addestrato sul Wall Street Journal, con "sezioni 15-18 come dati di addestramento (211727 token) e sezione 20 come dati di test (47377 token)". Si possono guardare le procedure utilizzate [qui](https://www.clips.uantwerpen.be/conll2000/chunking/) e i [risultati](https://ifarm.nl/erikt/research/np-chunking.html).

### Sfida: migliorare il bot con NPL

Nella lezione precedente si è creato un bot di domande e risposte molto semplice. Ora si renderà Marvin un po' più comprensivo analizzando l'input per il sentimento e stampando una risposta che corrisponda al sentimento. Si dovrà anche identificare una frase nominale `noun_phrase` e chiedere informazioni su di essa.

I passaggi durante la creazione di un bot conversazionale:

1. Stampare le istruzioni che consigliano all'utente come interagire con il bot
2. Avviare il ciclo
   1. Accettare l'input dell'utente
   2. Se l'utente ha chiesto di uscire, allora si esce
   3. Elaborare l'input dell'utente e determinare la risposta di sentimento appropriata
   4. Se viene rilevata una frase nominale nel sentimento, pluralizzala e chiedere ulteriori input su quell'argomento
   5. Stampare la risposta
3. Riprendere il ciclo dal passo 2

Ecco il frammento di codice per determinare il sentimento usando TextBlob. Si noti che ci sono solo quattro *gradienti* di risposta al sentimento (se ne potrebbero avere di più se lo si desidera):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "  # Oh caro, è terribile
elif user_input_blob.polarity <= 0: 
  response = "Hmm, that's not great. "  # Mmm, non è eccezionale
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "  # Bene, questo è positivo
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "  # Wow, sembra eccezionale
```

Ecco un risultato di esempio a scopo di guida (l'input utente è sulle righe che iniziano per >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Una possibile soluzione al compito è [qui](../solution/bot.py)

Verifica delle conoscenze

1. Si ritiene che le risposte casuali "ingannerebbero" qualcuno facendogli pensare che il bot le abbia effettivamente capite?
2. Identificare la frase nominale rende il bot più 'credibile'?
3. Perché estrarre una "frase nominale" da una frase sarebbe una cosa utile da fare?

---

Implementare il bot nel controllo delle conoscenze precedenti e testarlo su un amico. Può ingannarlo? Si può rendere il bot più 'credibile?'

## 🚀 Sfida

Prendere un'attività dalla verifica delle conoscenze qui sopra e provare a implementarla. Provare il bot su un amico. Può ingannarlo? Si può rendere il bot più 'credibile?'

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/34/?loc=it)

## Revisione e Auto Apprendimento

Nelle prossime lezioni si imparerà di più sull'analisi del sentiment. Fare ricerche su questa interessante tecnica in articoli come questi su [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Compito

[Fare rispondere un bot](assignment.it.md)
