<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3c4738bb0836dd838c552ab9cab7e09d",
  "translation_date": "2025-08-29T22:27:31+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "it"
}
-->
# Analisi del sentiment con recensioni di hotel - elaborazione dei dati

In questa sezione utilizzerai le tecniche apprese nelle lezioni precedenti per eseguire un'analisi esplorativa dei dati su un ampio dataset. Una volta acquisita una buona comprensione dell'utilità delle varie colonne, imparerai:

- come rimuovere le colonne non necessarie
- come calcolare nuovi dati basandoti sulle colonne esistenti
- come salvare il dataset risultante per utilizzarlo nella sfida finale

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Introduzione

Finora hai imparato che i dati testuali sono molto diversi dai dati numerici. Se il testo è stato scritto o pronunciato da un essere umano, può essere analizzato per trovare schemi, frequenze, sentiment e significato. Questa lezione ti introduce a un dataset reale con una sfida concreta: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, che include una [licenza CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). I dati sono stati estratti da Booking.com da fonti pubbliche. Il creatore del dataset è Jiashen Liu.

### Preparazione

Avrai bisogno di:

* La capacità di eseguire notebook .ipynb utilizzando Python 3
* pandas
* NLTK, [che dovresti installare localmente](https://www.nltk.org/install.html)
* Il dataset disponibile su Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). È di circa 230 MB una volta decompresso. Scaricalo nella cartella principale `/data` associata a queste lezioni di NLP.

## Analisi esplorativa dei dati

Questa sfida presuppone che tu stia costruendo un bot di raccomandazione per hotel utilizzando l'analisi del sentiment e i punteggi delle recensioni degli ospiti. Il dataset che utilizzerai include recensioni di 1493 hotel in 6 città.

Utilizzando Python, un dataset di recensioni di hotel e l'analisi del sentiment di NLTK, potresti scoprire:

* Quali sono le parole e le frasi più frequentemente utilizzate nelle recensioni?
* I *tag* ufficiali che descrivono un hotel sono correlati ai punteggi delle recensioni (ad esempio, ci sono più recensioni negative per un particolare hotel da parte di *Famiglie con bambini piccoli* rispetto a *Viaggiatori solitari*, indicando forse che è più adatto ai *Viaggiatori solitari*)?
* I punteggi di sentiment di NLTK "concordano" con il punteggio numerico del recensore?

#### Dataset

Esploriamo il dataset che hai scaricato e salvato localmente. Apri il file in un editor come VS Code o anche Excel.

Le intestazioni del dataset sono le seguenti:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Ecco come possono essere raggruppate per facilitarne l'esame: 
##### Colonne relative agli hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitudine), `lng` (longitudine)
  * Utilizzando *lat* e *lng* potresti creare una mappa con Python che mostra le posizioni degli hotel (magari codificate per colore in base alle recensioni positive e negative)
  * Hotel_Address non sembra essere particolarmente utile e probabilmente lo sostituiremo con il paese per facilitare l'ordinamento e la ricerca

**Colonne meta-recensione dell'hotel**

* `Average_Score`
  * Secondo il creatore del dataset, questa colonna rappresenta il *Punteggio medio dell'hotel, calcolato in base all'ultimo commento dell'ultimo anno*. Questo metodo di calcolo sembra insolito, ma dato che i dati sono stati estratti, per ora possiamo prenderli per buoni.
  
  ✅ Basandoti sulle altre colonne di questo dataset, riesci a pensare a un altro modo per calcolare il punteggio medio?

* `Total_Number_of_Reviews`
  * Il numero totale di recensioni ricevute da questo hotel - non è chiaro (senza scrivere del codice) se si riferisca alle recensioni presenti nel dataset.
* `Additional_Number_of_Scoring`
  * Questo indica che è stato dato un punteggio ma il recensore non ha scritto una recensione positiva o negativa

**Colonne relative alle recensioni**

- `Reviewer_Score`
  - Questo è un valore numerico con al massimo 1 cifra decimale, compreso tra i valori minimi e massimi di 2.5 e 10
  - Non è spiegato perché il punteggio minimo possibile sia 2.5
- `Negative_Review`
  - Se un recensore non ha scritto nulla, questo campo conterrà "**No Negative**"
  - Nota che un recensore potrebbe scrivere una recensione positiva nella colonna delle recensioni negative (ad esempio "non c'è nulla di negativo in questo hotel")
- `Review_Total_Negative_Word_Counts`
  - Un numero maggiore di parole negative indica un punteggio più basso (senza controllare il sentiment)
- `Positive_Review`
  - Se un recensore non ha scritto nulla, questo campo conterrà "**No Positive**"
  - Nota che un recensore potrebbe scrivere una recensione negativa nella colonna delle recensioni positive (ad esempio "non c'è nulla di buono in questo hotel")
- `Review_Total_Positive_Word_Counts`
  - Un numero maggiore di parole positive indica un punteggio più alto (senza controllare il sentiment)
- `Review_Date` e `days_since_review`
  - Si potrebbe applicare una misura di freschezza o obsolescenza a una recensione (le recensioni più vecchie potrebbero non essere accurate perché la gestione dell'hotel è cambiata, sono state fatte ristrutturazioni, è stata aggiunta una piscina, ecc.)
- `Tags`
  - Questi sono brevi descrittori che un recensore può selezionare per descrivere il tipo di ospite che era (ad esempio, solitario o in famiglia), il tipo di stanza che aveva, la durata del soggiorno e come è stata inviata la recensione.
  - Sfortunatamente, l'uso di questi tag è problematico, controlla la sezione sottostante che discute la loro utilità

**Colonne relative ai recensori**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Questo potrebbe essere un fattore in un modello di raccomandazione, ad esempio, se si potesse determinare che i recensori più prolifici con centinaia di recensioni tendono ad essere più negativi che positivi. Tuttavia, il recensore di una particolare recensione non è identificato con un codice univoco e quindi non può essere collegato a un set di recensioni. Ci sono 30 recensori con 100 o più recensioni, ma è difficile vedere come questo possa aiutare il modello di raccomandazione.
- `Reviewer_Nationality`
  - Alcune persone potrebbero pensare che certe nazionalità siano più inclini a dare recensioni positive o negative a causa di un'inclinazione nazionale. Fai attenzione a costruire tali visioni aneddotiche nei tuoi modelli. Questi sono stereotipi nazionali (e talvolta razziali), e ogni recensore è un individuo che ha scritto una recensione basandosi sulla propria esperienza. Questa potrebbe essere stata filtrata attraverso molteplici lenti, come i soggiorni precedenti in hotel, la distanza percorsa e il loro temperamento personale. Pensare che la loro nazionalità sia stata la ragione di un punteggio di recensione è difficile da giustificare.

##### Esempi

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Questo non è attualmente un hotel ma un cantiere. Sono stato disturbato fin dal mattino presto e per tutto il giorno da rumori di costruzione inaccettabili mentre cercavo di riposare dopo un lungo viaggio e lavorare nella stanza. Le persone lavoravano tutto il giorno, ad esempio con martelli pneumatici nelle stanze adiacenti. Ho chiesto di cambiare stanza, ma non c'era una stanza silenziosa disponibile. Per peggiorare le cose, mi hanno addebitato un costo eccessivo. Ho lasciato l'hotel la sera poiché avevo un volo molto presto e ho ricevuto una fattura adeguata. Un giorno dopo, l'hotel ha effettuato un altro addebito senza il mio consenso, superiore al prezzo prenotato. È un posto terribile. Non punirti prenotando qui. | Nulla. Posto terribile. Stanne lontano. | Viaggio di lavoro, Coppia, Camera doppia standard, Soggiorno di 2 notti |

Come puoi vedere, questo ospite non ha avuto un soggiorno felice in questo hotel. L'hotel ha un buon punteggio medio di 7.8 e 1945 recensioni, ma questo recensore gli ha dato 2.5 e ha scritto 115 parole su quanto negativa sia stata la sua esperienza. Se non avesse scritto nulla nella colonna Positive_Review, potresti dedurre che non ci fosse nulla di positivo, ma invece ha scritto 7 parole di avvertimento. Se contassimo solo le parole invece del significato o del sentiment delle parole, potremmo avere una visione distorta dell'intento del recensore. Stranamente, il loro punteggio di 2.5 è confuso, perché se il soggiorno in quell'hotel è stato così pessimo, perché assegnare comunque dei punti? Esaminando attentamente il dataset, vedrai che il punteggio minimo possibile è 2.5, non 0. Il punteggio massimo possibile è 10.

##### Tags

Come accennato sopra, a prima vista, l'idea di utilizzare i `Tags` per categorizzare i dati sembra sensata. Sfortunatamente, questi tag non sono standardizzati, il che significa che in un determinato hotel le opzioni potrebbero essere *Camera singola*, *Camera doppia*, e *Camera matrimoniale*, ma in un altro hotel potrebbero essere *Camera singola deluxe*, *Camera queen classica*, e *Camera king executive*. Potrebbero essere la stessa cosa, ma ci sono così tante variazioni che la scelta diventa:

1. Tentare di uniformare tutti i termini a uno standard unico, il che è molto difficile, perché non è chiaro quale sarebbe il percorso di conversione in ogni caso (ad esempio, *Camera singola classica* si mappa a *Camera singola*, ma *Camera queen superiore con vista sul giardino o sulla città* è molto più difficile da mappare)

2. Possiamo adottare un approccio NLP e misurare la frequenza di determinati termini come *Solitario*, *Viaggiatore d'affari*, o *Famiglia con bambini piccoli* per ogni hotel, e includere questo fattore nel modello di raccomandazione  

I tag sono di solito (ma non sempre) un singolo campo contenente un elenco di 5-6 valori separati da virgole che si riferiscono a *Tipo di viaggio*, *Tipo di ospiti*, *Tipo di stanza*, *Numero di notti*, e *Tipo di dispositivo utilizzato per inviare la recensione*. Tuttavia, poiché alcuni recensori non compilano ogni campo (potrebbero lasciarne uno vuoto), i valori non sono sempre nello stesso ordine.

Ad esempio, prendi il *Tipo di gruppo*. Ci sono 1025 possibilità uniche in questo campo nella colonna `Tags`, e sfortunatamente solo alcune di esse si riferiscono a un gruppo (alcune sono il tipo di stanza, ecc.). Se filtri solo quelle che menzionano famiglia, i risultati contengono molti risultati relativi a *Camera familiare*. Se includi il termine *con*, ad esempio conti i valori *Famiglia con*, i risultati sono migliori, con oltre 80.000 dei 515.000 risultati che contengono la frase "Famiglia con bambini piccoli" o "Famiglia con bambini più grandi".

Questo significa che la colonna dei tag non è completamente inutile, ma richiederà del lavoro per renderla utile.

##### Punteggio medio dell'hotel

Ci sono alcune stranezze o discrepanze nel dataset che non riesco a spiegare, ma che sono illustrate qui affinché tu ne sia consapevole quando costruisci i tuoi modelli. Se riesci a capirle, faccelo sapere nella sezione di discussione!

Il dataset ha le seguenti colonne relative al punteggio medio e al numero di recensioni: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

L'hotel con il maggior numero di recensioni in questo dataset è *Britannia International Hotel Canary Wharf* con 4789 recensioni su 515.000. Ma se guardiamo il valore di `Total_Number_of_Reviews` per questo hotel, è 9086. Potresti dedurre che ci siano molti più punteggi senza recensioni, quindi forse dovremmo aggiungere il valore della colonna `Additional_Number_of_Scoring`. Quel valore è 2682, e aggiungendolo a 4789 otteniamo 7471, che è comunque 1615 in meno rispetto a `Total_Number_of_Reviews`. 

Se prendi la colonna `Average_Score`, potresti supporre che sia la media delle recensioni nel dataset, ma la descrizione su Kaggle è "*Punteggio medio dell'hotel, calcolato in base all'ultimo commento dell'ultimo anno*". Questo non sembra molto utile, ma possiamo calcolare la nostra media basandoci sui punteggi delle recensioni nel dataset. Usando lo stesso hotel come esempio, il punteggio medio dell'hotel è dato come 7.1, ma il punteggio calcolato (media dei punteggi dei recensori *nel* dataset) è 6.8. Questo è vicino, ma non lo stesso valore, e possiamo solo ipotizzare che i punteggi dati nelle recensioni `Additional_Number_of_Scoring` abbiano aumentato la media a 7.1. Sfortunatamente, senza un modo per testare o dimostrare questa affermazione, è difficile utilizzare o fidarsi di `Average_Score`, `Additional_Number_of_Scoring` e `Total_Number_of_Reviews` quando si basano su, o si riferiscono a, dati che non abbiamo.

Per complicare ulteriormente le cose, l'hotel con il secondo maggior numero di recensioni ha un punteggio medio calcolato di 8.12 e il punteggio `Average_Score` del dataset è 8.1. Questo punteggio corretto è una coincidenza o la discrepanza riguarda solo il primo hotel?
Sulla possibilità che questi hotel possano essere un'eccezione, e che forse la maggior parte dei valori corrispondano (ma alcuni no per qualche motivo), scriveremo un breve programma per esplorare i valori nel dataset e determinare il corretto utilizzo (o non utilizzo) dei valori.

> 🚨 Una nota di cautela
>
> Quando lavori con questo dataset, scriverai codice che calcola qualcosa dal testo senza dover leggere o analizzare il testo tu stesso. Questa è l'essenza dell'NLP: interpretare il significato o il sentimento senza che un essere umano debba farlo. Tuttavia, è possibile che tu legga alcune recensioni negative. Ti esorto a non farlo, perché non è necessario. Alcune di esse sono sciocche o irrilevanti, come "Il tempo non era bello", qualcosa al di fuori del controllo dell'hotel, o di chiunque altro. Ma c'è anche un lato oscuro in alcune recensioni. A volte le recensioni negative sono razziste, sessiste o discriminatorie per età. Questo è spiacevole ma prevedibile in un dataset estratto da un sito web pubblico. Alcuni recensori lasciano recensioni che potresti trovare di cattivo gusto, scomode o sconvolgenti. Meglio lasciare che il codice misuri il sentimento piuttosto che leggerle tu stesso e rimanerne turbato. Detto ciò, è una minoranza a scrivere tali cose, ma esistono comunque.

## Esercizio - Esplorazione dei dati
### Caricare i dati

Basta esaminare i dati visivamente, ora scriverai del codice per ottenere delle risposte! Questa sezione utilizza la libreria pandas. Il tuo primo compito è assicurarti di poter caricare e leggere i dati CSV. La libreria pandas ha un caricatore CSV veloce, e il risultato viene posizionato in un dataframe, come nelle lezioni precedenti. Il CSV che stiamo caricando ha oltre mezzo milione di righe, ma solo 17 colonne. Pandas ti offre molti modi potenti per interagire con un dataframe, inclusa la possibilità di eseguire operazioni su ogni riga.

Da qui in avanti in questa lezione, ci saranno frammenti di codice, alcune spiegazioni del codice e alcune discussioni su cosa significano i risultati. Usa il file _notebook.ipynb_ incluso per il tuo codice.

Iniziamo caricando il file di dati che utilizzerai:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Ora che i dati sono caricati, possiamo eseguire alcune operazioni su di essi. Tieni questo codice all'inizio del tuo programma per la prossima parte.

## Esplorare i dati

In questo caso, i dati sono già *puliti*, il che significa che sono pronti per essere utilizzati e non contengono caratteri in altre lingue che potrebbero confondere gli algoritmi che si aspettano solo caratteri in inglese.

✅ Potresti dover lavorare con dati che richiedono una elaborazione iniziale per essere formattati prima di applicare tecniche NLP, ma non questa volta. Se dovessi farlo, come gestiresti i caratteri non in inglese?

Prenditi un momento per assicurarti che, una volta caricati i dati, tu possa esplorarli con il codice. È molto facile voler concentrarsi sulle colonne `Negative_Review` e `Positive_Review`. Sono piene di testo naturale per i tuoi algoritmi NLP da processare. Ma aspetta! Prima di immergerti nell'NLP e nel sentimento, segui il codice qui sotto per verificare se i valori forniti nel dataset corrispondono ai valori che calcoli con pandas.

## Operazioni sui dataframe

Il primo compito in questa lezione è verificare se le seguenti affermazioni sono corrette scrivendo del codice che esamina il dataframe (senza modificarlo).

> Come in molti compiti di programmazione, ci sono diversi modi per completare questo, ma un buon consiglio è farlo nel modo più semplice e facile possibile, specialmente se sarà più facile da capire quando tornerai su questo codice in futuro. Con i dataframe, c'è un'API completa che spesso avrà un modo per fare ciò che desideri in modo efficiente.

Tratta le seguenti domande come compiti di programmazione e prova a rispondere senza guardare la soluzione.

1. Stampa la *forma* del dataframe che hai appena caricato (la forma è il numero di righe e colonne).
2. Calcola la frequenza delle nazionalità dei recensori:
   1. Quanti valori distinti ci sono nella colonna `Reviewer_Nationality` e quali sono?
   2. Qual è la nazionalità del recensore più comune nel dataset (stampa il paese e il numero di recensioni)?
   3. Quali sono le successive 10 nazionalità più frequenti e il loro conteggio?
3. Qual è stato l'hotel più recensito per ciascuna delle 10 nazionalità di recensori più comuni?
4. Quante recensioni ci sono per hotel (conteggio delle frequenze degli hotel) nel dataset?
5. Sebbene ci sia una colonna `Average_Score` per ogni hotel nel dataset, puoi anche calcolare un punteggio medio (ottenendo la media di tutti i punteggi dei recensori nel dataset per ogni hotel). Aggiungi una nuova colonna al tuo dataframe con l'intestazione `Calc_Average_Score` che contiene quella media calcolata.
6. Ci sono hotel che hanno lo stesso valore (arrotondato a 1 decimale) per `Average_Score` e `Calc_Average_Score`?
   1. Prova a scrivere una funzione Python che prende una Serie (riga) come argomento e confronta i valori, stampando un messaggio quando i valori non sono uguali. Quindi usa il metodo `.apply()` per elaborare ogni riga con la funzione.
7. Calcola e stampa quante righe hanno valori della colonna `Negative_Review` pari a "No Negative".
8. Calcola e stampa quante righe hanno valori della colonna `Positive_Review` pari a "No Positive".
9. Calcola e stampa quante righe hanno valori della colonna `Positive_Review` pari a "No Positive" **e** valori della colonna `Negative_Review` pari a "No Negative".

### Risposte in codice

1. Stampa la *forma* del dataframe che hai appena caricato (la forma è il numero di righe e colonne).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcola la frequenza delle nazionalità dei recensori:

   1. Quanti valori distinti ci sono nella colonna `Reviewer_Nationality` e quali sono?
   2. Qual è la nazionalità del recensore più comune nel dataset (stampa il paese e il numero di recensioni)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Quali sono le successive 10 nazionalità più frequenti e il loro conteggio?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Qual è stato l'hotel più recensito per ciascuna delle 10 nazionalità di recensori più comuni?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Quante recensioni ci sono per hotel (conteggio delle frequenze degli hotel) nel dataset?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Nome_Hotel                 | Numero_Totale_Recensioni | Recensioni_Trovate |
   | :----------------------------------------: | :-----------------------: | :----------------: |
   | Britannia International Hotel Canary Wharf |          9086            |        4789        |
   |    Park Plaza Westminster Bridge London    |          12158           |        4169        |
   |   Copthorne Tara Hotel London Kensington   |          7105            |        3578        |
   |                    ...                     |           ...            |         ...        |
   |       Mercure Paris Porte d Orleans        |           110            |         10         |
   |                Hotel Wagner                |           135            |         10         |
   |            Hotel Gallitzinberg             |           173            |          8         |

   Potresti notare che i risultati *conteggiati nel dataset* non corrispondono al valore in `Total_Number_of_Reviews`. Non è chiaro se questo valore nel dataset rappresentasse il numero totale di recensioni che l'hotel aveva, ma non tutte sono state estratte, o qualche altro calcolo. `Total_Number_of_Reviews` non viene utilizzato nel modello a causa di questa incertezza.

5. Sebbene ci sia una colonna `Average_Score` per ogni hotel nel dataset, puoi anche calcolare un punteggio medio (ottenendo la media di tutti i punteggi dei recensori nel dataset per ogni hotel). Aggiungi una nuova colonna al tuo dataframe con l'intestazione `Calc_Average_Score` che contiene quella media calcolata. Stampa le colonne `Hotel_Name`, `Average_Score` e `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Potresti anche chiederti del valore `Average_Score` e perché a volte è diverso dal punteggio medio calcolato. Poiché non possiamo sapere perché alcuni valori corrispondono, ma altri hanno una differenza, è più sicuro in questo caso utilizzare i punteggi delle recensioni che abbiamo per calcolare la media noi stessi. Detto ciò, le differenze sono di solito molto piccole, ecco gli hotel con la maggiore deviazione tra la media del dataset e la media calcolata:

   | Differenza_Media_Punteggio | Media_Punteggio | Media_Calcolata |                                  Nome_Hotel |
   | :------------------------: | :-------------: | :-------------: | ------------------------------------------: |
   |           -0.8             |      7.7        |      8.5        |                  Best Western Hotel Astoria |
   |           -0.7             |      8.8        |      9.5        | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7             |      7.5        |      8.2        |               Mercure Paris Porte d Orleans |
   |           -0.7             |      7.9        |      8.6        |             Renaissance Paris Vendome Hotel |
   |           -0.5             |      7.0        |      7.5        |                         Hotel Royal Elys es |
   |           ...              |      ...        |      ...        |                                         ... |
   |           0.7              |      7.5        |      6.8        |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8              |      7.1        |      6.3        |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9              |      6.8        |      5.9        |                               Villa Eugenie |
   |           0.9              |      8.6        |      7.7        |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3              |      7.2        |      5.9        |                          Kube Hotel Ice Bar |

   Con solo 1 hotel che ha una differenza di punteggio maggiore di 1, significa che probabilmente possiamo ignorare la differenza e utilizzare il punteggio medio calcolato.

6. Calcola e stampa quante righe hanno valori della colonna `Negative_Review` pari a "No Negative".

7. Calcola e stampa quante righe hanno valori della colonna `Positive_Review` pari a "No Positive".

8. Calcola e stampa quante righe hanno valori della colonna `Positive_Review` pari a "No Positive" **e** valori della colonna `Negative_Review` pari a "No Negative".

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Un altro modo

Un altro modo per contare gli elementi senza Lambdas, e usare sum per contare le righe:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Potresti aver notato che ci sono 127 righe che hanno sia "No Negative" che "No Positive" come valori per le colonne `Negative_Review` e `Positive_Review` rispettivamente. Ciò significa che il recensore ha dato all'hotel un punteggio numerico, ma ha deciso di non scrivere né una recensione positiva né una negativa. Fortunatamente, si tratta di una piccola quantità di righe (127 su 515738, ovvero 0,02%), quindi probabilmente non influenzerà il nostro modello o i risultati in una direzione particolare, ma potresti non aspettarti che un dataset di recensioni contenga righe senza recensioni, quindi vale la pena esplorare i dati per scoprire righe come questa.

Ora che hai esplorato il dataset, nella prossima lezione filtrerai i dati e aggiungerai un'analisi del sentimento.

---
## 🚀Sfida

Questa lezione dimostra, come abbiamo visto nelle lezioni precedenti, quanto sia fondamentale comprendere i tuoi dati e le loro peculiarità prima di eseguire operazioni su di essi. I dati basati su testo, in particolare, richiedono un'analisi attenta. Esplora vari dataset ricchi di testo e vedi se riesci a scoprire aree che potrebbero introdurre bias o sentimenti distorti in un modello.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## Revisione e studio autonomo

Segui [questo percorso di apprendimento sull'NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) per scoprire strumenti da provare quando costruisci modelli basati su testo e voce.

## Compito

[NLTK](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.