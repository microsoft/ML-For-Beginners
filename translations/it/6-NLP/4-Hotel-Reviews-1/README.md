<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-06T07:39:33+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "it"
}
-->
# Analisi del sentiment con recensioni di hotel - elaborazione dei dati

In questa sezione utilizzerai le tecniche apprese nelle lezioni precedenti per effettuare un'analisi esplorativa dei dati su un ampio dataset. Una volta che avrai una buona comprensione dell'utilità delle varie colonne, imparerai:

- come rimuovere le colonne non necessarie
- come calcolare nuovi dati basati sulle colonne esistenti
- come salvare il dataset risultante per utilizzarlo nella sfida finale

## [Quiz preliminare alla lezione](https://ff-quizzes.netlify.app/en/ml/)

### Introduzione

Finora hai imparato che i dati testuali sono molto diversi dai dati numerici. Se il testo è stato scritto o parlato da un essere umano, può essere analizzato per trovare modelli, frequenze, sentiment e significato. Questa lezione ti introduce a un dataset reale con una sfida concreta: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, che include una [licenza CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). È stato estratto da Booking.com da fonti pubbliche. Il creatore del dataset è Jiashen Liu.

### Preparazione

Avrai bisogno di:

* La capacità di eseguire notebook .ipynb utilizzando Python 3
* pandas
* NLTK, [che dovresti installare localmente](https://www.nltk.org/install.html)
* Il dataset disponibile su Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). È di circa 230 MB una volta decompresso. Scaricalo nella cartella principale `/data` associata a queste lezioni di NLP.

## Analisi esplorativa dei dati

Questa sfida presuppone che tu stia costruendo un bot di raccomandazione per hotel utilizzando l'analisi del sentiment e i punteggi delle recensioni degli ospiti. Il dataset che utilizzerai include recensioni di 1493 hotel diversi in 6 città.

Utilizzando Python, un dataset di recensioni di hotel e l'analisi del sentiment di NLTK, potresti scoprire:

* Quali sono le parole e le frasi più frequentemente utilizzate nelle recensioni?
* I *tag* ufficiali che descrivono un hotel sono correlati ai punteggi delle recensioni (ad esempio, ci sono recensioni più negative per un particolare hotel da parte di *Famiglie con bambini piccoli* rispetto a *Viaggiatori solitari*, forse indicando che è più adatto ai *Viaggiatori solitari*)?
* I punteggi di sentiment di NLTK "concordano" con il punteggio numerico del recensore dell'hotel?

#### Dataset

Esploriamo il dataset che hai scaricato e salvato localmente. Apri il file in un editor come VS Code o anche Excel.

Le intestazioni nel dataset sono le seguenti:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Ecco come sono raggruppate per facilitarne l'esame: 
##### Colonne dell'hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitudine), `lng` (longitudine)
  * Utilizzando *lat* e *lng* potresti tracciare una mappa con Python che mostra le posizioni degli hotel (forse codificate per colore in base alle recensioni negative e positive)
  * Hotel_Address non sembra essere particolarmente utile e probabilmente lo sostituiremo con un paese per facilitare l'ordinamento e la ricerca

**Colonne meta-recensione dell'hotel**

* `Average_Score`
  * Secondo il creatore del dataset, questa colonna rappresenta il *Punteggio medio dell'hotel, calcolato sulla base del commento più recente dell'ultimo anno*. Questo sembra un modo insolito di calcolare il punteggio, ma è il dato estratto, quindi per ora lo accettiamo così com'è.
  
  ✅ Sulla base delle altre colonne di questo dataset, riesci a pensare a un altro modo per calcolare il punteggio medio?

* `Total_Number_of_Reviews`
  * Il numero totale di recensioni che questo hotel ha ricevuto - non è chiaro (senza scrivere del codice) se si riferisce alle recensioni nel dataset.
* `Additional_Number_of_Scoring`
  * Significa che è stato dato un punteggio di recensione ma il recensore non ha scritto una recensione positiva o negativa

**Colonne delle recensioni**

- `Reviewer_Score`
  - Questo è un valore numerico con al massimo 1 cifra decimale tra i valori minimi e massimi 2.5 e 10
  - Non è spiegato perché 2.5 sia il punteggio minimo possibile
- `Negative_Review`
  - Se un recensore non ha scritto nulla, questo campo avrà "**No Negative**"
  - Nota che un recensore potrebbe scrivere una recensione positiva nella colonna Negative review (ad esempio, "non c'è nulla di negativo in questo hotel")
- `Review_Total_Negative_Word_Counts`
  - Un numero maggiore di parole negative indica un punteggio più basso (senza verificare la sentimentalità)
- `Positive_Review`
  - Se un recensore non ha scritto nulla, questo campo avrà "**No Positive**"
  - Nota che un recensore potrebbe scrivere una recensione negativa nella colonna Positive review (ad esempio, "non c'è nulla di positivo in questo hotel")
- `Review_Total_Positive_Word_Counts`
  - Un numero maggiore di parole positive indica un punteggio più alto (senza verificare la sentimentalità)
- `Review_Date` e `days_since_review`
  - Si potrebbe applicare una misura di freschezza o obsolescenza a una recensione (le recensioni più vecchie potrebbero non essere accurate quanto quelle più recenti perché la gestione dell'hotel è cambiata, sono stati fatti lavori di ristrutturazione, è stata aggiunta una piscina ecc.)
- `Tags`
  - Questi sono brevi descrittori che un recensore può selezionare per descrivere il tipo di ospite che era (ad esempio, solitario o famiglia), il tipo di stanza che aveva, la durata del soggiorno e come è stata inviata la recensione.
  - Purtroppo, l'uso di questi tag è problematico, consulta la sezione seguente che discute la loro utilità

**Colonne del recensore**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Questo potrebbe essere un fattore in un modello di raccomandazione, ad esempio, se si potesse determinare che i recensori più prolifici con centinaia di recensioni erano più propensi a essere negativi piuttosto che positivi. Tuttavia, il recensore di una particolare recensione non è identificato con un codice univoco e quindi non può essere collegato a un set di recensioni. Ci sono 30 recensori con 100 o più recensioni, ma è difficile vedere come questo possa aiutare il modello di raccomandazione.
- `Reviewer_Nationality`
  - Alcune persone potrebbero pensare che alcune nazionalità siano più propense a dare una recensione positiva o negativa a causa di un'inclinazione nazionale. Fai attenzione a costruire tali visioni aneddotiche nei tuoi modelli. Questi sono stereotipi nazionali (e talvolta razziali), e ogni recensore era un individuo che ha scritto una recensione basata sulla propria esperienza. Potrebbe essere stata filtrata attraverso molte lenti, come i loro precedenti soggiorni in hotel, la distanza percorsa e il loro temperamento personale. Pensare che la loro nazionalità sia stata la ragione di un punteggio di recensione è difficile da giustificare.

##### Esempi

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Questo non è attualmente un hotel ma un cantiere edile Sono stato terrorizzato fin dal mattino presto e per tutto il giorno con rumori di costruzione inaccettabili mentre mi riposavo dopo un lungo viaggio e lavoravo nella stanza Le persone lavoravano tutto il giorno con martelli pneumatici nelle stanze adiacenti Ho chiesto di cambiare stanza ma non era disponibile nessuna stanza silenziosa Per peggiorare le cose mi è stato addebitato un costo eccessivo Ho fatto il check-out la sera poiché avevo un volo molto presto e ho ricevuto una fattura appropriata Il giorno dopo l'hotel ha effettuato un altro addebito senza il mio consenso superiore al prezzo prenotato È un posto terribile Non punirti prenotando qui | Nulla  Posto terribile Stai lontano | Viaggio di lavoro                                Coppia Camera doppia standard Soggiorno di 2 notti |

Come puoi vedere, questo ospite non ha avuto un soggiorno felice in questo hotel. L'hotel ha un buon punteggio medio di 7.8 e 1945 recensioni, ma questo recensore gli ha dato 2.5 e ha scritto 115 parole su quanto sia stato negativo il suo soggiorno. Se non avesse scritto nulla nella colonna Positive_Review, si potrebbe dedurre che non ci fosse nulla di positivo, ma invece ha scritto 7 parole di avvertimento. Se contassimo solo le parole invece del significato o del sentiment delle parole, potremmo avere una visione distorta dell'intento del recensore. Stranamente, il loro punteggio di 2.5 è confuso, perché se quel soggiorno in hotel è stato così pessimo, perché assegnare comunque dei punti? Esaminando attentamente il dataset, vedrai che il punteggio minimo possibile è 2.5, non 0. Il punteggio massimo possibile è 10.

##### Tags

Come accennato sopra, a prima vista, l'idea di utilizzare `Tags` per categorizzare i dati sembra sensata. Purtroppo questi tag non sono standardizzati, il che significa che in un determinato hotel le opzioni potrebbero essere *Camera singola*, *Camera doppia*, e *Camera matrimoniale*, ma nel prossimo hotel potrebbero essere *Camera singola deluxe*, *Camera queen classica*, e *Camera king executive*. Questi potrebbero essere la stessa cosa, ma ci sono così tante variazioni che la scelta diventa:

1. Tentare di modificare tutti i termini in uno standard unico, il che è molto difficile, perché non è chiaro quale sarebbe il percorso di conversione in ogni caso (ad esempio, *Camera singola classica* si mappa su *Camera singola* ma *Camera queen superiore con vista sul giardino o sulla città* è molto più difficile da mappare)

1. Possiamo adottare un approccio NLP e misurare la frequenza di determinati termini come *Solitario*, *Viaggiatore d'affari*, o *Famiglia con bambini piccoli* mentre si applicano a ciascun hotel, e includerlo nel modello di raccomandazione  

I tag sono solitamente (ma non sempre) un singolo campo contenente un elenco di 5 o 6 valori separati da virgole che si riferiscono a *Tipo di viaggio*, *Tipo di ospiti*, *Tipo di stanza*, *Numero di notti*, e *Tipo di dispositivo su cui è stata inviata la recensione*. Tuttavia, poiché alcuni recensori non compilano ogni campo (potrebbero lasciarne uno vuoto), i valori non sono sempre nello stesso ordine.

Ad esempio, prendi *Tipo di gruppo*. Ci sono 1025 possibilità uniche in questo campo nella colonna `Tags`, e purtroppo solo alcune di esse si riferiscono a un gruppo (alcune sono il tipo di stanza ecc.). Se filtri solo quelle che menzionano famiglia, i risultati contengono molti tipi di *Camera familiare*. Se includi il termine *con*, cioè conti i valori *Famiglia con*, i risultati sono migliori, con oltre 80.000 dei 515.000 risultati che contengono la frase "Famiglia con bambini piccoli" o "Famiglia con bambini più grandi".

Questo significa che la colonna dei tag non è completamente inutile per noi, ma ci vorrà del lavoro per renderla utile.

##### Punteggio medio dell'hotel

Ci sono una serie di stranezze o discrepanze con il dataset che non riesco a spiegare, ma sono illustrate qui affinché tu ne sia consapevole quando costruisci i tuoi modelli. Se riesci a capirle, faccelo sapere nella sezione di discussione!

Il dataset ha le seguenti colonne relative al punteggio medio e al numero di recensioni: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

L'hotel singolo con il maggior numero di recensioni in questo dataset è *Britannia International Hotel Canary Wharf* con 4789 recensioni su 515.000. Ma se guardiamo il valore `Total_Number_of_Reviews` per questo hotel, è 9086. Potresti dedurre che ci siano molti più punteggi senza recensioni, quindi forse dovremmo aggiungere il valore della colonna `Additional_Number_of_Scoring`. Quel valore è 2682, e aggiungendolo a 4789 otteniamo 7471, che è ancora 1615 in meno rispetto a `Total_Number_of_Reviews`. 

Se prendi la colonna `Average_Score`, potresti dedurre che sia la media delle recensioni nel dataset, ma la descrizione su Kaggle è "*Punteggio medio dell'hotel, calcolato sulla base del commento più recente dell'ultimo anno*". Questo non sembra molto utile, ma possiamo calcolare la nostra media basata sui punteggi delle recensioni nel dataset. Utilizzando lo stesso hotel come esempio, il punteggio medio dell'hotel è dato come 7.1 ma il punteggio calcolato (punteggio medio del recensore *nel* dataset) è 6.8. Questo è vicino, ma non lo stesso valore, e possiamo solo supporre che i punteggi dati nelle recensioni `Additional_Number_of_Scoring` abbiano aumentato la media a 7.1. Purtroppo, senza modo di testare o dimostrare questa affermazione, è difficile utilizzare o fidarsi di `Average_Score`, `Additional_Number_of_Scoring` e `Total_Number_of_Reviews` quando si basano su, o si riferiscono a, dati che non abbiamo.

Per complicare ulteriormente le cose, l'hotel con il secondo maggior numero di recensioni ha un punteggio medio calcolato di 8.12 e il punteggio `Average_Score` del dataset è 8.1. Questo punteggio corretto è una coincidenza o il primo hotel è una discrepanza? 

Nell'ipotesi che questi hotel possano essere un'eccezione e che forse la maggior parte dei valori corrisponda (ma alcuni no per qualche motivo), scriveremo un breve programma per esplorare i valori nel dataset e determinare l'uso corretto (o non utilizzo) dei valori.
> 🚨 Una nota di cautela  
>  
> Quando lavori con questo dataset, scriverai codice che calcola qualcosa dal testo senza dover leggere o analizzare il testo direttamente. Questa è l'essenza dell'NLP: interpretare il significato o il sentimento senza che sia necessario l'intervento umano. Tuttavia, è possibile che tu legga alcune delle recensioni negative. Ti consiglio di non farlo, perché non è necessario. Alcune di esse sono sciocche o irrilevanti, come recensioni negative sugli hotel del tipo "Il tempo non era bello", qualcosa che è al di fuori del controllo dell'hotel, o di chiunque altro. Ma c'è anche un lato oscuro in alcune recensioni. A volte le recensioni negative sono razziste, sessiste o discriminatorie verso l'età. Questo è spiacevole ma prevedibile in un dataset raccolto da un sito web pubblico. Alcuni recensori lasciano commenti che potresti trovare sgradevoli, imbarazzanti o sconvolgenti. È meglio lasciare che il codice misuri il sentimento piuttosto che leggerle personalmente e rimanere turbato. Detto ciò, è una minoranza a scrivere cose del genere, ma esistono comunque.
## Esercizio - Esplorazione dei dati
### Carica i dati

Basta esaminare visivamente i dati, ora scriverai del codice per ottenere delle risposte! Questa sezione utilizza la libreria pandas. Il tuo primo compito è assicurarti di poter caricare e leggere i dati CSV. La libreria pandas ha un caricatore CSV veloce, e il risultato viene posizionato in un dataframe, come nelle lezioni precedenti. Il CSV che stiamo caricando ha oltre mezzo milione di righe, ma solo 17 colonne. Pandas ti offre molti modi potenti per interagire con un dataframe, inclusa la possibilità di eseguire operazioni su ogni riga.

Da qui in avanti, in questa lezione, ci saranno frammenti di codice, alcune spiegazioni del codice e discussioni su cosa significano i risultati. Usa il file _notebook.ipynb_ incluso per il tuo codice.

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

Ora che i dati sono stati caricati, possiamo eseguire alcune operazioni su di essi. Tieni questo codice all'inizio del tuo programma per la parte successiva.

## Esplora i dati

In questo caso, i dati sono già *puliti*, il che significa che sono pronti per essere utilizzati e non contengono caratteri in altre lingue che potrebbero creare problemi agli algoritmi che si aspettano solo caratteri in inglese.

✅ Potresti dover lavorare con dati che richiedono una prima elaborazione per essere formattati prima di applicare tecniche NLP, ma non questa volta. Se fosse necessario, come gestiresti i caratteri non inglesi?

Prenditi un momento per assicurarti che, una volta caricati i dati, tu possa esplorarli con il codice. È molto facile voler concentrarsi sulle colonne `Negative_Review` e `Positive_Review`. Queste sono piene di testo naturale per i tuoi algoritmi NLP da elaborare. Ma aspetta! Prima di immergerti nell'NLP e nel sentiment, dovresti seguire il codice qui sotto per verificare se i valori forniti nel dataset corrispondono ai valori che calcoli con pandas.

## Operazioni sul dataframe

Il primo compito in questa lezione è verificare se le seguenti affermazioni sono corrette scrivendo del codice che esamina il dataframe (senza modificarlo).

> Come molti compiti di programmazione, ci sono diversi modi per completarlo, ma un buon consiglio è farlo nel modo più semplice e facile possibile, soprattutto se sarà più facile da capire quando tornerai su questo codice in futuro. Con i dataframe, c'è un'API completa che spesso avrà un modo per fare ciò che desideri in modo efficiente.

Tratta le seguenti domande come compiti di programmazione e prova a rispondere senza guardare la soluzione.

1. Stampa la *forma* del dataframe che hai appena caricato (la forma è il numero di righe e colonne).
2. Calcola il conteggio delle frequenze per le nazionalità dei recensori:
   1. Quanti valori distinti ci sono per la colonna `Reviewer_Nationality` e quali sono?
   2. Qual è la nazionalità del recensore più comune nel dataset (stampa il paese e il numero di recensioni)?
   3. Quali sono le 10 nazionalità più frequenti e il loro conteggio di frequenza?
3. Qual è l'hotel più recensito per ciascuna delle 10 nazionalità di recensori più comuni?
4. Quante recensioni ci sono per hotel (conteggio di frequenza degli hotel) nel dataset?
5. Sebbene ci sia una colonna `Average_Score` per ogni hotel nel dataset, puoi anche calcolare un punteggio medio (ottenendo la media di tutti i punteggi dei recensori nel dataset per ogni hotel). Aggiungi una nuova colonna al tuo dataframe con l'intestazione `Calc_Average_Score` che contiene quella media calcolata.
6. Ci sono hotel che hanno lo stesso valore (arrotondato a 1 decimale) per `Average_Score` e `Calc_Average_Score`?
   1. Prova a scrivere una funzione Python che prende una Serie (riga) come argomento e confronta i valori, stampando un messaggio quando i valori non sono uguali. Quindi usa il metodo `.apply()` per elaborare ogni riga con la funzione.
7. Calcola e stampa quante righe hanno valori "No Negative" nella colonna `Negative_Review`.
8. Calcola e stampa quante righe hanno valori "No Positive" nella colonna `Positive_Review`.
9. Calcola e stampa quante righe hanno valori "No Positive" nella colonna `Positive_Review` **e** valori "No Negative" nella colonna `Negative_Review`.

### Risposte in codice

1. Stampa la *forma* del dataframe che hai appena caricato (la forma è il numero di righe e colonne).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcola il conteggio delle frequenze per le nazionalità dei recensori:

   1. Quanti valori distinti ci sono per la colonna `Reviewer_Nationality` e quali sono?
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

   3. Quali sono le 10 nazionalità più frequenti e il loro conteggio di frequenza?

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

3. Qual è l'hotel più recensito per ciascuna delle 10 nazionalità di recensori più comuni?

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

4. Quante recensioni ci sono per hotel (conteggio di frequenza degli hotel) nel dataset?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |

   Potresti notare che i risultati *conteggiati nel dataset* non corrispondono al valore in `Total_Number_of_Reviews`. Non è chiaro se questo valore nel dataset rappresentasse il numero totale di recensioni che l'hotel aveva, ma non tutte sono state raccolte, o qualche altro calcolo. `Total_Number_of_Reviews` non viene utilizzato nel modello a causa di questa ambiguità.

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

   Potresti anche chiederti del valore `Average_Score` e perché a volte è diverso dal punteggio medio calcolato. Poiché non possiamo sapere perché alcuni valori corrispondono, ma altri hanno una differenza, è più sicuro in questo caso utilizzare i punteggi delle recensioni che abbiamo per calcolare la media da soli. Detto ciò, le differenze sono solitamente molto piccole, ecco gli hotel con la maggiore deviazione tra la media del dataset e la media calcolata:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Con solo 1 hotel che ha una differenza di punteggio maggiore di 1, significa che probabilmente possiamo ignorare la differenza e utilizzare il punteggio medio calcolato.

6. Calcola e stampa quante righe hanno valori "No Negative" nella colonna `Negative_Review`.

7. Calcola e stampa quante righe hanno valori "No Positive" nella colonna `Positive_Review`.

8. Calcola e stampa quante righe hanno valori "No Positive" nella colonna `Positive_Review` **e** valori "No Negative" nella colonna `Negative_Review`.

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

Un altro modo per contare gli elementi senza Lambdas, e utilizzare sum per contare le righe:

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

   Potresti aver notato che ci sono 127 righe che hanno sia "No Negative" che "No Positive" come valori per le colonne `Negative_Review` e `Positive_Review` rispettivamente. Ciò significa che il recensore ha dato all'hotel un punteggio numerico, ma ha deciso di non scrivere né una recensione positiva né negativa. Fortunatamente, si tratta di una piccola quantità di righe (127 su 515738, ovvero 0,02%), quindi probabilmente non influenzerà il nostro modello o i risultati in una direzione particolare, ma potresti non aspettarti che un dataset di recensioni abbia righe senza recensioni, quindi vale la pena esplorare i dati per scoprire righe come questa.

Ora che hai esplorato il dataset, nella prossima lezione filtrerai i dati e aggiungerai un'analisi del sentiment.

---
## 🚀Sfida

Questa lezione dimostra, come abbiamo visto nelle lezioni precedenti, quanto sia importante comprendere i tuoi dati e le loro peculiarità prima di eseguire operazioni su di essi. I dati basati su testo, in particolare, richiedono un'attenzione scrupolosa. Esplora vari dataset ricchi di testo e vedi se riesci a scoprire aree che potrebbero introdurre bias o sentiment distorti in un modello.

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Revisione e studio autonomo

Segui [questo percorso di apprendimento sull'NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) per scoprire strumenti da provare quando costruisci modelli basati su discorsi e testi.

## Compito

[NLTK](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.