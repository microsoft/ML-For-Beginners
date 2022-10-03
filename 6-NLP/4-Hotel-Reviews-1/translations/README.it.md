# Analisi del sentiment con le recensioni degli hotel - elaborazione dei dati

In questa sezione si utilizzeranno le tecniche delle lezioni precedenti per eseguire alcune analisi esplorative dei dati di un grande insieme di dati. Una volta compresa bene l'utilità delle varie colonne, si imparerà:

- come rimuovere le colonne non necessarie
- come calcolare alcuni nuovi dati in base alle colonne esistenti
- come salvare l'insieme di dati risultante per l'uso nella sfida finale

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/?loc=it)

### Introduzione

Finora si è appreso come i dati di testo siano abbastanza diversi dai tipi di dati numerici. Se è un testo scritto o parlato da un essere umano, può essere analizzato per trovare schemi e frequenze, sentiment e significati. Questa lezione entra in un vero insieme di dati con una vera sfida: **[515K dati di recensioni di hotel in Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** e include una [licenza CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). È stato ricavato da Booking.com da fonti pubbliche. Il creatore dell'insieme di dati è stato Jiashen Liu.

### Preparazione

Ecco l'occorrente:

* La possibilità di eseguire notebook .ipynb utilizzando Python 3
* pandas
* NLTK, [che si dovrebbe installare localmente](https://www.nltk.org/install.html)
* L'insieme di dati disponibile su Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Sono circa 230 MB decompressi. Scaricarlo nella cartella radice `/data` associata a queste lezioni di NLP.

## Analisi esplorativa dei dati

Questa sfida presuppone che si stia creando un bot di raccomandazione di hotel utilizzando l'analisi del sentiment e i punteggi delle recensioni degli ospiti. L'insieme di dati da utilizzare include recensioni di 1493 hotel diversi in 6 città.

Utilizzando Python, un insieme di dati di recensioni di hotel e l'analisi del sentiment di NLTK si potrebbe scoprire:

* quali sono le parole e le frasi più usate nelle recensioni?
* i *tag* ufficiali che descrivono un hotel correlato con punteggi di recensione (ad es. sono le più negative per un particolare hotel per *famiglia con bambini piccoli* rispetto a *viaggiatore singolo*, forse indicando che è meglio per i *viaggiatori sinogli*?)
* i punteggi del sentiment NLTK "concordano" con il punteggio numerico del recensore dell'hotel?

#### Insieme di dati

Per esplorare l'insieme di dati scaricato e salvato localmente, aprire il file in un editor tipo VS Code o anche Excel.

Le intestazioni nell'insieme di dati sono le seguenti:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Qui sono raggruppati in un modo che potrebbe essere più facile da esaminare:

##### Colonne Hotel

* `Hotel_Name` (nome hotel), `Hotel_Address` (indirizzo hotel), `lat` (latitudine)`,` lng (longitudine)
   * Usando *lat* e *lng* si può tracciare una mappa con Python che mostra le posizioni degli hotel (forse codificate a colori per recensioni negative e positive)
   * Hotel_Address non è ovviamente utile e probabilmente verrà sostituito con una nazione per semplificare l'ordinamento e la ricerca

**Colonne di meta-recensione dell'hotel**

* `Average_Score` (Punteggio medio)
   * Secondo il creatore dell'insieme di dati, questa colonna è il *punteggio medio dell'hotel, calcolato in base all'ultimo commento dell'ultimo anno*. Questo sembra un modo insolito per calcolare il punteggio, ma sono i dati recuperati, quindi per ora si potrebbero prendere come valore nominale.

   ✅ Sulla base delle altre colonne di questi dati, si riesce a pensare a un altro modo per calcolare il punteggio medio?

* `Total_Number_of_Reviews` (Numero totale di recensioni)
   * Il numero totale di recensioni ricevute da questo hotel - non è chiaro (senza scrivere del codice) se si riferisce alle recensioni nell'insieme di dati.
* `Additional_Number_of_Scoring` (Numero aggiuntivo di punteggio
   * Ciò significa che è stato assegnato un punteggio di recensione ma nessuna recensione positiva o negativa è stata scritta dal recensore

**Colonne di recensione**

- `Reviewer_Score` (Punteggio recensore)
   - Questo è un valore numerico con al massimo 1 cifra decimale tra i valori minimo e massimo 2,5 e 10
   - Non è spiegato perché 2,5 sia il punteggio più basso possibile
- `Negative_Review` (Recensione Negativa)
   - Se un recensore non ha scritto nulla, questo campo avrà "**No Negative" (Nessun negativo)**
   - Si tenga presente che un recensore può scrivere una recensione positiva nella colonna delle recensioni negative (ad es. "non c'è niente di negativo in questo hotel")
- `Review_Total_Negative_Word_Counts` (Conteggio parole negative totali per revisione)
   - Conteggi di parole negative più alti indicano un punteggio più basso (senza controllare il sentiment)
- `Positive_Review` (Recensioni positive)
   - Se un recensore non ha scritto nulla, questo campo avrà "**No Positive" (Nessun positivo)**
   - Si tenga presente che un recensore può scrivere una recensione negativa nella colonna delle recensioni positive (ad es. "non c'è niente di buono in questo hotel")
- `Review_Total_Positive_Word_Counts` (Conteggio parole positive totali per revisione)
   - Conteggi di parole positive più alti indicano un punteggio più alto (senza controllare il sentiment)
- `Review_Date` e `days_since_review` (Data revisione e giorni trascorsi dalla revisione)
   - Una misura di freschezza od obsolescenza potrebbe essere applicata a una recensione (le recensioni più vecchie potrebbero non essere accurate quanto quelle più recenti perché la gestione dell'hotel è cambiata, o sono stati effettuati lavori di ristrutturazione, o è stata aggiunta una piscina, ecc.)
- `Tag`
   - Questi sono brevi descrittori che un recensore può selezionare per descrivere il tipo di ospite in cui rientra (ad es. da soli o in famiglia), il tipo di camera che aveva, la durata del soggiorno e come è stata inviata la recensione.
   - Sfortunatamente, l'uso di questi tag è problematico, controllare la sezione sottostante che discute la loro utilità

**Colonne dei recensori**

- `Total_Number_of_Reviews_Reviewer_Has_Given` (Numero totale di revisioni per recensore)
   - Questo potrebbe essere un fattore in un modello di raccomandazione, ad esempio, se si potesse determinare che i recensori più prolifici con centinaia di recensioni avevano maggiori probabilità di essere negativi piuttosto che positivi. Tuttavia, il recensore di una particolare recensione non è identificato con un codice univoco e quindi non può essere collegato a un insieme di recensioni. Ci sono 30 recensori con 100 o più recensioni, ma è difficile vedere come questo possa aiutare il modello di raccomandazione.
- `Reviewer_Nationality` (Nazionalità recensore)
   - Alcune persone potrebbero pensare che alcune nazionalità abbiano maggiore propensione a dare una recensione positiva o negativa a causa di un'inclinazione nazionalista. Fare attenzione a creare tali punti di vista aneddotici nei propri modelli. Questi sono stereotipi nazionalisti (e talvolta razziali) e ogni recensore era un individuo che ha scritto una recensione in base alla propria esperienza. Potrebbe essere stata filtrata attraverso molte lenti come i loro precedenti soggiorni in hotel, la distanza percorsa e la loro indole. Pensare che la loro nazionalità sia stata la ragione per un punteggio di recensione è difficile da giustificare.

##### Esempi

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | This is  currently not a hotel but a construction site I was terroized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make thinks worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropiate  bill A day later the hotel made another charge without my concent in excess  of booked price It s a terrible place Don t punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

Come si può vedere, questo ospite non ha trascorso un soggiorno felice in questo hotel. L'hotel ha un buon punteggio medio di 7,8 e 1945 recensioni, ma questo recensore ha dato 2,5 e ha scritto 115 parole su quanto sia stato negativo il suo soggiorno. Se non è stato scritto nulla nella colonna Positive_Review, si potrebbe supporre che non ci sia stato nulla di positivo, ma purtroppo sono state scritte 7 parole di avvertimento. Se si contassero solo le parole invece del significato o del sentiment delle parole, si potrebbe avere una visione distorta dell'intento dei recensori. Stranamente, il punteggio di 2,5 è fonte di confusione, perché se il soggiorno in hotel è stato così negativo, perché dargli dei punti? Indagando da vicino l'insieme di dati, si vedrà che il punteggio più basso possibile è 2,5, non 0. Il punteggio più alto possibile è 10.

##### Tag

Come accennato in precedenza, a prima vista, l'idea di utilizzare i `tag` per classificare i dati ha senso. Sfortunatamente questi tag non sono standardizzati, il che significa che in un determinato hotel le opzioni potrebbero essere *Camera singola* , *Camera a due letti* e *Camera doppia*, ma nell'hotel successivo potrebbero essere *Camera singola deluxe*, *Camera matrimoniale classica* e *Camera king executive*. Potrebbero essere le stesse cose, ma ci sono così tante varianti che la scelta diventa:

1. Tentare di modificare tutti i termini in un unico standard, il che è molto difficile, perché non è chiaro quale sarebbe il percorso di conversione in ciascun caso (ad es. *Camera singola classica* mappata in *camera singola*, ma *camera Queen Superior con cortile giardino o vista città* è molto più difficile da mappare)

1. Si può adottare un approccio NLP e misurare la frequenza di determinati termini come *Solo*, *Business Traveller* (Viaggiatore d'affari) o *Family with young kids* (Famiglia con bambini piccoli) quando si applicano a ciascun hotel e tenerne conto nella raccomandazione

I tag sono in genere (ma non sempre) un singolo campo contenente un elenco di valori separati da 5 a 6 virgole allineati per *Type of trip* (Tipo di viaggio), *Type of guests* (Tipo di ospiti), *Type of room* (Tipo di camera), *Number of nights* (Numero di notti) e *Type of device* (Tipo di dispositivo con il quale è stata inviata la recensione).Tuttavia, poiché alcuni recensori non compilano ogni campo (potrebbero lasciarne uno vuoto), i valori non sono sempre nello stesso ordine.

Ad esempio, si prenda *Type of group*  (Tipo di gruppo). Ci sono 1025 possibilità uniche in questo campo nella colonna `Tag`, e purtroppo solo alcune di esse fanno riferimento a un gruppo (alcune sono il tipo di stanza ecc.). Se si filtra solo quelli che menzionano la famiglia, saranno ricavati molti risultati relativi al tipo di *Family room* . Se si include il termine *with* (con), cioè si conta i valori per *Family with*, i risultati sono migliori, con oltre 80.000 dei 515.000 risultati che contengono la frase "Family with young children" (Famiglia con figli piccoli) o "Family with older children" (Famiglia con figli grandi).

Ciò significa che la colonna dei tag non è completamente inutile allo scopo, ma richiederà del lavoro per renderla utile.

##### Punteggio medio dell'hotel

Ci sono una serie di stranezze o discrepanze con l'insieme di dati che non si riesce a capire, ma sono illustrate qui in modo che ci siano note quando si costruiscono i propri modelli. Se ci si capisce qualcosa, renderlo noto nella sezione discussione!

L'insieme di dati ha le seguenti colonne relative al punteggio medio e al numero di recensioni:

1. Hotel_Name (Nome Hotel)
2. Additional_Number_of_Scoring (Numero aggiuntivo di punteggio
3. Average_Score (Punteggio medio)
4. Total_Number_of_Reviews (Numero totale di recensioni)
5. Reviewer_Score (Punteggio recensore)

L'hotel con il maggior numero di recensioni in questo insieme di dati è *il Britannia International Hotel Canary Wharf* con 4789 recensioni su 515.000. Ma se si guarda al valore `Total_Number_of_Reviews` per questo hotel, è 9086. Si potrebbe supporre che ci siano molti più punteggi senza recensioni, quindi forse si dovrebbe aggiungere il valore della colonna `Additional_Number_of_Scoring` . Quel valore è 2682 e aggiungendolo a 4789 si ottiene 7.471 che è ancora 1615 in meno del valore di `Total_Number_of_Reviews`.

Se si prende la colonna `Average_Score`, si potrebbe supporre che sia la media delle recensioni nell'insieme di dati, ma la descrizione di Kaggle è "*Punteggio medio dell’hotel, calcolato in base all’ultimo commento nell’ultimo anno*". Non sembra così utile, ma si può calcolare la media in base ai punteggi delle recensioni nell'insieme di dati. Utilizzando lo stesso hotel come esempio, il punteggio medio dell'hotel è 7,1 ma il punteggio calcolato (punteggio medio del recensore nell'insieme di dati) è 6,8. Questo è vicino, ma non lo stesso valore, e si può solo supporre che i punteggi dati nelle recensioni `Additional_Number_of_Scoring` hanno aumentato la media a 7,1. Sfortunatamente, senza alcun modo per testare o dimostrare tale affermazione, è difficile utilizzare o fidarsi di `Average_Score`, `Additional_Number_of_Scoring` e `Total_Number_of_Reviews` quando si basano su o fanno riferimento a dati che non sono presenti.

Per complicare ulteriormente le cose, l'hotel con il secondo numero più alto di recensioni ha un punteggio medio calcolato di 8,12 e l'insieme di dati `Average_Score` è 8,1. Questo punteggio corretto è una coincidenza o il primo hotel è una discrepanza?

Sulla possibilità che questi hotel possano essere un valore anomalo e che forse la maggior parte dei valori coincidano (ma alcuni non lo fanno per qualche motivo)  si scriverà un breve programma per esplorare i valori nell'insieme di dati e determinare l'utilizzo corretto (o mancato utilizzo) dei valori.

> 🚨 Una nota di cautela
>
> Quando si lavora con questo insieme di dati, si scriverà un codice che calcola qualcosa dal testo senza dover leggere o analizzare il testo da soli. Questa è l'essenza di NLP, interpretare il significato o il sentiment senza che lo faccia un essere umano. Tuttavia, è possibile che si leggano alcune delle recensioni negative. Non è raccomandabile farlo. Alcune di esse sono recensioni negative sciocche o irrilevanti, come "Il tempo non era eccezionale", qualcosa al di fuori del controllo dell'hotel, o di chiunque, in effetti. Ma c'è anche un lato oscuro in alcune recensioni. A volte le recensioni negative sono razziste, sessiste o antietà. Questo è un peccato, ma è prevedibile in un insieme di dati recuperato da un sito web pubblico. Alcuni recensori lasciano recensioni che si potrebbe trovare sgradevoli, scomode o sconvolgenti. Meglio lasciare che il codice misuri il sentiment piuttosto che leggerle da soli e arrabbiarsi. Detto questo, è una minoranza che scrive queste cose, ma esistono lo stesso.

## Esercizio - Esplorazione dei dati

### Caricare i dati

Per ora l'esame visivo dei dati è sufficiente, adesso si scriverà del codice e si otterranno alcune risposte! Questa sezione utilizza la libreria pandas. Il primo compito è assicurarsi di poter caricare e leggere i dati CSV. La libreria pandas ha un veloce caricatore CSV e il risultato viene inserito in un dataframe, come nelle lezioni precedenti. Il CSV che si sta caricando ha oltre mezzo milione di righe, ma solo 17 colonne. Pandas offre molti modi potenti per interagire con un dataframe, inclusa la possibilità di eseguire operazioni su ogni riga.

Da qui in poi in questa lezione, ci saranno frammenti di codice e alcune spiegazioni del codice e alcune discussioni su cosa significano i risultati. Usare il _notebook.ipynb_ incluso per il proprio codice.

Si inizia con il caricamento del file di dati da utilizzare:

```python
# Carica il CSV con le recensioni degli hotel
import pandas as pd
import time
# importa time per determinare orario di inizio e fine caricamento per poterne calcolare la durata
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df è un 'DataFrame' - assicurarsi di aver scaricato il file nella cartelle data
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Ora che i dati sono stati caricati, si possono eseguire alcune operazioni su di essi. Tenere questo codice nella parte superiore del programma per la parte successiva.

## Esplorare i dati

In questo caso, i dati sono già *puliti*, il che significa che sono pronti per essere lavorati e non ci sono caratteri in altre lingue che potrebbero far scattare algoritmi che si aspettano solo caratteri inglesi.

✅ Potrebbe essere necessario lavorare con dati che richiedono un'elaborazione iniziale per formattarli prima di applicare le tecniche di NLP, ma non questa volta. Se si dovesse, come si gestirebero i caratteri non inglesi?

Si prenda un momento per assicurarsi che una volta caricati, i dati possano essere esplorarati con il codice. È molto facile volersi concentrare sulle colonne `Negative_Review` e `Positive_Review` . Sono pieni di testo naturale per essere elaborato dagli algoritmi di NLP. Ma non è finita qui! Prima di entrare nel NLP e nel sentiment, si dovrebbe seguire il codice seguente per accertarsi se i valori forniti nell'insieme di dati corrispondono ai valori calcolati con pandas.

## Operazioni con dataframe

Il primo compito di questa lezione è verificare se le seguenti asserzioni sono corrette scrivendo del codice che esamini il dataframe (senza modificarlo).

> Come molte attività di programmazione, ci sono diversi modi per completarla, ma un buon consiglio è farlo nel modo più semplice e facile possibile, soprattutto se sarà più facile da capire quando si riesaminerà questo codice in futuro. Con i dataframe, esiste un'API completa che spesso avrà un modo per fare ciò che serve in modo efficiente.

Trattare le seguenti domande come attività di codifica e provare a rispondere senza guardare la soluzione.

1. Stampare la *forma* del dataframe appena caricato (la forma è il numero di righe e colonne)
2. Calcolare il conteggio della frequenza per le nazionalità dei recensori:
   1. Quanti valori distinti ci sono per la colonna `Reviewer_Nationality` e quali sono?
   2. Quale nazionalità del recensore è la più comune nell'insieme di dati (stampare nazione e numero di recensioni)?
   3. Quali sono le prossime 10 nazionalità più frequenti e la loro frequenza?
3. Qual è stato l'hotel più recensito per ciascuna delle 10 nazionalità più recensite?
4. Quante recensioni ci sono per hotel (conteggio della frequenza dell'hotel) nell'insieme di dati?
5. Sebbene sia presente una colonna `Average_Score` per ogni hotel nell'insieme di dati, si può anche calcolare un punteggio medio (ottenendo la media di tutti i punteggi dei recensori nell'insieme di dati per ogni hotel). Aggiungere una nuova colonna al dataframe con l'intestazione della colonna `Calc_Average_Score` che contiene quella media calcolata.
6. Ci sono hotel che hanno lo stesso (arrotondato a 1 decimale) `Average_Score` e `Calc_Average_Score`?
   1. Provare a scrivere una funzione Python che accetta una serie (riga) come argomento e confronta i valori, stampando un messaggio quando i valori non sono uguali. Quindi usare il metodo `.apply()` per elaborare ogni riga con la funzione.
7. Calcolare e stampare quante righe contengono valori di "No Negative" nella colonna `Negative_Review` "
8. Calcolare e stampare quante righe contengono valori di "No Positive" nella colonna `Positive_Review` "
9. Calcolare e stampare quante righe contengono valori di "No Positive" nella colonna `Positive_Review` " **e** valori di "No Negative" nella colonna `Negative_Review`

### Risposte

1. Stampare la *forma* del dataframei appena caricato (la forma è il numero di righe e colonne)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcolare il conteggio della frequenza per le nazionalità dei recensori:

   1. Quanti valori distinti ci sono per la colonna `Reviewer_Nationality` e quali sono?
   2. Quale nazionalità del recensore è la più comune nell'insieme di dati (paese di stampa e numero di recensioni)?

   ```python
   # value_counts() crea un oggetto Series con indice e valori, in questo caso la nazione 
   # e la frequenza con la quale si manifestano nella nazionalità del recensore
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # stampa la prima e ultima riga della Series. Modificare in nationality_freq.to_string() per stampare tutti i dati
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

   3. Quali sono le prossime 10 nazionalità più frequenti e la loro frequenza?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notare che c'è uno spazio davanti ai valori, strip() lo rimuove per la stampa
      # Quale sono le 10 nazionalità più comuni e la loro frequenza?
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

3. Qual è stato l'hotel più recensito per ciascuna delle 10 nazionalità più recensite?

   ```python
   # Qual è stato l'hotel più recensito per ciascuna delle 10 nazionalità più recensite
   # In genere con pandas si cerca di evitare un ciclo esplicito, ma si vuole mostrare come si crea un
   # nuovo dataframe usando criteri (non fare questo con un grande volume di dati in quanto potrebbe essere molto lento)
   for nat in nationality_freq[:10].index:
      # Per prima cosa estrarre tutte le righe che corrispondono al criterio in un nuovo dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Ora ottenere la frequenza per l'hotel
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

4. Quante recensioni ci sono per hotel (conteggio della frequenza dell'hotel) nell'insieme di dati?

   ```python
   # Per prima cosa creare un nuovo dataframe in base a quello vecchio, togliendo le colonne non necessarie
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)

   # Raggruppre le righe per Hotel_Name, conteggiarle e inserire il risultato in una nuova colonna Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')

   # Eliminare tutte le righe duplicate
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df)
   ```
   
   |          Hotel_Name (Nome Hotel)           | Total_Number_of_Reviews (Numero totale di recensioni) | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------------------------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |                         9086                          |        4789         |
   |    Park Plaza Westminster Bridge Londra    |                         12158                         |        4169         |
   |   Copthorne Tara Hotel London Kensington   |                         7105                          |        3578         |
   |                    ...                     |                          ...                          |         ...         |
   |       Mercure Paris Porte d'Orléans        |                          110                          |         10          |
   |                Hotel Wagner                |                          135                          |         10          |
   |            Hotel Gallitzinberg             |                          173                          |          8          |
   

   Si potrebbe notare che il *conteggio nell'insieme di dati* non corrisponde al valore in `Total_Number_of_Reviews`. Non è chiaro se questo valore nell'insieme di dati rappresentasse il numero totale di recensioni che l'hotel aveva, ma non tutte sono state recuperate o qualche altro calcolo. `Total_Number_of_Reviews` non viene utilizzato nel modello a causa di questa non chiarezza.

5. Sebbene sia presente una colonna `Average_Score` per ogni hotel nell'insieme di dati, si può anche calcolare un punteggio medio (ottenendo la media di tutti i punteggi dei recensori nell'insieme di dati per ogni hotel). Aggiungere una nuova colonna al dataframe con l'intestazione della colonna `Calc_Average_Score` che contiene quella media calcolata. Stampare le colonne `Hotel_Name`, `Average_Score` e `Calc_Average_Score`.

   ```python
   # definisce una funzione che ottiene una riga ed esegue alcuni calcoli su di essa
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]

   # 'mean' è la definizione matematica per 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)

   # Aggiunge una nuova colonna con la differenza tra le due medie di punteggio
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)

   # Crea un df senza tutti i duplicati di Hotel_Name (quindi una sola riga per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])

   # Ordina il dataframe per trovare la differnza più bassa e più alta per il punteggio medio
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])

   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Ci si potrebbe anche chiedere del valore `Average_Score` e perché a volte è diverso dal punteggio medio calcolato. Poiché non è possibile sapere perché alcuni valori corrispondano, ma altri hanno una differenza, in questo caso è più sicuro utilizzare i punteggi delle recensioni a disposizione per calcolare autonomamente la media. Detto questo, le differenze sono solitamente molto piccole, ecco gli hotel con la maggiore deviazione dalla media dell'insieme di dati e dalla media calcolata:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                      Hotel_Name (Nome Hotel) |
   | :----------------------: | :-----------: | :----------------: | -------------------------------------------: |
   |           -0,8           |      7,7      |        8,5         |                   Best Western Hotel Astoria |
   |           -0,7           |      8,8      |        9,5         | Hotel Stendhal Place Vend me Parigi MGallery |
   |           -0,7           |      7,5      |        8.2         |                Mercure Paris Porte d'Orléans |
   |           -0,7           |      7,9      |        8,6         |              Renaissance Paris Vendome Hotel |
   |           -0,5           |      7,0      |        7,5         |                          Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                          ... |
   |           0.7            |      7,5      |        6.8         |      Mercure Paris Op ra Faubourg Montmartre |
   |           0,8            |      7,1      |        6.3         |       Holiday Inn Paris Montparnasse Pasteur |
   |           0,9            |      6.8      |        5,9         |                                Villa Eugenia |
   |           0,9            |      8,6      |        7,7         |   MARCHESE Faubourg St Honor Relais Ch teaux |
   |           1,3            |      7,2      |        5,9         |                           Kube Hotel Ice Bar |

   Con un solo hotel con una differenza di punteggio maggiore di 1, significa che probabilmente si può ignorare la differenza e utilizzare il punteggio medio calcolato.

6. Calcolare e stampare quante righe hanno la colonna `Negative_Review` valori di "No Negative"

7. Calcolare e stampare quante righe hanno la colonna `Positive_Review` valori di "No Positive"

8. Calcolare e stampare quante righe hanno la colonna `Positive_Review` valori di "No Positive" **e** `Negative_Review` valori di "No Negative"

   ```python
   # con funzini lambda:
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

## Un'altra strada

Un altro modo per contare gli elementi senza Lambda e utilizzare sum per contare le righe:

```python
# senza funzioni lambda (usando un misto di notazioni per mostrare che si possono usare entrambi)
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

Si potrebbe aver notato che ci sono 127 righe che hanno entrambi i valori "No Negative" e "No Positive" rispettivamente per le colonne `Negative_Review` e `Positive_Review` . Ciò significa che il recensore ha assegnato all'hotel un punteggio numerico, ma ha rifiutato di scrivere una recensione positiva o negativa. Fortunatamente questa è una piccola quantità di righe (127 su 515738, o 0,02%), quindi probabilmente non distorcerà il modello o i risultati in una direzione particolare, ma si potrebbe non aspettarsi che un insieme di dati di recensioni abbia righe con nessuna recensione, quindi vale la pena esplorare i dati per scoprire righe come questa.

Ora che si è esplorato l'insieme di dati, nella prossima lezione si filtreranno i dati e si aggiungerà un'analisi del sentiment.

---

## 🚀 Sfida

Questa lezione dimostra, come visto nelle lezioni precedenti, quanto sia di fondamentale importanza comprendere i dati e le loro debolezze prima di eseguire operazioni su di essi. I dati basati su testo, in particolare, sono oggetto di un attento esame. Esaminare vari insiemi di dati contenenti principalmente testo e vedere se si riesce a scoprire aree che potrebbero introdurre pregiudizi o sentiment distorti in un modello.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/?loc=it)

## Revisione e Auto Apprendimento

Seguire [questo percorso di apprendimento su NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) per scoprire gli strumenti da provare durante la creazione di modelli vocali e di testo.

## Compito

[NLTK](assignment.it.md)
