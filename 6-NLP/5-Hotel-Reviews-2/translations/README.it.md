# Analisi del sentiment con recensioni di hotel

Ora che si è  esplorato in dettaglio l'insieme di dati, è il momento di filtrare le colonne e quindi utilizzare le tecniche NLP sull'insieme di dati per ottenere nuove informazioni sugli hotel.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/?loc=it)

### Operazioni di Filtraggio e Analisi del Sentiment

Come probabilmente notato, l'insieme di dati presenta alcuni problemi. Alcune colonne sono piene di informazioni inutili, altre sembrano errate. Se sono corrette, non è chiaro come sono state calcolate e le risposte non possono essere verificate in modo indipendente dai propri calcoli.

## Esercizio: un po' più di elaborazione dei dati

Occorre pulire un po' di più i dati. Si aggiungono colonne che saranno utili in seguito, si modificano i valori in altre colonne e si eliminano completamente determinate colonne.

1. Elaborazione iniziale colonne

   1. Scartare `lat` e `lng`

   2. Sostituire i valori `Hotel_Address` con i seguenti valori (se l'indirizzo contiene lo stesso della città e del paese, si cambia solo con la città e la nazione).

      Queste sono le uniche città e nazioni nell'insieme di dati:

      Amsterdam, Netherlands

      Barcelona, Spain

      London, United Kingdom

      Milan, Italy

      Paris, France

      Vienna, Austria

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria"

      # Sostituisce tutti gli indirizzi con una forma ridotta più utile
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # La somma di value_counts() dovrebbe sommarsi al numero totale recensioni
      print(df["Hotel_Address"].value_counts())
      ```

      Ora si possono interrogare i dati a livello di nazione:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name (Nome Hotel) |
      | :--------------------- | :---------------------: |
      | Amsterdam, Paesi Bassi |           105           |
      | Barcellona, Spagna     |           211           |
      | Londra, Regno Unito    |           400           |
      | Milano, Italia         |           162           |
      | Parigi, Francia        |           458           |
      | Vienna, Austria        |           158           |

2. Elaborazione colonne di meta-recensione dell'hotel

1. Eliminare `Additional_Number_of_Scoring`

1. Sostituire `Total_Number_of_Reviews` con il numero totale di recensioni per quell'hotel che sono effettivamente nell'insieme di dati

1. Sostituire `Average_Score` con il punteggio calcolato via codice

```python
# Elimina `Additional_Number_of_Scoring`
df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
# Sostituisce `Total_Number_of_Reviews` e `Average_Score` con i propri valori calcolati
df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
```

3. Elaborazione delle colonne di recensione

   1. Eliminare `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` e `days_since_review`

   2. Mantenere `Reviewer_Score`, `Negative_Review` e `Positive_Review` così come sono

   3. Conservare i `Tags` per ora

   - Si faranno alcune operazioni di filtraggio aggiuntive sui tag nella prossima sezione, successivamente i tag verranno eliminati

4. Elaborazione delle colonne del recensore

1. Scartare `Total_Number_of_Reviews_Reviewer_Has_Given`

2. Mantenere `Reviewer_Nationality`

### Colonne tag

Le colonne `Tag` sono problematiche in quanto si tratta di un elenco (in formato testo) memorizzato nella colonna. Purtroppo l'ordine e il numero delle sottosezioni in questa colonna non sono sempre gli stessi. È difficile per un essere umano identificare le frasi corrette a cui essere interessato, perché ci sono 515.000 righe e 1427 hotel e ognuno ha opzioni leggermente diverse che un recensore potrebbe scegliere. È qui che la NLP brilla. Si può scansionare il testo, trovare le frasi più comuni e contarle.

Purtroppo non interessano parole singole, ma frasi composte da più parole (es. *Viaggio di lavoro*). L'esecuzione di un algoritmo di distribuzione della frequenza a più parole su così tanti dati (6762646 parole) potrebbe richiedere una quantità straordinaria di tempo, ma senza guardare i dati, sembrerebbe che sia una spesa necessaria. È qui che l'analisi dei dati esplorativi diventa utile, perché si è visto un esempio di tag come `["Business trip", "Solo traveler", "Single Room", "Stayed 5 nights", "Submitted from a mobile device"]` , si può iniziare a chiedersi se è possibile ridurre notevolmente l'elaborazione da fare. Fortunatamente lo è, ma prima occorre seguire alcuni passaggi per accertare i tag di interesse.

### Filtraggio tag

Ricordare che l'obiettivo dell'insieme di dati è aggiungere il sentiment e le colonne che aiuteranno a scegliere l'hotel migliore (per se stessi o forse per un cliente che incarica di creare un bot di raccomandazione dell'hotel). Occorre chiedersi se i tag sono utili o meno nell'insieme di dati finale. Ecco un'interpretazione (se serve l'insieme di dati per altri motivi diversi tag potrebbero rimanere dentro/fuori dalla selezione):

1. Il tipo di viaggio è rilevante e dovrebbe rimanere
2. Il tipo di gruppo di ospiti è importante e dovrebbe rimanere
3. Il tipo di camera, suite o monolocale in cui ha soggiornato l'ospite è irrilevante (tutti gli hotel hanno praticamente le stesse stanze)
4. Il dispositivo su cui è stata inviata la recensione è irrilevante
5. Il numero di notti in cui il recensore ha soggiornato *potrebbe* essere rilevante se si attribuisce a soggiorni più lunghi un gradimento maggiore per l'hotel, ma è una forzatura e probabilmente irrilevante

In sintesi, si **mantengono 2 tipi di tag e si rimuove il resto**.

Innanzitutto, non si vogliono contare i tag finché non sono in un formato migliore, quindi ciò significa rimuovere le parentesi quadre e le virgolette. Si può fare in diversi modi, ma serve il più veloce in quanto potrebbe richiedere molto tempo per elaborare molti dati. Fortunatamente, pandas ha un modo semplice per eseguire ciascuno di questi passaggi.

```Python
# Rimuove le parentesi quadre di apertura e chiusura
df.Tags = df.Tags.str.strip("[']")
# rimuove anche tutte le virgolette
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Ogni tag diventa qualcosa come: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`.

Successivamente si manifesta un problema. Alcune recensioni, o righe, hanno 5 colonne, altre 3, altre 6. Questo è il risultato di come è stato creato l'insieme di dati ed è difficile da risolvere. Si vuole ottenere un conteggio della frequenza di ogni frase, ma sono in ordine diverso in ogni recensione, quindi il conteggio potrebbe essere disattivato e un hotel potrebbe non ricevere un tag assegnato per ciò che meritava.

Si utilizzerà invece l'ordine diverso a proprio vantaggio, perché ogni tag è composto da più parole ma anche separato da una virgola! Il modo più semplice per farlo è creare 6 colonne temporanee con ogni tag inserito nella colonna corrispondente al suo ordine nel tag. Quindi si uniscono le 6 colonne in una grande colonna e si esegue il metodo `value_counts()` sulla colonna risultante. Stampandolo, si vedrà che c'erano 2428 tag univoci. Ecco un piccolo esempio:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Alcuni dei tag comuni come `Submitted from a mobile device` non sono di alcuna utilità, quindi potrebbe essere una cosa intelligente rimuoverli prima di contare l'occorrenza della frase, ma è un'operazione così veloce che si possono lasciare e ignorare.

### Rimozione della durata dai tag di soggiorno

La rimozione di questi tag è il passaggio 1, riduce leggermente il numero totale di tag da considerare. Notare che non si rimuovono dall'insieme di dati, si sceglie semplicemente di rimuoverli dalla considerazione come valori da contare/mantenere nell'insieme di dati delle recensioni.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

C'è una grande varietà di camere, suite, monolocali, appartamenti e così via. Significano tutti più o meno la stessa cosa e non sono rilevanti allo scopo, quindi si rimuovono dalla considerazione.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Infine, e questo è delizioso (perché non ha richiesto molta elaborazione), rimarranno i seguenti tag *utili*:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

Si potrebbe obiettare che `Travellers with friends` (Viaggiatori con amici) è più o meno lo stesso di `Group` (Gruppo), e sarebbe giusto combinare i due come fatto sopra. Il codice per identificare i tag corretti è [il notebook Tags](../solution/1-notebook.ipynb).

Il passaggio finale consiste nel creare nuove colonne per ciascuno di questi tag. Quindi, per ogni riga di recensione, se la colonna `Tag` corrisponde a una delle nuove colonne, aggiungere 1, in caso contrario aggiungere 0. Il risultato finale sarà un conteggio di quanti recensori hanno scelto questo hotel (in aggregato) per, ad esempio, affari o piacere, o per portare un animale domestico, e questa è un'informazione utile quando consiglia un hotel.

```python
# Elabora Tags in nuove colonne
# Il file Hotel_Reviews_Tags.py, identifica i tag più importanti
# Leisure trip, Couple, Solo traveler, Business trip, Group combinato con Travelers with friends,
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Salvare il file.

Infine, salvare l'insieme di dati così com'è ora con un nuovo nome.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Salvataggio del nuovo file dati con le colonne calcolate
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operazioni di Analisi del Sentiment

In questa sezione finale, si applicherà l'analisi del sentiment alle colonne di recensione e si salveranno i risultati in un insieme di dati.

## Esercizio: caricare e salvare i dati filtrati

Tenere presente che ora si sta caricando l'insieme di dati filtrato che è stato salvato nella sezione precedente, **non** quello originale.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Carica le recensioni di hotel filtrate dal CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# Il proprio codice andrà aggiunto qui


# Infine ricordarsi di salvare le recensioni di hotel con i nuovi dati NLP aggiunti
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Rimozione delle stop word

Se si dovesse eseguire l'analisi del sentiment sulle colonne delle recensioni negative e positive, potrebbe volerci molto tempo. Testato su un potente laptop di prova con CPU veloce, ci sono voluti 12 - 14 minuti a seconda della libreria di sentiment utilizzata. È un tempo (relativamente) lungo, quindi vale la pena indagare se può essere accelerato.

Il primo passo è rimuovere le stop word, o parole inglesi comuni che non cambiano il sentiment di una frase. Rimuovendole, l'analisi del sentiment dovrebbe essere eseguita più velocemente, ma non essere meno accurata (poiché le stop word non influiscono sul sentiment, ma rallentano l'analisi).

La recensione negativa più lunga è stata di 395 parole, ma dopo aver rimosso le stop word, è di 195 parole.

Anche la rimozione delle stop word è un'operazione rapida, poiché la rimozione di esse da 2 colonne di recensione su 515.000 righe ha richiesto 3,3 secondi sul dispositivo di test. Potrebbe volerci un po' più o meno tempo a seconda della velocità della CPU del proprio dispositivo, della RAM, del fatto che si abbia o meno un SSD e alcuni altri fattori. La relativa brevità dell'operazione significa che se migliora il tempo di analisi del sentiment, allora vale la pena farlo.

```python
from nltk.corpus import stopwords

# Carica le recensioni di hotel da CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Rimuove le stop word - potrebbe essere lento quando c'è molto testo!
# Ryan Han (ryanxjhan su Kaggle) ha un gran post riguardo al misurare le prestazioni di diversi approcci per la rimozione delle stop word
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # si usa l'approccio raccomandato da Ryan
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Rimuove le stop word da entrambe le colonne
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Esecuzione dell'analisi del sentiment

Ora si dovrebbe calcolare l'analisi del sentiment per le colonne di recensioni negative e positive e memorizzare il risultato in 2 nuove colonne. Il test del sentiment sarà quello di confrontarlo con il punteggio del recensore per la stessa recensione. Ad esempio, se il sentiment ritiene che la recensione negativa abbia avuto un sentiment pari a 1 (giudizio estremamente positivo) e un sentiment positivo della recensione pari a 1, ma il recensore ha assegnato all'hotel il punteggio più basso possibile, allora il testo della recensione non corrisponde al punteggio, oppure l'analizzatore del sentiment non è stato in grado di riconoscere correttamente il sentiment. Ci si dovrebbe aspettare che alcuni punteggi del sentiment siano completamente sbagliati, e spesso ciò sarà spiegabile, ad esempio la recensione potrebbe essere estremamente sarcastica "Certo che mi è piaciuto dormire in una stanza senza riscaldamento" e l'analizzatore del sentimento pensa che sia un sentimento positivo, anche se un un lettore umano avrebbe rilevato il sarcasmo.

NLTK fornisce diversi analizzatori di sentiment con cui imparare e si possono sostituire e vedere se il sentiment è più o meno accurato. Qui viene utilizzata l'analisi del sentiment di VADER.

> Hutto, CJ & Gilbert, EE (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Ottava Conferenza Internazionale su Weblog e Social Media (ICWSM-14). Ann Arbor, MI, giugno 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Crea l'analizzatore di sentiment vader (ce ne sono altri in NLTK che si possono provare)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# Ci sono tre possibilità di input per un recensore:
# Potrebbe essere "No Negative", nel qual caso ritorna 0
# Potrebbe essere "No Positive", nel qual caso ritorna 0
# Potrebbe essere una recensione, nel qual caso calcola il sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Più avanti nel programma, quando si è pronti per calcolare il sentiment, lo si può applicare a ciascuna recensione come segue:

```python
# Aggiunge una colonna di sentiment negativa e positiva
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Questo richiede circa 120 secondi sul computer utilizzato, ma varierà per ciascun computer. Se si vogliono stampare i risultati e vedere se il sentiment corrisponde alla recensione:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

L'ultima cosa da fare con il file prima di utilizzarlo nella sfida è salvarlo! Si dovrrebbe anche considerare di riordinare tutte le nuove colonne in modo che sia facile lavorarci (per un essere umano, è un cambiamento estetico).

```python
# Riordina le colonne (E' un estetismo ma facilita l'esplorazione successiva dei dati)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Si dovrebbe eseguire l'intero codice per [il notebook di analisi](../solution/3-notebook.ipynb) (dopo aver eseguito [il notebook di filtraggio](../solution/1-notebook.ipynb) per generare il file Hotel_Reviews_Filtered.csv).

Per riepilogare, i passaggi sono:

1. Il file del'insieme di dati originale **Hotel_Reviews.csv** è stato esplorato nella lezione precedente con [il notebook explorer](../../4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv viene filtrato [dal notebook di filtraggio](../solution/1-notebook.ipynb) risultante in **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv viene elaborato dal [notebook di analisi del sentiment](../solution/3-notebook.ipynb) risultante in **Hotel_Reviews_NLP.csv**
4. Usare Hotel_Reviews_NLP.csv nella Sfida NLP di seguito

### Conclusione

Quando si è iniziato, si disponeva di un insieme di dati con colonne e dati, ma non tutto poteva essere verificato o utilizzato. Si sono esplorati i dati, filtrato ciò che non serve, convertito i tag in qualcosa di utile, calcolato le proprie medie, aggiunto alcune colonne di sentiment e, si spera, imparato alcune cose interessanti sull'elaborazione del testo naturale.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/?loc=it)

## Sfida

Ora che si è analizzato il proprio insieme di dati per il sentiment, vedere se si possono usare le strategie apprese in questo programma di studi (clustering, forse?) per determinare modelli intorno al sentiment.

## recensione e Auto Apprendimento

Seguire [questo modulo di apprendimento](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) per saperne di più e utilizzare diversi strumenti per esplorare il sentiment nel testo.

## Compito

[Provare un insieme di dati diverso](assignment.it.md)
