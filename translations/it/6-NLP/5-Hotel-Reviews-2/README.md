<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T22:42:40+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "it"
}
-->
# Analisi del sentiment con recensioni di hotel

Ora che hai esplorato il dataset in dettaglio, è il momento di filtrare le colonne e utilizzare tecniche di NLP sul dataset per ottenere nuove informazioni sugli hotel.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Operazioni di filtraggio e analisi del sentiment

Come avrai notato, il dataset presenta alcuni problemi. Alcune colonne contengono informazioni inutili, altre sembrano errate. Anche se fossero corrette, non è chiaro come siano state calcolate, e le risposte non possono essere verificate indipendentemente con i tuoi calcoli.

## Esercizio: un po' più di elaborazione dei dati

Pulisci ulteriormente i dati. Aggiungi colonne che saranno utili in seguito, modifica i valori in altre colonne e elimina completamente alcune colonne.

1. Elaborazione iniziale delle colonne

   1. Elimina `lat` e `lng`

   2. Sostituisci i valori di `Hotel_Address` con i seguenti valori (se l'indirizzo contiene il nome della città e del paese, cambialo con solo la città e il paese).

      Queste sono le uniche città e paesi presenti nel dataset:

      Amsterdam, Paesi Bassi

      Barcellona, Spagna

      Londra, Regno Unito

      Milano, Italia

      Parigi, Francia

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
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Ora puoi interrogare i dati a livello di paese:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Paesi Bassi |    105     |
      | Barcellona, Spagna     |    211     |
      | Londra, Regno Unito    |    400     |
      | Milano, Italia         |    162     |
      | Parigi, Francia        |    458     |
      | Vienna, Austria        |    158     |

2. Elaborazione delle colonne meta-recensione degli hotel

   1. Elimina `Additional_Number_of_Scoring`

   2. Sostituisci `Total_Number_of_Reviews` con il numero totale di recensioni per quell'hotel effettivamente presenti nel dataset 

   3. Sostituisci `Average_Score` con il punteggio calcolato da noi

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Elaborazione delle colonne delle recensioni

   1. Elimina `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` e `days_since_review`

   2. Mantieni `Reviewer_Score`, `Negative_Review` e `Positive_Review` così come sono
     
   3. Mantieni `Tags` per ora

     - Effettueremo ulteriori operazioni di filtraggio sui tag nella prossima sezione e poi i tag verranno eliminati

4. Elaborazione delle colonne dei recensori

   1. Elimina `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Mantieni `Reviewer_Nationality`

### Colonne dei tag

La colonna `Tag` è problematica poiché è una lista (in formato testo) memorizzata nella colonna. Purtroppo l'ordine e il numero di sottosezioni in questa colonna non sono sempre gli stessi. È difficile per un essere umano identificare le frasi corrette di interesse, perché ci sono 515.000 righe e 1427 hotel, e ciascuno ha opzioni leggermente diverse che un recensore potrebbe scegliere. Qui entra in gioco il NLP. Puoi analizzare il testo e trovare le frasi più comuni, contando la loro frequenza.

Purtroppo, non siamo interessati a singole parole, ma a frasi composte da più parole (ad esempio *Viaggio di lavoro*). Eseguire un algoritmo di distribuzione di frequenza su frasi composte su una quantità così grande di dati (6762646 parole) potrebbe richiedere un tempo straordinario, ma senza guardare i dati, sembrerebbe una spesa necessaria. Qui l'analisi esplorativa dei dati è utile, perché hai visto un campione dei tag come `[' Viaggio di lavoro  ', ' Viaggiatore solitario ', ' Camera singola ', ' Soggiornato 5 notti ', ' Inviato da un dispositivo mobile ']`, puoi iniziare a chiederti se è possibile ridurre notevolmente l'elaborazione necessaria. Fortunatamente, è possibile - ma prima devi seguire alcuni passaggi per determinare i tag di interesse.

### Filtraggio dei tag

Ricorda che l'obiettivo del dataset è aggiungere sentiment e colonne che ti aiutino a scegliere il miglior hotel (per te stesso o magari per un cliente che ti chiede di creare un bot per raccomandare hotel). Devi chiederti se i tag sono utili o meno nel dataset finale. Ecco un'interpretazione (se avessi bisogno del dataset per altri motivi, potrebbero esserci tag diversi da includere/escludere):

1. Il tipo di viaggio è rilevante e dovrebbe rimanere
2. Il tipo di gruppo di ospiti è importante e dovrebbe rimanere
3. Il tipo di camera, suite o studio in cui l'ospite ha soggiornato è irrilevante (tutti gli hotel hanno sostanzialmente le stesse camere)
4. Il dispositivo su cui è stata inviata la recensione è irrilevante
5. Il numero di notti di soggiorno *potrebbe* essere rilevante se attribuisci soggiorni più lunghi al fatto che l'ospite abbia apprezzato di più l'hotel, ma è un'ipotesi debole e probabilmente irrilevante

In sintesi, **mantieni 2 tipi di tag e rimuovi gli altri**.

Per prima cosa, non vuoi contare i tag finché non sono in un formato migliore, quindi ciò significa rimuovere le parentesi quadre e le virgolette. Puoi farlo in diversi modi, ma vuoi il più veloce possibile poiché potrebbe richiedere molto tempo per elaborare una grande quantità di dati. Fortunatamente, pandas offre un modo semplice per eseguire ciascuno di questi passaggi.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Ogni tag diventa qualcosa come: `Viaggio di lavoro, Viaggiatore solitario, Camera singola, Soggiornato 5 notti, Inviato da un dispositivo mobile`. 

Successivamente troviamo un problema. Alcune recensioni, o righe, hanno 5 colonne, altre 3, altre 6. Questo è il risultato di come è stato creato il dataset ed è difficile da correggere. Vuoi ottenere un conteggio di frequenza di ogni frase, ma sono in ordine diverso in ogni recensione, quindi il conteggio potrebbe essere errato e un hotel potrebbe non ricevere un tag assegnato che meriterebbe.

Invece userai l'ordine diverso a nostro vantaggio, poiché ogni tag è composto da più parole ma anche separato da una virgola! Il modo più semplice per farlo è creare 6 colonne temporanee con ciascun tag inserito nella colonna corrispondente al suo ordine nel tag. Puoi quindi unire le 6 colonne in una grande colonna e eseguire il metodo `value_counts()` sulla colonna risultante. Stampando il risultato, vedrai che ci sono 2428 tag unici. Ecco un piccolo campione:

| Tag                            | Conteggio |
| ------------------------------ | --------- |
| Viaggio di piacere             | 417778    |
| Inviato da un dispositivo mobile | 307640 |
| Coppia                         | 252294    |
| Soggiornato 1 notte            | 193645    |
| Soggiornato 2 notti            | 133937    |
| Viaggiatore solitario          | 108545    |
| Soggiornato 3 notti            | 95821     |
| Viaggio di lavoro              | 82939     |
| Gruppo                         | 65392     |
| Famiglia con bambini piccoli   | 61015     |
| Soggiornato 4 notti            | 47817     |
| Camera doppia                  | 35207     |
| Camera doppia standard         | 32248     |
| Camera doppia superior         | 31393     |
| Famiglia con bambini grandi    | 26349     |
| Camera doppia deluxe           | 24823     |
| Camera doppia o twin           | 22393     |
| Soggiornato 5 notti            | 20845     |
| Camera doppia o twin standard  | 17483     |
| Camera doppia classica         | 16989     |
| Camera doppia o twin superior  | 13570     |
| 2 camere                       | 12393     |

Alcuni dei tag comuni come `Inviato da un dispositivo mobile` non sono utili per noi, quindi potrebbe essere intelligente rimuoverli prima di contare l'occorrenza delle frasi, ma è un'operazione così veloce che puoi lasciarli e ignorarli.

### Rimozione dei tag relativi alla durata del soggiorno

Rimuovere questi tag è il primo passo, riduce leggermente il numero totale di tag da considerare. Nota che non li rimuovi dal dataset, ma scegli di non considerarli come valori da contare/mantenere nel dataset delle recensioni.

| Durata del soggiorno | Conteggio |
| --------------------- | --------- |
| Soggiornato 1 notte   | 193645    |
| Soggiornato 2 notti   | 133937    |
| Soggiornato 3 notti   | 95821     |
| Soggiornato 4 notti   | 47817     |
| Soggiornato 5 notti   | 20845     |
| Soggiornato 6 notti   | 9776      |
| Soggiornato 7 notti   | 7399      |
| Soggiornato 8 notti   | 2502      |
| Soggiornato 9 notti   | 1293      |
| ...                   | ...       |

Ci sono una grande varietà di camere, suite, studi, appartamenti e così via. Tutti significano più o meno la stessa cosa e non sono rilevanti per te, quindi rimuovili dalla considerazione.

| Tipo di camera                  | Conteggio |
| ------------------------------- | --------- |
| Camera doppia                   | 35207     |
| Camera doppia standard          | 32248     |
| Camera doppia superior          | 31393     |
| Camera doppia deluxe            | 24823     |
| Camera doppia o twin            | 22393     |
| Camera doppia o twin standard   | 17483     |
| Camera doppia classica          | 16989     |
| Camera doppia o twin superior   | 13570     |

Infine, e questo è interessante (perché non ha richiesto molta elaborazione), rimarrai con i seguenti tag *utili*:

| Tag                                           | Conteggio |
| --------------------------------------------- | --------- |
| Viaggio di piacere                            | 417778    |
| Coppia                                        | 252294    |
| Viaggiatore solitario                         | 108545    |
| Viaggio di lavoro                             | 82939     |
| Gruppo (combinato con Viaggiatori con amici)  | 67535     |
| Famiglia con bambini piccoli                  | 61015     |
| Famiglia con bambini grandi                   | 26349     |
| Con un animale domestico                      | 1405      |

Si potrebbe sostenere che `Viaggiatori con amici` sia più o meno lo stesso di `Gruppo`, e sarebbe giusto combinarli come sopra. Il codice per identificare i tag corretti si trova nel [notebook dei tag](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

L'ultimo passo è creare nuove colonne per ciascuno di questi tag. Poi, per ogni riga di recensione, se la colonna `Tag` corrisponde a una delle nuove colonne, aggiungi un 1, altrimenti aggiungi uno 0. Il risultato finale sarà un conteggio di quanti recensori hanno scelto questo hotel (in aggregato) per, ad esempio, lavoro vs piacere, o per portare un animale domestico, e queste sono informazioni utili per raccomandare un hotel.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
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

### Salva il tuo file

Infine, salva il dataset così com'è ora con un nuovo nome.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operazioni di analisi del sentiment

In questa sezione finale, applicherai l'analisi del sentiment alle colonne delle recensioni e salverai i risultati in un dataset.

## Esercizio: carica e salva i dati filtrati

Nota che ora stai caricando il dataset filtrato che è stato salvato nella sezione precedente, **non** il dataset originale.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Rimozione delle stop words

Se dovessi eseguire l'analisi del sentiment sulle colonne delle recensioni negative e positive, potrebbe richiedere molto tempo. Testato su un potente laptop con CPU veloce, ha impiegato 12-14 minuti a seconda della libreria di sentiment utilizzata. È un tempo (relativamente) lungo, quindi vale la pena indagare se può essere accelerato.

Rimuovere le stop words, ovvero parole comuni in inglese che non cambiano il sentiment di una frase, è il primo passo. Rimuovendole, l'analisi del sentiment dovrebbe essere più veloce, ma non meno accurata (poiché le stop words non influenzano il sentiment, ma rallentano l'analisi).

La recensione negativa più lunga era di 395 parole, ma dopo la rimozione delle stop words, è di 195 parole.

Rimuovere le stop words è anche un'operazione veloce: rimuovere le stop words da 2 colonne di recensioni su 515.000 righe ha impiegato 3,3 secondi sul dispositivo di test. Potrebbe richiedere leggermente più o meno tempo a seconda della velocità della CPU del tuo dispositivo, della RAM, se hai un SSD o meno, e di altri fattori. La relativa brevità dell'operazione significa che, se migliora il tempo dell'analisi del sentiment, allora vale la pena farlo.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Esecuzione dell'analisi del sentiment
Ora dovresti calcolare l'analisi del sentiment per entrambe le colonne delle recensioni negative e positive, e memorizzare il risultato in 2 nuove colonne. Il test del sentiment sarà confrontarlo con il punteggio dato dal recensore per la stessa recensione. Ad esempio, se l'analisi del sentiment rileva che una recensione negativa ha un sentiment di 1 (sentiment estremamente positivo) e una recensione positiva ha un sentiment di 1, ma il recensore ha dato all'hotel il punteggio più basso possibile, allora o il testo della recensione non corrisponde al punteggio, oppure l'analizzatore di sentiment non è stato in grado di riconoscere correttamente il sentiment. Dovresti aspettarti che alcuni punteggi di sentiment siano completamente sbagliati, e spesso ciò sarà spiegabile, ad esempio la recensione potrebbe essere estremamente sarcastica: "Ovviamente ADORO dormire in una stanza senza riscaldamento", e l'analizzatore di sentiment potrebbe interpretarlo come un sentiment positivo, anche se un essere umano leggendo capirebbe che si tratta di sarcasmo.

NLTK fornisce diversi analizzatori di sentiment con cui fare pratica, e puoi sostituirli per vedere se il sentiment è più o meno accurato. Qui viene utilizzata l'analisi del sentiment VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, Giugno 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Successivamente, nel tuo programma, quando sei pronto per calcolare il sentiment, puoi applicarlo a ogni recensione come segue:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Questo richiede circa 120 secondi sul mio computer, ma il tempo può variare a seconda del computer. Se vuoi stampare i risultati e verificare se il sentiment corrisponde alla recensione:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

L'ultima cosa da fare con il file prima di utilizzarlo nella sfida è salvarlo! Dovresti anche considerare di riordinare tutte le nuove colonne in modo che siano facili da gestire (per un essere umano, è un cambiamento puramente estetico).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Dovresti eseguire l'intero codice per [il notebook di analisi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (dopo aver eseguito [il notebook di filtraggio](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) per generare il file Hotel_Reviews_Filtered.csv).

Per ricapitolare, i passaggi sono:

1. Il file del dataset originale **Hotel_Reviews.csv** è stato esplorato nella lezione precedente con [il notebook di esplorazione](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv è stato filtrato con [il notebook di filtraggio](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) risultando in **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv è stato elaborato con [il notebook di analisi del sentiment](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) risultando in **Hotel_Reviews_NLP.csv**
4. Usa Hotel_Reviews_NLP.csv nella sfida NLP qui sotto

### Conclusione

Quando hai iniziato, avevi un dataset con colonne e dati, ma non tutto poteva essere verificato o utilizzato. Hai esplorato i dati, filtrato ciò che non ti serviva, convertito tag in qualcosa di utile, calcolato le tue medie, aggiunto alcune colonne di sentiment e, si spera, imparato alcune cose interessanti sull'elaborazione del testo naturale.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Sfida

Ora che hai analizzato il tuo dataset per il sentiment, vedi se riesci a utilizzare le strategie che hai imparato in questo corso (ad esempio il clustering) per determinare schemi relativi al sentiment.

## Revisione e studio autonomo

Segui [questo modulo Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) per saperne di più e utilizzare strumenti diversi per esplorare il sentiment nel testo.

## Compito

[Prova un dataset diverso](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.