# Analisi del sentiment con recensioni di hotel

Ora che hai esplorato il dataset in dettaglio, è il momento di filtrare le colonne e poi utilizzare tecniche di NLP sul dataset per ottenere nuove informazioni sugli hotel.
## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Operazioni di Filtraggio e Analisi del Sentiment

Come avrai notato, il dataset presenta alcuni problemi. Alcune colonne sono piene di informazioni inutili, altre sembrano errate. Anche se fossero corrette, non è chiaro come siano state calcolate, e le risposte non possono essere verificate indipendentemente dai tuoi calcoli.

## Esercizio: un po' più di elaborazione dei dati

Pulisci i dati un po' di più. Aggiungi colonne che saranno utili in seguito, cambia i valori in altre colonne e elimina completamente alcune colonne.

1. Elaborazione iniziale delle colonne

   1. Elimina `lat` e `lng`

   2. Sostituisci i valori di `Hotel_Address` con i seguenti valori (se l'indirizzo contiene il nome della città e del paese, cambialo con solo la città e il paese).

      Queste sono le uniche città e paesi nel dataset:

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
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. Processa le colonne Meta-review degli Hotel

  1. Elimina `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` con il nostro punteggio calcolato

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Processa le colonne delle recensioni

   1. Elimina `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, puoi iniziare a chiederti se è possibile ridurre notevolmente l'elaborazione che devi fare. Fortunatamente, è possibile - ma prima devi seguire alcuni passaggi per accertarti dei tag di interesse.

### Filtraggio dei tag

Ricorda che l'obiettivo del dataset è aggiungere sentiment e colonne che ti aiuteranno a scegliere il miglior hotel (per te stesso o magari per un cliente che ti chiede di creare un bot di raccomandazione di hotel). Devi chiederti se i tag sono utili o meno nel dataset finale. Ecco un'interpretazione (se avessi bisogno del dataset per altri motivi diversi, potrebbero restare/fuori dalla selezione):

1. Il tipo di viaggio è rilevante e dovrebbe rimanere
2. Il tipo di gruppo di ospiti è importante e dovrebbe rimanere
3. Il tipo di stanza, suite o studio in cui l'ospite ha soggiornato è irrilevante (tutti gli hotel hanno fondamentalmente le stesse stanze)
4. Il dispositivo da cui è stata inviata la recensione è irrilevante
5. Il numero di notti di soggiorno del recensore *potrebbe* essere rilevante se attribuisci soggiorni più lunghi al fatto che gli sia piaciuto di più l'hotel, ma è un'ipotesi azzardata e probabilmente irrilevante

In sintesi, **mantieni 2 tipi di tag e rimuovi gli altri**.

Prima di tutto, non vuoi contare i tag finché non sono in un formato migliore, quindi significa rimuovere le parentesi quadre e le virgolette. Puoi farlo in diversi modi, ma vuoi il più veloce possibile poiché potrebbe richiedere molto tempo per elaborare molti dati. Fortunatamente, pandas ha un modo semplice per eseguire ciascuno di questi passaggi.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Ogni tag diventa qualcosa come: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

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

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

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

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

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

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

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

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` se la colonna corrisponde a una delle nuove colonne, aggiungi 1, altrimenti aggiungi 0. Il risultato finale sarà un conteggio di quanti recensori hanno scelto questo hotel (in aggregato) per, ad esempio, affari vs svago, o per portare un animale domestico, e queste sono informazioni utili quando si raccomanda un hotel.

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

Infine, salva il dataset come è ora con un nuovo nome.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operazioni di Analisi del Sentiment

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

Se dovessi eseguire l'analisi del sentiment sulle colonne delle recensioni negative e positive, potrebbe richiedere molto tempo. Testato su un potente laptop di prova con CPU veloce, ha impiegato 12 - 14 minuti a seconda della libreria di sentiment utilizzata. È un tempo (relativamente) lungo, quindi vale la pena indagare se può essere velocizzato.

Rimuovere le stop words, o parole comuni in inglese che non cambiano il sentiment di una frase, è il primo passo. Rimuovendole, l'analisi del sentiment dovrebbe essere più veloce, ma non meno accurata (poiché le stop words non influenzano il sentiment, ma rallentano l'analisi).

La recensione negativa più lunga era di 395 parole, ma dopo aver rimosso le stop words, è di 195 parole.

La rimozione delle stop words è anche un'operazione veloce, rimuovere le stop words da 2 colonne di recensioni su 515.000 righe ha impiegato 3,3 secondi sul dispositivo di prova. Potrebbe richiedere un po' più o meno tempo per te a seconda della velocità della CPU del tuo dispositivo, della RAM, se hai un SSD o meno, e di altri fattori. La relativa brevità dell'operazione significa che se migliora il tempo dell'analisi del sentiment, allora vale la pena farlo.

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

### Eseguire l'analisi del sentiment

Ora dovresti calcolare l'analisi del sentiment per entrambe le colonne delle recensioni negative e positive, e memorizzare il risultato in 2 nuove colonne. Il test del sentiment sarà confrontarlo con il punteggio del recensore per la stessa recensione. Ad esempio, se il sentiment ritiene che la recensione negativa abbia un sentiment di 1 (sentiment estremamente positivo) e un sentiment della recensione positiva di 1, ma il recensore ha dato all'hotel il punteggio più basso possibile, allora o il testo della recensione non corrisponde al punteggio, oppure l'analizzatore di sentiment non è riuscito a riconoscere correttamente il sentiment. Dovresti aspettarti che alcuni punteggi di sentiment siano completamente sbagliati, e spesso sarà spiegabile, ad esempio la recensione potrebbe essere estremamente sarcastica "Ovviamente HO ADORATO dormire in una stanza senza riscaldamento" e l'analizzatore di sentiment pensa che sia un sentiment positivo, anche se un essere umano leggendo capirebbe che è sarcasmo.

NLTK fornisce diversi analizzatori di sentiment con cui imparare, e puoi sostituirli e vedere se il sentiment è più o meno accurato. Qui viene utilizzata l'analisi del sentiment VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

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

Più avanti nel tuo programma, quando sei pronto per calcolare il sentiment, puoi applicarlo a ogni recensione come segue:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Questo richiede circa 120 secondi sul mio computer, ma varierà su ciascun computer. Se vuoi stampare i risultati e vedere se il sentiment corrisponde alla recensione:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

L'ultima cosa da fare con il file prima di usarlo nella sfida è salvarlo! Dovresti anche considerare di riordinare tutte le tue nuove colonne in modo che siano facili da lavorare (per un essere umano, è un cambiamento cosmetico).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Dovresti eseguire l'intero codice per [il notebook di analisi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (dopo aver eseguito [il tuo notebook di filtraggio](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) per generare il file Hotel_Reviews_Filtered.csv).

Per riepilogare, i passaggi sono:

1. Il file del dataset originale **Hotel_Reviews.csv** è stato esplorato nella lezione precedente con [il notebook di esplorazione](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv è filtrato dal [notebook di filtraggio](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) risultando in **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv è processato dal [notebook di analisi del sentiment](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) risultando in **Hotel_Reviews_NLP.csv**
4. Usa Hotel_Reviews_NLP.csv nella sfida NLP qui sotto

### Conclusione

Quando hai iniziato, avevi un dataset con colonne e dati ma non tutti potevano essere verificati o utilizzati. Hai esplorato i dati, filtrato ciò che non ti serve, convertito i tag in qualcosa di utile, calcolato le tue medie, aggiunto alcune colonne di sentiment e, si spera, imparato alcune cose interessanti sull'elaborazione del testo naturale.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Sfida

Ora che hai analizzato il tuo dataset per il sentiment, vedi se puoi utilizzare le strategie che hai imparato in questo curriculum (clustering, forse?) per determinare modelli attorno al sentiment.

## Revisione & Studio Autonomo

Segui [questo modulo di Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) per saperne di più e utilizzare strumenti diversi per esplorare il sentiment nel testo.
## Compito

[Prova un dataset diverso](assignment.md)

**Avvertenza**: 
Questo documento è stato tradotto utilizzando servizi di traduzione basati su intelligenza artificiale. Sebbene ci impegniamo per garantire l'accuratezza, si prega di essere consapevoli che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua madre dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale umana. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.