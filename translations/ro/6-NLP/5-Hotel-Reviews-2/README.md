<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T17:09:07+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ro"
}
-->
# Analiza sentimentelor cu recenzii de hotel

Acum că ai explorat în detaliu setul de date, este momentul să filtrezi coloanele și să utilizezi tehnici NLP pe setul de date pentru a obține noi perspective despre hoteluri.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

### Operațiuni de filtrare și analiză a sentimentelor

Așa cum probabil ai observat, setul de date are câteva probleme. Unele coloane sunt pline de informații inutile, altele par incorecte. Chiar dacă sunt corecte, nu este clar cum au fost calculate, iar răspunsurile nu pot fi verificate independent prin propriile calcule.

## Exercițiu: un pic mai mult procesare a datelor

Curăță datele puțin mai mult. Adaugă coloane care vor fi utile mai târziu, modifică valorile din alte coloane și elimină complet anumite coloane.

1. Procesarea inițială a coloanelor

   1. Elimină `lat` și `lng`

   2. Înlocuiește valorile din `Hotel_Address` cu următoarele valori (dacă adresa conține numele orașului și țării, schimbă-l doar la oraș și țară).

      Acestea sunt singurele orașe și țări din setul de date:

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

      Acum poți interoga date la nivel de țară:

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

2. Procesează coloanele meta-recenziei hotelului

  1. Elimină `Additional_Number_of_Scoring`

  1. Înlocuiește `Total_Number_of_Reviews` cu numărul total de recenzii pentru acel hotel care sunt efectiv în setul de date 

  1. Înlocuiește `Average_Score` cu scorul calculat de noi

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Procesează coloanele recenziilor

   1. Elimină `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` și `days_since_review`

   2. Păstrează `Reviewer_Score`, `Negative_Review` și `Positive_Review` așa cum sunt,
     
   3. Păstrează `Tags` pentru moment

     - Vom face câteva operațiuni suplimentare de filtrare pe tag-uri în secțiunea următoare, iar apoi tag-urile vor fi eliminate

4. Procesează coloanele recenzentului

  1. Elimină `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Păstrează `Reviewer_Nationality`

### Coloanele de tag-uri

Coloana `Tag` este problematică deoarece este o listă (în formă de text) stocată în coloană. Din păcate, ordinea și numărul de subsecțiuni din această coloană nu sunt întotdeauna aceleași. Este dificil pentru un om să identifice frazele corecte de interes, deoarece există 515.000 de rânduri și 1427 de hoteluri, iar fiecare are opțiuni ușor diferite pe care un recenzent le-ar putea alege. Aici intervine NLP. Poți scana textul și găsi cele mai comune fraze și să le numeri.

Din păcate, nu suntem interesați de cuvinte individuale, ci de fraze de mai multe cuvinte (de exemplu, *Călătorie de afaceri*). Rularea unui algoritm de distribuție a frecvenței frazelor pe atât de multe date (6762646 cuvinte) ar putea dura un timp extraordinar, dar fără a privi datele, ar părea că este o cheltuială necesară. Aici este utilă analiza exploratorie a datelor, deoarece ai văzut un eșantion de tag-uri precum `[' Călătorie de afaceri  ', ' Călător singur ', ' Cameră single ', ' A stat 5 nopți ', ' Trimis de pe un dispozitiv mobil ']`, poți începe să te întrebi dacă este posibil să reduci semnificativ procesarea pe care trebuie să o faci. Din fericire, este - dar mai întâi trebuie să urmezi câțiva pași pentru a determina tag-urile de interes.

### Filtrarea tag-urilor

Amintește-ți că scopul setului de date este să adaugi sentiment și coloane care te vor ajuta să alegi cel mai bun hotel (pentru tine sau poate pentru un client care îți cere să creezi un bot de recomandare pentru hoteluri). Trebuie să te întrebi dacă tag-urile sunt utile sau nu în setul de date final. Iată o interpretare (dacă ai nevoie de setul de date pentru alte motive, tag-urile ar putea rămâne/să fie excluse din selecție):

1. Tipul de călătorie este relevant și ar trebui să rămână
2. Tipul grupului de oaspeți este important și ar trebui să rămână
3. Tipul camerei, suitei sau studioului în care a stat oaspetele este irelevant (toate hotelurile au practic aceleași camere)
4. Dispozitivul pe care a fost trimisă recenzia este irelevant
5. Numărul de nopți în care recenzentul a stat *ar putea* fi relevant dacă atribui șederile mai lungi cu faptul că le-a plăcut mai mult hotelul, dar este o presupunere și probabil irelevant

În rezumat, **păstrează 2 tipuri de tag-uri și elimină celelalte**.

Mai întâi, nu vrei să numeri tag-urile până când acestea nu sunt într-un format mai bun, ceea ce înseamnă eliminarea parantezelor pătrate și a ghilimelelor. Poți face acest lucru în mai multe moduri, dar vrei cea mai rapidă metodă, deoarece ar putea dura mult timp să procesezi multe date. Din fericire, pandas are o metodă ușoară pentru fiecare dintre acești pași.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Fiecare tag devine ceva de genul: `Călătorie de afaceri, Călător singur, Cameră single, A stat 5 nopți, Trimis de pe un dispozitiv mobil`. 

Apoi găsim o problemă. Unele recenzii sau rânduri au 5 coloane, altele 3, altele 6. Acesta este un rezultat al modului în care a fost creat setul de date și este dificil de remediat. Vrei să obții o numărătoare a frecvenței fiecărei fraze, dar acestea sunt în ordine diferită în fiecare recenzie, astfel încât numărătoarea ar putea fi greșită, iar un hotel ar putea să nu primească un tag atribuit pe care îl merita.

În schimb, vei folosi ordinea diferită în avantajul nostru, deoarece fiecare tag este format din mai multe cuvinte, dar și separat prin virgulă! Cea mai simplă metodă este să creezi 6 coloane temporare, fiecare tag fiind inserat în coloana corespunzătoare ordinii sale în tag. Apoi poți combina cele 6 coloane într-o singură coloană mare și să rulezi metoda `value_counts()` pe coloana rezultată. Printând rezultatul, vei vedea că au existat 2428 de tag-uri unice. Iată un mic eșantion:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Călătorie de relaxare          | 417778 |
| Trimis de pe un dispozitiv mobil | 307640 |
| Cuplu                         | 252294 |
| A stat 1 noapte               | 193645 |
| A stat 2 nopți                | 133937 |
| Călător singur                | 108545 |
| A stat 3 nopți                | 95821  |
| Călătorie de afaceri          | 82939  |
| Grup                          | 65392  |
| Familie cu copii mici         | 61015  |
| A stat 4 nopți                | 47817  |
| Cameră dublă                  | 35207  |
| Cameră dublă standard         | 32248  |
| Cameră dublă superioară       | 31393  |
| Familie cu copii mai mari     | 26349  |
| Cameră dublă deluxe           | 24823  |
| Cameră dublă sau twin         | 22393  |
| A stat 5 nopți                | 20845  |
| Cameră dublă sau twin standard | 17483  |
| Cameră dublă clasică          | 16989  |
| Cameră dublă sau twin superioară | 13570 |
| 2 camere                      | 12393  |

Unele dintre tag-urile comune, cum ar fi `Trimis de pe un dispozitiv mobil`, nu ne sunt de folos, așa că ar fi inteligent să le eliminăm înainte de a număra frecvența frazelor, dar este o operațiune atât de rapidă încât le poți lăsa și să le ignori.

### Eliminarea tag-urilor legate de durata șederii

Eliminarea acestor tag-uri este pasul 1, reducând ușor numărul total de tag-uri care trebuie luate în considerare. Observă că nu le elimini din setul de date, ci doar alegi să le elimini din considerare ca valori de numărat/păstrat în setul de date al recenziilor.

| Durata șederii | Count  |
| -------------- | ------ |
| A stat 1 noapte | 193645 |
| A stat 2 nopți  | 133937 |
| A stat 3 nopți  | 95821  |
| A stat 4 nopți  | 47817  |
| A stat 5 nopți  | 20845  |
| A stat 6 nopți  | 9776   |
| A stat 7 nopți  | 7399   |
| A stat 8 nopți  | 2502   |
| A stat 9 nopți  | 1293   |
| ...             | ...    |

Există o mare varietate de camere, suite, studiouri, apartamente și așa mai departe. Toate înseamnă aproximativ același lucru și nu sunt relevante pentru tine, așa că elimină-le din considerare.

| Tipul camerei                | Count |
| ---------------------------- | ----- |
| Cameră dublă                | 35207 |
| Cameră dublă standard       | 32248 |
| Cameră dublă superioară     | 31393 |
| Cameră dublă deluxe         | 24823 |
| Cameră dublă sau twin       | 22393 |
| Cameră dublă sau twin standard | 17483 |
| Cameră dublă clasică        | 16989 |
| Cameră dublă sau twin superioară | 13570 |

În cele din urmă, și acest lucru este încântător (pentru că nu a necesitat prea multă procesare), vei rămâne cu următoarele tag-uri *utile*:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Călătorie de relaxare                         | 417778 |
| Cuplu                                         | 252294 |
| Călător singur                                | 108545 |
| Călătorie de afaceri                          | 82939  |
| Grup (combinat cu Călători cu prieteni)       | 67535  |
| Familie cu copii mici                         | 61015  |
| Familie cu copii mai mari                     | 26349  |
| Cu un animal de companie                      | 1405   |

Ai putea argumenta că `Călători cu prieteni` este același lucru cu `Grup` mai mult sau mai puțin, și ar fi corect să le combini, așa cum este indicat mai sus. Codul pentru identificarea tag-urilor corecte se află în [notebook-ul Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Ultimul pas este să creezi coloane noi pentru fiecare dintre aceste tag-uri. Apoi, pentru fiecare rând de recenzie, dacă coloana `Tag` se potrivește cu una dintre coloanele noi, adaugă un 1, dacă nu, adaugă un 0. Rezultatul final va fi o numărătoare a câți recenzenți au ales acest hotel (în agregat) pentru, de exemplu, afaceri vs relaxare, sau pentru a aduce un animal de companie, și aceasta este o informație utilă atunci când recomanzi un hotel.

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

### Salvează fișierul

În cele din urmă, salvează setul de date așa cum este acum, cu un nume nou.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operațiuni de analiză a sentimentelor

În această secțiune finală, vei aplica analiza sentimentelor pe coloanele de recenzii și vei salva rezultatele într-un set de date.

## Exercițiu: încarcă și salvează datele filtrate

Observă că acum încarci setul de date filtrat care a fost salvat în secțiunea anterioară, **nu** setul de date original.

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

### Eliminarea cuvintelor de umplutură

Dacă ai rula analiza sentimentelor pe coloanele de recenzii negative și pozitive, ar putea dura mult timp. Testat pe un laptop puternic cu CPU rapid, a durat 12 - 14 minute, în funcție de biblioteca de analiză a sentimentelor utilizată. Este un timp (relativ) lung, așa că merită investigat dacă poate fi accelerat. 

Eliminarea cuvintelor de umplutură, sau a cuvintelor comune în limba engleză care nu schimbă sentimentul unei propoziții, este primul pas. Prin eliminarea lor, analiza sentimentelor ar trebui să ruleze mai rapid, dar să nu fie mai puțin precisă (deoarece cuvintele de umplutură nu afectează sentimentul, dar încetinesc analiza). 

Cea mai lungă recenzie negativă avea 395 de cuvinte, dar după eliminarea cuvintelor de umplutură, are 195 de cuvinte.

Eliminarea cuvintelor de umplutură este, de asemenea, o operațiune rapidă; eliminarea lor din 2 coloane de recenzii pe 515.000 de rânduri a durat 3,3 secunde pe dispozitivul de testare. Ar putea dura puțin mai mult sau mai puțin pentru tine, în funcție de viteza CPU-ului dispozitivului, RAM, dacă ai un SSD sau nu și alți factori. Durata relativ scurtă a operațiunii înseamnă că, dacă îmbunătățește timpul de analiză a sentimentelor, atunci merită făcută.

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

### Realizarea analizei sentimentelor

Acum ar trebui să calculezi analiza sentimentelor pentru coloanele de recenzii negative și pozitive și să stochezi rezultatul în 2 coloane noi. Testul sentimentului va fi să-l compari cu scorul recenzentului pentru aceeași recenzie. De exemplu, dacă sentimentul consideră că recenzia negativă avea un sentiment de 1 (sentiment extrem de pozitiv) și un sentiment de 1 pentru recenzia pozitivă, dar recenzentul a dat hotelului cel mai mic scor posibil, atunci fie textul recenziei nu se potrivește cu scorul, fie analizatorul de sentimente nu a putut recunoaște corect sentimentul. Ar trebui să te aștepți ca unele scoruri de sentiment să fie complet greșite, și adesea acest lucru va fi explicabil, de exemplu, recenzia ar putea fi extrem de sarcastică: "Desigur, AM IUBIT să dorm într-o cameră fără încălzire", iar analizatorul de sentimente consideră că acesta este un sentiment pozitiv, deși un om care citește ar ști că este sarcasm.
NLTK oferă diferiți analizatori de sentiment pentru a învăța, iar tu poți să îi înlocuiești și să vezi dacă sentimentul este mai precis sau mai puțin precis. Analiza de sentiment VADER este utilizată aici.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Un model simplificat bazat pe reguli pentru analiza sentimentului textelor din social media. A opta Conferință Internațională despre Bloguri și Social Media (ICWSM-14). Ann Arbor, MI, iunie 2014.

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

Mai târziu, în programul tău, când ești pregătit să calculezi sentimentul, îl poți aplica fiecărei recenzii astfel:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Aceasta durează aproximativ 120 de secunde pe computerul meu, dar va varia în funcție de fiecare computer. Dacă vrei să tipărești rezultatele și să vezi dacă sentimentul se potrivește cu recenzia:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Ultimul lucru pe care trebuie să-l faci cu fișierul înainte de a-l folosi în provocare este să-l salvezi! De asemenea, ar trebui să iei în considerare reordonarea tuturor coloanelor noi astfel încât să fie ușor de lucrat cu ele (pentru un om, este o schimbare cosmetică).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Ar trebui să rulezi întregul cod din [notebook-ul de analiză](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (după ce ai rulat [notebook-ul de filtrare](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) pentru a genera fișierul Hotel_Reviews_Filtered.csv).

Pentru a recapitula, pașii sunt:

1. Fișierul original al datasetului **Hotel_Reviews.csv** este explorat în lecția anterioară cu [notebook-ul de explorare](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv este filtrat de [notebook-ul de filtrare](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), rezultând **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv este procesat de [notebook-ul de analiză a sentimentului](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), rezultând **Hotel_Reviews_NLP.csv**
4. Folosește Hotel_Reviews_NLP.csv în provocarea NLP de mai jos

### Concluzie

Când ai început, aveai un dataset cu coloane și date, dar nu toate puteau fi verificate sau utilizate. Ai explorat datele, ai filtrat ceea ce nu aveai nevoie, ai convertit etichetele în ceva util, ai calculat propriile medii, ai adăugat câteva coloane de sentiment și, sperăm, ai învățat lucruri interesante despre procesarea textului natural.

## [Quiz de după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Provocare

Acum că ai analizat datasetul pentru sentiment, vezi dacă poți folosi strategiile pe care le-ai învățat în acest curriculum (poate clustering?) pentru a determina modele legate de sentiment.

## Recapitulare & Studiu Individual

Parcurge [acest modul Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) pentru a învăța mai multe și pentru a folosi diferite instrumente pentru a explora sentimentul în text.
## Temă

[Încearcă un alt dataset](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.