<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T14:19:41+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "hr"
}
-->
# Analiza sentimenta s recenzijama hotela

Sada kada ste detaljno istražili skup podataka, vrijeme je da filtrirate stupce i primijenite NLP tehnike na skup podataka kako biste dobili nove uvide o hotelima.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

### Operacije filtriranja i analize sentimenta

Kao što ste vjerojatno primijetili, skup podataka ima nekoliko problema. Neki stupci sadrže beskorisne informacije, dok se drugi čine netočnima. Ako su točni, nije jasno kako su izračunati, a odgovore ne možete neovisno provjeriti vlastitim izračunima.

## Vježba: malo više obrade podataka

Očistite podatke još malo. Dodajte stupce koji će biti korisni kasnije, promijenite vrijednosti u drugim stupcima i potpuno uklonite određene stupce.

1. Početna obrada stupaca

   1. Uklonite `lat` i `lng`

   2. Zamijenite vrijednosti u `Hotel_Address` sljedećim vrijednostima (ako adresa sadrži ime grada i države, promijenite je tako da ostanu samo grad i država).

      Ovo su jedini gradovi i države u skupu podataka:

      Amsterdam, Nizozemska

      Barcelona, Španjolska

      London, Ujedinjeno Kraljevstvo

      Milano, Italija

      Pariz, Francuska

      Beč, Austrija 

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

      Sada možete upitima dohvatiti podatke na razini države:

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

2. Obrada stupaca meta-recenzija hotela

   1. Uklonite `Additional_Number_of_Scoring`

   2. Zamijenite `Total_Number_of_Reviews` ukupnim brojem recenzija za taj hotel koje su stvarno prisutne u skupu podataka 

   3. Zamijenite `Average_Score` vlastitim izračunatim prosjekom

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Obrada stupaca recenzija

   1. Uklonite `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` i `days_since_review`

   2. Zadržite `Reviewer_Score`, `Negative_Review` i `Positive_Review` kakvi jesu
     
   3. Zadržite `Tags` za sada

     - U sljedećem dijelu ćemo provesti dodatne operacije filtriranja na oznakama, a zatim će oznake biti uklonjene

4. Obrada stupaca recenzenta

   1. Uklonite `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Zadržite `Reviewer_Nationality`

### Stupac oznaka

Stupac `Tag` je problematičan jer je popis (u tekstualnom obliku) pohranjen u stupcu. Nažalost, redoslijed i broj podsekcija u ovom stupcu nisu uvijek isti. Teško je ljudima identificirati točne fraze koje bi ih mogle zanimati, jer postoji 515.000 redaka i 1427 hotela, a svaki ima malo drugačije opcije koje recenzent može odabrati. Tu dolazi NLP do izražaja. Možete skenirati tekst i pronaći najčešće fraze te ih prebrojati.

Nažalost, ne zanima nas pojedinačne riječi, već višerječne fraze (npr. *Poslovno putovanje*). Pokretanje algoritma za distribuciju frekvencije višerječnih fraza na tolikoj količini podataka (6762646 riječi) moglo bi potrajati iznimno dugo, ali bez pregleda podataka čini se da je to nužan trošak. Tu dolazi do izražaja istraživačka analiza podataka, jer ste vidjeli uzorak oznaka poput `[' Poslovno putovanje  ', ' Solo putnik ', ' Jednokrevetna soba ', ' Boravak od 5 noći ', ' Poslano s mobilnog uređaja ']`, možete početi postavljati pitanje je li moguće značajno smanjiti obradu koju morate obaviti. Srećom, jest - ali prvo morate slijediti nekoliko koraka kako biste utvrdili oznake od interesa.

### Filtriranje oznaka

Zapamtite da je cilj skupa podataka dodati sentiment i stupce koji će vam pomoći odabrati najbolji hotel (za sebe ili možda za klijenta koji od vas traži da napravite bot za preporuku hotela). Morate se zapitati jesu li oznake korisne ili ne u konačnom skupu podataka. Evo jedne interpretacije (ako vam je skup podataka potreban iz drugih razloga, različite oznake bi mogle ostati/izaći iz odabira):

1. Vrsta putovanja je relevantna i treba ostati
2. Vrsta grupe gostiju je važna i treba ostati
3. Vrsta sobe, apartmana ili studija u kojem je gost boravio nije relevantna (svi hoteli imaju otprilike iste sobe)
4. Uređaj s kojeg je recenzija poslana nije relevantan
5. Broj noći koje je recenzent boravio *mogao bi* biti relevantan ako dulji boravak povežete s većim zadovoljstvom hotelom, ali to je upitno i vjerojatno nije relevantno

Ukratko, **zadržite 2 vrste oznaka i uklonite ostale**.

Prvo, ne želite brojati oznake dok nisu u boljem formatu, što znači uklanjanje uglatih zagrada i navodnika. To možete učiniti na nekoliko načina, ali želite najbrži jer bi obrada velike količine podataka mogla potrajati dugo. Srećom, pandas ima jednostavan način za svaki od ovih koraka.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Svaka oznaka postaje nešto poput: `Poslovno putovanje, Solo putnik, Jednokrevetna soba, Boravak od 5 noći, Poslano s mobilnog uređaja`. 

Zatim nailazimo na problem. Neke recenzije ili redci imaju 5 stupaca, neke 3, neke 6. To je rezultat načina na koji je skup podataka stvoren i teško ga je popraviti. Želite dobiti frekvencijski broj svake fraze, ali one su u različitom redoslijedu u svakoj recenziji, pa brojanje može biti netočno, a hotel možda neće dobiti oznaku koja mu pripada.

Umjesto toga, iskoristit ćete različit redoslijed u svoju korist, jer je svaka oznaka višerječna, ali također odvojena zarezom! Najjednostavniji način za to je stvaranje 6 privremenih stupaca, pri čemu se svaka oznaka umetne u stupac koji odgovara njezinom redoslijedu u oznaci. Zatim možete spojiti tih 6 stupaca u jedan veliki stupac i pokrenuti metodu `value_counts()` na rezultirajućem stupcu. Kada to ispišete, vidjet ćete da je bilo 2428 jedinstvenih oznaka. Evo malog uzorka:

| Oznaka                          | Broj   |
| ------------------------------- | ------ |
| Rekreacijsko putovanje          | 417778 |
| Poslano s mobilnog uređaja      | 307640 |
| Par                             | 252294 |
| Boravak od 1 noći               | 193645 |
| Boravak od 2 noći               | 133937 |
| Solo putnik                     | 108545 |
| Boravak od 3 noći               | 95821  |
| Poslovno putovanje              | 82939  |
| Grupa                           | 65392  |
| Obitelj s malom djecom          | 61015  |
| Boravak od 4 noći               | 47817  |
| Dvokrevetna soba                | 35207  |
| Standardna dvokrevetna soba     | 32248  |
| Superior dvokrevetna soba       | 31393  |
| Obitelj sa starijom djecom      | 26349  |
| Deluxe dvokrevetna soba         | 24823  |
| Dvokrevetna ili twin soba       | 22393  |
| Boravak od 5 noći               | 20845  |
| Standardna dvokrevetna ili twin | 17483  |
| Klasična dvokrevetna soba       | 16989  |
| Superior dvokrevetna ili twin   | 13570  |
| 2 sobe                          | 12393  |

Neke od uobičajenih oznaka poput `Poslano s mobilnog uređaja` nisu nam korisne, pa bi bilo pametno ukloniti ih prije brojanja pojavljivanja fraza, ali to je tako brza operacija da ih možete ostaviti i ignorirati.

### Uklanjanje oznaka duljine boravka

Uklanjanje ovih oznaka je prvi korak, što malo smanjuje ukupan broj oznaka koje treba razmotriti. Napominjemo da ih ne uklanjate iz skupa podataka, već ih samo odlučujete ukloniti iz razmatranja kao vrijednosti koje treba brojati/zadržati u skupu podataka recenzija.

| Duljina boravka | Broj   |
| --------------- | ------ |
| Boravak od 1 noći | 193645 |
| Boravak od 2 noći | 133937 |
| Boravak od 3 noći | 95821  |
| Boravak od 4 noći | 47817  |
| Boravak od 5 noći | 20845  |
| Boravak od 6 noći | 9776   |
| Boravak od 7 noći | 7399   |
| Boravak od 8 noći | 2502   |
| Boravak od 9 noći | 1293   |
| ...              | ...    |

Postoji veliki broj soba, apartmana, studija, stanova i tako dalje. Svi oni znače otprilike isto i nisu relevantni za vas, pa ih uklonite iz razmatranja.

| Vrsta sobe                  | Broj |
| --------------------------- | ----- |
| Dvokrevetna soba            | 35207 |
| Standardna dvokrevetna soba | 32248 |
| Superior dvokrevetna soba   | 31393 |
| Deluxe dvokrevetna soba     | 24823 |
| Dvokrevetna ili twin soba   | 22393 |
| Standardna dvokrevetna ili twin | 17483 |
| Klasična dvokrevetna soba   | 16989 |
| Superior dvokrevetna ili twin | 13570 |

Na kraju, i ovo je sjajno (jer nije zahtijevalo puno obrade), ostat ćete s sljedećim *korisnim* oznakama:

| Oznaka                                       | Broj   |
| ------------------------------------------- | ------ |
| Rekreacijsko putovanje                      | 417778 |
| Par                                         | 252294 |
| Solo putnik                                 | 108545 |
| Poslovno putovanje                          | 82939  |
| Grupa (kombinirano s Putnici s prijateljima)| 67535  |
| Obitelj s malom djecom                      | 61015  |
| Obitelj sa starijom djecom                  | 26349  |
| S kućnim ljubimcem                          | 1405   |

Možete tvrditi da je `Putnici s prijateljima` isto što i `Grupa` više-manje, i bilo bi pošteno kombinirati te dvije oznake kao gore. Kod za identifikaciju ispravnih oznaka nalazi se u [bilježnici s oznakama](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Završni korak je stvaranje novih stupaca za svaku od ovih oznaka. Zatim, za svaki red recenzije, ako stupac `Tag` odgovara jednom od novih stupaca, dodajte 1, ako ne, dodajte 0. Krajnji rezultat bit će broj koliko je recenzenata odabralo ovaj hotel (u zbiru) za, recimo, poslovno naspram rekreacijskog putovanja, ili za dolazak s kućnim ljubimcem, i to su korisne informacije pri preporuci hotela.

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

### Spremite datoteku

Na kraju, spremite skup podataka kakav je sada s novim imenom.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operacije analize sentimenta

U ovom završnom dijelu primijenit ćete analizu sentimenta na stupce recenzija i spremiti rezultate u skup podataka.

## Vježba: učitajte i spremite filtrirane podatke

Napominjemo da sada učitavate filtrirani skup podataka koji je spremljen u prethodnom dijelu, **ne** originalni skup podataka.

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

### Uklanjanje stop riječi

Ako biste pokrenuli analizu sentimenta na stupcima negativnih i pozitivnih recenzija, to bi moglo potrajati dugo. Testirano na snažnom testnom prijenosnom računalu s brzim CPU-om, trajalo je 12 - 14 minuta ovisno o tome koja je biblioteka za analizu sentimenta korištena. To je (relativno) dugo vrijeme, pa vrijedi istražiti može li se ubrzati.

Uklanjanje stop riječi, odnosno uobičajenih engleskih riječi koje ne mijenjaju sentiment rečenice, prvi je korak. Uklanjanjem tih riječi analiza sentimenta trebala bi se brže izvršiti, ali ne biti manje točna (jer stop riječi ne utječu na sentiment, ali usporavaju analizu). 

Najduža negativna recenzija imala je 395 riječi, ali nakon uklanjanja stop riječi, ima 195 riječi.

Uklanjanje stop riječi također je brza operacija, uklanjanje stop riječi iz 2 stupca recenzija preko 515.000 redaka trajalo je 3,3 sekunde na testnom uređaju. Može potrajati malo više ili manje vremena ovisno o brzini vašeg CPU-a, RAM-u, imate li SSD ili ne, i nekim drugim čimbenicima. Relativna kratkoća operacije znači da, ako poboljšava vrijeme analize sentimenta, vrijedi je provesti.

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

### Provođenje analize sentimenta

Sada biste trebali izračunati analizu sentimenta za stupce negativnih i pozitivnih recenzija te spremiti rezultat u 2 nova stupca. Test sentimenta bit će usporedba s ocjenom recenzenta za istu recenziju. Na primjer, ako sentiment pokazuje da negativna recenzija ima sentiment 1 (iznimno pozitivan sentiment) i sentiment pozitivne recenzije također 1, ali recenzent je hotelu dao najnižu moguću ocjenu, tada tekst recenzije ne odgovara ocjeni ili analizator sentimenta nije mogao ispravno prepoznati sentiment. Trebali biste očekivati da će neki rezultati sentimenta biti potpuno pogrešni, a često će to biti objašnjivo, npr. recenzija bi mogla biti iznimno sarkastična "Naravno da sam OBOŽAVAO spavati u sobi bez grijanja" i analizator sentimenta misli da je to pozitivan sentiment, iako bi čovjek koji to čita znao da je to sarkazam.
NLTK nudi različite analizatore sentimenta za učenje, a možete ih zamijeniti i provjeriti je li sentiment precizniji ili manje precizan. Ovdje se koristi VADER analiza sentimenta.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, lipanj 2014.

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

Kasnije u programu, kada budete spremni za izračunavanje sentimenta, možete ga primijeniti na svaku recenziju na sljedeći način:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Ovo traje otprilike 120 sekundi na mom računalu, ali vrijeme će varirati ovisno o računalu. Ako želite ispisati rezultate i provjeriti odgovara li sentiment recenziji:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Posljednja stvar koju trebate učiniti s datotekom prije nego što je upotrijebite u izazovu jest spremiti je! Također biste trebali razmisliti o preuređivanju svih novih stupaca kako bi bili lakši za rad (za čovjeka, to je kozmetička promjena).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Trebali biste pokrenuti cijeli kod za [bilježnicu analize](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (nakon što ste pokrenuli [bilježnicu za filtriranje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) kako biste generirali datoteku Hotel_Reviews_Filtered.csv).

Za pregled, koraci su:

1. Izvorna datoteka **Hotel_Reviews.csv** istražena je u prethodnoj lekciji pomoću [bilježnice za istraživanje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv filtriran je pomoću [bilježnice za filtriranje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), što rezultira datotekom **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv obrađen je pomoću [bilježnice za analizu sentimenta](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), što rezultira datotekom **Hotel_Reviews_NLP.csv**
4. Koristite Hotel_Reviews_NLP.csv u NLP izazovu u nastavku

### Zaključak

Kada ste započeli, imali ste skup podataka sa stupcima i podacima, ali ne sve od njih mogli ste provjeriti ili koristiti. Istražili ste podatke, filtrirali ono što vam nije potrebno, pretvorili oznake u nešto korisno, izračunali vlastite prosjeke, dodali stupce sentimenta i, nadamo se, naučili nešto zanimljivo o obradi prirodnog teksta.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Izazov

Sada kada ste analizirali svoj skup podataka za sentiment, pokušajte koristiti strategije koje ste naučili u ovom kurikulumu (možda klasteriranje?) kako biste odredili obrasce vezane uz sentiment.

## Pregled i samostalno učenje

Proučite [ovaj modul na Learn platformi](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) kako biste saznali više i koristili različite alate za istraživanje sentimenta u tekstu.

## Zadatak

[Isprobajte drugi skup podataka](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prijevod [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.