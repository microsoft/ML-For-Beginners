<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T08:08:00+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "lt"
}
-->
# Sentimentų analizė su viešbučių apžvalgomis

Dabar, kai išsamiai išnagrinėjote duomenų rinkinį, metas filtruoti stulpelius ir pritaikyti NLP technikas, kad gautumėte naujų įžvalgų apie viešbučius.

## [Prieš paskaitą: testas](https://ff-quizzes.netlify.app/en/ml/)

### Filtravimo ir sentimentų analizės operacijos

Kaip turbūt pastebėjote, duomenų rinkinyje yra keletas problemų. Kai kurie stulpeliai užpildyti nereikalinga informacija, kiti atrodo neteisingi. Net jei jie teisingi, neaišku, kaip buvo apskaičiuoti, ir atsakymų negalima savarankiškai patikrinti pagal jūsų pačių skaičiavimus.

## Užduotis: šiek tiek daugiau duomenų apdorojimo

Išvalykite duomenis dar šiek tiek. Pridėkite stulpelius, kurie bus naudingi vėliau, pakeiskite kitų stulpelių reikšmes ir visiškai pašalinkite tam tikrus stulpelius.

1. Pradinis stulpelių apdorojimas

   1. Pašalinkite `lat` ir `lng`

   2. Pakeiskite `Hotel_Address` reikšmes taip (jei adresas turi miesto ir šalies pavadinimą, pakeiskite jį į tik miestą ir šalį).

      Štai vieninteliai miestai ir šalys duomenų rinkinyje:

      Amsterdamas, Nyderlandai

      Barselona, Ispanija

      Londonas, Jungtinė Karalystė

      Milanas, Italija

      Paryžius, Prancūzija

      Viena, Austrija 

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

      Dabar galite užklausti duomenis pagal šalį:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdamas, Nyderlandai |    105     |
      | Barselona, Ispanija       |    211     |
      | Londonas, Jungtinė Karalystė |    400     |
      | Milanas, Italija           |    162     |
      | Paryžius, Prancūzija       |    458     |
      | Viena, Austrija            |    158     |

2. Viešbučių meta-apžvalgų stulpelių apdorojimas

  1. Pašalinkite `Additional_Number_of_Scoring`

  1. Pakeiskite `Total_Number_of_Reviews` į bendrą apžvalgų skaičių, kuris iš tikrųjų yra duomenų rinkinyje

  1. Pakeiskite `Average_Score` į mūsų pačių apskaičiuotą vidurkį

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Apžvalgų stulpelių apdorojimas

   1. Pašalinkite `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ir `days_since_review`

   2. Palikite `Reviewer_Score`, `Negative_Review` ir `Positive_Review` kaip yra,
     
   3. Laikinai palikite `Tags`

     - Kitame skyriuje atliksime papildomas filtravimo operacijas su žymomis, o tada žymos bus pašalintos

4. Apžvalgininkų stulpelių apdorojimas

  1. Pašalinkite `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Palikite `Reviewer_Nationality`

### Žymų stulpeliai

`Tag` stulpelis yra problemiškas, nes jame saugomas sąrašas (teksto forma). Deja, šio stulpelio poskyrių tvarka ir skaičius ne visada yra vienodi. Žmogui sunku nustatyti, kurios frazės yra svarbios, nes yra 515 000 eilučių ir 1427 viešbučiai, o kiekvienas turi šiek tiek skirtingas galimybes, kurias apžvalgininkas galėjo pasirinkti. Čia NLP yra labai naudingas. Galite nuskaityti tekstą, rasti dažniausiai pasitaikančias frazes ir jas suskaičiuoti.

Deja, mūsų nedomina pavieniai žodžiai, o kelių žodžių frazės (pvz., *Verslo kelionė*). Kelių žodžių dažnio paskirstymo algoritmas tokiam dideliam duomenų kiekiui (6762646 žodžių) galėtų užtrukti nepaprastai daug laiko, tačiau, nepažvelgus į duomenis, atrodo, kad tai būtina. Čia naudinga duomenų tyrimo analizė, nes jau matėte žymių pavyzdį, pvz., `[' Verslo kelionė  ', ' Vienišas keliautojas ', ' Vienvietis kambarys ', ' Praleido 5 naktis ', ' Pateikta iš mobiliojo įrenginio ']`, galite pradėti klausti, ar įmanoma labai sumažinti apdorojimo apimtį. Laimei, tai įmanoma - bet pirmiausia reikia atlikti keletą žingsnių, kad nustatytumėte svarbias žymas.

### Žymų filtravimas

Prisiminkite, kad duomenų rinkinio tikslas yra pridėti sentimentus ir stulpelius, kurie padės pasirinkti geriausią viešbutį (sau arba galbūt klientui, kuris prašo sukurti viešbučių rekomendacijų botą). Turite savęs paklausti, ar žymos yra naudingos galutiniame duomenų rinkinyje. Štai viena interpretacija (jei duomenų rinkinys būtų reikalingas kitais tikslais, skirtingos žymos galėtų būti įtrauktos/neįtrauktos):

1. Kelionės tipas yra svarbus ir turėtų likti
2. Svečio grupės tipas yra svarbus ir turėtų likti
3. Kambario, apartamentų ar studijos tipas, kuriame svečias apsistojo, yra nesvarbus (visi viešbučiai turi iš esmės tuos pačius kambarius)
4. Įrenginys, iš kurio pateikta apžvalga, yra nesvarbus
5. Naktų, kurias apžvalgininkas praleido, skaičius *galėtų* būti svarbus, jei ilgesnės viešnagės būtų susijusios su viešbučio patikimu, tačiau tai abejotina ir greičiausiai nesvarbu

Apibendrinant, **palikite 2 žymų tipus ir pašalinkite kitus**.

Pirmiausia, nenorite skaičiuoti žymų, kol jos nėra geresniame formate, todėl reikia pašalinti kvadratinius skliaustus ir kabutes. Tai galite padaryti keliais būdais, tačiau norite greičiausio, nes apdoroti daug duomenų gali užtrukti ilgai. Laimei, pandas turi paprastą būdą atlikti kiekvieną iš šių žingsnių.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Kiekviena žyma tampa panaši į: `Verslo kelionė, Vienišas keliautojas, Vienvietis kambarys, Praleido 5 naktis, Pateikta iš mobiliojo įrenginio`. 

Tada susiduriame su problema. Kai kurios apžvalgos arba eilutės turi 5 stulpelius, kai kurios 3, kai kurios 6. Tai yra duomenų rinkinio kūrimo rezultatas ir sunku ištaisyti. Norite gauti kiekvienos frazės dažnio skaičių, tačiau jos yra skirtingoje tvarkoje kiekvienoje apžvalgoje, todėl skaičiavimas gali būti netikslus, o viešbutis gali negauti žymos, kurios jis nusipelnė.

Vietoj to, pasinaudosite skirtinga tvarka savo naudai, nes kiekviena žyma yra kelių žodžių, bet taip pat atskirta kableliu! Paprasčiausias būdas tai padaryti yra sukurti 6 laikinus stulpelius, kuriuose kiekviena žyma įterpiama į stulpelį, atitinkantį jos tvarką žymoje. Tada galite sujungti 6 stulpelius į vieną didelį stulpelį ir paleisti `value_counts()` metodą ant gauto stulpelio. Atspausdinus, pamatysite, kad buvo 2428 unikalios žymos. Štai nedidelis pavyzdys:

| Žyma                           | Skaičius |
| ------------------------------ | -------- |
| Laisvalaikio kelionė           | 417778   |
| Pateikta iš mobiliojo įrenginio| 307640   |
| Pora                           | 252294   |
| Praleido 1 naktį               | 193645   |
| Praleido 2 naktis              | 133937   |
| Vienišas keliautojas           | 108545   |
| Praleido 3 naktis              | 95821    |
| Verslo kelionė                 | 82939    |
| Grupė                          | 65392    |
| Šeima su mažais vaikais        | 61015    |
| Praleido 4 naktis              | 47817    |
| Dvivietis kambarys             | 35207    |
| Standartinis dvivietis kambarys| 32248    |
| Aukštesnės klasės dvivietis kambarys | 31393 |
| Šeima su vyresniais vaikais    | 26349    |
| Prabangus dvivietis kambarys   | 24823    |
| Dvivietis arba dvynis kambarys | 22393    |
| Praleido 5 naktis              | 20845    |
| Standartinis dvivietis arba dvynis kambarys | 17483 |
| Klasikinis dvivietis kambarys  | 16989    |
| Aukštesnės klasės dvivietis arba dvynis kambarys | 13570 |
| 2 kambariai                    | 12393    |

Kai kurios dažnos žymos, pvz., `Pateikta iš mobiliojo įrenginio`, mums nėra naudingos, todėl gali būti protinga jas pašalinti prieš skaičiuojant frazių pasikartojimą, tačiau tai yra tokia greita operacija, kad galite jas palikti ir ignoruoti.

### Viešnagės trukmės žymų pašalinimas

Šių žymų pašalinimas yra pirmas žingsnis, jis šiek tiek sumažina bendrą žymų skaičių, kurį reikia apsvarstyti. Atkreipkite dėmesį, kad jų nepašalinate iš duomenų rinkinio, tiesiog nusprendžiate jų neįtraukti į apžvalgų duomenų rinkinio skaičiavimus.

| Viešnagės trukmė | Skaičius |
| ---------------- | -------- |
| Praleido 1 naktį | 193645   |
| Praleido 2 naktis| 133937   |
| Praleido 3 naktis| 95821    |
| Praleido 4 naktis| 47817    |
| Praleido 5 naktis| 20845    |
| Praleido 6 naktis| 9776     |
| Praleido 7 naktis| 7399     |
| Praleido 8 naktis| 2502     |
| Praleido 9 naktis| 1293     |
| ...              | ...      |

Yra didelė kambarių, apartamentų, studijų, butų ir pan. įvairovė. Jie visi reiškia maždaug tą patį ir nėra svarbūs jums, todėl pašalinkite juos iš svarstymo.

| Kambario tipas                | Skaičius |
| ----------------------------- | -------- |
| Dvivietis kambarys            | 35207    |
| Standartinis dvivietis kambarys| 32248    |
| Aukštesnės klasės dvivietis kambarys | 31393 |
| Prabangus dvivietis kambarys  | 24823    |
| Dvivietis arba dvynis kambarys| 22393    |
| Standartinis dvivietis arba dvynis kambarys | 17483 |
| Klasikinis dvivietis kambarys | 16989    |
| Aukštesnės klasės dvivietis arba dvynis kambarys | 13570 |

Galiausiai, ir tai yra malonu (nes tam nereikėjo daug apdorojimo), liksite su šiomis *naudingomis* žymomis:

| Žyma                                         | Skaičius |
| -------------------------------------------- | -------- |
| Laisvalaikio kelionė                         | 417778   |
| Pora                                         | 252294   |
| Vienišas keliautojas                         | 108545   |
| Verslo kelionė                               | 82939    |
| Grupė (sujungta su Keliautojai su draugais)  | 67535    |
| Šeima su mažais vaikais                      | 61015    |
| Šeima su vyresniais vaikais                  | 26349    |
| Su augintiniu                                | 1405     |

Galite teigti, kad `Keliautojai su draugais` yra tas pats kaip `Grupė`, ir būtų teisinga juos sujungti, kaip parodyta aukščiau. Kodas, skirtas tinkamoms žymoms identifikuoti, yra [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Paskutinis žingsnis yra sukurti naujus stulpelius kiekvienai iš šių žymų. Tada, kiekvienai apžvalgos eilutei, jei `Tag` stulpelis atitinka vieną iš naujų stulpelių, pridėkite 1, jei ne, pridėkite 0. Galutinis rezultatas bus skaičius, kiek apžvalgininkų pasirinko šį viešbutį (bendrai) verslo ar laisvalaikio kelionei, arba, pavyzdžiui, su augintiniu, ir tai yra naudinga informacija rekomenduojant viešbutį.

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

### Išsaugokite savo failą

Galiausiai, išsaugokite duomenų rinkinį dabartinėje būsenoje su nauju pavadinimu.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentimentų analizės operacijos

Šiame paskutiniame skyriuje pritaikysite sentimentų analizę apžvalgų stulpeliams ir išsaugosite rezultatus duomenų rinkinyje.

## Užduotis: įkelkite ir išsaugokite filtruotus duomenis

Atkreipkite dėmesį, kad dabar įkeliate filtruotą duomenų rinkinį, kuris buvo išsaugotas ankstesniame skyriuje, **ne** originalų duomenų rinkinį.

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

### Stop žodžių pašalinimas

Jei atliktumėte sentimentų analizę neigiamų ir teigiamų apžvalgų stulpeliams, tai galėtų užtrukti ilgai. Testuota galingame nešiojamame kompiuteryje su greitu procesoriumi, tai užtruko 12–14 minučių, priklausomai nuo to, kuri sentimentų biblioteka buvo naudojama. Tai yra (santykinai) ilgas laikas, todėl verta ištirti, ar tai galima pagreitinti. 

Stop žodžių, arba dažnų anglų kalbos žodžių, kurie nekeičia sakinio sentimentų, pašalinimas yra pirmas žingsnis. Pašalinus juos, sentimentų analizė turėtų vykti greičiau, tačiau nebūti mažiau tiksli (nes stop žodžiai sentimentų nekeičia, tačiau jie sulėtina analizę). 

Ilgiausia neigiama apžvalga buvo 395 žodžių, tačiau pašalinus stop žodžius, ji tapo 195 žodžių.

Stop žodžių pašalinimas taip pat yra greita operacija, pašalinus stop žodžius iš 2 apžvalgų stulpelių per 515 000 eilučių, tai užtruko 3,3 sekundės testavimo įrenginyje. Tai galėtų užtrukti šiek tiek daugiau ar mažiau laiko, priklausomai nuo jūsų įrenginio procesoriaus greičio, RAM, ar turite SSD, ir kitų veiksnių. Operacijos santykinis trumpumas reiškia, kad jei tai pagerina sentimentų analizės laiką, verta tai atlikti.

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

### Sentimentų analizės atlikimas

Dabar turėtumėte apskaičiuoti sentimentų analizę tiek neigiamų, tiek teigiamų apžvalgų stulpeliams ir išsaugoti rezultatą 2 naujuose stulpeliuose. Sentimentų testas bus palyginti jį su apžvalgininko įvertinimu už tą pačią apžvalgą. Pavyzdžiui, jei sentimentų analizė mano, kad neigiama apžvalga turėjo sentimentą 1 (labai teigiamas sentimentas) ir
NLTK siūlo įvairius sentimentų analizės įrankius, kuriuos galite išbandyti ir pakeisti, kad pamatytumėte, ar sentimentų analizė yra tikslesnė ar mažiau tiksli. Čia naudojama VADER sentimentų analizė.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, 2014 m. birželis.

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

Vėliau, kai būsite pasiruošę skaičiuoti sentimentus savo programoje, galite juos pritaikyti kiekvienai apžvalgai taip:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Tai užtrunka maždaug 120 sekundžių mano kompiuteryje, tačiau laikas gali skirtis priklausomai nuo kompiuterio. Jei norite išspausdinti rezultatus ir patikrinti, ar sentimentai atitinka apžvalgą:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Paskutinis dalykas, kurį reikia padaryti su failu prieš naudojant jį iššūkyje, yra jį išsaugoti! Taip pat turėtumėte apsvarstyti galimybę pertvarkyti visas naujas stulpelius, kad jie būtų patogesni naudoti (tai kosmetinis pakeitimas, skirtas žmogui).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Turėtumėte paleisti visą kodą iš [analizės užrašų knygelės](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (po to, kai paleidote [filtravimo užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), kad sugeneruotumėte Hotel_Reviews_Filtered.csv failą).

Apžvelkime žingsnius:

1. Originalus duomenų failas **Hotel_Reviews.csv** buvo analizuotas ankstesnėje pamokoje naudojant [tyrimo užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv buvo filtruotas naudojant [filtravimo užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), rezultatas - **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv buvo apdorotas naudojant [sentimentų analizės užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), rezultatas - **Hotel_Reviews_NLP.csv**
4. Naudokite Hotel_Reviews_NLP.csv NLP iššūkyje žemiau

### Išvada

Pradėję turėjote duomenų rinkinį su stulpeliais ir duomenimis, tačiau ne visi jie buvo patikrinami ar naudojami. Jūs ištyrėte duomenis, išfiltravote nereikalingus, konvertavote žymes į naudingą informaciją, apskaičiavote savo vidurkius, pridėjote sentimentų stulpelius ir, tikėkimės, sužinojote įdomių dalykų apie natūralaus teksto apdorojimą.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Iššūkis

Dabar, kai jūsų duomenų rinkinys yra analizuotas sentimentų atžvilgiu, pabandykite pritaikyti strategijas, kurias išmokote šiame kurse (galbūt klasterizavimą?), kad nustatytumėte sentimentų tendencijas.

## Apžvalga ir savarankiškas mokymasis

Pereikite [šį mokymosi modulį](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), kad sužinotumėte daugiau ir naudotumėte skirtingus įrankius sentimentų analizei tekste.

## Užduotis

[Naudokite kitą duomenų rinkinį](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipiame dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudotis profesionalių vertėjų paslaugomis. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus aiškinimus, kylančius dėl šio vertimo naudojimo.