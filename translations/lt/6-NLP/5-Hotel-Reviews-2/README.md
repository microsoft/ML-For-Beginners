<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-03T19:09:08+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "lt"
}
-->
# Sentimentų analizė su viešbučių apžvalgomis

Dabar, kai išsamiai išnagrinėjote duomenų rinkinį, metas filtruoti stulpelius ir taikyti NLP technikas, kad gautumėte naujų įžvalgų apie viešbučius.
## [Prieš paskaitą: testas](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Filtravimo ir sentimentų analizės operacijos

Kaip turbūt pastebėjote, duomenų rinkinyje yra keletas problemų. Kai kurie stulpeliai užpildyti nereikalinga informacija, kiti atrodo neteisingi. Net jei jie teisingi, neaišku, kaip jie buvo apskaičiuoti, ir atsakymų negalima savarankiškai patikrinti atliekant savo skaičiavimus.

## Užduotis: šiek tiek daugiau duomenų apdorojimo

Dar šiek tiek išvalykite duomenis. Pridėkite stulpelius, kurie bus naudingi vėliau, pakeiskite kitų stulpelių reikšmes ir visiškai pašalinkite tam tikrus stulpelius.

1. Pradinis stulpelių apdorojimas

   1. Pašalinkite `lat` ir `lng`

   2. Pakeiskite `Hotel_Address` reikšmes šiomis reikšmėmis (jei adresas apima miesto ir šalies pavadinimą, pakeiskite jį tik į miestą ir šalį).

      Tai yra vieninteliai miestai ir šalys duomenų rinkinyje:

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

      Dabar galite užklausti duomenis šalies lygiu:

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

  1. Pakeiskite `Average_Score` į mūsų pačių apskaičiuotą rezultatą

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

`Tag` stulpelis yra problemiškas, nes jame saugomas sąrašas (teksto forma). Deja, šio stulpelio poskyrių tvarka ir skaičius ne visada yra vienodi. Žmogui sunku nustatyti tinkamas frazes, į kurias verta atkreipti dėmesį, nes yra 515 000 eilučių ir 1427 viešbučiai, o kiekvienas turi šiek tiek skirtingas galimybes, kurias apžvalgininkas galėjo pasirinkti. Čia praverčia NLP. Galite nuskaityti tekstą, rasti dažniausiai pasitaikančias frazes ir jas suskaičiuoti.

Deja, mūsų nedomina pavieniai žodžiai, o kelių žodžių frazės (pvz., *Verslo kelionė*). Paleisti kelių žodžių dažnio paskirstymo algoritmą tokiam dideliam duomenų kiekiui (6762646 žodžių) gali užtrukti nepaprastai daug laiko, tačiau, nepažvelgus į duomenis, atrodo, kad tai būtina. Čia praverčia duomenų tyrimo analizė, nes matėte žymų pavyzdį, pvz., `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, galite pradėti klausti, ar įmanoma labai sumažinti apdorojimo apimtį. Laimei, tai įmanoma - bet pirmiausia reikia atlikti keletą žingsnių, kad nustatytumėte dominančias žymas.

### Žymų filtravimas

Prisiminkite, kad duomenų rinkinio tikslas yra pridėti sentimentus ir stulpelius, kurie padės pasirinkti geriausią viešbutį (sau arba galbūt klientui, kuris prašo sukurti viešbučių rekomendacijų botą). Turite savęs paklausti, ar žymos yra naudingos galutiniame duomenų rinkinyje. Štai viena interpretacija (jei duomenų rinkinys būtų reikalingas kitais tikslais, skirtingos žymos galėtų būti įtrauktos/neįtrauktos):

1. Kelionės tipas yra svarbus ir turėtų likti
2. Svečio grupės tipas yra svarbus ir turėtų likti
3. Kambario, apartamentų ar studijos tipas, kuriame svečias apsistojo, yra nesvarbus (visi viešbučiai iš esmės turi tuos pačius kambarius)
4. Įrenginys, kuriuo pateikta apžvalga, yra nesvarbus
5. Nakvynių skaičius *galėtų* būti svarbus, jei ilgesnės viešnagės būtų siejamos su viešbučio patikimu, tačiau tai abejotina ir greičiausiai nesvarbu

Apibendrinant, **palikite 2 žymų tipus ir pašalinkite kitus**.

Pirmiausia nenorite skaičiuoti žymų, kol jos nėra geresniame formate, todėl reikia pašalinti kvadratinius skliaustus ir kabutes. Tai galite padaryti keliais būdais, tačiau norite greičiausio, nes apdoroti daug duomenų gali užtrukti ilgai. Laimei, pandas turi paprastą būdą atlikti kiekvieną iš šių žingsnių.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Kiekviena žyma tampa kažkuo panašiu į: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Tada susiduriame su problema. Kai kurios apžvalgos arba eilutės turi 5 stulpelius, kai kurios 3, kai kurios 6. Tai yra duomenų rinkinio kūrimo rezultatas ir sunku ištaisyti. Norite gauti kiekvienos frazės dažnio skaičių, tačiau jos yra skirtingoje eilėje kiekvienoje apžvalgoje, todėl skaičiavimas gali būti netikslus, o viešbutis gali negauti jam priskirtos žymos, kurios jis nusipelnė.

Vietoj to pasinaudosite skirtinga tvarka savo naudai, nes kiekviena žyma yra kelių žodžių, bet taip pat atskirta kableliu! Paprasčiausias būdas tai padaryti yra sukurti 6 laikinus stulpelius, kuriuose kiekviena žyma įterpiama į stulpelį, atitinkantį jos eilę žymoje. Tada galite sujungti 6 stulpelius į vieną didelį stulpelį ir paleisti `value_counts()` metodą ant gauto stulpelio. Atspausdinę pamatysite, kad buvo 2428 unikalios žymos. Štai nedidelis pavyzdys:

| Žyma                           | Skaičius |
| ------------------------------ | -------- |
| Laisvalaikio kelionė           | 417778   |
| Pateikta iš mobiliojo įrenginio| 307640   |
| Pora                           | 252294   |
| Apsistojo 1 nakčiai            | 193645   |
| Apsistojo 2 nakčiai            | 133937   |
| Vienišas keliautojas           | 108545   |
| Apsistojo 3 nakčiai            | 95821    |
| Verslo kelionė                 | 82939    |
| Grupė                          | 65392    |
| Šeima su mažais vaikais        | 61015    |
| Apsistojo 4 nakčiai            | 47817    |
| Dvivietis kambarys             | 35207    |
| Standartinis dvivietis kambarys| 32248    |
| Aukštesnės klasės dvivietis kambarys | 31393 |
| Šeima su vyresniais vaikais    | 26349    |
| Prabangus dvivietis kambarys   | 24823    |
| Dvivietis arba dvivietis su atskiromis lovomis | 22393 |
| Apsistojo 5 nakčiai            | 20845    |
| Standartinis dvivietis arba dvivietis su atskiromis lovomis | 17483 |
| Klasikinis dvivietis kambarys  | 16989    |
| Aukštesnės klasės dvivietis arba dvivietis su atskiromis lovomis | 13570 |
| 2 kambariai                    | 12393    |

Kai kurios dažnos žymos, pvz., `Pateikta iš mobiliojo įrenginio`, mums nėra naudingos, todėl gali būti protinga jas pašalinti prieš skaičiuojant frazių pasikartojimą, tačiau tai yra tokia greita operacija, kad galite jas palikti ir ignoruoti.

### Nakvynių trukmės žymų pašalinimas

Šių žymų pašalinimas yra pirmas žingsnis, jis šiek tiek sumažina žymų, kurias reikia apsvarstyti, skaičių. Atkreipkite dėmesį, kad jų nepašalinate iš duomenų rinkinio, tiesiog nusprendžiate jų neįtraukti į apžvalgų duomenų rinkinio skaičiavimus.

| Nakvynės trukmė | Skaičius |
| ---------------- | -------- |
| Apsistojo 1 nakčiai | 193645 |
| Apsistojo 2 nakčiai | 133937 |
| Apsistojo 3 nakčiai | 95821  |
| Apsistojo 4 nakčiai | 47817  |
| Apsistojo 5 nakčiai | 20845  |
| Apsistojo 6 nakčiai | 9776   |
| Apsistojo 7 nakčiai | 7399   |
| Apsistojo 8 nakčiai | 2502   |
| Apsistojo 9 nakčiai | 1293   |
| ...              | ...     |

Yra didelė įvairovė kambarių, apartamentų, studijų, butų ir pan. Jie visi reiškia maždaug tą patį ir nėra svarbūs jums, todėl pašalinkite juos iš svarstymo.

| Kambario tipas                | Skaičius |
| ----------------------------- | -------- |
| Dvivietis kambarys            | 35207    |
| Standartinis dvivietis kambarys | 32248   |
| Aukštesnės klasės dvivietis kambarys | 31393 |
| Prabangus dvivietis kambarys  | 24823    |
| Dvivietis arba dvivietis su atskiromis lovomis | 22393 |
| Standartinis dvivietis arba dvivietis su atskiromis lovomis | 17483 |
| Klasikinis dvivietis kambarys | 16989    |
| Aukštesnės klasės dvivietis arba dvivietis su atskiromis lovomis | 13570 |

Galiausiai, ir tai yra malonu (nes tam nereikėjo daug apdorojimo), liksite su šiomis *naudingomis* žymomis:

| Žyma                                         | Skaičius |
| -------------------------------------------- | -------- |
| Laisvalaikio kelionė                         | 417778   |
| Pora                                         | 252294   |
| Vienišas keliautojas                         | 108545   |
| Verslo kelionė                               | 82939    |
| Grupė (sujungta su Keliautojais su draugais) | 67535    |
| Šeima su mažais vaikais                      | 61015    |
| Šeima su vyresniais vaikais                  | 26349    |
| Su augintiniu                                | 1405     |

Galite teigti, kad `Keliautojai su draugais` yra tas pats kaip `Grupė`, ir būtų teisinga sujungti šiuos du, kaip parodyta aukščiau. Kodas, skirtas tinkamų žymų identifikavimui, yra [Žymų užrašų knygelėje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Paskutinis žingsnis yra sukurti naujus stulpelius kiekvienai iš šių žymų. Tada, kiekvienai apžvalgos eilutei, jei `Tag` stulpelis atitinka vieną iš naujų stulpelių, pridėkite 1, jei ne, pridėkite 0. Galutinis rezultatas bus skaičius, kiek apžvalgininkų pasirinko šį viešbutį (bendru mastu) verslo ar laisvalaikio kelionėms, arba, pavyzdžiui, atvykti su augintiniu, ir tai yra naudinga informacija rekomenduojant viešbutį.

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

Šiame paskutiniame skyriuje taikysite sentimentų analizę apžvalgų stulpeliams ir išsaugosite rezultatus duomenų rinkinyje.

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

Jei atliktumėte sentimentų analizę neigiamų ir teigiamų apžvalgų stulpeliuose, tai galėtų užtrukti ilgai. Testuota galingame testavimo nešiojamame kompiuteryje su greitu procesoriumi, tai užtruko 12–14 minučių, priklausomai nuo to, kuri sentimentų biblioteka buvo naudojama. Tai yra (santykinai) ilgas laikas, todėl verta ištirti, ar tai galima pagreitinti. 

Stop žodžių, arba dažnų anglų kalbos žodžių, kurie nekeičia sakinio sentimentų, pašalinimas yra pirmas žingsnis. Pašalinus juos, sentimentų analizė turėtų vykti greičiau, tačiau nebūti mažiau tiksli (nes stop žodžiai neturi įtakos sentimentams, tačiau jie sulėtina analizę). 

Ilgiausia neigiama apžvalga buvo 395 žodžiai, tačiau pašalinus stop žodžius, ji sumažėjo iki 195 žodžių.

Stop žodžių pašalinimas taip pat yra greita operacija, pašalinti stop žodžius iš 2 apžvalgų stulpelių per 515 000 eilučių užtruko 3,3 sekundės testavimo įrenginyje. Tai galėtų užtrukti šiek tiek daugiau ar mažiau laiko, priklausomai nuo jūsų įrenginio procesoriaus greičio, RAM, ar turite SSD, ir kitų veiksnių. Operacijos trumpumas reiškia, kad jei tai pagerina sentimentų analizės laiką, verta tai atlikti.

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
Dabar turėtumėte apskaičiuoti sentimentų analizę tiek neigiamų, tiek teigiamų atsiliepimų stulpeliams ir rezultatus išsaugoti dviejuose naujuose stulpeliuose. Sentimentų testas bus palyginti jį su recenzento įvertinimu už tą patį atsiliepimą. Pavyzdžiui, jei sentimentų analizė rodo, kad neigiamas atsiliepimas turi sentimentą 1 (labai teigiamas sentimentas) ir teigiamas atsiliepimas taip pat turi sentimentą 1, tačiau recenzentas viešbučiui suteikė žemiausią įmanomą įvertinimą, tuomet arba atsiliepimo tekstas neatitinka įvertinimo, arba sentimentų analizatorius negalėjo teisingai atpažinti sentimentų. Turėtumėte tikėtis, kad kai kurie sentimentų įvertinimai bus visiškai neteisingi, ir dažnai tai bus paaiškinama, pvz., atsiliepimas gali būti labai sarkastiškas: „Žinoma, man LABAI patiko miegoti kambaryje be šildymo“, o sentimentų analizatorius mano, kad tai teigiamas sentimentas, nors žmogus, skaitantis tai, suprastų, kad tai sarkazmas.

NLTK siūlo įvairius sentimentų analizatorius, kuriuos galima išbandyti, ir galite juos pakeisti, kad pamatytumėte, ar sentimentų analizė tampa tikslesnė. Čia naudojama VADER sentimentų analizė.

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

Vėliau savo programoje, kai būsite pasiruošę apskaičiuoti sentimentus, galite juos pritaikyti kiekvienam atsiliepimui taip:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Tai užtrunka maždaug 120 sekundžių mano kompiuteryje, tačiau laikas gali skirtis priklausomai nuo kompiuterio. Jei norite išspausdinti rezultatus ir patikrinti, ar sentimentai atitinka atsiliepimą:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Paskutinis dalykas, kurį reikia padaryti su failu prieš naudojant jį iššūkyje, yra jį išsaugoti! Taip pat turėtumėte apsvarstyti galimybę pertvarkyti visus naujus stulpelius, kad jie būtų patogesni naudoti (tai kosmetinis pakeitimas žmogui).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Turėtumėte paleisti visą kodą [analizės užrašų knygelėje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (po to, kai paleidote [filtravimo užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), kad sugeneruotumėte Hotel_Reviews_Filtered.csv failą).

Apžvelkime žingsnius:

1. Originalus duomenų rinkinys **Hotel_Reviews.csv** buvo analizuotas ankstesnėje pamokoje naudojant [tyrimo užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv buvo filtruotas naudojant [filtravimo užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), rezultatas - **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv buvo apdorotas naudojant [sentimentų analizės užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), rezultatas - **Hotel_Reviews_NLP.csv**
4. Naudokite Hotel_Reviews_NLP.csv NLP iššūkyje žemiau

### Išvada

Kai pradėjote, turėjote duomenų rinkinį su stulpeliais ir duomenimis, tačiau ne visi jie galėjo būti patikrinti ar panaudoti. Jūs ištyrėte duomenis, išfiltravote tai, ko nereikia, konvertavote žymes į kažką naudingo, apskaičiavote savo vidurkius, pridėjote sentimentų stulpelius ir, tikėkimės, sužinojote įdomių dalykų apie natūralios kalbos apdorojimą.

## [Po paskaitos testas](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Iššūkis

Dabar, kai jūsų duomenų rinkinys yra analizuotas sentimentų atžvilgiu, pabandykite naudoti strategijas, kurias išmokote šiame kurse (galbūt klasterizavimą?), kad nustatytumėte sentimentų modelius.

## Apžvalga ir savarankiškas mokymasis

Pereikite [šį mokymosi modulį](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), kad sužinotumėte daugiau ir naudotumėte skirtingus įrankius sentimentų analizei tekste.

## Užduotis

[Išbandykite kitą duomenų rinkinį](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipiame dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus aiškinimus, kylančius dėl šio vertimo naudojimo.