<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-10-11T11:32:40+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "et"
}
-->
# Sentimentaanalüüs hotelliarvustustega

Nüüd, kui olete andmestikku põhjalikult uurinud, on aeg filtreerida veerud ja kasutada NLP-tehnikaid, et saada hotellide kohta uusi teadmisi.

## [Eelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

### Filtreerimis- ja sentimentaanalüüsi toimingud

Nagu olete ilmselt märganud, on andmestikul mõned probleemid. Mõned veerud sisaldavad kasutut teavet, teised tunduvad ebatäpsed. Kui need on õiged, on ebaselge, kuidas need arvutati, ja vastuseid ei saa iseseisvalt oma arvutustega kontrollida.

## Harjutus: veidi rohkem andmetöötlust

Puhastage andmeid veelgi. Lisage veerud, mis on hiljem kasulikud, muutke väärtusi teistes veergudes ja kustutage teatud veerud täielikult.

1. Esmane veerutöötlus

   1. Kustutage `lat` ja `lng`

   2. Asendage `Hotel_Address` väärtused järgmiste väärtustega (kui aadress sisaldab linna ja riigi nime, muutke see ainult linnaks ja riigiks).

      Need on ainsad linnad ja riigid andmestikus:

      Amsterdam, Holland

      Barcelona, Hispaania

      London, Ühendkuningriik

      Milano, Itaalia

      Pariis, Prantsusmaa

      Viin, Austria 

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

      Nüüd saate pärida riigi tasemel andmeid:

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

2. Töötlege hotelli meta-arvustuste veerge

  1. Kustutage `Additional_Number_of_Scoring`

  1. Asendage `Total_Number_of_Reviews` selle hotelliga seotud arvustuste koguarvuga, mis tegelikult andmestikus on 

  1. Asendage `Average_Score` meie enda arvutatud skooriga

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Töötlege arvustuste veerge

   1. Kustutage `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ja `days_since_review`

   2. Säilitage `Reviewer_Score`, `Negative_Review` ja `Positive_Review` nii nagu need on,
     
   3. Säilitage `Tags` praegu

     - Järgmises osas teeme täiendavaid filtreerimistoiminguid siltidega ja siis need kustutatakse

4. Töötlege arvustaja veerge

  1. Kustutage `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Säilitage `Reviewer_Nationality`

### Siltide veerud

`Tag` veerg on probleemne, kuna see on loend (tekstivormis), mis on veerus salvestatud. Kahjuks ei ole selle veeru alajaotuste järjekord ja arv alati ühesugused. Inimesel on raske tuvastada õigeid fraase, mis võiksid huvi pakkuda, kuna andmestikus on 515 000 rida ja 1427 hotelli, ning igal hotellil on veidi erinevad valikud, mida arvustaja võiks valida. Siin tuleb appi NLP. Teksti saab skaneerida, leida kõige levinumad fraasid ja neid loendada.

Kahjuks ei huvita meid üksiksõnad, vaid mitmesõnalised fraasid (nt *Ärireis*). Mitmesõnalise sagedusjaotuse algoritmi käivitamine sellisel hulgal andmetel (6762646 sõna) võib võtta erakordselt palju aega, kuid ilma andmeid vaatamata tundub, et see on vajalik kulu. Siin tuleb kasuks uuriv andmeanalüüs, kuna olete näinud siltide näidist, nagu `[' Ärireis  ', ' Üksik reisija ', ' Üksik tuba ', ' Viibis 5 ööd ', ' Esitatud mobiilseadme kaudu ']`, saate hakata küsima, kas on võimalik oluliselt vähendada töötlemist, mida peate tegema. Õnneks on see võimalik - kuid kõigepealt peate järgima mõningaid samme, et kindlaks teha huvipakkuvad sildid.

### Siltide filtreerimine

Pidage meeles, et andmestiku eesmärk on lisada sentiment ja veerud, mis aitavad teil valida parima hotelli (enda jaoks või võib-olla kliendi jaoks, kes palub teil luua hotelli soovitamise bot). Peate endalt küsima, kas sildid on lõplikus andmestikus kasulikud või mitte. Siin on üks tõlgendus (kui vajate andmestikku muudel põhjustel, võivad erinevad sildid jääda valikusse või välja):

1. Reisi tüüp on asjakohane ja see peaks jääma
2. Külaliste grupi tüüp on oluline ja see peaks jääma
3. Tuba, sviit või stuudio, kus külaline viibis, on ebaoluline (kõigil hotellidel on põhimõtteliselt samad toad)
4. Seade, mille kaudu arvustus esitati, on ebaoluline
5. Ööbimiste arv *võiks* olla asjakohane, kui seostate pikema viibimise hotelliga rahuloluga, kuid see on kahtlane ja tõenäoliselt ebaoluline

Kokkuvõttes **säilitage 2 tüüpi silte ja eemaldage teised**.

Esiteks, te ei soovi silte loendada, kuni need on paremas vormingus, mis tähendab, et eemaldage nurksulud ja jutumärgid. Seda saab teha mitmel viisil, kuid soovite kiireimat, kuna suure hulga andmete töötlemine võib võtta kaua aega. Õnneks on pandasel lihtne viis iga sammu tegemiseks.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Iga silt muutub millekski selliseks: `Ärireis, Üksik reisija, Üksik tuba, Viibis 5 ööd, Esitatud mobiilseadme kaudu`. 

Järgmisena ilmneb probleem. Mõned arvustused või read sisaldavad 5 veergu, mõned 3, mõned 6. See on tingitud sellest, kuidas andmestik loodi, ja seda on raske parandada. Soovite saada iga fraasi sagedusloendit, kuid need on igas arvustuses erinevas järjekorras, mistõttu loend võib olla vale ja hotell ei pruugi saada talle kuuluvat silti.

Selle asemel kasutate erinevat järjekorda enda kasuks, kuna iga silt on mitmesõnaline, kuid ka komaga eraldatud! Lihtsaim viis seda teha on luua 6 ajutist veergu, kus iga silt sisestatakse veergu, mis vastab selle järjekorrale sildis. Seejärel saate need 6 veergu ühendada üheks suureks veeruks ja käivitada `value_counts()` meetodi tulemuseks saadud veerul. Selle välja trükkides näete, et oli 2428 unikaalset silti. Siin on väike näidis:

| Silt                           | Loendus |
| ------------------------------ | ------- |
| Vaba aja reis                 | 417778  |
| Esitatud mobiilseadme kaudu   | 307640  |
| Paar                          | 252294  |
| Viibis 1 öö                   | 193645  |
| Viibis 2 ööd                  | 133937  |
| Üksik reisija                 | 108545  |
| Viibis 3 ööd                  | 95821   |
| Ärireis                      | 82939   |
| Grupp                         | 65392   |
| Pere väikeste lastega         | 61015   |
| Viibis 4 ööd                  | 47817   |
| Kaheinimese tuba              | 35207   |
| Standard kaheinimese tuba     | 32248   |
| Superior kaheinimese tuba     | 31393   |
| Pere vanemate lastega         | 26349   |
| Deluxe kaheinimese tuba       | 24823   |
| Kaheinimese või kahe voodiga tuba | 22393 |
| Viibis 5 ööd                  | 20845   |
| Standard kaheinimese või kahe voodiga tuba | 17483 |
| Klassikaline kaheinimese tuba | 16989   |
| Superior kaheinimese või kahe voodiga tuba | 13570 |
| 2 tuba                        | 12393   |

Mõned levinud sildid, nagu `Esitatud mobiilseadme kaudu`, ei ole meile kasulikud, seega võib olla tark need enne fraasi esinemise loendamist eemaldada, kuid see on nii kiire toiming, et võite need sisse jätta ja ignoreerida.

### Ööbimiste pikkuse siltide eemaldamine

Nende siltide eemaldamine on esimene samm, see vähendab veidi arvestatavate siltide koguarvu. Pange tähele, et te ei eemalda neid andmestikust, vaid otsustate need arvustuste andmestikus loendamise/kasutamise väärtustest välja jätta.

| Ööbimise pikkus | Loendus |
| ---------------- | ------- |
| Viibis 1 öö      | 193645  |
| Viibis 2 ööd     | 133937  |
| Viibis 3 ööd     | 95821   |
| Viibis 4 ööd     | 47817   |
| Viibis 5 ööd     | 20845   |
| Viibis 6 ööd     | 9776    |
| Viibis 7 ööd     | 7399    |
| Viibis 8 ööd     | 2502    |
| Viibis 9 ööd     | 1293    |
| ...              | ...     |

Toatüüpe, sviite, stuudioid, kortereid ja nii edasi on tohutult palju. Need kõik tähendavad enam-vähem sama ja ei ole teile asjakohased, seega eemaldage need arvestusest.

| Toatüüp                     | Loendus |
| --------------------------- | ------- |
| Kaheinimese tuba            | 35207   |
| Standard kaheinimese tuba   | 32248   |
| Superior kaheinimese tuba   | 31393   |
| Deluxe kaheinimese tuba     | 24823   |
| Kaheinimese või kahe voodiga tuba | 22393 |
| Standard kaheinimese või kahe voodiga tuba | 17483 |
| Klassikaline kaheinimese tuba | 16989   |
| Superior kaheinimese või kahe voodiga tuba | 13570 |

Lõpuks, ja see on rõõmustav (sest see ei nõudnud peaaegu üldse töötlemist), jäävad teile järgmised *kasulikud* sildid:

| Silt                                         | Loendus |
| ------------------------------------------- | ------- |
| Vaba aja reis                               | 417778  |
| Paar                                        | 252294  |
| Üksik reisija                               | 108545  |
| Ärireis                                    | 82939   |
| Grupp (ühendatud sõpradega reisijatega)     | 67535   |
| Pere väikeste lastega                       | 61015   |
| Pere vanemate lastega                       | 26349   |
| Lemmikloomaga                               | 1405    |

Võite väita, et `Sõpradega reisijad` on enam-vähem sama mis `Grupp`, ja oleks õiglane need kaks ühendada, nagu eespool. Õige siltide tuvastamise koodi leiate [siltide märkmikust](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Viimane samm on luua uued veerud iga sildi jaoks. Seejärel, iga arvustuse rea puhul, kui `Tag` veerg vastab ühele uuest veerust, lisage 1, kui mitte, lisage 0. Lõpptulemusena saate loenduse, kui palju arvustajaid valis selle hotelli (kogumis) näiteks ärireisiks või vaba aja veetmiseks või lemmiklooma kaasa võtmiseks, ja see on kasulik teave hotelli soovitamisel.

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

### Salvestage oma fail

Lõpuks salvestage andmestik praegusel kujul uue nimega.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentimentaanalüüsi toimingud

Selles viimases osas rakendate sentimentaanalüüsi arvustuste veergudele ja salvestate tulemused andmestikku.

## Harjutus: laadige ja salvestage filtreeritud andmed

Pange tähele, et nüüd laadite filtreeritud andmestiku, mis salvestati eelmises osas, **mitte** algset andmestikku.

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

### Stoppsõnade eemaldamine

Kui teeksite sentimentaanalüüsi negatiivsete ja positiivsete arvustuste veergudel, võib see võtta kaua aega. Testitud võimsal test-sülearvutil kiire protsessoriga, kulus 12–14 minutit sõltuvalt sellest, millist sentimentaanalüüsi teeki kasutati. See on (suhteliselt) pikk aeg, seega tasub uurida, kas seda saab kiirendada. 

Stoppsõnade, ehk tavaliste ingliskeelsete sõnade, mis ei muuda lause sentimenti, eemaldamine on esimene samm. Nende eemaldamisega peaks sentimentaanalüüs töötama kiiremini, kuid mitte vähem täpselt (kuna stoppsõnad ei mõjuta sentimenti, kuid aeglustavad analüüsi). 

Pikim negatiivne arvustus oli 395 sõna, kuid pärast stoppsõnade eemaldamist on see 195 sõna.

Stoppsõnade eemaldamine on samuti kiire toiming, nende eemaldamine 2 arvustuste veerust üle 515 000 rea võttis testseadmel 3,3 sekundit. See võib teie jaoks võtta veidi rohkem või vähem aega sõltuvalt teie seadme protsessori kiirusest, RAM-ist, sellest, kas teil on SSD või mitte, ja mõnest muust tegurist. Toimingu suhteline lühidus tähendab, et kui see parandab sentimentaanalüüsi aega, siis tasub seda teha.

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

### Sentimentaanalüüsi läbiviimine

Nüüd peaksite arvutama sentimentaanalüüsi nii negatiivsete kui positiivsete arvustuste veergude jaoks ja salvestama tulemuse 2 uude veergu. Sentimendi testiks on võrrelda seda sama arvustuse arvustaja skooriga. Näiteks, kui sentiment arvab, et negatiivsel arvustusel oli sentiment 1 (äärmiselt positiivne sentiment) ja positiivse arvustuse sentiment 1, kuid arvustaja andis hotellile madalaima võimaliku skoori, siis kas arvustuse tekst ei vasta skoorile või sentimentaanalüsaator ei suutnud sentimenti õigesti tuvastada. Peaksite eeldama, et mõned sentimenti skoorid on täiesti valed, ja sageli on see seletatav, näiteks arvustus võib olla äärmiselt sarkastiline: "Muidugi MA ARMASTASIN magamist toas, kus polnud kütet" ja sentimentaanalüsaator arvab, et see on positiivne sentiment, kuigi inimene, kes seda loeb, teaks, et see oli sarkasm. 
NLTK pakub erinevaid sentimenti analüüsi tööriistu, millega saab õppida, ja neid saab asendada, et näha, kas sentimenti analüüs on täpsem või mitte. Siin kasutatakse VADER sentimenti analüüsi.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, juuni 2014.

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

Hiljem, kui oled valmis sentimenti arvutama, saad seda rakendada igale arvustusele järgmiselt:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

See võtab minu arvutis umbes 120 sekundit, kuid see võib varieeruda sõltuvalt arvutist. Kui soovid tulemusi välja printida ja näha, kas sentiment vastab arvustusele:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Viimane asi, mida failiga enne väljakutse kasutamist teha, on selle salvestamine! Samuti tasub kaaluda kõigi uute veergude ümberjärjestamist, et nendega oleks lihtsam töötada (inimese jaoks on see kosmeetiline muudatus).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Sa peaksid käivitama kogu koodi [analüüsi märkmikust](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (pärast seda, kui oled käivitanud [filtreerimise märkmiku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), et genereerida Hotel_Reviews_Filtered.csv fail).

Kokkuvõtteks, sammud on järgmised:

1. Algne andmefail **Hotel_Reviews.csv** uuritakse eelmises õppetükis [uurimise märkmiku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) abil
2. Hotel_Reviews.csv filtreeritakse [filtreerimise märkmiku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) abil, mille tulemuseks on **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv töödeldakse [sentimenti analüüsi märkmiku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) abil, mille tulemuseks on **Hotel_Reviews_NLP.csv**
4. Kasuta Hotel_Reviews_NLP.csv allpool NLP väljakutses

### Kokkuvõte

Kui alustasid, oli sul andmestik veergude ja andmetega, kuid mitte kõik ei olnud kontrollitav või kasutatav. Oled andmeid uurinud, filtreerinud välja, mida ei vaja, konverteerinud sildid millekski kasulikuks, arvutanud oma keskmised, lisanud sentimenti veerud ja loodetavasti õppinud huvitavaid asju loomuliku teksti töötlemise kohta.

## [Loengu järgne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Väljakutse

Nüüd, kui sinu andmestik on sentimenti analüüsitud, proovi kasutada õppekavas õpitud strateegiaid (näiteks klasterdamist), et määrata sentimenti mustreid.

## Ülevaade ja iseseisev õppimine

Võta [see Learn moodul](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), et õppida rohkem ja kasutada erinevaid tööriistu sentimenti uurimiseks tekstis.

## Ülesanne

[Proovi teistsugust andmestikku](assignment.md)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.