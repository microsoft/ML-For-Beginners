<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T14:20:45+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "sl"
}
-->
# Analiza sentimenta s hotelskimi ocenami

Zdaj, ko ste podrobno raziskali podatkovni niz, je čas, da filtrirate stolpce in nato uporabite tehnike NLP na podatkovnem nizu, da pridobite nove vpoglede o hotelih.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

### Operacije filtriranja in analize sentimenta

Kot ste verjetno opazili, ima podatkovni niz nekaj težav. Nekateri stolpci so napolnjeni z neuporabnimi informacijami, drugi se zdijo napačni. Če so pravilni, ni jasno, kako so bili izračunani, in odgovore ni mogoče neodvisno preveriti z lastnimi izračuni.

## Naloga: malo več obdelave podatkov

Podatke očistite še malo. Dodajte stolpce, ki bodo uporabni kasneje, spremenite vrednosti v drugih stolpcih in nekatere stolpce popolnoma odstranite.

1. Začetna obdelava stolpcev

   1. Odstranite `lat` in `lng`.

   2. Zamenjajte vrednosti `Hotel_Address` z naslednjimi vrednostmi (če naslov vsebuje ime mesta in države, ga spremenite v samo mesto in državo).

      To so edina mesta in države v podatkovnem nizu:

      Amsterdam, Nizozemska

      Barcelona, Španija

      London, Združeno kraljestvo

      Milano, Italija

      Pariz, Francija

      Dunaj, Avstrija 

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

      Zdaj lahko poizvedujete podatke na ravni države:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Nizozemska  |    105     |
      | Barcelona, Španija     |    211     |
      | London, Združeno kraljestvo | 400   |
      | Milano, Italija        |    162     |
      | Pariz, Francija        |    458     |
      | Dunaj, Avstrija        |    158     |

2. Obdelava stolpcev meta-ocene hotela

   1. Odstranite `Additional_Number_of_Scoring`.

   2. Zamenjajte `Total_Number_of_Reviews` s skupnim številom ocen za ta hotel, ki so dejansko v podatkovnem nizu.

   3. Zamenjajte `Average_Score` z lastno izračunano oceno.

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Obdelava stolpcev ocen

   1. Odstranite `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` in `days_since_review`.

   2. Obdržite `Reviewer_Score`, `Negative_Review` in `Positive_Review` takšne, kot so.

   3. Za zdaj obdržite `Tags`.

      - V naslednjem razdelku bomo izvedli dodatne operacije filtriranja na oznakah, nato pa bodo oznake odstranjene.

4. Obdelava stolpcev ocenjevalcev

   1. Odstranite `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Obdržite `Reviewer_Nationality`.

### Stolpec oznak

Stolpec `Tag` je problematičen, saj je seznam (v obliki besedila), shranjen v stolpcu. Na žalost vrstni red in število podsekcij v tem stolpcu nista vedno enaka. Težko je za človeka identificirati pravilne fraze, ki bi ga zanimale, ker je 515.000 vrstic in 1427 hotelov, vsak pa ima nekoliko drugačne možnosti, ki jih lahko izbere ocenjevalec. Tukaj pride NLP do izraza. Besedilo lahko pregledate, najdete najpogostejše fraze in jih preštejete.

Na žalost nas ne zanimajo posamezne besede, temveč večbesedne fraze (npr. *Poslovno potovanje*). Zagon algoritma za frekvenčno porazdelitev večbesednih fraz na toliko podatkov (6762646 besed) bi lahko trajal izjemno dolgo, vendar brez pregleda podatkov bi se zdelo, da je to nujen strošek. Tukaj je koristna raziskovalna analiza podatkov, saj ste videli vzorec oznak, kot so `[' Poslovno potovanje  ', ' Samostojni potnik ', ' Enoposteljna soba ', ' Bivanje 5 noči ', ' Oddano z mobilne naprave ']`, lahko začnete spraševati, ali je mogoče močno zmanjšati obdelavo, ki jo morate opraviti. Na srečo je to mogoče - vendar morate najprej slediti nekaj korakom, da določite zanimive oznake.

### Filtriranje oznak

Ne pozabite, da je cilj podatkovnega niza dodati sentiment in stolpce, ki vam bodo pomagali izbrati najboljši hotel (za vas ali morda za stranko, ki vas prosi, da ustvarite bot za priporočanje hotelov). Morate se vprašati, ali so oznake koristne ali ne v končnem podatkovnem nizu. Tukaj je ena interpretacija (če bi potrebovali podatkovni niz za druge namene, bi lahko bile oznake drugače vključene/izključene):

1. Vrsta potovanja je pomembna in naj ostane.
2. Vrsta skupine gostov je pomembna in naj ostane.
3. Vrsta sobe, apartmaja ali studia, v katerem je gost bival, je nepomembna (vsi hoteli imajo v bistvu enake sobe).
4. Naprava, na kateri je bila ocena oddana, je nepomembna.
5. Število noči, ki jih je ocenjevalec bival, *bi* lahko bilo pomembno, če bi daljše bivanje povezali z večjim zadovoljstvom hotela, vendar je to malo verjetno in verjetno nepomembno.

Povzetek: **obdržite 2 vrsti oznak in odstranite ostale**.

Najprej ne želite šteti oznak, dokler niso v boljši obliki, kar pomeni odstranitev oglatih oklepajev in narekovajev. To lahko storite na več načinov, vendar želite najhitrejši, saj bi obdelava velike količine podatkov lahko trajala dolgo. Na srečo ima pandas enostaven način za izvedbo vsakega od teh korakov.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Vsaka oznaka postane nekaj takega: `Poslovno potovanje, Samostojni potnik, Enoposteljna soba, Bivanje 5 noči, Oddano z mobilne naprave`.

Nato naletite na težavo. Nekatere ocene ali vrstice imajo 5 stolpcev, nekatere 3, nekatere 6. To je rezultat načina, kako je bil podatkovni niz ustvarjen, in težko ga je popraviti. Želite dobiti frekvenčno število vsake fraze, vendar so v različnih vrstah v vsaki oceni, zato je lahko število napačno, hotel pa morda ne dobi oznake, ki si jo zasluži.

Namesto tega boste uporabili različne vrste v svojo korist, saj je vsaka oznaka večbesedna, hkrati pa ločena z vejico! Najenostavnejši način za to je, da ustvarite 6 začasnih stolpcev, v katere vstavite vsako oznako glede na njen vrstni red v oznaki. Nato lahko združite teh 6 stolpcev v en velik stolpec in zaženete metodo `value_counts()` na nastalem stolpcu. Če to natisnete, boste videli, da je bilo 2428 edinstvenih oznak. Tukaj je majhen vzorec:

| Oznaka                          | Število |
| ------------------------------- | ------- |
| Prostočasno potovanje           | 417778  |
| Oddano z mobilne naprave        | 307640  |
| Par                             | 252294  |
| Bivanje 1 noč                   | 193645  |
| Bivanje 2 noči                  | 133937  |
| Samostojni potnik               | 108545  |
| Bivanje 3 noči                  | 95821   |
| Poslovno potovanje              | 82939   |
| Skupina                         | 65392   |
| Družina z majhnimi otroki       | 61015   |
| Bivanje 4 noči                  | 47817   |
| Dvoposteljna soba               | 35207   |
| Standardna dvoposteljna soba    | 32248   |
| Superior dvoposteljna soba      | 31393   |
| Družina z večjimi otroki        | 26349   |
| Deluxe dvoposteljna soba        | 24823   |
| Dvoposteljna ali enoposteljna soba | 22393 |
| Bivanje 5 noči                  | 20845   |
| Standardna dvoposteljna ali enoposteljna soba | 17483 |
| Klasična dvoposteljna soba      | 16989   |
| Superior dvoposteljna ali enoposteljna soba | 13570 |
| 2 sobe                          | 12393   |

Nekatere pogoste oznake, kot je `Oddano z mobilne naprave`, nam niso koristne, zato bi jih bilo pametno odstraniti, preden štejemo pojavnost fraz, vendar je to tako hitra operacija, da jih lahko pustite notri in jih ignorirate.

### Odstranjevanje oznak dolžine bivanja

Odstranjevanje teh oznak je prvi korak, saj nekoliko zmanjša skupno število oznak, ki jih je treba upoštevati. Upoštevajte, da jih ne odstranite iz podatkovnega niza, temveč se odločite, da jih ne upoštevate kot vrednosti za štetje/obdržanje v podatkovnem nizu ocen.

| Dolžina bivanja | Število |
| --------------- | ------- |
| Bivanje 1 noč   | 193645  |
| Bivanje 2 noči  | 133937  |
| Bivanje 3 noči  | 95821   |
| Bivanje 4 noči  | 47817   |
| Bivanje 5 noči  | 20845   |
| Bivanje 6 noči  | 9776    |
| Bivanje 7 noči  | 7399    |
| Bivanje 8 noči  | 2502    |
| Bivanje 9 noči  | 1293    |
| ...             | ...     |

Obstaja ogromna raznolikost sob, apartmajev, studiev, stanovanj in podobno. Vse to pomeni približno isto stvar in ni relevantno za vas, zato jih odstranite iz obravnave.

| Vrsta sobe                     | Število |
| ------------------------------ | ------- |
| Dvoposteljna soba              | 35207   |
| Standardna dvoposteljna soba   | 32248   |
| Superior dvoposteljna soba     | 31393   |
| Deluxe dvoposteljna soba       | 24823   |
| Dvoposteljna ali enoposteljna soba | 22393 |
| Standardna dvoposteljna ali enoposteljna soba | 17483 |
| Klasična dvoposteljna soba     | 16989   |
| Superior dvoposteljna ali enoposteljna soba | 13570 |

Na koncu, in to je razveseljivo (ker ni zahtevalo veliko obdelave), boste ostali z naslednjimi *koristnimi* oznakami:

| Oznaka                                       | Število |
| ------------------------------------------- | ------- |
| Prostočasno potovanje                       | 417778  |
| Par                                         | 252294  |
| Samostojni potnik                           | 108545  |
| Poslovno potovanje                          | 82939   |
| Skupina (združeno s Potniki s prijatelji)   | 67535   |
| Družina z majhnimi otroki                   | 61015   |
| Družina z večjimi otroki                    | 26349   |
| S hišnim ljubljenčkom                      | 1405    |

Lahko bi trdili, da je `Potniki s prijatelji` približno enako kot `Skupina`, in to bi bilo pošteno združiti, kot je prikazano zgoraj. Koda za identifikacijo pravilnih oznak je v [zvezku z oznakami](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Zadnji korak je ustvariti nove stolpce za vsako od teh oznak. Nato za vsako vrstico ocene, če stolpec `Tag` ustreza enemu od novih stolpcev, dodajte 1, če ne, dodajte 0. Končni rezultat bo število, koliko ocenjevalcev je izbralo ta hotel (v agregatu) za, recimo, poslovno ali prostočasno potovanje, ali za bivanje s hišnim ljubljenčkom, kar je koristna informacija pri priporočanju hotela.

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

### Shranite datoteko

Na koncu shranite podatkovni niz, kot je zdaj, z novim imenom.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operacije analize sentimenta

V tem zadnjem razdelku boste uporabili analizo sentimenta na stolpcih ocen in shranili rezultate v podatkovni niz.

## Naloga: naložite in shranite filtrirane podatke

Upoštevajte, da zdaj nalagate filtrirani podatkovni niz, ki je bil shranjen v prejšnjem razdelku, **ne** originalnega podatkovnega niza.

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

### Odstranjevanje nepomembnih besed

Če bi izvedli analizo sentimenta na stolpcih negativnih in pozitivnih ocen, bi to lahko trajalo dolgo. Testirano na zmogljivem prenosniku z hitrim procesorjem je trajalo 12–14 minut, odvisno od uporabljenega knjižničnega orodja za analizo sentimenta. To je (relativno) dolgo, zato je vredno raziskati, ali je mogoče to pospešiti.

Odstranjevanje nepomembnih besed, ali pogostih angleških besed, ki ne spreminjajo sentimenta stavka, je prvi korak. Z odstranitvijo teh besed bi morala analiza sentimenta potekati hitreje, vendar ne biti manj natančna (saj nepomembne besede ne vplivajo na sentiment, vendar upočasnjujejo analizo).

Najdaljša negativna ocena je imela 395 besed, vendar po odstranitvi nepomembnih besed ima 195 besed.

Odstranjevanje nepomembnih besed je tudi hitra operacija; odstranjevanje nepomembnih besed iz 2 stolpcev ocen čez 515.000 vrstic je trajalo 3,3 sekunde na testni napravi. Lahko traja nekoliko več ali manj časa, odvisno od hitrosti procesorja vaše naprave, RAM-a, ali imate SSD in nekaterih drugih dejavnikov. Relativna kratkost operacije pomeni, da če izboljša čas analize sentimenta, je vredno to storiti.

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

### Izvajanje analize sentimenta

Zdaj morate izračunati analizo sentimenta za oba stolpca ocen, negativne in pozitivne, ter rezultat shraniti v 2 nova stolpca. Test sentimenta bo primerjava z oceno ocenjevalca za isto oceno. Na primer, če sentiment meni, da ima negativna ocena sentiment 1 (izjemno pozitiven sentiment) in pozitivna ocena sentiment 1, vendar je ocenjevalec hotelu dal najnižjo možno oceno, potem bodisi besedilo ocene ne ustreza oceni, bodisi analizator sentimenta ni mogel pravilno prepoznati sentimenta. Pričakujete lahko, da bodo nekatere ocene sentimenta popolnoma napačne, kar bo pogosto razložljivo, npr. ocena bi lahko bila izjemno sarkastična "Seveda sem OBOŽEVAL spanje v sobi brez ogrevanja", analizator sentimenta pa meni, da je to pozitiven sentiment, čeprav bi človek, ki to bere, vedel, da gre za sarkazem.
NLTK ponuja različne analizatorje sentimenta, s katerimi se lahko učite, in jih lahko zamenjate ter preverite, ali je sentiment bolj ali manj natančen. Tukaj je uporabljena analiza sentimenta VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, junij 2014.

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

Kasneje v programu, ko ste pripravljeni izračunati sentiment, ga lahko uporabite za vsako oceno na naslednji način:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

To traja približno 120 sekund na mojem računalniku, vendar se bo čas razlikoval glede na računalnik. Če želite natisniti rezultate in preveriti, ali sentiment ustreza oceni:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Zadnja stvar, ki jo morate narediti s datoteko, preden jo uporabite v izzivu, je, da jo shranite! Prav tako bi morali razmisliti o preurejanju vseh novih stolpcev, da bodo bolj pregledni (za človeka je to kozmetična sprememba).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Zaženite celotno kodo za [analizni zvezek](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (po tem, ko ste zagnali [zvezek za filtriranje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), da ustvarite datoteko Hotel_Reviews_Filtered.csv).

Če povzamemo, koraki so:

1. Izvirna datoteka **Hotel_Reviews.csv** je bila raziskana v prejšnji lekciji z [raziskovalnim zvezkom](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv je filtriran z [zvezkom za filtriranje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), kar ustvari **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv je obdelan z [zvezkom za analizo sentimenta](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), kar ustvari **Hotel_Reviews_NLP.csv**
4. Uporabite Hotel_Reviews_NLP.csv v NLP izzivu spodaj

### Zaključek

Ko ste začeli, ste imeli podatkovno zbirko s stolpci in podatki, vendar ni bilo vse mogoče preveriti ali uporabiti. Raziskali ste podatke, filtrirali, kar ni bilo potrebno, pretvorili oznake v nekaj uporabnega, izračunali svoje povprečje, dodali nekaj stolpcev sentimenta in upajmo, da ste se naučili nekaj zanimivega o obdelavi naravnega besedila.

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Izziv

Zdaj, ko imate analizirano podatkovno zbirko za sentiment, preverite, ali lahko uporabite strategije, ki ste se jih naučili v tem učnem načrtu (morda razvrščanje v skupine?), da določite vzorce glede sentimenta.

## Pregled in samostojno učenje

Vzemite [ta modul Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), da se naučite več in uporabite različna orodja za raziskovanje sentimenta v besedilu.

## Naloga

[Preizkusite drugo podatkovno zbirko](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.