<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T17:08:11+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "sk"
}
-->
# Analýza sentimentu pomocou recenzií hotelov

Teraz, keď ste podrobne preskúmali dataset, je čas filtrovať stĺpce a použiť techniky NLP na získanie nových poznatkov o hoteloch.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

### Operácie filtrovania a analýzy sentimentu

Ako ste si pravdepodobne všimli, dataset má niekoľko problémov. Niektoré stĺpce sú plné zbytočných informácií, iné sa zdajú byť nesprávne. Ak sú správne, nie je jasné, ako boli vypočítané, a odpovede nemôžete nezávisle overiť vlastnými výpočtami.

## Cvičenie: ďalšie spracovanie dát

Vyčistite dáta ešte o niečo viac. Pridajte stĺpce, ktoré budú užitočné neskôr, zmeňte hodnoty v iných stĺpcoch a niektoré stĺpce úplne odstráňte.

1. Počiatočné spracovanie stĺpcov

   1. Odstráňte `lat` a `lng`

   2. Nahraďte hodnoty `Hotel_Address` nasledujúcimi hodnotami (ak adresa obsahuje názov mesta a krajiny, zmeňte ju na iba mesto a krajinu).

      Toto sú jediné mestá a krajiny v datasete:

      Amsterdam, Holandsko

      Barcelona, Španielsko

      Londýn, Spojené kráľovstvo

      Miláno, Taliansko

      Paríž, Francúzsko

      Viedeň, Rakúsko 

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

      Teraz môžete dotazovať údaje na úrovni krajiny:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Holandsko   |    105     |
      | Barcelona, Španielsko  |    211     |
      | Londýn, Spojené kráľovstvo | 400     |
      | Miláno, Taliansko      |    162     |
      | Paríž, Francúzsko      |    458     |
      | Viedeň, Rakúsko        |    158     |

2. Spracovanie stĺpcov meta-recenzie hotela

  1. Odstráňte `Additional_Number_of_Scoring`

  1. Nahraďte `Total_Number_of_Reviews` celkovým počtom recenzií pre daný hotel, ktoré sú skutočne v datasete 

  1. Nahraďte `Average_Score` vlastným vypočítaným skóre

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Spracovanie stĺpcov recenzií

   1. Odstráňte `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` a `days_since_review`

   2. Zachovajte `Reviewer_Score`, `Negative_Review` a `Positive_Review` tak, ako sú,
     
   3. Zatiaľ ponechajte `Tags`

     - V ďalšej časti vykonáme ďalšie operácie filtrovania na tagoch a potom tagy odstránime

4. Spracovanie stĺpcov recenzentov

  1. Odstráňte `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Zachovajte `Reviewer_Nationality`

### Stĺpce tagov

Stĺpec `Tag` je problematický, pretože je to zoznam (vo forme textu) uložený v stĺpci. Bohužiaľ, poradie a počet podsekcií v tomto stĺpci nie sú vždy rovnaké. Je ťažké pre človeka identifikovať správne frázy, ktoré by ho mohli zaujímať, pretože existuje 515 000 riadkov a 1427 hotelov, pričom každý má mierne odlišné možnosti, ktoré si recenzent mohol vybrať. Tu prichádza na rad NLP. Môžete prehľadať text a nájsť najčastejšie frázy a spočítať ich.

Bohužiaľ, nezaujímajú nás jednotlivé slová, ale viacslovné frázy (napr. *Služobná cesta*). Spustenie algoritmu na frekvenčnú distribúciu viacslovných fráz na takom množstve dát (6762646 slov) by mohlo trvať mimoriadne dlho, ale bez pohľadu na dáta by sa zdalo, že je to nevyhnutný výdavok. Tu je užitočná prieskumná analýza dát, pretože ste videli vzorku tagov, ako napríklad `[' Služobná cesta  ', ' Cestovateľ sólo ', ' Jednolôžková izba ', ' Pobyt na 5 nocí ', ' Odoslané z mobilného zariadenia ']`, môžete začať uvažovať, či je možné výrazne znížiť spracovanie, ktoré musíte vykonať. Našťastie je to možné - ale najprv musíte dodržať niekoľko krokov na určenie zaujímavých tagov.

### Filtrovanie tagov

Pamätajte, že cieľom datasetu je pridať sentiment a stĺpce, ktoré vám pomôžu vybrať najlepší hotel (pre seba alebo možno pre klienta, ktorý vás poveril vytvorením bota na odporúčanie hotelov). Musíte sa sami seba opýtať, či sú tagy užitočné alebo nie vo finálnom datasete. Tu je jedna interpretácia (ak by ste dataset potrebovali na iné účely, mohli by zostať/vypadnúť iné tagy):

1. Typ cesty je relevantný a mal by zostať
2. Typ skupiny hostí je dôležitý a mal by zostať
3. Typ izby, apartmánu alebo štúdia, v ktorom hosť býval, je irelevantný (všetky hotely majú v podstate rovnaké izby)
4. Zariadenie, na ktorom bola recenzia odoslaná, je irelevantné
5. Počet nocí, ktoré recenzent zostal, *môže* byť relevantný, ak by ste pripisovali dlhšie pobyty tomu, že sa hosťovi hotel páčil viac, ale je to diskutabilné a pravdepodobne irelevantné

V súhrne, **ponechajte 2 druhy tagov a ostatné odstráňte**.

Najprv nechcete počítať tagy, kým nie sú v lepšom formáte, čo znamená odstránenie hranatých zátvoriek a úvodzoviek. Môžete to urobiť niekoľkými spôsobmi, ale chcete najrýchlejší, pretože spracovanie veľkého množstva dát môže trvať dlho. Našťastie, pandas má jednoduchý spôsob, ako vykonať každý z týchto krokov.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Každý tag sa stane niečím ako: `Služobná cesta, Cestovateľ sólo, Jednolôžková izba, Pobyt na 5 nocí, Odoslané z mobilného zariadenia`. 

Ďalej narazíme na problém. Niektoré recenzie alebo riadky majú 5 stĺpcov, niektoré 3, niektoré 6. To je výsledok toho, ako bol dataset vytvorený, a je ťažké to opraviť. Chcete získať frekvenčný počet každej frázy, ale sú v rôznom poradí v každej recenzii, takže počet môže byť nesprávny a hotel nemusí dostať tag, ktorý si zaslúžil.

Namiesto toho využijete rôzne poradie vo svoj prospech, pretože každý tag je viacslovný, ale tiež oddelený čiarkou! Najjednoduchší spôsob, ako to urobiť, je vytvoriť 6 dočasných stĺpcov, pričom každý tag vložíte do stĺpca zodpovedajúceho jeho poradiu v tagu. Potom môžete zlúčiť 6 stĺpcov do jedného veľkého stĺpca a spustiť metódu `value_counts()` na výslednom stĺpci. Po vytlačení uvidíte, že existovalo 2428 unikátnych tagov. Tu je malá vzorka:

| Tag                            | Počet  |
| ------------------------------ | ------ |
| Rekreačná cesta                | 417778 |
| Odoslané z mobilného zariadenia | 307640 |
| Pár                            | 252294 |
| Pobyt na 1 noc                 | 193645 |
| Pobyt na 2 noci                | 133937 |
| Cestovateľ sólo                | 108545 |
| Pobyt na 3 noci                | 95821  |
| Služobná cesta                 | 82939  |
| Skupina                        | 65392  |
| Rodina s malými deťmi          | 61015  |
| Pobyt na 4 noci                | 47817  |
| Dvojlôžková izba               | 35207  |
| Štandardná dvojlôžková izba    | 32248  |
| Superior dvojlôžková izba      | 31393  |
| Rodina so staršími deťmi       | 26349  |
| Deluxe dvojlôžková izba        | 24823  |
| Dvojlôžková alebo dvojposteľová izba | 22393  |
| Pobyt na 5 nocí                | 20845  |
| Štandardná dvojlôžková alebo dvojposteľová izba | 17483  |
| Klasická dvojlôžková izba      | 16989  |
| Superior dvojlôžková alebo dvojposteľová izba | 13570  |
| 2 izby                         | 12393  |

Niektoré z bežných tagov, ako napríklad `Odoslané z mobilného zariadenia`, sú pre nás zbytočné, takže by mohlo byť rozumné ich odstrániť pred počítaním výskytu fráz, ale je to tak rýchla operácia, že ich môžete ponechať a ignorovať.

### Odstránenie tagov o dĺžke pobytu

Odstránenie týchto tagov je krok 1, mierne znižuje celkový počet tagov, ktoré treba zvážiť. Všimnite si, že ich neodstraňujete z datasetu, len sa rozhodnete ich nebrať do úvahy ako hodnoty na počítanie/zachovanie v datasete recenzií.

| Dĺžka pobytu   | Počet  |
| -------------- | ------ |
| Pobyt na 1 noc | 193645 |
| Pobyt na 2 noci| 133937 |
| Pobyt na 3 noci| 95821  |
| Pobyt na 4 noci| 47817  |
| Pobyt na 5 nocí| 20845  |
| Pobyt na 6 nocí| 9776   |
| Pobyt na 7 nocí| 7399   |
| Pobyt na 8 nocí| 2502   |
| Pobyt na 9 nocí| 1293   |
| ...            | ...    |

Existuje obrovská rozmanitosť izieb, apartmánov, štúdií, bytov a podobne. Všetky znamenajú približne to isté a nie sú pre vás relevantné, takže ich odstráňte z úvahy.

| Typ izby                     | Počet |
| ---------------------------- | ----- |
| Dvojlôžková izba             | 35207 |
| Štandardná dvojlôžková izba  | 32248 |
| Superior dvojlôžková izba    | 31393 |
| Deluxe dvojlôžková izba      | 24823 |
| Dvojlôžková alebo dvojposteľová izba | 22393 |
| Štandardná dvojlôžková alebo dvojposteľová izba | 17483 |
| Klasická dvojlôžková izba    | 16989 |
| Superior dvojlôžková alebo dvojposteľová izba | 13570 |

Nakoniec, a to je potešujúce (pretože to nevyžadovalo veľa spracovania), zostanú vám nasledujúce *užitočné* tagy:

| Tag                                           | Počet  |
| --------------------------------------------- | ------ |
| Rekreačná cesta                               | 417778 |
| Pár                                           | 252294 |
| Cestovateľ sólo                               | 108545 |
| Služobná cesta                                | 82939  |
| Skupina (spojené s Cestovatelia s priateľmi)  | 67535  |
| Rodina s malými deťmi                         | 61015  |
| Rodina so staršími deťmi                      | 26349  |
| S domácim miláčikom                           | 1405   |

Môžete argumentovať, že `Cestovatelia s priateľmi` je viac-menej to isté ako `Skupina`, a bolo by spravodlivé spojiť tieto dva tagy, ako je uvedené vyššie. Kód na identifikáciu správnych tagov je [notebook Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Posledným krokom je vytvoriť nové stĺpce pre každý z týchto tagov. Potom, pre každý riadok recenzie, ak stĺpec `Tag` zodpovedá jednému z nových stĺpcov, pridajte 1, ak nie, pridajte 0. Konečným výsledkom bude počet recenzentov, ktorí si vybrali tento hotel (v súhrne) napríklad na služobnú cestu, rekreačnú cestu alebo na pobyt s domácim miláčikom, a to sú užitočné informácie pri odporúčaní hotela.

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

### Uložte svoj súbor

Nakoniec uložte dataset v aktuálnom stave pod novým názvom.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operácie analýzy sentimentu

V tejto poslednej časti aplikujete analýzu sentimentu na stĺpce recenzií a uložíte výsledky do datasetu.

## Cvičenie: načítanie a uloženie filtrovaných dát

Všimnite si, že teraz načítavate filtrovaný dataset, ktorý bol uložený v predchádzajúcej časti, **nie** pôvodný dataset.

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

### Odstránenie stop slov

Ak by ste spustili analýzu sentimentu na stĺpcoch negatívnych a pozitívnych recenzií, mohlo by to trvať dlho. Testované na výkonnom testovacom notebooku s rýchlym CPU, trvalo to 12 - 14 minút v závislosti od použitej knižnice sentimentu. To je (relatívne) dlhý čas, takže stojí za to preskúmať, či sa to dá urýchliť. 

Odstránenie stop slov, alebo bežných anglických slov, ktoré nemenia sentiment vety, je prvým krokom. Ich odstránením by mala analýza sentimentu prebiehať rýchlejšie, ale nebude menej presná (keďže stop slová neovplyvňujú sentiment, ale spomaľujú analýzu). 

Najdlhšia negatívna recenzia mala 395 slov, ale po odstránení stop slov má 195 slov.

Odstránenie stop slov je tiež rýchla operácia, odstránenie stop slov z 2 stĺpcov recenzií cez 515 000 riadkov trvalo 3,3 sekundy na testovacom zariadení. Môže to trvať o niečo viac alebo menej času v závislosti od rýchlosti vášho CPU, RAM, či máte SSD alebo nie, a niektorých ďalších faktorov. Relatívna krátkosť operácie znamená, že ak zlepší čas analýzy sentimentu, potom sa oplatí vykonať.

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

### Vykonanie analýzy sentimentu

Teraz by ste mali vypočítať analýzu sentimentu pre stĺpce negatívnych a pozitívnych recenzií a uložiť výsledok do 2 nových stĺpcov. Test sentimentu bude porovnať ho so skóre recenzenta pre tú istú recenziu. Napríklad, ak sentiment ukazuje, že negatívna recenzia mala sentiment 1 (extrémne pozitívny sentiment) a pozitívna recenzia sentiment 1, ale recenzent dal hotelu najnižšie možné skóre, potom buď text recenzie nezodpovedá skóre, alebo sentimentový analyzátor nedokázal správne rozpoznať sentiment. Mali by ste očakávať, že niektoré skóre sentimentu budú úplne nesprávne, a často to bude vysvetliteľné, napríklad recenzia môže byť extrémne sarkastická "Samozrejme, že som MILOVAL spanie v izbe bez kúrenia" a sentimentový analyzátor si myslí, že je to pozitívny sentiment, aj keď človek, ktorý to číta, by vedel, že ide o sarkazmus.
NLTK poskytuje rôzne analyzátory sentimentu, s ktorými sa môžete učiť, a môžete ich zameniť, aby ste zistili, či je sentiment presnejší alebo menej presný. Na analýzu sentimentu sa tu používa VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, jún 2014.

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

Neskôr vo vašom programe, keď budete pripravení vypočítať sentiment, môžete ho aplikovať na každú recenziu nasledovne:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Toto trvá približne 120 sekúnd na mojom počítači, ale na každom počítači sa to môže líšiť. Ak chcete vytlačiť výsledky a zistiť, či sentiment zodpovedá recenzii:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Posledná vec, ktorú treba urobiť so súborom pred jeho použitím vo výzve, je uložiť ho! Mali by ste tiež zvážiť preusporiadanie všetkých nových stĺpcov tak, aby sa s nimi ľahšie pracovalo (pre človeka je to kozmetická zmena).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Mali by ste spustiť celý kód pre [analytický notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (po tom, čo ste spustili [notebook na filtrovanie](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) na generovanie súboru Hotel_Reviews_Filtered.csv).

Na zhrnutie, kroky sú:

1. Pôvodný súbor datasetu **Hotel_Reviews.csv** je preskúmaný v predchádzajúcej lekcii pomocou [exploračného notebooku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv je filtrovaný pomocou [notebooku na filtrovanie](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), čo vedie k súboru **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv je spracovaný pomocou [notebooku na analýzu sentimentu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), čo vedie k súboru **Hotel_Reviews_NLP.csv**
4. Použite Hotel_Reviews_NLP.csv vo výzve NLP nižšie

### Záver

Keď ste začínali, mali ste dataset so stĺpcami a údajmi, ale nie všetky z nich mohli byť overené alebo použité. Preskúmali ste údaje, odfiltrovali to, čo nepotrebujete, konvertovali značky na niečo užitočné, vypočítali svoje vlastné priemery, pridali niektoré stĺpce sentimentu a dúfajme, že ste sa naučili niečo zaujímavé o spracovaní prirodzeného textu.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Výzva

Teraz, keď máte dataset analyzovaný na sentiment, skúste použiť stratégie, ktoré ste sa naučili v tomto kurze (napríklad zhlukovanie), aby ste určili vzory okolo sentimentu.

## Prehľad a samostatné štúdium

Vezmite si [tento modul Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), aby ste sa dozvedeli viac a použili rôzne nástroje na skúmanie sentimentu v texte.

## Zadanie

[Vyskúšajte iný dataset](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.