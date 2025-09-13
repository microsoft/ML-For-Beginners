<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T17:07:14+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "hu"
}
-->
# Érzelemfelismerés szállodai vélemények alapján

Most, hogy részletesen megvizsgáltad az adatállományt, itt az ideje, hogy szűrd az oszlopokat, majd NLP technikákat alkalmazz az adatállományon, hogy új betekintést nyerj a szállodákról.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Szűrési és érzelemfelismerési műveletek

Ahogy valószínűleg észrevetted, az adatállományban van néhány probléma. Néhány oszlop haszontalan információval van tele, mások hibásnak tűnnek. Ha helyesek is, nem világos, hogyan számították ki őket, és az eredmények nem ellenőrizhetők független számításokkal.

## Gyakorlat: további adatfeldolgozás

Tisztítsd meg az adatokat egy kicsit jobban. Adj hozzá oszlopokat, amelyek később hasznosak lesznek, módosítsd más oszlopok értékeit, és teljesen törölj bizonyos oszlopokat.

1. Kezdeti oszlopfeldolgozás

   1. Töröld a `lat` és `lng` oszlopokat.

   2. Cseréld ki a `Hotel_Address` értékeit az alábbi értékekre (ha a cím tartalmazza a város és az ország nevét, változtasd meg csak a városra és az országra).

      Ezek az adatállományban szereplő egyetlen városok és országok:

      Amszterdam, Hollandia

      Barcelona, Spanyolország

      London, Egyesült Királyság

      Milánó, Olaszország

      Párizs, Franciaország

      Bécs, Ausztria 

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

      Most már lekérdezheted az ország szintű adatokat:

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

2. Szállodai meta-vélemény oszlopok feldolgozása

  1. Töröld az `Additional_Number_of_Scoring` oszlopot.

  1. Cseréld ki a `Total_Number_of_Reviews` értékét az adott szállodához tartozó vélemények tényleges számával az adatállományban.

  1. Cseréld ki az `Average_Score` értékét a saját számított átlagos pontszámunkkal.

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Vélemény oszlopok feldolgozása

   1. Töröld a `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` és `days_since_review` oszlopokat.

   2. Hagyd meg a `Reviewer_Score`, `Negative_Review` és `Positive_Review` oszlopokat változatlanul.
     
   3. Egyelőre hagyd meg a `Tags` oszlopot.

     - A következő szakaszban további szűrési műveleteket végzünk a címkéken, majd a címkék törlésre kerülnek.

4. Véleményező oszlopok feldolgozása

  1. Töröld a `Total_Number_of_Reviews_Reviewer_Has_Given` oszlopot.
  
  2. Hagyd meg a `Reviewer_Nationality` oszlopot.

### Címke oszlopok

A `Tag` oszlop problémás, mivel egy lista (szöveg formájában) van tárolva az oszlopban. Sajnos az alrészek sorrendje és száma nem mindig ugyanaz. Nehéz az ember számára azonosítani a releváns kifejezéseket, mivel 515,000 sor és 1427 szálloda van, és mindegyiknek kissé eltérő lehetőségei vannak, amelyeket a véleményező választhatott. Itt jön képbe az NLP. A szöveget átvizsgálva megtalálhatod a leggyakoribb kifejezéseket, és megszámolhatod őket.

Sajnos nem az egyes szavak érdekelnek minket, hanem több szóból álló kifejezések (pl. *Üzleti út*). Egy több szóból álló gyakorisági eloszlás algoritmus futtatása ekkora adatmennyiségen (6762646 szó) rendkívül sok időt vehet igénybe, de anélkül, hogy megnéznénk az adatokat, úgy tűnik, hogy ez szükséges költség. Itt jön jól az exploratív adatvizsgálat, mivel láttál egy mintát a címkékből, például `[' Üzleti út  ', ' Egyedül utazó ', ' Egyágyas szoba ', ' 5 éjszakát maradt ', ' Mobil eszközről küldve ']`, elkezdheted megkérdezni, hogy lehetséges-e jelentősen csökkenteni a feldolgozást. Szerencsére lehetséges - de először néhány lépést kell követned, hogy meghatározd az érdekes címkéket.

### Címkék szűrése

Ne feledd, hogy az adatállomány célja az érzelmek és oszlopok hozzáadása, amelyek segítenek kiválasztani a legjobb szállodát (számodra vagy esetleg egy ügyfél számára, aki szállodai ajánló botot szeretne készíteni). Fel kell tenned magadnak a kérdést, hogy a címkék hasznosak-e vagy sem a végső adatállományban. Íme egy értelmezés (ha más célból lenne szükséged az adatállományra, különböző címkék maradhatnak benne/kieshetnek):

1. Az utazás típusa releváns, ezt meg kell tartani.
2. A vendégcsoport típusa fontos, ezt meg kell tartani.
3. Az a szoba, lakosztály vagy stúdió típusa, amelyben a vendég tartózkodott, irreleváns (minden szállodában alapvetően ugyanazok a szobák vannak).
4. Az eszköz, amelyről a véleményt beküldték, irreleváns.
5. Az éjszakák száma, amelyet a véleményező ott töltött, *lehet*, hogy releváns, ha hosszabb tartózkodást a szálloda kedvelésével társítasz, de ez erőltetett, és valószínűleg irreleváns.

Összefoglalva, **tarts meg 2 fajta címkét, és távolítsd el a többit**.

Először nem akarod megszámolni a címkéket, amíg nem kerülnek jobb formátumba, tehát ez azt jelenti, hogy el kell távolítani a szögletes zárójeleket és az idézőjeleket. Ezt többféleképpen megteheted, de a leggyorsabbat szeretnéd, mivel sok adat feldolgozása hosszú időt vehet igénybe. Szerencsére a pandas egyszerű módot kínál mindegyik lépés elvégzésére.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Minden címke valami ilyesmivé válik: `Üzleti út, Egyedül utazó, Egyágyas szoba, 5 éjszakát maradt, Mobil eszközről küldve`.

Ezután találunk egy problémát. Néhány vélemény vagy sor 5 oszlopot tartalmaz, néhány 3-at, néhány 6-ot. Ez az adatállomány létrehozásának eredménye, és nehéz javítani. Szeretnéd megszámolni az egyes kifejezések gyakoriságát, de ezek különböző sorrendben vannak minden véleményben, így a számolás hibás lehet, és egy szálloda nem kapja meg azt a címkét, amelyet megérdemelt volna.

Ehelyett a különböző sorrendet az előnyünkre fordítjuk, mivel minden címke több szóból áll, de vesszővel is el van választva! A legegyszerűbb módja ennek az, hogy létrehozunk 6 ideiglenes oszlopot, amelyekbe minden címkét beillesztünk a címke sorrendjének megfelelő oszlopba. Ezután egyesítheted a 6 oszlopot egy nagy oszlopba, és futtathatod a `value_counts()` metódust az eredményoszlopon. Ha ezt kinyomtatod, látni fogod, hogy 2428 egyedi címke volt. Íme egy kis minta:

| Címke                          | Szám   |
| ------------------------------ | ------ |
| Szabadidős utazás              | 417778 |
| Mobil eszközről küldve         | 307640 |
| Pár                            | 252294 |
| 1 éjszakát maradt              | 193645 |
| 2 éjszakát maradt              | 133937 |
| Egyedül utazó                  | 108545 |
| 3 éjszakát maradt              | 95821  |
| Üzleti út                      | 82939  |
| Csoport                        | 65392  |
| Fiatal gyermekes család        | 61015  |
| 4 éjszakát maradt              | 47817  |
| Kétszemélyes szoba             | 35207  |
| Standard kétszemélyes szoba    | 32248  |
| Superior kétszemélyes szoba    | 31393  |
| Idősebb gyermekes család       | 26349  |
| Deluxe kétszemélyes szoba      | 24823  |
| Kétszemélyes vagy ikerszoba    | 22393  |
| 5 éjszakát maradt              | 20845  |
| Standard kétszemélyes vagy ikerszoba | 17483  |
| Klasszikus kétszemélyes szoba  | 16989  |
| Superior kétszemélyes vagy ikerszoba | 13570 |
| 2 szoba                        | 12393  |

Néhány gyakori címke, mint például `Mobil eszközről küldve`, nem hasznos számunkra, így okos dolog lehet eltávolítani őket, mielőtt megszámolnánk a kifejezések előfordulását, de ez olyan gyors művelet, hogy benne hagyhatod őket, és figyelmen kívül hagyhatod.

### Az éjszakák számát jelző címkék eltávolítása

Ezeknek a címkéknek az eltávolítása az első lépés, amely kissé csökkenti a figyelembe veendő címkék számát. Ne feledd, hogy nem távolítod el őket az adatállományból, csak úgy döntesz, hogy nem veszed figyelembe őket értékként a vélemények adatállományában.

| Tartózkodás hossza | Szám   |
| ------------------ | ------ |
| 1 éjszakát maradt  | 193645 |
| 2 éjszakát maradt  | 133937 |
| 3 éjszakát maradt  | 95821  |
| 4 éjszakát maradt  | 47817  |
| 5 éjszakát maradt  | 20845  |
| 6 éjszakát maradt  | 9776   |
| 7 éjszakát maradt  | 7399   |
| 8 éjszakát maradt  | 2502   |
| 9 éjszakát maradt  | 1293   |
| ...                | ...    |

Számos különféle szoba, lakosztály, stúdió, apartman stb. van. Mindegyik nagyjából ugyanazt jelenti, és nem releváns számodra, így távolítsd el őket a figyelembe vételből.

| Szobatípus                  | Szám   |
| --------------------------- | ------ |
| Kétszemélyes szoba          | 35207  |
| Standard kétszemélyes szoba | 32248  |
| Superior kétszemélyes szoba | 31393  |
| Deluxe kétszemélyes szoba   | 24823  |
| Kétszemélyes vagy ikerszoba | 22393  |
| Standard kétszemélyes vagy ikerszoba | 17483 |
| Klasszikus kétszemélyes szoba | 16989 |
| Superior kétszemélyes vagy ikerszoba | 13570 |

Végül, és ez örömteli (mivel nem igényelt sok feldolgozást), a következő *hasznos* címkék maradnak:

| Címke                                         | Szám   |
| --------------------------------------------- | ------ |
| Szabadidős utazás                             | 417778 |
| Pár                                           | 252294 |
| Egyedül utazó                                 | 108545 |
| Üzleti út                                     | 82939  |
| Csoport (összevonva Barátokkal utazók címkével) | 67535  |
| Fiatal gyermekes család                       | 61015  |
| Idősebb gyermekes család                      | 26349  |
| Háziállattal                                  | 1405   |

Érvelhetsz azzal, hogy a `Barátokkal utazók` nagyjából ugyanaz, mint a `Csoport`, és ez igaz lenne, ha összevonnád őket, ahogy fentebb. A helyes címkék azonosításához szükséges kód megtalálható itt: [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Az utolsó lépés az, hogy új oszlopokat hozz létre ezekhez a címkékhez. Ezután minden véleménysor esetében, ha a `Tag` oszlop megegyezik az egyik új oszloppal, adj hozzá egy 1-est, ha nem, adj hozzá egy 0-t. Az eredmény egy összesítés lesz arról, hogy hány véleményező választotta ezt a szállodát (összesítve) például üzleti vagy szabadidős célra, vagy hogy háziállattal érkezett-e, és ez hasznos információ lehet szálloda ajánlásakor.

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

### Az adatállomány mentése

Végül mentsd el az adatállományt az aktuális állapotában egy új néven.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Érzelemfelismerési műveletek

Ebben az utolsó szakaszban érzelemfelismerést alkalmazol a vélemény oszlopokra, és az eredményeket elmented egy adatállományban.

## Gyakorlat: a szűrt adatok betöltése és mentése

Ne feledd, hogy most a korábban mentett szűrt adatállományt töltöd be, **nem** az eredeti adatállományt.

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

### Stop szavak eltávolítása

Ha érzelemfelismerést futtatnál a negatív és pozitív vélemény oszlopokon, az hosszú időt vehet igénybe. Egy erős teszt laptopon, gyors CPU-val tesztelve 12-14 percet vett igénybe, attól függően, hogy melyik érzelemfelismerési könyvtárat használták. Ez viszonylag hosszú idő, így érdemes megvizsgálni, hogy lehet-e gyorsítani.

A stop szavak, vagyis olyan gyakori angol szavak eltávolítása, amelyek nem változtatják meg egy mondat érzelmi töltetét, az első lépés. Ezek eltávolításával az érzelemfelismerés gyorsabban fut, de nem lesz kevésbé pontos (mivel a stop szavak nem befolyásolják az érzelmi töltetet, de lassítják az elemzést).

A leghosszabb negatív vélemény 395 szóból állt, de a stop szavak eltávolítása után 195 szóból.

A stop szavak eltávolítása szintén gyors művelet, 515,000 sorból álló 2 vélemény oszlopból a stop szavak eltávolítása 3,3 másodpercet vett igénybe a teszt eszközön. Ez kissé több vagy kevesebb időt vehet igénybe nálad, attól függően, hogy milyen gyors a CPU-d, mennyi RAM-od van, van-e SSD-d, és néhány más tényezőtől. A művelet relatív rövidsége azt jelenti, hogy ha javítja az érzelemfelismerés idejét, akkor érdemes elvégezni.

@@CODE_BLOCK_
Az NLTK különböző érzelemelemzőket kínál, amelyekkel tanulhatsz, és helyettesítheted őket, hogy megnézd, az érzelem mennyire pontos vagy kevésbé pontos. Itt a VADER érzelemelemzést használjuk.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, 2014. június.

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

Később, amikor a programodban készen állsz az érzelem kiszámítására, alkalmazhatod azt minden egyes értékelésre az alábbi módon:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Ez körülbelül 120 másodpercet vesz igénybe a számítógépemen, de ez minden gépen eltérő lehet. Ha ki szeretnéd nyomtatni az eredményeket, és megnézni, hogy az érzelem megfelel-e az értékelésnek:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Az utolsó dolog, amit a fájllal meg kell tenni, mielőtt a kihívásban használnád, az az, hogy elmented! Érdemes átrendezni az összes új oszlopot is, hogy könnyebb legyen velük dolgozni (emberi szempontból ez csak egy kozmetikai változtatás).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Az egész kódot futtatnod kell [az elemző jegyzetfüzethez](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (miután futtattad [a szűrő jegyzetfüzetet](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), hogy létrehozd a Hotel_Reviews_Filtered.csv fájlt).

Összefoglalva, a lépések:

1. Az eredeti adatállomány **Hotel_Reviews.csv** a korábbi leckében került feltárásra [az explorer jegyzetfüzettel](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. A Hotel_Reviews.csv fájlt [a szűrő jegyzetfüzet](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) szűri, amelynek eredménye **Hotel_Reviews_Filtered.csv**
3. A Hotel_Reviews_Filtered.csv fájlt [az érzelemelemző jegyzetfüzet](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) dolgozza fel, amelynek eredménye **Hotel_Reviews_NLP.csv**
4. Használd a Hotel_Reviews_NLP.csv fájlt az alábbi NLP kihívásban

### Következtetés

Amikor elkezdted, volt egy adatállományod oszlopokkal és adatokkal, de nem mindegyik volt ellenőrizhető vagy használható. Feltártad az adatokat, kiszűrted, amire nincs szükséged, átalakítottad a címkéket valami hasznosabbá, kiszámítottad a saját átlagokat, hozzáadtál néhány érzelem oszlopot, és remélhetőleg érdekes dolgokat tanultál a természetes szöveg feldolgozásáról.

## [Utólagos kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Kihívás

Most, hogy elemezted az adatállományt érzelem szempontjából, próbáld meg alkalmazni azokat a stratégiákat, amelyeket ebben a tananyagban tanultál (például klaszterezést), hogy mintákat találj az érzelmek körül.

## Áttekintés és önálló tanulás

Vedd fel [ezt a Learn modult](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), hogy többet tanulj, és különböző eszközöket használj az érzelmek feltárására a szövegben.

## Feladat

[Próbálj ki egy másik adatállományt](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Fontos információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.