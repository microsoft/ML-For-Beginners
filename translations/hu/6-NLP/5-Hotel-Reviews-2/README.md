# Érzelemelemzés szállodai értékelésekkel

Miután részletesen megvizsgáltad az adatkészletet, itt az ideje, hogy kiszűrd a megfelelő oszlopokat, majd NLP technikákat alkalmazz az adatkészleten, hogy új betekintést nyerhess a szállodákról.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Szűrés és érzelemelemzési műveletek

Valószínűleg észrevetted, hogy az adatkészletnek vannak problémái. Egyes oszlopok haszontalan információkat tartalmaznak, mások hibásnak tűnnek. Ha helyesek, akkor sem világos, hogyan számították ki őket, és az eredményeket nem lehet saját számításokkal függetlenül ellenőrizni.

## Gyakorlat: egy kicsit több adatfeldolgozás

Tiszítsd meg az adatokat még egy kicsit. Adj hozzá hasznos oszlopokat, módosítsd más oszlopok értékeit, és bizonyos oszlopokat teljesen távolíts el.

1. Kezdeti oszlopfeldolgozás

   1. Távolítsd el a `lat` és `lng` oszlopokat

   2. Cseréld ki a `Hotel_Address` értékeket az alábbiakra (ha a cím tartalmazza a város és az ország nevét is, csak a várost és az országot hagyd meg).

      Ezek a városok és országok találhatók az adatkészletben:

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
      
      # Cserélje le az összes címet egy rövidebb, hasznosabb formára
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # A value_counts() összege meg kell, hogy egyezzen a teljes értékelésszámmal
      print(df["Hotel_Address"].value_counts())
      ```

      Most már lekérdezheted az adatokat ország szinten is:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Cím              | Szálloda_Neve |
      | :--------------------- | :--------:   |
      | Amszterdam, Hollandia  |    105       |
      | Barcelona, Spanyolország |    211     |
      | London, Egyesült Királyság |    400     |
      | Milánó, Olaszország    |    162       |
      | Párizs, Franciaország  |    458       |
      | Bécs, Ausztria         |    158       |

2. Szállodai meta-értékelési oszlopok feldolgozása

  1. Távolítsd el az `Additional_Number_of_Scoring` oszlopot

  1. Cseréld le a `Total_Number_of_Reviews` értékét a valós, az adatkészletben szereplő értékre

  1. Cseréld le az `Average_Score` értékét a saját, kiszámított pontra

  ```python
  # Töröld az `Additional_Number_of_Scoring` értéket
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Cseréld le a `Total_Number_of_Reviews` és az `Average_Score` értékeket a saját számított értékeinkre
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Értékelési oszlopok feldolgozása

   1. Távolítsd el a `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` és `days_since_review` oszlopokat

   2. Hagyd meg változatlanul a `Reviewer_Score`, `Negative_Review` és `Positive_Review` oszlopokat,
     
   3. Hagyd meg most a `Tags` oszlopot is

     - A következő szakaszban további szűrőműveleteket végzünk a címkéken, majd a címkék eltávolításra kerülnek

4. Értékelői oszlopok feldolgozása

  1. Távolítsd el a `Total_Number_of_Reviews_Reviewer_Has_Given` oszlopot
  
  2. Hagyd meg a `Reviewer_Nationality` oszlopot

### Címke oszlopok

A `Tag` oszlop problémás, mivel egy felsorolás (szöveges formában) van benne tárolva. Sajnos az alcímek száma és sorrendje nem mindig egyezik meg. Nehéz az embernek azonosítani a fontos kifejezéseket, mert 515 000 sor, és 1427 szálloda van, mindegyik egy kicsit eltérő lehetőségekkel, amiből a véleményező választhatott. Itt jön jól az NLP. Átvizsgálhatod a szöveget, megtalálhatod a leggyakoribb kifejezéseket, és megszámolhatod őket.

Sajnos nem egyszavas kifejezések érdekelnek, hanem több szóból álló kifejezések (pl. *Üzleti út*). Egy több szavas gyakorisági eloszlás végrehajtása ekkora adatmennyiségen (6 762 646 szó) rendkívül hosszú időt venne igénybe, de az adatok ismerete nélkül ez szükségesnek tűnik. Itt jön jól a felderítő adat elemzés, mert láttál már címkemintát, mint például `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, így elkezdheted keresni a módját, hogyan csökkentsd jelentősen a feldolgozandó adatot. Szerencsére lehet – de először néhány lépést kell követni a fontos címkék meghatározásához.

### Címkék szűrése

Ne feledd, hogy az adatkészlet célja érzelem hozzáadása és olyan oszlopok létrehozása, amelyek segítenek a legjobb szálloda kiválasztásában (saját magad vagy egy ügyfél számára, aki szállodai ajánló bot fejlesztését kéri). El kell döntened, hogy a címkék hasznosak-e vagy sem a végső adatkészletben. Íme egy értelmezés (más célra más címkék maradhatnak bent vagy kerülhetnek ki):

1. Az utazás típusa releváns, ennek meg kell maradnia
2. A vendégcsoport típusa fontos, ennek meg kell maradnia
3. A vendég által használt szoba, lakosztály vagy stúdió típusa nem releváns (minden szállodában lényegében ugyanazok a szobák vannak)
4. Az eszköz, amin az értékelést benyújtották, nem releváns
5. Az eltöltött éjszakák száma *lehet* releváns, ha azt feltételezed, hogy a hosszabb tartózkodás a szálloda jobban tetszésével jár, de ez megkérdőjelezhető és valószínűleg nem számít

Összefoglalva, **2 fajta címkét tarts meg, a többit távolítsd el**.

Először nem akarod megszámolni a címkéket, amíg nincsenek megfelelő formátumban, ezért el kell távolítani a szögletes zárójeleket és a idézőjeleket. Többféleképpen lehet ezt megtenni, de a leggyorsabbat érdemes választani, mert sok adat feldolgozása hosszú időt vehet igénybe. Szerencsére a pandas egyszerű eszközöket kínál ezekhez a lépésekhez.

```Python
# Távolítsa el a nyitó és záró zárójeleket
df.Tags = df.Tags.str.strip("[']")
# távolítsa el az összes idézőjelet is
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Minden címke olyanná válik, mint: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`.

Ezután egy problémát találunk. Néhány értékelés vagy sor 5 oszlopból áll, néhány 3-ból, mások 6-ból. Ez az adatkészlet létrehozásának módjából ered, és nehéz orvosolni. Meg akarod számolni az egyes kifejezéseket, de azok különböző sorrendben vannak az értékelésekben, így a megszámlálás hibázhat, és lehet, hogy egy szálloda nem kap megérdemelt címkét.

Ehelyett a sorrendet a javunkra fordítjuk, mert mindegyik címke több szóból áll, de vesszővel van elválasztva! A legegyszerűbb megoldás 6 ideiglenes oszlop létrehozása, mindegyikbe a címke a maga sorrendjében kerül. Ezeket azután egy nagy oszlopba egyesítjük, és ráfuttatjuk a `value_counts()` metódust. Az eredmény 2428 egyedi címke volt. Íme egy kis minta:

| Címke                         | Darabszám |
| ------------------------------ | --------- |
| Szabadidős utazás             | 417778    |
| Mobil eszközről benyújtva    | 307640    |
| Pár                           | 252294    |
| 1 éjszakát eltöltött          | 193645    |
| 2 éjszakát eltöltött          | 133937    |
| Egyedül utazó                 | 108545    |
| 3 éjszakát eltöltött          | 95821     |
| Üzleti út                    | 82939     |
| Csoport                      | 65392     |
| Fiatal gyerekes család       | 61015     |
| 4 éjszakát eltöltött          | 47817     |
| Kétszemélyes szoba           | 35207     |
| Standard kétszemélyes szoba  | 32248     |
| Superior kétszemélyes szoba  | 31393     |
| Nagyobb gyerekes család      | 26349     |
| Deluxe kétszemélyes szoba    | 24823     |
| Kétszemélyes vagy iker szoba| 22393     |
| 5 éjszakát eltöltött          | 20845     |
| Standard kétszemélyes vagy iker szoba | 17483     |
| Klasszikus kétszemélyes szoba | 16989     |
| Superior kétszemélyes vagy iker szoba | 13570     |
| 2 szoba                     | 12393     |

Néhány gyakori címke, mint a `Mobil eszközről benyújtva` haszontalan számunkra, ezért okos dolog lehet eltávolítani őket a címke előfordulás megszámlálása előtt, de a művelet olyan gyors, hogy bent is hagyhatod és figyelmen kívül hagyhatod.

### Az eltöltött éjszakák címkék eltávolítása

Ezek eltávolítása az első lépés, enyhén csökkenti a figyelembe vett címkék számát. Megjegyzendő, hogy nem az adatkészletből távolítod el őket, csak a számlálás/megőrzés céljára nem veszed figyelembe.

| Tartózkodás hossza | Darabszám |
| ------------------ | --------- |
| 1 éjszakát eltöltött | 193645    |
| 2 éjszakát eltöltött | 133937    |
| 3 éjszakát eltöltött | 95821     |
| 4 éjszakát eltöltött | 47817     |
| 5 éjszakát eltöltött | 20845     |
| 6 éjszakát eltöltött | 9776      |
| 7 éjszakát eltöltött | 7399      |
| 8 éjszakát eltöltött | 2502      |
| 9 éjszakát eltöltött | 1293      |
| ...                | ...       |

Rengeteg különböző szoba, lakosztály, stúdió, apartman stb. van. Ezek nagyjából ugyanazt jelentik és nem relevánsak számodra, ezért távolítsd el őket a figyelembevételből.

| Szoba típusa                     | Darabszám |
| ------------------------------- | --------- |
| Kétszemélyes szoba              | 35207     |
| Standard kétszemélyes szoba     | 32248     |
| Superior kétszemélyes szoba     | 31393     |
| Deluxe kétszemélyes szoba       | 24823     |
| Kétszemélyes vagy iker szoba   | 22393     |
| Standard kétszemélyes vagy iker szoba | 17483     |
| Klasszikus kétszemélyes szoba   | 16989     |
| Superior kétszemélyes vagy iker szoba | 13570     |

Végül, és ez örömteli (mert alig volt feldolgozás), a következő *hasznos* címke marad meg:

| Címke                                         | Darabszám |
| --------------------------------------------- | --------- |
| Szabadidős utazás                            | 417778    |
| Pár                                           | 252294    |
| Egyedül utazó                                | 108545    |
| Üzleti út                                    | 82939     |
| Csoport (egybevonva a baráti utazókkal)     | 67535     |
| Fiatal gyerekes család                        | 61015     |
| Nagyobb gyerekes család                       | 26349     |
| Háziállattal                                  | 1405      |

Az is vitatható, hogy a `Travellers with friends` többé-kevésbé ugyanaz, mint a `Group`, és jogos őket összevonni a fentiek szerint. A helyes címkék azonosításának kódja a [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Az utolsó lépés, hogy új oszlopokat hozz létre ezeknek a címkéknek. Minden értékelői sorban, ha a `Tag` oszlop megegyezik az új oszlop egyikével, akkor 1-et adj hozzá, ha nem, akkor 0-t. Az eredményként megkapod, hogy összesen hány értékelő választotta adott szállodát, például üzleti útra vagy szabadidős utazásra, vagy háziállat hozatalára, és ez hasznos információ a szállodai ajánláshoz.

```python
# A címkék feldolgozása új oszlopokká
# A Hotel_Reviews_Tags.py fájl azonosítja a legfontosabb címkéket
# Szabadidős utazás, Pár, Egyéni utazó, Üzleti út, Csoport kombinálva Baráti utazókkal,
# Család kisgyermekkel, Család nagyobb gyerekkel, Háziállattal
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Mentés

Végül mentsd az adatkészletet az aktuális állapotában egy új néven.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Új adatfájl mentése kiszámított oszlopokkal
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Érzelemelemzés műveletek

Ebben a végső szakaszban végezd el az érzelemelemzést az értékelői oszlopokon, és mentsd az eredményt egy új adatkészletbe.

## Gyakorlat: a szűrt adatok betöltése és mentése

Jegyezd meg, hogy most a szűrt adatkészletet töltöd be, amelyet a korábbi szakaszban mentettél, **nem** az eredetit.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Töltse be a szűrt szállodai véleményeket CSV-ből
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# Az Ön kódja ide kerül hozzáadásra


# Végül ne felejtse el elmenteni a szállodai véleményeket az új NLP adatokkal együtt
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Stop szavak eltávolítása

Ha az érzelemelemzést futtatnád a negatív és pozitív értékelői oszlopokon, az hosszú időt venne igénybe. Egy erős tesztlaptopon, gyors CPU-val 12-14 percig tartott, attól függően mely érzelemelemző könyvtárat használták. Ez viszonylag hosszú idő, ezért érdemes megvizsgálni a gyorsítás lehetőségét.

A stop szavak, azaz gyakori angol szavak, amelyek nem befolyásolják a mondat érzelmi töltetét, eltávolítása az első lépés. Eltávolításukkal gyorsabb lehet az érzelemelemzés, de nem lesz kevésbé pontos (mivel a stop szavak nem befolyásolják az érzelmet, csak lassítják az elemzést).

A leghosszabb negatív értékelés 395 szóból állt, de a stop szavak eltávolítása után már csak 195 szóból.

A stop szavak eltávolítása szintén gyors művelet volt: két értékelési oszlopból több mint 515 000 sorból 3,3 másodperc alatt megtörtént teszt környezetben. A tényleges idő kissé eltérhet a készülék CPU sebességétől, RAM-tól, SSD meglététől és egyéb tényezőktől függően. Az elég rövid végrehajtás miatt, ha ezzel csökkenthető az érzelemelemzés ideje, akkor megéri megtenni.

```python
from nltk.corpus import stopwords

# Töltse be a szállodai értékeléseket CSV-ből
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Távolítsa el a stop szavakat - sok szöveg esetén lassú lehet!
# Ryan Han (ryanxjhan a Kaggle-en) nagyszerű bejegyzést írt a különböző stop szavak eltávolítási módszerek teljesítményének méréséről
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # Ryan által ajánlott módszer használata
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Távolítsa el a stop szavakat mindkét oszlopból
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Érzelemelemzés végrehajtása

Most kiszámolod az érzelemelemzést mind a negatív, mind a pozitív értékelésekhez, és az eredményt két új oszlopban tárolod. Az érzelem pontosságát az fogja mutatni, hogy összehasonlítod a véleményező pontszámával ugyanarra az értékelésre. Például, ha a negatív értékelés érzelmi töltete 1 (rendkívül pozitív), és a pozitív értékelés is 1, de a véleményező a legalacsonyabb pontszámot adta a szállodának, akkor vagy az értékelő szövege nem felel meg a pontszámnak, vagy az érzelemelemző nem ismerte fel helyesen az érzelmet. Várható, hogy az érzelem pontszámok néha teljesen hibásak lesznek, és ezt gyakran meg lehet magyarázni, pl. az értékelés lehet nagyon szarkasztikus: "Persze, nagyon élveztem, hogy fűtés nélküli szobában aludtam", és az érzelemelemző pozitívnak gondolja, míg egy ember tudja, hogy szarkazmus.

Az NLTK többféle érzelemelemzőt is kínál, amiket kipróbálhatsz, hogy lássad, melyik mennyire pontos. Itt a VADER érzelemelemzést használjuk.


> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Egy takarékos, szabályalapú modell a közösségi média szövegeinek érzelemelemzésére. Nyolcadik Nemzetközi Konferencia a Webblogokról és a Közösségi Médiumokról (ICWSM-14). Ann Arbor, MI, 2014. június.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Hozd létre a vader érzelemelemzőt (az NLTK-ban vannak más lehetőségek is, amelyeket kipróbálhatsz)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: Egy takarékos, szabályalapú modell a közösségi média szövegének érzelemelemzéséhez. Nyolcadik Nemzetközi Weblogok és Közösségi Média Konferencia (ICWSM-14). Ann Arbor, MI, 2014 június.

# Egy véleményhez 3 bemeneti lehetőség van:
# Lehet "Nincs negatív", ekkor térj vissza 0-val
# Lehet "Nincs pozitív", ekkor térj vissza 0-val
# Lehet egy vélemény, ekkor számold ki az érzelmet
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Később a programban, amikor készen állsz az érzelem kiszámítására, ezt alkalmazhatod minden értékelésre a következő módon:

```python
# Adj hozzá egy negatív érzelmi töltetű és egy pozitív érzelmi töltetű oszlopot
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Ez körülbelül 120 másodpercet vesz igénybe a számítógépemen, de gépenként eltérő lehet. Ha ki akarod nyomtatni az eredményeket és megnézni, hogy az érzelem egyezik-e az értékeléssel:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

A legutolsó teendő a fájllal a kihívás előtt, hogy elmentsd! Érdemes megfontolnod az új oszlopok átrendezését is, hogy könnyebben kezelhetők legyenek (egy ember számára ez csak esztétikai változtatás).

```python
# Átrendezni az oszlopokat (Ez csak kozmetikai, de megkönnyíti a későbbi adatfelfedezést)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Futtasd le az egész kódot a [elemzési jegyzetfüzethez](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (miután lefuttattad a [szűrő jegyzetfüzetet](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) a Hotel_Reviews_Filtered.csv fájl előállításához).

Ismétlésképpen, a lépések a következők:

1. Az eredeti adathalmaz fájl, a **Hotel_Reviews.csv** az előző leckében került feltérképezésre a [feltérképező jegyzetfüzet](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) segítségével
2. A Hotel_Reviews.csv a [szűrő jegyzetfüzet](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) által szűrésre kerül, ami **Hotel_Reviews_Filtered.csv** fájlt eredményez
3. A Hotel_Reviews_Filtered.csv a [érzelemelemző jegyzetfüzet](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) által feldolgozásra kerül, amely **Hotel_Reviews_NLP.csv** fájlt eredményez
4. Használd a Hotel_Reviews_NLP.csv-t az alábbi NLP kihívásban

### Következtetés

Amikor elkezdted, volt egy adathalmazod oszlopokkal és adatokkal, de nem mindegyiket lehetett ellenőrizni vagy használni. Feltérképezted az adatokat, kiszűrted, amire nincs szükséged, címkéket hasznos információvá alakítottad, kiszámoltad a saját átlagaidat, hozzáadtál néhány érzelem oszlopot, és remélhetőleg érdekes dolgokat tanultál a természetes szöveg feldolgozásáról.

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Kihívás

Most, hogy az adathalmazod elemzésre került érzelem szempontjából, nézd meg, hogy a tanult stratégiákat (pl. klaszterezés) alkalmazva meg tudod-e találni az érzelem körüli mintázatokat.

## Áttekintés és önálló tanulás

Vedd [ezt a Learn modult](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) hogy többet tanulj és különböző eszközöket használj a szöveg érzelemének feltérképezésére.
## Feladat 

[Próbálj ki egy másik adathalmazt](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Fontos információk esetén professzionális emberi fordítást javasolunk. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely ebből a fordításból ered.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->