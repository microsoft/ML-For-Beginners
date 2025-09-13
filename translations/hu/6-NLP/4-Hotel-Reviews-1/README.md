<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T16:54:12+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "hu"
}
-->
# Érzelemfelismerés szállodai véleményekkel - adatok feldolgozása

Ebben a részben az előző leckékben tanult technikákat fogod használni egy nagy adatállomány feltáró elemzéséhez. Miután jól megérted az egyes oszlopok hasznosságát, megtanulod:

- hogyan távolítsd el a felesleges oszlopokat
- hogyan számíts új adatokat a meglévő oszlopok alapján
- hogyan mentsd el az eredményül kapott adatállományt a végső kihívás során történő felhasználásra

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Bevezetés

Eddig megtanultad, hogy a szöveges adatok jelentősen eltérnek a numerikus adatoktól. Ha az adatokat ember írta vagy mondta, elemezhetők minták, gyakoriságok, érzelmek és jelentések szempontjából. Ez a lecke egy valós adatállományt és egy valós kihívást mutat be: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, amely [CC0: Public Domain licenccel](https://creativecommons.org/publicdomain/zero/1.0/) rendelkezik. Az adatokat a Booking.com nyilvános forrásaiból gyűjtötték össze. Az adatállomány készítője Jiashen Liu.

### Felkészülés

Amire szükséged lesz:

* Python 3-at futtató .ipynb notebookok használata
* pandas
* NLTK, [amit helyben telepítened kell](https://www.nltk.org/install.html)
* Az adatállomány, amely elérhető a Kaggle-en: [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Kibontva körülbelül 230 MB. Töltsd le az NLP leckékhez tartozó gyökér `/data` mappába.

## Feltáró adatvizsgálat

Ez a kihívás azt feltételezi, hogy egy szállodai ajánló botot építesz érzelemfelismerés és vendégértékelések alapján. Az adatállomány, amelyet használni fogsz, 1493 különböző szálloda véleményeit tartalmazza 6 városban.

Python, szállodai vélemények adatállománya és az NLTK érzelemfelismerő eszköze segítségével megtudhatod:

* Melyek a leggyakrabban használt szavak és kifejezések a véleményekben?
* A szállodát leíró hivatalos *címkék* összefüggésben vannak-e az értékelési pontszámokkal (pl. több negatív vélemény érkezik-e egy adott szállodára *Fiatal gyerekes család* címkével, mint *Egyedül utazó* címkével, ami esetleg azt jelezheti, hogy az egyedül utazóknak jobban megfelel)?
* Az NLTK érzelemfelismerő pontszámai "egyeznek-e" a szállodai vélemények numerikus pontszámával?

#### Adatállomány

Vizsgáljuk meg az adatállományt, amelyet letöltöttél és helyben elmentettél. Nyisd meg a fájlt egy szerkesztőben, például VS Code-ban vagy akár Excelben.

Az adatállomány fejlécének oszlopai a következők:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Itt csoportosítva vannak, hogy könnyebb legyen áttekinteni őket:  
##### Szállodai oszlopok

* `Hotel_Name`, `Hotel_Address`, `lat` (szélességi fok), `lng` (hosszúsági fok)
  * A *lat* és *lng* segítségével térképet készíthetsz Pythonban, amely megmutatja a szállodák helyét (esetleg színkódolva a negatív és pozitív vélemények alapján)
  * A Hotel_Address nem tűnik különösebben hasznosnak, valószínűleg országra cseréljük, hogy könnyebb legyen rendezni és keresni

**Szállodai meta-vélemény oszlopok**

* `Average_Score`
  * Az adatállomány készítője szerint ez az oszlop a *szálloda átlagos pontszáma, amelyet az elmúlt év legfrissebb véleményei alapján számítottak ki*. Ez szokatlan módja a pontszám kiszámításának, de mivel az adatokat így gyűjtötték, egyelőre elfogadhatjuk.

  ✅ Az adatállomány többi oszlopa alapján tudsz más módot kitalálni az átlagos pontszám kiszámítására?

* `Total_Number_of_Reviews`
  * A szálloda által kapott vélemények teljes száma - nem egyértelmű (kód írása nélkül), hogy ez az adatállományban szereplő véleményekre vonatkozik-e.
* `Additional_Number_of_Scoring`
  * Ez azt jelenti, hogy pontszámot adtak, de a véleményező nem írt pozitív vagy negatív véleményt.

**Vélemény oszlopok**

- `Reviewer_Score`
  - Ez egy numerikus érték, amely legfeljebb 1 tizedesjegyet tartalmaz, és 2.5 és 10 közötti minimum és maximum értékek között mozog.
  - Nem magyarázzák meg, miért 2.5 a legalacsonyabb lehetséges pontszám.
- `Negative_Review`
  - Ha a véleményező nem írt semmit, ez a mező "**No Negative**" értéket kap.
  - Figyelj arra, hogy a véleményező pozitív véleményt is írhat a negatív vélemény mezőbe (pl. "semmi rossz nincs ebben a szállodában").
- `Review_Total_Negative_Word_Counts`
  - Magasabb negatív szószám alacsonyabb pontszámot jelez (az érzelmi töltet ellenőrzése nélkül).
- `Positive_Review`
  - Ha a véleményező nem írt semmit, ez a mező "**No Positive**" értéket kap.
  - Figyelj arra, hogy a véleményező negatív véleményt is írhat a pozitív vélemény mezőbe (pl. "semmi jó nincs ebben a szállodában").
- `Review_Total_Positive_Word_Counts`
  - Magasabb pozitív szószám magasabb pontszámot jelez (az érzelmi töltet ellenőrzése nélkül).
- `Review_Date` és `days_since_review`
  - Frissességi vagy elavultsági mérőszámot lehet alkalmazni a véleményekre (régebbi vélemények nem biztos, hogy olyan pontosak, mint az újabbak, mert a szálloda vezetése megváltozott, felújításokat végeztek, vagy például medencét építettek).
- `Tags`
  - Ezek rövid leírások, amelyeket a véleményező választhat, hogy leírja, milyen típusú vendég volt (pl. egyedül vagy családdal), milyen típusú szobában szállt meg, mennyi ideig tartózkodott, és hogyan nyújtotta be a véleményt.
  - Sajnos ezeknek a címkéknek a használata problémás, lásd az alábbi szakaszt, amely a hasznosságukat tárgyalja.

**Véleményező oszlopok**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Ez egy tényező lehet az ajánlási modellben, például ha meg tudod állapítani, hogy a több száz véleményt író véleményezők inkább negatívak, mint pozitívak. Azonban az adott vélemény véleményezője nem azonosítható egyedi kóddal, és ezért nem kapcsolható össze egy véleményhalmazzal. 30 véleményező van, akik 100 vagy több véleményt írtak, de nehéz látni, hogyan segítheti ez az ajánlási modellt.
- `Reviewer_Nationality`
  - Egyesek azt gondolhatják, hogy bizonyos nemzetiségek hajlamosabbak pozitív vagy negatív véleményt adni nemzeti hajlamuk miatt. Légy óvatos, ha ilyen anekdotikus nézeteket építesz be a modelljeidbe. Ezek nemzeti (és néha faji) sztereotípiák, és minden véleményező egyén volt, aki a saját tapasztalatai alapján írt véleményt. Ez sok szűrőn keresztül történhetett, például korábbi szállodai tartózkodásaik, az utazott távolság, és személyes temperamentumuk alapján. Nehéz igazolni azt a feltételezést, hogy a véleménypontszám oka a nemzetiségük volt.

##### Példák

| Átlagos pontszám | Vélemények száma | Véleményező pontszám | Negatív <br />Vélemény                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pozitív vélemény                 | Címkék                                                                                      |
| ---------------- | ---------------- | -------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8              | 1945             | 2.5                  | Ez jelenleg nem szálloda, hanem építkezési terület. Korán reggel és egész nap elfogadhatatlan építési zajjal terrorizáltak, miközben egy hosszú utazás után pihentem és dolgoztam a szobában. Az emberek egész nap dolgoztak, például légkalapáccsal a szomszédos szobákban. Kértem szobacserét, de nem volt csendes szoba elérhető. Ráadásul túlszámláztak. Este kijelentkeztem, mivel korán kellett indulnom a repülőjáratomhoz, és megfelelő számlát kaptam. Egy nappal később a szálloda további díjat számolt fel a beleegyezésem nélkül, a foglalási ár felett. Ez egy szörnyű hely. Ne büntesd magad azzal, hogy itt foglalsz. | Semmi. Szörnyű hely. Kerüld el. | Üzleti út                                Pár Standard Double Room 2 éjszakát töltött |

Amint láthatod, ez a vendég nem volt elégedett a szállodai tartózkodásával. A szálloda jó átlagos pontszámmal rendelkezik (7.8) és 1945 véleménnyel, de ez a véleményező 2.5 pontot adott, és 115 szót írt arról, mennyire negatív volt az ott tartózkodása. Ha semmit sem írt volna a Pozitív vélemény oszlopba, feltételezhetnéd, hogy semmi pozitív nem volt, de mégis írt 7 figyelmeztető szót. Ha csak a szavak számát néznénk, a szavak jelentése vagy érzelmi töltete helyett, torz képet kaphatnánk a véleményező szándékáról. Furcsa módon a 2.5 pontszám zavaró, mert ha a szállodai tartózkodás ennyire rossz volt, miért adott egyáltalán pontot? Az adatállomány alapos vizsgálata során láthatod, hogy a legalacsonyabb lehetséges pontszám 2.5, nem 0. A legmagasabb lehetséges pontszám 10.

##### Címkék

Ahogy fentebb említettük, első pillantásra a `Tags` oszlop használata az adatok kategorizálására logikusnak tűnik. Sajnos ezek a címkék nem szabványosítottak, ami azt jelenti, hogy egy adott szállodában az opciók lehetnek *Single room*, *Twin room* és *Double room*, míg egy másik szállodában *Deluxe Single Room*, *Classic Queen Room* és *Executive King Room*. Ezek lehetnek ugyanazok, de annyi variáció van, hogy a választás a következő:

1. Minden kifejezést egyetlen szabványra próbálunk átalakítani, ami nagyon nehéz, mert nem világos, hogy mi lenne az átalakítási útvonal minden esetben (pl. *Classic single room* átalakítása *Single room*-ra, de *Superior Queen Room with Courtyard Garden or City View* sokkal nehezebb).

1. NLP megközelítést alkalmazunk, és mérjük bizonyos kifejezések, mint például *Solo*, *Business Traveller* vagy *Family with young kids* gyakoriságát, ahogy azok az egyes szállodákra vonatkoznak, és ezt beépítjük az ajánlásba.

A címkék általában (de nem mindig) egyetlen mezőt tartalmaznak, amely 5-6 vesszővel elválasztott értéket sorol fel, amelyek a *Utazás típusa*, *Vendégek típusa*, *Szoba típusa*, *Éjszakák száma* és *Eszköz típusa, amelyen a véleményt benyújtották* kategóriákhoz igazodnak. Azonban mivel néhány véleményező nem tölti ki az összes mezőt (egy mezőt üresen hagyhat), az értékek nem mindig ugyanabban a sorrendben vannak.

Például vegyük a *Csoport típusa* kategóriát. Ebben az oszlopban a `Tags` mezőben 1025 egyedi lehetőség van, és sajnos csak néhányuk utal csoportokra (néhány a szoba típusára stb.). Ha csak azokat szűröd, amelyek családot említenek, az eredmények sok *Family room* típusú eredményt tartalmaznak. Ha hozzáadod a *with* kifejezést, azaz számolod a *Family with* értékeket, az eredmények jobbak, több mint 80,000 a 515,000 eredményből tartalmazza a "Family with young children" vagy "Family with older children" kifejezést.

Ez azt jelenti, hogy a címkék oszlop nem teljesen haszontalan számunkra, de némi munkát igényel, hogy hasznossá váljon.

##### Szállodai átlagos pontszám

Az adatállományban számos furcsaság vagy eltérés van, amelyeket nem tudok megfejteni, de itt bemutatom őket, hogy tisztában legyél velük, amikor a modelljeidet építed. Ha megfejted, kérlek, oszd meg velünk a vitafórumon!

Az adatállomány az alábbi oszlopokat tartalmazza az átlagos pontszám és a vélemények száma kapcsán:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Az adatállományban a legtöbb véleménnyel rendelkező szálloda a *Britannia International Hotel Canary Wharf*, amely 4789 véleményt tartalmaz az 515,000-ből. De ha megnézzük a `Total_Number_of_Reviews` értéket ennél a szállodánál, az 9086. Feltételezheted, hogy sokkal több pontszám van vélemények nélkül, így talán hozzá kellene adni az `Additional_Number_of_Scoring` oszlop értékét. Az érték 2682, és ha hozzáadjuk a 4789-hez, 7471-et kapunk, ami még mindig 1615-tel kevesebb, mint a `Total_Number_of_Reviews`.

Ha az `Average_Score` oszlopot nézed, feltételezheted, hogy az az adatállományban szereplő vélemények átlaga, de a Kaggle leírása szerint "*A szálloda átlagos pontszáma, amelyet az elmúlt év legfrissebb véleményei alapján számítottak ki*". Ez nem tűnik túl hasznosnak, de kiszámíthatjuk a saját átlagunkat az adatállományban szereplő véleménypontszámok alapján. Ugyanazt a szállodát példaként használva, az átlagos szállodai pontszám 7.1, de az adatállományban szereplő véleményező pontszámok átlaga 6.8. Ez közel van, de nem ugyanaz az érték,
> 🚨 Egy figyelmeztetés

> Amikor ezzel az adathalmazzal dolgozol, olyan kódot fogsz írni, amely kiszámít valamit a szövegből anélkül, hogy magát a szöveget el kellene olvasnod vagy elemezned. Ez az NLP lényege: jelentés vagy érzelem értelmezése anélkül, hogy emberi beavatkozásra lenne szükség. Azonban előfordulhat, hogy elolvasol néhány negatív értékelést. Arra biztatlak, hogy ne tedd, mert nincs rá szükség. Néhányuk nevetséges vagy irreleváns negatív hotelértékelés, például: "Nem volt jó az időjárás", ami a hotel, vagy bárki más számára nem befolyásolható tényező. De van egy sötét oldala is néhány értékelésnek. Néha a negatív értékelések rasszista, szexista vagy életkorral kapcsolatos előítéleteket tartalmaznak. Ez sajnálatos, de várható egy nyilvános weboldalról lekapart adathalmaz esetében. Néhány értékelő olyan véleményeket hagy, amelyeket ízléstelennek, kényelmetlennek vagy felkavarónak találhatsz. Jobb, ha a kód méri az érzelmeket, mintha magad olvasnád el őket és felzaklatnád magad. Ennek ellenére csak kisebbség ír ilyen dolgokat, de mégis léteznek.
## Feladat - Adatfeltárás
### Adatok betöltése

Elég volt az adatok vizuális vizsgálatából, most írj néhány kódot, hogy válaszokat kapj! Ebben a részben a pandas könyvtárat fogjuk használni. Az első feladatod az, hogy megbizonyosodj arról, hogy be tudod tölteni és olvasni a CSV adatokat. A pandas könyvtár gyors CSV betöltőt kínál, és az eredményt egy dataframe-be helyezi, ahogy azt korábbi leckékben láttuk. A betöltendő CSV több mint félmillió sort tartalmaz, de csak 17 oszlopot. A pandas számos hatékony módot kínál a dataframe-ekkel való interakcióra, beleértve a műveletek végrehajtását minden soron.

Ettől a ponttól kezdve a leckében kódrészletek és magyarázatok lesznek a kódról, valamint némi vita arról, hogy mit jelentenek az eredmények. Használd a mellékelt _notebook.ipynb_-t a kódodhoz.

Kezdjük azzal, hogy betöltjük az adatfájlt, amelyet használni fogsz:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Most, hogy az adatok betöltve vannak, végezhetünk rajtuk néhány műveletet. Tartsd ezt a kódot a programod tetején a következő részhez.

## Az adatok feltárása

Ebben az esetben az adatok már *tiszták*, ami azt jelenti, hogy készen állnak a feldolgozásra, és nem tartalmaznak más nyelveken írt karaktereket, amelyek problémát okozhatnak az algoritmusoknak, amelyek csak angol karaktereket várnak.

✅ Lehet, hogy olyan adatokkal kell dolgoznod, amelyek kezdeti feldolgozást igényelnek, mielőtt NLP technikákat alkalmaznál, de most nem. Ha mégis, hogyan kezelnéd a nem angol karaktereket?

Szánj egy pillanatot arra, hogy megbizonyosodj arról, hogy az adatok betöltése után kóddal tudod feltárni őket. Könnyű lenne a `Negative_Review` és `Positive_Review` oszlopokra koncentrálni. Ezek természetes szövegekkel vannak tele, amelyeket az NLP algoritmusok feldolgozhatnak. De várj! Mielőtt belevágnál az NLP-be és az érzelemfelismerésbe, kövesd az alábbi kódot, hogy megbizonyosodj arról, hogy az adatkészletben megadott értékek megfelelnek a pandas segítségével számított értékeknek.

## Dataframe műveletek

Az első feladat ebben a leckében az, hogy ellenőrizd, helyesek-e az alábbi állítások, azáltal, hogy írsz néhány kódot, amely megvizsgálja a dataframe-et (anélkül, hogy megváltoztatnád).

> Mint sok programozási feladatnál, itt is többféle módon lehet megoldani, de jó tanács, hogy a legegyszerűbb, legkönnyebb módon csináld, különösen, ha később könnyebb lesz megérteni, amikor visszatérsz ehhez a kódhoz. A dataframe-ekkel egy átfogó API áll rendelkezésre, amely gyakran hatékony módot kínál arra, hogy elvégezd, amit szeretnél.

Tekintsd az alábbi kérdéseket kódolási feladatoknak, és próbáld meg megválaszolni őket anélkül, hogy megnéznéd a megoldást.

1. Írd ki a dataframe *alakját* (shape), amelyet éppen betöltöttél (az alak a sorok és oszlopok száma).
2. Számítsd ki az értékek gyakoriságát a reviewer nemzetiségek esetében:
   1. Hány különböző érték van a `Reviewer_Nationality` oszlopban, és mik ezek?
   2. Melyik reviewer nemzetiség a leggyakoribb az adatkészletben (ország és értékelések száma)?
   3. Melyek a következő 10 leggyakrabban előforduló nemzetiségek, és azok gyakorisága?
3. Melyik volt a leggyakrabban értékelt hotel a 10 leggyakoribb reviewer nemzetiség esetében?
4. Hány értékelés van hotelenként (hotel gyakoriság az adatkészletben)?
5. Bár van egy `Average_Score` oszlop minden hotel esetében az adatkészletben, kiszámíthatod az átlagos pontszámot is (az összes reviewer pontszámának átlaga az adatkészletben hotelenként). Adj hozzá egy új oszlopot a dataframe-hez `Calc_Average_Score` oszlopfejléccel, amely tartalmazza a kiszámított átlagot.
6. Van-e olyan hotel, amelynek ugyanaz az (1 tizedesjegyre kerekített) `Average_Score` és `Calc_Average_Score` értéke?
   1. Próbálj meg írni egy Python függvényt, amely egy Series-t (sor) vesz argumentumként, és összehasonlítja az értékeket, üzenetet nyomtatva, ha az értékek nem egyenlőek. Ezután használd a `.apply()` metódust, hogy minden sort feldolgozz a függvénnyel.
7. Számítsd ki és írd ki, hány sorban van a `Negative_Review` oszlop értéke "No Negative".
8. Számítsd ki és írd ki, hány sorban van a `Positive_Review` oszlop értéke "No Positive".
9. Számítsd ki és írd ki, hány sorban van a `Positive_Review` oszlop értéke "No Positive" **és** a `Negative_Review` oszlop értéke "No Negative".

### Kódválaszok

1. Írd ki a dataframe *alakját* (shape), amelyet éppen betöltöttél (az alak a sorok és oszlopok száma).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Számítsd ki az értékek gyakoriságát a reviewer nemzetiségek esetében:

   1. Hány különböző érték van a `Reviewer_Nationality` oszlopban, és mik ezek?
   2. Melyik reviewer nemzetiség a leggyakoribb az adatkészletben (ország és értékelések száma)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Melyek a következő 10 leggyakrabban előforduló nemzetiségek, és azok gyakorisága?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Melyik volt a leggyakrabban értékelt hotel a 10 leggyakoribb reviewer nemzetiség esetében?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Hány értékelés van hotelenként (hotel gyakoriság az adatkészletben)?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Név                  | Összes_Értékelés_Száma | Talált_Értékelések_Száma |
   | :----------------------------------------: | :---------------------: | :----------------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789             |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169             |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578             |
   |                    ...                     |           ...           |         ...             |
   |       Mercure Paris Porte d Orleans        |           110           |         10              |
   |                Hotel Wagner                |           135           |         10              |
   |            Hotel Gallitzinberg             |           173           |          8              |

   Észreveheted, hogy az *adathalmazban számolt* eredmények nem egyeznek a `Total_Number_of_Reviews` értékével. Nem világos, hogy ez az érték az adathalmazban a hotel összes értékelését képviselte-e, de nem mindet kaparták le, vagy valamilyen más számítást. A `Total_Number_of_Reviews` nem kerül felhasználásra a modellben, mivel nem egyértelmű.

5. Bár van egy `Average_Score` oszlop minden hotel esetében az adatkészletben, kiszámíthatod az átlagos pontszámot is (az összes reviewer pontszámának átlaga az adatkészletben hotelenként). Adj hozzá egy új oszlopot a dataframe-hez `Calc_Average_Score` oszlopfejléccel, amely tartalmazza a kiszámított átlagot. Írd ki a `Hotel_Név`, `Average_Score` és `Calc_Average_Score` oszlopokat.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Érdekelhet, hogy miért különbözik néha az `Average_Score` érték a kiszámított átlagos pontszámtól. Mivel nem tudhatjuk, miért egyeznek néhány értékek, de mások eltérnek, ebben az esetben a legbiztonságosabb, ha a rendelkezésre álló értékelési pontszámokat használjuk az átlag kiszámításához. Az eltérések általában nagyon kicsik, itt vannak a legnagyobb eltéréssel rendelkező hotelek:

   | Átlagos_Pontszám_Eltérés | Átlagos_Pontszám | Számított_Átlagos_Pontszám |                                  Hotel_Név |
   | :----------------------: | :--------------: | :------------------------: | ------------------------------------------: |
   |           -0.8           |      7.7         |        8.5                |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8         |        9.5                | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5         |        8.2                |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9         |        8.6                |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0         |        7.5                |                         Hotel Royal Elys es |
   |           ...            |      ...         |        ...                |                                         ... |
   |           0.7            |      7.5         |        6.8                |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1         |        6.3                |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8         |        5.9                |                               Villa Eugenie |
   |           0.9            |      8.6         |        7.7                |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2         |        5.9                |                          Kube Hotel Ice Bar |

   Mivel csak 1 hotel esetében van 1-nél nagyobb eltérés, valószínűleg figyelmen kívül hagyhatjuk az eltérést, és használhatjuk a számított átlagos pontszámot.

6. Számítsd ki és írd ki, hány sorban van a `Negative_Review` oszlop értéke "No Negative".

7. Számítsd ki és írd ki, hány sorban van a `Positive_Review` oszlop értéke "No Positive".

8. Számítsd ki és írd ki, hány sorban van a `Positive_Review` oszlop értéke "No Positive" **és** a `Negative_Review` oszlop értéke "No Negative".

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Egy másik módszer

Egy másik mód az elemek számolására lambdák nélkül, és a sorok számolására a sum használatával:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Észrevehetted, hogy 127 sor van, amelyben mind a `Negative_Review` oszlop értéke "No Negative", mind a `Positive_Review` oszlop értéke "No Positive". Ez azt jelenti, hogy az értékelő adott egy numerikus pontszámot a hotelnek, de nem írt sem pozitív, sem negatív értékelést. Szerencsére ez csak kis mennyiségű sor (127 a 515738-ból, vagyis 0,02%), így valószínűleg nem torzítja a modellünket vagy az eredményeinket semmilyen irányba, de lehet, hogy nem számítottál arra, hogy egy értékeléseket tartalmazó adathalmazban lesznek sorok értékelések nélkül, ezért érdemes feltárni az adatokat, hogy felfedezzük az ilyen sorokat.

Most, hogy feltártad az adatkészletet, a következő leckében szűrni fogod az adatokat, és hozzáadsz némi érzelemfelismerést.

---
## 🚀Kihívás

Ez a lecke bemutatja, ahogy azt korábbi leckékben láttuk, hogy mennyire kritikus fontosságú az adatok és azok sajátosságainak megértése, mielőtt műveleteket végeznénk rajtuk. Különösen a szövegalapú adatok alapos vizsgálatot igényelnek. Ásd bele magad különböző szövegközpontú adathalmazokba, és nézd meg, felfedezhetsz-e olyan területeket, amelyek torzítást vagy ferde érzelmeket vihetnek be egy modellbe.

## [Utó-leckekvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Vedd fel [ezt az NLP tanulási útvonalat](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), hogy felfedezd azokat az eszközöket, amelyeket beszéd- és szövegközpontú modellek építésekor kipróbálhatsz.

## Feladat 

[NLTK](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.