<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T16:55:34+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "sk"
}
-->
# Analýza sentimentu pomocou recenzií hotelov - spracovanie údajov

V tejto časti použijete techniky z predchádzajúcich lekcií na prieskumnú analýzu veľkého datasetu. Keď získate dobré pochopenie užitočnosti jednotlivých stĺpcov, naučíte sa:

- ako odstrániť nepotrebné stĺpce
- ako vypočítať nové údaje na základe existujúcich stĺpcov
- ako uložiť výsledný dataset na použitie vo finálnej výzve

## [Kvíz pred lekciou](https://ff-quizzes.netlify.app/en/ml/)

### Úvod

Doteraz ste sa naučili, že textové údaje sú dosť odlišné od číselných typov údajov. Ak ide o text, ktorý napísal alebo povedal človek, je možné ho analyzovať na hľadanie vzorcov, frekvencií, sentimentu a významu. Táto lekcia vás zavedie do reálneho datasetu s reálnou výzvou: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, ktorý obsahuje [CC0: Public Domain licenciu](https://creativecommons.org/publicdomain/zero/1.0/). Dataset bol získaný z verejných zdrojov na Booking.com. Autorom datasetu je Jiashen Liu.

### Príprava

Budete potrebovať:

* Schopnosť spúšťať .ipynb notebooky pomocou Pythonu 3
* pandas
* NLTK, [ktorý by ste si mali nainštalovať lokálne](https://www.nltk.org/install.html)
* Dataset, ktorý je dostupný na Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Po rozbalení má približne 230 MB. Stiahnite ho do koreňového priečinka `/data` priradeného k týmto lekciám NLP.

## Prieskumná analýza údajov

Táto výzva predpokladá, že vytvárate odporúčací bot pre hotely pomocou analýzy sentimentu a hodnotení hostí. Dataset, ktorý budete používať, obsahuje recenzie na 1493 rôznych hotelov v 6 mestách.

Pomocou Pythonu, datasetu hotelových recenzií a analýzy sentimentu NLTK môžete zistiť:

* Aké sú najčastejšie používané slová a frázy v recenziách?
* Korelujú oficiálne *tagy* opisujúce hotel s hodnoteniami recenzií (napr. sú negatívnejšie recenzie pre konkrétny hotel od *Rodiny s malými deťmi* než od *Samostatného cestovateľa*, čo by mohlo naznačovať, že je lepší pre *Samostatných cestovateľov*)?
* Sú skóre sentimentu NLTK v súlade s číselným hodnotením recenzenta?

#### Dataset

Preskúmajme dataset, ktorý ste si stiahli a uložili lokálne. Otvorte súbor v editore ako VS Code alebo dokonca Excel.

Hlavičky v datasete sú nasledovné:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Tu sú zoskupené spôsobom, ktorý môže byť jednoduchší na preskúmanie: 
##### Stĺpce hotela

* `Hotel_Name`, `Hotel_Address`, `lat` (zemepisná šírka), `lng` (zemepisná dĺžka)
  * Pomocou *lat* a *lng* môžete vytvoriť mapu v Pythone, ktorá zobrazuje polohy hotelov (možno farebne odlíšené podľa negatívnych a pozitívnych recenzií)
  * Hotel_Address nie je pre nás zjavne užitočný a pravdepodobne ho nahradíme krajinou pre jednoduchšie triedenie a vyhľadávanie

**Meta-recenzie hotela**

* `Average_Score`
  * Podľa autora datasetu tento stĺpec predstavuje *Priemerné skóre hotela vypočítané na základe najnovšieho komentára za posledný rok*. Toto sa zdá byť nezvyčajný spôsob výpočtu skóre, ale ide o získané údaje, takže ich zatiaľ môžeme brať ako fakt. 
  
  ✅ Na základe ostatných stĺpcov v tomto datasete, dokážete si predstaviť iný spôsob výpočtu priemerného skóre?

* `Total_Number_of_Reviews`
  * Celkový počet recenzií, ktoré hotel dostal - nie je jasné (bez napísania kódu), či sa to vzťahuje na recenzie v datasete.
* `Additional_Number_of_Scoring`
  * To znamená, že bolo udelené hodnotenie, ale recenzent nenapísal pozitívnu ani negatívnu recenziu

**Stĺpce recenzií**

- `Reviewer_Score`
  - Ide o číselnú hodnotu s maximálne jedným desatinným miestom medzi minimálnymi a maximálnymi hodnotami 2.5 a 10
  - Nie je vysvetlené, prečo je najnižšie možné skóre 2.5
- `Negative_Review`
  - Ak recenzent nenapísal nič, toto pole bude obsahovať "**No Negative**"
  - Všimnite si, že recenzent môže napísať pozitívnu recenziu do stĺpca Negative review (napr. "na tomto hoteli nie je nič zlé")
- `Review_Total_Negative_Word_Counts`
  - Vyšší počet negatívnych slov naznačuje nižšie skóre (bez kontroly sentimentu)
- `Positive_Review`
  - Ak recenzent nenapísal nič, toto pole bude obsahovať "**No Positive**"
  - Všimnite si, že recenzent môže napísať negatívnu recenziu do stĺpca Positive review (napr. "na tomto hoteli nie je vôbec nič dobré")
- `Review_Total_Positive_Word_Counts`
  - Vyšší počet pozitívnych slov naznačuje vyššie skóre (bez kontroly sentimentu)
- `Review_Date` a `days_since_review`
  - Na recenziu by sa mohol aplikovať ukazovateľ čerstvosti alebo zastaranosti (staršie recenzie nemusia byť tak presné ako novšie, pretože sa mohlo zmeniť vedenie hotela, prebehli renovácie alebo bol pridaný bazén atď.)
- `Tags`
  - Ide o krátke popisy, ktoré si recenzent môže vybrať na opis typu hosťa (napr. samostatný alebo rodina), typu izby, dĺžky pobytu a spôsobu, akým bola recenzia odoslaná. 
  - Bohužiaľ, použitie týchto tagov je problematické, pozrite si nižšie uvedenú sekciu, ktorá diskutuje o ich užitočnosti

**Stĺpce recenzenta**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Toto by mohlo byť faktorom v odporúčacom modeli, napríklad ak by ste mohli určiť, že plodnejší recenzenti s stovkami recenzií boli skôr negatívni než pozitívni. Avšak recenzent konkrétnej recenzie nie je identifikovaný jedinečným kódom, a preto ho nemožno prepojiť so súborom recenzií. Existuje 30 recenzentov so 100 alebo viac recenziami, ale je ťažké vidieť, ako by to mohlo pomôcť odporúčaciemu modelu.
- `Reviewer_Nationality`
  - Niektorí ľudia si môžu myslieť, že určité národnosti majú väčšiu tendenciu dávať pozitívne alebo negatívne recenzie kvôli národnej inklinácii. Buďte opatrní pri budovaní takýchto anekdotických názorov do svojich modelov. Ide o národné (a niekedy rasové) stereotypy a každý recenzent bol jednotlivec, ktorý napísal recenziu na základe svojej skúsenosti. Mohla byť filtrovaná cez mnoho šošoviek, ako sú ich predchádzajúce pobyty v hoteloch, vzdialenosť, ktorú precestovali, a ich osobný temperament. Myslieť si, že ich národnosť bola dôvodom hodnotenia recenzie, je ťažké odôvodniť.

##### Príklady

| Priemerné skóre | Celkový počet recenzií | Skóre recenzenta | Negatívna <br />recenzia                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pozitívna recenzia                 | Tagy                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Toto momentálne nie je hotel, ale stavenisko. Bol som terorizovaný od skorého rána a celý deň neznesiteľným stavebným hlukom, zatiaľ čo som odpočíval po dlhej ceste a pracoval v izbe. Ľudia pracovali celý deň, napr. s pneumatickými kladivami v susedných izbách. Požiadal som o zmenu izby, ale žiadna tichá izba nebola dostupná. Aby toho nebolo málo, bol som preplatený. Odhlásil som sa večer, pretože som musel odísť na skorý let a dostal som primeraný účet. O deň neskôr hotel vykonal ďalší poplatok bez môjho súhlasu nad rámec rezervovanej ceny. Je to strašné miesto. Nepunujte sa rezerváciou tu. | Nič. Strašné miesto. Držte sa ďalej. | Služobná cesta                                Pár Štandardná dvojlôžková izba Pobyt na 2 noci |

Ako vidíte, tento hosť nemal šťastný pobyt v tomto hoteli. Hotel má dobré priemerné skóre 7.8 a 1945 recenzií, ale tento recenzent mu dal 2.5 a napísal 115 slov o tom, aký negatívny bol jeho pobyt. Ak by nenapísal nič do stĺpca Positive_Review, mohli by ste predpokladať, že nebolo nič pozitívne, ale napriek tomu napísal 7 varovných slov. Ak by sme len počítali slová namiesto významu alebo sentimentu slov, mohli by sme mať skreslený pohľad na zámer recenzenta. Zvláštne je, že jeho skóre 2.5 je mätúce, pretože ak bol pobyt v hoteli taký zlý, prečo mu dal vôbec nejaké body? Pri podrobnom preskúmaní datasetu uvidíte, že najnižšie možné skóre je 2.5, nie 0. Najvyššie možné skóre je 10.

##### Tagy

Ako bolo uvedené vyššie, na prvý pohľad sa zdá, že použitie `Tags` na kategorizáciu údajov dáva zmysel. Bohužiaľ, tieto tagy nie sú štandardizované, čo znamená, že v danom hoteli môžu byť možnosti *Jednolôžková izba*, *Dvojlôžková izba* a *Manželská izba*, ale v ďalšom hoteli sú to *Deluxe jednolôžková izba*, *Klasická izba s kráľovskou posteľou* a *Exekutívna izba s kráľovskou posteľou*. Môžu to byť tie isté veci, ale existuje toľko variácií, že voľba sa stáva:

1. Pokúsiť sa zmeniť všetky termíny na jeden štandard, čo je veľmi náročné, pretože nie je jasné, aká by bola cesta konverzie v každom prípade (napr. *Klasická jednolôžková izba* sa mapuje na *Jednolôžková izba*, ale *Superior Queen Room with Courtyard Garden or City View* je oveľa ťažšie mapovať)

1. Môžeme použiť prístup NLP a merať frekvenciu určitých termínov ako *Samostatný*, *Obchodný cestovateľ* alebo *Rodina s malými deťmi*, ako sa vzťahujú na každý hotel, a zahrnúť to do odporúčania  

Tagy sú zvyčajne (ale nie vždy) jedno pole obsahujúce zoznam 5 až 6 hodnôt oddelených čiarkami, ktoré sa vzťahujú na *Typ cesty*, *Typ hostí*, *Typ izby*, *Počet nocí* a *Typ zariadenia, na ktorom bola recenzia odoslaná*. Avšak, pretože niektorí recenzenti nevyplnia každé pole (môžu nechať jedno prázdne), hodnoty nie sú vždy v rovnakom poradí.

Ako príklad vezmite *Typ skupiny*. V tomto poli v stĺpci `Tags` je 1025 jedinečných možností a bohužiaľ iba niektoré z nich sa vzťahujú na skupinu (niektoré sú typ izby atď.). Ak filtrujete iba tie, ktoré spomínajú rodinu, výsledky obsahujú mnoho typov izieb *Rodinná izba*. Ak zahrniete termín *s*, t.j. počítate hodnoty *Rodina s*, výsledky sú lepšie, s viac ako 80 000 z 515 000 výsledkov obsahujúcich frázu "Rodina s malými deťmi" alebo "Rodina so staršími deťmi".

To znamená, že stĺpec tagov nie je pre nás úplne zbytočný, ale bude si vyžadovať určitú prácu, aby bol užitočný.

##### Priemerné skóre hotela

Existuje niekoľko zvláštností alebo nezrovnalostí v datasete, ktoré neviem vysvetliť, ale sú tu ilustrované, aby ste si ich boli vedomí pri budovaní svojich modelov. Ak ich vyriešite, dajte nám vedieť v diskusnej sekcii!

Dataset má nasledujúce stĺpce týkajúce sa priemerného skóre a počtu recenzií: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel s najväčším počtom recenzií v tomto datasete je *Britannia International Hotel Canary Wharf* s 4789 recenziami z 515 000. Ale ak sa pozrieme na hodnotu `Total_Number_of_Reviews` pre tento hotel, je to 9086. Mohli by ste predpokladať, že existuje oveľa viac skóre bez recenzií, takže možno by sme mali pridať hodnotu stĺpca `Additional_Number_of_Scoring`. Táto hodnota je 2682 a pridaním k 4789 dostaneme 7471, čo je stále o 1615 menej ako `Total_Number_of_Reviews`. 

Ak vezmete stĺpec `Average_Score`, mohli by ste predpokladať, že ide o priemer recenzií v datasete, ale popis z Kaggle je "*Priemerné skóre hotela vypočítané na základe najnovšieho komentára za posledný rok*". To sa nezdá byť veľmi užitočné, ale môžeme vypočítať vlastný priemer na základe skóre recenzií v datasete. Použitím toho istého hotela ako príkladu je priemerné skóre hotela uvedené ako 7.1, ale vypočítané skóre (priemerné skóre recenzenta *v* datasete) je 6.8. To je blízko, ale nie rovnaká hodnota, a môžeme len hádať, že skóre uvedené v recenziách `Additional_Number_of_Scoring` zvýšilo priemer na 7.1. Bohužiaľ, bez možnosti testovania alebo overenia tohto tvrdenia je ťažké použiť alebo dôverovať hodnotám `Average_Score`, `Additional_Number_of_Scoring` a `Total_Number_of_Reviews`, keď sú založené na údajoch, ktoré nemáme.

Aby to bolo ešte komplikovanejšie, hotel s druhým najväčším počtom recenzií má vypočítané priemerné skóre 8.12 a dataset `Average_Score` je 8.1. Je toto správne skóre náhoda alebo je prvý hotel nezrovnalosť? 

Na základe možnosti, že tieto hotely môžu byť odľahlé hodnoty a že možno väčšina hodnôt sa zhoduje (ale niektoré z nejakého dôvodu nie), napíšeme krátky program na preskúmanie hodnôt v datasete a určenie správneho použitia (alebo nepoužitia) hodnôt.
> 🚨 Upozornenie
>
> Pri práci s touto dátovou sadou budete písať kód, ktorý vypočíta niečo z textu bez toho, aby ste museli text sami čítať alebo analyzovať. Toto je podstata NLP – interpretovať význam alebo sentiment bez toho, aby to musel robiť človek. Je však možné, že si prečítate niektoré negatívne recenzie. Dôrazne vás vyzývam, aby ste to nerobili, pretože to nie je potrebné. Niektoré z nich sú hlúpe alebo nepodstatné negatívne recenzie na hotely, ako napríklad „Počasie nebolo dobré“, čo je mimo kontroly hotela alebo kohokoľvek iného. Ale niektoré recenzie majú aj temnú stránku. Niekedy sú negatívne recenzie rasistické, sexistické alebo ageistické. To je nešťastné, ale očakávané v dátovej sade získanej z verejnej webovej stránky. Niektorí recenzenti zanechávajú recenzie, ktoré by ste považovali za nevkusné, nepríjemné alebo znepokojujúce. Je lepšie nechať kód zmerať sentiment, než si ich čítať sami a byť znepokojení. To povedané, je to menšina, ktorá takéto veci píše, ale aj tak existujú.
## Cvičenie - Prieskum údajov
### Načítanie údajov

To je dosť vizuálneho skúmania údajov, teraz napíšete kód a získate odpovede! Táto sekcia používa knižnicu pandas. Vašou prvou úlohou je zabezpečiť, že dokážete načítať a prečítať údaje z CSV. Knižnica pandas má rýchly načítač CSV a výsledok je uložený v dataframe, ako v predchádzajúcich lekciách. CSV, ktoré načítavame, má viac ako pol milióna riadkov, ale iba 17 stĺpcov. Pandas vám poskytuje množstvo výkonných spôsobov interakcie s dataframe, vrátane možnosti vykonávať operácie na každom riadku.

Od tohto bodu v lekcii budú kódové úryvky, vysvetlenia kódu a diskusia o tom, čo výsledky znamenajú. Použite priložený _notebook.ipynb_ na svoj kód.

Začnime načítaním súboru s údajmi, ktorý budete používať:

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

Keď sú údaje načítané, môžeme na nich vykonávať operácie. Tento kód ponechajte na začiatku svojho programu pre ďalšiu časť.

## Preskúmajte údaje

V tomto prípade sú údaje už *čisté*, čo znamená, že sú pripravené na prácu a neobsahujú znaky v iných jazykoch, ktoré by mohli spôsobiť problémy algoritmom očakávajúcim iba anglické znaky.

✅ Môže sa stať, že budete pracovať s údajmi, ktoré vyžadujú počiatočné spracovanie na ich formátovanie pred aplikáciou techník NLP, ale tentokrát to nie je potrebné. Ak by ste museli, ako by ste riešili znaky v iných jazykoch?

Uistite sa, že po načítaní údajov ich dokážete preskúmať pomocou kódu. Je veľmi lákavé zamerať sa na stĺpce `Negative_Review` a `Positive_Review`. Sú naplnené prirodzeným textom, ktorý vaše NLP algoritmy môžu spracovať. Ale počkajte! Predtým, než sa pustíte do NLP a sentimentu, mali by ste postupovať podľa kódu nižšie, aby ste zistili, či hodnoty uvedené v datasete zodpovedajú hodnotám, ktoré vypočítate pomocou pandas.

## Operácie s dataframe

Prvou úlohou v tejto lekcii je overiť, či sú nasledujúce tvrdenia správne, napísaním kódu, ktorý preskúma dataframe (bez jeho zmeny).

> Ako pri mnohých programovacích úlohách, existuje niekoľko spôsobov, ako to dokončiť, ale dobrá rada je urobiť to najjednoduchším a najľahším spôsobom, aký môžete, najmä ak to bude jednoduchšie pochopiť, keď sa k tomuto kódu vrátite v budúcnosti. Pri práci s dataframe existuje komplexné API, ktoré často obsahuje spôsob, ako efektívne dosiahnuť, čo chcete.

Považujte nasledujúce otázky za programovacie úlohy a pokúste sa na ne odpovedať bez toho, aby ste sa pozreli na riešenie.

1. Vypíšte *tvar* dataframe, ktorý ste práve načítali (tvar je počet riadkov a stĺpcov).
2. Vypočítajte frekvenčný počet národností recenzentov:
   1. Koľko rôznych hodnôt je v stĺpci `Reviewer_Nationality` a aké sú?
   2. Ktorá národnosť recenzenta je najčastejšia v datasete (vypíšte krajinu a počet recenzií)?
   3. Aké sú ďalších 10 najčastejšie sa vyskytujúcich národností a ich frekvenčný počet?
3. Ktorý hotel bol najčastejšie recenzovaný pre každú z 10 najčastejších národností recenzentov?
4. Koľko recenzií je na každý hotel (frekvenčný počet hotelov) v datasete?
5. Hoci existuje stĺpec `Average_Score` pre každý hotel v datasete, môžete tiež vypočítať priemerné skóre (získaním priemeru všetkých skóre recenzentov v datasete pre každý hotel). Pridajte nový stĺpec do svojho dataframe s názvom stĺpca `Calc_Average_Score`, ktorý obsahuje vypočítaný priemer.
6. Majú niektoré hotely rovnaké (zaokrúhlené na 1 desatinné miesto) hodnoty `Average_Score` a `Calc_Average_Score`?
   1. Skúste napísať funkciu v Pythone, ktorá berie Series (riadok) ako argument a porovnáva hodnoty, pričom vypíše správu, keď hodnoty nie sú rovnaké. Potom použite metódu `.apply()` na spracovanie každého riadku pomocou funkcie.
7. Vypočítajte a vypíšte, koľko riadkov má stĺpec `Negative_Review` hodnotu "No Negative".
8. Vypočítajte a vypíšte, koľko riadkov má stĺpec `Positive_Review` hodnotu "No Positive".
9. Vypočítajte a vypíšte, koľko riadkov má stĺpec `Positive_Review` hodnotu "No Positive" **a** stĺpec `Negative_Review` hodnotu "No Negative".

### Odpovede na kód

1. Vypíšte *tvar* dataframe, ktorý ste práve načítali (tvar je počet riadkov a stĺpcov).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Vypočítajte frekvenčný počet národností recenzentov:

   1. Koľko rôznych hodnôt je v stĺpci `Reviewer_Nationality` a aké sú?
   2. Ktorá národnosť recenzenta je najčastejšia v datasete (vypíšte krajinu a počet recenzií)?

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

   3. Aké sú ďalších 10 najčastejšie sa vyskytujúcich národností a ich frekvenčný počet?

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

3. Ktorý hotel bol najčastejšie recenzovaný pre každú z 10 najčastejších národností recenzentov?

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

4. Koľko recenzií je na každý hotel (frekvenčný počet hotelov) v datasete?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   Môžete si všimnúť, že výsledky *počítané v datasete* nezodpovedajú hodnote v `Total_Number_of_Reviews`. Nie je jasné, či táto hodnota v datasete predstavovala celkový počet recenzií, ktoré hotel mal, ale nie všetky boli zoškrabané, alebo nejaký iný výpočet. `Total_Number_of_Reviews` sa nepoužíva v modeli kvôli tejto nejasnosti.

5. Hoci existuje stĺpec `Average_Score` pre každý hotel v datasete, môžete tiež vypočítať priemerné skóre (získaním priemeru všetkých skóre recenzentov v datasete pre každý hotel). Pridajte nový stĺpec do svojho dataframe s názvom stĺpca `Calc_Average_Score`, ktorý obsahuje vypočítaný priemer. Vypíšte stĺpce `Hotel_Name`, `Average_Score` a `Calc_Average_Score`.

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

   Môžete sa tiež čudovať nad hodnotou `Average_Score` a prečo je niekedy odlišná od vypočítaného priemerného skóre. Keďže nemôžeme vedieť, prečo niektoré hodnoty zodpovedajú, ale iné majú rozdiel, je najbezpečnejšie v tomto prípade použiť skóre recenzií, ktoré máme, na výpočet priemeru sami. Napriek tomu sú rozdiely zvyčajne veľmi malé, tu sú hotely s najväčšou odchýlkou od priemeru v datasete a vypočítaného priemeru:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   S iba 1 hotelom, ktorý má rozdiel skóre väčší ako 1, to znamená, že pravdepodobne môžeme ignorovať rozdiel a použiť vypočítané priemerné skóre.

6. Vypočítajte a vypíšte, koľko riadkov má stĺpec `Negative_Review` hodnotu "No Negative".

7. Vypočítajte a vypíšte, koľko riadkov má stĺpec `Positive_Review` hodnotu "No Positive".

8. Vypočítajte a vypíšte, koľko riadkov má stĺpec `Positive_Review` hodnotu "No Positive" **a** stĺpec `Negative_Review` hodnotu "No Negative".

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

## Iný spôsob

Iný spôsob, ako počítať položky bez Lambdas, a použiť sum na počítanie riadkov:

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

   Môžete si všimnúť, že existuje 127 riadkov, ktoré majú hodnoty "No Negative" a "No Positive" pre stĺpce `Negative_Review` a `Positive_Review`. To znamená, že recenzent dal hotelu číselné skóre, ale odmietol napísať pozitívnu alebo negatívnu recenziu. Našťastie ide o malý počet riadkov (127 z 515738, alebo 0,02%), takže pravdepodobne neovplyvní náš model alebo výsledky žiadnym konkrétnym smerom, ale možno ste nečakali, že dataset recenzií bude obsahovať riadky bez recenzií, takže stojí za to preskúmať údaje, aby ste objavili takéto riadky.

Teraz, keď ste preskúmali dataset, v ďalšej lekcii budete filtrovať údaje a pridávať analýzu sentimentu.

---
## 🚀Výzva

Táto lekcia demonštruje, ako sme videli v predchádzajúcich lekciách, aké kriticky dôležité je pochopiť svoje údaje a ich zvláštnosti pred vykonaním operácií na nich. Textové údaje si obzvlášť vyžadujú dôkladné preskúmanie. Prejdite rôzne textovo bohaté datasety a zistite, či dokážete objaviť oblasti, ktoré by mohli zaviesť zaujatosť alebo skreslený sentiment do modelu.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Vezmite si [túto vzdelávaciu cestu o NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), aby ste objavili nástroje, ktoré môžete vyskúšať pri budovaní modelov založených na reči a texte.

## Zadanie

[NLTK](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.