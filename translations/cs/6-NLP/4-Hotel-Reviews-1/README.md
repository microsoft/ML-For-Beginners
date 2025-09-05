<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T01:25:44+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "cs"
}
-->
# Analýza sentimentu pomocí recenzí hotelů - zpracování dat

V této části použijete techniky z předchozích lekcí k provedení průzkumné analýzy velkého datového souboru. Jakmile získáte dobré porozumění užitečnosti jednotlivých sloupců, naučíte se:

- jak odstranit nepotřebné sloupce
- jak vypočítat nová data na základě existujících sloupců
- jak uložit výsledný datový soubor pro použití v závěrečné výzvě

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

### Úvod

Doposud jste se naučili, že textová data se výrazně liší od číselných typů dat. Pokud jde o text napsaný nebo vyslovený člověkem, lze jej analyzovat za účelem nalezení vzorců, frekvencí, sentimentu a významu. Tato lekce vás zavede do skutečného datového souboru s reálnou výzvou: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, který je dostupný pod [licencí CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Data byla získána z Booking.com z veřejných zdrojů. Tvůrcem datového souboru je Jiashen Liu.

### Příprava

Budete potřebovat:

* Schopnost spouštět .ipynb notebooky pomocí Pythonu 3
* pandas
* NLTK, [který byste měli nainstalovat lokálně](https://www.nltk.org/install.html)
* Datový soubor, který je dostupný na Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Po rozbalení má přibližně 230 MB. Stáhněte jej do kořenové složky `/data` spojené s těmito lekcemi NLP.

## Průzkumná analýza dat

Tato výzva předpokládá, že vytváříte doporučovacího bota pro hotely pomocí analýzy sentimentu a hodnocení hostů. Datový soubor, který budete používat, obsahuje recenze 1493 různých hotelů v 6 městech.

Pomocí Pythonu, datového souboru recenzí hotelů a analýzy sentimentu NLTK můžete zjistit:

* Jaká jsou nejčastěji používaná slova a fráze v recenzích?
* Korelují oficiální *tagy* popisující hotel s hodnocením recenzí (např. jsou negativnější recenze pro konkrétní hotel od *Rodiny s malými dětmi* než od *Samostatného cestovatele*, což by mohlo naznačovat, že je lepší pro *Samostatné cestovatele*)?
* Souhlasí skóre sentimentu NLTK s číselným hodnocením recenzenta?

#### Datový soubor

Prozkoumejme datový soubor, který jste stáhli a uložili lokálně. Otevřete soubor v editoru, jako je VS Code nebo dokonce Excel.

Hlavičky v datovém souboru jsou následující:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Zde jsou seskupeny způsobem, který může být snazší prozkoumat: 
##### Sloupce hotelu

* `Hotel_Name`, `Hotel_Address`, `lat` (zeměpisná šířka), `lng` (zeměpisná délka)
  * Pomocí *lat* a *lng* můžete vytvořit mapu v Pythonu zobrazující polohy hotelů (možná barevně odlišené podle negativních a pozitivních recenzí)
  * Hotel_Address pro nás není zjevně užitečný a pravděpodobně jej nahradíme zemí pro snazší třídění a vyhledávání

**Sloupce meta-recenze hotelu**

* `Average_Score`
  * Podle tvůrce datového souboru tento sloupec představuje *Průměrné skóre hotelu, vypočítané na základě posledního komentáře za poslední rok*. To se zdá být neobvyklý způsob výpočtu skóre, ale jedná se o získaná data, takže je prozatím můžeme brát jako daná. 
  
  ✅ Na základě ostatních sloupců v těchto datech, dokážete vymyslet jiný způsob výpočtu průměrného skóre?

* `Total_Number_of_Reviews`
  * Celkový počet recenzí, které hotel obdržel - není jasné (bez napsání nějakého kódu), zda se to týká recenzí v datovém souboru.
* `Additional_Number_of_Scoring`
  * To znamená, že bylo uděleno hodnocení, ale recenzent nenapsal žádnou pozitivní ani negativní recenzi.

**Sloupce recenzí**

- `Reviewer_Score`
  - Jedná se o číselnou hodnotu s maximálně 1 desetinným místem mezi minimální a maximální hodnotou 2.5 a 10
  - Není vysvětleno, proč je nejnižší možné skóre 2.5
- `Negative_Review`
  - Pokud recenzent nic nenapsal, toto pole bude obsahovat "**No Negative**"
  - Všimněte si, že recenzent může napsat pozitivní recenzi do sloupce Negative review (např. "na tomto hotelu není nic špatného")
- `Review_Total_Negative_Word_Counts`
  - Vyšší počet negativních slov naznačuje nižší skóre (bez kontroly sentimentu)
- `Positive_Review`
  - Pokud recenzent nic nenapsal, toto pole bude obsahovat "**No Positive**"
  - Všimněte si, že recenzent může napsat negativní recenzi do sloupce Positive review (např. "na tomto hotelu není vůbec nic dobrého")
- `Review_Total_Positive_Word_Counts`
  - Vyšší počet pozitivních slov naznačuje vyšší skóre (bez kontroly sentimentu)
- `Review_Date` a `days_since_review`
  - Na recenzi by mohl být aplikován ukazatel čerstvosti nebo zastaralosti (starší recenze nemusí být tak přesné jako novější, protože se změnilo vedení hotelu, proběhla renovace nebo byl přidán bazén atd.)
- `Tags`
  - Jedná se o krátké popisné štítky, které si recenzent může vybrat k popisu typu hosta (např. samostatný nebo rodina), typu pokoje, délky pobytu a způsobu, jakým byla recenze odeslána. 
  - Bohužel použití těchto štítků je problematické, viz níže uvedená část, která pojednává o jejich užitečnosti.

**Sloupce recenzenta**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - To by mohlo být faktorem v doporučovacím modelu, například pokud byste mohli určit, že plodnější recenzenti se stovkami recenzí byli spíše negativní než pozitivní. Nicméně recenzent konkrétní recenze není identifikován jedinečným kódem, a proto nemůže být propojen se sadou recenzí. Existuje 30 recenzentů se 100 nebo více recenzemi, ale je těžké vidět, jak by to mohlo pomoci doporučovacímu modelu.
- `Reviewer_Nationality`
  - Někteří lidé by si mohli myslet, že určité národnosti mají větší tendenci dávat pozitivní nebo negativní recenze kvůli národnímu sklonu. Buďte opatrní při začleňování takových anekdotických názorů do svých modelů. Jedná se o národní (a někdy rasové) stereotypy a každý recenzent byl jedinec, který napsal recenzi na základě své zkušenosti. Mohla být filtrována skrze mnoho hledisek, jako jsou jejich předchozí pobyty v hotelu, vzdálenost, kterou cestovali, a jejich osobní temperament. Myslet si, že jejich národnost byla důvodem hodnocení, je těžké ospravedlnit.

##### Příklady

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Toto aktuálně není hotel, ale staveniště. Byl jsem terorizován od brzkého rána a celý den nepřijatelným stavebním hlukem, zatímco jsem odpočíval po dlouhé cestě a pracoval v pokoji. Lidé pracovali celý den, např. s bouracími kladivy v sousedních pokojích. Požádal jsem o změnu pokoje, ale žádný tichý pokoj nebyl k dispozici. Aby toho nebylo málo, byl jsem přeúčtován. Odhlásil jsem se večer, protože jsem měl velmi brzy let a obdržel jsem odpovídající účet. O den později hotel provedl další poplatek bez mého souhlasu nad rámec rezervované ceny. Je to hrozné místo. Nezničte si pobyt tím, že si zde rezervujete. | Nic. Hrozné místo. Držte se dál. | Služební cesta, Pár, Standardní dvoulůžkový pokoj, Pobyt na 2 noci |

Jak vidíte, tento host neměl šťastný pobyt v tomto hotelu. Hotel má dobré průměrné skóre 7.8 a 1945 recenzí, ale tento recenzent mu dal 2.5 a napsal 115 slov o tom, jak negativní byl jeho pobyt. Pokud by nenapsal nic do sloupce Positive_Review, mohli byste usoudit, že nebylo nic pozitivního, ale přesto napsal 7 slov varování. Pokud bychom pouze počítali slova místo významu nebo sentimentu slov, mohli bychom mít zkreslený pohled na záměr recenzenta. Podivně je jejich skóre 2.5 matoucí, protože pokud byl pobyt v hotelu tak špatný, proč mu vůbec dát nějaké body? Při bližším zkoumání datového souboru zjistíte, že nejnižší možné skóre je 2.5, nikoli 0. Nejvyšší možné skóre je 10.

##### Tagy

Jak bylo uvedeno výše, na první pohled se zdá, že použití `Tags` k kategorizaci dat dává smysl. Bohužel tyto tagy nejsou standardizované, což znamená, že v daném hotelu mohou být možnosti *Jednolůžkový pokoj*, *Dvoulůžkový pokoj* a *Pokoj s manželskou postelí*, ale v dalším hotelu jsou to *Deluxe jednolůžkový pokoj*, *Klasický pokoj s královskou postelí* a *Pokoj Executive s královskou postelí*. Mohou to být stejné věci, ale existuje tolik variací, že volba se stává:

1. Pokusit se změnit všechny termíny na jeden standard, což je velmi obtížné, protože není jasné, jak by měl být převod proveden v každém případě (např. *Klasický jednolůžkový pokoj* mapuje na *Jednolůžkový pokoj*, ale *Superior Queen Room with Courtyard Garden or City View* je mnohem těžší mapovat)

1. Můžeme použít přístup NLP a měřit frekvenci určitých termínů, jako je *Samostatný*, *Obchodní cestovatel* nebo *Rodina s malými dětmi*, jak se vztahují na každý hotel, a zahrnout to do doporučení  

Tagy jsou obvykle (ale ne vždy) jedno pole obsahující seznam 5 až 6 hodnot oddělených čárkami odpovídajících *Typu cesty*, *Typu hostů*, *Typu pokoje*, *Počtu nocí* a *Typu zařízení, na kterém byla recenze odeslána*. Nicméně, protože někteří recenzenti nevyplní každé pole (mohou jedno pole nechat prázdné), hodnoty nejsou vždy ve stejném pořadí.

Například vezměte *Typ skupiny*. V tomto poli ve sloupci `Tags` je 1025 unikátních možností a bohužel pouze některé z nich se týkají skupiny (některé jsou typ pokoje atd.). Pokud filtrujete pouze ty, které zmiňují rodinu, výsledky obsahují mnoho typů *Rodinný pokoj*. Pokud zahrnete termín *s*, tj. počítáte hodnoty *Rodina s*, výsledky jsou lepší, s více než 80 000 z 515 000 výsledků obsahujících frázi "Rodina s malými dětmi" nebo "Rodina se staršími dětmi".

To znamená, že sloupec tagů pro nás není úplně zbytečný, ale bude vyžadovat určitou práci, aby byl užitečný.

##### Průměrné skóre hotelu

Existuje řada zvláštností nebo nesrovnalostí v datovém souboru, které nemohu vysvětlit, ale jsou zde ilustrovány, abyste si je byli vědomi při vytváření svých modelů. Pokud na to přijdete, dejte nám prosím vědět v diskusní sekci!

Datový soubor má následující sloupce týkající se průměrného skóre a počtu recenzí: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel s největším počtem recenzí v tomto datovém souboru je *Britannia International Hotel Canary Wharf* s 4789 recenzemi z 515 000. Ale pokud se podíváme na hodnotu `Total_Number_of_Reviews` pro tento hotel, je to 9086. Mohli byste usoudit, že existuje mnohem více skóre bez recenzí, takže bychom možná měli přidat hodnotu sloupce `Additional_Number_of_Scoring`. Tato hodnota je 2682 a přidáním k 4789 dostaneme 7471, což je stále o 1615 méně než `Total_Number_of_Reviews`. 

Pokud vezmete sloupec `Average_Score`, mohli byste usoudit, že se jedná o průměr recenzí v datovém souboru, ale popis z Kaggle je "*Průměrné skóre hotelu, vypočítané na základě posledního komentáře za poslední rok*". To se nezdá být příliš užitečné, ale můžeme vypočítat vlastní průměr na základě skóre recenzí v datovém souboru. Použitím stejného hotelu jako příkladu je průměrné skóre hotelu uvedeno jako 7.1, ale vypočítané skóre (průměrné skóre recenzenta *v* datovém souboru) je 6.8. To je blízko, ale ne stejná hodnota, a můžeme pouze hádat, že skóre uvedená v recenzích `Additional_Number_of_Scoring` zvýšila průměr na 7.1. Bohužel bez možnosti testování nebo ověření tohoto tvrzení je obtížné použít nebo důvěřovat `Average_Score`, `Additional_Number_of_Scoring` a `Total_Number_of_Reviews`, když jsou založeny na datech, která nemáme.

Aby to bylo ještě složitější, hotel s druhým nejvyšším počtem recenzí má vypočítané průměrné skóre 8.12 a průměrné skóre v datovém souboru je 8.1. Je toto správné skóre náhoda, nebo je první hotel nesrovnalostí? 

S možností, že tyto hotely mohou být odlehlé hodnoty, a že možná většina hodnot odpovídá (ale některé z nějakého důvodu ne), napíšeme krátký program, který prozkoumá hodnoty v datovém souboru a určí správné použití (nebo nepoužití) hodnot.
> 🚨 Poznámka k opatrnosti
>
> Při práci s touto datovou sadou budete psát kód, který něco vypočítá z textu, aniž byste museli text sami číst nebo analyzovat. To je podstata NLP, interpretace významu nebo sentimentu bez nutnosti lidského zásahu. Je však možné, že některé negativní recenze přečtete. Důrazně vás žádám, abyste to nedělali, protože to není nutné. Některé z nich jsou hloupé nebo irelevantní negativní recenze na hotely, například "Počasí nebylo skvělé", což je něco, co hotel, ani nikdo jiný, nemůže ovlivnit. Ale některé recenze mají i temnou stránku. Někdy jsou negativní recenze rasistické, sexistické nebo ageistické. To je nešťastné, ale očekávané u datové sady získané z veřejné webové stránky. Někteří recenzenti zanechávají recenze, které by vám mohly připadat odpudivé, nepříjemné nebo znepokojivé. Je lepší nechat kód měřit sentiment, než je číst sami a být znepokojeni. To znamená, že takové recenze píše menšina, ale přesto existují.
## Cvičení - Průzkum dat
### Načtení dat

To bylo dost vizuálního zkoumání dat, teď napíšete nějaký kód a získáte odpovědi! Tato část využívá knihovnu pandas. Vaším úplně prvním úkolem je zajistit, že dokážete načíst a přečíst data z CSV souboru. Knihovna pandas má rychlý nástroj pro načítání CSV, jehož výsledek je uložen do dataframe, stejně jako v předchozích lekcích. CSV, které načítáme, obsahuje přes půl milionu řádků, ale pouze 17 sloupců. Pandas vám nabízí mnoho výkonných způsobů, jak pracovat s dataframe, včetně možnosti provádět operace na každém řádku.

Od této chvíle v této lekci budou ukázky kódu, vysvětlení kódu a diskuze o tom, co výsledky znamenají. Použijte přiložený _notebook.ipynb_ pro svůj kód.

Začněme načtením datového souboru, který budete používat:

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

Jakmile jsou data načtena, můžeme na nich provádět některé operace. Tento kód ponechte na začátku svého programu pro další část.

## Průzkum dat

V tomto případě jsou data již *čistá*, což znamená, že jsou připravena k práci a neobsahují znaky v jiných jazycích, které by mohly způsobit problémy algoritmům očekávajícím pouze anglické znaky.

✅ Může se stát, že budete pracovat s daty, která vyžadují počáteční zpracování, aby byla připravena pro aplikaci NLP technik, ale tentokrát to není nutné. Pokud by to bylo potřeba, jak byste se vypořádali s neanglickými znaky?

Ujistěte se, že jakmile jsou data načtena, můžete je prozkoumat pomocí kódu. Je velmi snadné zaměřit se na sloupce `Negative_Review` a `Positive_Review`. Tyto sloupce obsahují přirozený text, který vaše NLP algoritmy mohou zpracovat. Ale počkejte! Než se pustíte do NLP a analýzy sentimentu, měli byste podle níže uvedeného kódu ověřit, zda hodnoty uvedené v datasetu odpovídají hodnotám, které vypočítáte pomocí pandas.

## Operace s dataframe

Prvním úkolem v této lekci je ověřit, zda následující tvrzení jsou správná, tím, že napíšete kód, který zkoumá dataframe (bez jeho změny).

> Stejně jako u mnoha programovacích úkolů existuje několik způsobů, jak to provést, ale dobrá rada je udělat to co nejjednodušším a nejpřehlednějším způsobem, zejména pokud bude snazší pochopit váš kód, když se k němu v budoucnu vrátíte. U dataframe existuje komplexní API, které často nabízí efektivní způsob, jak dosáhnout toho, co potřebujete.

Považujte následující otázky za programovací úkoly a pokuste se na ně odpovědět bez nahlížení do řešení.

1. Vytiskněte *rozměry* dataframe, který jste právě načetli (rozměry jsou počet řádků a sloupců).
2. Vypočítejte frekvenční počet pro národnosti recenzentů:
   1. Kolik různých hodnot je ve sloupci `Reviewer_Nationality` a jaké jsou?
   2. Která národnost recenzentů je v datasetu nejčastější (vytiskněte zemi a počet recenzí)?
   3. Jakých je dalších 10 nejčastěji se vyskytujících národností a jejich frekvenční počet?
3. Který hotel byl nejčastěji recenzován pro každou z 10 nejčastějších národností recenzentů?
4. Kolik recenzí je na každý hotel (frekvenční počet hotelů) v datasetu?
5. Ačkoli dataset obsahuje sloupec `Average_Score` pro každý hotel, můžete také vypočítat průměrné skóre (získáním průměru všech skóre recenzentů v datasetu pro každý hotel). Přidejte nový sloupec do svého dataframe s názvem `Calc_Average_Score`, který obsahuje tento vypočítaný průměr. 
6. Mají některé hotely stejné (zaokrouhlené na 1 desetinné místo) hodnoty `Average_Score` a `Calc_Average_Score`?
   1. Zkuste napsat Python funkci, která přijímá Series (řádek) jako argument a porovnává hodnoty, přičemž tiskne zprávu, když hodnoty nejsou stejné. Poté použijte metodu `.apply()` k zpracování každého řádku pomocí této funkce.
7. Vypočítejte a vytiskněte, kolik řádků má ve sloupci `Negative_Review` hodnotu "No Negative".
8. Vypočítejte a vytiskněte, kolik řádků má ve sloupci `Positive_Review` hodnotu "No Positive".
9. Vypočítejte a vytiskněte, kolik řádků má ve sloupci `Positive_Review` hodnotu "No Positive" **a** ve sloupci `Negative_Review` hodnotu "No Negative".

### Odpovědi na kód

1. Vytiskněte *rozměry* dataframe, který jste právě načetli (rozměry jsou počet řádků a sloupců).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Vypočítejte frekvenční počet pro národnosti recenzentů:

   1. Kolik různých hodnot je ve sloupci `Reviewer_Nationality` a jaké jsou?
   2. Která národnost recenzentů je v datasetu nejčastější (vytiskněte zemi a počet recenzí)?

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

   3. Jakých je dalších 10 nejčastěji se vyskytujících národností a jejich frekvenční počet?

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

3. Který hotel byl nejčastěji recenzován pro každou z 10 nejčastějších národností recenzentů?

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

4. Kolik recenzí je na každý hotel (frekvenční počet hotelů) v datasetu?

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
   
   Můžete si všimnout, že výsledky *počítané v datasetu* neodpovídají hodnotě ve `Total_Number_of_Reviews`. Není jasné, zda tato hodnota v datasetu představovala celkový počet recenzí, které hotel měl, ale ne všechny byly získány, nebo nějaký jiný výpočet. `Total_Number_of_Reviews` není použit v modelu kvůli této nejasnosti.

5. Ačkoli dataset obsahuje sloupec `Average_Score` pro každý hotel, můžete také vypočítat průměrné skóre (získáním průměru všech skóre recenzentů v datasetu pro každý hotel). Přidejte nový sloupec do svého dataframe s názvem `Calc_Average_Score`, který obsahuje tento vypočítaný průměr. Vytiskněte sloupce `Hotel_Name`, `Average_Score` a `Calc_Average_Score`.

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

   Můžete se také divit hodnotě `Average_Score` a proč se někdy liší od vypočítaného průměrného skóre. Jelikož nemůžeme vědět, proč některé hodnoty odpovídají, ale jiné mají rozdíl, je v tomto případě nejbezpečnější použít skóre recenzí, které máme, k výpočtu průměru sami. Nicméně rozdíly jsou obvykle velmi malé, zde jsou hotely s největší odchylkou od průměru datasetu a vypočítaného průměru:

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

   Pouze 1 hotel má rozdíl skóre větší než 1, což znamená, že rozdíl můžeme pravděpodobně ignorovat a použít vypočítané průměrné skóre.

6. Vypočítejte a vytiskněte, kolik řádků má ve sloupci `Negative_Review` hodnotu "No Negative".

7. Vypočítejte a vytiskněte, kolik řádků má ve sloupci `Positive_Review` hodnotu "No Positive".

8. Vypočítejte a vytiskněte, kolik řádků má ve sloupci `Positive_Review` hodnotu "No Positive" **a** ve sloupci `Negative_Review` hodnotu "No Negative".

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

## Jiný způsob

Jiný způsob, jak počítat položky bez použití Lambdas, a použít sum k počítání řádků:

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

   Můžete si všimnout, že existuje 127 řádků, které mají hodnoty "No Negative" a "No Positive" ve sloupcích `Negative_Review` a `Positive_Review`. To znamená, že recenzent dal hotelu číselné skóre, ale odmítl napsat pozitivní nebo negativní recenzi. Naštěstí je to malý počet řádků (127 z 515738, tedy 0,02 %), takže to pravděpodobně neovlivní náš model nebo výsledky žádným konkrétním směrem, ale možná jste nečekali, že dataset recenzí bude obsahovat řádky bez recenzí, takže stojí za to prozkoumat data a objevit takové řádky.

Nyní, když jste prozkoumali dataset, v další lekci budete filtrovat data a přidávat analýzu sentimentu.

---
## 🚀Výzva

Tato lekce ukazuje, jak je, jak jsme viděli v předchozích lekcích, kriticky důležité porozumět svým datům a jejich zvláštnostem před prováděním operací na nich. Textová data obzvláště vyžadují pečlivé zkoumání. Prozkoumejte různé datové sady bohaté na text a zjistěte, zda dokážete objevit oblasti, které by mohly zavést zkreslení nebo ovlivnit sentiment modelu.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Recenze & Samostudium

Vezměte [tuto vzdělávací cestu o NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), abyste objevili nástroje, které můžete vyzkoušet při vytváření modelů zaměřených na řeč a text.

## Zadání 

[NLTK](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.