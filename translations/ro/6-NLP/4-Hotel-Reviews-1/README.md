<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T16:56:57+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "ro"
}
-->
# Analiza sentimentului cu recenzii de hotel - procesarea datelor

În această secțiune vei folosi tehnicile din lecțiile anterioare pentru a realiza o analiză exploratorie a unui set de date mare. După ce vei avea o înțelegere bună a utilității diferitelor coloane, vei învăța:

- cum să elimini coloanele inutile
- cum să calculezi date noi bazate pe coloanele existente
- cum să salvezi setul de date rezultat pentru utilizare în provocarea finală

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

### Introducere

Până acum ai învățat despre cum datele text sunt destul de diferite față de datele numerice. Dacă textul a fost scris sau rostit de un om, acesta poate fi analizat pentru a găsi modele și frecvențe, sentimente și semnificații. Această lecție te introduce într-un set de date real cu o provocare reală: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, care include o [licență CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Datele au fost extrase de pe Booking.com din surse publice. Creatorul setului de date este Jiashen Liu.

### Pregătire

Vei avea nevoie de:

* Abilitatea de a rula notebook-uri .ipynb folosind Python 3
* pandas
* NLTK, [pe care ar trebui să-l instalezi local](https://www.nltk.org/install.html)
* Setul de date disponibil pe Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Are aproximativ 230 MB dezarhivat. Descarcă-l în folderul rădăcină `/data` asociat acestor lecții NLP.

## Analiza exploratorie a datelor

Această provocare presupune că construiești un bot de recomandare pentru hoteluri folosind analiza sentimentului și scorurile recenziilor oaspeților. Setul de date pe care îl vei folosi include recenzii pentru 1493 hoteluri diferite din 6 orașe.

Folosind Python, un set de date cu recenzii de hotel și analiza sentimentului din NLTK, ai putea afla:

* Care sunt cele mai frecvent utilizate cuvinte și expresii în recenzii?
* Corelează *etichetele* oficiale care descriu un hotel cu scorurile recenziilor (de exemplu, există recenzii mai negative pentru un hotel de la *Familie cu copii mici* decât de la *Călător singur*, indicând poate că este mai potrivit pentru *Călători singuri*?)
* Sunt scorurile sentimentului din NLTK în acord cu scorul numeric al recenziei hotelului?

#### Setul de date

Să explorăm setul de date pe care l-ai descărcat și salvat local. Deschide fișierul într-un editor precum VS Code sau chiar Excel.

Anteturile din setul de date sunt următoarele:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Iată-le grupate într-un mod care ar putea fi mai ușor de examinat: 
##### Coloane despre hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitudine), `lng` (longitudine)
  * Folosind *lat* și *lng* ai putea crea o hartă cu Python care să arate locațiile hotelurilor (poate codificate pe culori pentru recenzii negative și pozitive)
  * Hotel_Address nu pare să fie util pentru noi și probabil îl vom înlocui cu o țară pentru o sortare și căutare mai ușoară

**Coloane meta-recenzie hotel**

* `Average_Score`
  * Conform creatorului setului de date, această coloană reprezintă *Scorul mediu al hotelului, calculat pe baza celui mai recent comentariu din ultimul an*. Acesta pare un mod neobișnuit de a calcula scorul, dar este datele extrase, așa că le putem lua ca atare pentru moment.
  
  ✅ Pe baza celorlalte coloane din acest set de date, te poți gândi la un alt mod de a calcula scorul mediu?

* `Total_Number_of_Reviews`
  * Numărul total de recenzii pe care acest hotel le-a primit - nu este clar (fără a scrie cod) dacă se referă la recenziile din setul de date.
* `Additional_Number_of_Scoring`
  * Acest lucru înseamnă că a fost dat un scor de recenzie, dar nu a fost scrisă nicio recenzie pozitivă sau negativă de către recenzor

**Coloane despre recenzii**

- `Reviewer_Score`
  - Acesta este o valoare numerică cu cel mult 1 zecimală între valorile minime și maxime 2.5 și 10
  - Nu este explicat de ce 2.5 este cel mai mic scor posibil
- `Negative_Review`
  - Dacă un recenzor nu a scris nimic, acest câmp va avea "**No Negative**"
  - Reține că un recenzor poate scrie o recenzie pozitivă în coloana Negative review (de exemplu, "nu este nimic rău la acest hotel")
- `Review_Total_Negative_Word_Counts`
  - Numărul mai mare de cuvinte negative indică un scor mai mic (fără a verifica sentimentul)
- `Positive_Review`
  - Dacă un recenzor nu a scris nimic, acest câmp va avea "**No Positive**"
  - Reține că un recenzor poate scrie o recenzie negativă în coloana Positive review (de exemplu, "nu este nimic bun la acest hotel")
- `Review_Total_Positive_Word_Counts`
  - Numărul mai mare de cuvinte pozitive indică un scor mai mare (fără a verifica sentimentul)
- `Review_Date` și `days_since_review`
  - Se poate aplica o măsură de prospețime sau vechime unei recenzii (recenziile mai vechi s-ar putea să nu fie la fel de precise ca cele mai noi, deoarece managementul hotelului s-a schimbat, s-au făcut renovări, s-a adăugat o piscină etc.)
- `Tags`
  - Acestea sunt descrieri scurte pe care un recenzor le poate selecta pentru a descrie tipul de oaspete care a fost (de exemplu, singur sau familie), tipul de cameră pe care a avut-o, durata șederii și modul în care recenzia a fost trimisă.
  - Din păcate, utilizarea acestor etichete este problematică, verifică secțiunea de mai jos care discută utilitatea lor

**Coloane despre recenzor**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Acesta ar putea fi un factor într-un model de recomandare, de exemplu, dacă ai putea determina că recenzorii mai prolifici cu sute de recenzii sunt mai predispuși să fie negativi decât pozitivi. Totuși, recenzorul oricărei recenzii nu este identificat cu un cod unic și, prin urmare, nu poate fi legat de un set de recenzii. Există 30 de recenzori cu 100 sau mai multe recenzii, dar este greu de văzut cum acest lucru poate ajuta modelul de recomandare.
- `Reviewer_Nationality`
  - Unii oameni ar putea crede că anumite naționalități sunt mai predispuse să ofere o recenzie pozitivă sau negativă din cauza unei înclinații naționale. Fii atent când construiești astfel de opinii anecdotice în modelele tale. Acestea sunt stereotipuri naționale (și uneori rasiale), iar fiecare recenzor a fost un individ care a scris o recenzie bazată pe experiența sa. Aceasta ar fi putut fi filtrată prin multe lentile, cum ar fi șederile anterioare la hotel, distanța parcursă și temperamentul personal. Este greu de justificat ideea că naționalitatea a fost motivul unui scor de recenzie.

##### Exemple

| Scor mediu | Număr total de recenzii | Scor recenzor | Recenzie <br />Negativă                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Recenzie Pozitivă                 | Etichete                                                                                      |
| ---------- | ----------------------- | ------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------- |
| 7.8        | 1945                   | 2.5           | Acesta nu este în prezent un hotel, ci un șantier de construcții. Am fost terorizat de dimineață devreme și toată ziua cu zgomot de construcție inacceptabil în timp ce mă odihneam după o călătorie lungă și lucram în cameră. Oamenii lucrau toată ziua, adică cu ciocane pneumatice în camerele adiacente. Am cerut schimbarea camerei, dar nu era disponibilă nicio cameră liniștită. Ca să fie și mai rău, am fost suprataxat. Am plecat seara, deoarece aveam un zbor foarte devreme și am primit o factură corespunzătoare. O zi mai târziu, hotelul a făcut o altă taxare fără consimțământul meu, peste prețul rezervat. Este un loc groaznic. Nu te pedepsi rezervând aici. | Nimic. Loc groaznic. Stai departe. | Călătorie de afaceri. Cuplu. Cameră dublă standard. Ședere 2 nopți. |

După cum poți vedea, acest oaspete nu a avut o ședere fericită la acest hotel. Hotelul are un scor mediu bun de 7.8 și 1945 de recenzii, dar acest recenzor i-a dat 2.5 și a scris 115 cuvinte despre cât de negativă a fost șederea sa. Dacă nu ar fi scris nimic în coloana Positive_Review, ai putea presupune că nu a fost nimic pozitiv, dar totuși au scris 7 cuvinte de avertizare. Dacă am număra doar cuvintele în loc de semnificația sau sentimentul cuvintelor, am putea avea o viziune distorsionată asupra intenției recenzorului. Ciudat, scorul lor de 2.5 este confuz, deoarece dacă șederea la hotel a fost atât de rea, de ce să-i acorde vreun punct? Investigând setul de date mai atent, vei vedea că cel mai mic scor posibil este 2.5, nu 0. Cel mai mare scor posibil este 10.

##### Etichete

Așa cum s-a menționat mai sus, la prima vedere, ideea de a folosi `Tags` pentru a categorisi datele are sens. Din păcate, aceste etichete nu sunt standardizate, ceea ce înseamnă că într-un hotel, opțiunile ar putea fi *Single room*, *Twin room* și *Double room*, dar în următorul hotel, acestea sunt *Deluxe Single Room*, *Classic Queen Room* și *Executive King Room*. Acestea ar putea fi aceleași lucruri, dar există atât de multe variații încât alegerea devine:

1. Încercarea de a schimba toți termenii într-un standard unic, ceea ce este foarte dificil, deoarece nu este clar care ar fi calea de conversie în fiecare caz (de exemplu, *Classic single room* se mapează la *Single room*, dar *Superior Queen Room with Courtyard Garden or City View* este mult mai greu de mapat)

1. Putem adopta o abordare NLP și măsura frecvența anumitor termeni precum *Solo*, *Business Traveller* sau *Family with young kids* pe măsură ce se aplică fiecărui hotel și să includem acest lucru în recomandare  

Etichetele sunt de obicei (dar nu întotdeauna) un câmp unic care conține o listă de 5 până la 6 valori separate prin virgulă, aliniate la *Tipul de călătorie*, *Tipul de oaspeți*, *Tipul de cameră*, *Numărul de nopți* și *Tipul de dispozitiv pe care a fost trimisă recenzia*. Totuși, deoarece unii recenzori nu completează fiecare câmp (pot lăsa unul gol), valorile nu sunt întotdeauna în aceeași ordine.

De exemplu, ia *Tipul de grup*. Există 1025 de posibilități unice în acest câmp din coloana `Tags`, și din păcate doar unele dintre ele se referă la un grup (unele sunt tipul de cameră etc.). Dacă filtrezi doar cele care menționează familie, rezultatele conțin multe rezultate de tip *Family room*. Dacă incluzi termenul *with*, adică numeri valorile *Family with*, rezultatele sunt mai bune, cu peste 80.000 din cele 515.000 de rezultate conținând expresia "Family with young children" sau "Family with older children".

Aceasta înseamnă că coloana etichete nu este complet inutilă pentru noi, dar va necesita ceva muncă pentru a o face utilă.

##### Scorul mediu al hotelului

Există o serie de ciudățenii sau discrepanțe în setul de date pe care nu le pot explica, dar sunt ilustrate aici pentru a fi conștient de ele atunci când construiești modelele tale. Dacă le descoperi, te rugăm să ne anunți în secțiunea de discuții!

Setul de date are următoarele coloane legate de scorul mediu și numărul de recenzii:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotelul cu cele mai multe recenzii din acest set de date este *Britannia International Hotel Canary Wharf* cu 4789 de recenzii din 515.000. Dar dacă ne uităm la valoarea `Total_Number_of_Reviews` pentru acest hotel, este 9086. Ai putea presupune că există mult mai multe scoruri fără recenzii, așa că poate ar trebui să adăugăm valoarea din coloana `Additional_Number_of_Scoring`. Acea valoare este 2682, iar adăugând-o la 4789 obținem 7471, care este încă cu 1615 mai puțin decât `Total_Number_of_Reviews`. 

Dacă iei coloana `Average_Score`, ai putea presupune că este media recenziilor din setul de date, dar descrierea de pe Kaggle este "*Scorul mediu al hotelului, calculat pe baza celui mai recent comentariu din ultimul an*". Acest lucru nu pare foarte util, dar putem calcula propria noastră medie bazată pe scorurile recenziilor din setul de date. Folosind același hotel ca exemplu, scorul mediu al hotelului este dat ca 7.1, dar scorul calculat (scorul mediu al recenzorilor *din* setul de date) este 6.8. Acesta este apropiat, dar nu aceeași valoare, și putem doar presupune că scorurile date în recenziile `Additional_Number_of_Scoring` au crescut media la 7.1. Din păcate, fără nicio modalitate de a testa sau de a dovedi această afirmație, este dificil să folosim sau să avem încredere în `Average_Score`, `Additional_Number_of_Scoring` și `Total_Number_of_Reviews` atunci când se bazează pe, sau se referă la, date pe care nu le avem.

Pentru a complica lucrurile și mai mult, hotelul cu al doilea cel mai mare număr de recenzii are un scor mediu calculat de 8.12, iar scorul mediu din setul de date este 8.1. Este acest scor corect o coincidență sau este primul hotel o discrepanță? 

Pe posibilitatea ca acest hotel să fie un punct de excepție și că poate majoritatea valorilor se potrivesc (dar unele nu dintr-un motiv oarecare), vom scrie un program scurt în continuare pentru a explora valorile din setul de date și a determina utilizarea corectă (sau neutilizarea) valorilor.
> 🚨 O notă de precauție  
>  
> Când lucrați cu acest set de date, veți scrie cod care calculează ceva din text fără a fi nevoie să citiți sau să analizați textul în mod direct. Aceasta este esența NLP-ului, interpretarea semnificației sau sentimentului fără a fi nevoie ca un om să o facă. Totuși, este posibil să citiți unele dintre recenziile negative. Vă îndemn să nu o faceți, deoarece nu este necesar. Unele dintre ele sunt absurde sau irelevante, cum ar fi recenzii negative despre hotel de genul „Vremea nu a fost grozavă”, ceva ce este dincolo de controlul hotelului sau, de fapt, al oricui. Dar există și o latură întunecată a unor recenzii. Uneori, recenziile negative sunt rasiste, sexiste sau discriminatorii pe criterii de vârstă. Acest lucru este regretabil, dar de așteptat într-un set de date extras de pe un site public. Unii recenzori lasă comentarii pe care le-ați putea considera dezgustătoare, inconfortabile sau deranjante. Este mai bine să lăsați codul să măsoare sentimentul decât să le citiți voi înșivă și să fiți afectați. Acestea fiind spuse, este vorba de o minoritate care scrie astfel de lucruri, dar ele există totuși.
## Exercițiu - Explorarea datelor
### Încarcă datele

Ajunge cu examinarea vizuală a datelor, acum vei scrie cod și vei obține răspunsuri! Această secțiune folosește biblioteca pandas. Prima ta sarcină este să te asiguri că poți încărca și citi datele CSV. Biblioteca pandas are un loader rapid pentru fișiere CSV, iar rezultatul este plasat într-un dataframe, la fel ca în lecțiile anterioare. Fișierul CSV pe care îl încărcăm are peste jumătate de milion de rânduri, dar doar 17 coloane. Pandas îți oferă multe modalități puternice de a interacționa cu un dataframe, inclusiv abilitatea de a efectua operații pe fiecare rând.

De aici înainte, în această lecție, vor fi fragmente de cod și câteva explicații ale codului, precum și discuții despre ce înseamnă rezultatele. Folosește _notebook.ipynb_ inclus pentru codul tău.

Să începem cu încărcarea fișierului de date pe care îl vei folosi:

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

Acum că datele sunt încărcate, putem efectua câteva operații asupra lor. Păstrează acest cod în partea de sus a programului pentru următoarea parte.

## Explorează datele

În acest caz, datele sunt deja *curate*, ceea ce înseamnă că sunt gata de lucru și nu conțin caractere în alte limbi care ar putea încurca algoritmii care se așteaptă doar la caractere în limba engleză.

✅ Este posibil să trebuiască să lucrezi cu date care necesită o procesare inițială pentru a le formata înainte de a aplica tehnici NLP, dar nu de această dată. Dacă ar fi necesar, cum ai gestiona caracterele non-engleze?

Ia un moment pentru a te asigura că, odată ce datele sunt încărcate, le poți explora cu cod. Este foarte ușor să vrei să te concentrezi pe coloanele `Negative_Review` și `Positive_Review`. Acestea sunt pline de text natural pentru algoritmii tăi NLP. Dar stai! Înainte de a te arunca în NLP și sentiment, ar trebui să urmezi codul de mai jos pentru a verifica dacă valorile date în setul de date se potrivesc cu valorile pe care le calculezi cu pandas.

## Operații pe dataframe

Prima sarcină din această lecție este să verifici dacă următoarele afirmații sunt corecte, scriind cod care examinează dataframe-ul (fără a-l modifica).

> La fel ca multe sarcini de programare, există mai multe moduri de a le finaliza, dar un sfat bun este să o faci în cel mai simplu și ușor mod posibil, mai ales dacă va fi mai ușor de înțeles când te vei întoarce la acest cod în viitor. Cu dataframe-uri, există un API cuprinzător care va avea adesea o modalitate eficientă de a face ceea ce dorești.

Tratează următoarele întrebări ca sarcini de cod și încearcă să le răspunzi fără a te uita la soluție.

1. Afișează *forma* dataframe-ului pe care tocmai l-ai încărcat (forma este numărul de rânduri și coloane).
2. Calculează frecvența naționalităților recenzorilor:
   1. Câte valori distincte există pentru coloana `Reviewer_Nationality` și care sunt acestea?
   2. Care este naționalitatea recenzorului cea mai comună în setul de date (afișează țara și numărul de recenzii)?
   3. Care sunt următoarele 10 cele mai frecvente naționalități și frecvența lor?
3. Care a fost hotelul cel mai frecvent recenzat pentru fiecare dintre cele mai frecvente 10 naționalități ale recenzorilor?
4. Câte recenzii sunt per hotel (frecvența recenziilor hotelului) în setul de date?
5. Deși există o coloană `Average_Score` pentru fiecare hotel în setul de date, poți calcula și un scor mediu (obținând media tuturor scorurilor recenzorilor din setul de date pentru fiecare hotel). Adaugă o nouă coloană în dataframe-ul tău cu antetul `Calc_Average_Score` care conține acea medie calculată.
6. Există hoteluri care au același `Average_Score` și `Calc_Average_Score` (rotunjite la o zecimală)?
   1. Încearcă să scrii o funcție Python care ia un Series (rând) ca argument și compară valorile, afișând un mesaj când valorile nu sunt egale. Apoi folosește metoda `.apply()` pentru a procesa fiecare rând cu funcția.
7. Calculează și afișează câte rânduri au valori "No Negative" în coloana `Negative_Review`.
8. Calculează și afișează câte rânduri au valori "No Positive" în coloana `Positive_Review`.
9. Calculează și afișează câte rânduri au valori "No Positive" în coloana `Positive_Review` **și** valori "No Negative" în coloana `Negative_Review`.

### Răspunsuri în cod

1. Afișează *forma* dataframe-ului pe care tocmai l-ai încărcat (forma este numărul de rânduri și coloane).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calculează frecvența naționalităților recenzorilor:

   1. Câte valori distincte există pentru coloana `Reviewer_Nationality` și care sunt acestea?
   2. Care este naționalitatea recenzorului cea mai comună în setul de date (afișează țara și numărul de recenzii)?

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

   3. Care sunt următoarele 10 cele mai frecvente naționalități și frecvența lor?

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

3. Care a fost hotelul cel mai frecvent recenzat pentru fiecare dintre cele mai frecvente 10 naționalități ale recenzorilor?

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

4. Câte recenzii sunt per hotel (frecvența recenziilor hotelului) în setul de date?

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
   
   Poți observa că rezultatele *numărate în setul de date* nu se potrivesc cu valoarea din `Total_Number_of_Reviews`. Nu este clar dacă această valoare din setul de date reprezenta numărul total de recenzii pe care hotelul le avea, dar nu toate au fost extrase, sau alt calcul. `Total_Number_of_Reviews` nu este utilizat în model din cauza acestei neclarități.

5. Deși există o coloană `Average_Score` pentru fiecare hotel în setul de date, poți calcula și un scor mediu (obținând media tuturor scorurilor recenzorilor din setul de date pentru fiecare hotel). Adaugă o nouă coloană în dataframe-ul tău cu antetul `Calc_Average_Score` care conține acea medie calculată. Afișează coloanele `Hotel_Name`, `Average_Score` și `Calc_Average_Score`.

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

   Poți să te întrebi despre valoarea `Average_Score` și de ce este uneori diferită de scorul mediu calculat. Deoarece nu putem ști de ce unele valori se potrivesc, dar altele au o diferență, este mai sigur în acest caz să folosim scorurile recenzorilor pe care le avem pentru a calcula media noi înșine. Totuși, diferențele sunt de obicei foarte mici, iată hotelurile cu cea mai mare deviație între media din setul de date și media calculată:

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

   Cu doar 1 hotel având o diferență de scor mai mare de 1, înseamnă că probabil putem ignora diferența și să folosim scorul mediu calculat.

6. Calculează și afișează câte rânduri au valori "No Negative" în coloana `Negative_Review`.

7. Calculează și afișează câte rânduri au valori "No Positive" în coloana `Positive_Review`.

8. Calculează și afișează câte rânduri au valori "No Positive" în coloana `Positive_Review` **și** valori "No Negative" în coloana `Negative_Review`.

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

## O altă metodă

O altă metodă de a număra elementele fără Lambdas și de a folosi sum pentru a număra rândurile:

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

   Poți să fi observat că există 127 rânduri care au atât valori "No Negative", cât și "No Positive" pentru coloanele `Negative_Review` și `Positive_Review` respectiv. Asta înseamnă că recenzorul a dat hotelului un scor numeric, dar a refuzat să scrie fie o recenzie pozitivă, fie negativă. Din fericire, aceasta este o cantitate mică de rânduri (127 din 515738, sau 0.02%), deci probabil nu va influența modelul sau rezultatele într-o direcție particulară, dar s-ar putea să nu te fi așteptat ca un set de date de recenzii să aibă rânduri fără recenzii, așa că merită să explorezi datele pentru a descoperi astfel de rânduri.

Acum că ai explorat setul de date, în lecția următoare vei filtra datele și vei adăuga o analiză a sentimentului.

---
## 🚀Provocare

Această lecție demonstrează, așa cum am văzut în lecțiile anterioare, cât de important este să înțelegi datele și particularitățile lor înainte de a efectua operații asupra lor. Datele bazate pe text, în special, necesită o examinare atentă. Explorează diverse seturi de date bogate în text și vezi dacă poți descoperi zone care ar putea introduce bias sau sentiment distorsionat într-un model.

## [Quiz post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Urmează [acest parcurs de învățare despre NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) pentru a descoperi instrumente pe care să le încerci atunci când construiești modele bazate pe vorbire și text.

## Temă

[NLTK](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.