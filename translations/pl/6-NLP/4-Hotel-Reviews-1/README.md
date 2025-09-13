<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T08:30:04+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "pl"
}
-->
# Analiza sentymentu w recenzjach hoteli - przetwarzanie danych

W tej sekcji wykorzystasz techniki z poprzednich lekcji, aby przeprowadzić eksploracyjną analizę danych dużego zbioru danych. Gdy zrozumiesz użyteczność różnych kolumn, nauczysz się:

- jak usuwać niepotrzebne kolumny,
- jak obliczać nowe dane na podstawie istniejących kolumn,
- jak zapisać wynikowy zbiór danych do wykorzystania w końcowym wyzwaniu.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

### Wprowadzenie

Do tej pory nauczyłeś się, że dane tekstowe różnią się od danych numerycznych. Jeśli tekst został napisany lub wypowiedziany przez człowieka, można go analizować, aby znaleźć wzorce, częstotliwości, sentyment i znaczenie. Ta lekcja wprowadza Cię w prawdziwy zbiór danych z rzeczywistym wyzwaniem: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, który posiada [licencję CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Dane zostały zebrane z Booking.com z publicznych źródeł. Twórcą zbioru danych jest Jiashen Liu.

### Przygotowanie

Będziesz potrzebować:

* Możliwości uruchamiania notebooków .ipynb za pomocą Python 3,
* pandas,
* NLTK, [który należy zainstalować lokalnie](https://www.nltk.org/install.html),
* Zbioru danych dostępnego na Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Po rozpakowaniu zajmuje około 230 MB. Pobierz go do folderu głównego `/data` powiązanego z tymi lekcjami NLP.

## Eksploracyjna analiza danych

To wyzwanie zakłada, że budujesz bota rekomendującego hotele, wykorzystując analizę sentymentu i oceny gości. Zbiór danych, który będziesz używać, zawiera recenzje 1493 różnych hoteli w 6 miastach.

Korzystając z Pythona, zbioru danych recenzji hoteli oraz analizy sentymentu NLTK, możesz dowiedzieć się:

* Jakie są najczęściej używane słowa i frazy w recenzjach?
* Czy oficjalne *tagi* opisujące hotel korelują z ocenami recenzji (np. czy bardziej negatywne recenzje dla danego hotelu pochodzą od *Rodzin z małymi dziećmi* niż od *Podróżujących samotnie*, co może wskazywać, że hotel jest lepszy dla *Podróżujących samotnie*)?
* Czy oceny sentymentu NLTK "zgadzają się" z numeryczną oceną recenzenta?

#### Zbiór danych

Przyjrzyjmy się zbiorowi danych, który pobrałeś i zapisałeś lokalnie. Otwórz plik w edytorze, takim jak VS Code lub nawet Excel.

Nagłówki w zbiorze danych są następujące:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Tutaj są pogrupowane w sposób, który może być łatwiejszy do analizy: 
##### Kolumny hotelowe

* `Hotel_Name`, `Hotel_Address`, `lat` (szerokość geograficzna), `lng` (długość geograficzna)
  * Korzystając z *lat* i *lng*, możesz stworzyć mapę w Pythonie pokazującą lokalizacje hoteli (może z kodowaniem kolorów dla recenzji negatywnych i pozytywnych).
  * Hotel_Address nie wydaje się być dla nas szczególnie użyteczny, prawdopodobnie zastąpimy go krajem dla łatwiejszego sortowania i wyszukiwania.

**Kolumny meta-recenzji hotelowych**

* `Average_Score`
  * Według twórcy zbioru danych, ta kolumna to *Średnia ocena hotelu, obliczona na podstawie najnowszego komentarza z ostatniego roku*. To wydaje się być nietypowym sposobem obliczania oceny, ale ponieważ dane zostały zebrane, możemy na razie przyjąć je za dobrą monetę.
  
  ✅ Na podstawie innych kolumn w tych danych, czy możesz wymyślić inny sposób obliczenia średniej oceny?

* `Total_Number_of_Reviews`
  * Całkowita liczba recenzji, które otrzymał hotel - nie jest jasne (bez napisania kodu), czy odnosi się to do recenzji w zbiorze danych.
* `Additional_Number_of_Scoring`
  * Oznacza, że ocena została podana, ale recenzent nie napisał pozytywnej ani negatywnej recenzji.

**Kolumny recenzji**

- `Reviewer_Score`
  - Jest to wartość numeryczna z maksymalnie jednym miejscem dziesiętnym, w zakresie od 2.5 do 10.
  - Nie wyjaśniono, dlaczego najniższa możliwa ocena to 2.5.
- `Negative_Review`
  - Jeśli recenzent nic nie napisał, to pole będzie zawierać "**No Negative**".
  - Zauważ, że recenzent może napisać pozytywną recenzję w kolumnie Negative review (np. "nie ma nic złego w tym hotelu").
- `Review_Total_Negative_Word_Counts`
  - Wyższa liczba słów negatywnych wskazuje na niższą ocenę (bez sprawdzania sentymentalności).
- `Positive_Review`
  - Jeśli recenzent nic nie napisał, to pole będzie zawierać "**No Positive**".
  - Zauważ, że recenzent może napisać negatywną recenzję w kolumnie Positive review (np. "w tym hotelu nie ma absolutnie nic dobrego").
- `Review_Total_Positive_Word_Counts`
  - Wyższa liczba słów pozytywnych wskazuje na wyższą ocenę (bez sprawdzania sentymentalności).
- `Review_Date` i `days_since_review`
  - Można zastosować miarę świeżości lub przestarzałości recenzji (starsze recenzje mogą być mniej dokładne niż nowsze, ponieważ zarząd hotelu się zmienił, przeprowadzono remonty, dodano basen itp.).
- `Tags`
  - Są to krótkie opisy, które recenzent może wybrać, aby opisać typ gościa (np. samotny lub rodzina), typ pokoju, długość pobytu i sposób przesłania recenzji.
  - Niestety, użycie tych tagów jest problematyczne, sprawdź sekcję poniżej, która omawia ich użyteczność.

**Kolumny recenzenta**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Może to być czynnik w modelu rekomendacji, na przykład jeśli można ustalić, że bardziej płodni recenzenci z setkami recenzji częściej są negatywni niż pozytywni. Jednak recenzent konkretnej recenzji nie jest identyfikowany za pomocą unikalnego kodu, więc nie można go powiązać z zestawem recenzji. Jest 30 recenzentów z 100 lub więcej recenzjami, ale trudno dostrzec, jak może to pomóc w modelu rekomendacji.
- `Reviewer_Nationality`
  - Niektórzy mogą sądzić, że pewne narodowości częściej wystawiają pozytywne lub negatywne recenzje z powodu narodowych skłonności. Należy uważać, aby nie budować takich anegdotycznych poglądów w modelach. Są to narodowe (a czasem rasowe) stereotypy, a każdy recenzent był indywidualnością, która napisała recenzję na podstawie swojego doświadczenia. Mogło ono być filtrowane przez wiele czynników, takich jak wcześniejsze pobyty w hotelach, odległość podróży i osobisty temperament. Trudno uzasadnić, że narodowość była powodem oceny recenzji.

##### Przykłady

| Średnia ocena | Całkowita liczba recenzji | Ocena recenzenta | Negatywna <br />recenzja                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pozytywna recenzja                 | Tagi                                                                                      |
| -------------- | ------------------------ | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                     | 2.5              | To obecnie nie jest hotel, ale plac budowy. Byłem terroryzowany od wczesnego rana i przez cały dzień nieakceptowalnym hałasem budowlanym, odpoczywając po długiej podróży i pracując w pokoju. Ludzie pracowali cały dzień, np. młotami pneumatycznymi w sąsiednich pokojach. Poprosiłem o zmianę pokoju, ale nie było dostępnego cichego pokoju. Co gorsza, zostałem obciążony nadmierną opłatą. Wymeldowałem się wieczorem, ponieważ musiałem wcześnie wyjechać na lot i otrzymałem odpowiedni rachunek. Dzień później hotel dokonał kolejnego obciążenia bez mojej zgody, przekraczając cenę rezerwacji. To okropne miejsce. Nie karz się, rezerwując tutaj. | Nic. Okropne miejsce. Trzymaj się z daleka. | Podróż służbowa, Para, Standardowy pokój dwuosobowy, Pobyt 2 noce |

Jak widać, ten gość nie miał udanego pobytu w hotelu. Hotel ma dobrą średnią ocenę 7.8 i 1945 recenzji, ale ten recenzent wystawił ocenę 2.5 i napisał 115 słów o tym, jak negatywny był jego pobyt. Jeśli nic nie napisał w kolumnie Positive_Review, można by przypuszczać, że nie było nic pozytywnego, ale jednak napisał 7 słów ostrzeżenia. Jeśli liczymy tylko słowa zamiast znaczenia lub sentymentu słów, możemy mieć wypaczone spojrzenie na intencje recenzenta. Co dziwne, jego ocena 2.5 jest myląca, ponieważ jeśli pobyt w hotelu był tak zły, dlaczego w ogóle przyznał jakieś punkty? Analizując zbiór danych dokładnie, zauważysz, że najniższa możliwa ocena to 2.5, a nie 0. Najwyższa możliwa ocena to 10.

##### Tagi

Jak wspomniano powyżej, na pierwszy rzut oka pomysł użycia `Tags` do kategoryzacji danych ma sens. Niestety, te tagi nie są ustandaryzowane, co oznacza, że w danym hotelu opcje mogą być *Pokój jednoosobowy*, *Pokój dwuosobowy*, i *Pokój małżeński*, ale w następnym hotelu są to *Deluxe Single Room*, *Classic Queen Room*, i *Executive King Room*. Mogą to być te same rzeczy, ale istnieje tak wiele wariacji, że wybór staje się:

1. Próba zmiany wszystkich terminów na jeden standard, co jest bardzo trudne, ponieważ nie jest jasne, jaka byłaby ścieżka konwersji w każdym przypadku (np. *Classic single room* mapuje się na *Single room*, ale *Superior Queen Room with Courtyard Garden or City View* jest znacznie trudniejsze do zmapowania).

1. Możemy podejść do tego za pomocą NLP i zmierzyć częstotliwość pewnych terminów, takich jak *Solo*, *Podróżujący służbowo*, lub *Rodzina z małymi dziećmi*, w odniesieniu do każdego hotelu i uwzględnić to w rekomendacji.

Tagi zazwyczaj (ale nie zawsze) są pojedynczym polem zawierającym listę 5 do 6 wartości oddzielonych przecinkami, odpowiadających *Typowi podróży*, *Typowi gości*, *Typowi pokoju*, *Liczbie nocy* i *Typowi urządzenia, na którym przesłano recenzję*. Jednak ponieważ niektórzy recenzenci nie wypełniają każdego pola (mogą zostawić jedno puste), wartości nie zawsze są w tej samej kolejności.

Na przykład, weź *Typ grupy*. W tej kolumnie `Tags` znajduje się 1025 unikalnych możliwości, a niestety tylko niektóre z nich odnoszą się do grupy (niektóre dotyczą typu pokoju itp.). Jeśli przefiltrujesz tylko te, które wspominają rodzinę, wyniki zawierają wiele wyników typu *Pokój rodzinny*. Jeśli uwzględnisz termin *z*, tj. policzysz wartości *Rodzina z*, wyniki są lepsze, z ponad 80 000 z 515 000 wyników zawierających frazę "Rodzina z małymi dziećmi" lub "Rodzina ze starszymi dziećmi".

Oznacza to, że kolumna tagów nie jest dla nas całkowicie bezużyteczna, ale wymaga pracy, aby była użyteczna.

##### Średnia ocena hotelu

Istnieje kilka dziwności lub rozbieżności w zbiorze danych, których nie mogę rozgryźć, ale są tutaj zilustrowane, abyś był ich świadomy podczas budowania swoich modeli. Jeśli je rozgryziesz, daj nam znać w sekcji dyskusji!

Zbiór danych zawiera następujące kolumny dotyczące średniej oceny i liczby recenzji:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel z największą liczbą recenzji w tym zbiorze danych to *Britannia International Hotel Canary Wharf* z 4789 recenzjami z 515 000. Ale jeśli spojrzymy na wartość `Total_Number_of_Reviews` dla tego hotelu, wynosi ona 9086. Można by przypuszczać, że istnieje wiele więcej ocen bez recenzji, więc może powinniśmy dodać wartość z kolumny `Additional_Number_of_Scoring`. Ta wartość wynosi 2682, a dodanie jej do 4789 daje nam 7471, co nadal jest o 1615 mniej niż `Total_Number_of_Reviews`.

Jeśli weźmiesz kolumnę `Average_Score`, możesz przypuszczać, że jest to średnia recenzji w zbiorze danych, ale opis z Kaggle mówi: "*Średnia ocena hotelu, obliczona na podstawie najnowszego komentarza z ostatniego roku*". To nie wydaje się być szczególnie użyteczne, ale możemy obliczyć własną średnią na podstawie ocen recenzentów w zbiorze danych. Korzystając z tego samego hotelu jako przykładu, średnia ocena hotelu wynosi 7.1, ale obliczona ocena (średnia ocena recenzentów *w* zbiorze danych) wynosi 6.8. Jest to bliskie, ale nie ta sama wartość, i możemy tylko przypuszczać, że oceny podane w recenzjach `Additional_Number_of_Scoring` zwiększyły średnią do 7.1. Niestety, bez możliwości przetestowania lub udowodnienia tego założenia, trudno jest używać lub ufać `Average_Score`, `Additional_Number_of_Scoring` i `Total_Number_of_Reviews`, gdy są one oparte na danych, których nie posiadamy.

Aby jeszcze bardziej skomplikować sprawę, hotel z drugą największą liczbą recenzji ma obliczoną średnią ocenę 8.12, a `Average_Score` w zbiorze danych wynosi 8.1. Czy ta poprawna ocena to przypadek, czy pierwszy hotel to rozbieżność?

Zakładając, że te hotele mogą być odstającymi wartościami, a może większość wartości się zgadza (ale niektóre z jakiegoś powodu nie), napiszemy krótki program, aby zbadać wartości w zbiorze danych i określić poprawne użycie (lub brak użycia) tych wartości.
> 🚨 Uwaga  
>  
> Pracując z tym zestawem danych, będziesz pisać kod, który oblicza coś na podstawie tekstu, bez konieczności czytania lub analizowania tekstu samodzielnie. To jest istota NLP – interpretowanie znaczenia lub nastroju bez udziału człowieka. Jednak możliwe jest, że przeczytasz niektóre negatywne recenzje. Zachęcam, aby tego nie robić, ponieważ nie musisz. Niektóre z nich są absurdalne lub nieistotne, jak na przykład negatywne opinie o hotelu typu: „Pogoda była kiepska”, coś, co jest poza kontrolą hotelu, a nawet kogokolwiek. Ale istnieje też ciemna strona niektórych recenzji. Czasami negatywne opinie są rasistowskie, seksistowskie lub dyskryminujące ze względu na wiek. To jest przykre, ale niestety spodziewane w zestawie danych zebranym z publicznej strony internetowej. Niektórzy recenzenci zostawiają opinie, które mogą być niesmaczne, niekomfortowe lub wręcz przykre. Lepiej pozwolić kodowi zmierzyć nastrój niż czytać je samemu i się tym przejmować. Powiedziawszy to, jest to mniejszość, która pisze takie rzeczy, ale mimo wszystko istnieją.
## Ćwiczenie - Eksploracja danych
### Wczytaj dane

Wystarczy już wizualnego badania danych, teraz napiszesz trochę kodu, aby uzyskać odpowiedzi! W tej sekcji używamy biblioteki pandas. Twoim pierwszym zadaniem jest upewnienie się, że potrafisz wczytać i odczytać dane z pliku CSV. Biblioteka pandas ma szybki loader CSV, a wynik jest umieszczany w dataframe, tak jak w poprzednich lekcjach. CSV, który wczytujemy, zawiera ponad pół miliona wierszy, ale tylko 17 kolumn. Pandas oferuje wiele potężnych sposobów interakcji z dataframe, w tym możliwość wykonywania operacji na każdym wierszu.

Od tego momentu w tej lekcji znajdziesz fragmenty kodu, wyjaśnienia dotyczące kodu oraz dyskusje na temat znaczenia wyników. Użyj dołączonego pliku _notebook.ipynb_ do swojego kodu.

Zacznijmy od wczytania pliku danych, którego będziesz używać:

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

Teraz, gdy dane zostały wczytane, możemy wykonać na nich pewne operacje. Umieść ten kod na początku swojego programu na potrzeby kolejnej części.

## Eksploracja danych

W tym przypadku dane są już *czyste*, co oznacza, że są gotowe do pracy i nie zawierają znaków w innych językach, które mogłyby sprawić trudności algorytmom oczekującym wyłącznie znaków angielskich.

✅ Możesz mieć do czynienia z danymi, które wymagają wstępnego przetwarzania, aby je sformatować przed zastosowaniem technik NLP, ale tym razem nie musisz tego robić. Jeśli musiałbyś, jak poradziłbyś sobie z nieangielskimi znakami?

Poświęć chwilę, aby upewnić się, że po wczytaniu danych możesz je eksplorować za pomocą kodu. Bardzo łatwo jest skupić się na kolumnach `Negative_Review` i `Positive_Review`. Są one wypełnione naturalnym tekstem, który Twoje algorytmy NLP mogą przetwarzać. Ale poczekaj! Zanim przejdziesz do NLP i analizy sentymentu, powinieneś skorzystać z poniższego kodu, aby upewnić się, że wartości podane w zestawie danych odpowiadają wartościom, które obliczasz za pomocą pandas.

## Operacje na dataframe

Pierwszym zadaniem w tej lekcji jest sprawdzenie, czy poniższe założenia są poprawne, poprzez napisanie kodu, który bada dataframe (bez jego zmieniania).

> Podobnie jak w przypadku wielu zadań programistycznych, istnieje kilka sposobów na ich wykonanie, ale dobrą radą jest zrobienie tego w najprostszy, najłatwiejszy sposób, zwłaszcza jeśli będzie to łatwiejsze do zrozumienia, gdy wrócisz do tego kodu w przyszłości. W przypadku dataframe istnieje kompleksowe API, które często ma sposób na efektywne wykonanie tego, czego potrzebujesz.

Potraktuj poniższe pytania jako zadania programistyczne i spróbuj odpowiedzieć na nie bez zaglądania do rozwiązania.

1. Wypisz *kształt* dataframe, który właśnie wczytałeś (kształt to liczba wierszy i kolumn).
2. Oblicz częstotliwość występowania narodowości recenzentów:
   1. Ile jest unikalnych wartości w kolumnie `Reviewer_Nationality` i jakie one są?
   2. Jaka narodowość recenzentów jest najczęstsza w zestawie danych (wypisz kraj i liczbę recenzji)?
   3. Jakie są kolejne 10 najczęściej występujących narodowości i ich liczba?
3. Jaki hotel był najczęściej recenzowany dla każdej z 10 najczęstszych narodowości recenzentów?
4. Ile recenzji przypada na każdy hotel (częstotliwość recenzji hotelu) w zestawie danych?
5. Chociaż w zestawie danych znajduje się kolumna `Average_Score` dla każdego hotelu, możesz również obliczyć średnią ocenę (uzyskując średnią wszystkich ocen recenzentów w zestawie danych dla każdego hotelu). Dodaj nową kolumnę do swojego dataframe z nagłówkiem kolumny `Calc_Average_Score`, która zawiera tę obliczoną średnią.
6. Czy są jakieś hotele, które mają tę samą (zaokrągloną do 1 miejsca po przecinku) wartość `Average_Score` i `Calc_Average_Score`?
   1. Spróbuj napisać funkcję w Pythonie, która przyjmuje Series (wiersz) jako argument i porównuje wartości, wypisując wiadomość, gdy wartości nie są równe. Następnie użyj metody `.apply()`, aby przetworzyć każdy wiersz za pomocą tej funkcji.
7. Oblicz i wypisz, ile wierszy ma wartości "No Negative" w kolumnie `Negative_Review`.
8. Oblicz i wypisz, ile wierszy ma wartości "No Positive" w kolumnie `Positive_Review`.
9. Oblicz i wypisz, ile wierszy ma wartości "No Positive" w kolumnie `Positive_Review` **i** wartości "No Negative" w kolumnie `Negative_Review`.

### Odpowiedzi w kodzie

1. Wypisz *kształt* dataframe, który właśnie wczytałeś (kształt to liczba wierszy i kolumn).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Oblicz częstotliwość występowania narodowości recenzentów:

   1. Ile jest unikalnych wartości w kolumnie `Reviewer_Nationality` i jakie one są?
   2. Jaka narodowość recenzentów jest najczęstsza w zestawie danych (wypisz kraj i liczbę recenzji)?

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

   3. Jakie są kolejne 10 najczęściej występujących narodowości i ich liczba?

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

3. Jaki hotel był najczęściej recenzowany dla każdej z 10 najczęstszych narodowości recenzentów?

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

4. Ile recenzji przypada na każdy hotel (częstotliwość recenzji hotelu) w zestawie danych?

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
   
   Możesz zauważyć, że wyniki *policzone w zestawie danych* nie odpowiadają wartości w `Total_Number_of_Reviews`. Nie jest jasne, czy wartość w zestawie danych reprezentowała całkowitą liczbę recenzji, które hotel miał, ale nie wszystkie zostały zebrane, czy też była to inna kalkulacja. `Total_Number_of_Reviews` nie jest używana w modelu z powodu tej niejasności.

5. Chociaż w zestawie danych znajduje się kolumna `Average_Score` dla każdego hotelu, możesz również obliczyć średnią ocenę (uzyskując średnią wszystkich ocen recenzentów w zestawie danych dla każdego hotelu). Dodaj nową kolumnę do swojego dataframe z nagłówkiem kolumny `Calc_Average_Score`, która zawiera tę obliczoną średnią. Wypisz kolumny `Hotel_Name`, `Average_Score` i `Calc_Average_Score`.

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

   Możesz również zastanawiać się nad wartością `Average_Score` i dlaczego czasami różni się od obliczonej średniej. Ponieważ nie możemy wiedzieć, dlaczego niektóre wartości się zgadzają, a inne mają różnicę, najbezpieczniej w tym przypadku jest użyć ocen recenzentów, które mamy, aby samodzielnie obliczyć średnią. Powiedziawszy to, różnice są zazwyczaj bardzo małe, oto hotele z największym odchyleniem od średniej zestawu danych i obliczonej średniej:

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

   Tylko 1 hotel ma różnicę w ocenie większą niż 1, co oznacza, że prawdopodobnie możemy zignorować różnicę i użyć obliczonej średniej oceny.

6. Oblicz i wypisz, ile wierszy ma wartości "No Negative" w kolumnie `Negative_Review`.

7. Oblicz i wypisz, ile wierszy ma wartości "No Positive" w kolumnie `Positive_Review`.

8. Oblicz i wypisz, ile wierszy ma wartości "No Positive" w kolumnie `Positive_Review` **i** wartości "No Negative" w kolumnie `Negative_Review`.

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

## Inny sposób

Inny sposób na liczenie elementów bez użycia Lambd i wykorzystanie sumy do zliczania wierszy:

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

   Możesz zauważyć, że istnieje 127 wierszy, które mają zarówno wartości "No Negative", jak i "No Positive" w kolumnach `Negative_Review` i `Positive_Review`. Oznacza to, że recenzent podał hotelowi ocenę liczbową, ale odmówił napisania zarówno pozytywnej, jak i negatywnej recenzji. Na szczęście jest to niewielka liczba wierszy (127 z 515738, czyli 0,02%), więc prawdopodobnie nie wpłynie to na nasz model ani wyniki w żadnym konkretnym kierunku, ale możesz nie spodziewać się, że zestaw danych recenzji będzie zawierał wiersze bez recenzji, więc warto eksplorować dane, aby odkryć takie wiersze.

Teraz, gdy eksplorowałeś zestaw danych, w następnej lekcji przefiltrujesz dane i dodasz analizę sentymentu.

---
## 🚀Wyzwanie

Ta lekcja pokazuje, jak widzieliśmy w poprzednich lekcjach, jak niezwykle ważne jest zrozumienie swoich danych i ich niuansów przed wykonaniem operacji na nich. Dane tekstowe w szczególności wymagają dokładnej analizy. Przejrzyj różne zestawy danych bogate w tekst i sprawdź, czy możesz odkryć obszary, które mogłyby wprowadzić uprzedzenia lub zniekształcony sentyment do modelu.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Weź [ten ścieżkę nauki o NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), aby odkryć narzędzia, które możesz wypróbować podczas budowania modeli opartych na mowie i tekście.

## Zadanie

[NLTK](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za źródło autorytatywne. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.