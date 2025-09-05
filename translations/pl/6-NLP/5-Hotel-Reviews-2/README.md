<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T08:33:12+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "pl"
}
-->
# Analiza sentymentu na podstawie recenzji hoteli

Teraz, gdy dokładnie przeanalizowałeś zbiór danych, czas przefiltrować kolumny i zastosować techniki NLP, aby uzyskać nowe informacje o hotelach.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

### Operacje filtrowania i analizy sentymentu

Jak zapewne zauważyłeś, zbiór danych ma kilka problemów. Niektóre kolumny zawierają nieprzydatne informacje, inne wydają się niepoprawne. Nawet jeśli są poprawne, nie jest jasne, jak zostały obliczone, a wyniki nie mogą być niezależnie zweryfikowane na podstawie własnych obliczeń.

## Ćwiczenie: trochę więcej przetwarzania danych

Oczyść dane jeszcze bardziej. Dodaj kolumny, które będą przydatne później, zmień wartości w innych kolumnach i całkowicie usuń niektóre kolumny.

1. Wstępne przetwarzanie kolumn

   1. Usuń `lat` i `lng`

   2. Zastąp wartości w kolumnie `Hotel_Address` następującymi wartościami (jeśli adres zawiera nazwę miasta i kraju, zmień go na samo miasto i kraj).

      Oto jedyne miasta i kraje w zbiorze danych:

      Amsterdam, Netherlands

      Barcelona, Spain

      London, United Kingdom

      Milan, Italy

      Paris, France

      Vienna, Austria 

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

      Teraz możesz zapytać o dane na poziomie kraju:

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

2. Przetwarzanie kolumn meta-recenzji hoteli

   1. Usuń `Additional_Number_of_Scoring`

   2. Zastąp `Total_Number_of_Reviews` całkowitą liczbą recenzji dla danego hotelu, które faktycznie znajdują się w zbiorze danych 

   3. Zastąp `Average_Score` własnym obliczonym wynikiem

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Przetwarzanie kolumn recenzji

   1. Usuń `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` i `days_since_review`

   2. Zachowaj `Reviewer_Score`, `Negative_Review` i `Positive_Review` bez zmian
     
   3. Zachowaj `Tags` na razie

     - W następnej sekcji przeprowadzimy dodatkowe operacje filtrowania na tagach, a następnie je usuniemy

4. Przetwarzanie kolumn recenzentów

   1. Usuń `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Zachowaj `Reviewer_Nationality`

### Kolumna Tag

Kolumna `Tag` jest problematyczna, ponieważ zawiera listę (w formie tekstowej) przechowywaną w kolumnie. Niestety, kolejność i liczba podsekcji w tej kolumnie nie zawsze są takie same. Trudno jest człowiekowi zidentyfikować odpowiednie frazy, które mogą być interesujące, ponieważ zbiór danych zawiera 515 000 wierszy i 1427 hoteli, a każdy z nich ma nieco inne opcje, które recenzent mógł wybrać. Tutaj przydaje się NLP. Możesz przeanalizować tekst, znaleźć najczęstsze frazy i je policzyć.

Niestety, nie interesują nas pojedyncze słowa, ale frazy wielowyrazowe (np. *Podróż służbowa*). Uruchomienie algorytmu częstotliwości fraz wielowyrazowych na tak dużej ilości danych (6762646 słów) mogłoby zająć niezwykle dużo czasu, ale bez analizy danych wydaje się, że jest to konieczny wydatek. Tutaj przydaje się eksploracyjna analiza danych, ponieważ widząc próbkę tagów, takich jak `[' Podróż służbowa  ', ' Podróżujący samotnie ', ' Pokój jednoosobowy ', ' Pobyt 5 nocy ', ' Wysłano z urządzenia mobilnego ']`, możesz zacząć zastanawiać się, czy można znacznie zmniejszyć ilość przetwarzania, które musisz wykonać. Na szczęście jest to możliwe - ale najpierw musisz wykonać kilka kroków, aby określić interesujące tagi.

### Filtrowanie tagów

Pamiętaj, że celem zbioru danych jest dodanie sentymentu i kolumn, które pomogą Ci wybrać najlepszy hotel (dla siebie lub może dla klienta, który zleca Ci stworzenie bota rekomendującego hotele). Musisz zadać sobie pytanie, czy tagi są przydatne w ostatecznym zbiorze danych. Oto jedna interpretacja (jeśli potrzebowałbyś zbioru danych do innych celów, różne tagi mogłyby zostać uwzględnione/wykluczone):

1. Rodzaj podróży jest istotny i powinien zostać
2. Rodzaj grupy gości jest ważny i powinien zostać
3. Rodzaj pokoju, apartamentu lub studia, w którym przebywał gość, jest nieistotny (wszystkie hotele mają w zasadzie te same pokoje)
4. Urządzenie, z którego wysłano recenzję, jest nieistotne
5. Liczba nocy spędzonych przez recenzenta *mogłaby* być istotna, jeśli przypiszesz dłuższe pobyty do większego zadowolenia z hotelu, ale to mało prawdopodobne i raczej nieistotne

Podsumowując, **zachowaj 2 rodzaje tagów i usuń pozostałe**.

Najpierw nie chcesz liczyć tagów, dopóki nie będą w lepszym formacie, co oznacza usunięcie nawiasów kwadratowych i cudzysłowów. Możesz to zrobić na kilka sposobów, ale chcesz wybrać najszybszy, ponieważ przetwarzanie dużej ilości danych może zająć dużo czasu. Na szczęście pandas oferuje łatwy sposób na wykonanie każdego z tych kroków.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Każdy tag staje się czymś w rodzaju: `Podróż służbowa, Podróżujący samotnie, Pokój jednoosobowy, Pobyt 5 nocy, Wysłano z urządzenia mobilnego`. 

Następnie pojawia się problem. Niektóre recenzje lub wiersze mają 5 kolumn, inne 3, jeszcze inne 6. Jest to wynik sposobu, w jaki zbiór danych został utworzony, i trudno to naprawić. Chcesz uzyskać częstotliwość występowania każdej frazy, ale są one w różnej kolejności w każdej recenzji, więc liczba może być nieprecyzyjna, a hotel może nie otrzymać przypisanego tagu, na który zasługuje.

Zamiast tego wykorzystasz różną kolejność na swoją korzyść, ponieważ każdy tag jest wielowyrazowy, ale także oddzielony przecinkiem! Najprostszym sposobem na to jest utworzenie 6 tymczasowych kolumn, w których każdy tag zostanie wstawiony do kolumny odpowiadającej jego kolejności w tagu. Następnie możesz połączyć 6 kolumn w jedną dużą kolumnę i uruchomić metodę `value_counts()` na wynikowej kolumnie. Po jej wydrukowaniu zobaczysz, że było 2428 unikalnych tagów. Oto mała próbka:

| Tag                            | Liczba |
| ------------------------------ | ------ |
| Podróż wypoczynkowa            | 417778 |
| Wysłano z urządzenia mobilnego | 307640 |
| Para                           | 252294 |
| Pobyt 1 noc                   | 193645 |
| Pobyt 2 noce                  | 133937 |
| Podróżujący samotnie           | 108545 |
| Pobyt 3 noce                  | 95821  |
| Podróż służbowa                | 82939  |
| Grupa                          | 65392  |
| Rodzina z małymi dziećmi       | 61015  |
| Pobyt 4 noce                  | 47817  |
| Pokój dwuosobowy               | 35207  |
| Standardowy pokój dwuosobowy   | 32248  |
| Superior pokój dwuosobowy      | 31393  |
| Rodzina ze starszymi dziećmi   | 26349  |
| Deluxe pokój dwuosobowy        | 24823  |
| Pokój dwuosobowy lub typu twin | 22393  |
| Pobyt 5 nocy                  | 20845  |
| Standardowy pokój dwuosobowy lub typu twin | 17483  |
| Klasyczny pokój dwuosobowy     | 16989  |
| Superior pokój dwuosobowy lub typu twin | 13570 |
| 2 pokoje                       | 12393  |

Niektóre z popularnych tagów, takich jak `Wysłano z urządzenia mobilnego`, są dla nas bezużyteczne, więc może być rozsądne usunięcie ich przed liczeniem występowania fraz, ale jest to tak szybka operacja, że możesz je zostawić i po prostu je zignorować.

### Usuwanie tagów dotyczących długości pobytu

Usunięcie tych tagów to pierwszy krok, który nieco zmniejsza całkowitą liczbę tagów do rozważenia. Zauważ, że nie usuwasz ich ze zbioru danych, a jedynie decydujesz się na ich pominięcie przy liczeniu/zachowywaniu w zbiorze recenzji.

| Długość pobytu | Liczba |
| --------------- | ------ |
| Pobyt 1 noc     | 193645 |
| Pobyt 2 noce    | 133937 |
| Pobyt 3 noce    | 95821  |
| Pobyt 4 noce    | 47817  |
| Pobyt 5 nocy    | 20845  |
| Pobyt 6 nocy    | 9776   |
| Pobyt 7 nocy    | 7399   |
| Pobyt 8 nocy    | 2502   |
| Pobyt 9 nocy    | 1293   |
| ...             | ...    |

Istnieje ogromna różnorodność pokoi, apartamentów, studiów, mieszkań i tak dalej. Wszystkie one oznaczają mniej więcej to samo i nie są dla Ciebie istotne, więc usuń je z rozważań.

| Rodzaj pokoju               | Liczba |
| --------------------------- | ------ |
| Pokój dwuosobowy           | 35207  |
| Standardowy pokój dwuosobowy | 32248  |
| Superior pokój dwuosobowy   | 31393  |
| Deluxe pokój dwuosobowy     | 24823  |
| Pokój dwuosobowy lub typu twin | 22393 |
| Standardowy pokój dwuosobowy lub typu twin | 17483 |
| Klasyczny pokój dwuosobowy  | 16989  |
| Superior pokój dwuosobowy lub typu twin | 13570 |

Na koniec, i to jest satysfakcjonujące (ponieważ nie wymagało dużego przetwarzania), zostaniesz z następującymi *przydatnymi* tagami:

| Tag                                           | Liczba |
| --------------------------------------------- | ------ |
| Podróż wypoczynkowa                           | 417778 |
| Para                                          | 252294 |
| Podróżujący samotnie                          | 108545 |
| Podróż służbowa                               | 82939  |
| Grupa (połączona z Podróżującymi z przyjaciółmi) | 67535  |
| Rodzina z małymi dziećmi                      | 61015  |
| Rodzina ze starszymi dziećmi                  | 26349  |
| Z pupilem                                     | 1405   |

Można by argumentować, że `Podróżujący z przyjaciółmi` to w zasadzie to samo co `Grupa`, i byłoby to uzasadnione, aby połączyć te dwa tagi, jak powyżej. Kod do identyfikacji odpowiednich tagów znajduje się w [notatniku Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Ostatnim krokiem jest utworzenie nowych kolumn dla każdego z tych tagów. Następnie, dla każdego wiersza recenzji, jeśli kolumna `Tag` pasuje do jednej z nowych kolumn, dodaj 1, jeśli nie, dodaj 0. Końcowym wynikiem będzie liczba recenzentów, którzy wybrali ten hotel (w sumie) np. na podróż służbową lub wypoczynkową, albo na pobyt z pupilem, co jest przydatną informacją przy rekomendowaniu hotelu.

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

### Zapisz swój plik

Na koniec zapisz zbiór danych w obecnej formie pod nową nazwą.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operacje analizy sentymentu

W tej ostatniej sekcji zastosujesz analizę sentymentu do kolumn recenzji i zapiszesz wyniki w zbiorze danych.

## Ćwiczenie: załaduj i zapisz przefiltrowane dane

Zauważ, że teraz ładujesz przefiltrowany zbiór danych zapisany w poprzedniej sekcji, **a nie** oryginalny zbiór danych.

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

### Usuwanie stop słów

Jeśli uruchomisz analizę sentymentu na kolumnach recenzji negatywnych i pozytywnych, może to zająć dużo czasu. Testowane na wydajnym laptopie z szybkim procesorem, zajęło to 12-14 minut w zależności od użytej biblioteki analizy sentymentu. To (stosunkowo) długi czas, więc warto sprawdzić, czy można go skrócić. 

Usuwanie stop słów, czyli powszechnych angielskich słów, które nie wpływają na sentyment zdania, to pierwszy krok. Usunięcie ich powinno przyspieszyć analizę sentymentu, ale nie zmniejszyć jej dokładności (ponieważ stop słowa nie wpływają na sentyment, ale spowalniają analizę). 

Najdłuższa negatywna recenzja miała 395 słów, ale po usunięciu stop słów, ma 195 słów.

Usuwanie stop słów to również szybka operacja, usunięcie ich z 2 kolumn recenzji w 515 000 wierszy zajęło 3,3 sekundy na urządzeniu testowym. Może to zająć nieco więcej lub mniej czasu w zależności od szybkości procesora, pamięci RAM, posiadania dysku SSD i innych czynników. Relatywnie krótki czas operacji oznacza, że jeśli poprawi czas analizy sentymentu, to warto ją wykonać.

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

### Przeprowadzanie analizy sentymentu

Teraz powinieneś obliczyć analizę sentymentu dla kolumn recenzji negatywnych i pozytywnych oraz zapisać wynik w 2 nowych kolumnach. Test analizy sentymentu będzie polegał na porównaniu go z oceną recenzenta dla tej samej recenzji. Na przykład, jeśli analiza sentymentu uzna, że negatywna recenzja miała sentyment 1 (bardzo pozytywny sentyment), a pozytywna recenzja również sentyment 1, ale recenzent dał hotelowi najniższą możliwą ocenę, to albo tekst recenzji nie odpowiada ocenie, albo analizator sentymentu nie rozpoznał poprawnie sentymentu. Powinieneś spodziewać się, że niektóre wyniki analizy sentymentu będą całkowicie błędne, co często będzie można wyjaśnić, np. recenzja może być bardzo sarkastyczna: "Oczywiście UWIELBIAŁEM spać w pokoju bez ogrzewania", a analizator sentymentu uzna to za pozytywny sentyment, mimo że człowiek czytający to wiedziałby, że to sarkazm.
NLTK dostarcza różne analizatory sentymentu, z którymi można eksperymentować, i możesz je zamieniać, aby sprawdzić, czy analiza sentymentu jest bardziej lub mniej dokładna. W tym przypadku używana jest analiza sentymentu VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, czerwiec 2014.

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

Później, w swoim programie, gdy będziesz gotowy do obliczenia sentymentu, możesz zastosować go do każdej recenzji w następujący sposób:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Na moim komputerze zajmuje to około 120 sekund, ale czas ten może się różnić w zależności od urządzenia. Jeśli chcesz wydrukować wyniki i sprawdzić, czy sentyment odpowiada recenzji:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Ostatnią rzeczą, którą należy zrobić z plikiem przed użyciem go w wyzwaniu, jest zapisanie go! Warto również rozważyć uporządkowanie wszystkich nowych kolumn w taki sposób, aby były łatwe do obsługi (dla człowieka to zmiana kosmetyczna).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Powinieneś uruchomić cały kod z [notatnika analizy](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (po uruchomieniu [notatnika filtrowania](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), aby wygenerować plik Hotel_Reviews_Filtered.csv).

Podsumowując, kroki są następujące:

1. Oryginalny plik danych **Hotel_Reviews.csv** został przeanalizowany w poprzedniej lekcji za pomocą [notatnika eksploracyjnego](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv został przefiltrowany za pomocą [notatnika filtrowania](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), co dało wynikowy plik **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv został przetworzony za pomocą [notatnika analizy sentymentu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), co dało wynikowy plik **Hotel_Reviews_NLP.csv**
4. Użyj Hotel_Reviews_NLP.csv w poniższym wyzwaniu NLP

### Podsumowanie

Na początku miałeś zestaw danych z kolumnami i danymi, ale nie wszystkie z nich mogły być zweryfikowane lub użyte. Przeanalizowałeś dane, odfiltrowałeś to, czego nie potrzebujesz, przekształciłeś tagi w coś użytecznego, obliczyłeś własne średnie, dodałeś kolumny sentymentu i, miejmy nadzieję, nauczyłeś się kilku ciekawych rzeczy o przetwarzaniu tekstu naturalnego.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Wyzwanie

Teraz, gdy przeanalizowałeś swój zestaw danych pod kątem sentymentu, sprawdź, czy możesz wykorzystać strategie, których nauczyłeś się w tym kursie (np. klasteryzację?), aby określić wzorce związane z sentymentem.

## Przegląd i samodzielna nauka

Skorzystaj z [tego modułu Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), aby dowiedzieć się więcej i użyć różnych narzędzi do eksploracji sentymentu w tekście.

## Zadanie

[Spróbuj innego zestawu danych](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.