# Zacznij z Pythonem i Scikit-learn dla modeli regresji

![Podsumowanie regresji w formie notatki szkicowej](../../../../translated_images/pl/ml-regression.4e4f70e3b3ed446e.webp)

> Notatka szkicowa autorstwa [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ta lekcja jest dostępna również w R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Wprowadzenie

W tych czterech lekcjach odkryjesz, jak budować modele regresyjne. Wkrótce omówimy, do czego one służą. Ale zanim zaczniesz cokolwiek robić, upewnij się, że masz właściwe narzędzia, by rozpocząć proces!

W tej lekcji nauczysz się jak:

- Skonfigurować komputer do lokalnych zadań uczenia maszynowego.
- Pracować z Jupyter Notebooks.
- Korzystać ze Scikit-learn, w tym instalacji.
- Poznać regresję liniową przez ćwiczenie praktyczne.

## Instalacje i konfiguracje

[![ML dla początkujących - Przygotuj swoje narzędzia do budowania modeli uczenia maszynowego](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML dla początkujących - Przygotuj swoje narzędzia do budowania modeli uczenia maszynowego")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film pokazujący konfigurację komputera do ML.

1. **Zainstaluj Pythona**. Upewnij się, że [Python](https://www.python.org/downloads/) jest zainstalowany na twoim komputerze. Będziesz używać Pythona do wielu zadań związanych z data science i uczeniem maszynowym. Większość systemów komputerowych ma już zainstalowanego Pythona. Są też dostępne przydatne [pakiety do kodowania w Pythonie](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott), które ułatwiają instalację niektórym użytkownikom.

   Niektóre zastosowania Pythona wymagają jednak różnych wersji oprogramowania. Dlatego korzystne jest pracowanie w ramach [wirtualnego środowiska](https://docs.python.org/3/library/venv.html).

2. **Zainstaluj Visual Studio Code**. Upewnij się, że masz zainstalowany Visual Studio Code na swoim komputerze. Postępuj według tych instrukcji, by [zainstalować Visual Studio Code](https://code.visualstudio.com/) - podstawowa instalacja. W tym kursie będziesz używać Pythona w Visual Studio Code, więc warto nauczyć się, jak [konfigurować Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) do pracy z Pythonem.

   > Zapoznaj się z Pythonem, pracując z tym zbiorem [modułów Learn](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Konfiguracja Pythona z Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Konfiguracja Pythona z Visual Studio Code")
   >
   > 🎥 Kliknij powyższy obraz, by obejrzeć film: użycie Pythona w VS Code.

3. **Zainstaluj Scikit-learn**, postępując zgodnie z [tym instrukcjami](https://scikit-learn.org/stable/install.html). Ponieważ musisz używać Pythona 3, zalecane jest korzystanie z wirtualnego środowiska. Jeśli instalujesz tę bibliotekę na Macu M1, na powyższej stronie są specjalne wskazówki.

1. **Zainstaluj Jupyter Notebook**. Będziesz musiał [zainstalować paczkę Jupyter](https://pypi.org/project/jupyter/).

## Twoje środowisko do tworzenia ML

Będziesz korzystać z **notebooków** do tworzenia kodu w Pythonie i budowania modeli uczenia maszynowego. Ten typ pliku jest powszechnym narzędziem dla data scientistów i można go rozpoznać po rozszerzeniu `.ipynb`.

Notebooki to interaktywne środowiska, które pozwalają programiście kodować oraz dodawać notatki i dokumentację do kodu, co jest bardzo pomocne w projektach eksperymentalnych lub badawczych.

[![ML dla początkujących - Skonfiguruj Jupyter Notebooks, by zacząć budować modele regresji](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML dla początkujących - Skonfiguruj Jupyter Notebooks, by zacząć budować modele regresji")

> 🎥 Kliknij powyższy obraz, by obejrzeć krótki film z tym ćwiczeniem.

### Ćwiczenie - praca z notebookiem

W tym folderze znajdziesz plik _notebook.ipynb_.

1. Otwórz _notebook.ipynb_ w Visual Studio Code.

   Uruchomi się serwer Jupyter z Pythonem 3+. Znajdziesz obszary notebooka, które możesz `uruchomić`, fragmenty kodu. Możesz wykonać blok kodu, wybierając ikonę przypominającą przycisk „play”.

1. Wybierz ikonę `md` i dodaj trochę markdownu, wstawiając tekst **# Witaj w swoim notebooku**.

   Następnie dodaj trochę kodu Pythona.

1. Wpisz **print('hello notebook')** w bloku kodu.
1. Wybierz strzałkę, aby uruchomić kod.

   Powinieneś zobaczyć wydrukowane zdanie:

    ```output
    hello notebook
    ```

![VS Code z otwartym notebookiem](../../../../translated_images/pl/notebook.4a3ee31f396b8832.webp)

Możesz przeplatać swój kod komentarzami, by samodokumentować notebook.

✅ Pomyśl przez chwilę, jak bardzo różni się środowisko pracy programisty webowego od środowiska data scientist.

## Start ze Scikit-learn

Teraz, gdy Python jest skonfigurowany w twoim lokalnym środowisku, a ty czujesz się komfortowo z Jupyter Notebooks, poznajmy równie dobrze Scikit-learn (wymowa `sci` jak w „science”). Scikit-learn oferuje [obszerny API](https://scikit-learn.org/stable/modules/classes.html#api-ref), które pomaga wykonać zadania ML.

Według ich [strony internetowej](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn to biblioteka open source do uczenia maszynowego wspierająca uczenie nadzorowane i nienadzorowane. Dostarcza także narzędzia do dopasowywania modeli, przetwarzania danych, wyboru i oceny modeli oraz wiele innych użyteczności."

W tym kursie będziesz używać Scikit-learn i innych narzędzi do budowania modeli uczenia maszynowego w tzw. „tradycyjnych” zadaniach uczenia maszynowego. Świadomie uniknęliśmy sieci neuronowych i głębokiego uczenia, ponieważ są one lepiej omówione w naszym nadchodzącym kursie „AI dla początkujących”.

Scikit-learn ułatwia budowanie modeli i ich ewaluację. Skupia się głównie na danych liczbowych i zawiera kilka gotowych zestawów danych do użytku edukacyjnego. Obejmuje też wbudowane modele do przetestowania przez uczniów. Poznajmy proces ładowania gotowych danych i używania wbudowanego estymatora, aby stworzyć Twój pierwszy model ML w Scikit-learn na podstawie podstawowych danych.

## Ćwiczenie - twój pierwszy notebook Scikit-learn

> Ten samouczek został zainspirowany [przykładem regresji liniowej](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) na stronie Scikit-learn.


[![ML dla początkujących - Twój pierwszy projekt regresji liniowej w Pythonie](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML dla początkujących - Twój pierwszy projekt regresji liniowej w Pythonie")

> 🎥 Kliknij powyższy obraz, aby obejrzeć krótki film z tym ćwiczeniem.

W pliku _notebook.ipynb_ powiązanym z tą lekcją, wyczyść wszystkie komórki, klikając ikonę „kosza”.

W tej sekcji będziesz pracować z małym zestawem danych o cukrzycy, który jest wbudowany w Scikit-learn do celów edukacyjnych. Wyobraź sobie, że chcesz przetestować leczenie dla pacjentów cukrzycowych. Modele uczenia maszynowego mogą pomóc określić, którzy pacjenci zareagują lepiej na terapię, na podstawie kombinacji zmiennych. Nawet bardzo podstawowy model regresji, gdy jest wizualizowany, może pokazać informacje o zmiennych, które pomogłyby Ci zaplanować teoretyczne badania kliniczne.

✅ Istnieje wiele metod regresji, a której użyjesz zależy od pytania, na które chcesz odpowiedzieć. Jeśli chcesz przewidzieć prawdopodobny wzrost osoby w danym wieku, użyjesz regresji liniowej, bo szukasz **wartości liczbowej**. Jeśli chcesz sprawdzić, czy dana kuchnia powinna być uznana za wegańską, interesuje Cię **przypisanie do kategorii**, więc użyjesz regresji logistycznej. O regresji logistycznej dowiesz się więcej później. Pomyśl chwilę o pytaniach, które możesz zadać danym, i która z tych metod byłaby bardziej odpowiednia.

Zaczynajmy zadanie.

### Import bibliotek

Do tego zadania zaimportujemy kilka bibliotek:

- **matplotlib**. To przydatne [narzędzie do tworzenia wykresów](https://matplotlib.org/). Użyjemy go do stworzenia wykresu liniowego.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) to przydatna biblioteka do obsługi danych numerycznych w Pythonie.
- **sklearn**. To biblioteka [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Zaimportuj biblioteki, które pomogą Ci w zadaniach.

1. Dodaj importy, wpisując następujący kod:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Powyżej importujesz `matplotlib`, `numpy` oraz importujesz `datasets`, `linear_model` i `model_selection` z `sklearn`. `model_selection` służy do dzielenia danych na zestawy treningowe i testowe.

### Zbiór danych o cukrzycy

Wbudowany [zbiór danych o cukrzycy](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) zawiera 442 próbki danych dotyczących cukrzycy, z 10 zmiennymi cech, z których niektóre to:

- wiek: wiek w latach
- bmi: wskaźnik masy ciała
- bp: średnie ciśnienie krwi
- s1 tc: komórki T (rodzaj białych krwinek)

✅ Ten zbiór danych zawiera koncepcję „płci” jako zmiennej cechowej ważnej w badaniach nad cukrzycą. Wiele medycznych zbiorów zawiera tego rodzaju binarne klasyfikacje. Pomyśl trochę, jak takie kategoryzacje mogą wykluczać części populacji z leczenia.

Teraz załaduj dane X i y.

> 🎓 Pamiętaj, to jest uczenie nadzorowane i potrzebujemy nazwanego celu 'y'.

W nowej komórce kodu załaduj zbiór danych o cukrzycy, wywołując `load_diabetes()`. Parametr `return_X_y=True` oznacza, że `X` będzie macierzą danych, a `y` celem regresji.

1. Dodaj kilka poleceń print, aby pokazać kształt macierzy danych i jej pierwszy element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    To co otrzymujesz w odpowiedzi, to krotka. Przypisujesz dwie pierwsze wartości krotki do zmiennych `X` i `y`. Dowiedz się więcej [o krotkach](https://wikipedia.org/wiki/Tuple).

    Widzisz, że dane zawierają 442 elementy ułożone w tablice o 10 elementach:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pomyśl trochę o relacji między danymi a celem regresji. Regresja liniowa przewiduje zależności między cechą X a zmienną celu y. Czy potrafisz znaleźć [cel](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) dla zbioru danych o cukrzycy w dokumentacji? Co ten zbiór danych demonstruje, biorąc pod uwagę cel?

2. Następnie wybierz część tego zbioru, którą wyświetlisz, wybierając 3 kolumnę zbioru. Zrób to, używając operatora `:`, by wybrać wszystkie wiersze, a potem wybierz trzecią kolumnę za pomocą indeksu (2). Możesz także przekształcić dane do tablicy 2D - jak wymagane do wykresu - używając `reshape(n_rows, n_columns)`. Jeśli jeden z parametrów to -1, odpowiedni wymiar jest wyliczany automatycznie.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ W każdej chwili wypisz dane, aby sprawdzić ich kształt.

3. Teraz, gdy masz dane gotowe do wykresu, zobacz, czy maszyna może pomóc określić logiczny podział danych w zbiorze. Aby to zrobić, musisz podzielić dane (X) i cel (y) na zestawy testowe i treningowe. Scikit-learn oferuje prosty sposób na to; możesz podzielić swoje dane testowe w określonym miejscu.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Teraz jesteś gotów do trenowania modelu! Wczytaj model regresji liniowej i wytrenuj go na swoich zestawach treningowych X i y, używając `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` to funkcja, którą zobaczysz w wielu bibliotekach ML, jak TensorFlow

5. Następnie stwórz predykcję używając danych testowych, korzystając z funkcji `predict()`. Posłuży ona do narysowania linii pomiędzy grupami danych

    ```python
    y_pred = model.predict(X_test)
    ```

6. Czas pokazać dane na wykresie. Matplotlib to bardzo przydatne narzędzie do tego zadania. Stwórz wykres rozrzutu wszystkich danych testowych X i y, a następnie użyj predykcji, aby narysować linię w najbardziej odpowiednim miejscu, pomiędzy grupowaniami danych modelu.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![wykres rozrzutu pokazujący punkty danych o cukrzycy](../../../../translated_images/pl/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Pomyśl chwilę, co tu się dzieje. Prosta linia przechodzi przez wiele małych punktów danych, ale co ona właściwie robi? Czy widzisz, jak możesz użyć tej linii, by przewidzieć, gdzie powinien pasować nowy, niewidziany punkt danych w relacji do osi y wykresu? Spróbuj ubrać w słowa praktyczne zastosowanie tego modelu.

Gratulacje, stworzyłeś pierwszy model regresji liniowej, wygenerowałeś prognozę i wyświetliłeś ją na wykresie!

---
## 🚀Wyzwanie

Narysuj wykres innej zmiennej z tego zbioru danych. Podpowiedź: edytuj tę linię: `X = X[:,2]`. Biorąc pod uwagę cel tego zbioru, co możesz odkryć na temat postępu cukrzycy jako choroby?
## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Powtórka & Samodzielna nauka

W tym tutorialu pracowałeś z prostą regresją liniową, a nie regresją jednowymiarową lub wieloraką. Przeczytaj trochę o różnicach między tymi metodami lub obejrzyj [ten film](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Przeczytaj więcej na temat koncepcji regresji i zastanów się, jakie pytania można odpowiedzieć za pomocą tej techniki. Skorzystaj z tego [samouczka](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott), aby pogłębić swoją wiedzę.

## Zadanie

[Inny zbiór danych](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Choć dążymy do dokładności, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w jego języku źródłowym należy uznawać za autorytatywne źródło. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->