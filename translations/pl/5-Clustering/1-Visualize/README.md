<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T08:16:39+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "pl"
}
-->
# Wprowadzenie do klasteryzacji

Klasteryzacja to rodzaj [uczenia nienadzorowanego](https://wikipedia.org/wiki/Uczenie_nienadzorowane), który zakłada, że zbiór danych jest nieoznaczony lub że jego dane wejściowe nie są powiązane z wcześniej zdefiniowanymi wynikami. Wykorzystuje różne algorytmy do analizy nieoznaczonych danych i tworzenia grup na podstawie wzorców wykrytych w danych.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć wideo. Podczas nauki o klasteryzacji w uczeniu maszynowym, posłuchaj nigeryjskich utworów Dance Hall - to wysoko oceniana piosenka z 2014 roku autorstwa PSquare.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

### Wprowadzenie

[Klasteryzacja](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) jest bardzo przydatna w eksploracji danych. Zobaczmy, czy może pomóc w odkrywaniu trendów i wzorców w sposobie, w jaki nigeryjscy odbiorcy konsumują muzykę.

✅ Zastanów się przez chwilę nad zastosowaniami klasteryzacji. W codziennym życiu klasteryzacja ma miejsce, gdy masz stos prania i musisz posortować ubrania członków rodziny 🧦👕👖🩲. W data science klasteryzacja występuje podczas analizy preferencji użytkownika lub określania cech dowolnego nieoznaczonego zbioru danych. Klasteryzacja w pewnym sensie pomaga uporządkować chaos, jak w przypadku szuflady na skarpetki.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć wideo: John Guttag z MIT wprowadza klasteryzację.

W środowisku zawodowym klasteryzacja może być używana do określania takich rzeczy jak segmentacja rynku, np. ustalanie, jakie grupy wiekowe kupują jakie produkty. Innym zastosowaniem może być wykrywanie anomalii, np. w celu wykrycia oszustw w zbiorze danych o transakcjach kartami kredytowymi. Możesz również użyć klasteryzacji do identyfikacji guzów w serii skanów medycznych.

✅ Zastanów się przez chwilę, jak mogłeś spotkać się z klasteryzacją „w terenie”, w bankowości, e-commerce lub biznesie.

> 🎓 Co ciekawe, analiza klastrów wywodzi się z dziedzin antropologii i psychologii w latach 30. XX wieku. Wyobraź sobie, jak mogła być wtedy używana.

Alternatywnie, możesz użyć jej do grupowania wyników wyszukiwania - na przykład według linków zakupowych, obrazów lub recenzji. Klasteryzacja jest przydatna, gdy masz duży zbiór danych, który chcesz zredukować i na którym chcesz przeprowadzić bardziej szczegółową analizę, więc technika ta może być używana do poznania danych przed skonstruowaniem innych modeli.

✅ Gdy dane są zorganizowane w klastry, przypisujesz im identyfikator klastra. Ta technika może być przydatna przy zachowaniu prywatności zbioru danych; zamiast odnosić się do punktu danych za pomocą bardziej ujawniających danych identyfikacyjnych, możesz odwoływać się do niego za pomocą identyfikatora klastra. Czy możesz wymyślić inne powody, dla których warto odwoływać się do identyfikatora klastra zamiast innych elementów klastra, aby go zidentyfikować?

Pogłęb swoją wiedzę na temat technik klasteryzacji w tym [module nauki](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Rozpoczęcie pracy z klasteryzacją

[Scikit-learn oferuje szeroki wachlarz](https://scikit-learn.org/stable/modules/clustering.html) metod do przeprowadzania klasteryzacji. Typ, który wybierzesz, będzie zależał od Twojego przypadku użycia. Według dokumentacji każda metoda ma różne zalety. Oto uproszczona tabela metod obsługiwanych przez Scikit-learn i ich odpowiednich zastosowań:

| Nazwa metody                 | Zastosowanie                                                          |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | ogólne zastosowanie, indukcyjne                                       |
| Affinity propagation         | wiele, nierówne klastry, indukcyjne                                   |
| Mean-shift                   | wiele, nierówne klastry, indukcyjne                                   |
| Spectral clustering          | kilka, równe klastry, transdukcyjne                                  |
| Ward hierarchical clustering | wiele, ograniczone klastry, transdukcyjne                            |
| Agglomerative clustering     | wiele, ograniczone, odległości nieeuklidesowe, transdukcyjne          |
| DBSCAN                       | geometria niepłaska, nierówne klastry, transdukcyjne                  |
| OPTICS                       | geometria niepłaska, nierówne klastry o zmiennej gęstości, transdukcyjne |
| Gaussian mixtures            | geometria płaska, indukcyjne                                         |
| BIRCH                        | duży zbiór danych z wartościami odstającymi, indukcyjne              |

> 🎓 Sposób, w jaki tworzymy klastry, ma wiele wspólnego z tym, jak grupujemy punkty danych w grupy. Rozpakujmy trochę terminologię:
>
> 🎓 ['Transdukcyjne' vs. 'indukcyjne'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Wnioskowanie transdukcyjne opiera się na zaobserwowanych przypadkach treningowych, które są mapowane na konkretne przypadki testowe. Wnioskowanie indukcyjne opiera się na przypadkach treningowych, które są mapowane na ogólne reguły, które dopiero potem są stosowane do przypadków testowych.
> 
> Przykład: Wyobraź sobie, że masz zbiór danych, który jest tylko częściowo oznaczony. Niektóre rzeczy to „płyty”, inne „CD”, a niektóre są puste. Twoim zadaniem jest przypisanie etykiet do pustych danych. Jeśli wybierzesz podejście indukcyjne, wytrenujesz model szukający „płyt” i „CD” i zastosujesz te etykiety do nieoznaczonych danych. Podejście to będzie miało trudności z klasyfikacją rzeczy, które są faktycznie „kasetami”. Podejście transdukcyjne, z drugiej strony, radzi sobie z tymi nieznanymi danymi bardziej efektywnie, ponieważ działa na grupowaniu podobnych elementów razem, a następnie przypisuje etykietę do grupy. W tym przypadku klastry mogą odzwierciedlać „okrągłe muzyczne rzeczy” i „kwadratowe muzyczne rzeczy”.
> 
> 🎓 ['Geometria niepłaska' vs. 'płaska'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Wywodzące się z terminologii matematycznej, geometria niepłaska vs. płaska odnosi się do pomiaru odległości między punktami za pomocą metod geometrycznych „płaskich” ([euklidesowych](https://wikipedia.org/wiki/Geometria_euklidesowa)) lub „niepłaskich” (nieeuklidesowych).
>
>'Płaska' w tym kontekście odnosi się do geometrii euklidesowej (części której są nauczane jako „geometria płaszczyzny”), a 'niepłaska' odnosi się do geometrii nieeuklidesowej. Co geometria ma wspólnego z uczeniem maszynowym? Cóż, jako dwie dziedziny zakorzenione w matematyce, musi istnieć wspólny sposób mierzenia odległości między punktami w klastrach, a to można zrobić w sposób „płaski” lub „niepłaski”, w zależności od charakteru danych. [Odległości euklidesowe](https://wikipedia.org/wiki/Odległość_euklidesowa) są mierzone jako długość odcinka między dwoma punktami. [Odległości nieeuklidesowe](https://wikipedia.org/wiki/Geometria_nieeuklidesowa) są mierzone wzdłuż krzywej. Jeśli Twoje dane, wizualizowane, wydają się nie istnieć na płaszczyźnie, możesz potrzebować specjalistycznego algorytmu do ich obsługi.
>
![Flat vs Nonflat Geometry Infographic](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografika autorstwa [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Odległości'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Klastry są definiowane przez ich macierz odległości, np. odległości między punktami. Odległość tę można mierzyć na kilka sposobów. Klastry euklidesowe są definiowane przez średnią wartości punktów i zawierają 'centroid', czyli punkt centralny. Odległości są więc mierzone względem tego centroidu. Odległości nieeuklidesowe odnoszą się do 'clustroidów', punktu najbliższego innym punktom. Clustroidy z kolei mogą być definiowane na różne sposoby.
> 
> 🎓 ['Ograniczone'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Ograniczona klasteryzacja](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) wprowadza 'uczenie półnadzorowane' do tej metody nienadzorowanej. Relacje między punktami są oznaczane jako 'nie można połączyć' lub 'muszą być połączone', więc na zbiór danych nakładane są pewne reguły.
>
>Przykład: Jeśli algorytm zostanie uwolniony na partii nieoznaczonych lub półoznaczonych danych, klastry, które produkuje, mogą być niskiej jakości. W powyższym przykładzie klastry mogą grupować „okrągłe muzyczne rzeczy”, „kwadratowe muzyczne rzeczy”, „trójkątne rzeczy” i „ciastka”. Jeśli zostaną wprowadzone pewne ograniczenia lub reguły do przestrzegania („przedmiot musi być wykonany z plastiku”, „przedmiot musi być w stanie produkować muzykę”), może to pomóc „ograniczyć” algorytm do podejmowania lepszych decyzji.
> 
> 🎓 'Gęstość'
> 
> Dane, które są „szumne”, są uważane za „gęste”. Odległości między punktami w każdym z jego klastrów mogą okazać się, po zbadaniu, bardziej lub mniej gęste, czyli „zatłoczone”, i dlatego te dane muszą być analizowane za pomocą odpowiedniej metody klasteryzacji. [Ten artykuł](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) pokazuje różnicę między użyciem klasteryzacji K-Means a algorytmami HDBSCAN do eksploracji szumnego zbioru danych o nierównej gęstości klastrów.

## Algorytmy klasteryzacji

Istnieje ponad 100 algorytmów klasteryzacji, a ich zastosowanie zależy od charakteru danych. Omówmy niektóre z najważniejszych:

- **Klasteryzacja hierarchiczna**. Jeśli obiekt jest klasyfikowany na podstawie swojej bliskości do pobliskiego obiektu, a nie do bardziej odległego, klastry są tworzone na podstawie odległości ich członków od innych obiektów. Klasteryzacja aglomeracyjna w Scikit-learn jest hierarchiczna.

   ![Hierarchical clustering Infographic](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografika autorstwa [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Klasteryzacja centroidowa**. Ten popularny algorytm wymaga wyboru 'k', czyli liczby klastrów do utworzenia, po czym algorytm określa punkt centralny klastra i gromadzi dane wokół tego punktu. [Klasteryzacja K-means](https://wikipedia.org/wiki/K-means_clustering) jest popularną wersją klasteryzacji centroidowej. Centrum jest określane przez najbliższą średnią, stąd nazwa. Kwadratowa odległość od klastra jest minimalizowana.

   ![Centroid clustering Infographic](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografika autorstwa [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Klasteryzacja oparta na rozkładzie**. Opierając się na modelowaniu statystycznym, klasteryzacja oparta na rozkładzie koncentruje się na określeniu prawdopodobieństwa, że punkt danych należy do klastra, i przypisaniu go odpowiednio. Metody mieszanki Gaussa należą do tego typu.

- **Klasteryzacja oparta na gęstości**. Punkty danych są przypisywane do klastrów na podstawie ich gęstości, czyli ich grupowania wokół siebie. Punkty danych oddalone od grupy są uważane za wartości odstające lub szum. DBSCAN, Mean-shift i OPTICS należą do tego typu klasteryzacji.

- **Klasteryzacja oparta na siatce**. Dla wielowymiarowych zbiorów danych tworzona jest siatka, a dane są dzielone między komórki siatki, tworząc w ten sposób klastry.

## Ćwiczenie - klasteryzuj swoje dane

Klasteryzacja jako technika jest bardzo wspomagana przez odpowiednią wizualizację, więc zacznijmy od wizualizacji naszych danych muzycznych. To ćwiczenie pomoże nam zdecydować, którą z metod klasteryzacji powinniśmy najskuteczniej zastosować do charakteru tych danych.

1. Otwórz plik [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) w tym folderze.

1. Zaimportuj pakiet `Seaborn` dla dobrej wizualizacji danych.

    ```python
    !pip install seaborn
    ```

1. Dodaj dane o piosenkach z pliku [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Załaduj dataframe z danymi o piosenkach. Przygotuj się do eksploracji tych danych, importując biblioteki i wyświetlając dane:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Sprawdź pierwsze kilka wierszy danych:

    |     | nazwa                    | album                        | artysta             | główny gatunek artysty | data wydania | długość | popularność | taneczność   | akustyczność | energia | instrumentalność | żywotność | głośność | mówność     | tempo   | podpis czasowy |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b       | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop                | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Uzyskaj informacje o dataframe, wywołując `info()`:

    ```python
    df.info()
    ```

   Wynik wygląda następująco:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Sprawdź ponownie, czy są wartości null, wywołując `isnull()` i upewniając się, że suma wynosi 0:

    ```python
    df.isnull().sum()
    ```

    Wygląda dobrze:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Opisz dane:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Jeśli pracujemy z klastrowaniem, metodą nienadzorowaną, która nie wymaga danych z etykietami, dlaczego pokazujemy te dane z etykietami? W fazie eksploracji danych są one przydatne, ale nie są konieczne do działania algorytmów klastrowania. Można równie dobrze usunąć nagłówki kolumn i odwoływać się do danych według numeru kolumny.

Spójrz na ogólne wartości danych. Zauważ, że popularność może wynosić '0', co oznacza utwory, które nie mają rankingu. Usuńmy je wkrótce.

1. Użyj wykresu słupkowego, aby dowiedzieć się, które gatunki są najpopularniejsze:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Jeśli chcesz zobaczyć więcej wartości, zmień top `[:5]` na większą wartość lub usuń go, aby zobaczyć wszystko.

Zauważ, że gdy najpopularniejszy gatunek jest opisany jako 'Missing', oznacza to, że Spotify go nie sklasyfikował, więc usuńmy go.

1. Usuń brakujące dane, filtrując je:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Teraz sprawdź ponownie gatunki:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Trzy najpopularniejsze gatunki zdecydowanie dominują w tym zbiorze danych. Skoncentrujmy się na `afro dancehall`, `afropop` i `nigerian pop`, dodatkowo filtrując zbiór danych, aby usunąć wszystko z wartością popularności 0 (co oznacza, że nie zostało sklasyfikowane jako popularne w zbiorze danych i może być uznane za szum w naszych celach):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Przeprowadź szybki test, aby sprawdzić, czy dane korelują w szczególnie silny sposób:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Jedyną silną korelacją jest związek między `energy` a `loudness`, co nie jest zaskakujące, biorąc pod uwagę, że głośna muzyka jest zwykle dość energetyczna. Poza tym korelacje są stosunkowo słabe. Ciekawe będzie zobaczenie, co algorytm klastrowania może zrobić z tymi danymi.

    > 🎓 Pamiętaj, że korelacja nie oznacza przyczynowości! Mamy dowód korelacji, ale brak dowodu przyczynowości. [Zabawna strona internetowa](https://tylervigen.com/spurious-correlations) zawiera wizualizacje, które podkreślają ten punkt.

Czy w tym zbiorze danych występuje zbieżność wokół postrzeganej popularności utworu i jego taneczności? FacetGrid pokazuje, że istnieją koncentryczne okręgi, które się pokrywają, niezależnie od gatunku. Czy może być tak, że gusta nigeryjskie koncentrują się na pewnym poziomie taneczności dla tego gatunku?  

✅ Wypróbuj różne punkty danych (energy, loudness, speechiness) i więcej lub inne gatunki muzyczne. Co możesz odkryć? Spójrz na tabelę `df.describe()`, aby zobaczyć ogólny rozkład punktów danych.

### Ćwiczenie - rozkład danych

Czy te trzy gatunki różnią się znacząco w postrzeganiu ich taneczności, w zależności od ich popularności?

1. Zbadaj rozkład danych dla popularności i taneczności w naszych trzech najpopularniejszych gatunkach wzdłuż osi x i y.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Możesz odkryć koncentryczne okręgi wokół ogólnego punktu zbieżności, pokazujące rozkład punktów.

    > 🎓 Zauważ, że ten przykład używa wykresu KDE (Kernel Density Estimate), który reprezentuje dane za pomocą ciągłej krzywej gęstości prawdopodobieństwa. Pozwala to interpretować dane podczas pracy z wieloma rozkładami.

    Ogólnie rzecz biorąc, trzy gatunki luźno się pokrywają pod względem ich popularności i taneczności. Określenie klastrów w tych luźno powiązanych danych będzie wyzwaniem:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Utwórz wykres punktowy:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Wykres punktowy dla tych samych osi pokazuje podobny wzór zbieżności

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Ogólnie rzecz biorąc, w przypadku klastrowania można używać wykresów punktowych do pokazywania klastrów danych, więc opanowanie tego typu wizualizacji jest bardzo przydatne. W następnej lekcji weźmiemy te przefiltrowane dane i użyjemy klastrowania k-średnich, aby odkryć grupy w tych danych, które wydają się nakładać w interesujący sposób.

---

## 🚀Wyzwanie

W ramach przygotowania do następnej lekcji, stwórz wykres dotyczący różnych algorytmów klastrowania, które możesz odkryć i użyć w środowisku produkcyjnym. Jakie problemy próbuje rozwiązać klastrowanie?

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Zanim zastosujesz algorytmy klastrowania, jak się nauczyliśmy, warto zrozumieć naturę swojego zbioru danych. Przeczytaj więcej na ten temat [tutaj](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ten pomocny artykuł](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) przeprowadza Cię przez różne sposoby działania algorytmów klastrowania, biorąc pod uwagę różne kształty danych.

## Zadanie

[Zbadaj inne wizualizacje dla klastrowania](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.