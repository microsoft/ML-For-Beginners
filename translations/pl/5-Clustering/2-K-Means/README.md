<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T08:17:54+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "pl"
}
-->
# Klasteryzacja metodą K-Means

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

W tej lekcji nauczysz się tworzyć klastry za pomocą biblioteki Scikit-learn oraz nigeryjskiego zestawu danych muzycznych, który zaimportowałeś wcześniej. Omówimy podstawy metody K-Means dla klasteryzacji. Pamiętaj, że jak nauczyłeś się w poprzedniej lekcji, istnieje wiele sposobów pracy z klastrami, a metoda, którą wybierzesz, zależy od Twoich danych. Spróbujemy metody K-Means, ponieważ jest to najczęściej stosowana technika klasteryzacji. Zaczynajmy!

Pojęcia, które poznasz:

- Ocena sylwetki (Silhouette scoring)
- Metoda łokcia (Elbow method)
- Inercja
- Wariancja

## Wprowadzenie

[Klasteryzacja metodą K-Means](https://wikipedia.org/wiki/K-means_clustering) to metoda wywodząca się z dziedziny przetwarzania sygnałów. Służy do dzielenia i grupowania danych w 'k' klastry za pomocą serii obserwacji. Każda obserwacja działa na rzecz grupowania danego punktu danych najbliżej jego 'średniej', czyli punktu centralnego klastra.

Klastry można wizualizować jako [diagramy Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), które zawierają punkt (lub 'nasiono') i odpowiadający mu obszar.

![diagram Voronoi](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper)

Proces klasteryzacji metodą K-Means [wykonuje się w trzech krokach](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algorytm wybiera k liczbę punktów centralnych, próbkując z zestawu danych. Następnie wykonuje pętlę:
    1. Przypisuje każdą próbkę do najbliższego centroidu.
    2. Tworzy nowe centroidy, obliczając średnią wartość wszystkich próbek przypisanych do poprzednich centroidów.
    3. Następnie oblicza różnicę między nowymi a starymi centroidami i powtarza proces, aż centroidy się ustabilizują.

Jednym z minusów metody K-Means jest konieczność ustalenia 'k', czyli liczby centroidów. Na szczęście metoda 'łokcia' pomaga oszacować dobrą wartość początkową dla 'k'. Zaraz ją wypróbujesz.

## Wymagania wstępne

Będziesz pracować w pliku [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb), który zawiera import danych i wstępne czyszczenie, które wykonałeś w poprzedniej lekcji.

## Ćwiczenie - przygotowanie

Zacznij od ponownego przyjrzenia się danym o piosenkach.

1. Utwórz wykres pudełkowy, wywołując `boxplot()` dla każdej kolumny:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    Dane są trochę hałaśliwe: obserwując każdą kolumnę jako wykres pudełkowy, możesz zauważyć wartości odstające.

    ![wartości odstające](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Możesz przejrzeć zestaw danych i usunąć te wartości odstające, ale to sprawiłoby, że dane byłyby dość ograniczone.

1. Na razie wybierz kolumny, które wykorzystasz w ćwiczeniu klasteryzacji. Wybierz te o podobnych zakresach i zakoduj kolumnę `artist_top_genre` jako dane numeryczne:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Teraz musisz wybrać, ile klastrów chcesz utworzyć. Wiesz, że w zestawie danych wyodrębniliśmy 3 gatunki muzyczne, więc spróbujmy z 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

Zobaczysz wydrukowaną tablicę z przewidywanymi klastrami (0, 1 lub 2) dla każdego wiersza ramki danych.

1. Użyj tej tablicy, aby obliczyć 'ocenę sylwetki':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Ocena sylwetki

Szukaj oceny sylwetki bliższej 1. Wynik ten waha się od -1 do 1, a jeśli wynik wynosi 1, klaster jest gęsty i dobrze oddzielony od innych klastrów. Wartość bliska 0 oznacza nakładające się klastry, w których próbki znajdują się bardzo blisko granicy decyzyjnej sąsiednich klastrów. [(Źródło)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Nasz wynik to **0.53**, czyli w samym środku. Wskazuje to, że nasze dane nie są szczególnie dobrze dopasowane do tego typu klasteryzacji, ale kontynuujmy.

### Ćwiczenie - budowa modelu

1. Zaimportuj `KMeans` i rozpocznij proces klasteryzacji.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Jest tu kilka elementów, które warto wyjaśnić.

    > 🎓 range: Są to iteracje procesu klasteryzacji.

    > 🎓 random_state: "Określa generowanie liczb losowych dla inicjalizacji centroidów." [Źródło](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "suma kwadratów wewnątrz klastra" mierzy średnią kwadratową odległość wszystkich punktów w klastrze od centroidu klastra. [Źródło](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inercja: Algorytmy K-Means próbują wybrać centroidy, aby zminimalizować 'inercję', "miarę spójności wewnętrznej klastrów." [Źródło](https://scikit-learn.org/stable/modules/clustering.html). Wartość jest dodawana do zmiennej wcss przy każdej iteracji.

    > 🎓 k-means++: W [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) możesz użyć optymalizacji 'k-means++', która "inicjalizuje centroidy tak, aby były (zazwyczaj) odległe od siebie, co prowadzi do prawdopodobnie lepszych wyników niż losowa inicjalizacja."

### Metoda łokcia

Wcześniej założyłeś, że ponieważ wyodrębniłeś 3 gatunki muzyczne, powinieneś wybrać 3 klastry. Ale czy na pewno?

1. Użyj metody 'łokcia', aby się upewnić.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Użyj zmiennej `wcss`, którą zbudowałeś w poprzednim kroku, aby stworzyć wykres pokazujący, gdzie znajduje się 'zgięcie' łokcia, co wskazuje optymalną liczbę klastrów. Może rzeczywiście jest to **3**!

    ![metoda łokcia](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Ćwiczenie - wyświetlanie klastrów

1. Spróbuj ponownie przeprowadzić proces, tym razem ustawiając trzy klastry, i wyświetl klastry jako wykres punktowy:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. Sprawdź dokładność modelu:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Dokładność tego modelu nie jest zbyt dobra, a kształt klastrów daje Ci wskazówkę dlaczego.

    ![klastry](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Dane są zbyt niezrównoważone, zbyt mało skorelowane, a między wartościami kolumn występuje zbyt duża wariancja, aby dobrze je sklasteryzować. W rzeczywistości klastry, które się tworzą, są prawdopodobnie mocno wpływane lub zniekształcone przez trzy kategorie gatunków, które zdefiniowaliśmy powyżej. To był proces nauki!

    W dokumentacji Scikit-learn możesz zobaczyć, że model taki jak ten, z klastrami niezbyt dobrze oddzielonymi, ma problem z 'wariancją':

    ![problematyczne modele](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografika z Scikit-learn

## Wariancja

Wariancja jest definiowana jako "średnia kwadratów różnic od średniej" [(Źródło)](https://www.mathsisfun.com/data/standard-deviation.html). W kontekście tego problemu klasteryzacji odnosi się do danych, w których liczby w naszym zestawie danych mają tendencję do zbytniego odchylenia od średniej.

✅ To świetny moment, aby pomyśleć o wszystkich sposobach, w jakie możesz rozwiązać ten problem. Dopracować dane? Użyć innych kolumn? Wypróbować inny algorytm? Podpowiedź: Spróbuj [skalować dane](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/), aby je znormalizować i przetestować inne kolumny.

> Wypróbuj ten '[kalkulator wariancji](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)', aby lepiej zrozumieć to pojęcie.

---

## 🚀Wyzwanie

Spędź trochę czasu z tym notebookiem, dopracowując parametry. Czy możesz poprawić dokładność modelu, bardziej oczyszczając dane (na przykład usuwając wartości odstające)? Możesz użyć wag, aby nadać większą wagę określonym próbkom danych. Co jeszcze możesz zrobić, aby stworzyć lepsze klastry?

Podpowiedź: Spróbuj skalować dane. W notebooku znajdziesz zakomentowany kod, który dodaje standardowe skalowanie, aby kolumny danych bardziej przypominały siebie nawzajem pod względem zakresu. Zauważysz, że chociaż ocena sylwetki spada, 'zgięcie' na wykresie łokcia wygładza się. Dzieje się tak, ponieważ pozostawienie danych nieskalowanych pozwala danym o mniejszej wariancji mieć większy wpływ. Przeczytaj więcej o tym problemie [tutaj](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Spójrz na symulator K-Means [taki jak ten](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Możesz użyć tego narzędzia, aby wizualizować próbki danych i określić ich centroidy. Możesz edytować losowość danych, liczbę klastrów i liczbę centroidów. Czy pomaga Ci to zrozumieć, jak dane mogą być grupowane?

Zobacz także [ten materiał o K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) ze Stanfordu.

## Zadanie

[Wypróbuj różne metody klasteryzacji](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.