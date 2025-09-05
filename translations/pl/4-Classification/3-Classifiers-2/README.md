<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T08:25:46+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "pl"
}
-->
# Klasyfikatory kuchni 2

W tej drugiej lekcji dotyczącej klasyfikacji poznasz więcej sposobów klasyfikowania danych numerycznych. Dowiesz się również, jakie są konsekwencje wyboru jednego klasyfikatora zamiast innego.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

### Wymagania wstępne

Zakładamy, że ukończyłeś poprzednie lekcje i masz wyczyszczony zbiór danych w folderze `data`, nazwany _cleaned_cuisines.csv_, znajdujący się w głównym katalogu tego czteroczęściowego kursu.

### Przygotowanie

Załadowaliśmy Twój plik _notebook.ipynb_ z wyczyszczonym zbiorem danych i podzieliliśmy go na ramki danych X i y, gotowe do procesu budowy modelu.

## Mapa klasyfikacji

Wcześniej nauczyłeś się o różnych opcjach klasyfikacji danych, korzystając z ściągi Microsoftu. Scikit-learn oferuje podobną, ale bardziej szczegółową ściągę, która może pomóc jeszcze bardziej zawęzić wybór estymatorów (inaczej klasyfikatorów):

![Mapa ML ze Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Wskazówka: [odwiedź tę mapę online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) i klikaj po ścieżkach, aby przeczytać dokumentację.

### Plan

Ta mapa jest bardzo pomocna, gdy masz jasne zrozumienie swoich danych, ponieważ możesz „przechodzić” jej ścieżkami, aby podjąć decyzję:

- Mamy >50 próbek
- Chcemy przewidzieć kategorię
- Mamy dane z etykietami
- Mamy mniej niż 100 tys. próbek
- ✨ Możemy wybrać Linear SVC
- Jeśli to nie zadziała, ponieważ mamy dane numeryczne:
    - Możemy spróbować ✨ KNeighbors Classifier 
      - Jeśli to nie zadziała, spróbuj ✨ SVC i ✨ Ensemble Classifiers

To bardzo pomocna ścieżka do naśladowania.

## Ćwiczenie - podziel dane

Podążając tą ścieżką, powinniśmy zacząć od zaimportowania potrzebnych bibliotek.

1. Zaimportuj potrzebne biblioteki:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Podziel dane na zestawy treningowe i testowe:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Klasyfikator Linear SVC

Support-Vector Clustering (SVC) to metoda z rodziny maszyn wektorów nośnych (Support-Vector Machines) w technikach uczenia maszynowego (więcej o nich poniżej). W tej metodzie możesz wybrać „jądro” (kernel), aby zdecydować, jak grupować etykiety. Parametr 'C' odnosi się do 'regularyzacji', która kontroluje wpływ parametrów. Jądro może być jednym z [kilku](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tutaj ustawiamy je na 'linear', aby wykorzystać liniowy SVC. Domyślnie prawdopodobieństwo jest ustawione na 'false'; tutaj ustawiamy je na 'true', aby uzyskać oszacowania prawdopodobieństwa. Parametr random state ustawiamy na '0', aby przetasować dane i uzyskać prawdopodobieństwa.

### Ćwiczenie - zastosuj Linear SVC

Zacznij od stworzenia tablicy klasyfikatorów. Będziesz stopniowo dodawać do tej tablicy, testując różne metody.

1. Zacznij od Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Wytrenuj model, używając Linear SVC, i wyświetl raport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Wynik jest całkiem dobry:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Klasyfikator K-Neighbors

K-Neighbors należy do rodziny metod ML „sąsiadów”, które mogą być używane zarówno w uczeniu nadzorowanym, jak i nienadzorowanym. W tej metodzie tworzona jest z góry określona liczba punktów, a dane są grupowane wokół tych punktów, aby można było przewidzieć ogólne etykiety dla danych.

### Ćwiczenie - zastosuj klasyfikator K-Neighbors

Poprzedni klasyfikator był dobry i dobrze działał z danymi, ale może uda się uzyskać lepszą dokładność. Spróbuj klasyfikatora K-Neighbors.

1. Dodaj linię do swojej tablicy klasyfikatorów (dodaj przecinek po elemencie Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Wynik jest trochę gorszy:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Dowiedz się więcej o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Klasyfikator Support Vector

Klasyfikatory Support-Vector należą do rodziny metod ML [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine), które są używane do zadań klasyfikacji i regresji. SVM „mapuje przykłady treningowe na punkty w przestrzeni”, aby zmaksymalizować odległość między dwiema kategoriami. Kolejne dane są mapowane w tej przestrzeni, aby przewidzieć ich kategorię.

### Ćwiczenie - zastosuj klasyfikator Support Vector

Spróbujmy uzyskać nieco lepszą dokładność za pomocą klasyfikatora Support Vector.

1. Dodaj przecinek po elemencie K-Neighbors, a następnie dodaj tę linię:

    ```python
    'SVC': SVC(),
    ```

    Wynik jest całkiem dobry!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Dowiedz się więcej o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Klasyfikatory zespołowe (Ensemble Classifiers)

Podążajmy ścieżką do samego końca, mimo że poprzedni test był całkiem dobry. Wypróbujmy kilka klasyfikatorów zespołowych, w szczególności Random Forest i AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Wynik jest bardzo dobry, szczególnie dla Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Dowiedz się więcej o [Klasyfikatorach zespołowych](https://scikit-learn.org/stable/modules/ensemble.html)

Ta metoda uczenia maszynowego „łączy przewidywania kilku podstawowych estymatorów”, aby poprawić jakość modelu. W naszym przykładzie użyliśmy Random Trees i AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda uśredniania, buduje „las” „drzew decyzyjnych” z elementami losowości, aby uniknąć przeuczenia. Parametr n_estimators określa liczbę drzew.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) dopasowuje klasyfikator do zbioru danych, a następnie dopasowuje kopie tego klasyfikatora do tego samego zbioru danych. Skupia się na wagach błędnie sklasyfikowanych elementów i dostosowuje dopasowanie kolejnego klasyfikatora, aby je poprawić.

---

## 🚀 Wyzwanie

Każda z tych technik ma dużą liczbę parametrów, które możesz dostosować. Zbadaj domyślne parametry każdej z nich i zastanów się, co zmiana tych parametrów oznaczałaby dla jakości modelu.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

W tych lekcjach pojawia się wiele żargonu, więc poświęć chwilę, aby przejrzeć [tę listę](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) przydatnej terminologii!

## Zadanie 

[Zabawa z parametrami](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji o krytycznym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.