<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T08:16:06+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "pl"
}
-->
# Prognozowanie szeregów czasowych za pomocą Support Vector Regressor

W poprzedniej lekcji nauczyłeś się, jak używać modelu ARIMA do prognozowania szeregów czasowych. Teraz przyjrzymy się modelowi Support Vector Regressor, który jest modelem regresji używanym do przewidywania danych ciągłych.

## [Quiz przed lekcją](https://ff-quizzes.netlify.app/en/ml/) 

## Wprowadzenie

W tej lekcji odkryjesz specyficzny sposób budowania modeli za pomocą [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) dla regresji, czyli **SVR: Support Vector Regressor**. 

### SVR w kontekście szeregów czasowych [^1]

Zanim zrozumiesz znaczenie SVR w prognozowaniu szeregów czasowych, oto kilka ważnych pojęć, które musisz znać:

- **Regresja:** Technika uczenia nadzorowanego służąca do przewidywania wartości ciągłych na podstawie zestawu danych wejściowych. Idea polega na dopasowaniu krzywej (lub linii) w przestrzeni cech, która obejmuje maksymalną liczbę punktów danych. [Kliknij tutaj](https://en.wikipedia.org/wiki/Regression_analysis), aby dowiedzieć się więcej.
- **Support Vector Machine (SVM):** Rodzaj nadzorowanego modelu uczenia maszynowego używanego do klasyfikacji, regresji i wykrywania wartości odstających. Model jest hiperpłaszczyzną w przestrzeni cech, która w przypadku klasyfikacji działa jako granica, a w przypadku regresji jako linia najlepszego dopasowania. W SVM funkcja jądra jest zazwyczaj używana do przekształcenia zestawu danych w przestrzeń o większej liczbie wymiarów, aby dane były łatwiej rozdzielne. [Kliknij tutaj](https://en.wikipedia.org/wiki/Support-vector_machine), aby dowiedzieć się więcej o SVM.
- **Support Vector Regressor (SVR):** Rodzaj SVM, który znajduje linię najlepszego dopasowania (która w przypadku SVM jest hiperpłaszczyzną) obejmującą maksymalną liczbę punktów danych.

### Dlaczego SVR? [^1]

W ostatniej lekcji nauczyłeś się o ARIMA, która jest bardzo skuteczną statystyczną metodą liniową do prognozowania danych szeregów czasowych. Jednak w wielu przypadkach dane szeregów czasowych mają *nieliniowość*, której modele liniowe nie mogą odwzorować. W takich przypadkach zdolność SVM do uwzględniania nieliniowości w danych w zadaniach regresji sprawia, że SVR jest skuteczny w prognozowaniu szeregów czasowych.

## Ćwiczenie - budowa modelu SVR

Pierwsze kroki przygotowania danych są takie same jak w poprzedniej lekcji dotyczącej [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Otwórz folder [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) w tej lekcji i znajdź plik [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Uruchom notebook i zaimportuj potrzebne biblioteki: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. Załaduj dane z pliku `/data/energy.csv` do dataframe Pandas i przejrzyj je: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Zobrazuj wszystkie dostępne dane dotyczące energii od stycznia 2012 do grudnia 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![pełne dane](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Teraz zbudujmy nasz model SVR.

### Tworzenie zbiorów danych treningowych i testowych

Po załadowaniu danych możesz podzielić je na zbiory treningowe i testowe. Następnie przekształcisz dane, aby stworzyć zestaw danych oparty na krokach czasowych, który będzie potrzebny dla SVR. Model zostanie wytrenowany na zbiorze treningowym. Po zakończeniu treningu ocenisz jego dokładność na zbiorze treningowym, testowym, a następnie na pełnym zestawie danych, aby zobaczyć ogólną wydajność. Musisz upewnić się, że zbiór testowy obejmuje późniejszy okres czasu niż zbiór treningowy, aby model nie uzyskał informacji z przyszłych okresów czasu [^2] (sytuacja znana jako *przeuczenie*).

1. Przypisz dwumiesięczny okres od 1 września do 31 października 2014 do zbioru treningowego. Zbiór testowy będzie obejmował dwumiesięczny okres od 1 listopada do 31 grudnia 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Zobrazuj różnice: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![dane treningowe i testowe](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Przygotowanie danych do treningu

Teraz musisz przygotować dane do treningu, wykonując filtrowanie i skalowanie danych. Przefiltruj zestaw danych, aby uwzględnić tylko potrzebne okresy czasu i kolumny, a także skalowanie, aby dane były przedstawione w przedziale 0,1.

1. Przefiltruj oryginalny zestaw danych, aby uwzględnić tylko wspomniane okresy czasu dla każdego zestawu oraz tylko potrzebną kolumnę 'load' i datę: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. Skaluj dane treningowe, aby były w zakresie (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Teraz skaluj dane testowe: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Tworzenie danych z krokami czasowymi [^1]

Dla SVR przekształcasz dane wejściowe w formę `[batch, timesteps]`. Przekształcasz istniejące `train_data` i `test_data`, tak aby powstał nowy wymiar odnoszący się do kroków czasowych. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

W tym przykładzie przyjmujemy `timesteps = 5`. Dane wejściowe dla modelu to dane z pierwszych 4 kroków czasowych, a dane wyjściowe to dane z 5. kroku czasowego.

```python
timesteps=5
```

Konwersja danych treningowych na tensor 2D za pomocą zagnieżdżonej listy:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Konwersja danych testowych na tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Wybór danych wejściowych i wyjściowych z danych treningowych i testowych:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### Implementacja SVR [^1]

Teraz czas na implementację SVR. Aby dowiedzieć się więcej o tej implementacji, możesz odnieść się do [tej dokumentacji](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). W naszej implementacji wykonujemy następujące kroki:

  1. Zdefiniuj model, wywołując `SVR()` i przekazując hiperparametry modelu: kernel, gamma, c i epsilon
  2. Przygotuj model do danych treningowych, wywołując funkcję `fit()`
  3. Dokonaj prognoz, wywołując funkcję `predict()`

Teraz tworzymy model SVR. Używamy [jądra RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) i ustawiamy hiperparametry gamma, C i epsilon na odpowiednio 0.5, 10 i 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Dopasowanie modelu do danych treningowych [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Prognozy modelu [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Zbudowałeś swój SVR! Teraz musimy go ocenić.

### Ocena modelu [^1]

Aby ocenić model, najpierw przeskalujemy dane z powrotem do oryginalnej skali. Następnie, aby sprawdzić wydajność, zobrazujemy oryginalny i prognozowany wykres szeregów czasowych oraz wydrukujemy wynik MAPE.

Skalowanie prognozowanych i oryginalnych danych wyjściowych:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### Sprawdzenie wydajności modelu na danych treningowych i testowych [^1]

Wyciągamy znaczniki czasu z zestawu danych, aby pokazać je na osi x naszego wykresu. Zauważ, że używamy pierwszych ```timesteps-1``` wartości jako danych wejściowych dla pierwszego wyniku, więc znaczniki czasu dla wyniku zaczną się po tym.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Zobrazowanie prognoz dla danych treningowych:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![prognoza danych treningowych](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Wydrukowanie MAPE dla danych treningowych

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Zobrazowanie prognoz dla danych testowych

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![prognoza danych testowych](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Wydrukowanie MAPE dla danych testowych

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Uzyskałeś bardzo dobry wynik na zbiorze danych testowych!

### Sprawdzenie wydajności modelu na pełnym zestawie danych [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![prognoza pełnych danych](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Bardzo ładne wykresy, pokazujące model o dobrej dokładności. Świetna robota!

---

## 🚀Wyzwanie

- Spróbuj dostosować hiperparametry (gamma, C, epsilon) podczas tworzenia modelu i ocenić dane, aby zobaczyć, który zestaw hiperparametrów daje najlepsze wyniki na danych testowych. Aby dowiedzieć się więcej o tych hiperparametrach, możesz odnieść się do dokumentu [tutaj](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Spróbuj użyć różnych funkcji jądra dla modelu i przeanalizuj ich wydajność na zestawie danych. Pomocny dokument znajdziesz [tutaj](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Spróbuj użyć różnych wartości dla `timesteps`, aby model mógł spojrzeć wstecz i dokonać prognozy.

## [Quiz po lekcji](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Ta lekcja miała na celu wprowadzenie zastosowania SVR do prognozowania szeregów czasowych. Aby dowiedzieć się więcej o SVR, możesz odnieść się do [tego bloga](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Ta [dokumentacja scikit-learn](https://scikit-learn.org/stable/modules/svm.html) zawiera bardziej kompleksowe wyjaśnienie na temat SVM ogólnie, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) oraz innych szczegółów implementacji, takich jak różne [funkcje jądra](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), które można używać, oraz ich parametry.

## Zadanie

[Nowy model SVR](assignment.md)

## Podziękowania

[^1]: Tekst, kod i wyniki w tej sekcji zostały dostarczone przez [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Tekst, kod i wyniki w tej sekcji zostały zaczerpnięte z [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby zapewnić dokładność, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji o krytycznym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.