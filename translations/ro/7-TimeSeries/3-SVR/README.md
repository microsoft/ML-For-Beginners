<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T15:38:01+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "ro"
}
-->
# Predicția seriilor temporale cu Support Vector Regressor

În lecția anterioară, ai învățat cum să folosești modelul ARIMA pentru a face predicții ale seriilor temporale. Acum vei explora modelul Support Vector Regressor, un model de regresie utilizat pentru a prezice date continue.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/) 

## Introducere

În această lecție, vei descoperi o metodă specifică de a construi modele cu [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) pentru regresie, sau **SVR: Support Vector Regressor**. 

### SVR în contextul seriilor temporale [^1]

Înainte de a înțelege importanța SVR în predicția seriilor temporale, iată câteva concepte importante pe care trebuie să le cunoști:

- **Regresie:** Tehnică de învățare supravegheată pentru a prezice valori continue dintr-un set dat de intrări. Ideea este de a ajusta o curbă (sau o linie) în spațiul caracteristicilor care să includă cât mai multe puncte de date. [Click aici](https://en.wikipedia.org/wiki/Regression_analysis) pentru mai multe informații.
- **Support Vector Machine (SVM):** Un tip de model de învățare automată supravegheat utilizat pentru clasificare, regresie și detectarea anomaliilor. Modelul este un hiperplan în spațiul caracteristicilor, care, în cazul clasificării, acționează ca o graniță, iar în cazul regresiei, acționează ca linia de ajustare optimă. În SVM, o funcție Kernel este utilizată în general pentru a transforma setul de date într-un spațiu cu un număr mai mare de dimensiuni, astfel încât să fie mai ușor separabil. [Click aici](https://en.wikipedia.org/wiki/Support-vector_machine) pentru mai multe informații despre SVM.
- **Support Vector Regressor (SVR):** Un tip de SVM, utilizat pentru a găsi linia de ajustare optimă (care, în cazul SVM, este un hiperplan) ce include cât mai multe puncte de date.

### De ce SVR? [^1]

În lecția anterioară ai învățat despre ARIMA, care este o metodă statistică liniară foarte eficientă pentru a prezice datele seriilor temporale. Totuși, în multe cazuri, datele seriilor temporale prezintă *non-liniaritate*, care nu poate fi modelată de metodele liniare. În astfel de cazuri, abilitatea SVM de a considera non-liniaritatea datelor pentru sarcinile de regresie face ca SVR să fie eficient în predicția seriilor temporale.

## Exercițiu - construirea unui model SVR

Primele câteva etape pentru pregătirea datelor sunt aceleași ca în lecția anterioară despre [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Deschide folderul [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) din această lecție și găsește fișierul [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Rulează notebook-ul și importă bibliotecile necesare: [^2]

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

2. Încarcă datele din fișierul `/data/energy.csv` într-un dataframe Pandas și analizează-le: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Plotează toate datele disponibile despre energie din ianuarie 2012 până în decembrie 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![date complete](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Acum, să construim modelul SVR.

### Crearea seturilor de date pentru antrenare și testare

Acum datele tale sunt încărcate, așa că le poți separa în seturi de antrenare și testare. Apoi vei rearanja datele pentru a crea un set de date bazat pe pași de timp, necesar pentru SVR. Vei antrena modelul pe setul de antrenare. După ce modelul a terminat antrenarea, îi vei evalua acuratețea pe setul de antrenare, setul de testare și apoi pe întregul set de date pentru a vedea performanța generală. Trebuie să te asiguri că setul de testare acoperă o perioadă ulterioară în timp față de setul de antrenare pentru a te asigura că modelul nu obține informații din perioadele viitoare [^2] (o situație cunoscută sub numele de *Overfitting*).

1. Alocă o perioadă de două luni, de la 1 septembrie până la 31 octombrie 2014, pentru setul de antrenare. Setul de testare va include perioada de două luni de la 1 noiembrie până la 31 decembrie 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Vizualizează diferențele: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![date de antrenare și testare](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Pregătirea datelor pentru antrenare

Acum trebuie să pregătești datele pentru antrenare prin filtrarea și scalarea acestora. Filtrează setul de date pentru a include doar perioadele de timp și coloanele necesare, și scalează datele pentru a te asigura că sunt proiectate în intervalul 0,1.

1. Filtrează setul de date original pentru a include doar perioadele de timp menționate anterior pentru fiecare set și doar coloana necesară 'load' plus data: [^2]

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
   
2. Scalează datele de antrenare pentru a fi în intervalul (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Acum scalează datele de testare: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Crearea datelor cu pași de timp [^1]

Pentru SVR, transformi datele de intrare în forma `[batch, timesteps]`. Așadar, rearanjezi `train_data` și `test_data` existente astfel încât să existe o nouă dimensiune care se referă la pașii de timp. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Pentru acest exemplu, luăm `timesteps = 5`. Așadar, intrările modelului sunt datele pentru primii 4 pași de timp, iar ieșirea va fi datele pentru al 5-lea pas de timp.

```python
timesteps=5
```

Conversia datelor de antrenare într-un tensor 2D folosind list comprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Conversia datelor de testare într-un tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Selectarea intrărilor și ieșirilor din datele de antrenare și testare:

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

### Implementarea SVR [^1]

Acum este momentul să implementezi SVR. Pentru a citi mai multe despre această implementare, poți consulta [această documentație](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Pentru implementarea noastră, urmăm acești pași:

  1. Definim modelul apelând `SVR()` și trecând hiperparametrii modelului: kernel, gamma, c și epsilon
  2. Pregătim modelul pentru datele de antrenare apelând funcția `fit()`
  3. Facem predicții apelând funcția `predict()`

Acum creăm un model SVR. Aici folosim [kernelul RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) și setăm hiperparametrii gamma, C și epsilon la 0.5, 10 și 0.05 respectiv.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Antrenează modelul pe datele de antrenare [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Realizează predicții cu modelul [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Ai construit modelul SVR! Acum trebuie să-l evaluăm.

### Evaluează modelul [^1]

Pentru evaluare, mai întâi vom scala înapoi datele la scara originală. Apoi, pentru a verifica performanța, vom plota graficul seriei temporale originale și prezise și vom afișa rezultatul MAPE.

Scalează ieșirea prezisă și cea originală:

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

#### Verifică performanța modelului pe datele de antrenare și testare [^1]

Extragem marcajele temporale din setul de date pentru a le afișa pe axa x a graficului nostru. Observă că folosim primele ```timesteps-1``` valori ca intrare pentru prima ieșire, astfel încât marcajele temporale pentru ieșire vor începe după aceea.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Plotează predicțiile pentru datele de antrenare:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![predicția datelor de antrenare](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Afișează MAPE pentru datele de antrenare

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Plotează predicțiile pentru datele de testare

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![predicția datelor de testare](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Afișează MAPE pentru datele de testare

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Ai obținut un rezultat foarte bun pe setul de date de testare!

### Verifică performanța modelului pe întregul set de date [^1]

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

![predicția datelor complete](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Grafice foarte frumoase, care arată un model cu o acuratețe bună. Felicitări!

---

## 🚀Provocare

- Încearcă să ajustezi hiperparametrii (gamma, C, epsilon) în timp ce creezi modelul și evaluează-l pe date pentru a vedea care set de hiperparametri oferă cele mai bune rezultate pe datele de testare. Pentru a afla mai multe despre acești hiperparametri, poți consulta documentul [aici](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Încearcă să folosești funcții kernel diferite pentru model și analizează performanțele acestora pe setul de date. Un document util poate fi găsit [aici](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Încearcă să folosești valori diferite pentru `timesteps` pentru ca modelul să privească înapoi pentru a face predicții.

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Această lecție a fost pentru a introduce aplicația SVR pentru predicția seriilor temporale. Pentru a citi mai multe despre SVR, poți consulta [acest blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Această [documentație despre scikit-learn](https://scikit-learn.org/stable/modules/svm.html) oferă o explicație mai cuprinzătoare despre SVM-uri în general, [SVR-uri](https://scikit-learn.org/stable/modules/svm.html#regression) și, de asemenea, alte detalii de implementare, cum ar fi diferitele [funcții kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) care pot fi utilizate și parametrii acestora.

## Temă

[Un nou model SVR](assignment.md)

## Credite

[^1]: Textul, codul și rezultatele din această secțiune au fost contribuite de [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Textul, codul și rezultatele din această secțiune au fost preluate din [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.