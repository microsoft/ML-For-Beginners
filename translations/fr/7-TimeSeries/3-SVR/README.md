<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T22:56:04+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "fr"
}
-->
# Prévision des séries temporelles avec le Support Vector Regressor

Dans la leçon précédente, vous avez appris à utiliser le modèle ARIMA pour effectuer des prédictions sur des séries temporelles. Maintenant, vous allez découvrir le modèle Support Vector Regressor, un modèle de régression utilisé pour prédire des données continues.

## [Quiz avant la leçon](https://ff-quizzes.netlify.app/en/ml/) 

## Introduction

Dans cette leçon, vous allez découvrir une méthode spécifique pour construire des modèles avec [**SVM** : **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) pour la régression, ou **SVR : Support Vector Regressor**.

### SVR dans le contexte des séries temporelles [^1]

Avant de comprendre l'importance du SVR dans la prévision des séries temporelles, voici quelques concepts importants que vous devez connaître :

- **Régression :** Technique d'apprentissage supervisé pour prédire des valeurs continues à partir d'un ensemble d'entrées donné. L'idée est d'ajuster une courbe (ou une ligne) dans l'espace des caractéristiques qui contient le maximum de points de données. [Cliquez ici](https://en.wikipedia.org/wiki/Regression_analysis) pour plus d'informations.
- **Support Vector Machine (SVM) :** Type de modèle d'apprentissage supervisé utilisé pour la classification, la régression et la détection des anomalies. Le modèle est un hyperplan dans l'espace des caractéristiques, qui agit comme une frontière dans le cas de la classification, et comme une ligne de meilleur ajustement dans le cas de la régression. Dans le SVM, une fonction Kernel est généralement utilisée pour transformer le jeu de données dans un espace de dimensions supérieures, afin qu'ils soient facilement séparables. [Cliquez ici](https://en.wikipedia.org/wiki/Support-vector_machine) pour plus d'informations sur les SVM.
- **Support Vector Regressor (SVR) :** Type de SVM, utilisé pour trouver la ligne de meilleur ajustement (qui, dans le cas du SVM, est un hyperplan) contenant le maximum de points de données.

### Pourquoi utiliser le SVR ? [^1]

Dans la dernière leçon, vous avez appris à utiliser ARIMA, une méthode statistique linéaire très efficace pour prévoir les données des séries temporelles. Cependant, dans de nombreux cas, les données des séries temporelles présentent une *non-linéarité*, qui ne peut pas être modélisée par des modèles linéaires. Dans de tels cas, la capacité du SVM à prendre en compte la non-linéarité des données pour les tâches de régression rend le SVR efficace pour la prévision des séries temporelles.

## Exercice - Construire un modèle SVR

Les premières étapes de préparation des données sont les mêmes que celles de la leçon précédente sur [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA).

Ouvrez le dossier [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) dans cette leçon et trouvez le fichier [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Exécutez le notebook et importez les bibliothèques nécessaires : [^2]

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

2. Chargez les données du fichier `/data/energy.csv` dans un dataframe Pandas et examinez-les : [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Tracez toutes les données énergétiques disponibles de janvier 2012 à décembre 2014 : [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![données complètes](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Maintenant, construisons notre modèle SVR.

### Créer des ensembles de données d'entraînement et de test

Vos données sont maintenant chargées, vous pouvez donc les séparer en ensembles d'entraînement et de test. Ensuite, vous allez remodeler les données pour créer un ensemble de données basé sur des étapes temporelles, ce qui sera nécessaire pour le SVR. Vous entraînerez votre modèle sur l'ensemble d'entraînement. Une fois l'entraînement terminé, vous évaluerez sa précision sur l'ensemble d'entraînement, l'ensemble de test, puis l'ensemble complet pour voir les performances globales. Vous devez vous assurer que l'ensemble de test couvre une période ultérieure à celle de l'ensemble d'entraînement afin que le modèle ne tire pas d'informations des périodes futures [^2] (une situation connue sous le nom de *surapprentissage*).

1. Allouez une période de deux mois du 1er septembre au 31 octobre 2014 à l'ensemble d'entraînement. L'ensemble de test inclura la période de deux mois du 1er novembre au 31 décembre 2014 : [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualisez les différences : [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![données d'entraînement et de test](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Préparer les données pour l'entraînement

Maintenant, vous devez préparer les données pour l'entraînement en effectuant un filtrage et une mise à l'échelle de vos données. Filtrez votre jeu de données pour inclure uniquement les périodes et colonnes nécessaires, et mettez à l'échelle pour garantir que les données sont projetées dans l'intervalle 0,1.

1. Filtrez le jeu de données original pour inclure uniquement les périodes mentionnées par ensemble et uniquement la colonne nécessaire 'load' ainsi que la date : [^2]

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
   
2. Mettez à l'échelle les données d'entraînement pour qu'elles soient dans la plage (0, 1) : [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Maintenant, mettez à l'échelle les données de test : [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Créer des données avec des étapes temporelles [^1]

Pour le SVR, vous transformez les données d'entrée pour qu'elles soient sous la forme `[batch, timesteps]`. Ainsi, vous remodelez les `train_data` et `test_data` existants de manière à ajouter une nouvelle dimension qui fait référence aux étapes temporelles.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Pour cet exemple, nous prenons `timesteps = 5`. Ainsi, les entrées du modèle sont les données des 4 premières étapes temporelles, et la sortie sera les données de la 5ème étape temporelle.

```python
timesteps=5
```

Conversion des données d'entraînement en tenseur 2D à l'aide de la compréhension de listes imbriquées :

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Conversion des données de test en tenseur 2D :

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Sélection des entrées et sorties des données d'entraînement et de test :

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

### Implémenter le SVR [^1]

Il est maintenant temps d'implémenter le SVR. Pour en savoir plus sur cette implémentation, vous pouvez consulter [cette documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Pour notre implémentation, nous suivons ces étapes :

  1. Définir le modèle en appelant `SVR()` et en passant les hyperparamètres du modèle : kernel, gamma, c et epsilon
  2. Préparer le modèle pour les données d'entraînement en appelant la fonction `fit()`
  3. Faire des prédictions en appelant la fonction `predict()`

Nous créons maintenant un modèle SVR. Ici, nous utilisons le [kernel RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), et définissons les hyperparamètres gamma, C et epsilon à 0.5, 10 et 0.05 respectivement.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Ajuster le modèle sur les données d'entraînement [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Faire des prédictions avec le modèle [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Vous avez construit votre SVR ! Maintenant, nous devons l'évaluer.

### Évaluer votre modèle [^1]

Pour l'évaluation, nous allons d'abord remettre les données à leur échelle originale. Ensuite, pour vérifier les performances, nous tracerons le graphique des séries temporelles originales et prédites, et imprimerons également le résultat du MAPE.

Remettre à l'échelle les sorties prédites et originales :

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

#### Vérifier les performances du modèle sur les données d'entraînement et de test [^1]

Nous extrayons les horodatages du jeu de données pour les afficher sur l'axe x de notre graphique. Notez que nous utilisons les ```timesteps-1``` premières valeurs comme entrée pour la première sortie, donc les horodatages pour la sortie commenceront après cela.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Tracer les prédictions pour les données d'entraînement :

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![prédictions des données d'entraînement](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Imprimer le MAPE pour les données d'entraînement

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Tracer les prédictions pour les données de test

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![prédictions des données de test](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Imprimer le MAPE pour les données de test

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Vous avez obtenu un très bon résultat sur l'ensemble de test !

### Vérifier les performances du modèle sur l'ensemble complet [^1]

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

![prédictions des données complètes](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Très beaux graphiques, montrant un modèle avec une bonne précision. Bien joué !

---

## 🚀Défi

- Essayez de modifier les hyperparamètres (gamma, C, epsilon) lors de la création du modèle et évaluez les données pour voir quel ensemble d'hyperparamètres donne les meilleurs résultats sur les données de test. Pour en savoir plus sur ces hyperparamètres, vous pouvez consulter le document [ici](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Essayez d'utiliser différentes fonctions kernel pour le modèle et analysez leurs performances sur le jeu de données. Un document utile peut être trouvé [ici](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Essayez d'utiliser différentes valeurs pour `timesteps` pour que le modèle puisse regarder en arrière pour faire des prédictions.

## [Quiz après la leçon](https://ff-quizzes.netlify.app/en/ml/)

## Révision et auto-apprentissage

Cette leçon avait pour but d'introduire l'application du SVR pour la prévision des séries temporelles. Pour en savoir plus sur le SVR, vous pouvez consulter [ce blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Cette [documentation sur scikit-learn](https://scikit-learn.org/stable/modules/svm.html) fournit une explication plus complète sur les SVM en général, les [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) et également d'autres détails d'implémentation tels que les différentes [fonctions kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) qui peuvent être utilisées, ainsi que leurs paramètres.

## Devoir

[Un nouveau modèle SVR](assignment.md)

## Remerciements

[^1]: Le texte, le code et les résultats de cette section ont été contribué par [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Le texte, le code et les résultats de cette section ont été tirés de [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction professionnelle humaine. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.