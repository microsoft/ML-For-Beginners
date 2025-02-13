# Classificateurs de cuisine 2

Dans cette deuxième leçon de classification, vous explorerez d'autres façons de classifier des données numériques. Vous apprendrez également les implications du choix d'un classificateur plutôt qu'un autre.

## [Quiz pré-cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/23/)

### Prérequis

Nous supposons que vous avez terminé les leçons précédentes et que vous disposez d'un ensemble de données nettoyé dans votre dossier `data` appelé _cleaned_cuisines.csv_ à la racine de ce dossier de 4 leçons.

### Préparation

Nous avons chargé votre fichier _notebook.ipynb_ avec l'ensemble de données nettoyé et l'avons divisé en dataframes X et y, prêts pour le processus de construction du modèle.

## Une carte de classification

Auparavant, vous avez appris les différentes options qui s'offrent à vous lors de la classification des données en utilisant la feuille de triche de Microsoft. Scikit-learn propose une feuille de triche similaire, mais plus détaillée, qui peut encore vous aider à affiner vos estimateurs (un autre terme pour classificateurs) :

![Carte ML de Scikit-learn](../../../../translated_images/map.e963a6a51349425ab107b38f6c7307eb4c0d0c7ccdd2e81a5e1919292bab9ac7.fr.png)
> Conseil : [visitez cette carte en ligne](https://scikit-learn.org/stable/tutorial/machine_learning_map/) et cliquez le long du chemin pour lire la documentation.

### Le plan

Cette carte est très utile une fois que vous avez une bonne compréhension de vos données, car vous pouvez 'marcher' le long de ses chemins vers une décision :

- Nous avons >50 échantillons
- Nous voulons prédire une catégorie
- Nous avons des données étiquetées
- Nous avons moins de 100K échantillons
- ✨ Nous pouvons choisir un SVC linéaire
- Si cela ne fonctionne pas, puisque nous avons des données numériques
    - Nous pouvons essayer un ✨ classificateur KNeighbors 
      - Si cela ne fonctionne pas, essayez un ✨ SVC et des ✨ classificateurs d'ensemble

C'est un chemin très utile à suivre.

## Exercice - diviser les données

En suivant ce chemin, nous devrions commencer par importer certaines bibliothèques à utiliser.

1. Importez les bibliothèques nécessaires :

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Divisez vos données d'entraînement et de test :

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Classificateur SVC linéaire

Le clustering par Support-Vector (SVC) est un membre de la famille des techniques ML des machines à vecteurs de support (apprenez-en plus sur celles-ci ci-dessous). Dans cette méthode, vous pouvez choisir un 'noyau' pour décider comment regrouper les étiquettes. Le paramètre 'C' fait référence à la 'régularisation' qui régule l'influence des paramètres. Le noyau peut être l'un des [plusieurs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) ; ici, nous le définissons sur 'linéaire' pour nous assurer que nous tirons parti du SVC linéaire. La probabilité par défaut est 'fausse' ; ici, nous la définissons sur 'vrai' pour recueillir des estimations de probabilité. Nous définissons l'état aléatoire sur '0' pour mélanger les données afin d'obtenir des probabilités.

### Exercice - appliquer un SVC linéaire

Commencez par créer un tableau de classificateurs. Vous ajouterez progressivement à ce tableau au fur et à mesure de nos tests. 

1. Commencez avec un SVC linéaire :

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Entraînez votre modèle en utilisant le SVC linéaire et imprimez un rapport :

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Le résultat est plutôt bon :

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

## Classificateur K-Neighbors

K-Neighbors fait partie de la famille des méthodes ML "voisins", qui peuvent être utilisées pour l'apprentissage supervisé et non supervisé. Dans cette méthode, un nombre prédéfini de points est créé et des données sont rassemblées autour de ces points de manière à ce que des étiquettes généralisées puissent être prédites pour les données.

### Exercice - appliquer le classificateur K-Neighbors

Le classificateur précédent était bon et a bien fonctionné avec les données, mais peut-être que nous pouvons obtenir une meilleure précision. Essayez un classificateur K-Neighbors.

1. Ajoutez une ligne à votre tableau de classificateurs (ajoutez une virgule après l'élément SVC linéaire) :

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Le résultat est un peu moins bon :

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

    ✅ Apprenez-en plus sur [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Classificateur à vecteurs de support

Les classificateurs à vecteurs de support font partie de la famille des [machines à vecteurs de support](https://wikipedia.org/wiki/Support-vector_machine) des méthodes ML utilisées pour les tâches de classification et de régression. Les SVM "cartographient les exemples d'entraînement en points dans l'espace" pour maximiser la distance entre deux catégories. Les données suivantes sont cartographiées dans cet espace afin que leur catégorie puisse être prédite.

### Exercice - appliquer un classificateur à vecteurs de support

Essayons d'obtenir une précision un peu meilleure avec un classificateur à vecteurs de support.

1. Ajoutez une virgule après l'élément K-Neighbors, puis ajoutez cette ligne :

    ```python
    'SVC': SVC(),
    ```

    Le résultat est assez bon !

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

    ✅ Apprenez-en plus sur [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Classificateurs d'ensemble

Suivons le chemin jusqu'à la fin, même si le test précédent était assez bon. Essayons quelques 'classificateurs d'ensemble', en particulier Random Forest et AdaBoost :

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Le résultat est très bon, surtout pour Random Forest :

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

✅ Apprenez-en plus sur [classificateurs d'ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Cette méthode d'apprentissage automatique "combine les prédictions de plusieurs estimateurs de base" pour améliorer la qualité du modèle. Dans notre exemple, nous avons utilisé des arbres aléatoires et AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), une méthode de moyennage, construit une 'forêt' d'arbres de décision infusés de hasard pour éviter le surajustement. Le paramètre n_estimators est défini sur le nombre d'arbres.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ajuste un classificateur à un ensemble de données, puis ajuste des copies de ce classificateur au même ensemble de données. Il se concentre sur les poids des éléments mal classés et ajuste l'ajustement pour le prochain classificateur afin de corriger.

---

## 🚀Défi

Chacune de ces techniques a un grand nombre de paramètres que vous pouvez ajuster. Renseignez-vous sur les paramètres par défaut de chacun et réfléchissez à ce que l'ajustement de ces paramètres signifierait pour la qualité du modèle.

## [Quiz post-cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/24/)

## Révision & Auto-étude

Il y a beaucoup de jargon dans ces leçons, alors prenez une minute pour revoir [cette liste](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) de terminologie utile !

## Devoir 

[Jeu de paramètres](assignment.md)

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autorisée. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées découlant de l'utilisation de cette traduction.