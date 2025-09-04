<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-04T00:54:58+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "fr"
}
-->
# Analyse de sentiment avec les avis d'hôtels

Maintenant que vous avez exploré le jeu de données en détail, il est temps de filtrer les colonnes et d'utiliser des techniques de NLP sur le jeu de données pour obtenir de nouvelles informations sur les hôtels.

## [Quiz avant le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Opérations de filtrage et d'analyse de sentiment

Comme vous l'avez probablement remarqué, le jeu de données présente quelques problèmes. Certaines colonnes contiennent des informations inutiles, d'autres semblent incorrectes. Si elles sont correctes, il est difficile de comprendre comment elles ont été calculées, et les réponses ne peuvent pas être vérifiées indépendamment par vos propres calculs.

## Exercice : un peu plus de traitement des données

Nettoyez les données un peu plus. Ajoutez des colonnes qui seront utiles plus tard, modifiez les valeurs dans d'autres colonnes, et supprimez certaines colonnes complètement.

1. Traitement initial des colonnes

   1. Supprimez `lat` et `lng`.

   2. Remplacez les valeurs de `Hotel_Address` par les valeurs suivantes (si l'adresse contient le nom de la ville et du pays, changez-la pour inclure uniquement la ville et le pays).

      Voici les seules villes et pays présents dans le jeu de données :

      Amsterdam, Pays-Bas

      Barcelone, Espagne

      Londres, Royaume-Uni

      Milan, Italie

      Paris, France

      Vienne, Autriche 

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

      Vous pouvez maintenant interroger les données au niveau des pays :

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Pays-Bas    |    105     |
      | Barcelone, Espagne     |    211     |
      | Londres, Royaume-Uni   |    400     |
      | Milan, Italie          |    162     |
      | Paris, France          |    458     |
      | Vienne, Autriche       |    158     |

2. Traitez les colonnes de méta-avis des hôtels

   1. Supprimez `Additional_Number_of_Scoring`.

   2. Remplacez `Total_Number_of_Reviews` par le nombre total d'avis pour cet hôtel qui sont réellement présents dans le jeu de données.

   3. Remplacez `Average_Score` par notre propre score calculé.

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Traitez les colonnes des avis

   1. Supprimez `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` et `days_since_review`.

   2. Conservez `Reviewer_Score`, `Negative_Review` et `Positive_Review` tels quels.

   3. Conservez `Tags` pour l'instant.

      - Nous effectuerons des opérations de filtrage supplémentaires sur les tags dans la section suivante, puis les tags seront supprimés.

4. Traitez les colonnes des évaluateurs

   1. Supprimez `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Conservez `Reviewer_Nationality`.

### Colonnes des tags

La colonne `Tag` est problématique car elle contient une liste (sous forme de texte) stockée dans la colonne. Malheureusement, l'ordre et le nombre de sous-sections dans cette colonne ne sont pas toujours les mêmes. Il est difficile pour un humain d'identifier les phrases correctes à analyser, car il y a 515 000 lignes et 1427 hôtels, et chacun propose des options légèrement différentes que le critique pourrait choisir. C'est là que le NLP est utile. Vous pouvez analyser le texte et trouver les phrases les plus courantes, puis les compter.

Malheureusement, nous ne sommes pas intéressés par les mots individuels, mais par les expressions multi-mots (par exemple, *Voyage d'affaires*). Exécuter un algorithme de distribution de fréquence pour les expressions multi-mots sur autant de données (6762646 mots) pourrait prendre un temps extraordinaire, mais sans examiner les données, il semblerait que ce soit une dépense nécessaire. C'est là que l'analyse exploratoire des données est utile, car vous avez vu un échantillon des tags tels que `[' Voyage d'affaires  ', ' Voyageur solo ', ' Chambre simple ', ' Séjour de 5 nuits ', ' Soumis depuis un appareil mobile ']`, vous pouvez commencer à vous demander s'il est possible de réduire considérablement le traitement à effectuer. Heureusement, c'est possible - mais vous devez d'abord suivre quelques étapes pour identifier les tags d'intérêt.

### Filtrage des tags

Rappelez-vous que l'objectif du jeu de données est d'ajouter des sentiments et des colonnes qui vous aideront à choisir le meilleur hôtel (pour vous-même ou peut-être pour un client vous demandant de créer un bot de recommandation d'hôtel). Vous devez vous demander si les tags sont utiles ou non dans le jeu de données final. Voici une interprétation (si vous aviez besoin du jeu de données pour d'autres raisons, différents tags pourraient rester ou être exclus) :

1. Le type de voyage est pertinent et doit rester.
2. Le type de groupe de voyageurs est important et doit rester.
3. Le type de chambre, suite ou studio dans lequel le client a séjourné est sans importance (tous les hôtels ont essentiellement les mêmes chambres).
4. L'appareil sur lequel l'avis a été soumis est sans importance.
5. Le nombre de nuits passées par le critique *pourrait* être pertinent si vous attribuez des séjours plus longs à une appréciation accrue de l'hôtel, mais c'est une hypothèse peu probable et probablement sans importance.

En résumé, **conservez 2 types de tags et supprimez les autres**.

Tout d'abord, vous ne voulez pas compter les tags tant qu'ils ne sont pas dans un format plus approprié, ce qui signifie supprimer les crochets et les guillemets. Vous pouvez le faire de plusieurs façons, mais vous voulez la méthode la plus rapide car cela pourrait prendre beaucoup de temps pour traiter autant de données. Heureusement, pandas propose une méthode simple pour effectuer chacune de ces étapes.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Chaque tag devient quelque chose comme : `Voyage d'affaires, Voyageur solo, Chambre simple, Séjour de 5 nuits, Soumis depuis un appareil mobile`.

Ensuite, nous rencontrons un problème. Certains avis, ou lignes, ont 5 colonnes, d'autres 3, d'autres encore 6. Cela résulte de la manière dont le jeu de données a été créé, et il est difficile de corriger cela. Vous voulez obtenir un décompte de fréquence de chaque expression, mais elles sont dans un ordre différent dans chaque avis, donc le décompte pourrait être incorrect, et un hôtel pourrait ne pas recevoir un tag qui lui était dû.

Au lieu de cela, vous utiliserez l'ordre différent à votre avantage, car chaque tag est une expression multi-mots mais également séparée par une virgule ! La manière la plus simple de procéder est de créer 6 colonnes temporaires avec chaque tag inséré dans la colonne correspondant à son ordre dans le tag. Vous pouvez ensuite fusionner les 6 colonnes en une grande colonne et exécuter la méthode `value_counts()` sur la colonne résultante. En imprimant cela, vous verrez qu'il y avait 2428 tags uniques. Voici un petit échantillon :

| Tag                            | Count  |
| ------------------------------ | ------ |
| Voyage de loisirs              | 417778 |
| Soumis depuis un appareil mobile | 307640 |
| Couple                         | 252294 |
| Séjour de 1 nuit               | 193645 |
| Séjour de 2 nuits              | 133937 |
| Voyageur solo                  | 108545 |
| Séjour de 3 nuits              | 95821  |
| Voyage d'affaires              | 82939  |
| Groupe                         | 65392  |
| Famille avec jeunes enfants    | 61015  |
| Séjour de 4 nuits              | 47817  |
| Chambre double                 | 35207  |
| Chambre double standard        | 32248  |
| Chambre double supérieure      | 31393  |
| Famille avec enfants plus âgés | 26349  |
| Chambre double deluxe          | 24823  |
| Chambre double ou twin         | 22393  |
| Séjour de 5 nuits              | 20845  |
| Chambre double ou twin standard | 17483  |
| Chambre double classique       | 16989  |
| Chambre double ou twin supérieure | 13570 |
| 2 chambres                     | 12393  |

Certains tags courants comme `Soumis depuis un appareil mobile` ne nous sont d'aucune utilité, donc il pourrait être judicieux de les supprimer avant de compter les occurrences des expressions, mais c'est une opération si rapide que vous pouvez les laisser et les ignorer.

### Suppression des tags liés à la durée du séjour

Supprimer ces tags est la première étape, cela réduit légèrement le nombre total de tags à considérer. Notez que vous ne les supprimez pas du jeu de données, mais choisissez de les exclure de la considération comme valeurs à compter/conserver dans le jeu de données des avis.

| Durée du séjour | Count  |
| ---------------- | ------ |
| Séjour de 1 nuit | 193645 |
| Séjour de 2 nuits | 133937 |
| Séjour de 3 nuits | 95821  |
| Séjour de 4 nuits | 47817  |
| Séjour de 5 nuits | 20845  |
| Séjour de 6 nuits | 9776   |
| Séjour de 7 nuits | 7399   |
| Séjour de 8 nuits | 2502   |
| Séjour de 9 nuits | 1293   |
| ...              | ...    |

Il existe une grande variété de chambres, suites, studios, appartements, etc. Ils signifient tous à peu près la même chose et ne sont pas pertinents pour vous, donc supprimez-les de la considération.

| Type de chambre               | Count |
| ----------------------------- | ----- |
| Chambre double                | 35207 |
| Chambre double standard       | 32248 |
| Chambre double supérieure     | 31393 |
| Chambre double deluxe         | 24823 |
| Chambre double ou twin        | 22393 |
| Chambre double ou twin standard | 17483 |
| Chambre double classique      | 16989 |
| Chambre double ou twin supérieure | 13570 |

Enfin, et c'est une bonne nouvelle (car cela n'a pas nécessité beaucoup de traitement), vous serez laissé avec les tags *utiles* suivants :

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Voyage de loisirs                             | 417778 |
| Couple                                        | 252294 |
| Voyageur solo                                 | 108545 |
| Voyage d'affaires                             | 82939  |
| Groupe (combiné avec Voyageurs avec amis)     | 67535  |
| Famille avec jeunes enfants                   | 61015  |
| Famille avec enfants plus âgés                | 26349  |
| Avec un animal                                | 1405   |

Vous pourriez argumenter que `Voyageurs avec amis` est similaire à `Groupe` plus ou moins, et il serait juste de combiner les deux comme ci-dessus. Le code pour identifier les tags corrects se trouve dans [le notebook des tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

La dernière étape consiste à créer de nouvelles colonnes pour chacun de ces tags. Ensuite, pour chaque ligne d'avis, si la colonne `Tag` correspond à l'une des nouvelles colonnes, ajoutez un 1, sinon ajoutez un 0. Le résultat final sera un décompte du nombre de critiques ayant choisi cet hôtel (en agrégé) pour, par exemple, affaires vs loisirs, ou pour y amener un animal, et cela constitue une information utile pour recommander un hôtel.

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

### Sauvegardez votre fichier

Enfin, sauvegardez le jeu de données tel qu'il est maintenant avec un nouveau nom.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Opérations d'analyse de sentiment

Dans cette dernière section, vous appliquerez une analyse de sentiment aux colonnes des avis et enregistrerez les résultats dans un jeu de données.

## Exercice : charger et sauvegarder les données filtrées

Notez que vous chargez maintenant le jeu de données filtré qui a été sauvegardé dans la section précédente, **et non** le jeu de données original.

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

### Suppression des mots vides

Si vous deviez exécuter une analyse de sentiment sur les colonnes des avis négatifs et positifs, cela pourrait prendre beaucoup de temps. Testé sur un ordinateur portable puissant avec un processeur rapide, cela a pris 12 à 14 minutes selon la bibliothèque de sentiment utilisée. C'est un temps relativement long, donc cela vaut la peine d'examiner si cela peut être accéléré.

La suppression des mots vides, ou des mots courants en anglais qui n'affectent pas le sentiment d'une phrase, est la première étape. En les supprimant, l'analyse de sentiment devrait être plus rapide, sans être moins précise (car les mots vides n'affectent pas le sentiment, mais ralentissent l'analyse).

Le plus long avis négatif comptait 395 mots, mais après suppression des mots vides, il en compte 195.

La suppression des mots vides est également une opération rapide. Supprimer les mots vides de 2 colonnes d'avis sur 515 000 lignes a pris 3,3 secondes sur l'appareil de test. Cela pourrait prendre légèrement plus ou moins de temps pour vous en fonction de la vitesse de votre processeur, de votre RAM, de la présence ou non d'un SSD, et d'autres facteurs. La relative rapidité de l'opération signifie que si elle améliore le temps d'analyse de sentiment, alors cela vaut la peine de le faire.

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

### Effectuer une analyse de sentiment
Maintenant, vous devez calculer l'analyse de sentiment pour les colonnes de critiques négatives et positives, et stocker le résultat dans 2 nouvelles colonnes. Le test du sentiment consistera à le comparer à la note donnée par le critique pour la même critique. Par exemple, si l'analyse de sentiment estime que la critique négative a un sentiment de 1 (sentiment extrêmement positif) et que la critique positive a également un sentiment de 1, mais que le critique a donné à l'hôtel la note la plus basse possible, alors soit le texte de la critique ne correspond pas à la note, soit l'analyseur de sentiment n'a pas réussi à reconnaître correctement le sentiment. Vous devez vous attendre à ce que certains scores de sentiment soient complètement erronés, et cela sera souvent explicable, par exemple la critique pourrait être extrêmement sarcastique : "Bien sûr, j'ai ADORÉ dormir dans une chambre sans chauffage", et l'analyseur de sentiment pense que c'est un sentiment positif, alors qu'un humain lisant cela saurait qu'il s'agit de sarcasme.

NLTK propose différents analyseurs de sentiment à expérimenter, et vous pouvez les substituer pour voir si le sentiment est plus ou moins précis. L'analyse de sentiment VADER est utilisée ici.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, juin 2014.

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

Plus tard dans votre programme, lorsque vous serez prêt à calculer le sentiment, vous pouvez l'appliquer à chaque critique comme suit :

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Cela prend environ 120 secondes sur mon ordinateur, mais cela variera selon chaque machine. Si vous souhaitez imprimer les résultats et vérifier si le sentiment correspond à la critique :

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

La toute dernière chose à faire avec le fichier avant de l'utiliser dans le défi est de le sauvegarder ! Vous devriez également envisager de réorganiser toutes vos nouvelles colonnes pour qu'elles soient faciles à manipuler (pour un humain, c'est un changement cosmétique).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Vous devez exécuter l'intégralité du code pour [le notebook d'analyse](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (après avoir exécuté [votre notebook de filtrage](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) pour générer le fichier Hotel_Reviews_Filtered.csv).

Pour récapituler, les étapes sont :

1. Le fichier de dataset original **Hotel_Reviews.csv** est exploré dans la leçon précédente avec [le notebook d'exploration](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv est filtré par [le notebook de filtrage](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), ce qui donne **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv est traité par [le notebook d'analyse de sentiment](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), ce qui donne **Hotel_Reviews_NLP.csv**
4. Utilisez Hotel_Reviews_NLP.csv dans le défi NLP ci-dessous

### Conclusion

Lorsque vous avez commencé, vous aviez un dataset avec des colonnes et des données, mais tout ne pouvait pas être vérifié ou utilisé. Vous avez exploré les données, filtré ce dont vous n'aviez pas besoin, converti des balises en quelque chose d'utile, calculé vos propres moyennes, ajouté des colonnes de sentiment et, espérons-le, appris des choses intéressantes sur le traitement du texte naturel.

## [Quiz post-lecture](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Défi

Maintenant que vous avez analysé le sentiment de votre dataset, voyez si vous pouvez utiliser les stratégies que vous avez apprises dans ce programme (le clustering, peut-être ?) pour déterminer des motifs autour du sentiment.

## Révision & Étude personnelle

Suivez [ce module Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) pour en apprendre davantage et utiliser différents outils pour explorer le sentiment dans le texte.

## Devoir

[Essayez un dataset différent](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.