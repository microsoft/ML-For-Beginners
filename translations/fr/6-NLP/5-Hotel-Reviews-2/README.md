# Analyse de sentiment avec les avis d'hôtels

Maintenant que vous avez exploré le jeu de données en détail, il est temps de filtrer les colonnes et d'utiliser des techniques de NLP sur le jeu de données pour obtenir de nouvelles informations sur les hôtels.
## [Quiz pré-conférence](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Opérations de filtrage et d'analyse de sentiment

Comme vous l'avez probablement remarqué, le jeu de données présente quelques problèmes. Certaines colonnes sont remplies d'informations inutiles, d'autres semblent incorrectes. Si elles sont correctes, il n'est pas clair comment elles ont été calculées, et les réponses ne peuvent pas être vérifiées indépendamment par vos propres calculs.

## Exercice : un peu plus de traitement des données

Nettoyez les données un peu plus. Ajoutez des colonnes qui seront utiles plus tard, modifiez les valeurs dans d'autres colonnes et supprimez certaines colonnes complètement.

1. Traitement initial des colonnes

   1. Supprimez `lat` et `lng`

   2. Remplacez les valeurs `Hotel_Address` par les valeurs suivantes (si l'adresse contient à la fois la ville et le pays, changez-la simplement en la ville et le pays).

      Voici les seules villes et pays dans le jeu de données :

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

      Maintenant, vous pouvez interroger les données au niveau du pays :

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Pays-Bas    |    105     |
      | Barcelone, Espagne      |    211     |
      | Londres, Royaume-Uni    |    400     |
      | Milan, Italie           |    162     |
      | Paris, France           |    458     |
      | Vienne, Autriche       |    158     |

2. Traitement des colonnes de méta-avis d'hôtel

  1. Supprimez `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` avec notre propre score calculé

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Traitement des colonnes d'avis

   1. Supprimez `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, vous pouvez commencer à vous demander s'il est possible de réduire considérablement le traitement que vous devez effectuer. Heureusement, c'est le cas - mais d'abord, vous devez suivre quelques étapes pour déterminer les tags d'intérêt.

### Filtrage des tags

Rappelez-vous que l'objectif du jeu de données est d'ajouter du sentiment et des colonnes qui vous aideront à choisir le meilleur hôtel (pour vous-même ou peut-être pour un client vous demandant de créer un bot de recommandation d'hôtels). Vous devez vous demander si les tags sont utiles ou non dans le jeu de données final. Voici une interprétation (si vous aviez besoin du jeu de données pour d'autres raisons, d'autres tags pourraient rester ou être exclus de la sélection) :

1. Le type de voyage est pertinent, et cela doit rester
2. Le type de groupe de clients est important, et cela doit rester
3. Le type de chambre, suite ou studio dans lequel le client a séjourné est sans rapport (tous les hôtels ont essentiellement les mêmes chambres)
4. L'appareil sur lequel l'avis a été soumis est sans rapport
5. Le nombre de nuits que le critique a passées *pourrait* être pertinent si vous attribuez des séjours plus longs à un plus grand plaisir de l'hôtel, mais c'est un peu tiré par les cheveux et probablement sans rapport

En résumé, **conservez 2 types de tags et supprimez les autres**.

Tout d'abord, vous ne voulez pas compter les tags tant qu'ils ne sont pas dans un meilleur format, donc cela signifie enlever les crochets et les guillemets. Vous pouvez le faire de plusieurs manières, mais vous voulez la méthode la plus rapide car cela pourrait prendre beaucoup de temps pour traiter une grande quantité de données. Heureusement, pandas a un moyen facile de faire chacune de ces étapes.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Chaque tag devient quelque chose comme : `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` la colonne correspond à l'une des nouvelles colonnes, ajoutez un 1, sinon, ajoutez un 0. Le résultat final sera un compte du nombre de critiques qui ont choisi cet hôtel (au total) pour, disons, affaires contre loisirs, ou pour amener un animal de compagnie, et c'est une information utile lors de la recommandation d'un hôtel.

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

### Enregistrez votre fichier

Enfin, enregistrez le jeu de données tel qu'il est maintenant avec un nouveau nom.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Opérations d'analyse de sentiment

Dans cette dernière section, vous appliquerez une analyse de sentiment aux colonnes d'avis et enregistrerez les résultats dans un jeu de données.

## Exercice : charger et enregistrer les données filtrées

Notez que maintenant vous chargez le jeu de données filtré qui a été enregistré dans la section précédente, **pas** le jeu de données original.

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

Si vous deviez effectuer une analyse de sentiment sur les colonnes d'avis négatifs et positifs, cela pourrait prendre beaucoup de temps. Testé sur un ordinateur portable puissant avec un processeur rapide, cela a pris 12 à 14 minutes selon la bibliothèque de sentiment utilisée. C'est un temps (relativement) long, donc cela vaut la peine d'enquêter sur la possibilité d'accélérer le processus. 

La suppression des mots vides, ou des mots anglais courants qui ne changent pas le sentiment d'une phrase, est la première étape. En les supprimant, l'analyse de sentiment devrait s'exécuter plus rapidement, mais pas être moins précise (car les mots vides n'affectent pas le sentiment, mais ralentissent l'analyse). 

Le plus long avis négatif faisait 395 mots, mais après suppression des mots vides, il ne fait plus que 195 mots.

La suppression des mots vides est également une opération rapide, retirer les mots vides de 2 colonnes d'avis sur plus de 515 000 lignes a pris 3,3 secondes sur l'appareil de test. Cela pourrait prendre un peu plus ou moins de temps pour vous en fonction de la vitesse de votre processeur, de votre RAM, de la présence ou non d'un SSD, et d'autres facteurs. La relative brièveté de l'opération signifie que si cela améliore le temps d'analyse de sentiment, cela vaut la peine d'être fait.

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

### Effectuer l'analyse de sentiment

Maintenant, vous devez calculer l'analyse de sentiment pour les colonnes d'avis négatifs et positifs, et stocker le résultat dans 2 nouvelles colonnes. Le test du sentiment consistera à le comparer au score du critique pour le même avis. Par exemple, si le sentiment pense que l'avis négatif avait un sentiment de 1 (sentiment extrêmement positif) et un sentiment d'avis positif de 1, mais que le critique a donné à l'hôtel la note la plus basse possible, alors soit le texte de l'avis ne correspond pas au score, soit l'analyste de sentiment n'a pas pu reconnaître correctement le sentiment. Vous devriez vous attendre à ce que certains scores de sentiment soient complètement erronés, et souvent cela sera explicable, par exemple, l'avis pourrait être extrêmement sarcastique "Bien sûr, j'AI ADORÉ dormir dans une chambre sans chauffage" et l'analyste de sentiment pense que c'est un sentiment positif, même si un humain le lirait et comprendrait qu'il s'agit de sarcasme.

NLTK fournit différents analyseurs de sentiment à apprendre avec, et vous pouvez les substituer et voir si le sentiment est plus ou moins précis. L'analyse de sentiment VADER est utilisée ici.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER : Un modèle basé sur des règles parcimonieuses pour l'analyse de sentiment de textes sur les médias sociaux. Huitième Conférence internationale sur les blogs et les médias sociaux (ICWSM-14). Ann Arbor, MI, juin 2014.

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

Plus tard dans votre programme, lorsque vous êtes prêt à calculer le sentiment, vous pouvez l'appliquer à chaque avis comme suit :

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Cela prend environ 120 secondes sur mon ordinateur, mais cela variera d'un ordinateur à l'autre. Si vous voulez imprimer les résultats et voir si le sentiment correspond à l'avis :

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

La toute dernière chose à faire avec le fichier avant de l'utiliser dans le défi est de l'enregistrer ! Vous devriez également envisager de réorganiser toutes vos nouvelles colonnes afin qu'elles soient faciles à manipuler (pour un humain, c'est un changement cosmétique).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Vous devriez exécuter l'ensemble du code pour [le carnet d'analyse](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (après avoir exécuté [votre carnet de filtrage](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) pour générer le fichier Hotel_Reviews_Filtered.csv).

Pour résumer, les étapes sont :

1. Le fichier de jeu de données original **Hotel_Reviews.csv** a été exploré dans la leçon précédente avec [le carnet d'exploration](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv est filtré par [le carnet de filtrage](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ce qui donne **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv est traité par [le carnet d'analyse de sentiment](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ce qui donne **Hotel_Reviews_NLP.csv**
4. Utilisez Hotel_Reviews_NLP.csv dans le défi NLP ci-dessous

### Conclusion

Lorsque vous avez commencé, vous aviez un jeu de données avec des colonnes et des données, mais tout ne pouvait pas être vérifié ou utilisé. Vous avez exploré les données, filtré ce dont vous n'avez pas besoin, converti les tags en quelque chose d'utile, calculé vos propres moyennes, ajouté quelques colonnes de sentiment et, espérons-le, appris des choses intéressantes sur le traitement du texte naturel.

## [Quiz post-conférence](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Défi

Maintenant que vous avez analysé votre jeu de données pour le sentiment, voyez si vous pouvez utiliser des stratégies que vous avez apprises dans ce cursus (clustering, peut-être ?) pour déterminer des motifs autour du sentiment. 

## Revue et auto-apprentissage

Prenez [ce module Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) pour en savoir plus et utiliser différents outils pour explorer le sentiment dans le texte.
## Mission 

[Essayez un autre jeu de données](assignment.md)

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.