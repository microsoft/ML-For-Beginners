<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-04T23:06:47+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "fr"
}
-->
# Analyse de sentiment avec les avis d'hôtels - traitement des données

Dans cette section, vous utiliserez les techniques des leçons précédentes pour effectuer une analyse exploratoire des données sur un grand ensemble de données. Une fois que vous aurez une bonne compréhension de l'utilité des différentes colonnes, vous apprendrez :

- comment supprimer les colonnes inutiles
- comment calculer de nouvelles données à partir des colonnes existantes
- comment sauvegarder l'ensemble de données résultant pour l'utiliser dans le défi final

## [Quiz avant la leçon](https://ff-quizzes.netlify.app/en/ml/)

### Introduction

Jusqu'à présent, vous avez appris que les données textuelles sont très différentes des données numériques. Si le texte a été écrit ou prononcé par un humain, il peut être analysé pour trouver des motifs, des fréquences, des sentiments et des significations. Cette leçon vous plonge dans un ensemble de données réel avec un défi réel : **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, qui inclut une [licence CC0 : Domaine Public](https://creativecommons.org/publicdomain/zero/1.0/). Les données ont été extraites de Booking.com à partir de sources publiques. Le créateur de l'ensemble de données est Jiashen Liu.

### Préparation

Vous aurez besoin de :

* La capacité d'exécuter des notebooks .ipynb avec Python 3
* pandas
* NLTK, [que vous devez installer localement](https://www.nltk.org/install.html)
* L'ensemble de données disponible sur Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Il pèse environ 230 Mo une fois décompressé. Téléchargez-le dans le dossier racine `/data` associé à ces leçons sur le NLP.

## Analyse exploratoire des données

Ce défi suppose que vous construisez un bot de recommandation d'hôtels en utilisant l'analyse de sentiment et les scores des avis des clients. L'ensemble de données que vous utiliserez inclut des avis sur 1493 hôtels différents dans 6 villes.

Avec Python, un ensemble de données d'avis d'hôtels et l'analyse de sentiment de NLTK, vous pourriez découvrir :

* Quels sont les mots et expressions les plus fréquemment utilisés dans les avis ?
* Les *tags* officiels décrivant un hôtel sont-ils corrélés aux scores des avis (par exemple, les avis plus négatifs pour un hôtel particulier concernent-ils davantage les *Familles avec jeunes enfants* que les *Voyageurs en solo*, ce qui pourrait indiquer qu'il est mieux adapté aux *Voyageurs en solo*) ?
* Les scores de sentiment de NLTK "s'accordent-ils" avec le score numérique donné par le client ?

#### Ensemble de données

Explorons l'ensemble de données que vous avez téléchargé et sauvegardé localement. Ouvrez le fichier dans un éditeur comme VS Code ou même Excel.

Les en-têtes de l'ensemble de données sont les suivants :

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Voici une présentation regroupée pour faciliter l'examen :  
##### Colonnes des hôtels

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Avec *lat* et *lng*, vous pourriez tracer une carte avec Python montrant les emplacements des hôtels (peut-être codés par couleur pour les avis négatifs et positifs).
  * Hotel_Address n'est pas évidemment utile pour nous, et nous remplacerons probablement cette colonne par un pays pour faciliter le tri et la recherche.

**Colonnes méta-avis des hôtels**

* `Average_Score`
  * Selon le créateur de l'ensemble de données, cette colonne représente le *Score moyen de l'hôtel, calculé sur la base du dernier commentaire de l'année écoulée*. Cela semble être une manière inhabituelle de calculer le score, mais ce sont les données extraites, donc nous pouvons les prendre telles quelles pour l'instant.

  ✅ En vous basant sur les autres colonnes de cet ensemble de données, pouvez-vous imaginer une autre manière de calculer le score moyen ?

* `Total_Number_of_Reviews`
  * Le nombre total d'avis reçus par cet hôtel - il n'est pas clair (sans écrire du code) si cela fait référence aux avis présents dans l'ensemble de données.
* `Additional_Number_of_Scoring`
  * Cela signifie qu'un score a été donné mais qu'aucun avis positif ou négatif n'a été écrit par le client.

**Colonnes des avis**

- `Reviewer_Score`
  - C'est une valeur numérique avec au maximum 1 chiffre après la virgule, entre les valeurs minimales et maximales de 2.5 et 10.
  - Il n'est pas expliqué pourquoi 2.5 est le score le plus bas possible.
- `Negative_Review`
  - Si un client n'a rien écrit, ce champ contiendra "**No Negative**".
  - Notez qu'un client peut écrire un avis positif dans la colonne des avis négatifs (par exemple, "il n'y a rien de mauvais dans cet hôtel").
- `Review_Total_Negative_Word_Counts`
  - Un nombre élevé de mots négatifs indique un score plus bas (sans vérifier la tonalité).
- `Positive_Review`
  - Si un client n'a rien écrit, ce champ contiendra "**No Positive**".
  - Notez qu'un client peut écrire un avis négatif dans la colonne des avis positifs (par exemple, "il n'y a rien de bien dans cet hôtel").
- `Review_Total_Positive_Word_Counts`
  - Un nombre élevé de mots positifs indique un score plus élevé (sans vérifier la tonalité).
- `Review_Date` et `days_since_review`
  - Une mesure de fraîcheur ou de vieillissement pourrait être appliquée à un avis (les avis plus anciens pourraient ne pas être aussi précis que les plus récents en raison de changements dans la gestion de l'hôtel, de rénovations, ou de l'ajout d'une piscine, etc.).
- `Tags`
  - Ce sont de courtes descriptions qu'un client peut sélectionner pour décrire le type de client qu'il était (par exemple, solo ou en famille), le type de chambre qu'il avait, la durée de son séjour et la manière dont l'avis a été soumis.
  - Malheureusement, l'utilisation de ces tags est problématique, consultez la section ci-dessous qui discute de leur utilité.

**Colonnes des clients**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Cela pourrait être un facteur dans un modèle de recommandation, par exemple, si vous pouviez déterminer que les clients plus prolifiques avec des centaines d'avis étaient plus susceptibles d'être négatifs que positifs. Cependant, le client de tout avis particulier n'est pas identifié par un code unique, et ne peut donc pas être lié à un ensemble d'avis. Il y a 30 clients avec 100 avis ou plus, mais il est difficile de voir comment cela peut aider le modèle de recommandation.
- `Reviewer_Nationality`
  - Certains pourraient penser que certaines nationalités sont plus susceptibles de donner un avis positif ou négatif en raison d'une inclination nationale. Soyez prudent en intégrant de telles vues anecdotiques dans vos modèles. Ce sont des stéréotypes nationaux (et parfois raciaux), et chaque client était un individu qui a écrit un avis basé sur son expérience. Cela peut avoir été filtré à travers de nombreux prismes tels que ses séjours précédents, la distance parcourue, et son tempérament personnel. Penser que leur nationalité était la raison d'un score d'avis est difficile à justifier.

##### Exemples

| Score moyen | Nombre total d'avis | Score du client | Avis négatif                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Avis positif                     | Tags                                                                                      |
| ----------- | ------------------- | --------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8         | 1945                | 2.5             | Cet endroit n'est actuellement pas un hôtel mais un chantier de construction. J'ai été terrorisé dès le matin et toute la journée par des bruits de travaux inacceptables alors que je me reposais après un long voyage et travaillais dans la chambre. Les ouvriers travaillaient toute la journée, par exemple avec des marteaux-piqueurs dans les chambres adjacentes. J'ai demandé à changer de chambre mais aucune chambre silencieuse n'était disponible. Pour aggraver les choses, j'ai été surfacturé. J'ai quitté l'hôtel le soir car je devais prendre un vol très tôt et j'ai reçu une facture appropriée. Un jour plus tard, l'hôtel a effectué un autre prélèvement sans mon consentement, supérieur au prix réservé. C'est un endroit terrible. Ne vous punissez pas en réservant ici. | Rien. Endroit terrible. Fuyez.   | Voyage d'affaires Couple Chambre double standard Séjour de 2 nuits |

Comme vous pouvez le voir, ce client n'a pas eu un séjour heureux dans cet hôtel. L'hôtel a un bon score moyen de 7.8 et 1945 avis, mais ce client lui a attribué un score de 2.5 et a écrit 115 mots sur la négativité de son séjour. S'il n'a rien écrit dans la colonne Positive_Review, vous pourriez en déduire qu'il n'y avait rien de positif, mais il a écrit 7 mots d'avertissement. Si nous comptions simplement les mots au lieu de leur signification ou de leur sentiment, nous pourrions avoir une vision biaisée de l'intention du client. Étrangement, son score de 2.5 est déroutant, car si ce séjour à l'hôtel était si mauvais, pourquoi lui attribuer des points ? En examinant attentivement l'ensemble de données, vous verrez que le score le plus bas possible est de 2.5, pas 0. Le score le plus élevé possible est de 10.

##### Tags

Comme mentionné ci-dessus, à première vue, l'idée d'utiliser `Tags` pour catégoriser les données semble logique. Malheureusement, ces tags ne sont pas standardisés, ce qui signifie que dans un hôtel donné, les options pourraient être *Chambre simple*, *Chambre double*, et *Chambre twin*, mais dans le prochain hôtel, elles sont *Chambre simple deluxe*, *Chambre queen classique*, et *Chambre king exécutive*. Ces options pourraient être les mêmes, mais il y a tellement de variations que le choix devient :

1. Tenter de convertir tous les termes en une norme unique, ce qui est très difficile, car il n'est pas clair quel serait le chemin de conversion dans chaque cas (par exemple, *Chambre simple classique* correspond à *Chambre simple*, mais *Chambre queen supérieure avec vue sur cour ou ville* est beaucoup plus difficile à mapper).

1. Nous pouvons adopter une approche NLP et mesurer la fréquence de certains termes comme *Solo*, *Voyageur d'affaires*, ou *Famille avec jeunes enfants* lorsqu'ils s'appliquent à chaque hôtel, et intégrer cela dans la recommandation.

Les tags sont généralement (mais pas toujours) un champ unique contenant une liste de 5 à 6 valeurs séparées par des virgules correspondant à *Type de voyage*, *Type de client*, *Type de chambre*, *Nombre de nuits*, et *Type d'appareil utilisé pour soumettre l'avis*. Cependant, comme certains clients ne remplissent pas chaque champ (ils peuvent en laisser un vide), les valeurs ne sont pas toujours dans le même ordre.

Par exemple, prenez *Type de groupe*. Il y a 1025 possibilités uniques dans ce champ de la colonne `Tags`, et malheureusement, seulement certaines d'entre elles se réfèrent à un groupe (certaines concernent le type de chambre, etc.). Si vous filtrez uniquement celles qui mentionnent famille, les résultats contiennent de nombreux types *Chambre familiale*. Si vous incluez le terme *avec*, c'est-à-dire comptez les valeurs *Famille avec*, les résultats sont meilleurs, avec plus de 80 000 des 515 000 résultats contenant la phrase "Famille avec jeunes enfants" ou "Famille avec enfants plus âgés".

Cela signifie que la colonne tags n'est pas complètement inutile pour nous, mais il faudra du travail pour la rendre utile.

##### Score moyen des hôtels

Il y a un certain nombre d'étrangetés ou de divergences dans l'ensemble de données que je ne peux pas expliquer, mais elles sont illustrées ici pour que vous en soyez conscient lorsque vous construisez vos modèles. Si vous trouvez une explication, merci de nous en informer dans la section de discussion !

L'ensemble de données contient les colonnes suivantes relatives au score moyen et au nombre d'avis :

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

L'hôtel avec le plus grand nombre d'avis dans cet ensemble de données est *Britannia International Hotel Canary Wharf* avec 4789 avis sur 515 000. Mais si nous regardons la valeur `Total_Number_of_Reviews` pour cet hôtel, elle est de 9086. Vous pourriez en déduire qu'il y a beaucoup plus de scores sans avis, donc peut-être devrions-nous ajouter la valeur de la colonne `Additional_Number_of_Scoring`. Cette valeur est de 2682, et en l'ajoutant à 4789, nous obtenons 7471, ce qui est encore 1615 de moins que le `Total_Number_of_Reviews`.

Si vous prenez la colonne `Average_Score`, vous pourriez en déduire qu'il s'agit de la moyenne des avis dans l'ensemble de données, mais la description sur Kaggle est "*Score moyen de l'hôtel, calculé sur la base du dernier commentaire de l'année écoulée*". Cela ne semble pas très utile, mais nous pouvons calculer notre propre moyenne basée sur les scores des avis dans l'ensemble de données. En prenant le même hôtel comme exemple, le score moyen de l'hôtel est donné comme 7.1, mais le score calculé (moyenne des scores des clients *dans* l'ensemble de données) est de 6.8. Cela est proche, mais pas identique, et nous pouvons seulement supposer que les scores donnés dans les avis `Additional_Number_of_Scoring` ont augmenté la moyenne à 7.1. Malheureusement, sans moyen de tester ou de prouver cette assertion, il est difficile d'utiliser ou de faire confiance à `Average_Score`, `Additional_Number_of_Scoring` et `Total_Number_of_Reviews` lorsqu'ils sont basés sur, ou se réfèrent à, des données que nous n'avons pas.

Pour compliquer les choses davantage, l'hôtel avec le deuxième plus grand nombre d'avis a un score moyen calculé de 8.12 et le `Average_Score` de l'ensemble de données est de 8.1. Ce score correct est-il une coïncidence ou le premier hôtel est-il une anomalie ?

En supposant que ces hôtels pourraient être des valeurs aberrantes, et que peut-être la plupart des valeurs correspondent (mais certaines ne le font pas pour une raison quelconque), nous écrirons un court programme ensuite pour explorer les valeurs dans l'ensemble de données et déterminer l'utilisation correcte (ou non-utilisation) des valeurs.
> 🚨 Une note de prudence  
>  
> Lorsque vous travaillez avec cet ensemble de données, vous écrirez du code qui calcule quelque chose à partir du texte sans avoir à lire ou analyser le texte vous-même. C'est l'essence du NLP : interpréter le sens ou le sentiment sans qu'un humain ait à le faire. Cependant, il est possible que vous lisiez certains des avis négatifs. Je vous encourage vivement à ne pas le faire, car ce n'est pas nécessaire. Certains d'entre eux sont absurdes ou concernent des critiques négatives de l'hôtel qui sont hors de propos, comme "Le temps n'était pas génial", quelque chose qui échappe au contrôle de l'hôtel, ou de quiconque d'ailleurs. Mais il y a aussi un côté sombre à certains avis. Parfois, les avis négatifs sont racistes, sexistes ou âgistes. C'est regrettable, mais prévisible dans un ensemble de données extrait d'un site web public. Certains auteurs laissent des avis que vous pourriez trouver de mauvais goût, inconfortables ou bouleversants. Il vaut mieux laisser le code mesurer le sentiment plutôt que de les lire vous-même et d'en être affecté. Cela dit, c'est une minorité qui écrit de telles choses, mais ils existent tout de même.
## Exercice - Exploration des données
### Charger les données

C'est suffisant pour examiner visuellement les données, maintenant vous allez écrire du code et obtenir des réponses ! Cette section utilise la bibliothèque pandas. Votre toute première tâche est de vous assurer que vous pouvez charger et lire les données CSV. La bibliothèque pandas dispose d'un chargeur CSV rapide, et le résultat est placé dans un dataframe, comme dans les leçons précédentes. Le CSV que nous chargeons contient plus d'un demi-million de lignes, mais seulement 17 colonnes. Pandas vous offre de nombreuses façons puissantes d'interagir avec un dataframe, y compris la possibilité d'effectuer des opérations sur chaque ligne.

À partir de maintenant dans cette leçon, il y aura des extraits de code, des explications sur le code et des discussions sur ce que signifient les résultats. Utilisez le fichier _notebook.ipynb_ inclus pour votre code.

Commençons par charger le fichier de données que vous allez utiliser :

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Maintenant que les données sont chargées, nous pouvons effectuer certaines opérations dessus. Gardez ce code en haut de votre programme pour la prochaine partie.

## Explorer les données

Dans ce cas, les données sont déjà *propres*, ce qui signifie qu'elles sont prêtes à être utilisées et ne contiennent pas de caractères dans d'autres langues qui pourraient perturber les algorithmes qui attendent uniquement des caractères anglais.

✅ Vous pourriez avoir à travailler avec des données nécessitant un traitement initial pour les formater avant d'appliquer des techniques de PNL, mais pas cette fois. Si vous deviez le faire, comment géreriez-vous les caractères non anglais ?

Prenez un moment pour vous assurer qu'une fois les données chargées, vous pouvez les explorer avec du code. Il est très tentant de se concentrer sur les colonnes `Negative_Review` et `Positive_Review`. Elles sont remplies de texte naturel pour vos algorithmes de PNL. Mais attendez ! Avant de plonger dans la PNL et l'analyse de sentiment, vous devriez suivre le code ci-dessous pour vérifier si les valeurs données dans le dataset correspondent aux valeurs que vous calculez avec pandas.

## Opérations sur le dataframe

La première tâche de cette leçon est de vérifier si les assertions suivantes sont correctes en écrivant du code qui examine le dataframe (sans le modifier).

> Comme pour de nombreuses tâches de programmation, il existe plusieurs façons de les accomplir, mais un bon conseil est de le faire de la manière la plus simple et la plus facile possible, surtout si cela sera plus compréhensible lorsque vous reviendrez à ce code à l'avenir. Avec les dataframes, il existe une API complète qui aura souvent une méthode efficace pour faire ce que vous voulez.

Considérez les questions suivantes comme des tâches de codage et essayez de répondre sans regarder la solution.

1. Affichez la *forme* du dataframe que vous venez de charger (la forme correspond au nombre de lignes et de colonnes).
2. Calculez le nombre de fréquences pour les nationalités des évaluateurs :
   1. Combien de valeurs distinctes y a-t-il pour la colonne `Reviewer_Nationality` et quelles sont-elles ?
   2. Quelle nationalité d'évaluateur est la plus courante dans le dataset (affichez le pays et le nombre d'évaluations) ?
   3. Quelles sont les 10 nationalités les plus fréquentes suivantes et leur nombre de fréquences ?
3. Quel hôtel a été le plus évalué pour chacune des 10 nationalités d'évaluateurs les plus fréquentes ?
4. Combien d'évaluations y a-t-il par hôtel (nombre de fréquences des hôtels) dans le dataset ?
5. Bien qu'il existe une colonne `Average_Score` pour chaque hôtel dans le dataset, vous pouvez également calculer un score moyen (en obtenant la moyenne de tous les scores des évaluateurs dans le dataset pour chaque hôtel). Ajoutez une nouvelle colonne à votre dataframe avec l'en-tête `Calc_Average_Score` contenant cette moyenne calculée.
6. Est-ce que certains hôtels ont le même `Average_Score` (arrondi à une décimale) et `Calc_Average_Score` ?
   1. Essayez d'écrire une fonction Python qui prend une série (ligne) comme argument et compare les valeurs, en affichant un message lorsque les valeurs ne sont pas égales. Ensuite, utilisez la méthode `.apply()` pour traiter chaque ligne avec la fonction.
7. Calculez et affichez combien de lignes ont des valeurs "No Negative" dans la colonne `Negative_Review`.
8. Calculez et affichez combien de lignes ont des valeurs "No Positive" dans la colonne `Positive_Review`.
9. Calculez et affichez combien de lignes ont des valeurs "No Positive" dans la colonne `Positive_Review` **et** des valeurs "No Negative" dans la colonne `Negative_Review`.

### Réponses en code

1. Affichez la *forme* du dataframe que vous venez de charger (la forme correspond au nombre de lignes et de colonnes).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calculez le nombre de fréquences pour les nationalités des évaluateurs :

   1. Combien de valeurs distinctes y a-t-il pour la colonne `Reviewer_Nationality` et quelles sont-elles ?
   2. Quelle nationalité d'évaluateur est la plus courante dans le dataset (affichez le pays et le nombre d'évaluations) ?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Quelles sont les 10 nationalités les plus fréquentes suivantes et leur nombre de fréquences ?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Quel hôtel a été le plus évalué pour chacune des 10 nationalités d'évaluateurs les plus fréquentes ?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Combien d'évaluations y a-t-il par hôtel (nombre de fréquences des hôtels) dans le dataset ?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   Vous remarquerez peut-être que les résultats *comptés dans le dataset* ne correspondent pas à la valeur dans `Total_Number_of_Reviews`. Il est difficile de savoir si cette valeur dans le dataset représente le nombre total d'évaluations que l'hôtel a reçues, mais que toutes n'ont pas été extraites, ou si un autre calcul a été effectué. `Total_Number_of_Reviews` n'est pas utilisé dans le modèle en raison de cette incertitude.

5. Bien qu'il existe une colonne `Average_Score` pour chaque hôtel dans le dataset, vous pouvez également calculer un score moyen (en obtenant la moyenne de tous les scores des évaluateurs dans le dataset pour chaque hôtel). Ajoutez une nouvelle colonne à votre dataframe avec l'en-tête `Calc_Average_Score` contenant cette moyenne calculée. Affichez les colonnes `Hotel_Name`, `Average_Score` et `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Vous pourriez également vous demander pourquoi la valeur `Average_Score` est parfois différente du score moyen calculé. Comme nous ne pouvons pas savoir pourquoi certaines valeurs correspondent, mais d'autres présentent une différence, il est plus sûr dans ce cas d'utiliser les scores des évaluations que nous avons pour calculer la moyenne nous-mêmes. Cela dit, les différences sont généralement très petites, voici les hôtels avec la plus grande déviation entre la moyenne du dataset et la moyenne calculée :

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Avec seulement 1 hôtel ayant une différence de score supérieure à 1, cela signifie que nous pouvons probablement ignorer la différence et utiliser le score moyen calculé.

6. Calculez et affichez combien de lignes ont des valeurs "No Negative" dans la colonne `Negative_Review`.

7. Calculez et affichez combien de lignes ont des valeurs "No Positive" dans la colonne `Positive_Review`.

8. Calculez et affichez combien de lignes ont des valeurs "No Positive" dans la colonne `Positive_Review` **et** des valeurs "No Negative" dans la colonne `Negative_Review`.

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Une autre méthode

Une autre méthode pour compter les éléments sans Lambdas, et utiliser sum pour compter les lignes :

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Vous avez peut-être remarqué qu'il y a 127 lignes qui ont à la fois des valeurs "No Negative" et "No Positive" pour les colonnes `Negative_Review` et `Positive_Review` respectivement. Cela signifie que l'évaluateur a donné à l'hôtel un score numérique, mais a refusé d'écrire une évaluation positive ou négative. Heureusement, cela représente une petite quantité de lignes (127 sur 515738, soit 0,02 %), donc cela ne devrait pas fausser notre modèle ou nos résultats dans une direction particulière, mais vous pourriez ne pas avoir attendu qu'un dataset d'évaluations contienne des lignes sans évaluations, donc cela vaut la peine d'explorer les données pour découvrir des lignes comme celle-ci.

Maintenant que vous avez exploré le dataset, dans la prochaine leçon, vous filtrerez les données et ajouterez une analyse de sentiment.

---
## 🚀Défi

Cette leçon démontre, comme nous l'avons vu dans les leçons précédentes, à quel point il est crucial de comprendre vos données et leurs particularités avant d'effectuer des opérations dessus. Les données basées sur du texte, en particulier, nécessitent une attention particulière. Explorez divers datasets riches en texte et voyez si vous pouvez découvrir des zones qui pourraient introduire des biais ou des sentiments biaisés dans un modèle.

## [Quiz post-lecture](https://ff-quizzes.netlify.app/en/ml/)

## Révision & Auto-apprentissage

Suivez [ce parcours d'apprentissage sur la PNL](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) pour découvrir des outils à essayer lors de la création de modèles basés sur la parole et le texte.

## Devoir 

[NLTK](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle effectuée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.