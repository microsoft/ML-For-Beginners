# Analyse de sentiment avec les avis d'hôtels - traitement des données

Dans cette section, vous utiliserez les techniques des leçons précédentes pour effectuer une analyse exploratoire des données sur un grand ensemble de données. Une fois que vous aurez une bonne compréhension de l'utilité des différentes colonnes, vous apprendrez :

- comment supprimer les colonnes inutiles
- comment calculer de nouvelles données basées sur les colonnes existantes
- comment sauvegarder l'ensemble de données résultant pour l'utiliser dans le défi final

## [Quiz pré-conférence](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Introduction

Jusqu'à présent, vous avez appris que les données textuelles sont très différentes des types de données numériques. Si le texte a été écrit ou prononcé par un humain, il peut être analysé pour trouver des motifs et des fréquences, des sentiments et des significations. Cette leçon vous plonge dans un ensemble de données réel avec un véritable défi : **[515K Avis d'Hôtels en Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** et comprend une [licence CC0 : Domaine public](https://creativecommons.org/publicdomain/zero/1.0/). Il a été extrait de Booking.com à partir de sources publiques. Le créateur de l'ensemble de données est Jiashen Liu.

### Préparation

Vous aurez besoin de :

* La capacité d'exécuter des notebooks .ipynb en utilisant Python 3
* pandas
* NLTK, [que vous devez installer localement](https://www.nltk.org/install.html)
* L'ensemble de données qui est disponible sur Kaggle [515K Avis d'Hôtels en Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Il fait environ 230 Mo une fois décompressé. Téléchargez-le dans le dossier racine `/data` associé à ces leçons de NLP.

## Analyse exploratoire des données

Ce défi suppose que vous construisez un bot de recommandation d'hôtels utilisant l'analyse de sentiment et les scores des avis des clients. L'ensemble de données que vous allez utiliser comprend des avis sur 1493 hôtels différents dans 6 villes.

En utilisant Python, un ensemble de données d'avis d'hôtels et l'analyse de sentiment de NLTK, vous pourriez découvrir :

* Quels sont les mots et phrases les plus fréquemment utilisés dans les avis ?
* Les *tags* officiels décrivant un hôtel sont-ils corrélés avec les scores des avis (par exemple, les avis plus négatifs pour un hôtel particulier sont-ils pour *Famille avec de jeunes enfants* plutôt que pour *Voyageur solo*, ce qui indiquerait peut-être qu'il est mieux pour les *Voyageurs solo* ?) 
* Les scores de sentiment de NLTK 's'accordent-ils' avec le score numérique de l'examinateur de l'hôtel ?

#### Ensemble de données

Explorons l'ensemble de données que vous avez téléchargé et sauvegardé localement. Ouvrez le fichier dans un éditeur comme VS Code ou même Excel.

Les en-têtes dans l'ensemble de données sont les suivants :

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Voici comment ils sont regroupés d'une manière qui pourrait être plus facile à examiner :
##### Colonnes de l'hôtel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * En utilisant *lat* et *lng*, vous pourriez tracer une carte avec Python montrant les emplacements des hôtels (peut-être codée par couleur pour les avis négatifs et positifs)
  * Hotel_Address n'est pas évidemment utile pour nous, et nous allons probablement le remplacer par un pays pour un tri et une recherche plus faciles

**Colonnes de méta-avis sur l'hôtel**

* `Average_Score`
  * Selon le créateur de l'ensemble de données, cette colonne est le *Score moyen de l'hôtel, calculé sur la base du dernier commentaire dans l'année écoulée*. Cela semble être une manière inhabituelle de calculer le score, mais c'est les données extraites donc nous pouvons le prendre pour ce qu'il est pour l'instant.
  
  ✅ En vous basant sur les autres colonnes de ces données, pouvez-vous penser à une autre façon de calculer le score moyen ?

* `Total_Number_of_Reviews`
  * Le nombre total d'avis que cet hôtel a reçus - il n'est pas clair (sans écrire un peu de code) si cela fait référence aux avis dans l'ensemble de données.
* `Additional_Number_of_Scoring`
  * Cela signifie qu'un score d'avis a été donné mais qu'aucun avis positif ou négatif n'a été écrit par l'examinateur

**Colonnes d'avis**

- `Reviewer_Score`
  - Il s'agit d'une valeur numérique avec au maximum 1 décimale entre les valeurs minimales et maximales 2.5 et 10
  - Il n'est pas expliqué pourquoi 2.5 est le score le plus bas possible
- `Negative_Review`
  - Si un examinateur n'a rien écrit, ce champ aura "**No Negative**"
  - Notez qu'un examinateur peut écrire un avis positif dans la colonne Negative review (par exemple, "il n'y a rien de mauvais dans cet hôtel")
- `Review_Total_Negative_Word_Counts`
  - Un nombre de mots négatifs plus élevé indique un score plus bas (sans vérifier la sentimentalité)
- `Positive_Review`
  - Si un examinateur n'a rien écrit, ce champ aura "**No Positive**"
  - Notez qu'un examinateur peut écrire un avis négatif dans la colonne Positive review (par exemple, "il n'y a rien de bon dans cet hôtel")
- `Review_Total_Positive_Word_Counts`
  - Un nombre de mots positifs plus élevé indique un score plus élevé (sans vérifier la sentimentalité)
- `Review_Date` et `days_since_review`
  - Une mesure de fraîcheur ou de stagnation pourrait être appliquée à un avis (les avis plus anciens pourraient ne pas être aussi précis que les plus récents en raison de changements de gestion d'hôtel, de rénovations effectuées, ou d'une piscine ajoutée, etc.)
- `Tags`
  - Ce sont de courts descripteurs qu'un examinateur peut sélectionner pour décrire le type de client qu'il était (par exemple, solo ou famille), le type de chambre qu'il avait, la durée du séjour et comment l'avis a été soumis.
  - Malheureusement, l'utilisation de ces tags pose problème, consultez la section ci-dessous qui discute de leur utilité

**Colonnes d'examinateur**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Cela pourrait être un facteur dans un modèle de recommandation, par exemple, si vous pouviez déterminer que les examinateurs plus prolifiques avec des centaines d'avis étaient plus susceptibles d'être négatifs plutôt que positifs. Cependant, l'examinateur d'un avis particulier n'est pas identifié par un code unique, et ne peut donc pas être lié à un ensemble d'avis. Il y a 30 examinateurs avec 100 avis ou plus, mais il est difficile de voir comment cela peut aider le modèle de recommandation.
- `Reviewer_Nationality`
  - Certaines personnes pourraient penser que certaines nationalités sont plus susceptibles de donner un avis positif ou négatif en raison d'une inclination nationale. Faites attention à ne pas intégrer de telles vues anecdotiques dans vos modèles. Ce sont des stéréotypes nationaux (et parfois raciaux), et chaque examinateur était un individu qui a écrit un avis basé sur son expérience. Cela peut avoir été filtré à travers de nombreuses lentilles telles que ses précédents séjours à l'hôtel, la distance parcourue, et son tempérament personnel. Penser que leur nationalité était la raison d'un score d'avis est difficile à justifier.

##### Exemples

| Score Moyen | Nombre Total d'Avis | Score de l'Examinateur | Avis Négatif <br />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Avis Positif                 | Tags                                                                                      |
| ------------ | --------------------- | ---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8          | 1945                  | 2.5                    | Actuellement, ce n'est pas un hôtel mais un chantier de construction. J'ai été terrorisé dès le matin et toute la journée par un bruit de construction inacceptable tout en essayant de me reposer après un long voyage et de travailler dans la chambre. Des personnes travaillaient toute la journée avec des perceuses dans les chambres adjacentes. J'ai demandé un changement de chambre, mais aucune chambre silencieuse n'était disponible. Pour aggraver les choses, j'ai été surfacturé. J'ai quitté l'hôtel le soir puisque je devais partir très tôt pour un vol et j'ai reçu une facture appropriée. Un jour plus tard, l'hôtel a effectué un autre prélèvement sans mon consentement, supérieur au prix réservé. C'est un endroit terrible. Ne vous punissez pas en réservant ici. | Rien. Endroit terrible. Restez à l'écart. | Voyage d'affaires                                Couple, Chambre Double Standard, Séjour de 2 nuits |

Comme vous pouvez le voir, ce client n'a pas eu un séjour heureux dans cet hôtel. L'hôtel a un bon score moyen de 7.8 et 1945 avis, mais cet examinateur lui a donné 2.5 et a écrit 115 mots sur la négativité de son séjour. S'il n'avait rien écrit du tout dans la colonne Positive_Review, vous pourriez supposer qu'il n'y avait rien de positif, mais hélas, il a écrit 7 mots d'avertissement. Si nous ne comptions que les mots au lieu de la signification ou du sentiment des mots, nous pourrions avoir une vision biaisée de l'intention de l'examinateur. Étrangement, leur score de 2.5 est déroutant, car si ce séjour à l'hôtel était si mauvais, pourquoi lui donner des points du tout ? En examinant de près l'ensemble de données, vous verrez que le score le plus bas possible est 2.5, pas 0. Le score le plus élevé possible est 10.

##### Tags

Comme mentionné ci-dessus, à première vue, l'idée d'utiliser `Tags` pour catégoriser les données a du sens. Malheureusement, ces tags ne sont pas standardisés, ce qui signifie que dans un hôtel donné, les options pourraient être *Chambre simple*, *Chambre twin*, et *Chambre double*, mais dans l'hôtel suivant, elles sont *Chambre Simple Deluxe*, *Chambre Reine Classique*, et *Chambre Roi Exécutive*. Ces options pourraient être les mêmes, mais il y a tellement de variations que le choix devient :

1. Essayer de changer tous les termes en une seule norme, ce qui est très difficile, car il n'est pas clair quel serait le chemin de conversion dans chaque cas (par exemple, *Chambre simple classique* correspond à *Chambre simple* mais *Chambre Reine Supérieure avec Jardin Cour ou Vue sur la Ville* est beaucoup plus difficile à mapper)

2. Nous pouvons adopter une approche NLP et mesurer la fréquence de certains termes comme *Solo*, *Voyageur d'affaires*, ou *Famille avec de jeunes enfants* tels qu'ils s'appliquent à chaque hôtel, et en tenir compte dans la recommandation  

Les tags sont généralement (mais pas toujours) un champ unique contenant une liste de 5 à 6 valeurs séparées par des virgules correspondant à *Type de voyage*, *Type de clients*, *Type de chambre*, *Nombre de nuits*, et *Type de dispositif sur lequel l'avis a été soumis*. Cependant, comme certains examinateurs ne remplissent pas chaque champ (ils peuvent en laisser un vide), les valeurs ne sont pas toujours dans le même ordre.

Prenons un exemple, le champ *Type de groupe*. Il y a 1025 possibilités uniques dans ce champ de la colonne `Tags`, et malheureusement, seules certaines d'entre elles font référence à un groupe (certaines sont le type de chambre, etc.). Si vous filtrez uniquement celles qui mentionnent la famille, les résultats contiennent de nombreux types de résultats *Chambre familiale*. Si vous incluez le terme *avec*, c'est-à-dire compter les valeurs *Famille avec*, les résultats sont meilleurs, avec plus de 80 000 des 515 000 résultats contenant la phrase "Famille avec de jeunes enfants" ou "Famille avec des enfants plus âgés".

Cela signifie que la colonne des tags n'est pas complètement inutile pour nous, mais il faudra du travail pour la rendre utile.

##### Score moyen de l'hôtel

Il y a un certain nombre d'étrangetés ou de divergences avec l'ensemble de données que je ne peux pas comprendre, mais qui sont illustrées ici afin que vous en soyez conscient lors de la construction de vos modèles. Si vous le comprenez, merci de nous le faire savoir dans la section discussion !

L'ensemble de données a les colonnes suivantes concernant le score moyen et le nombre d'avis :

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

L'hôtel avec le plus d'avis dans cet ensemble de données est *Britannia International Hotel Canary Wharf* avec 4789 avis sur 515 000. Mais si nous regardons la valeur `Total_Number_of_Reviews` pour cet hôtel, elle est de 9086. Vous pourriez supposer qu'il y a beaucoup plus de scores sans avis, donc peut-être devrions-nous ajouter la valeur de la colonne `Additional_Number_of_Scoring`. Cette valeur est de 2682, et l'ajouter à 4789 nous donne 7471, ce qui est encore 1615 de moins que le `Total_Number_of_Reviews`. 

Si vous prenez les colonnes `Average_Score`, vous pourriez supposer qu'il s'agit de la moyenne des avis dans l'ensemble de données, mais la description de Kaggle est "*Score moyen de l'hôtel, calculé sur la base du dernier commentaire dans l'année écoulée*". Cela ne semble pas très utile, mais nous pouvons calculer notre propre moyenne basée sur les scores des avis dans l'ensemble de données. En utilisant le même hôtel comme exemple, le score moyen de l'hôtel est donné comme 7.1 mais le score calculé (score moyen des examinateurs *dans* l'ensemble de données) est de 6.8. C'est proche, mais pas la même valeur, et nous ne pouvons que deviner que les scores donnés dans les avis `Additional_Number_of_Scoring` ont augmenté la moyenne à 7.1. Malheureusement, sans moyen de tester ou de prouver cette assertion, il est difficile d'utiliser ou de faire confiance à `Average_Score`, `Additional_Number_of_Scoring` et `Total_Number_of_Reviews` lorsqu'ils sont basés sur, ou se réfèrent à, des données que nous n'avons pas.

Pour compliquer encore les choses, l'hôtel avec le deuxième plus grand nombre d'avis a un score moyen calculé de 8.12 et l'ensemble de données `Average_Score` est de 8.1. Ce score correct est-il une coïncidence ou le premier hôtel est-il une anomalie ? 

Dans l'éventualité où ces hôtels pourraient être des cas extrêmes, et que peut-être la plupart des valeurs s'additionnent (mais certaines ne le font pas pour une raison quelconque), nous allons écrire un court programme ensuite pour explorer les valeurs dans l'ensemble de données et déterminer l'utilisation correcte (ou non-utilisation) des valeurs.

> 🚨 Une note de prudence
>
> Lorsque vous travaillez avec cet ensemble de données, vous écrirez du code qui calcule quelque chose à partir du texte sans avoir à lire ou analyser le texte vous-même. C'est l'essence du NLP, interpréter la signification ou le sentiment sans qu'un humain le fasse. Cependant, il est possible que vous lisiez certains des avis négatifs. Je vous conseillerais de ne pas le faire, car vous n'avez pas besoin de. Certains d'entre eux sont ridicules ou des avis négatifs sans pertinence, comme "Le temps n'était pas super", quelque chose qui échappe au contrôle de l'hôtel, ou en effet, de quiconque. Mais il y a aussi un côté sombre à certains avis. Parfois, les avis négatifs sont racistes, sexistes ou âgistes. C'est malheureux mais à prévoir dans un ensemble de données extrait d'un site web public. Certains examinateurs laissent des avis que vous trouveriez de mauvais goût, inconfortables ou troublants. Il vaut mieux laisser le code mesurer le sentiment plutôt que de les lire vous-même et d'être contrarié. Cela dit, c'est une minorité qui écrit de telles choses, mais elles existent néanmoins. 

## Exercice - Exploration des données
### Charger les données

C'est assez d'examiner les données visuellement, maintenant vous allez écrire un peu de code et obtenir des réponses ! Cette section utilise la bibliothèque pandas. Votre toute première tâche est de vous assurer que vous pouvez charger et lire les données CSV. La bibliothèque pandas a un chargeur CSV rapide, et le résultat est placé dans un dataframe, comme dans les leçons précédentes. Le CSV que nous chargeons a plus d'un demi-million de lignes, mais seulement 17 colonnes. Pandas vous offre de nombreuses façons puissantes d'interagir avec un dataframe, y compris la capacité d'effectuer des opérations sur chaque ligne. 

À partir de maintenant dans cette leçon, il y aura des extraits de code et quelques explications du code ainsi que des discussions sur ce que les résultats signifient. Utilisez le _notebook.ipynb_ inclus pour votre code.

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

Dans ce cas, les données sont déjà *propres*, cela signifie qu'elles sont prêtes à être utilisées, et n'ont pas de caractères dans d'autres langues qui pourraient perturber les algorithmes s'attendant uniquement à des caractères anglais. 

✅ Vous pourriez avoir à travailler avec des données qui nécessitaient un traitement initial pour les formater avant d'appliquer des techniques NLP, mais pas cette fois. Si vous deviez le faire, comment géreriez-vous les caractères non anglais ?

Prenez un moment pour vous assurer qu'une fois les données chargées, vous pouvez les
les lignes ont des valeurs de colonne `Positive_Review` de "Aucun Positif" 9. Calculez et imprimez combien de lignes ont des valeurs de colonne `Positive_Review` de "Aucun Positif" **et** des valeurs `Negative_Review` de "Aucun Négatif" ### Réponses au code 1. Imprimez la *forme* du cadre de données que vous venez de charger (la forme est le nombre de lignes et de colonnes) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Calculez le nombre de fréquences pour les nationalités des examinateurs : 1. Combien de valeurs distinctes y a-t-il pour la colonne `Reviewer_Nationality` et quelles sont-elles ? 2. Quelle nationalité d'examinateur est la plus courante dans l'ensemble de données (imprimez le pays et le nombre de critiques) ? ```python
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
   ``` 3. Quelles sont les 10 nationalités les plus fréquemment trouvées, et leur nombre de fréquences ? ```python
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
      ``` 3. Quel était l'hôtel le plus fréquemment évalué pour chacune des 10 nationalités d'examinateurs les plus représentées ? ```python
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
   ``` 4. Combien de critiques y a-t-il par hôtel (nombre de fréquences de l'hôtel) dans l'ensemble de données ? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Nom_Hôtel | Nombre_Total_de_Critiques | Total_Critiques_Trouvées | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hôtel Wagner | 135 | 10 | | Hôtel Gallitzinberg | 173 | 8 | Vous remarquerez peut-être que les résultats *comptés dans l'ensemble de données* ne correspondent pas à la valeur dans `Total_Number_of_Reviews`. Il n'est pas clair si cette valeur dans l'ensemble de données représentait le nombre total de critiques que l'hôtel avait, mais que toutes n'ont pas été extraites, ou un autre calcul. `Total_Number_of_Reviews` n'est pas utilisé dans le modèle en raison de cette incertitude. 5. Bien qu'il y ait une colonne `Average_Score` pour chaque hôtel dans l'ensemble de données, vous pouvez également calculer un score moyen (obtenant la moyenne de tous les scores des examinateurs dans l'ensemble de données pour chaque hôtel). Ajoutez une nouvelle colonne à votre cadre de données avec l'en-tête de colonne `Calc_Average_Score` qui contient cette moyenne calculée. Imprimez les colonnes `Hotel_Name`, `Average_Score`, et `Calc_Average_Score`. ```python
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
   ``` Vous vous demandez peut-être également pourquoi la valeur `Average_Score` est parfois différente du score moyen calculé. Comme nous ne pouvons pas savoir pourquoi certaines des valeurs correspondent, mais d'autres ont une différence, il est plus sûr dans ce cas d'utiliser les scores de critique que nous avons pour calculer la moyenne nous-mêmes. Cela dit, les différences sont généralement très petites, voici les hôtels avec la plus grande déviation par rapport à la moyenne de l'ensemble de données et à la moyenne calculée : | Différence_Score_Moyen | Score_Moyen | Calc_Average_Score | Nom_Hôtel | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hôtel Stendhal Place Vendôme Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendôme Hôtel | | -0.5 | 7.0 | 7.5 | Hôtel Royal Élysées | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Opéra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Châteaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | Avec seulement 1 hôtel ayant une différence de score supérieure à 1, cela signifie que nous pouvons probablement ignorer la différence et utiliser le score moyen calculé. 6. Calculez et imprimez combien de lignes ont des valeurs de colonne `Negative_Review` de "Aucun Négatif" 7. Calculez et imprimez combien de lignes ont des valeurs de colonne `Positive_Review` de "Aucun Positif" 8. Calculez et imprimez combien de lignes ont des valeurs de colonne `Positive_Review` de "Aucun Positif" **et** des valeurs `Negative_Review` de "Aucun Négatif" ```python
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
   ``` ## Une autre façon Une autre façon de compter les éléments sans Lambdas, et d'utiliser sum pour compter les lignes : ```python
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
   ``` Vous avez peut-être remarqué qu'il y a 127 lignes qui ont à la fois des valeurs "Aucun Négatif" et "Aucun Positif" pour les colonnes `Negative_Review` et `Positive_Review` respectivement. Cela signifie que l'examinateur a donné à l'hôtel un score numérique, mais a refusé d'écrire soit une critique positive, soit une critique négative. Heureusement, c'est un petit nombre de lignes (127 sur 515738, ou 0,02 %), donc cela ne faussera probablement pas notre modèle ou nos résultats dans une direction particulière, mais vous ne vous attendiez peut-être pas à ce qu'un ensemble de données de critiques ait des lignes sans critiques, donc il vaut la peine d'explorer les données pour découvrir des lignes comme celle-ci. Maintenant que vous avez exploré l'ensemble de données, dans la prochaine leçon, vous filtrerez les données et ajouterez une analyse de sentiment. --- ## 🚀Défi Cette leçon démontre, comme nous l'avons vu dans les leçons précédentes, à quel point il est crucial de comprendre vos données et ses caprices avant d'effectuer des opérations dessus. Les données textuelles, en particulier, nécessitent un examen attentif. Fouillez à travers divers ensembles de données riches en texte et voyez si vous pouvez découvrir des domaines qui pourraient introduire des biais ou des sentiments faussés dans un modèle. ## [Quiz post-lecture](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Révision & Auto-apprentissage Suivez [ce parcours d'apprentissage sur le NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) pour découvrir des outils à essayer lors de la construction de modèles lourds en discours et en texte. ## Devoir [NLTK](assignment.md) Veuillez écrire la sortie de gauche à droite.

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatisés basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des erreurs d'interprétation résultant de l'utilisation de cette traduction.