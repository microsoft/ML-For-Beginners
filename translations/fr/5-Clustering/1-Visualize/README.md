<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0ab69b161efd7a41d325ee28b29415d7",
  "translation_date": "2025-09-03T22:57:40+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "fr"
}
-->
# Introduction au clustering

Le clustering est un type d'[apprentissage non supervisé](https://wikipedia.org/wiki/Apprentissage_non_supervis%C3%A9) qui suppose qu'un ensemble de données est non étiqueté ou que ses entrées ne sont pas associées à des sorties prédéfinies. Il utilise divers algorithmes pour trier les données non étiquetées et fournir des regroupements en fonction des motifs qu'il discerne dans les données.

[![No One Like You par PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You par PSquare")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo. Pendant que vous étudiez l'apprentissage automatique avec le clustering, profitez de quelques morceaux de Dance Hall nigérian - voici une chanson très appréciée de 2014 par PSquare.

## [Quiz avant le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/27/)

### Introduction

Le [clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) est très utile pour explorer les données. Voyons s'il peut aider à découvrir des tendances et des motifs dans la manière dont les audiences nigérianes consomment de la musique.

✅ Prenez une minute pour réfléchir aux utilisations du clustering. Dans la vie quotidienne, le clustering se produit chaque fois que vous avez une pile de linge et que vous devez trier les vêtements des membres de votre famille 🧦👕👖🩲. En science des données, le clustering se produit lorsqu'on essaie d'analyser les préférences d'un utilisateur ou de déterminer les caractéristiques d'un ensemble de données non étiqueté. Le clustering, d'une certaine manière, aide à donner un sens au chaos, comme un tiroir à chaussettes.

[![Introduction au ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction au clustering")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : John Guttag du MIT introduit le clustering.

Dans un cadre professionnel, le clustering peut être utilisé pour déterminer des choses comme la segmentation de marché, par exemple pour savoir quels groupes d'âge achètent quels articles. Une autre utilisation serait la détection d'anomalies, peut-être pour détecter des fraudes dans un ensemble de données de transactions par carte de crédit. Ou encore, vous pourriez utiliser le clustering pour identifier des tumeurs dans un lot de scans médicaux.

✅ Prenez une minute pour réfléchir à la manière dont vous avez pu rencontrer le clustering 'dans la nature', dans un contexte bancaire, e-commerce ou commercial.

> 🎓 Fait intéressant, l'analyse de clusters a vu le jour dans les domaines de l'anthropologie et de la psychologie dans les années 1930. Pouvez-vous imaginer comment elle aurait pu être utilisée ?

Alternativement, vous pourriez l'utiliser pour regrouper des résultats de recherche - par exemple des liens d'achat, des images ou des avis. Le clustering est utile lorsque vous avez un grand ensemble de données que vous souhaitez réduire et sur lequel vous voulez effectuer une analyse plus détaillée. Cette technique peut donc être utilisée pour mieux comprendre les données avant de construire d'autres modèles.

✅ Une fois vos données organisées en clusters, vous leur attribuez un identifiant de cluster. Cette technique peut être utile pour préserver la confidentialité d'un ensemble de données ; vous pouvez alors vous référer à un point de données par son identifiant de cluster, plutôt que par des données identifiables plus révélatrices. Pouvez-vous penser à d'autres raisons pour lesquelles vous préféreriez utiliser un identifiant de cluster plutôt que d'autres éléments du cluster pour l'identifier ?

Approfondissez votre compréhension des techniques de clustering dans ce [module d'apprentissage](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Premiers pas avec le clustering

[Scikit-learn propose une large gamme](https://scikit-learn.org/stable/modules/clustering.html) de méthodes pour effectuer du clustering. Le type que vous choisissez dépendra de votre cas d'utilisation. Selon la documentation, chaque méthode présente divers avantages. Voici un tableau simplifié des méthodes prises en charge par Scikit-learn et leurs cas d'utilisation appropriés :

| Nom de la méthode            | Cas d'utilisation                                                    |
| :--------------------------- | :------------------------------------------------------------------- |
| K-Means                      | usage général, inductif                                              |
| Propagation d'affinité       | nombreux clusters, clusters inégaux, inductif                       |
| Mean-shift                   | nombreux clusters, clusters inégaux, inductif                       |
| Clustering spectral          | quelques clusters, clusters égaux, transductif                      |
| Clustering hiérarchique Ward | nombreux clusters contraints, transductif                           |
| Clustering agglomératif      | nombreux clusters contraints, distances non euclidiennes, transductif |
| DBSCAN                       | géométrie non plate, clusters inégaux, transductif                  |
| OPTICS                       | géométrie non plate, clusters inégaux avec densité variable, transductif |
| Mélanges gaussiens           | géométrie plate, inductif                                            |
| BIRCH                        | grand ensemble de données avec des valeurs aberrantes, inductif     |

> 🎓 La manière dont nous créons des clusters dépend beaucoup de la façon dont nous regroupons les points de données. Décomposons quelques notions :
>
> 🎓 ['Transductif' vs. 'inductif'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> L'inférence transductive est dérivée des cas d'entraînement observés qui correspondent à des cas de test spécifiques. L'inférence inductive est dérivée des cas d'entraînement qui mènent à des règles générales, lesquelles sont ensuite appliquées aux cas de test.
> 
> Un exemple : Imaginez que vous avez un ensemble de données partiellement étiqueté. Certains éléments sont des 'disques', d'autres des 'CD', et certains sont non étiquetés. Votre tâche est de fournir des étiquettes pour les éléments non étiquetés. Si vous choisissez une approche inductive, vous entraîneriez un modèle en recherchant des 'disques' et des 'CD', et appliqueriez ces étiquettes aux données non étiquetées. Cette approche aurait du mal à classer des éléments qui sont en réalité des 'cassettes'. Une approche transductive, en revanche, gère ces données inconnues plus efficacement en regroupant des éléments similaires et en appliquant une étiquette à un groupe. Dans ce cas, les clusters pourraient refléter 'objets musicaux ronds' et 'objets musicaux carrés'.
> 
> 🎓 ['Géométrie non plate' vs. 'plate'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Tirée de la terminologie mathématique, la géométrie non plate vs. plate fait référence à la mesure des distances entre les points par des méthodes géométriques 'plates' ([euclidiennes](https://wikipedia.org/wiki/G%C3%A9om%C3%A9trie_euclidienne)) ou 'non plates' (non euclidiennes).
>
>'Plate' dans ce contexte fait référence à la géométrie euclidienne (dont certaines parties sont enseignées comme géométrie 'plane'), et 'non plate' fait référence à la géométrie non euclidienne. Quel est le lien entre la géométrie et l'apprentissage automatique ? Eh bien, en tant que deux domaines enracinés dans les mathématiques, il doit y avoir une manière commune de mesurer les distances entre les points dans les clusters, et cela peut être fait de manière 'plate' ou 'non plate', selon la nature des données. Les [distances euclidiennes](https://wikipedia.org/wiki/Distance_euclidienne) sont mesurées comme la longueur d'un segment de ligne entre deux points. Les [distances non euclidiennes](https://wikipedia.org/wiki/G%C3%A9om%C3%A9trie_non_euclidienne) sont mesurées le long d'une courbe. Si vos données, visualisées, semblent ne pas exister sur un plan, vous pourriez avoir besoin d'utiliser un algorithme spécialisé pour les traiter.
>
![Infographie Géométrie Plate vs Non Plate](../../../../translated_images/flat-nonflat.d1c8c6e2a96110c1d57fa0b72913f6aab3c245478524d25baf7f4a18efcde224.fr.png)
> Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Les clusters sont définis par leur matrice de distances, c'est-à-dire les distances entre les points. Cette distance peut être mesurée de plusieurs façons. Les clusters euclidiens sont définis par la moyenne des valeurs des points et contiennent un 'centroïde' ou point central. Les distances sont donc mesurées par rapport à ce centroïde. Les distances non euclidiennes font référence aux 'clustroïdes', le point le plus proche des autres points. Les clustroïdes peuvent à leur tour être définis de différentes manières.
> 
> 🎓 ['Contraint'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> Le [clustering contraint](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduit l'apprentissage 'semi-supervisé' dans cette méthode non supervisée. Les relations entre les points sont marquées comme 'ne peut pas lier' ou 'doit lier', ce qui impose certaines règles à l'ensemble de données.
>
>Un exemple : Si un algorithme est laissé libre sur un lot de données non étiquetées ou semi-étiquetées, les clusters qu'il produit peuvent être de mauvaise qualité. Dans l'exemple ci-dessus, les clusters pourraient regrouper 'objets musicaux ronds', 'objets musicaux carrés', 'objets triangulaires' et 'biscuits'. Si on lui donne des contraintes ou des règles à suivre ("l'objet doit être en plastique", "l'objet doit pouvoir produire de la musique"), cela peut aider à 'contraindre' l'algorithme à faire de meilleurs choix.
> 
> 🎓 'Densité'
> 
> Les données 'bruyantes' sont considérées comme 'denses'. Les distances entre les points dans chacun de ses clusters peuvent s'avérer, après examen, plus ou moins denses, ou 'concentrées', et ces données doivent donc être analysées avec la méthode de clustering appropriée. [Cet article](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) montre la différence entre l'utilisation du clustering K-Means et des algorithmes HDBSCAN pour explorer un ensemble de données bruyant avec une densité de cluster inégale.

## Algorithmes de clustering

Il existe plus de 100 algorithmes de clustering, et leur utilisation dépend de la nature des données disponibles. Discutons de quelques-uns des principaux :

- **Clustering hiérarchique**. Si un objet est classé par sa proximité avec un objet voisin, plutôt qu'avec un objet plus éloigné, les clusters sont formés en fonction de la distance de leurs membres par rapport aux autres objets. Le clustering agglomératif de Scikit-learn est hiérarchique.

   ![Infographie Clustering Hiérarchique](../../../../translated_images/hierarchical.bf59403aa43c8c47493bfdf1cc25230f26e45f4e38a3d62e8769cd324129ac15.fr.png)
   > Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering par centroïde**. Cet algorithme populaire nécessite de choisir 'k', ou le nombre de clusters à former, après quoi l'algorithme détermine le point central d'un cluster et regroupe les données autour de ce point. Le [clustering K-means](https://wikipedia.org/wiki/K-means_clustering) est une version populaire du clustering par centroïde. Le centre est déterminé par la moyenne la plus proche, d'où son nom. La distance au cluster est minimisée.

   ![Infographie Clustering par Centroïde](../../../../translated_images/centroid.097fde836cf6c9187d0b2033e9f94441829f9d86f4f0b1604dd4b3d1931aee34.fr.png)
   > Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering basé sur la distribution**. Basé sur la modélisation statistique, le clustering basé sur la distribution se concentre sur la détermination de la probabilité qu'un point de données appartienne à un cluster, et l'y attribue en conséquence. Les méthodes de mélange gaussien appartiennent à ce type.

- **Clustering basé sur la densité**. Les points de données sont attribués à des clusters en fonction de leur densité, ou de leur regroupement les uns autour des autres. Les points de données éloignés du groupe sont considérés comme des valeurs aberrantes ou du bruit. DBSCAN, Mean-shift et OPTICS appartiennent à ce type de clustering.

- **Clustering basé sur une grille**. Pour les ensembles de données multidimensionnels, une grille est créée et les données sont réparties entre les cellules de la grille, créant ainsi des clusters.

## Exercice - regroupez vos données

Le clustering en tant que technique est grandement facilité par une bonne visualisation, alors commençons par visualiser nos données musicales. Cet exercice nous aidera à décider quelle méthode de clustering utiliser le plus efficacement en fonction de la nature de ces données.

1. Ouvrez le fichier [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) dans ce dossier.

1. Importez le package `Seaborn` pour une bonne visualisation des données.

    ```python
    !pip install seaborn
    ```

1. Ajoutez les données des chansons à partir de [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Chargez un dataframe avec des données sur les chansons. Préparez-vous à explorer ces données en important les bibliothèques et en affichant les données :

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Vérifiez les premières lignes de données :

    |     | nom                     | album                        | artiste              | genre_principal_artiste | date_sortie | durée | popularité | dansabilité | acoustique | énergie | instrumentalité | vivacité | volume   | discours    | tempo   | signature_temps |
    | --- | ------------------------ | ---------------------------- | -------------------- | ----------------------- | ------------ | ------ | ---------- | ----------- | ---------- | ------- | --------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino        | r&b alternatif          | 2019         | 144000 | 48         | 0.666       | 0.851      | 0.42    | 0.534           | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine)  | afropop                 | 2020         | 89488  | 30         | 0.71        | 0.0822     | 0.683   | 0.000169        | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Obtenez des informations sur le dataframe en appelant `info()` :

    ```python
    df.info()
    ```

   Le résultat ressemble à ceci :

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Vérifiez les valeurs nulles en appelant `isnull()` et en vérifiant que la somme est égale à 0 :

    ```python
    df.isnull().sum()
    ```

    Tout semble correct :

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Décrivez les données :

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Si nous travaillons avec le clustering, une méthode non supervisée qui ne nécessite pas de données étiquetées, pourquoi montrons-nous ces données avec des étiquettes ? Lors de la phase d'exploration des données, elles sont utiles, mais elles ne sont pas nécessaires pour que les algorithmes de clustering fonctionnent. Vous pourriez tout aussi bien supprimer les en-têtes de colonnes et vous référer aux données par numéro de colonne.

Regardez les valeurs générales des données. Notez que la popularité peut être '0', ce qui montre des chansons sans classement. Supprimons ces données bientôt.

1. Utilisez un barplot pour découvrir les genres les plus populaires :

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/popular.9c48d84b3386705f98bf44e26e9655bee9eb7c849d73be65195e37895bfedb5d.fr.png)

✅ Si vous souhaitez voir plus de valeurs en haut du classement, modifiez le top `[:5]` pour une valeur plus grande ou supprimez-le pour voir tout.

Notez que lorsque le genre principal est décrit comme 'Missing', cela signifie que Spotify ne l'a pas classifié. Supprimons-le.

1. Supprimez les données manquantes en les filtrant :

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Re-vérifiez maintenant les genres :

    ![most popular](../../../../translated_images/all-genres.1d56ef06cefbfcd61183023834ed3cb891a5ee638a3ba5c924b3151bf80208d7.fr.png)

1. Les trois genres principaux dominent largement ce dataset. Concentrons-nous sur `afro dancehall`, `afropop` et `nigerian pop`, et filtrons également le dataset pour supprimer tout ce qui a une valeur de popularité égale à 0 (ce qui signifie qu'il n'a pas été classé dans le dataset et peut être considéré comme du bruit pour nos objectifs) :

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Faites un test rapide pour voir si les données présentent une corrélation particulièrement forte :

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/correlation.a9356bb798f5eea51f47185968e1ebac5c078c92fce9931e28ccf0d7fab71c2b.fr.png)

    La seule corrélation forte est entre `energy` et `loudness`, ce qui n'est pas très surprenant, étant donné que la musique forte est généralement assez énergique. Sinon, les corrélations sont relativement faibles. Il sera intéressant de voir ce qu'un algorithme de clustering peut tirer de ces données.

    > 🎓 Notez que la corrélation n'implique pas la causalité ! Nous avons une preuve de corrélation mais aucune preuve de causalité. Un [site web amusant](https://tylervigen.com/spurious-correlations) propose des visuels qui soulignent ce point.

Y a-t-il une convergence dans ce dataset autour de la popularité perçue d'une chanson et de sa capacité à faire danser ? Une FacetGrid montre qu'il existe des cercles concentriques qui s'alignent, quel que soit le genre. Est-il possible que les goûts nigérians convergent à un certain niveau de capacité à faire danser pour ce genre ?  

✅ Essayez différents points de données (énergie, loudness, speechiness) et plus ou différents genres musicaux. Que pouvez-vous découvrir ? Consultez le tableau `df.describe()` pour voir la répartition générale des points de données.

### Exercice - distribution des données

Ces trois genres sont-ils significativement différents dans la perception de leur capacité à faire danser, en fonction de leur popularité ?

1. Examinez la distribution des données de nos trois genres principaux pour la popularité et la capacité à faire danser le long d'un axe x et y donné.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Vous pouvez découvrir des cercles concentriques autour d'un point de convergence général, montrant la distribution des points.

    > 🎓 Notez que cet exemple utilise un graphique KDE (Kernel Density Estimate) qui représente les données à l'aide d'une courbe de densité de probabilité continue. Cela nous permet d'interpréter les données lorsque nous travaillons avec plusieurs distributions.

    En général, les trois genres s'alignent vaguement en termes de popularité et de capacité à faire danser. Déterminer des clusters dans ces données faiblement alignées sera un défi :

    ![distribution](../../../../translated_images/distribution.9be11df42356ca958dc8e06e87865e09d77cab78f94fe4fea8a1e6796c64dc4b.fr.png)

1. Créez un scatter plot :

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Un scatter plot des mêmes axes montre un schéma similaire de convergence.

    ![Facetgrid](../../../../translated_images/facetgrid.9b2e65ce707eba1f983b7cdfed5d952e60f385947afa3011df6e3cc7d200eb5b.fr.png)

En général, pour le clustering, vous pouvez utiliser des scatter plots pour montrer des clusters de données, donc maîtriser ce type de visualisation est très utile. Dans la prochaine leçon, nous utiliserons ces données filtrées et appliquerons le clustering k-means pour découvrir des groupes dans ces données qui semblent se chevaucher de manière intéressante.

---

## 🚀Défi

En préparation de la prochaine leçon, créez un tableau sur les différents algorithmes de clustering que vous pourriez découvrir et utiliser dans un environnement de production. Quels types de problèmes le clustering cherche-t-il à résoudre ?

## [Quiz post-lecture](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/28/)

## Révision et auto-apprentissage

Avant d'appliquer des algorithmes de clustering, comme nous l'avons appris, il est judicieux de comprendre la nature de votre dataset. Lisez-en davantage sur ce sujet [ici](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Cet article utile](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) vous guide à travers les différentes façons dont divers algorithmes de clustering se comportent, en fonction des formes des données.

## Devoir

[Recherchez d'autres visualisations pour le clustering](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction humaine professionnelle. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.