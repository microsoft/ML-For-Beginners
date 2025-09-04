<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a683e1fe430bb0d4a10b68f6ca15e0a6",
  "translation_date": "2025-09-03T22:36:59+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "fr"
}
-->
# Construire un modèle de régression avec Scikit-learn : préparer et visualiser les données

![Infographie sur la visualisation des données](../../../../translated_images/data-visualization.54e56dded7c1a804d00d027543f2881cb32da73aeadda2d4a4f10f3497526114.fr.png)

Infographie par [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz avant le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/11/)

> ### [Cette leçon est disponible en R !](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introduction

Maintenant que vous êtes équipé des outils nécessaires pour commencer à construire des modèles d'apprentissage automatique avec Scikit-learn, vous êtes prêt à poser des questions à vos données. Lorsque vous travaillez avec des données et appliquez des solutions d'apprentissage automatique, il est très important de savoir poser les bonnes questions pour exploiter pleinement le potentiel de votre ensemble de données.

Dans cette leçon, vous apprendrez :

- Comment préparer vos données pour la construction de modèles.
- Comment utiliser Matplotlib pour la visualisation des données.

## Poser les bonnes questions à vos données

La question que vous souhaitez résoudre déterminera le type d'algorithmes d'apprentissage automatique que vous utiliserez. La qualité de la réponse que vous obtiendrez dépendra fortement de la nature de vos données.

Examinez les [données](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) fournies pour cette leçon. Vous pouvez ouvrir ce fichier .csv dans VS Code. Un rapide coup d'œil montre immédiatement qu'il y a des valeurs manquantes et un mélange de chaînes de caractères et de données numériques. Il y a aussi une colonne étrange appelée 'Package' où les données sont un mélange de 'sacks', 'bins' et d'autres valeurs. Les données, en fait, sont un peu désordonnées.

[![ML pour débutants - Comment analyser et nettoyer un ensemble de données](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML pour débutants - Comment analyser et nettoyer un ensemble de données")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant comment préparer les données pour cette leçon.

En réalité, il est rare de recevoir un ensemble de données entièrement prêt à être utilisé pour créer un modèle d'apprentissage automatique. Dans cette leçon, vous apprendrez à préparer un ensemble de données brut en utilisant des bibliothèques Python standard. Vous apprendrez également différentes techniques pour visualiser les données.

## Étude de cas : 'le marché des citrouilles'

Dans ce dossier, vous trouverez un fichier .csv dans le dossier racine `data` appelé [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) qui contient 1757 lignes de données sur le marché des citrouilles, triées par ville. Ce sont des données brutes extraites des [Rapports standard des marchés terminaux des cultures spécialisées](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribués par le Département de l'Agriculture des États-Unis.

### Préparer les données

Ces données sont dans le domaine public. Elles peuvent être téléchargées en plusieurs fichiers distincts, par ville, depuis le site web de l'USDA. Pour éviter trop de fichiers séparés, nous avons concaténé toutes les données des villes en une seule feuille de calcul, ce qui signifie que nous avons déjà _préparé_ un peu les données. Ensuite, examinons de plus près ces données.

### Les données sur les citrouilles - premières conclusions

Que remarquez-vous à propos de ces données ? Vous avez déjà vu qu'il y a un mélange de chaînes de caractères, de nombres, de valeurs manquantes et de valeurs étranges qu'il faut interpréter.

Quelle question pouvez-vous poser à ces données en utilisant une technique de régression ? Que diriez-vous de "Prédire le prix d'une citrouille en vente pendant un mois donné". En regardant à nouveau les données, il y a des modifications à apporter pour créer la structure de données nécessaire à cette tâche.

## Exercice - analyser les données sur les citrouilles

Utilisons [Pandas](https://pandas.pydata.org/), (le nom signifie `Python Data Analysis`) un outil très utile pour façonner les données, afin d'analyser et de préparer ces données sur les citrouilles.

### Premièrement, vérifier les dates manquantes

Vous devrez d'abord prendre des mesures pour vérifier les dates manquantes :

1. Convertir les dates au format mois (ce sont des dates américaines, donc le format est `MM/DD/YYYY`).
2. Extraire le mois dans une nouvelle colonne.

Ouvrez le fichier _notebook.ipynb_ dans Visual Studio Code et importez la feuille de calcul dans un nouveau dataframe Pandas.

1. Utilisez la fonction `head()` pour afficher les cinq premières lignes.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Quelle fonction utiliseriez-vous pour afficher les cinq dernières lignes ?

1. Vérifiez s'il y a des données manquantes dans le dataframe actuel :

    ```python
    pumpkins.isnull().sum()
    ```

    Il y a des données manquantes, mais peut-être que cela n'aura pas d'importance pour la tâche à accomplir.

1. Pour rendre votre dataframe plus facile à manipuler, sélectionnez uniquement les colonnes dont vous avez besoin, en utilisant la fonction `loc` qui extrait du dataframe original un groupe de lignes (passé en premier paramètre) et de colonnes (passé en second paramètre). L'expression `:` dans le cas ci-dessous signifie "toutes les lignes".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Deuxièmement, déterminer le prix moyen des citrouilles

Réfléchissez à la manière de déterminer le prix moyen d'une citrouille pour un mois donné. Quelles colonnes choisiriez-vous pour cette tâche ? Indice : vous aurez besoin de 3 colonnes.

Solution : prenez la moyenne des colonnes `Low Price` et `High Price` pour remplir la nouvelle colonne Price, et convertissez la colonne Date pour n'afficher que le mois. Heureusement, selon la vérification ci-dessus, il n'y a pas de données manquantes pour les dates ou les prix.

1. Pour calculer la moyenne, ajoutez le code suivant :

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ N'hésitez pas à imprimer les données que vous souhaitez vérifier en utilisant `print(month)`.

2. Maintenant, copiez vos données converties dans un nouveau dataframe Pandas :

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    En imprimant votre dataframe, vous verrez un ensemble de données propre et ordonné sur lequel vous pourrez construire votre nouveau modèle de régression.

### Mais attendez ! Il y a quelque chose d'étrange ici

Si vous regardez la colonne `Package`, les citrouilles sont vendues dans de nombreuses configurations différentes. Certaines sont vendues en mesures de '1 1/9 bushel', d'autres en '1/2 bushel', certaines par citrouille, certaines par livre, et d'autres dans de grandes boîtes de largeurs variées.

> Les citrouilles semblent très difficiles à peser de manière cohérente

En examinant les données originales, il est intéressant de noter que tout ce qui a `Unit of Sale` égal à 'EACH' ou 'PER BIN' a également le type `Package` par pouce, par bin, ou 'each'. Les citrouilles semblent être très difficiles à peser de manière cohérente, alors filtrons-les en sélectionnant uniquement les citrouilles contenant le mot 'bushel' dans leur colonne `Package`.

1. Ajoutez un filtre en haut du fichier, sous l'importation initiale du fichier .csv :

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Si vous imprimez les données maintenant, vous verrez que vous obtenez uniquement les 415 lignes de données contenant des citrouilles par bushel.

### Mais attendez ! Il y a encore une chose à faire

Avez-vous remarqué que la quantité de bushel varie selon les lignes ? Vous devez normaliser les prix pour afficher les prix par bushel, alors faites quelques calculs pour les standardiser.

1. Ajoutez ces lignes après le bloc créant le dataframe new_pumpkins :

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Selon [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), le poids d'un bushel dépend du type de produit, car c'est une mesure de volume. "Un bushel de tomates, par exemple, est censé peser 56 livres... Les feuilles et les légumes verts prennent plus de place avec moins de poids, donc un bushel d'épinards ne pèse que 20 livres." Tout cela est assez compliqué ! Ne nous embêtons pas à convertir un bushel en livres, et affichons plutôt les prix par bushel. Toute cette étude des bushels de citrouilles montre cependant à quel point il est important de comprendre la nature de vos données !

Maintenant, vous pouvez analyser les prix par unité en fonction de leur mesure en bushel. Si vous imprimez les données une fois de plus, vous verrez comment elles sont standardisées.

✅ Avez-vous remarqué que les citrouilles vendues par demi-bushel sont très chères ? Pouvez-vous comprendre pourquoi ? Indice : les petites citrouilles sont beaucoup plus chères que les grandes, probablement parce qu'il y en a beaucoup plus par bushel, étant donné l'espace inutilisé occupé par une grande citrouille creuse pour tarte.

## Stratégies de visualisation

Une partie du rôle du data scientist est de démontrer la qualité et la nature des données avec lesquelles il travaille. Pour ce faire, il crée souvent des visualisations intéressantes, comme des graphiques, des diagrammes et des tableaux, montrant différents aspects des données. De cette manière, il peut montrer visuellement des relations et des lacunes qui sont autrement difficiles à découvrir.

[![ML pour débutants - Comment visualiser les données avec Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML pour débutants - Comment visualiser les données avec Matplotlib")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant comment visualiser les données pour cette leçon.

Les visualisations peuvent également aider à déterminer la technique d'apprentissage automatique la plus appropriée pour les données. Un nuage de points qui semble suivre une ligne, par exemple, indique que les données sont un bon candidat pour un exercice de régression linéaire.

Une bibliothèque de visualisation de données qui fonctionne bien dans les notebooks Jupyter est [Matplotlib](https://matplotlib.org/) (que vous avez également vue dans la leçon précédente).

> Obtenez plus d'expérience avec la visualisation des données dans [ces tutoriels](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Exercice - expérimenter avec Matplotlib

Essayez de créer des graphiques simples pour afficher le nouveau dataframe que vous venez de créer. Que montrerait un graphique linéaire de base ?

1. Importez Matplotlib en haut du fichier, sous l'importation de Pandas :

    ```python
    import matplotlib.pyplot as plt
    ```

1. Relancez tout le notebook pour actualiser.
1. En bas du notebook, ajoutez une cellule pour tracer les données sous forme de boîte :

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Un nuage de points montrant la relation entre le prix et le mois](../../../../translated_images/scatterplot.b6868f44cbd2051c6680ccdbb1510697d06a3ff6cd4abda656f5009c0ed4e3fc.fr.png)

    Ce graphique est-il utile ? Quelque chose vous surprend-il ?

    Ce n'est pas particulièrement utile car tout ce qu'il fait est d'afficher vos données sous forme de points répartis dans un mois donné.

### Rendez-le utile

Pour obtenir des graphiques affichant des données utiles, vous devez généralement regrouper les données d'une manière ou d'une autre. Essayons de créer un graphique où l'axe y montre les mois et les données démontrent la distribution des données.

1. Ajoutez une cellule pour créer un graphique en barres groupées :

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Un graphique en barres montrant la relation entre le prix et le mois](../../../../translated_images/barchart.a833ea9194346d769c77a3a870f7d8aee51574cd1138ca902e5500830a41cbce.fr.png)

    Ce graphique est une visualisation de données plus utile ! Il semble indiquer que le prix le plus élevé des citrouilles se produit en septembre et octobre. Cela correspond-il à vos attentes ? Pourquoi ou pourquoi pas ?

---

## 🚀Défi

Explorez les différents types de visualisation que Matplotlib propose. Quels types sont les plus appropriés pour les problèmes de régression ?

## [Quiz après le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/12/)

## Révision et auto-apprentissage

Examinez les nombreuses façons de visualiser les données. Faites une liste des différentes bibliothèques disponibles et notez celles qui sont les meilleures pour certains types de tâches, par exemple les visualisations 2D contre les visualisations 3D. Que découvrez-vous ?

## Devoir

[Explorer la visualisation](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.