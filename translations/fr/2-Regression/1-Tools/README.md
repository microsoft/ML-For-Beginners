# Commencez avec Python et Scikit-learn pour les modèles de régression

![Résumé des régressions dans un sketchnote](../../../../translated_images/fr/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote par [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz avant la leçon](https://ff-quizzes.netlify.app/en/ml/)

> ### [Cette leçon est disponible en R !](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introduction

Dans ces quatre leçons, vous allez découvrir comment construire des modèles de régression. Nous discuterons brièvement de leur utilité. Mais avant de faire quoi que ce soit, assurez-vous d'avoir les bons outils en place pour commencer le processus !

Dans cette leçon, vous apprendrez à :

- Configurer votre ordinateur pour des tâches d'apprentissage automatique local.
- Travailler avec Jupyter Notebooks.
- Utiliser Scikit-learn, y compris l'installation.
- Explorer la régression linéaire avec un exercice pratique.

## Installations et configurations

[![ML pour débutants - Préparez vos outils pour construire des modèles d'apprentissage automatique](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML pour débutants - Préparez vos outils pour construire des modèles d'apprentissage automatique")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant la configuration de votre ordinateur pour le ML.

1. **Installez Python**. Assurez-vous que [Python](https://www.python.org/downloads/) est installé sur votre ordinateur. Vous utiliserez Python pour de nombreuses tâches de science des données et d'apprentissage automatique. La plupart des systèmes informatiques incluent déjà une installation de Python. Il existe aussi des [packs de codage Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) utiles pour faciliter la configuration pour certains utilisateurs.

   Certaines utilisations de Python nécessitent cependant une version du logiciel, tandis que d'autres en requièrent une différente. Pour cette raison, il est utile de travailler dans un [environnement virtuel](https://docs.python.org/3/library/venv.html).

2. **Installez Visual Studio Code**. Assurez-vous d'avoir Visual Studio Code installé sur votre ordinateur. Suivez ces instructions pour [installer Visual Studio Code](https://code.visualstudio.com/) pour l'installation de base. Vous allez utiliser Python dans Visual Studio Code dans ce cours, donc vous voudrez peut-être vous familiariser avec la façon de [configurer Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) pour le développement Python.

   > Familiarisez-vous avec Python en parcourant cette collection de [modules d'apprentissage](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configurez Python avec Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configurez Python avec Visual Studio Code")
   >
   > 🎥 Cliquez sur l'image ci-dessus pour une vidéo : utiliser Python dans VS Code.

3. **Installez Scikit-learn**, en suivant [ces instructions](https://scikit-learn.org/stable/install.html). Comme vous devez vous assurer d'utiliser Python 3, il est recommandé d'utiliser un environnement virtuel. Notez, si vous installez cette bibliothèque sur un Mac M1, des instructions spéciales sont disponibles sur la page liée ci-dessus.

1. **Installez Jupyter Notebook**. Vous devrez [installer le package Jupyter](https://pypi.org/project/jupyter/).

## Votre environnement d'écriture ML

Vous allez utiliser des **notebooks** pour développer votre code Python et créer des modèles d'apprentissage automatique. Ce type de fichier est un outil courant pour les data scientists, et il se reconnaît par son suffixe ou extension `.ipynb`.

Les notebooks sont un environnement interactif qui permet au développeur de coder tout en ajoutant des notes et en rédigeant de la documentation autour du code, ce qui est très utile pour les projets expérimentaux ou de recherche.

[![ML pour débutants - Configurez Jupyter Notebooks pour commencer à construire des modèles de régression](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML pour débutants - Configurez Jupyter Notebooks pour commencer à construire des modèles de régression")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant cet exercice.

### Exercice - travaillez avec un notebook

Dans ce dossier, vous trouverez le fichier _notebook.ipynb_.

1. Ouvrez _notebook.ipynb_ dans Visual Studio Code.

   Un serveur Jupyter va démarrer avec Python 3+ lancé. Vous trouverez des zones du notebook qui peuvent être `exécutées`, des morceaux de code. Vous pouvez exécuter un bloc de code en sélectionnant l'icône qui ressemble à un bouton de lecture.

1. Sélectionnez l'icône `md` et ajoutez un peu de markdown, ainsi que le texte suivant : **# Bienvenue dans votre notebook**.

   Ensuite, ajoutez un peu de code Python.

1. Tapez **print('hello notebook')** dans le bloc de code.
1. Sélectionnez la flèche pour exécuter le code.

   Vous devriez voir la ligne imprimée :

    ```output
    hello notebook
    ```

![VS Code avec un notebook ouvert](../../../../translated_images/fr/notebook.4a3ee31f396b8832.webp)

Vous pouvez alterner votre code avec des commentaires pour auto-documenter le notebook.

✅ Réfléchissez un instant à quel point l'environnement de travail d'un développeur web est différent de celui d'un data scientist.

## Prise en main de Scikit-learn

Maintenant que Python est configuré dans votre environnement local, et que vous êtes à l'aise avec Jupyter Notebooks, familiarisons-nous de la même façon avec Scikit-learn (prononcez `sci` comme dans `science`). Scikit-learn fournit une [API étendue](https://scikit-learn.org/stable/modules/classes.html#api-ref) pour vous aider à réaliser des tâches de ML.

Selon leur [site web](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn est une bibliothèque open source d'apprentissage automatique qui prend en charge l'apprentissage supervisé et non supervisé. Elle fournit également divers outils pour l'ajustement de modèles, le prétraitement des données, la sélection et l'évaluation de modèles, ainsi que de nombreuses autres utilités."

Dans ce cours, vous utiliserez Scikit-learn et d'autres outils pour construire des modèles d'apprentissage automatique pour effectuer ce que nous appelons des tâches de "machine learning traditionnel". Nous avons délibérément évité les réseaux neuronaux et l'apprentissage profond, car ils sont mieux abordés dans notre futur programme "IA pour débutants".

Scikit-learn facilite la construction et l'évaluation des modèles pour une utilisation pratique. Il se concentre principalement sur l'utilisation de données numériques et contient plusieurs ensembles de données préconçus pour servir d'outils d'apprentissage. Il inclut aussi des modèles préconstruits pour que les étudiants puissent les essayer. Explorons d'abord le processus de chargement des données préemballées et l'utilisation d'un estimateur intégré pour un premier modèle ML avec Scikit-learn et des données basiques.

## Exercice - votre premier notebook Scikit-learn

> Ce tutoriel s'inspire de l'[exemple de régression linéaire](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) sur le site de Scikit-learn.


[![ML pour débutants - Votre premier projet de régression linéaire en Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML pour débutants - Votre premier projet de régression linéaire en Python")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant cet exercice.

Dans le fichier _notebook.ipynb_ associé à cette leçon, videz toutes les cellules en appuyant sur l'icône de la 'poubelle'.

Dans cette section, vous allez travailler avec un petit ensemble de données sur le diabète intégré à Scikit-learn à des fins d'apprentissage. Imaginez que vous vouliez tester un traitement pour des patients diabétiques. Les modèles d'apprentissage automatique pourraient vous aider à déterminer quels patients répondraient mieux au traitement, en fonction de combinaisons de variables. Même un modèle de régression très basique, une fois visualisé, pourrait montrer des informations sur les variables qui vous aideraient à organiser vos essais cliniques théoriques.

✅ Il existe de nombreux types de méthodes de régression, et celle que vous choisissez dépend de la réponse que vous cherchez. Si vous voulez prédire la taille probable d'une personne à un âge donné, vous utiliserez la régression linéaire, car vous cherchez une **valeur numérique**. Si vous vous intéressez à savoir si un type de cuisine doit être considéré comme végan ou non, vous recherchez une **attribution de catégorie**, donc vous utiliseriez la régression logistique. Vous en apprendrez davantage sur la régression logistique plus tard. Réfléchissez un peu à quelques questions que vous pouvez poser aux données, et à laquelle de ces méthodes serait la plus appropriée.

Commençons cette tâche.

### Importer des bibliothèques

Pour cette tâche, nous allons importer quelques bibliothèques :

- **matplotlib**. C'est un [outil de graphique](https://matplotlib.org/) utile que nous utiliserons pour créer un graphique en ligne.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) est une bibliothèque utile pour manipuler des données numériques en Python.
- **sklearn**. C'est la bibliothèque [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importez quelques bibliothèques pour aider dans vos tâches.

1. Ajoutez les imports en tapant le code suivant :

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Ci-dessus, vous importez `matplotlib`, `numpy` et vous importez `datasets`, `linear_model` et `model_selection` de `sklearn`. `model_selection` est utilisé pour séparer les données en ensembles d'entraînement et de test.

### Le jeu de données sur le diabète

Le [jeu de données diabète intégré](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) comprend 442 échantillons de données autour du diabète, avec 10 variables de caractéristiques, dont certaines incluent :

- âge : âge en années
- bmi : indice de masse corporelle
- bp : pression sanguine moyenne
- s1 tc : cellules T (un type de globules blancs)

✅ Ce jeu de données comprend la notion de 'sexe' comme variable caractéristique importante pour la recherche autour du diabète. Beaucoup de jeux de données médicales incluent ce type de classification binaire. Réfléchissez un peu à la façon dont ce type de catégorisation pourrait exclure certaines parties d'une population des traitements.

Maintenant, chargez les données X et y.

> 🎓 Rappelez-vous, il s'agit d'un apprentissage supervisé, et nous avons besoin d'une cible nommée 'y'.

Dans une nouvelle cellule de code, chargez le jeu de données diabète en appelant `load_diabetes()`. L'entrée `return_X_y=True` signifie que `X` sera une matrice de données, et que `y` sera la cible de la régression.

1. Ajoutez quelques commandes print pour afficher la forme de la matrice de données et son premier élément :

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Ce que vous obtenez en réponse est un tuple. Ce que vous faites, c'est attribuer les deux premières valeurs du tuple à `X` et `y` respectivement. Apprenez-en plus [sur les tuples](https://wikipedia.org/wiki/Tuple).

    Vous pouvez voir que ces données ont 442 items formés en tableaux de 10 éléments :

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Réfléchissez un peu à la relation entre les données et la cible de régression. La régression linéaire prédit les relations entre la caractéristique X et la variable cible y. Pouvez-vous trouver la [cible](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) pour le jeu de données diabète dans la documentation ? Que démontre ce jeu de données, étant donné cette cible ?

2. Ensuite, sélectionnez une portion de ce jeu de données à tracer en sélectionnant la 3e colonne du jeu de données. Vous pouvez le faire en utilisant l'opérateur `:` pour sélectionner toutes les lignes, puis en sélectionnant la 3e colonne en utilisant l'index (2). Vous pouvez aussi remodeler les données pour qu'elles soient un tableau 2D - comme requis pour le tracé - en utilisant `reshape(n_rows, n_columns)`. Si un des paramètres est -1, la dimension correspondante est calculée automatiquement.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ À tout moment, affichez les données pour vérifier leur forme.

3. Maintenant que vous avez des données prêtes à être tracées, vous pouvez voir si une machine peut aider à déterminer une séparation logique parmi les nombres de ce dataset. Pour ce faire, vous devez diviser à la fois les données (X) et la cible (y) en ensembles de test et d'entraînement. Scikit-learn offre un moyen simple de faire cela ; vous pouvez diviser vos données de test à un point donné.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Maintenant vous êtes prêt à entraîner votre modèle ! Chargez le modèle de régression linéaire et entraînez-le avec vos ensembles d'entraînement X et y en utilisant `model.fit()` :

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` est une fonction que vous verrez dans de nombreuses bibliothèques ML telles que TensorFlow

5. Puis, créez une prédiction en utilisant les données de test, avec la fonction `predict()`. Cela sera utilisé pour tracer la ligne entre les groupes de données

    ```python
    y_pred = model.predict(X_test)
    ```

6. Il est maintenant temps d'afficher les données dans un graphique. Matplotlib est un outil très utile pour cette tâche. Créez un nuage de points de toutes les données de test X et y, et utilisez la prédiction pour tracer une ligne à l'endroit le plus approprié, entre les groupes de données du modèle.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![Un nuage de points montrant des données autour du diabète](../../../../translated_images/fr/scatterplot.ad8b356bcbb33be6.webp)


   ✅ Réfléchissez un peu à ce qui se passe ici. Une ligne droite traverse de nombreux petits points de données, mais que fait-elle exactement ? Pouvez-vous voir comment vous devriez pouvoir utiliser cette ligne pour prédire où un nouveau point de données non vu devrait se situer par rapport à l'axe y du graphique ? Essayez de mettre en mots l'utilisation pratique de ce modèle.

Félicitations, vous avez construit votre premier modèle de régression linéaire, créé une prédiction avec celui-ci, et l'avez affiché dans un graphique !

---
## 🚀Défi

Tracez une variable différente de ce jeu de données. Astuce : modifiez cette ligne : `X = X[:,2]`. Étant donné la cible de ce jeu de données, que pouvez-vous découvrir sur la progression du diabète en tant que maladie ?
## [Quiz post-conférence](https://ff-quizzes.netlify.app/en/ml/)

## Revue & Auto-apprentissage

Dans ce tutoriel, vous avez travaillé avec une régression linéaire simple, plutôt qu’avec une régression univariée ou multiple. Lisez un peu sur les différences entre ces méthodes, ou jetez un œil à [cette vidéo](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Lisez davantage à propos du concept de régression et réfléchissez aux types de questions auxquelles cette technique peut répondre. Suivez ce [tutoriel](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) pour approfondir votre compréhension.

## Devoir

[Un autre jeu de données](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Avertissement** :
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforçions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source faisant autorité. Pour les informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous ne saurions être tenus responsables des malentendus ou erreurs d'interprétation découlant de l'utilisation de cette traduction.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->