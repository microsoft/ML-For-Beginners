# Techniques d'apprentissage automatique

Le processus de construction, d'utilisation et de maintenance des modèles d'apprentissage automatique et des données qu'ils utilisent est très différent de nombreux autres flux de travail de développement. Dans cette leçon, nous allons démystifier le processus et présenter les principales techniques que vous devez connaître. Vous allez :

- Comprendre les processus qui sous-tendent l'apprentissage automatique à un niveau très élevé.
- Explorer les concepts de base tels que les "modèles", les "prédictions" et les "données d'entraînement".

## [Quiz de prélecture](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/7/)

## Introduction

À un niveau élevé, la création de processus d'apprentissage machine (ML) comprend un certain nombre d'étapes :

1. **Décider de la question**. La plupart des processus ML commencent par poser une question à laquelle un simple programme conditionnel ou un moteur à base de règles ne peut répondre. Ces questions tournent souvent autour de prédictions basées sur une collection de données.
2. **Collecte et préparation des données**. Pour être en mesure de répondre à votre question, vous avez besoin de données. La qualité et, parfois, la quantité de vos données détermineront dans quelle mesure vous pourrez répondre à votre question initiale. La visualisation des données est un aspect important de cette phase. Cette phase comprend également la division des données en un groupe d'entraînement et un groupe de test pour construire un modèle.
3. **Choisissez une méthode de formation**. En fonction de votre question et de la nature de vos données, vous devez choisir la méthode d'entraînement d'un modèle pour refléter au mieux vos données et faire des prédictions précises par rapport à celles-ci. C'est la partie de votre processus ML qui nécessite une expertise spécifique et, souvent, une quantité considérable d'expérimentation.
4. **Formation du modèle**. En utilisant vos données de formation, vous utiliserez divers algorithmes pour former un modèle pour reconnaître les modèles dans les données. Le modèle peut s'appuyer sur des pondérations internes qui peuvent être ajustées pour privilégier certaines parties des données par rapport à d'autres afin de construire un meilleur modèle.
5. **Evaluez le modèle**. Vous utilisez des données jamais vues auparavant (vos données de test) à partir de votre ensemble collecté pour voir comment le modèle se comporte.
6. **Réglage des paramètres**. En fonction des performances de votre modèle, vous pouvez refaire le processus en utilisant différents paramètres, ou variables, qui contrôlent le comportement des algorithmes utilisés pour former le modèle.
7. **Prédire**. Utilisez de nouvelles entrées pour tester la précision de votre modèle.

## Quelle question poser ?

Les ordinateurs sont particulièrement doués pour découvrir des modèles cachés dans les données. Cette fonctionnalité est très utile pour les chercheurs qui ont des questions sur un domaine donné auxquelles il n'est pas facile de répondre en créant un moteur de règles basé sur des conditions. Dans le cadre d'une tâche actuarielle, par exemple, un spécialiste des données pourrait être en mesure de construire des règles artisanales concernant la mortalité des fumeurs par rapport aux non-fumeurs.

Toutefois, lorsque de nombreuses autres variables entrent dans l'équation, un modèle ML peut s'avérer plus efficace pour prédire les taux de mortalité futurs sur la base des antécédents médicaux. Un exemple plus réjouissant pourrait être de faire des prévisions météorologiques pour le mois d'avril dans un endroit donné, sur la base de données comprenant la latitude, la longitude, le changement climatique, la proximité de l'océan, les schémas du courant-jet, etc.

✅ Ce [diaporama](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sur les modèles météorologiques offre une perspective historique de l'utilisation du ML dans l'analyse météorologique.    

## Tâches préalables à la construction

Avant de commencer à construire votre modèle, vous devez accomplir plusieurs tâches. Pour tester votre question et formuler une hypothèse sur la base des prédictions d'un modèle, vous devez identifier et configurer plusieurs éléments.

### Données

Pour pouvoir répondre avec certitude à votre question, vous avez besoin d'une bonne quantité de données du bon type. Il y a deux choses que vous devez faire à ce stade :

- **Collecter des données**. En gardant à l'esprit la leçon précédente sur l'équité dans l'analyse des données, collectez vos données avec soin. Soyez conscient des sources de ces données, de tout biais inhérent qu'elles pourraient avoir, et documentez leur origine.
- **Préparez les données**. Le processus de préparation des données comporte plusieurs étapes. Vous devrez peut-être rassembler les données et les normaliser si elles proviennent de sources diverses. Vous pouvez améliorer la qualité et la quantité des données à l'aide de diverses méthodes telles que la conversion de chaînes de caractères en nombres (comme nous le faisons dans [Clustering](../../5-Clustering/1-Visualize/README.md)). Vous pouvez également générer de nouvelles données, basées sur les données originales (comme nous le faisons dans [Classification](../../4-Classification/1-Introduction/README.md)). Vous pouvez nettoyer et modifier les données (comme nous le ferons avant la leçon [Application Web](../../3-Web-App/README.md)). Enfin, vous pouvez également avoir besoin de les rendre aléatoires et de les mélanger, en fonction de vos techniques de formation.

✅ Après avoir collecté et traité vos données, prenez un moment pour voir si leur forme vous permettra de répondre à la question que vous vous êtes posée. Il se peut que les données ne soient pas performantes dans votre tâche donnée, comme nous le découvrons dans nos leçons de [Clustering](../../5-Clustering/1-Visualize/README.md) !

### Caractéristiques et cible

Une [caractéristique] (https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) est une propriété mesurable de vos données. Dans de nombreux ensembles de données, elle est exprimée sous la forme d'un titre de colonne comme "date", "taille" ou "couleur". Votre variable caractéristique, généralement représentée par `X` dans le code, représente la variable d'entrée qui sera utilisée pour former le modèle.

Une cible est une chose que vous essayez de prédire. La cible, généralement représentée par `y` dans le code, représente la réponse à la question que vous essayez de poser à vos données : en décembre, quelle **couleur** de citrouille sera la moins chère ? à San Francisco, quels quartiers auront les meilleurs **prix** immobiliers ? Parfois, la cible est également appelée attribut label.

### Sélection de votre variable caractéristique

🎓 **Sélection de caractéristiques et extraction de caractéristiques** Comment savoir quelle variable choisir lors de la construction d'un modèle ? Vous passerez probablement par un processus de sélection de caractéristiques ou d'extraction de caractéristiques afin de choisir les bonnes variables pour le modèle le plus performant. Ce n'est cependant pas la même chose : "L'extraction de caractéristiques crée de nouvelles caractéristiques à partir de fonctions des caractéristiques d'origine, alors que la sélection de caractéristiques renvoie un sous-ensemble des caractéristiques." ([source](https://wikipedia.org/wiki/Feature_selection))

### Visualisez vos données

Un aspect important de la boîte à outils du data scientist est le pouvoir de visualiser les données à l'aide de plusieurs excellentes bibliothèques telles que Seaborn ou MatPlotLib. La représentation visuelle de vos données peut vous permettre de découvrir des corrélations cachées dont vous pouvez tirer parti. Vos visualisations peuvent également vous aider à découvrir des données biaisées ou déséquilibrées (comme nous le découvrons dans [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Divisez votre ensemble de données

Avant la formation, vous devez diviser votre ensemble de données en deux ou plusieurs parties de taille inégale qui représentent toujours bien les données.

- **Entraînement**. Cette partie de l'ensemble de données est adaptée à votre modèle pour l'entraîner. Cet ensemble constitue la majorité de l'ensemble de données original.
- **Testing**. Un ensemble de données de test est un groupe indépendant de données, souvent recueillies à partir des données d'origine, que vous utilisez pour confirmer la performance du modèle construit.
- **Validation**. Un ensemble de validation est un groupe indépendant plus petit d'exemples que vous utilisez pour ajuster les hyperparamètres du modèle, ou l'architecture, afin d'améliorer le modèle. En fonction de la taille de vos données et de la question que vous posez, vous n'aurez peut-être pas besoin de construire ce troisième ensemble (comme nous l'indiquons dans [Prédiction des séries temporelles](../../7-TimeSeries/1-Introduction/README.md)).

## Construction d'un modèle

À partir de vos données d'entraînement, votre objectif est de construire un modèle, ou une représentation statistique de vos données, en utilisant divers algorithmes pour **l'entraîner**. L'entraînement d'un modèle l'expose aux données et lui permet de formuler des hypothèses sur les modèles perçus qu'il découvre, valide, et accepte ou rejette.

### Décider d'une méthode d'entraînement

En fonction de votre question et de la nature de vos données, vous choisirez une méthode d'entraînement. En parcourant la [documentation de Scikit-learn] (https://scikit-learn.org/stable/user_guide.html) - que nous utilisons dans ce cours - vous pouvez explorer de nombreuses façons d'entraîner un modèle. En fonction de votre expérience, vous devrez peut-être essayer plusieurs méthodes différentes pour construire le meilleur modèle. Il est probable que vous passiez par un processus au cours duquel les spécialistes des données évaluent les performances d'un modèle en lui fournissant des données non vues, en vérifiant la précision, les biais et autres problèmes de dégradation de la qualité, et en sélectionnant la méthode d'entraînement la plus appropriée pour la tâche à accomplir.

### Former un modèle

Armé de vos données d'entraînement, vous êtes prêt à les "adapter" pour créer un modèle. Vous remarquerez que dans de nombreuses bibliothèques ML, vous trouverez le code " model.fit ". C'est à ce moment-là que vous envoyez votre variable caractéristique sous forme de tableau de valeurs (généralement " X ") et une variable cible (généralement " y ").

### Évaluer le modèle

Une fois le processus d'apprentissage terminé (l'apprentissage d'un grand modèle peut nécessiter de nombreuses itérations, ou "époques"), vous pourrez évaluer la qualité du modèle en utilisant des données de test pour mesurer ses performances. Ces données sont un sous-ensemble des données d'origine que le modèle n'a pas analysé auparavant. Vous pouvez imprimer un tableau de mesures de la qualité de votre modèle.

🎓 **Ajustement de modèle**

Dans le contexte de l'apprentissage automatique, l'ajustement du modèle fait référence à la précision de la fonction sous-jacente du modèle lorsqu'il tente d'analyser des données qui ne lui sont pas familières.

🎓 **L'underfitting** et le **overfitting** sont des problèmes courants qui dégradent la qualité du modèle, car le modèle s'ajuste soit pas assez bien, soit trop bien. Cela amène le modèle à faire des prédictions trop proches ou trop éloignées de ses données d'apprentissage. Un modèle surajusté prédit trop bien les données d'apprentissage car il a trop bien appris les détails et le bruit des données. Un modèle sous-adapté n'est pas précis, car il ne peut analyser avec précision ni ses données d'apprentissage ni les données qu'il n'a pas encore "vues".

![modèle d'overfitting](images/overfitting.png)
> Infographie par [Jen Looper](https://twitter.com/jenlooper)

## Réglage des paramètres

Une fois la formation initiale terminée, observez la qualité du modèle et envisagez de l'améliorer en modifiant ses "hyperparamètres". Pour en savoir plus sur le processus [dans la documentation] (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prédiction

C'est le moment où vous pouvez utiliser des données totalement nouvelles pour tester la précision de votre modèle. Dans un contexte de ML "appliqué", où vous construisez des ressources Web pour utiliser le modèle en production, ce processus peut impliquer la collecte de données utilisateur (une pression sur un bouton, par exemple) pour définir une variable et l'envoyer au modèle pour inférence, ou évaluation.

Dans ces leçons, vous découvrirez comment utiliser ces étapes pour préparer, construire, tester, évaluer et prédire - tous les gestes d'un scientifique des données et plus encore, à mesure que vous progressez dans votre voyage pour devenir un ingénieur ML " full stack ".

---

## 🚀Défi

Dessinez un organigramme reflétant les étapes d'un praticien de la ML. Où vous voyez-vous en ce moment dans le processus ? Où pensez-vous rencontrer des difficultés ? Qu'est-ce qui vous semble facile ?

## [Quiz post-lecture] (https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/8/)

## Révision et autoformation

Recherchez en ligne des interviews de scientifiques des données qui parlent de leur travail quotidien. En voici [une] (https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Affectation

[Entretien avec un scientifique des données](assignment.md)
