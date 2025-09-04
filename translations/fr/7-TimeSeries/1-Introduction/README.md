<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3150d40f36a77857316ecaed5f31e856",
  "translation_date": "2025-09-03T22:47:22+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "fr"
}
-->
# Introduction à la prévision des séries temporelles

![Résumé des séries temporelles dans un sketchnote](../../../../translated_images/ml-timeseries.fb98d25f1013fc0c59090030080b5d1911ff336427bec31dbaf1ad08193812e9.fr.png)

> Sketchnote par [Tomomi Imura](https://www.twitter.com/girlie_mac)

Dans cette leçon et la suivante, vous allez découvrir la prévision des séries temporelles, une partie intéressante et précieuse du répertoire d'un scientifique en apprentissage automatique, qui est un peu moins connue que d'autres sujets. La prévision des séries temporelles est une sorte de "boule de cristal" : en se basant sur les performances passées d'une variable comme le prix, vous pouvez prédire sa valeur potentielle future.

[![Introduction à la prévision des séries temporelles](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introduction à la prévision des séries temporelles")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo sur la prévision des séries temporelles

## [Quiz avant la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/41/)

C'est un domaine utile et intéressant avec une réelle valeur pour les entreprises, étant donné son application directe aux problèmes de tarification, d'inventaire et de chaîne d'approvisionnement. Bien que les techniques d'apprentissage profond aient commencé à être utilisées pour obtenir davantage d'informations afin de mieux prédire les performances futures, la prévision des séries temporelles reste un domaine largement influencé par les techniques classiques d'apprentissage automatique.

> Le programme utile sur les séries temporelles de Penn State est disponible [ici](https://online.stat.psu.edu/stat510/lesson/1)

## Introduction

Supposons que vous gérez un ensemble de parcmètres intelligents qui fournissent des données sur leur fréquence d'utilisation et leur durée d'utilisation au fil du temps.

> Et si vous pouviez prédire, en fonction des performances passées du parcmètre, sa valeur future selon les lois de l'offre et de la demande ?

Prédire avec précision le moment où agir pour atteindre votre objectif est un défi qui pourrait être relevé grâce à la prévision des séries temporelles. Cela ne rendrait pas les gens heureux d'être facturés davantage pendant les périodes de forte affluence lorsqu'ils cherchent une place de parking, mais ce serait un moyen sûr de générer des revenus pour nettoyer les rues !

Explorons certains types d'algorithmes de séries temporelles et commençons un notebook pour nettoyer et préparer des données. Les données que vous allez analyser proviennent de la compétition de prévision GEFCom2014. Elles consistent en 3 années de valeurs horaires de charge électrique et de température entre 2012 et 2014. Étant donné les modèles historiques de charge électrique et de température, vous pouvez prédire les valeurs futures de charge électrique.

Dans cet exemple, vous apprendrez à prévoir une étape temporelle à l'avance, en utilisant uniquement les données historiques de charge. Avant de commencer, cependant, il est utile de comprendre ce qui se passe en coulisses.

## Quelques définitions

Lorsque vous rencontrez le terme "séries temporelles", vous devez comprendre son utilisation dans plusieurs contextes différents.

🎓 **Séries temporelles**

En mathématiques, "une série temporelle est une série de points de données indexés (ou listés ou tracés) dans l'ordre temporel. Le plus souvent, une série temporelle est une séquence prise à des intervalles successifs équidistants dans le temps." Un exemple de série temporelle est la valeur de clôture quotidienne du [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). L'utilisation de graphiques de séries temporelles et de modélisation statistique est fréquemment rencontrée dans le traitement du signal, la prévision météorologique, la prédiction des tremblements de terre et d'autres domaines où des événements se produisent et des points de données peuvent être tracés dans le temps.

🎓 **Analyse des séries temporelles**

L'analyse des séries temporelles est l'analyse des données de séries temporelles mentionnées ci-dessus. Les données de séries temporelles peuvent prendre des formes distinctes, y compris les "séries temporelles interrompues" qui détectent des modèles dans l'évolution d'une série temporelle avant et après un événement perturbateur. Le type d'analyse nécessaire pour les séries temporelles dépend de la nature des données. Les données de séries temporelles elles-mêmes peuvent prendre la forme de séries de nombres ou de caractères.

L'analyse à effectuer utilise une variété de méthodes, y compris le domaine fréquentiel et le domaine temporel, linéaire et non linéaire, et plus encore. [En savoir plus](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) sur les nombreuses façons d'analyser ce type de données.

🎓 **Prévision des séries temporelles**

La prévision des séries temporelles est l'utilisation d'un modèle pour prédire des valeurs futures en fonction des modèles affichés par les données précédemment collectées telles qu'elles se sont produites dans le passé. Bien qu'il soit possible d'utiliser des modèles de régression pour explorer les données de séries temporelles, avec des indices temporels comme variables x sur un graphique, ces données sont mieux analysées à l'aide de types de modèles spécifiques.

Les données de séries temporelles sont une liste d'observations ordonnées, contrairement aux données qui peuvent être analysées par régression linéaire. Le modèle le plus courant est ARIMA, un acronyme qui signifie "Autoregressive Integrated Moving Average".

[Les modèles ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relient la valeur actuelle d'une série aux valeurs passées et aux erreurs de prédiction passées." Ils sont les plus appropriés pour analyser les données du domaine temporel, où les données sont ordonnées dans le temps.

> Il existe plusieurs types de modèles ARIMA, que vous pouvez découvrir [ici](https://people.duke.edu/~rnau/411arim.htm) et que vous aborderez dans la prochaine leçon.

Dans la prochaine leçon, vous construirez un modèle ARIMA en utilisant [les séries temporelles univariées](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), qui se concentrent sur une variable qui change de valeur au fil du temps. Un exemple de ce type de données est [ce jeu de données](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) qui enregistre la concentration mensuelle de CO2 à l'observatoire de Mauna Loa :

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Identifiez la variable qui change au fil du temps dans ce jeu de données.

## Caractéristiques des données de séries temporelles à prendre en compte

Lorsque vous examinez des données de séries temporelles, vous pourriez remarquer qu'elles présentent [certaines caractéristiques](https://online.stat.psu.edu/stat510/lesson/1/1.1) que vous devez prendre en compte et atténuer pour mieux comprendre leurs modèles. Si vous considérez les données de séries temporelles comme fournissant potentiellement un "signal" que vous souhaitez analyser, ces caractéristiques peuvent être considérées comme du "bruit". Vous devrez souvent réduire ce "bruit" en compensant certaines de ces caractéristiques à l'aide de techniques statistiques.

Voici quelques concepts que vous devriez connaître pour travailler avec les séries temporelles :

🎓 **Tendances**

Les tendances sont définies comme des augmentations et des diminutions mesurables au fil du temps. [En savoir plus](https://machinelearningmastery.com/time-series-trends-in-python). Dans le contexte des séries temporelles, il s'agit de savoir comment utiliser et, si nécessaire, supprimer les tendances de vos séries temporelles.

🎓 **[Saisonnalité](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

La saisonnalité est définie comme des fluctuations périodiques, telles que les périodes de forte affluence pendant les vacances qui pourraient affecter les ventes, par exemple. [Découvrez](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) comment différents types de graphiques affichent la saisonnalité dans les données.

🎓 **Valeurs aberrantes**

Les valeurs aberrantes sont éloignées de la variance standard des données.

🎓 **Cycle à long terme**

Indépendamment de la saisonnalité, les données peuvent afficher un cycle à long terme, comme une récession économique qui dure plus d'un an.

🎓 **Variance constante**

Au fil du temps, certaines données affichent des fluctuations constantes, comme la consommation d'énergie par jour et par nuit.

🎓 **Changements brusques**

Les données peuvent afficher un changement brusque qui pourrait nécessiter une analyse approfondie. La fermeture soudaine des entreprises en raison de la COVID, par exemple, a provoqué des changements dans les données.

✅ Voici un [exemple de graphique de séries temporelles](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) montrant les dépenses quotidiennes en monnaie virtuelle dans un jeu sur plusieurs années. Pouvez-vous identifier certaines des caractéristiques mentionnées ci-dessus dans ces données ?

![Dépenses en monnaie virtuelle dans un jeu](../../../../translated_images/currency.e7429812bfc8c6087b2d4c410faaa4aaa11b2fcaabf6f09549b8249c9fbdb641.fr.png)

## Exercice - démarrer avec les données de consommation d'énergie

Commençons par créer un modèle de séries temporelles pour prédire la consommation d'énergie future en fonction de la consommation passée.

> Les données de cet exemple proviennent de la compétition de prévision GEFCom2014. Elles consistent en 3 années de valeurs horaires de charge électrique et de température entre 2012 et 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli et Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, juillet-septembre, 2016.

1. Dans le dossier `working` de cette leçon, ouvrez le fichier _notebook.ipynb_. Commencez par ajouter des bibliothèques qui vous aideront à charger et visualiser les données :

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Notez que vous utilisez les fichiers du dossier `common` inclus, qui configurent votre environnement et gèrent le téléchargement des données.

2. Ensuite, examinez les données sous forme de dataframe en appelant `load_data()` et `head()` :

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Vous pouvez voir qu'il y a deux colonnes représentant la date et la charge :

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Maintenant, tracez les données en appelant `plot()` :

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Graphique de consommation d'énergie](../../../../translated_images/energy-plot.5fdac3f397a910bc6070602e9e45bea8860d4c239354813fa8fc3c9d556f5bad.fr.png)

4. Ensuite, tracez la première semaine de juillet 2014, en la fournissant comme entrée à `energy` dans le format `[date de début]:[date de fin]` :

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Juillet](../../../../translated_images/july-2014.9e1f7c318ec6d5b30b0d7e1e20be3643501f64a53f3d426d7c7d7b62addb335e.fr.png)

    Un graphique magnifique ! Examinez ces graphiques et voyez si vous pouvez déterminer certaines des caractéristiques mentionnées ci-dessus. Que pouvons-nous déduire en visualisant les données ?

Dans la prochaine leçon, vous créerez un modèle ARIMA pour effectuer des prévisions.

---

## 🚀Défi

Faites une liste de toutes les industries et domaines de recherche que vous pouvez imaginer qui bénéficieraient de la prévision des séries temporelles. Pouvez-vous penser à une application de ces techniques dans les arts ? En économétrie ? En écologie ? En commerce de détail ? En industrie ? En finance ? Où encore ?

## [Quiz après la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/42/)

## Révision et auto-apprentissage

Bien que nous ne les couvrions pas ici, les réseaux neuronaux sont parfois utilisés pour améliorer les méthodes classiques de prévision des séries temporelles. Lisez-en plus [dans cet article](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Devoir

[Visualisez davantage de séries temporelles](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.