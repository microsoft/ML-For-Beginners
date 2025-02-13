# Introduction à l'apprentissage par renforcement

L'apprentissage par renforcement, RL, est considéré comme l'un des paradigmes fondamentaux de l'apprentissage automatique, aux côtés de l'apprentissage supervisé et de l'apprentissage non supervisé. Le RL est entièrement axé sur les décisions : prendre les bonnes décisions ou, du moins, apprendre de celles-ci.

Imaginez que vous avez un environnement simulé comme le marché boursier. Que se passe-t-il si vous imposez une réglementation donnée ? A-t-elle un effet positif ou négatif ? Si quelque chose de négatif se produit, vous devez prendre ce _renforcement négatif_, en tirer des leçons et changer de cap. Si c'est un résultat positif, vous devez capitaliser sur ce _renforcement positif_.

![peter and the wolf](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.fr.png)

> Peter et ses amis doivent échapper au loup affamé ! Image par [Jen Looper](https://twitter.com/jenlooper)

## Sujet régional : Pierre et le Loup (Russie)

[Pierre et le Loup](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) est un conte musical écrit par un compositeur russe [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). C'est l'histoire du jeune pionnier Pierre, qui s'aventure courageusement hors de sa maison vers la clairière pour chasser le loup. Dans cette section, nous allons entraîner des algorithmes d'apprentissage automatique qui aideront Pierre :

- **Explorer** les environs et construire une carte de navigation optimale
- **Apprendre** à utiliser un skateboard et à s'y équilibrer, afin de se déplacer plus rapidement.

[![Pierre et le Loup](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Cliquez sur l'image ci-dessus pour écouter Pierre et le Loup de Prokofiev

## Apprentissage par renforcement

Dans les sections précédentes, vous avez vu deux exemples de problèmes d'apprentissage automatique :

- **Supervisé**, où nous avons des ensembles de données qui suggèrent des solutions types au problème que nous voulons résoudre. [Classification](../4-Classification/README.md) et [régression](../2-Regression/README.md) sont des tâches d'apprentissage supervisé.
- **Non supervisé**, où nous n'avons pas de données d'entraînement étiquetées. L'exemple principal de l'apprentissage non supervisé est [Clustering](../5-Clustering/README.md).

Dans cette section, nous allons vous introduire à un nouveau type de problème d'apprentissage qui ne nécessite pas de données d'entraînement étiquetées. Il existe plusieurs types de tels problèmes :

- **[Apprentissage semi-supervisé](https://wikipedia.org/wiki/Semi-supervised_learning)**, où nous avons beaucoup de données non étiquetées qui peuvent être utilisées pour préformer le modèle.
- **[Apprentissage par renforcement](https://wikipedia.org/wiki/Reinforcement_learning)**, dans lequel un agent apprend comment se comporter en réalisant des expériences dans un environnement simulé.

### Exemple - jeu vidéo

Supposons que vous souhaitiez apprendre à un ordinateur à jouer à un jeu, comme les échecs ou [Super Mario](https://wikipedia.org/wiki/Super_Mario). Pour que l'ordinateur puisse jouer à un jeu, nous devons lui faire prédire quel mouvement effectuer dans chacun des états du jeu. Bien que cela puisse sembler être un problème de classification, ce n'est pas le cas - car nous n'avons pas d'ensemble de données avec des états et des actions correspondantes. Bien que nous puissions avoir des données comme des parties d'échecs existantes ou des enregistrements de joueurs jouant à Super Mario, il est probable que ces données ne couvrent pas suffisamment un nombre assez large d'états possibles.

Au lieu de chercher des données de jeu existantes, **l'apprentissage par renforcement** (RL) repose sur l'idée de *faire jouer l'ordinateur* de nombreuses fois et d'observer le résultat. Ainsi, pour appliquer l'apprentissage par renforcement, nous avons besoin de deux choses :

- **Un environnement** et **un simulateur** qui nous permettent de jouer à un jeu plusieurs fois. Ce simulateur définirait toutes les règles du jeu ainsi que les états et actions possibles.

- **Une fonction de récompense**, qui nous indiquerait à quel point nous avons bien joué à chaque mouvement ou partie.

La principale différence entre les autres types d'apprentissage automatique et le RL est qu'en RL, nous ne savons généralement pas si nous gagnons ou perdons jusqu'à ce que nous terminions le jeu. Ainsi, nous ne pouvons pas dire si un certain mouvement à lui seul est bon ou non - nous ne recevons une récompense qu'à la fin du jeu. Et notre objectif est de concevoir des algorithmes qui nous permettront de former un modèle dans des conditions d'incertitude. Nous allons apprendre un algorithme de RL appelé **Q-learning**.

## Leçons

1. [Introduction à l'apprentissage par renforcement et Q-Learning](1-QLearning/README.md)
2. [Utiliser un environnement de simulation gym](2-Gym/README.md)

## Crédits

"Introduction à l'apprentissage par renforcement" a été écrit avec ♥️ par [Dmitry Soshnikov](http://soshnikov.com)

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des erreurs d'interprétation résultant de l'utilisation de cette traduction.