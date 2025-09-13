<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-04T00:13:30+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "fr"
}
-->
# Introduction à l'apprentissage par renforcement

L'apprentissage par renforcement, ou RL, est considéré comme l'un des paradigmes fondamentaux de l'apprentissage automatique, aux côtés de l'apprentissage supervisé et non supervisé. Le RL concerne les décisions : prendre les bonnes décisions ou, à défaut, apprendre de celles-ci.

Imaginez que vous avez un environnement simulé, comme le marché boursier. Que se passe-t-il si vous imposez une réglementation donnée ? A-t-elle un effet positif ou négatif ? Si quelque chose de négatif se produit, vous devez tirer parti de ce _renforcement négatif_, en apprendre et changer de cap. Si le résultat est positif, vous devez vous appuyer sur ce _renforcement positif_.

![peter et le loup](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.fr.png)

> Peter et ses amis doivent échapper au loup affamé ! Image par [Jen Looper](https://twitter.com/jenlooper)

## Sujet régional : Pierre et le Loup (Russie)

[Pierre et le Loup](https://fr.wikipedia.org/wiki/Pierre_et_le_Loup) est un conte musical écrit par le compositeur russe [Sergei Prokofiev](https://fr.wikipedia.org/wiki/Serge_Prokofiev). C'est l'histoire du jeune pionnier Pierre, qui sort courageusement de sa maison pour aller dans la clairière de la forêt et chasser le loup. Dans cette section, nous allons entraîner des algorithmes d'apprentissage automatique qui aideront Pierre à :

- **Explorer** les environs et construire une carte de navigation optimale.
- **Apprendre** à utiliser un skateboard et à garder l'équilibre dessus, afin de se déplacer plus rapidement.

[![Pierre et le Loup](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Cliquez sur l'image ci-dessus pour écouter Pierre et le Loup de Prokofiev

## Apprentissage par renforcement

Dans les sections précédentes, vous avez vu deux exemples de problèmes d'apprentissage automatique :

- **Supervisé**, où nous avons des ensembles de données qui suggèrent des solutions possibles au problème que nous voulons résoudre. [La classification](../4-Classification/README.md) et [la régression](../2-Regression/README.md) sont des tâches d'apprentissage supervisé.
- **Non supervisé**, où nous n'avons pas de données d'entraînement étiquetées. L'exemple principal d'apprentissage non supervisé est [le clustering](../5-Clustering/README.md).

Dans cette section, nous allons vous présenter un nouveau type de problème d'apprentissage qui ne nécessite pas de données d'entraînement étiquetées. Il existe plusieurs types de tels problèmes :

- **[Apprentissage semi-supervisé](https://fr.wikipedia.org/wiki/Apprentissage_semi-supervis%C3%A9)**, où nous avons beaucoup de données non étiquetées qui peuvent être utilisées pour pré-entraîner le modèle.
- **[Apprentissage par renforcement](https://fr.wikipedia.org/wiki/Apprentissage_par_renforcement)**, dans lequel un agent apprend à se comporter en réalisant des expériences dans un environnement simulé.

### Exemple - jeu vidéo

Supposons que vous voulez apprendre à un ordinateur à jouer à un jeu, comme les échecs ou [Super Mario](https://fr.wikipedia.org/wiki/Super_Mario). Pour que l'ordinateur joue à un jeu, nous devons lui apprendre à prédire quel mouvement effectuer dans chacun des états du jeu. Bien que cela puisse sembler être un problème de classification, ce n'est pas le cas - car nous n'avons pas d'ensemble de données avec des états et des actions correspondantes. Bien que nous puissions avoir des données comme des parties d'échecs existantes ou des enregistrements de joueurs jouant à Super Mario, il est probable que ces données ne couvrent pas suffisamment un grand nombre d'états possibles.

Au lieu de chercher des données de jeu existantes, **l'apprentissage par renforcement** (RL) repose sur l'idée de *faire jouer l'ordinateur* plusieurs fois et d'observer le résultat. Ainsi, pour appliquer l'apprentissage par renforcement, nous avons besoin de deux éléments :

- **Un environnement** et **un simulateur** qui nous permettent de jouer au jeu plusieurs fois. Ce simulateur définirait toutes les règles du jeu ainsi que les états et actions possibles.

- **Une fonction de récompense**, qui nous indiquerait à quel point nous avons bien joué à chaque mouvement ou partie.

La principale différence entre les autres types d'apprentissage automatique et le RL est que dans le RL, nous ne savons généralement pas si nous gagnons ou perdons avant de terminer la partie. Ainsi, nous ne pouvons pas dire si un certain mouvement seul est bon ou non - nous ne recevons une récompense qu'à la fin de la partie. Et notre objectif est de concevoir des algorithmes qui nous permettront d'entraîner un modèle dans des conditions incertaines. Nous allons apprendre un algorithme de RL appelé **Q-learning**.

## Leçons

1. [Introduction à l'apprentissage par renforcement et au Q-Learning](1-QLearning/README.md)
2. [Utilisation d'un environnement de simulation gym](2-Gym/README.md)

## Crédits

"Introduction à l'apprentissage par renforcement" a été écrit avec ♥️ par [Dmitry Soshnikov](http://soshnikov.com)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction humaine professionnelle. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.