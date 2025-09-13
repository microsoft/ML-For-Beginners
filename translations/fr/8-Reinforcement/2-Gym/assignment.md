<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-04T00:30:57+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "fr"
}
-->
# Entraîner une voiture de montagne

[OpenAI Gym](http://gym.openai.com) a été conçu de manière à ce que tous les environnements fournissent la même API - c'est-à-dire les mêmes méthodes `reset`, `step` et `render`, ainsi que les mêmes abstractions de **espace d'action** et **espace d'observation**. Ainsi, il devrait être possible d'adapter les mêmes algorithmes d'apprentissage par renforcement à différents environnements avec des modifications minimales du code.

## Un environnement de voiture de montagne

[L'environnement de voiture de montagne](https://gym.openai.com/envs/MountainCar-v0/) contient une voiture coincée dans une vallée :

L'objectif est de sortir de la vallée et de capturer le drapeau, en effectuant à chaque étape l'une des actions suivantes :

| Valeur | Signification |
|---|---|
| 0 | Accélérer vers la gauche |
| 1 | Ne pas accélérer |
| 2 | Accélérer vers la droite |

Le principal défi de ce problème est cependant que le moteur de la voiture n'est pas assez puissant pour gravir la montagne en un seul passage. Par conséquent, la seule façon de réussir est de faire des allers-retours pour accumuler de l'élan.

L'espace d'observation se compose de seulement deux valeurs :

| Num | Observation  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Position de la voiture | -1.2| 0.6 |
|  1  | Vitesse de la voiture | -0.07 | 0.07 |

Le système de récompense pour la voiture de montagne est assez complexe :

 * Une récompense de 0 est attribuée si l'agent atteint le drapeau (position = 0.5) au sommet de la montagne.
 * Une récompense de -1 est attribuée si la position de l'agent est inférieure à 0.5.

L'épisode se termine si la position de la voiture dépasse 0.5, ou si la durée de l'épisode dépasse 200 étapes.

## Instructions

Adaptez notre algorithme d'apprentissage par renforcement pour résoudre le problème de la voiture de montagne. Commencez avec le code existant dans [notebook.ipynb](notebook.ipynb), substituez le nouvel environnement, modifiez les fonctions de discrétisation de l'état, et essayez de faire en sorte que l'algorithme existant s'entraîne avec des modifications minimales du code. Optimisez le résultat en ajustant les hyperparamètres.

> **Note** : L'ajustement des hyperparamètres sera probablement nécessaire pour que l'algorithme converge.

## Critères d'évaluation

| Critères | Exemplaire | Adéquat | À améliorer |
| -------- | --------- | -------- | ----------------- |
|          | L'algorithme Q-Learning est adapté avec succès à partir de l'exemple CartPole, avec des modifications minimales du code, et parvient à résoudre le problème de capture du drapeau en moins de 200 étapes. | Un nouvel algorithme Q-Learning a été adopté à partir d'Internet, mais est bien documenté ; ou l'algorithme existant a été adopté, mais n'atteint pas les résultats souhaités. | L'étudiant n'a pas réussi à adopter un algorithme, mais a fait des progrès substantiels vers la solution (implémentation de la discrétisation de l'état, structure de données Q-Table, etc.). |

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction humaine professionnelle. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.