# Entraîner la Voiture de Montagne

[OpenAI Gym](http://gym.openai.com) a été conçu de manière à ce que tous les environnements offrent la même API - c'est-à-dire les mêmes méthodes `reset`, `step` et `render`, ainsi que les mêmes abstractions de **espace d'action** et **espace d'observation**. Ainsi, il devrait être possible d'adapter les mêmes algorithmes d'apprentissage par renforcement à différents environnements avec des modifications de code minimales.

## Un Environnement de Voiture de Montagne

L'[environnement de la Voiture de Montagne](https://gym.openai.com/envs/MountainCar-v0/) contient une voiture coincée dans une vallée :
Vous êtes formé sur des données jusqu'en octobre 2023.

L'objectif est de sortir de la vallée et de capturer le drapeau, en effectuant à chaque étape l'une des actions suivantes :

| Valeur | Signification |
|---|---|
| 0 | Accélérer vers la gauche |
| 1 | Ne pas accélérer |
| 2 | Accélérer vers la droite |

Le principal piège de ce problème est, cependant, que le moteur de la voiture n'est pas assez puissant pour gravir la montagne en un seul passage. Par conséquent, le seul moyen de réussir est de faire des allers-retours pour accumuler de l'élan.

L'espace d'observation se compose de seulement deux valeurs :

| Num | Observation  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Position de la voiture | -1.2| 0.6 |
|  1  | Vitesse de la voiture | -0.07 | 0.07 |

Le système de récompense pour la voiture de montagne est plutôt délicat :

 * Une récompense de 0 est accordée si l'agent atteint le drapeau (position = 0.5) au sommet de la montagne.
 * Une récompense de -1 est accordée si la position de l'agent est inférieure à 0.5.

L'épisode se termine si la position de la voiture est supérieure à 0.5, ou si la durée de l'épisode est supérieure à 200.

## Instructions

Adaptez notre algorithme d'apprentissage par renforcement pour résoudre le problème de la voiture de montagne. Commencez avec le code existant [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), substituez le nouvel environnement, changez les fonctions de discrétisation d'état, et essayez de faire en sorte que l'algorithme existant s'entraîne avec des modifications de code minimales. Optimisez le résultat en ajustant les hyperparamètres.

> **Note** : L'ajustement des hyperparamètres sera probablement nécessaire pour faire converger l'algorithme.

## Rubrique

| Critères | Exemplaire | Adéquat | Besoin d'Amélioration |
| -------- | --------- | -------- | ----------------- |
|          | L'algorithme Q-Learning est adapté avec succès de l'exemple CartPole, avec des modifications de code minimales, et est capable de résoudre le problème de capture du drapeau en moins de 200 étapes. | Un nouvel algorithme Q-Learning a été adopté depuis Internet, mais est bien documenté ; ou un algorithme existant a été adopté, mais n'atteint pas les résultats souhaités. | L'étudiant n'a pas réussi à adopter d'algorithme, mais a fait des progrès substantiels vers la solution (implémentation de la discrétisation d'état, structure de données Q-Table, etc.) |

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction professionnelle effectuée par un humain est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.