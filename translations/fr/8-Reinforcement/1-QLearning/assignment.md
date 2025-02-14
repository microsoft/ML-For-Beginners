# Un Monde Plus Réaliste

Dans notre situation, Peter pouvait se déplacer presque sans se fatiguer ni avoir faim. Dans un monde plus réaliste, il devait s'asseoir et se reposer de temps en temps, et aussi se nourrir. Rendons notre monde plus réaliste en mettant en œuvre les règles suivantes :

1. En se déplaçant d'un endroit à un autre, Peter perd de **l'énergie** et accumule de la **fatigue**.
2. Peter peut regagner de l'énergie en mangeant des pommes.
3. Peter peut se débarrasser de sa fatigue en se reposant sous un arbre ou sur l'herbe (c'est-à-dire en se déplaçant vers un emplacement avec un arbre ou de l'herbe - champ vert).
4. Peter doit trouver et tuer le loup.
5. Pour tuer le loup, Peter doit avoir certains niveaux d'énergie et de fatigue, sinon il perd la bataille.
## Instructions

Utilisez le [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) original comme point de départ pour votre solution.

Modifiez la fonction de récompense ci-dessus selon les règles du jeu, exécutez l'algorithme d'apprentissage par renforcement pour apprendre la meilleure stratégie pour gagner le jeu, et comparez les résultats de la marche aléatoire avec votre algorithme en termes de nombre de jeux gagnés et perdus.

> **Note** : Dans votre nouveau monde, l'état est plus complexe et, en plus de la position humaine, inclut également les niveaux de fatigue et d'énergie. Vous pouvez choisir de représenter l'état sous forme de tuple (Board, énergie, fatigue), ou de définir une classe pour l'état (vous pouvez également vouloir en dériver une de `Board`), ou même modifier la classe `Board` originale dans [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Dans votre solution, veuillez garder le code responsable de la stratégie de marche aléatoire et comparez les résultats de votre algorithme avec la marche aléatoire à la fin.

> **Note** : Vous devrez peut-être ajuster les hyperparamètres pour que cela fonctionne, en particulier le nombre d'époques. Étant donné que le succès du jeu (combattre le loup) est un événement rare, vous pouvez vous attendre à un temps d'entraînement beaucoup plus long.
## Rubrique

| Critères | Exemplaire                                                                                                                                                                                         | Adéquat                                                                                                                                                                              | Besoin d'Amélioration                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
|          | Un notebook est présenté avec la définition des nouvelles règles du monde, l'algorithme Q-Learning et quelques explications textuelles. Q-Learning est capable d'améliorer significativement les résultats par rapport à la marche aléatoire. | Le notebook est présenté, Q-Learning est implémenté et améliore les résultats par rapport à la marche aléatoire, mais pas de manière significative ; ou le notebook est mal documenté et le code n'est pas bien structuré | Une certaine tentative de redéfinir les règles du monde est faite, mais l'algorithme Q-Learning ne fonctionne pas, ou la fonction de récompense n'est pas entièrement définie. |

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue natale doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des erreurs d'interprétation résultant de l'utilisation de cette traduction.