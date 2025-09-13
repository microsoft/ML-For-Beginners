<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-04T00:23:10+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "fr"
}
-->
# Un monde plus réaliste

Dans notre situation, Peter pouvait se déplacer presque sans se fatiguer ni avoir faim. Dans un monde plus réaliste, il doit s'asseoir et se reposer de temps en temps, et aussi se nourrir. Rendons notre monde plus réaliste en appliquant les règles suivantes :

1. En se déplaçant d'un endroit à un autre, Peter perd de **l'énergie** et accumule de la **fatigue**.
2. Peter peut récupérer de l'énergie en mangeant des pommes.
3. Peter peut se débarrasser de la fatigue en se reposant sous un arbre ou sur l'herbe (c'est-à-dire en marchant vers une case contenant un arbre ou de l'herbe - champ vert).
4. Peter doit trouver et tuer le loup.
5. Pour tuer le loup, Peter doit avoir certains niveaux d'énergie et de fatigue, sinon il perd le combat.

## Instructions

Utilisez le [notebook.ipynb](notebook.ipynb) original comme point de départ pour votre solution.

Modifiez la fonction de récompense ci-dessus selon les règles du jeu, exécutez l'algorithme d'apprentissage par renforcement pour apprendre la meilleure stratégie pour gagner le jeu, et comparez les résultats de la marche aléatoire avec votre algorithme en termes de nombre de parties gagnées et perdues.

> **Note** : Dans votre nouveau monde, l'état est plus complexe et inclut, en plus de la position de l'humain, les niveaux de fatigue et d'énergie. Vous pouvez choisir de représenter l'état comme un tuple (Plateau, énergie, fatigue), ou définir une classe pour l'état (vous pouvez également envisager de la dériver de `Board`), ou même modifier la classe `Board` originale dans [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Dans votre solution, veuillez conserver le code responsable de la stratégie de marche aléatoire et comparer les résultats de votre algorithme avec la marche aléatoire à la fin.

> **Note** : Vous devrez peut-être ajuster les hyperparamètres pour que cela fonctionne, en particulier le nombre d'époques. Étant donné que le succès du jeu (combattre le loup) est un événement rare, vous pouvez vous attendre à un temps d'entraînement beaucoup plus long.

## Grille d'évaluation

| Critères | Exemplaire                                                                                                                                                                                             | Adéquat                                                                                                                                                                                 | À améliorer                                                                                                                                 |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Un notebook est présenté avec la définition des nouvelles règles du monde, l'algorithme Q-Learning et quelques explications textuelles. Q-Learning améliore significativement les résultats par rapport à la marche aléatoire. | Un notebook est présenté, Q-Learning est implémenté et améliore les résultats par rapport à la marche aléatoire, mais pas de manière significative ; ou le notebook est mal documenté et le code est mal structuré. | Une tentative de redéfinir les règles du monde est faite, mais l'algorithme Q-Learning ne fonctionne pas, ou la fonction de récompense n'est pas entièrement définie. |

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.