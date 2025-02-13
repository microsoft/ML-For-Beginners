## Vérification de la politique

Puisque la Q-Table répertorie l'« attractivité » de chaque action à chaque état, il est assez facile de l'utiliser pour définir la navigation efficace dans notre monde. Dans le cas le plus simple, nous pouvons sélectionner l'action correspondant à la valeur la plus élevée de la Q-Table : (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Si vous essayez le code ci-dessus plusieurs fois, vous remarquerez peut-être qu'il "se bloque" parfois, et que vous devez appuyer sur le bouton STOP dans le notebook pour l'interrompre. Cela se produit car il peut y avoir des situations où deux états "pointent" l'un vers l'autre en termes de valeur Q optimale, auquel cas les agents finissent par se déplacer indéfiniment entre ces états.

## 🚀Défi

> **Tâche 1 :** Modifiez le `walk` function to limit the maximum length of path by a certain number of steps (say, 100), and watch the code above return this value from time to time.

> **Task 2:** Modify the `walk` function so that it does not go back to the places where it has already been previously. This will prevent `walk` from looping, however, the agent can still end up being "trapped" in a location from which it is unable to escape.

## Navigation

A better navigation policy would be the one that we used during training, which combines exploitation and exploration. In this policy, we will select each action with a certain probability, proportional to the values in the Q-Table. This strategy may still result in the agent returning back to a position it has already explored, but, as you can see from the code below, it results in a very short average path to the desired location (remember that `print_statistics` pour exécuter la simulation 100 fois : (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Après avoir exécuté ce code, vous devriez obtenir une longueur de chemin moyenne beaucoup plus petite qu'auparavant, dans la plage de 3 à 6.

## Enquête sur le processus d'apprentissage

Comme nous l'avons mentionné, le processus d'apprentissage est un équilibre entre exploration et exploitation des connaissances acquises sur la structure de l'espace problème. Nous avons vu que les résultats de l'apprentissage (la capacité à aider un agent à trouver un chemin court vers l'objectif) se sont améliorés, mais il est également intéressant d'observer comment la longueur moyenne du chemin se comporte pendant le processus d'apprentissage :

Les apprentissages peuvent être résumés comme suit :

- **La longueur moyenne du chemin augmente**. Ce que nous voyons ici, c'est qu'au début, la longueur moyenne du chemin augmente. Cela est probablement dû au fait que lorsque nous ne savons rien sur l'environnement, nous avons tendance à nous retrouver coincés dans de mauvais états, comme l'eau ou le loup. À mesure que nous en apprenons davantage et commençons à utiliser ces connaissances, nous pouvons explorer l'environnement plus longtemps, mais nous ne savons toujours pas très bien où se trouvent les pommes.

- **La longueur du chemin diminue, à mesure que nous apprenons davantage**. Une fois que nous avons suffisamment appris, il devient plus facile pour l'agent d'atteindre l'objectif, et la longueur du chemin commence à diminuer. Cependant, nous restons ouverts à l'exploration, donc nous nous écartons souvent du meilleur chemin et explorons de nouvelles options, rendant le chemin plus long que l'optimal.

- **Augmentation brutale de la longueur**. Ce que nous observons également sur ce graphique, c'est qu'à un certain moment, la longueur a augmenté de manière brutale. Cela indique la nature stochastique du processus, et que nous pouvons à un moment "gâcher" les coefficients de la Q-Table en les écrasant avec de nouvelles valeurs. Cela devrait idéalement être minimisé en diminuant le taux d'apprentissage (par exemple, vers la fin de l'entraînement, nous n'ajustons les valeurs de la Q-Table que d'une petite valeur).

Dans l'ensemble, il est important de se rappeler que le succès et la qualité du processus d'apprentissage dépendent fortement des paramètres, tels que le taux d'apprentissage, la décote du taux d'apprentissage et le facteur d'actualisation. Ceux-ci sont souvent appelés **hyperparamètres**, pour les distinguer des **paramètres**, que nous optimisons pendant l'entraînement (par exemple, les coefficients de la Q-Table). Le processus de recherche des meilleures valeurs d'hyperparamètres est appelé **optimisation des hyperparamètres**, et cela mérite un sujet à part entière.

## [Quiz post-lecture](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Devoir 
[Un monde plus réaliste](assignment.md)

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue natale doit être considéré comme la source autorisée. Pour des informations critiques, une traduction professionnelle par un humain est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.