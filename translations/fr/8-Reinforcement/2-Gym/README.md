# Patinage CartPole

Le problème que nous avons résolu dans la leçon précédente peut sembler être un problème trivial, pas vraiment applicable à des scénarios de la vie réelle. Ce n'est pas le cas, car de nombreux problèmes du monde réel partagent également ce scénario - y compris le jeu d'échecs ou de go. Ils sont similaires, car nous avons également un plateau avec des règles données et un **état discret**.

## [Quiz pré-lecture](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/47/)

## Introduction

Dans cette leçon, nous appliquerons les mêmes principes de Q-Learning à un problème avec un **état continu**, c'est-à-dire un état qui est défini par un ou plusieurs nombres réels. Nous allons traiter le problème suivant :

> **Problème** : Si Peter veut échapper au loup, il doit être capable de se déplacer plus vite. Nous verrons comment Peter peut apprendre à patiner, en particulier, à garder son équilibre, en utilisant le Q-Learning.

![La grande évasion !](../../../../translated_images/escape.18862db9930337e3fce23a9b6a76a06445f229dadea2268e12a6f0a1fde12115.fr.png)

> Peter et ses amis font preuve de créativité pour échapper au loup ! Image par [Jen Looper](https://twitter.com/jenlooper)

Nous utiliserons une version simplifiée de l'équilibre connue sous le nom de problème **CartPole**. Dans le monde de cartpole, nous avons un curseur horizontal qui peut se déplacer à gauche ou à droite, et l'objectif est de maintenir un poteau vertical au sommet du curseur.

## Prérequis

Dans cette leçon, nous utiliserons une bibliothèque appelée **OpenAI Gym** pour simuler différents **environnements**. Vous pouvez exécuter le code de cette leçon localement (par exemple, depuis Visual Studio Code), auquel cas la simulation s'ouvrira dans une nouvelle fenêtre. Lorsque vous exécutez le code en ligne, vous devrez peut-être apporter quelques modifications au code, comme décrit [ici](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Dans la leçon précédente, les règles du jeu et l'état étaient donnés par la classe `Board` que nous avons définie nous-mêmes. Ici, nous utiliserons un **environnement de simulation** spécial, qui simulera la physique derrière l'équilibre du poteau. L'un des environnements de simulation les plus populaires pour entraîner des algorithmes d'apprentissage par renforcement est appelé [Gym](https://gym.openai.com/), qui est maintenu par [OpenAI](https://openai.com/). En utilisant ce gym, nous pouvons créer différents **environnements**, allant de la simulation de cartpole aux jeux Atari.

> **Note** : Vous pouvez voir d'autres environnements disponibles dans OpenAI Gym [ici](https://gym.openai.com/envs/#classic_control).

Tout d'abord, installons le gym et importons les bibliothèques nécessaires (bloc de code 1) :

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Exercice - initialiser un environnement cartpole

Pour travailler avec un problème d'équilibre de cartpole, nous devons initialiser l'environnement correspondant. Chaque environnement est associé à un :

- **Espace d'observation** qui définit la structure des informations que nous recevons de l'environnement. Pour le problème cartpole, nous recevons la position du poteau, la vitesse et d'autres valeurs.

- **Espace d'action** qui définit les actions possibles. Dans notre cas, l'espace d'action est discret et se compose de deux actions - **gauche** et **droite**. (bloc de code 2)

1. Pour initialiser, tapez le code suivant :

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Pour voir comment l'environnement fonctionne, exécutons une courte simulation pendant 100 étapes. À chaque étape, nous fournissons l'une des actions à effectuer - dans cette simulation, nous sélectionnons simplement une action au hasard dans `action_space`.

1. Exécutez le code ci-dessous et voyez ce que cela donne.

    ✅ Rappelez-vous qu'il est préférable d'exécuter ce code sur une installation Python locale ! (bloc de code 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Vous devriez voir quelque chose de similaire à cette image :

    ![cartpole non équilibré](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Pendant la simulation, nous devons obtenir des observations afin de décider comment agir. En fait, la fonction d'étape renvoie les observations actuelles, une fonction de récompense et le drapeau done qui indique s'il est judicieux de continuer la simulation ou non : (bloc de code 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Vous finirez par voir quelque chose comme ceci dans la sortie du notebook :

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    Le vecteur d'observation qui est renvoyé à chaque étape de la simulation contient les valeurs suivantes :
    - Position du chariot
    - Vitesse du chariot
    - Angle du poteau
    - Taux de rotation du poteau

1. Obtenez la valeur minimale et maximale de ces nombres : (bloc de code 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Vous remarquerez également que la valeur de la récompense à chaque étape de simulation est toujours 1. Cela est dû au fait que notre objectif est de survivre le plus longtemps possible, c'est-à-dire de maintenir le poteau dans une position raisonnablement verticale pendant la plus longue période de temps.

    ✅ En fait, la simulation CartPole est considérée comme résolue si nous parvenons à obtenir une récompense moyenne de 195 sur 100 essais consécutifs.

## Discrétisation de l'état

Dans le Q-Learning, nous devons construire une Q-Table qui définit quoi faire à chaque état. Pour pouvoir le faire, nous avons besoin que l'état soit **discret**, plus précisément, il doit contenir un nombre fini de valeurs discrètes. Ainsi, nous devons d'une manière ou d'une autre **discrétiser** nos observations, en les mappant à un ensemble fini d'états.

Il existe plusieurs façons de procéder :

- **Diviser en bacs**. Si nous connaissons l'intervalle d'une certaine valeur, nous pouvons diviser cet intervalle en un certain nombre de **bacs**, puis remplacer la valeur par le numéro du bac auquel elle appartient. Cela peut être fait en utilisant la méthode numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Dans ce cas, nous connaîtrons précisément la taille de l'état, car elle dépendra du nombre de bacs que nous sélectionnons pour la numérisation.

✅ Nous pouvons utiliser l'interpolation linéaire pour amener les valeurs à un certain intervalle fini (disons, de -20 à 20), puis convertir les nombres en entiers en les arrondissant. Cela nous donne un peu moins de contrôle sur la taille de l'état, surtout si nous ne connaissons pas les plages exactes des valeurs d'entrée. Par exemple, dans notre cas, 2 des 4 valeurs n'ont pas de limites supérieures/inférieures, ce qui peut entraîner un nombre infini d'états.

Dans notre exemple, nous allons opter pour la deuxième approche. Comme vous le remarquerez plus tard, malgré l'absence de limites supérieures/inférieures, ces valeurs prennent rarement des valeurs en dehors de certains intervalles finis, donc ces états avec des valeurs extrêmes seront très rares.

1. Voici la fonction qui prendra l'observation de notre modèle et produira un tuple de 4 valeurs entières : (bloc de code 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Explorons également une autre méthode de discrétisation utilisant des bacs : (bloc de code 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Exécutons maintenant une courte simulation et observons ces valeurs d'environnement discrètes. N'hésitez pas à essayer à la fois `discretize` and `discretize_bins` et voir s'il y a une différence.

    ✅ discretize_bins renvoie le numéro du bac, qui est basé sur 0. Ainsi, pour des valeurs de variable d'entrée autour de 0, cela renvoie le numéro du milieu de l'intervalle (10). Dans discretize, nous ne nous sommes pas souciés de l'intervalle des valeurs de sortie, leur permettant d'être négatives, donc les valeurs d'état ne sont pas décalées, et 0 correspond à 0. (bloc de code 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Décommentez la ligne commençant par env.render si vous voulez voir comment l'environnement s'exécute. Sinon, vous pouvez l'exécuter en arrière-plan, ce qui est plus rapide. Nous utiliserons cette exécution "invisible" lors de notre processus de Q-Learning.

## La structure de la Q-Table

Dans notre leçon précédente, l'état était une simple paire de nombres de 0 à 8, et il était donc pratique de représenter la Q-Table par un tenseur numpy de forme 8x8x2. Si nous utilisons la discrétisation par bacs, la taille de notre vecteur d'état est également connue, donc nous pouvons utiliser la même approche et représenter l'état par un tableau de forme 20x20x10x10x2 (ici 2 est la dimension de l'espace d'action, et les premières dimensions correspondent au nombre de bacs que nous avons sélectionnés pour chacun des paramètres de l'espace d'observation).

Cependant, parfois, les dimensions précises de l'espace d'observation ne sont pas connues. Dans le cas de la fonction `discretize`, nous ne pouvons jamais être sûrs que notre état reste dans certaines limites, car certaines des valeurs d'origine ne sont pas bornées. Ainsi, nous utiliserons une approche légèrement différente et représenterons la Q-Table par un dictionnaire.

1. Utilisez la paire *(état, action)* comme clé du dictionnaire, et la valeur correspondra à la valeur d'entrée de la Q-Table. (bloc de code 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Ici, nous définissons également une fonction `qvalues()`, qui renvoie une liste des valeurs de la Q-Table pour un état donné qui correspond à toutes les actions possibles. Si l'entrée n'est pas présente dans la Q-Table, nous renverrons 0 par défaut.

## Commençons le Q-Learning

Maintenant, nous sommes prêts à apprendre à Peter à équilibrer !

1. Tout d'abord, définissons quelques hyperparamètres : (bloc de code 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Ici, `alpha` is the **learning rate** that defines to which extent we should adjust the current values of Q-Table at each step. In the previous lesson we started with 1, and then decreased `alpha` to lower values during training. In this example we will keep it constant just for simplicity, and you can experiment with adjusting `alpha` values later.

    `gamma` is the **discount factor** that shows to which extent we should prioritize future reward over current reward.

    `epsilon` is the **exploration/exploitation factor** that determines whether we should prefer exploration to exploitation or vice versa. In our algorithm, we will in `epsilon` percent of the cases select the next action according to Q-Table values, and in the remaining number of cases we will execute a random action. This will allow us to explore areas of the search space that we have never seen before. 

    ✅ In terms of balancing - choosing random action (exploration) would act as a random punch in the wrong direction, and the pole would have to learn how to recover the balance from those "mistakes"

### Improve the algorithm

We can also make two improvements to our algorithm from the previous lesson:

- **Calculate average cumulative reward**, over a number of simulations. We will print the progress each 5000 iterations, and we will average out our cumulative reward over that period of time. It means that if we get more than 195 point - we can consider the problem solved, with even higher quality than required.
  
- **Calculate maximum average cumulative result**, `Qmax`, and we will store the Q-Table corresponding to that result. When you run the training you will notice that sometimes the average cumulative result starts to drop, and we want to keep the values of Q-Table that correspond to the best model observed during training.

1. Collect all cumulative rewards at each simulation at `rewards` vecteur pour un traçage ultérieur. (bloc de code 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Ce que vous pouvez remarquer à partir de ces résultats :

- **Proche de notre objectif**. Nous sommes très proches d'atteindre l'objectif d'obtenir 195 récompenses cumulées sur 100+ exécutions consécutives de la simulation, ou nous l'avons peut-être déjà atteint ! Même si nous obtenons des chiffres plus petits, nous ne le savons toujours pas, car nous faisons la moyenne sur 5000 exécutions, et seulement 100 exécutions sont requises dans les critères formels.

- **La récompense commence à diminuer**. Parfois, la récompense commence à diminuer, ce qui signifie que nous pouvons "détruire" les valeurs déjà apprises dans la Q-Table avec celles qui aggravent la situation.

Cette observation est plus clairement visible si nous traçons les progrès de l'entraînement.

## Traçage des progrès de l'entraînement

Pendant l'entraînement, nous avons collecté la valeur de la récompense cumulée à chacune des itérations dans le vecteur `rewards`. Voici à quoi cela ressemble lorsque nous le traçons par rapport au numéro d'itération :

```python
plt.plot(rewards)
```

![progrès brut](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.fr.png)

À partir de ce graphique, il n'est pas possible de dire quoi que ce soit, car en raison de la nature du processus d'entraînement stochastique, la durée des sessions d'entraînement varie considérablement. Pour donner plus de sens à ce graphique, nous pouvons calculer la **moyenne mobile** sur une série d'expériences, disons 100. Cela peut être fait facilement en utilisant `np.convolve` : (bloc de code 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progrès de l'entraînement](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.fr.png)

## Variation des hyperparamètres

Pour rendre l'apprentissage plus stable, il est judicieux d'ajuster certains de nos hyperparamètres pendant l'entraînement. En particulier :

- **Pour le taux d'apprentissage**, `alpha`, we may start with values close to 1, and then keep decreasing the parameter. With time, we will be getting good probability values in the Q-Table, and thus we should be adjusting them slightly, and not overwriting completely with new values.

- **Increase epsilon**. We may want to increase the `epsilon` slowly, in order to explore less and exploit more. It probably makes sense to start with lower value of `epsilon`, et passez à presque 1.

> **Tâche 1** : Jouez avec les valeurs des hyperparamètres et voyez si vous pouvez atteindre une récompense cumulative plus élevée. Obtenez-vous plus de 195 ?

> **Tâche 2** : Pour résoudre formellement le problème, vous devez obtenir une récompense moyenne de 195 sur 100 exécutions consécutives. Mesurez cela pendant l'entraînement et assurez-vous que vous avez formellement résolu le problème !

## Voir le résultat en action

Il serait intéressant de voir comment le modèle entraîné se comporte. Exécutons la simulation et suivons la même stratégie de sélection d'actions que pendant l'entraînement, en échantillonnant selon la distribution de probabilité dans la Q-Table : (bloc de code 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Vous devriez voir quelque chose comme ceci :

![un cartpole équilibré](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Défi

> **Tâche 3** : Ici, nous utilisions la copie finale de la Q-Table, qui peut ne pas être la meilleure. N'oubliez pas que nous avons stocké la Q-Table la plus performante dans `Qbest` variable! Try the same example with the best-performing Q-Table by copying `Qbest` over to `Q` and see if you notice the difference.

> **Task 4**: Here we were not selecting the best action on each step, but rather sampling with corresponding probability distribution. Would it make more sense to always select the best action, with the highest Q-Table value? This can be done by using `np.argmax` fonction pour découvrir le numéro d'action correspondant à la valeur la plus élevée de la Q-Table. Implémentez cette stratégie et voyez si cela améliore l'équilibre.

## [Quiz post-lecture](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/)

## Devoir
[Entraînez une voiture de montagne](assignment.md)

## Conclusion

Nous avons maintenant appris comment entraîner des agents pour obtenir de bons résultats simplement en leur fournissant une fonction de récompense qui définit l'état souhaité du jeu, et en leur donnant l'occasion d'explorer intelligemment l'espace de recherche. Nous avons appliqué avec succès l'algorithme Q-Learning dans les cas d'environnements discrets et continus, mais avec des actions discrètes.

Il est également important d'étudier des situations où l'état d'action est également continu, et lorsque l'espace d'observation est beaucoup plus complexe, comme l'image de l'écran de jeu Atari. Dans ces problèmes, nous devons souvent utiliser des techniques d'apprentissage automatique plus puissantes, telles que les réseaux neuronaux, afin d'obtenir de bons résultats. Ces sujets plus avancés sont le sujet de notre prochain cours d'IA plus avancé.

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.