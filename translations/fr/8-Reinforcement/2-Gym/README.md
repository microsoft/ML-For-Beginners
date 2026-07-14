# Patinage CartPole

Le problème que nous avons résolu dans la leçon précédente peut sembler être un problème de jouet, pas vraiment applicable aux scénarios de la vie réelle. Ce n'est pas le cas, car de nombreux problèmes réels partagent également ce scénario - y compris jouer aux échecs ou au Go. Ils sont similaires, car nous avons aussi un plateau avec des règles données et un **état discret**.

## [Quiz avant le cours](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

Dans cette leçon, nous appliquerons les mêmes principes du Q-Learning à un problème avec un **état continu**, c’est-à-dire un état donné par un ou plusieurs nombres réels. Nous allons traiter le problème suivant :

> **Problème** : Si Peter veut échapper au loup, il doit pouvoir se déplacer plus vite. Nous verrons comment Peter peut apprendre à patiner, en particulier, à garder l’équilibre, en utilisant le Q-Learning.

![La grande évasion !](../../../../translated_images/fr/escape.18862db9930337e3.webp)

> Peter et ses amis font preuve de créativité pour échapper au loup ! Image par [Jen Looper](https://twitter.com/jenlooper)

Nous utiliserons une version simplifiée de l’équilibre connue sous le nom de problème **CartPole**. Dans le monde du cartpole, nous avons un curseur horizontal qui peut se déplacer à gauche ou à droite, et l’objectif est d’équilibrer un poteau vertical au sommet du curseur.

<img alt="un cartpole" src="../../../../translated_images/fr/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Prérequis

Dans cette leçon, nous utiliserons une bibliothèque appelée **OpenAI Gym** pour simuler différents **environnements**. Vous pouvez exécuter le code de cette leçon localement (par exemple depuis Visual Studio Code), dans ce cas la simulation s’ouvrira dans une nouvelle fenêtre. En exécutant le code en ligne, vous devrez peut-être effectuer quelques ajustements du code, comme décrit [ici](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Dans la leçon précédente, les règles du jeu et l’état étaient donnés par la classe `Board` que nous avions définie nous-mêmes. Ici, nous utiliserons un **environnement de simulation** spécial, qui simulera la physique derrière le poteau en équilibre. Un des environnements de simulation les plus populaires pour l’entraînement des algorithmes d’apprentissage par renforcement s’appelle un [Gym](https://gym.openai.com/), maintenu par [OpenAI](https://openai.com/). En utilisant ce gym, nous pouvons créer différents **environnements** allant d’une simulation de cartpole à des jeux Atari.

> **Note** : Vous pouvez voir d’autres environnements disponibles sur OpenAI Gym [ici](https://gym.openai.com/envs/#classic_control).

Tout d’abord, installons le gym et importons les bibliothèques requises (bloc de code 1) :

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Exercice - initialiser un environnement cartpole

Pour travailler avec un problème d’équilibre cartpole, nous devons initialiser l’environnement correspondant. Chaque environnement est associé à un :

- **espace d’observation** qui définit la structure des informations que nous recevons de l’environnement. Pour le problème cartpole, nous recevons la position du poteau, la vitesse et d’autres valeurs.

- **espace d’action** qui définit les actions possibles. Dans notre cas, l’espace d’actions est discret et se compose de deux actions - **gauche** et **droite**. (bloc de code 2)

1. Pour initialiser, tapez le code suivant :

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Pour voir comment l’environnement fonctionne, exécutons une courte simulation sur 100 étapes. À chaque étape, nous fournissons une des actions à prendre – dans cette simulation nous sélectionnons simplement une action au hasard parmi `action_space`.

1. Exécutez le code ci-dessous et voyez ce que cela donne.

    ✅ N’oubliez pas qu’il est préférable d’exécuter ce code sur une installation Python locale ! (bloc de code 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Vous devriez voir quelque chose de similaire à cette image :

    ![cartpole sans équilibre](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Pendant la simulation, nous devons obtenir des observations pour décider comment agir. En fait, la fonction step renvoie les observations actuelles, une fonction de récompense, et le drapeau done qui indique s’il est judicieux de continuer la simulation ou non : (bloc de code 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Vous devriez voir quelque chose comme ceci dans la sortie du notebook :

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

    Le vecteur d’observations renvoyé à chaque étape de la simulation contient les valeurs suivantes :
    - Position du chariot
    - Vitesse du chariot
    - Angle du poteau
    - Vitesse de rotation du poteau

1. Obtenez les valeurs minimum et maximum de ces nombres : (bloc de code 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Vous remarquerez aussi que la valeur de la récompense à chaque étape de la simulation est toujours 1. Cela est dû au fait que notre objectif est de survivre aussi longtemps que possible, c’est-à-dire garder le poteau dans une position raisonnablement verticale le plus longtemps possible.

    ✅ En fait, la simulation CartPole est considérée comme résolue si nous obtenons une récompense moyenne de 195 sur 100 essais consécutifs.

## Discrétisation de l’état

En Q-Learning, nous devons construire une Q-Table qui définit quoi faire à chaque état. Pour cela, l’état doit être **discret**, plus précisément, il doit contenir un nombre fini de valeurs discrètes. Ainsi, nous devons d’une certaine manière **discrétiser** nos observations, en les mappant à un ensemble fini d’états.

Il y a plusieurs façons de faire cela :

- **Diviser en bacs**. Si nous connaissons l’intervalle d’une certaine valeur, nous pouvons diviser cet intervalle en un nombre de **bacs**, puis remplacer la valeur par le numéro du bac auquel elle appartient. Cela peut être fait à l’aide de la méthode numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Dans ce cas, nous connaîtrons précisément la taille de l’état, car elle dépendra du nombre de bacs sélectionné pour la digitalisation.
  
✅ Nous pouvons utiliser une interpolation linéaire pour ramener les valeurs à un intervalle fini (disons, de -20 à 20), puis convertir les nombres en entiers en les arrondissant. Cela nous donne un peu moins de contrôle sur la taille de l’état, surtout si nous ne connaissons pas les plages exactes des valeurs d’entrée. Par exemple, dans notre cas, 2 des 4 valeurs n’ont pas de bornes supérieures/inférieures, ce qui peut conduire à un nombre infini d’états.

Dans notre exemple, nous allons choisir la deuxième approche. Comme vous le remarquerez plus tard, malgré les bornes supérieures/inférieures indéfinies, ces valeurs prennent rarement des valeurs en dehors de certains intervalles finis, donc ces états avec des valeurs extrêmes seront très rares.

1. Voici la fonction qui prendra l’observation de notre modèle et produira un tuple de 4 valeurs entières : (bloc de code 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Explorons aussi une autre méthode de discrétisation utilisant des bacs : (bloc de code 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervalles de valeurs pour chaque paramètre
    nbins = [20,20,10,10] # nombre de classes pour chaque paramètre
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Faisons maintenant une courte simulation et observons ces valeurs d’environnement discrètes. N’hésitez pas à essayer `discretize` et `discretize_bins` et à voir s’il y a une différence.

    ✅ discretize_bins renvoie le numéro du bac, qui commence à 0. Ainsi, pour des valeurs de la variable d’entrée proches de 0, il renvoie le numéro du milieu de l’intervalle (10). Dans discretize, nous ne nous sommes pas souciés de la plage des valeurs de sortie, les autorisant à être négatives, ainsi les valeurs d’état ne sont pas décalées, et 0 correspond à 0. (bloc de code 8)

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

    ✅ Décochez la ligne commençant par env.render si vous voulez voir comment l’environnement s’exécute. Sinon, vous pouvez l’exécuter en arrière-plan, ce qui est plus rapide. Nous utiliserons cette exécution "invisible" durant notre processus de Q-Learning.

## La structure de la Q-Table

Dans notre leçon précédente, l’état était une simple paire de nombres de 0 à 8, et donc il était pratique de représenter la Q-Table par un tenseur numpy de forme 8x8x2. Si nous utilisons la discrétisation par bacs, la taille de notre vecteur d’état est également connue, donc nous pouvons utiliser la même approche et représenter l’état par un tableau de forme 20x20x10x10x2 (ici 2 est la dimension de l’espace d’action, et les premières dimensions correspondent au nombre de bacs sélectionnés pour chacun des paramètres dans l’espace d’observation).

Cependant, parfois les dimensions précises de l’espace d’observation ne sont pas connues. Dans le cas de la fonction `discretize`, nous ne pouvons jamais être sûrs que notre état reste dans certaines limites, car certaines des valeurs originales ne sont pas bornées. Ainsi, nous utiliserons une approche légèrement différente et représenterons la Q-Table par un dictionnaire.

1. Utilisez la paire *(état, action)* comme clé du dictionnaire, et la valeur correspondra à la valeur de l’entrée dans la Q-Table. (bloc de code 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Ici, nous définissons aussi une fonction `qvalues()`, qui renvoie une liste des valeurs de la Q-Table pour un état donné qui correspond à toutes les actions possibles. Si l’entrée n’est pas présente dans la Q-Table, nous retournerons 0 par défaut.

## Commençons le Q-Learning

Maintenant, nous sommes prêts à apprendre à Peter à garder l’équilibre !

1. D’abord, définissons quelques hyperparamètres : (bloc de code 10)

    ```python
    # hyperparamètres
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Ici, `alpha` est le **taux d’apprentissage** qui définit dans quelle mesure nous devons ajuster les valeurs actuelles de la Q-Table à chaque étape. Dans la leçon précédente, nous avons commencé avec 1, puis diminué `alpha` vers des valeurs plus basses pendant l’entraînement. Dans cet exemple, nous le garderons constant par simplicité, et vous pouvez expérimenter en ajustant `alpha` plus tard.

    `gamma` est le **facteur d’actualisation** qui indique dans quelle mesure nous devons privilégier la récompense future par rapport à la récompense actuelle.

    `epsilon` est le **facteur exploration/exploitation** qui détermine si nous devons privilégier l’exploration à l’exploitation ou vice versa. Dans notre algorithme, dans `epsilon` pourcents des cas, nous sélectionnerons l’action suivante selon les valeurs de la Q-Table, et dans le reste des cas nous exécuterons une action aléatoire. Cela nous permettra d’explorer des zones de l’espace de recherche que nous n’avons jamais vues auparavant.

    ✅ En termes d’équilibre - choisir une action aléatoire (exploration) agit comme un coup aléatoire dans la mauvaise direction, et le poteau devra apprendre comment récupérer l’équilibre après ces "erreurs".

### Améliorer l’algorithme

Nous pouvons aussi apporter deux améliorations à notre algorithme de la leçon précédente :

- **Calculer la récompense cumulative moyenne**, sur un certain nombre de simulations. Nous afficherons la progression toutes les 5000 itérations, et nous ferons la moyenne de notre récompense cumulative sur cette période. Cela signifie que si nous obtenons plus de 195 points - nous pouvons considérer le problème résolu, avec une qualité encore meilleure que requise.
  
- **Calculer le maximum de la récompense cumulative moyenne**, `Qmax`, et nous stockerons la Q-Table correspondant à ce résultat. Lorsque vous exécuterez l’entraînement, vous remarquerez que parfois, la récompense moyenne cumulative commence à chuter, et nous voulons garder les valeurs de la Q-Table correspondant au meilleur modèle observé pendant l’entraînement.

1. Rassembler toutes les récompenses cumulatives à chaque simulation dans le vecteur `rewards` pour un tracé ultérieur. (bloc de code  11)

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
        # == faire la simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - choisir l'action selon les probabilités de la Q-Table
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - choisir l'action aléatoirement
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Imprimer périodiquement les résultats et calculer la récompense moyenne ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Ce que vous pouvez remarquer d’après ces résultats :

- **Proche de notre objectif**. Nous sommes très proches d’atteindre l’objectif d’obtenir 195 récompenses cumulées sur 100+ exécutions consécutives de la simulation, ou nous l’avons peut-être réellement atteint ! Même si nous obtenons des chiffres plus petits, nous ne savons pas encore, car nous faisons la moyenne sur 5000 exécutions alors que 100 exécutions seulement sont requises officiellement.
  
- **La récompense commence à chuter**. Parfois, la récompense commence à diminuer, ce qui signifie que nous pouvons "détruire" les valeurs déjà apprises dans la Q-Table avec des valeurs qui aggravent la situation.

Cette observation est plus clairement visible si nous traçons la progression de l’entraînement.

## Tracer la progression de l’entraînement

Pendant l’entraînement, nous avons collecté la valeur de la récompense cumulative à chaque itération dans le vecteur `rewards`. Voici à quoi cela ressemble lorsque nous la traçons en fonction du numéro d’itération :

```python
plt.plot(rewards)
```

![progression brute](../../../../translated_images/fr/train_progress_raw.2adfdf2daea09c59.webp)

Sur ce graphique, il est impossible de tirer des conclusions, car en raison de la nature stochastique du processus d’entraînement, la durée des sessions d’entraînement varie fortement. Pour mieux comprendre ce graphique, nous pouvons calculer la **moyenne glissante** sur une série d’expériences, disons 100. Cela peut être fait aisément avec `np.convolve` : (bloc de code 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progression de l’entraînement](../../../../translated_images/fr/train_progress_runav.c71694a8fa9ab359.webp)

## Variation des hyperparamètres

Pour rendre l’apprentissage plus stable, il est judicieux d’ajuster certains de nos hyperparamètres durant l’entraînement. En particulier :

- **Pour le taux d’apprentissage**, `alpha`, nous pouvons commencer avec des valeurs proches de 1, puis continuer à diminuer ce paramètre. Avec le temps, nous obtiendrons de bonnes valeurs de probabilité dans la Q-Table, et donc nous devrions les ajuster légèrement, sans les écraser complètement avec des nouvelles valeurs.

- **Augmenter epsilon**. Nous pourrions augmenter lentement `epsilon`, afin d’explorer moins et d’exploiter plus. Il est probablement judicieux de commencer avec une valeur basse de `epsilon`, puis la faire monter presque à 1.

> **Tâche 1** : Testez les valeurs des hyperparamètres et voyez si vous pouvez obtenir une récompense cumulative plus élevée. Parvenez-vous à dépasser 195 ?


> **Tâche 2** : Pour résoudre formellement le problème, vous devez obtenir une récompense moyenne de 195 sur 100 exécutions consécutives. Mesurez cela pendant l'entraînement et assurez-vous d'avoir formellement résolu le problème !

## Voir le résultat en action

Il serait intéressant de voir comment le modèle entraîné se comporte réellement. Exécutons la simulation et suivons la même stratégie de sélection d'action que pendant l'entraînement, en échantillonnant selon la distribution de probabilité dans la Q-Table : (bloc de code 13)

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

![un pendule inversé en équilibre](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Défi

> **Tâche 3** : Ici, nous utilisions la copie finale de la Q-Table, qui peut ne pas être la meilleure. Rappelez-vous que nous avons stocké la Q-Table la plus performante dans la variable `Qbest` ! Essayez le même exemple avec la Q-Table la plus performante en recopiant `Qbest` dans `Q` et voyez si vous remarquez la différence.

> **Tâche 4** : Ici, nous ne sélectionnions pas la meilleure action à chaque étape, mais plutôt un échantillonnage selon la distribution de probabilité correspondante. Ne serait-il pas plus logique de toujours sélectionner la meilleure action, avec la valeur la plus haute dans la Q-Table ? Cela peut se faire en utilisant la fonction `np.argmax` pour trouver le numéro de l'action correspondant à la valeur la plus élevée dans la Q-Table. Implémentez cette stratégie et voyez si cela améliore l'équilibre.

## [Quiz après cours](https://ff-quizzes.netlify.app/en/ml/)

## Devoir
[Entraîner une Mountain Car](assignment.md)

## Conclusion

Nous avons maintenant appris comment entraîner des agents à obtenir de bons résultats en leur fournissant simplement une fonction de récompense qui définit l'état désiré du jeu, et en leur donnant l'opportunité d'explorer intelligemment l'espace de recherche. Nous avons appliqué avec succès l'algorithme de Q-Learning dans des environnements discrets et continus, mais avec des actions discrètes.

Il est aussi important d'étudier les situations où l'état d'action est également continu, et où l'espace d'observation est beaucoup plus complexe, comme l'image de l'écran du jeu Atari. Dans ces problèmes, nous devons souvent utiliser des techniques d'apprentissage automatique plus puissantes, telles que les réseaux neuronaux, afin d'obtenir de bons résultats. Ces sujets plus avancés feront l'objet de notre prochain cours d'IA plus avancé.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Avertissement** :
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforçions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source faisant autorité. Pour les informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous ne saurions être tenus responsables des malentendus ou erreurs d'interprétation découlant de l'utilisation de cette traduction.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->