<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9660fbd80845c59c15715cb418cd6e23",
  "translation_date": "2025-09-04T00:25:46+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "fr"
}
-->
## Prérequis

Dans cette leçon, nous utiliserons une bibliothèque appelée **OpenAI Gym** pour simuler différents **environnements**. Vous pouvez exécuter le code de cette leçon localement (par exemple, depuis Visual Studio Code), auquel cas la simulation s'ouvrira dans une nouvelle fenêtre. Si vous exécutez le code en ligne, vous devrez peut-être apporter quelques ajustements au code, comme décrit [ici](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Dans la leçon précédente, les règles du jeu et l'état étaient définis par la classe `Board` que nous avons créée nous-mêmes. Ici, nous utiliserons un **environnement de simulation** spécial, qui simulera la physique derrière le balancement du poteau. L'un des environnements de simulation les plus populaires pour entraîner des algorithmes d'apprentissage par renforcement est appelé [Gym](https://gym.openai.com/), maintenu par [OpenAI](https://openai.com/). Grâce à ce gym, nous pouvons créer différents **environnements**, allant de la simulation de CartPole aux jeux Atari.

> **Note** : Vous pouvez voir les autres environnements disponibles dans OpenAI Gym [ici](https://gym.openai.com/envs/#classic_control).

Tout d'abord, installons le gym et importons les bibliothèques nécessaires (bloc de code 1) :

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Exercice - initialiser un environnement CartPole

Pour travailler sur le problème d'équilibrage du CartPole, nous devons initialiser l'environnement correspondant. Chaque environnement est associé à :

- **Observation space** qui définit la structure des informations que nous recevons de l'environnement. Pour le problème CartPole, nous recevons la position du poteau, la vitesse et quelques autres valeurs.

- **Action space** qui définit les actions possibles. Dans notre cas, l'espace d'action est discret et se compose de deux actions : **gauche** et **droite**. (bloc de code 2)

1. Pour initialiser, tapez le code suivant :

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Pour voir comment fonctionne l'environnement, exécutons une courte simulation de 100 étapes. À chaque étape, nous fournissons une action à effectuer - dans cette simulation, nous sélectionnons simplement une action au hasard dans `action_space`.

1. Exécutez le code ci-dessous et observez le résultat.

    ✅ N'oubliez pas qu'il est préférable d'exécuter ce code sur une installation locale de Python ! (bloc de code 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Vous devriez voir quelque chose de similaire à cette image :

    ![CartPole sans équilibre](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Pendant la simulation, nous devons obtenir des observations pour décider comment agir. En fait, la fonction step retourne les observations actuelles, une fonction de récompense et un indicateur `done` qui indique s'il est pertinent de continuer la simulation ou non : (bloc de code 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Vous verrez quelque chose comme ceci dans la sortie du notebook :

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

    Le vecteur d'observation retourné à chaque étape de la simulation contient les valeurs suivantes :
    - Position du chariot
    - Vitesse du chariot
    - Angle du poteau
    - Taux de rotation du poteau

1. Obtenez les valeurs minimales et maximales de ces nombres : (bloc de code 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Vous remarquerez également que la valeur de récompense à chaque étape de la simulation est toujours 1. Cela s'explique par le fait que notre objectif est de survivre le plus longtemps possible, c'est-à-dire de maintenir le poteau dans une position raisonnablement verticale pendant la période la plus longue possible.

    ✅ En fait, la simulation CartPole est considérée comme résolue si nous parvenons à obtenir une récompense moyenne de 195 sur 100 essais consécutifs.

## Discrétisation de l'état

Dans le Q-Learning, nous devons construire une Q-Table qui définit quoi faire à chaque état. Pour ce faire, l'état doit être **discret**, plus précisément, il doit contenir un nombre fini de valeurs discrètes. Ainsi, nous devons d'une manière ou d'une autre **discrétiser** nos observations, en les mappant à un ensemble fini d'états.

Il existe plusieurs façons de procéder :

- **Diviser en intervalles**. Si nous connaissons l'intervalle d'une certaine valeur, nous pouvons diviser cet intervalle en un certain nombre d'**intervalles**, puis remplacer la valeur par le numéro de l'intervalle auquel elle appartient. Cela peut être fait en utilisant la méthode [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) de numpy. Dans ce cas, nous connaîtrons précisément la taille de l'état, car elle dépendra du nombre d'intervalles que nous sélectionnons pour la digitalisation.

✅ Nous pouvons utiliser l'interpolation linéaire pour ramener les valeurs à un certain intervalle fini (par exemple, de -20 à 20), puis convertir les nombres en entiers en les arrondissant. Cela nous donne un peu moins de contrôle sur la taille de l'état, surtout si nous ne connaissons pas les plages exactes des valeurs d'entrée. Par exemple, dans notre cas, 2 des 4 valeurs n'ont pas de limites supérieures/inférieures, ce qui peut entraîner un nombre infini d'états.

Dans notre exemple, nous opterons pour la deuxième approche. Comme vous le remarquerez plus tard, malgré l'absence de limites supérieures/inférieures définies, ces valeurs prennent rarement des valeurs en dehors de certains intervalles finis, donc ces états avec des valeurs extrêmes seront très rares.

1. Voici la fonction qui prendra l'observation de notre modèle et produira un tuple de 4 valeurs entières : (bloc de code 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Explorons également une autre méthode de discrétisation utilisant des intervalles : (bloc de code 7)

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

1. Exécutons maintenant une courte simulation et observons ces valeurs discrètes de l'environnement. N'hésitez pas à essayer `discretize` et `discretize_bins` et à voir s'il y a une différence.

    ✅ `discretize_bins` retourne le numéro de l'intervalle, qui commence à 0. Ainsi, pour les valeurs de la variable d'entrée autour de 0, il retourne le numéro du milieu de l'intervalle (10). Dans `discretize`, nous ne nous sommes pas souciés de la plage des valeurs de sortie, leur permettant d'être négatives, donc les valeurs d'état ne sont pas décalées, et 0 correspond à 0. (bloc de code 8)

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

    ✅ Décommentez la ligne commençant par env.render si vous souhaitez voir comment l'environnement s'exécute. Sinon, vous pouvez l'exécuter en arrière-plan, ce qui est plus rapide. Nous utiliserons cette exécution "invisible" pendant notre processus de Q-Learning.

## La structure de la Q-Table

Dans notre leçon précédente, l'état était une simple paire de nombres allant de 0 à 8, et il était donc pratique de représenter la Q-Table par un tenseur numpy de forme 8x8x2. Si nous utilisons la discrétisation par intervalles, la taille de notre vecteur d'état est également connue, nous pouvons donc utiliser la même approche et représenter l'état par un tableau de forme 20x20x10x10x2 (ici 2 est la dimension de l'espace d'action, et les premières dimensions correspondent au nombre d'intervalles que nous avons choisi d'utiliser pour chacun des paramètres dans l'espace d'observation).

Cependant, parfois, les dimensions précises de l'espace d'observation ne sont pas connues. Dans le cas de la fonction `discretize`, nous ne pouvons jamais être sûrs que notre état reste dans certaines limites, car certaines des valeurs originales ne sont pas bornées. Ainsi, nous utiliserons une approche légèrement différente et représenterons la Q-Table par un dictionnaire.

1. Utilisez la paire *(state,action)* comme clé du dictionnaire, et la valeur correspondra à la valeur de l'entrée de la Q-Table. (bloc de code 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Ici, nous définissons également une fonction `qvalues()`, qui retourne une liste de valeurs de la Q-Table pour un état donné correspondant à toutes les actions possibles. Si l'entrée n'est pas présente dans la Q-Table, nous retournerons 0 par défaut.

## Commençons le Q-Learning

Nous sommes maintenant prêts à apprendre à Peter à maintenir l'équilibre !

1. Tout d'abord, définissons quelques hyperparamètres : (bloc de code 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Ici, `alpha` est le **taux d'apprentissage** qui définit dans quelle mesure nous devons ajuster les valeurs actuelles de la Q-Table à chaque étape. Dans la leçon précédente, nous avons commencé avec 1, puis diminué `alpha` à des valeurs plus faibles pendant l'entraînement. Dans cet exemple, nous le garderons constant pour simplifier, et vous pourrez expérimenter avec l'ajustement des valeurs de `alpha` plus tard.

    `gamma` est le **facteur d'actualisation** qui montre dans quelle mesure nous devons privilégier la récompense future par rapport à la récompense actuelle.

    `epsilon` est le **facteur d'exploration/exploitation** qui détermine si nous devons préférer l'exploration à l'exploitation ou vice versa. Dans notre algorithme, nous sélectionnerons dans `epsilon` pourcentage des cas la prochaine action selon les valeurs de la Q-Table, et dans le reste des cas, nous exécuterons une action aléatoire. Cela nous permettra d'explorer des zones de l'espace de recherche que nous n'avons jamais vues auparavant.

    ✅ En termes d'équilibrage - choisir une action aléatoire (exploration) agirait comme un coup aléatoire dans la mauvaise direction, et le poteau devrait apprendre à récupérer l'équilibre à partir de ces "erreurs".

### Améliorer l'algorithme

Nous pouvons également apporter deux améliorations à notre algorithme de la leçon précédente :

- **Calculer la récompense cumulative moyenne**, sur un certain nombre de simulations. Nous imprimerons les progrès tous les 5000 itérations, et nous ferons la moyenne de notre récompense cumulative sur cette période. Cela signifie que si nous obtenons plus de 195 points, nous pouvons considérer le problème comme résolu, avec une qualité encore supérieure à celle requise.

- **Calculer le résultat cumulatif moyen maximum**, `Qmax`, et nous stockerons la Q-Table correspondant à ce résultat. Lorsque vous exécutez l'entraînement, vous remarquerez que parfois le résultat cumulatif moyen commence à diminuer, et nous voulons conserver les valeurs de la Q-Table qui correspondent au meilleur modèle observé pendant l'entraînement.

1. Collectez toutes les récompenses cumulatives à chaque simulation dans le vecteur `rewards` pour un futur tracé. (bloc de code 11)

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

- **Proche de notre objectif**. Nous sommes très proches d'atteindre l'objectif de 195 récompenses cumulatives sur 100+ exécutions consécutives de la simulation, ou nous l'avons peut-être déjà atteint ! Même si nous obtenons des nombres plus faibles, nous ne le savons pas, car nous faisons la moyenne sur 5000 exécutions, et seuls 100 exécutions sont nécessaires dans les critères formels.

- **La récompense commence à diminuer**. Parfois, la récompense commence à diminuer, ce qui signifie que nous pouvons "détruire" les valeurs déjà apprises dans la Q-Table avec celles qui aggravent la situation.

Cette observation est plus clairement visible si nous traçons les progrès de l'entraînement.

## Tracer les progrès de l'entraînement

Pendant l'entraînement, nous avons collecté la valeur de la récompense cumulative à chaque itération dans le vecteur `rewards`. Voici à quoi cela ressemble lorsque nous le traçons par rapport au numéro d'itération :

```python
plt.plot(rewards)
```

![progrès brut](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.fr.png)

À partir de ce graphique, il n'est pas possible de tirer des conclusions, car en raison de la nature du processus d'entraînement stochastique, la durée des sessions d'entraînement varie considérablement. Pour donner plus de sens à ce graphique, nous pouvons calculer la **moyenne mobile** sur une série d'expériences, disons 100. Cela peut être fait facilement en utilisant `np.convolve` : (bloc de code 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progrès de l'entraînement](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.fr.png)

## Variation des hyperparamètres

Pour rendre l'apprentissage plus stable, il est judicieux d'ajuster certains de nos hyperparamètres pendant l'entraînement. En particulier :

- **Pour le taux d'apprentissage**, `alpha`, nous pouvons commencer avec des valeurs proches de 1, puis diminuer progressivement le paramètre. Avec le temps, nous obtiendrons de bonnes probabilités dans la Q-Table, et nous devrions donc les ajuster légèrement, et non les écraser complètement avec de nouvelles valeurs.

- **Augmenter epsilon**. Nous pouvons vouloir augmenter lentement `epsilon`, afin d'explorer moins et d'exploiter davantage. Il est probablement judicieux de commencer avec une valeur plus faible de `epsilon`, et de monter jusqu'à presque 1.
> **Tâche 1** : Expérimentez avec les valeurs des hyperparamètres et voyez si vous pouvez obtenir une récompense cumulative plus élevée. Atteignez-vous plus de 195 ?
> **Tâche 2** : Pour résoudre formellement le problème, vous devez atteindre une récompense moyenne de 195 sur 100 exécutions consécutives. Mesurez cela pendant l'entraînement et assurez-vous d'avoir résolu le problème de manière formelle !

## Voir le résultat en action

Il serait intéressant de voir comment le modèle entraîné se comporte réellement. Lançons la simulation et suivons la même stratégie de sélection d'actions qu'au cours de l'entraînement, en échantillonnant selon la distribution de probabilité dans la Q-Table : (bloc de code 13)

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

![un chariot en équilibre](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Défi

> **Tâche 3** : Ici, nous utilisions la copie finale de la Q-Table, qui n'est peut-être pas la meilleure. Rappelez-vous que nous avons stocké la Q-Table la plus performante dans la variable `Qbest` ! Essayez le même exemple avec la Q-Table la plus performante en copiant `Qbest` dans `Q` et voyez si vous remarquez une différence.

> **Tâche 4** : Ici, nous ne sélectionnions pas la meilleure action à chaque étape, mais plutôt en échantillonnant avec la distribution de probabilité correspondante. Serait-il plus logique de toujours sélectionner la meilleure action, celle avec la valeur la plus élevée dans la Q-Table ? Cela peut être fait en utilisant la fonction `np.argmax` pour trouver le numéro de l'action correspondant à la valeur la plus élevée dans la Q-Table. Implémentez cette stratégie et voyez si cela améliore l'équilibre.

## [Quiz post-conférence](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/)

## Devoir
[Entraîner une voiture de montagne](assignment.md)

## Conclusion

Nous avons maintenant appris à entraîner des agents pour obtenir de bons résultats simplement en leur fournissant une fonction de récompense qui définit l'état souhaité du jeu, et en leur donnant l'opportunité d'explorer intelligemment l'espace de recherche. Nous avons appliqué avec succès l'algorithme de Q-Learning dans des environnements discrets et continus, mais avec des actions discrètes.

Il est également important d'étudier des situations où l'état des actions est continu, et où l'espace d'observation est beaucoup plus complexe, comme l'image de l'écran d'un jeu Atari. Dans ces problèmes, nous devons souvent utiliser des techniques d'apprentissage automatique plus puissantes, telles que les réseaux neuronaux, pour obtenir de bons résultats. Ces sujets plus avancés seront abordés dans notre prochain cours d'IA plus avancé.

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.