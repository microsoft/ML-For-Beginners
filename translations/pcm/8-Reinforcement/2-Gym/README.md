<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-11-18T18:13:07+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "pcm"
}
-->
# CartPole Skating

Di problem we dey solve for di last lesson fit look like play-play problem wey no go fit work for real life. But e no be so, because plenty real life problem dey like dis one - like to play Chess or Go. Dem be di same, because we get board wey get rules and **discrete state**.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

For dis lesson, we go use di same Q-Learning principles for problem wey get **continuous state**, wey mean say di state dey show as one or more real numbers. Di problem wey we go look na:

> **Problem**: If Peter wan run comot from di wolf, e need sabi move fast. We go see how Peter go fit learn how to skate, especially to balance, using Q-Learning.

![Di great escape!](../../../../translated_images/escape.18862db9930337e3fce23a9b6a76a06445f229dadea2268e12a6f0a1fde12115.pcm.png)

> Peter and im padi dem dey creative to run comot from di wolf! Image by [Jen Looper](https://twitter.com/jenlooper)

We go use one simple version of balancing wey dem dey call **CartPole** problem. For di cartpole world, we get one horizontal slider wey fit move left or right, and di goal na to balance one vertical pole for di top of di slider.

<img alt="a cartpole" src="../../../../translated_images/cartpole.b5609cc0494a14f75d121299495ae24fd8f1c30465e7b40961af94ecda2e1cd0.pcm.png" width="200"/>

## Prerequisites

For dis lesson, we go use one library wey dem dey call **OpenAI Gym** to simulate different **environments**. You fit run di code for dis lesson for your computer (like for Visual Studio Code), and di simulation go open for new window. If you dey run di code online, you fit need change di code small, as dem explain [here](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

For di last lesson, di rules of di game and di state dey inside di `Board` class wey we create by ourselves. For here, we go use one special **simulation environment**, wey go simulate di physics wey dey behind di balancing pole. One of di popular simulation environments wey dem dey use train reinforcement learning algorithms na [Gym](https://gym.openai.com/), wey [OpenAI](https://openai.com/) dey maintain. With dis gym, we fit create different **environments** from cartpole simulation to Atari games.

> **Note**: You fit see di other environments wey OpenAI Gym get [here](https://gym.openai.com/envs/#classic_control). 

First, make we install di gym and import di libraries wey we need (code block 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Exercise - initialize a cartpole environment

To work with di cartpole balancing problem, we need to initialize di environment wey dey follow am. Each environment get:

- **Observation space** wey dey show di structure of di information wey we dey get from di environment. For di cartpole problem, we dey get di position of di pole, velocity and some other values.

- **Action space** wey dey show di actions wey we fit take. For our case, di action space dey discrete, and e get two actions - **left** and **right**. (code block 2)

1. To initialize, type dis code:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

To see how di environment dey work, make we run one short simulation for 100 steps. For each step, we go provide one action wey dem go take - for dis simulation we go just dey randomly select action from `action_space`. 

1. Run di code below and see wetin e go do.

    ✅ Remember say e better make you run dis code for local Python installation! (code block 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    You suppose dey see something wey resemble dis image:

    ![non-balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. For di simulation, we need dey collect observations to decide wetin we go do. Di step function dey return di current observations, reward function, and di done flag wey dey show whether e make sense to continue di simulation or not: (code block 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    You go see something like dis for di notebook output:

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

    Di observation vector wey dem dey return for each step of di simulation get di following values:
    - Position of cart
    - Velocity of cart
    - Angle of pole
    - Rotation rate of pole

1. Get di minimum and maximum value of di numbers: (code block 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    You fit notice say di reward value for each simulation step dey always 1. Dis na because our goal na to survive as long as we fit, wey mean say make di pole dey reasonably vertical for di longest time.

    ✅ Di CartPole simulation dey considered solved if we fit get di average reward of 195 for 100 consecutive trials.

## State discretization

For Q-Learning, we need to build Q-Table wey go show wetin to do for each state. To fit do dis, di state need dey **discreet**, wey mean say e go get finite number of discrete values. So, we need find way to **discretize** our observations, wey go map dem to one finite set of states.

Some ways dey to do dis:

- **Divide into bins**. If we sabi di interval of one value, we fit divide di interval into number of **bins**, and then replace di value with di bin number wey e belong to. We fit use numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) method to do dis. For dis case, we go sabi di state size well, because e go depend on di number of bins wey we select for di digitalization.
  
✅ We fit use linear interpolation to bring values to one finite interval (like, from -20 to 20), and then convert di numbers to integers by rounding dem. Dis one no go give us control for di size of di state well, especially if we no sabi di exact ranges of di input values. For example, for our case 2 out of 4 values no get upper/lower bounds for their values, wey fit make di number of states infinite.

For our example, we go use di second approach. As you go notice later, even though di upper/lower bounds no dey defined, di values rarely dey go outside certain finite intervals, so di states wey get extreme values go dey very rare.

1. Dis na di function wey go take di observation from our model and produce one tuple of 4 integer values: (code block 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Make we also try another discretization method wey dey use bins: (code block 7)

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

1. Make we now run one short simulation and observe di discrete environment values. You fit try both `discretize` and `discretize_bins` and see if difference dey.

    ✅ discretize_bins dey return di bin number, wey dey start from 0. So for values of input variable wey dey around 0 e dey return di number from di middle of di interval (10). For discretize, we no care about di range of output values, we allow dem to dey negative, so di state values no shift, and 0 dey correspond to 0. (code block 8)

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

    ✅ Uncomment di line wey start with env.render if you wan see how di environment dey execute. If not, you fit execute am for background, wey go fast pass. We go use dis "invisible" execution during our Q-Learning process.

## Di Q-Table structure

For di last lesson, di state na simple pair of numbers from 0 to 8, so e dey easy to represent Q-Table as numpy tensor wey get shape of 8x8x2. If we dey use bins discretization, di size of our state vector dey known too, so we fit use di same approach and represent state as array wey get shape 20x20x10x10x2 (here 2 na di dimension of action space, and di first dimensions dey correspond to di number of bins wey we select to use for each of di parameters for observation space).

But sometimes di exact dimensions of di observation space no dey known. For di `discretize` function, we no fit sure say our state go dey inside certain limits, because some of di original values no dey bound. So, we go use one different approach and represent Q-Table as dictionary. 

1. Use di pair *(state,action)* as di dictionary key, and di value go correspond to Q-Table entry value. (code block 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Here we also define one function `qvalues()`, wey dey return list of Q-Table values for one given state wey dey correspond to all possible actions. If di entry no dey for di Q-Table, we go return 0 as default.

## Make we start Q-Learning

Now we don ready to teach Peter how to balance!

1. First, make we set some hyperparameters: (code block 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Here, `alpha` na di **learning rate** wey dey show how we go adjust di current values of Q-Table for each step. For di last lesson we start with 1, and then reduce `alpha` to lower values during training. For dis example we go keep am constant just to make am simple, and you fit try experiment with adjusting `alpha` values later.

    `gamma` na di **discount factor** wey dey show how we go prioritize future reward over current reward.

    `epsilon` na di **exploration/exploitation factor** wey dey determine whether we go prefer exploration to exploitation or vice versa. For our algorithm, we go select di next action according to Q-Table values for `epsilon` percent of di cases, and for di remaining cases we go execute random action. Dis go allow us explore areas of di search space wey we never see before. 

    ✅ For balancing - to choose random action (exploration) go act like random punch for wrong direction, and di pole go need learn how to recover di balance from di "mistakes."

### Improve di algorithm

We fit also make two improvements to our algorithm from di last lesson:

- **Calculate average cumulative reward**, over number of simulations. We go print di progress every 5000 iterations, and we go average di cumulative reward over dat time. If we fit get more than 195 point - we fit consider di problem solved, with even better quality than dem require.
  
- **Calculate maximum average cumulative result**, `Qmax`, and we go store di Q-Table wey dey correspond to dat result. When you dey run di training you go notice say sometimes di average cumulative result go start to drop, and we wan keep di values of Q-Table wey dey correspond to di best model wey we observe during training.

1. Collect all cumulative rewards for each simulation for `rewards` vector for plotting later. (code block 11)

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

Wetin you fit notice from di results:

- **Close to our goal**. We dey very close to achieve di goal of getting 195 cumulative rewards over 100+ consecutive runs of di simulation, or we fit don achieve am! Even if we dey get smaller numbers, we no go know, because we dey average over 5000 runs, and only 100 runs dey required for di formal criteria.
  
- **Reward dey start to drop**. Sometimes di reward go start to drop, wey mean say we fit "scatter" di values wey we don learn for di Q-Table with di ones wey dey make di situation worse.

Dis observation go show well if we plot di training progress.

## Plotting Training Progress

During training, we don collect di cumulative reward value for each of di iterations into `rewards` vector. Dis na how e go look if we plot am against di iteration number:

```python
plt.plot(rewards)
```

![raw progress](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.pcm.png)

From dis graph, e no dey possible to talk anything, because di stochastic training process dey make di length of training sessions vary well. To make di graph make sense, we fit calculate di **running average** over series of experiments, like 100. We fit do dis well using `np.convolve`: (code block 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![training progress](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.pcm.png)

## Varying hyperparameters

To make di learning stable, e make sense to adjust some of our hyperparameters during training. Especially:

- **For learning rate**, `alpha`, we fit start with values wey dey close to 1, and then dey reduce di parameter. With time, we go dey get good probability values for di Q-Table, and so we go dey adjust dem small-small, and no dey overwrite dem completely with new values.

- **Increase epsilon**. We fit wan increase di `epsilon` small-small, so we go dey explore less and exploit more. E fit make sense to start with lower value of `epsilon`, and move am up to almost 1.

> **Task 1**: Play with di hyperparameter values and see if you fit achieve higher cumulative reward. You dey get above 195?
> **Task 2**: To solve di problem well-well, you go need get 195 average reward for 100 runs wey follow each other. Dey measure am during di training and make sure say you don solve di problem well!

## See how di result dey work

E go dey interesting to see how di model wey you don train go behave. Make we run di simulation and follow di same action selection strategy wey we use during training, dey sample based on di probability distribution wey dey for Q-Table: (code block 13)

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

You suppose see something like dis:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Challenge

> **Task 3**: For here, we dey use di final copy of Q-Table, but e fit no be di best one. Remember say we don store di best-performing Q-Table for `Qbest` variable! Try di same example with di best-performing Q-Table by copying `Qbest` go `Q` and see whether you go notice any difference.

> **Task 4**: For here, we no dey select di best action for each step, but we dey sample based on di probability distribution. E no go make sense to always select di best action, wey get di highest Q-Table value? You fit use `np.argmax` function to find di action number wey get di highest Q-Table value. Try implement dis strategy and see whether e go improve di balancing.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Assignment
[Train a Mountain Car](assignment.md)

## Conclusion

We don learn how to train agents to get better results just by giving dem reward function wey define di kind state wey we want for di game, and by giving dem chance to explore di search space with sense. We don use di Q-Learning algorithm for cases wey get discrete and continuous environments, but with discrete actions.

E dey important to also study situations wey di action state go dey continuous, and when di observation space go dey more complex, like di image from Atari game screen. For dis kind problems, we go need use stronger machine learning techniques, like neural networks, to get better results. Dis advanced topics go dey for our next advanced AI course.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis docu don use AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator) take translate am. Even though we dey try make sure say e correct, abeg no forget say automatic translation fit get mistake or no dey accurate well. Di original docu for di language wey dem write am first na di main correct one. For important information, e good make una use professional human translation. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because of dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->