<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T09:09:36+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "zh"
}
-->
# CartPole 滑行

我们在上一课中解决的问题可能看起来像一个玩具问题，似乎与现实生活场景无关。但事实并非如此，因为许多现实世界的问题也具有类似的场景——包括下棋或围棋。这些问题类似，因为我们也有一个带有规则的棋盘和一个**离散状态**。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

## 介绍

在本课中，我们将把 Q-Learning 的相同原理应用于一个具有**连续状态**的问题，即状态由一个或多个实数表示。我们将处理以下问题：

> **问题**：如果彼得想要逃离狼的追捕，他需要能够移动得更快。我们将看到彼得如何通过 Q-Learning 学习滑行，特别是保持平衡。

![伟大的逃亡！](../../../../8-Reinforcement/2-Gym/images/escape.png)

> 彼得和他的朋友们发挥创意逃离狼的追捕！图片由 [Jen Looper](https://twitter.com/jenlooper) 提供

我们将使用一种称为 **CartPole** 的简化平衡问题。在 CartPole 世界中，我们有一个可以左右移动的水平滑块，目标是让滑块顶部的垂直杆保持平衡。

## 前置知识

在本课中，我们将使用一个名为 **OpenAI Gym** 的库来模拟不同的**环境**。你可以在本地运行本课的代码（例如在 Visual Studio Code 中），此时模拟会在新窗口中打开。如果在线运行代码，你可能需要对代码进行一些调整，具体描述见[这里](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7)。

## OpenAI Gym

在上一课中，游戏规则和状态由我们自己定义的 `Board` 类提供。在这里，我们将使用一个特殊的**模拟环境**，它会模拟平衡杆的物理过程。训练强化学习算法最流行的模拟环境之一是 [Gym](https://gym.openai.com/)，由 [OpenAI](https://openai.com/) 维护。通过使用这个 Gym，我们可以创建不同的**环境**，从 CartPole 模拟到 Atari 游戏。

> **注意**：你可以在 OpenAI Gym 中查看其他可用的环境 [这里](https://gym.openai.com/envs/#classic_control)。

首先，让我们安装 Gym 并导入所需的库（代码块 1）：

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## 练习 - 初始化一个 CartPole 环境

要处理 CartPole 平衡问题，我们需要初始化相应的环境。每个环境都与以下内容相关联：

- **观察空间**：定义我们从环境中接收到的信息结构。对于 CartPole 问题，我们接收到杆的位置、速度以及其他一些值。

- **动作空间**：定义可能的动作。在我们的例子中，动作空间是离散的，由两个动作组成——**左**和**右**。（代码块 2）

1. 要初始化，请输入以下代码：

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

为了了解环境如何工作，让我们运行一个短暂的模拟，持续 100 步。在每一步中，我们提供一个动作——在这个模拟中，我们只是随机选择一个来自 `action_space` 的动作。

1. 运行以下代码并查看结果。

    ✅ 请记住，最好在本地 Python 安装中运行此代码！（代码块 3）

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    你应该会看到类似于以下图片的内容：

    ![未平衡的 CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. 在模拟过程中，我们需要获取观察值以决定如何行动。实际上，`step` 函数会返回当前的观察值、奖励函数以及一个表示是否继续模拟的完成标志：（代码块 4）

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    你将在笔记本输出中看到类似以下的内容：

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

    在模拟的每一步返回的观察向量包含以下值：
    - 小车的位置
    - 小车的速度
    - 杆的角度
    - 杆的旋转速率

1. 获取这些数值的最小值和最大值：（代码块 5）

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    你可能还会注意到，每次模拟步骤的奖励值始终为 1。这是因为我们的目标是尽可能长时间地保持杆在合理的垂直位置。

    ✅ 实际上，如果我们在 100 次连续试验中平均奖励达到 195，则认为 CartPole 模拟问题已解决。

## 状态离散化

在 Q-Learning 中，我们需要构建 Q-Table 来定义在每个状态下的行动。为了做到这一点，我们需要状态是**离散的**，更确切地说，它应该包含有限数量的离散值。因此，我们需要以某种方式**离散化**我们的观察值，将它们映射到有限的状态集合。

有几种方法可以做到这一点：

- **划分为区间**。如果我们知道某个值的范围，我们可以将这个范围划分为若干**区间**，然后用该值所属的区间编号替换原值。这可以使用 numpy 的 [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) 方法来完成。在这种情况下，我们将准确知道状态的大小，因为它将取决于我们为离散化选择的区间数量。

✅ 我们可以使用线性插值将值映射到某个有限区间（例如，从 -20 到 20），然后通过四舍五入将数字转换为整数。这种方法对状态大小的控制稍弱，特别是当我们不知道输入值的确切范围时。例如，在我们的例子中，观察值中的 4 个值中有 2 个没有上下界，这可能导致状态数量无限。

在我们的例子中，我们将采用第二种方法。正如你稍后可能注意到的，尽管没有明确的上下界，这些值很少会超出某些有限区间，因此具有极端值的状态将非常罕见。

1. 以下是一个函数，它将从模型中获取观察值并生成一个包含 4 个整数值的元组：（代码块 6）

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. 我们还可以探索另一种使用区间的离散化方法：（代码块 7）

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

1. 现在让我们运行一个短暂的模拟并观察这些离散化的环境值。可以尝试 `discretize` 和 `discretize_bins`，看看是否有区别。

    ✅ `discretize_bins` 返回区间编号，从 0 开始。因此，对于输入变量值接近 0 的情况，它返回区间中间的编号（10）。在 `discretize` 中，我们没有关心输出值的范围，允许它们为负，因此状态值没有偏移，0 对应于 0。（代码块 8）

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

    ✅ 如果你想查看环境如何执行，可以取消注释以 `env.render` 开头的行。否则，你可以在后台执行，这样速度更快。在我们的 Q-Learning 过程中，我们将使用这种“不可见”的执行方式。

## Q-Table 结构

在上一课中，状态是一个简单的数字对，从 0 到 8，因此用形状为 8x8x2 的 numpy 张量表示 Q-Table 很方便。如果我们使用区间离散化，状态向量的大小也是已知的，因此我们可以使用相同的方法，用形状为 20x20x10x10x2 的数组表示状态（这里的 2 是动作空间的维度，前几个维度对应于我们为观察空间中每个参数选择的区间数量）。

然而，有时观察空间的精确维度是未知的。在使用 `discretize` 函数的情况下，我们可能无法确定状态是否保持在某些限制范围内，因为某些原始值是没有界限的。因此，我们将使用稍微不同的方法，用字典表示 Q-Table。

1. 使用 *(state, action)* 对作为字典键，值对应于 Q-Table 的条目值。（代码块 9）

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    在这里我们还定义了一个函数 `qvalues()`，它返回给定状态对应于所有可能动作的 Q-Table 值列表。如果 Q-Table 中没有该条目，我们将返回默认值 0。

## 开始 Q-Learning

现在我们准备教彼得如何保持平衡了！

1. 首先，让我们设置一些超参数：（代码块 10）

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    这里，`alpha` 是**学习率**，定义了我们在每一步中应该在多大程度上调整 Q-Table 的当前值。在上一课中，我们从 1 开始，然后在训练过程中将 `alpha` 降低到较低的值。在这个例子中，为了简单起见，我们将保持它不变，你可以稍后尝试调整 `alpha` 值。

    `gamma` 是**折扣因子**，表示我们应该在多大程度上优先考虑未来奖励而不是当前奖励。

    `epsilon` 是**探索/利用因子**，决定我们是否应该更倾向于探索还是利用。在我们的算法中，我们将在 `epsilon` 百分比的情况下根据 Q-Table 值选择下一个动作，而在剩余情况下执行随机动作。这将允许我们探索以前从未见过的搜索空间区域。

    ✅ 在平衡方面——选择随机动作（探索）就像是一个随机的错误方向的推力，杆需要学习如何从这些“错误”中恢复平衡。

### 改进算法

我们还可以对上一课的算法进行两项改进：

- **计算平均累计奖励**，在多次模拟中进行。我们将每 5000 次迭代打印一次进度，并在这段时间内对累计奖励进行平均。这意味着如果我们获得超过 195 分——我们可以认为问题已经解决，质量甚至高于要求。

- **计算最大平均累计结果**，`Qmax`，并存储对应于该结果的 Q-Table。当你运行训练时，你会注意到有时平均累计结果开始下降，我们希望保留训练过程中观察到的最佳模型对应的 Q-Table 值。

1. 在每次模拟中将所有累计奖励收集到 `rewards` 向量中，以便进一步绘图。（代码块 11）

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

你可能从这些结果中注意到：

- **接近目标**。我们非常接近实现目标，即在 100 次以上的连续模拟中获得 195 的累计奖励，或者我们实际上已经实现了！即使我们获得较小的数字，我们仍然不知道，因为我们平均了 5000 次运行，而正式标准只需要 100 次运行。

- **奖励开始下降**。有时奖励开始下降，这意味着我们可能会用使情况变得更糟的新值“破坏” Q-Table 中已经学习到的值。

如果我们绘制训练进度，这种观察会更加清晰。

## 绘制训练进度

在训练过程中，我们将每次迭代的累计奖励值收集到 `rewards` 向量中。以下是将其与迭代次数绘制在一起的样子：

```python
plt.plot(rewards)
```

![原始进度](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

从这个图表中无法看出任何信息，因为由于随机训练过程的性质，训练会话的长度变化很大。为了让这个图表更有意义，我们可以计算一系列实验的**运行平均值**，比如 100 次。这可以使用 `np.convolve` 方便地完成：（代码块 12）

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![训练进度](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## 调整超参数

为了使学习更加稳定，有必要在训练过程中调整一些超参数。特别是：

- **学习率** `alpha`，我们可以从接近 1 的值开始，然后逐渐降低该参数。随着时间的推移，我们将在 Q-Table 中获得良好的概率值，因此我们应该稍微调整它们，而不是完全用新值覆盖。

- **增加 epsilon**。我们可能希望慢慢增加 `epsilon`，以便减少探索，更多地利用。可能合理的是从较低的 `epsilon` 值开始，然后逐渐增加到接近 1。
> **任务 1**：尝试调整超参数的值，看看是否能获得更高的累计奖励。你的得分是否超过了195？
> **任务 2**：为了正式解决这个问题，你需要在连续100次运行中获得195的平均奖励。在训练过程中进行测量，并确保你已经正式解决了这个问题！

## 查看结果的实际表现

观察训练好的模型如何表现会非常有趣。让我们运行模拟，并遵循与训练时相同的动作选择策略，根据Q表中的概率分布进行采样：（代码块13）

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

你应该会看到类似这样的画面：

![一个保持平衡的Cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀挑战

> **任务 3**：在这里，我们使用的是Q表的最终版本，但它可能不是表现最好的版本。记住，我们已经将表现最好的Q表存储在变量`Qbest`中！尝试用表现最好的Q表替换当前的Q表，看看是否能观察到差异。

> **任务 4**：在这里，我们并没有在每一步选择最佳动作，而是根据对应的概率分布进行采样。是否总是选择具有最高Q表值的最佳动作会更合理？这可以通过使用`np.argmax`函数找到对应于最高Q表值的动作编号来实现。尝试实施这种策略，看看是否能改善平衡效果。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 作业
[训练一个山地车](assignment.md)

## 总结

我们现在已经学会了如何通过提供一个定义游戏目标状态的奖励函数，并让智能体有机会智能地探索搜索空间，来训练智能体以获得良好的结果。我们成功地在离散和连续环境中应用了Q学习算法，但动作是离散的。

研究动作状态也是连续的情况，以及观察空间更复杂的情况（例如来自Atari游戏屏幕的图像）也很重要。在这些问题中，我们通常需要使用更强大的机器学习技术，例如神经网络，以获得良好的结果。这些更高级的主题将是我们即将推出的高级AI课程的内容。

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。因使用本翻译而导致的任何误解或误读，我们概不负责。