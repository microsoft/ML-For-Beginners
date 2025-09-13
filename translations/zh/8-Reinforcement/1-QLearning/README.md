<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T09:09:02+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "zh"
}
-->
# 强化学习与Q学习简介

![机器学习中强化学习的总结图](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

强化学习涉及三个重要概念：代理、状态和每个状态的一组动作。通过在指定状态下执行一个动作，代理会获得奖励。想象一下电脑游戏《超级马里奥》。你是马里奥，处于一个游戏关卡中，站在悬崖边上。你的上方有一个金币。你作为马里奥，处于游戏关卡中的特定位置……这就是你的状态。向右移动一步（一个动作）会让你掉下悬崖，这会给你一个较低的数值分数。然而，按下跳跃按钮会让你得分并保持存活。这是一个积极的结果，应该奖励你一个正数分数。

通过使用强化学习和模拟器（游戏），你可以学习如何玩游戏以最大化奖励，即保持存活并尽可能多地得分。

[![强化学习简介](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 点击上方图片观看 Dmitry 讨论强化学习

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

## 前提条件与设置

在本课中，我们将用 Python 实验一些代码。你应该能够在你的电脑或云端运行本课的 Jupyter Notebook 代码。

你可以打开[课程笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb)，并按照课程内容进行学习。

> **注意：** 如果你从云端打开代码，还需要获取 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) 文件，该文件在笔记本代码中使用。将其添加到与笔记本相同的目录中。

## 简介

在本课中，我们将探索**《彼得与狼》**的世界，这个故事灵感来源于俄罗斯作曲家[谢尔盖·普罗科菲耶夫](https://en.wikipedia.org/wiki/Sergei_Prokofiev)创作的音乐童话。我们将使用**强化学习**让彼得探索他的环境，收集美味的苹果并避免遇到狼。

**强化学习**（RL）是一种学习技术，它通过运行许多实验让我们学习代理在某个**环境**中的最佳行为。代理在这个环境中应该有某种**目标**，由**奖励函数**定义。

## 环境

为了简化，我们将彼得的世界设定为一个大小为 `width` x `height` 的方形棋盘，如下所示：

![彼得的环境](../../../../8-Reinforcement/1-QLearning/images/environment.png)

棋盘中的每个单元格可以是：

* **地面**，彼得和其他生物可以在上面行走。
* **水域**，显然无法在上面行走。
* **树**或**草地**，可以休息的地方。
* **苹果**，彼得很高兴找到的食物。
* **狼**，危险的生物，应避免接触。

有一个单独的 Python 模块 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py)，包含了与这个环境交互的代码。由于这些代码对理解我们的概念并不重要，我们将导入模块并使用它创建示例棋盘（代码块 1）：

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

这段代码应该打印出类似上图的环境。

## 动作与策略

在我们的示例中，彼得的目标是找到苹果，同时避免狼和其他障碍物。为此，他可以在棋盘上四处走动，直到找到苹果。

因此，在任何位置，他可以选择以下动作之一：向上、向下、向左和向右。

我们将这些动作定义为一个字典，并将它们映射到对应的坐标变化。例如，向右移动（`R`）对应于坐标对 `(1,0)`。（代码块 2）：

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

总结一下，这个场景的策略和目标如下：

- **策略**：我们的代理（彼得）的策略由所谓的**策略函数**定义。策略函数在任何给定状态下返回动作。在我们的例子中，问题的状态由棋盘表示，包括玩家的当前位置。

- **目标**：强化学习的目标是最终学习一个好的策略，使我们能够高效地解决问题。然而，作为基线，我们可以考虑最简单的策略，称为**随机游走**。

## 随机游走

首先，我们通过实现随机游走策略来解决问题。在随机游走中，我们会随机选择允许的动作，直到到达苹果（代码块 3）。

1. 使用以下代码实现随机游走：

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    调用 `walk` 应返回对应路径的长度，该长度可能因运行而异。

1. 多次运行游走实验（例如，100 次），并打印结果统计数据（代码块 4）：

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    注意，路径的平均长度约为 30-40 步，这相当多，考虑到到最近苹果的平均距离约为 5-6 步。

    你还可以看到彼得在随机游走中的移动情况：

    ![彼得的随机游走](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## 奖励函数

为了让我们的策略更智能，我们需要了解哪些动作比其他动作“更好”。为此，我们需要定义目标。

目标可以通过**奖励函数**定义，该函数为每个状态返回一些分数值。分数越高，奖励函数越好。（代码块 5）

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

奖励函数的一个有趣之处在于，大多数情况下，*我们只有在游戏结束时才会获得实质性奖励*。这意味着我们的算法应该以某种方式记住导致最终正奖励的“好”步骤，并增加它们的重要性。同样，所有导致不良结果的动作应该被抑制。

## Q学习

我们将讨论的算法称为**Q学习**。在这个算法中，策略由一个称为**Q表**的函数（或数据结构）定义。它记录了在给定状态下每个动作的“好坏程度”。

之所以称为 Q表，是因为将其表示为表格或多维数组通常很方便。由于我们的棋盘维度为 `width` x `height`，我们可以使用形状为 `width` x `height` x `len(actions)` 的 numpy 数组来表示 Q表：（代码块 6）

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

注意，我们将 Q表的所有值初始化为相等值，在我们的例子中为 0.25。这对应于“随机游走”策略，因为每个状态中的所有动作都同样好。我们可以将 Q表传递给 `plot` 函数，以便在棋盘上可视化表格：`m.plot(Q)`。

![彼得的环境](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

每个单元格的中心有一个“箭头”，指示移动的优选方向。由于所有方向都相等，显示的是一个点。

现在我们需要运行模拟，探索环境，并学习 Q表值的更好分布，这将使我们更快找到苹果的路径。

## Q学习的核心：贝尔曼方程

一旦我们开始移动，每个动作都会有相应的奖励，即我们理论上可以根据最高的即时奖励选择下一个动作。然而，在大多数状态下，动作不会立即实现我们到达苹果的目标，因此我们无法立即决定哪个方向更好。

> 请记住，重要的不是即时结果，而是最终结果，即我们将在模拟结束时获得的结果。

为了考虑这种延迟奖励，我们需要使用**[动态规划](https://en.wikipedia.org/wiki/Dynamic_programming)**的原理，这使我们能够递归地思考问题。

假设我们现在处于状态 *s*，并希望移动到下一个状态 *s'*。通过这样做，我们将获得即时奖励 *r(s,a)*，由奖励函数定义，加上某些未来奖励。如果我们假设我们的 Q表正确反映了每个动作的“吸引力”，那么在状态 *s'* 我们将选择一个动作 *a'*，其对应的值为 *Q(s',a')* 的最大值。因此，我们在状态 *s* 能够获得的最佳未来奖励将定义为 `max`

## 检查策略

由于 Q-Table 列出了每个状态下每个动作的“吸引力”，因此使用它来定义我们世界中的高效导航非常简单。在最简单的情况下，我们可以选择对应于最高 Q-Table 值的动作：（代码块 9）

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> 如果多次尝试上面的代码，你可能会注意到有时它会“卡住”，需要按下笔记本中的 STOP 按钮来中断。这是因为可能存在两种状态在最佳 Q 值方面“指向”彼此的情况，这样代理就会在这些状态之间无限移动。

## 🚀挑战

> **任务 1：** 修改 `walk` 函数以限制路径的最大长度为一定步数（例如 100），并观察上面的代码是否会不时返回该值。

> **任务 2：** 修改 `walk` 函数，使其不返回到之前已经到过的地方。这将防止 `walk` 进入循环，但代理仍可能最终被“困”在无法逃脱的位置。

## 导航

更好的导航策略是我们在训练期间使用的策略，它结合了利用和探索。在此策略中，我们将以一定的概率选择每个动作，该概率与 Q-Table 中的值成比例。此策略可能仍会导致代理返回到已经探索过的位置，但正如你从下面的代码中看到的，它会导致到达目标位置的平均路径非常短（记住 `print_statistics` 会运行 100 次模拟）：（代码块 10）

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

运行此代码后，你应该会得到比之前小得多的平均路径长度，范围在 3-6 之间。

## 调查学习过程

正如我们提到的，学习过程是在探索和利用已获得的关于问题空间结构的知识之间的平衡。我们已经看到学习的结果（帮助代理找到到达目标的短路径的能力）有所改善，但观察平均路径长度在学习过程中的变化也很有趣：

学习总结如下：

- **平均路径长度增加**。我们看到的是，起初平均路径长度增加。这可能是因为当我们对环境一无所知时，很容易陷入糟糕的状态，比如水或狼。随着我们学习更多并开始使用这些知识，我们可以更长时间地探索环境，但仍然不太清楚苹果的位置。

- **路径长度随着学习增加而减少**。一旦我们学到足够多，代理更容易实现目标，路径长度开始减少。然而，我们仍然开放探索，因此经常偏离最佳路径，探索新的选项，使路径比最优路径更长。

- **长度突然增加**。我们在图表上还观察到某些时候长度突然增加。这表明过程的随机性，并且我们可能会在某些时候通过用新值覆盖 Q-Table 系数来“破坏”它们。这应该通过降低学习率来尽量减少（例如，在训练结束时，我们仅通过小值调整 Q-Table 值）。

总体而言，重要的是要记住，学习过程的成功和质量在很大程度上取决于参数，例如学习率、学习率衰减和折扣因子。这些通常被称为 **超参数**，以区别于 **参数**，后者是在训练期间优化的（例如 Q-Table 系数）。寻找最佳超参数值的过程称为 **超参数优化**，它值得单独讨论。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 作业 
[一个更真实的世界](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于重要信息，建议使用专业人工翻译。我们不对因使用此翻译而产生的任何误解或误读承担责任。