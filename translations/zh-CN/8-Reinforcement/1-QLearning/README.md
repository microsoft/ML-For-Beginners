# 强化学习与 Q 学习简介

![机器学习中强化学习的概要手绘笔记](../../../../translated_images/zh-CN/ml-reinforcement.94024374d63348db.webp)
> 手绘笔记作者：[Tomomi Imura](https://www.twitter.com/girlie_mac)

强化学习包含三个重要概念：智能体(agent)、一些状态(states)以及每个状态下的一组动作(actions)。通过在指定状态下执行动作，智能体会获得一个奖励。再想象一下电脑游戏超级马里奥。你是马里奥，处在游戏关卡中，站在悬崖边上。你的上方有一枚金币。你作为马里奥，在游戏关卡中的一个具体位置……这就是你的状态。向右迈一步（动作）会让你跌落悬崖，这会给你较低的分数。然而，按跳跃键会让你得分并保持存活。这是一个积极的结果，应当奖励你一个正分。

通过使用强化学习和模拟器（游戏），你能学习如何玩游戏以最大化奖励，即保持存活并尽可能多地得分。

[![强化学习简介](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 点击上方图片，听 Dmitry 讲述强化学习

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

## 前置条件与环境搭建

在本课中，我们将用 Python 进行一些代码实验。你需要能够运行本课的 Jupyter Notebook 代码，无论是在你的电脑还是云端。

你可以打开[本课笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb)并跟随课程内容进行构建。

> **注意：** 如果你在云端打开此代码，你还需要获取笔记本代码中用到的 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) 文件，并将它放在与笔记本相同的目录。

## 介绍

本课中，我们将探索<strong>[彼得和狼](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)</strong>的世界，灵感来源于俄罗斯作曲家[谢尔盖·普罗科菲耶夫](https://en.wikipedia.org/wiki/Sergei_Prokofiev)的音乐童话。我们将用<strong>强化学习</strong>来让彼得探索他的环境，收集美味的苹果并避免遇见狼。

<strong>强化学习</strong>（RL）是一种学习技巧，通过多次实验让我们能学习智能体在某个环境下的最优行为。智能体在该环境中应有某个<strong>目标</strong>，由<strong>奖励函数</strong>定义。

## 环境

简化起见，我们将彼得的世界视为一个 `宽度` x `高度` 的方形棋盘，如下：

![彼得的环境](../../../../translated_images/zh-CN/environment.40ba3cb66256c93f.webp)

棋盘中的每个格子可能是：

* <strong>地面</strong>，彼得和其他生物可以在其上行走。
* <strong>水域</strong>，显然无法行走。
* <strong>树木</strong>或<strong>草地</strong>，可以休息的地方。
* <strong>苹果</strong>，代表彼得高兴找到用以喂养自己的东西。
* <strong>狼</strong>，危险且应躲避。

有一个单独的 Python 模块 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py)，包含与该环境互动的代码。由于这些代码对理解概念不重要，我们将导入该模块并用它创建示例棋盘（代码块1）：

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

这段代码应打印出与上图类似的环境画面。

## 动作与策略

在我们的示例中，彼得的目标是找到一个苹果，同时避开狼和其他障碍物。为此，他可以四处走动直到找到苹果。

因此，在任何位置，他可以选择以下动作之一：上、下、左、右。

我们将把这些动作定义为字典，并映射到对应的坐标变化对。例如向右移动 (`R`) 对应坐标变化对 `(1,0)`。（代码块2）：

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

总结来说，此场景的策略和目标如下：

- <strong>策略</strong>，由所谓的<strong>策略函数</strong>（policy）定义。策略是一个函数，给定任意状态返回采取的动作。在本例中，状态由棋盘表示，含玩家当前位置。

- <strong>目标</strong>，强化学习的目的是最终学习一个良好策略，能高效解决问题。但作为基线，先考虑最简单的策略——<strong>随机游走</strong>。

## 随机游走

我们先用随机游走策略解决问题。随机游走时，我们从允许的动作中随机选择下一步动作，直到到达苹果（代码块3）。

1. 用下面代码实现随机游走：

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # 步数
        # 设置初始位置
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # 成功！
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # 被狼吃掉或淹死
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # 执行实际移动
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    调用 `walk` 应返回对应路径长度，运行结果每次可能不同。 

1. 运行多次游走实验（例如100次），并打印统计结果（代码块4）：

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

    注意路径平均长度约为 30-40 步，这很大，考虑到最邻近苹果的平均距离仅 5-6 步。

    你也可以看到彼得随机游走时的移动轨迹：

    ![彼得的随机游走](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## 奖励函数

为了让策略更加智能，我们需要理解哪些移动比其他的“更好”。为此，我们需要定义目标。

目标可以通过<strong>奖励函数</strong>定义，该函数为每个状态返回分值。分数越高，奖励函数越好。（代码块5）

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

一个有趣的事实是，在大多数情况下，<em>我们只在游戏结束时得到实质性的奖励</em>。这意味着算法应记住通往最终正向奖励的“好”步骤，并提高它们的重要性。同理，导致糟糕结果的所有动作应被抑制。

## Q 学习

本文将讨论一种算法称为<strong>Q 学习</strong>。在此算法中，策略由称为<strong>Q 表</strong>的函数（或数据结构）定义。它记录给定状态下各动作的“价值”。

因为方便起见，它通常被表示为表格或多维数组。由于棋盘尺寸为 `宽度` x `高度`，我们可以用形状为 `宽度` x `高度` x `动作数` 的 numpy 数组表示 Q 表：（代码块6）

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

注意我们用相等的值初始化 Q 表的所有项，在本例中为 0.25。这对应随机游走策略，因为每个状态的所有动作值相等。我们可以将 Q 表传给 `plot` 函数，在棋盘上可视化表格：`m.plot(Q)`。

![彼得的环境](../../../../translated_images/zh-CN/env_init.04e8f26d2d60089e.webp)

每个格子中心有个“箭头”指示优选移动方向。所有方向等值时显示为点。

现在我们需要运行模拟，探索环境，学习更好的 Q 表数值分布，从而更快找到通往苹果的路径。

## Q 学习的本质：贝尔曼方程

开始移动时，每个动作会对应一个奖励，即理论上我们可根据即时奖励最高选择下一步动作。然而，大多数状态下，动作不会立即达成目标（吃到苹果），因此无法马上决定哪个方向更优。

> 记住，重要的不是即时结果，而是最终结果，即模拟结束时获得的结果。

为了考虑这种延后奖励，我们需要利用<strong>[动态规划](https://en.wikipedia.org/wiki/Dynamic_programming)</strong>原理，递归思考问题。

假设我们现在在状态 *s*，想移动到下一状态 *s'*。此时，会获得即时奖励 *r(s,a)*（由奖励函数定义），加上未来可能获得的一些奖励。如果假设我们的 Q 表正确反映动作“吸引力”，在状态 *s'* 会选动作 *a*，对应最大值 *Q(s',a')*。因此，在状态 *s*，能获得的最大未来奖励定义为 `max`<sub>a'</sub>*Q(s',a')*（最大值在状态 *s'* 的所有动作 *a'* 中计算）。

这给出了状态 *s* 下动作 *a* 的 Q 表计算的<strong>贝尔曼公式</strong>：

<img src="../../../../translated_images/zh-CN/bellman-equation.7c0c4c722e5a6b7c.webp"/>

其中 γ 是所谓的<strong>折扣因子</strong>，决定你应该多大程度上倾向当前奖励而非未来奖励，反之亦然。

## 学习算法

根据上面公式，我们现在可以写出我们的学习算法的伪代码：

* 用相等数字初始化 Q 表 Q，覆盖所有状态和动作
* 设定学习率 α ← 1
* 多次重复模拟
   1. 随机选择起始位置
   1. 重复
        1. 在状态 *s* 选择动作 *a*
        2. 执行动作，移动到新状态 *s'*
        3. 若遇到游戏结束条件或总奖励过低，退出模拟  
        4. 计算新状态下奖励 *r*
        5. 根据贝尔曼方程更新 Q 函数：*Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. 更新总奖励并减小 α。

## 利用与探索

在上面算法中，我们没具体说明如何选择步骤 2.1 的动作。如果随机选择动作，我们会随机<strong>探索</strong>环境，可能频繁死亡且进入通常不会去的区域。另一种方法是<strong>利用</strong>已知 Q 表值，选择最佳动作（Q 值高的动作）在状态 *s*。但这会阻止探索新的状态，可能导致未能找到最优解。

因此，最佳方法是探索与利用间取得平衡。可通过根据 Q 表值的比例概率选择动作，在初期 Q 值相同，等同随机选择；随着学习深入，更倾向沿最优路径，同时偶尔允许智能体选择未探索路径。

## Python 实现

现在我们准备实现学习算法。之前还需准备一个函数，能将 Q 表中的任意数转换成对应动作的概率向量。

1. 创建函数 `probs()`：

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

    为避免初始向量所有分量相同时除以 0，向原始向量添加少量 `eps`。

运行学习算法共5000次实验，也称为<strong>epochs</strong>：（代码块8）
```python
    for epoch in range(5000):
    
        # 选择初始点
        m.random_start()
        
        # 开始移动
        n=0
        cum_reward = 0
        while True:
            x,y = m.human
            v = probs(Q[x,y])
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            m.move(dpos,check_correctness=False) # 我们允许玩家移动到棋盘外，这将终止回合
            r = reward(m)
            cum_reward += r
            if r==end_reward or cum_reward < -1000:
                lpath.append(n)
                break
            alpha = np.exp(-n / 10e5)
            gamma = 0.5
            ai = action_idx[a]
            Q[x,y,ai] = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
            n+=1
```

执行此算法后，Q 表应更新为定义不同动作吸引力的值。我们可以尝试通过在每个格子绘制一个指向期望移动方向的向量来可视化 Q 表。为简化起见，使用小圆代替箭头。

<img src="../../../../translated_images/zh-CN/learned.ed28bcd8484b5287.webp"/>

## 检查策略

因为 Q 表列出了每个状态下动作的“吸引力”，所以可以轻松用来定义有效的世界导航。最简单情况，可选择对应最大 Q 表值的动作：（代码块9）

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```


> 如果你多次尝试上面的代码，你可能会注意到有时它会“卡住”，你需要按笔记本中的停止按钮来中断它。之所以会发生这种情况，是因为可能存在两个状态在最优Q值方面“互相指向”，在这种情况下，智能体最终会在这些状态之间无限循环。

## 🚀挑战

> **任务1：** 修改 `walk` 函数，将路径的最大长度限制为一定步数（比如100），并观察上述代码时不时返回这个值。

> **任务2：** 修改 `walk` 函数，使其不返回之前已经去过的地方。这将防止 `walk` 出现循环，但智能体仍可能被困在无法逃脱的位置。

## 导航

更好的导航策略是我们训练时使用的策略，它结合了利用和探索。在此策略中，我们将以某种概率选择每个动作，该概率与Q表中的值成比例。这个策略仍可能导致智能体返回已探索的位置，但正如下面代码所示，它能使到目标位置的平均路径非常短（记住 `print_statistics` 运行了100次仿真）：(代码块 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

运行此代码后，你应得到比之前更小的平均路径长度，约在3到6之间。

## 探索学习过程

如前所述，学习过程是在探索和利用已获得关于问题空间结构知识之间的平衡。我们看到学习结果（帮助智能体找到短路径的能力）有所改善，同时观察平均路径长度在学习过程中如何变化也很有趣：

<img src="../../../../translated_images/zh-CN/lpathlen1.0534784add58d4eb.webp"/>

学习要点总结如下：

- <strong>平均路径长度增加</strong>。这里看到的是，开始时平均路径长度增加。这很可能是因为当我们对环境一无所知时，很可能陷入糟糕的状态，比如水域或狼附近。随着学习深入并开始利用这些知识，我们可以更长时间地探索环境，但仍不清楚苹果位置。

- <strong>路径长度随学习增多而减少</strong>。当我们学到足够多的知识后，智能体更容易达到目标，路径长度开始减少。然而，我们仍保持探索，因此经常会偏离最佳路径，探索新的选项，使路径比最优路径更长。

- <strong>路径长度突然增加</strong>。图中还显示，在某些时刻路径长度突然增加。这反映了过程的随机性，也表明Q表系数可能会被新值覆盖而“破坏”。理想情况下，应通过降低学习率（例如，训练后期只对Q表值进行小幅调整）来最小化这种情况。

总体而言，重要的是要记住，学习过程的成功与质量很大程度上依赖于学习率、学习率衰减和折扣因子等参数。这些通常被称为<strong>超参数</strong>，以区别于我们在训练过程中优化的<strong>参数</strong>（例如Q表系数）。寻找最优超参数的过程称为<strong>超参数优化</strong>，这是一个独立的话题。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 作业
[一个更现实的世界](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：
本文件由 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译完成。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版文件应视为权威来源。对于重要信息，建议使用专业人工翻译。我们对因使用本翻译而产生的任何误解或误释不承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->