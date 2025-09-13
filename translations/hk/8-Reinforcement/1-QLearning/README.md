<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T09:33:22+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "hk"
}
-->
# 簡介：強化學習與 Q-Learning

![機器學習中強化學習的摘要示意圖](../../../../sketchnotes/ml-reinforcement.png)
> 示意圖由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 提供

強化學習涉及三個重要概念：代理（agent）、一些狀態（states）以及每個狀態的一組行動（actions）。通過在指定狀態下執行某個行動，代理會獲得一個獎勵。想像一下電腦遊戲《超級瑪利歐》。你是瑪利歐，處於遊戲關卡中，站在懸崖邊上。你的上方有一枚硬幣。你作為瑪利歐，處於遊戲關卡中的特定位置……這就是你的狀態。向右移動一步（行動）會讓你掉下懸崖，這會給你一個低的數值分數。然而，按下跳躍按鈕可以讓你得分並保持存活。這是一個正面的結果，應該給你一個正的數值分數。

通過使用強化學習和模擬器（遊戲），你可以學習如何玩遊戲以最大化獎勵，即保持存活並盡可能多地得分。

[![強化學習簡介](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 點擊上方圖片觀看 Dmitry 討論強化學習

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

## 前置條件與設置

在本課程中，我們將使用 Python 實驗一些代碼。你應該能夠在你的電腦或雲端環境中運行本課程的 Jupyter Notebook 代碼。

你可以打開[課程筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb)，並按照課程步驟進行。

> **注意：** 如果你從雲端打開此代碼，你還需要下載 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) 文件，該文件在筆記本代碼中使用。將其添加到與筆記本相同的目錄中。

## 簡介

在本課程中，我們將探索 **[彼得與狼](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)** 的世界，靈感來自俄羅斯作曲家 [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) 的音樂童話。我們將使用 **強化學習** 讓彼得探索他的環境，收集美味的蘋果並避免遇到狼。

**強化學習**（RL）是一種學習技術，通過多次實驗讓我們學習代理在某個**環境**中的最佳行為。代理在這個環境中應該有某些**目標**，由**獎勵函數**定義。

## 環境

為了簡化，我們將彼得的世界視為一個大小為 `width` x `height` 的方形棋盤，如下所示：

![彼得的環境](../../../../8-Reinforcement/1-QLearning/images/environment.png)

棋盤中的每個格子可以是：

* **地面**，彼得和其他生物可以在上面行走。
* **水域**，顯然不能行走。
* **樹木**或**草地**，可以休息的地方。
* **蘋果**，彼得希望找到以餵飽自己。
* **狼**，危險且應該避免。

有一個單獨的 Python 模塊 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py)，包含與此環境交互的代碼。由於這些代碼對理解概念並不重要，我們將導入該模塊並使用它創建示例棋盤（代碼塊 1）：

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

此代碼應該打印出類似上圖的環境。

## 行動與策略

在我們的例子中，彼得的目標是找到蘋果，同時避免狼和其他障礙物。為此，他可以在棋盤上四處走動，直到找到蘋果。

因此，在任何位置，他可以選擇以下行動之一：向上、向下、向左和向右。

我們將這些行動定義為一個字典，並將它們映射到相應的坐標變化。例如，向右移動（`R`）對應於坐標對 `(1,0)`。（代碼塊 2）：

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

總結一下，此場景的策略和目標如下：

- **策略**：代理（彼得）的策略由所謂的**策略函數**（policy）定義。策略函數在任何給定狀態下返回行動。在我們的例子中，問題的狀態由棋盤表示，包括玩家的當前位置。

- **目標**：強化學習的目標是最終學習一個良好的策略，能夠高效地解決問題。然而，作為基準，我們先考慮最簡單的策略，稱為**隨機行走**。

## 隨機行走

首先，我們通過實現隨機行走策略來解決問題。在隨機行走中，我們將從允許的行動中隨機選擇下一步，直到到達蘋果（代碼塊 3）。

1. 使用以下代碼實現隨機行走：

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

    `walk` 函數的調用應返回相應路徑的長度，該長度可能因每次運行而異。

1. 多次運行行走實驗（例如，100 次），並打印結果統計數據（代碼塊 4）：

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

    注意，路徑的平均長度約為 30-40 步，這相當多，考慮到到最近蘋果的平均距離約為 5-6 步。

    你還可以看到彼得在隨機行走中的移動情況：

    ![彼得的隨機行走](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## 獎勵函數

為了讓我們的策略更智能，我們需要了解哪些移動比其他移動“更好”。為此，我們需要定義我們的目標。

目標可以通過**獎勵函數**來定義，該函數會為每個狀態返回一些分數值。數值越高，獎勵函數越好。（代碼塊 5）

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

獎勵函數的一個有趣之處在於，大多數情況下，*只有在遊戲結束時才會給予實質性獎勵*。這意味著我們的算法應該以某種方式記住導致正面獎勵的“好”步驟，並增加它們的重要性。同樣，所有導致不良結果的移動應該被抑制。

## Q-Learning

我們將討論的算法稱為 **Q-Learning**。在此算法中，策略由一個函數（或數據結構）定義，稱為 **Q-Table**。它記錄了在給定狀態下每個行動的“好壞程度”。

之所以稱為 Q-Table，是因為將其表示為表格或多維數組通常很方便。由於我們的棋盤尺寸為 `width` x `height`，我們可以使用形狀為 `width` x `height` x `len(actions)` 的 numpy 數組來表示 Q-Table：（代碼塊 6）

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

注意，我們將 Q-Table 的所有值初始化為相等值，在我們的例子中為 0.25。這對應於“隨機行走”策略，因為每個狀態中的所有移動都是同樣好的。我們可以將 Q-Table 傳遞給 `plot` 函數，以便在棋盤上可視化該表：`m.plot(Q)`。

![彼得的環境](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

在每個格子的中心有一個“箭頭”，指示移動的首選方向。由於所有方向都是相等的，因此顯示為一個點。

現在我們需要運行模擬，探索環境，並學習 Q-Table 值的更好分佈，這將使我們能夠更快地找到蘋果的路徑。

## Q-Learning 的核心：貝爾曼方程

一旦我們開始移動，每個行動都會有相應的獎勵，即我們理論上可以根據最高的即時獎勵選擇下一個行動。然而，在大多數狀態下，移動並不能實現我們到達蘋果的目標，因此我們無法立即決定哪個方向更好。

> 記住，重要的不是即時結果，而是最終結果，即我們在模擬結束時獲得的結果。

為了考慮這種延遲獎勵，我們需要使用 **[動態規劃](https://en.wikipedia.org/wiki/Dynamic_programming)** 的原則，這使我們能夠以遞歸方式思考問題。

假設我們現在處於狀態 *s*，並希望移動到下一個狀態 *s'*。通過這樣做，我們將獲得由獎勵函數定義的即時獎勵 *r(s,a)*，加上一些未來的獎勵。如果我們假設 Q-Table 正確反映了每個行動的“吸引力”，那麼在狀態 *s'* 我們將選擇對應於 *Q(s',a')* 最大值的行動 *a*。因此，我們在狀態 *s* 能夠獲得的最佳未來獎勵將由 `max` 定義。

學習過程可以總結如下：

- **平均路徑長度增加**。我們觀察到，起初平均路徑長度會增加。這可能是因為當我們對環境一無所知時，很容易陷入不良狀態，例如水或狼。隨著我們學到更多並開始利用這些知識，我們可以探索更久的環境，但仍然不太清楚蘋果的位置。

- **隨著學習的深入，路徑長度減少**。當我們學到足夠多時，代理更容易達成目標，路徑長度開始減少。然而，我們仍然保持探索的開放性，因此經常偏離最佳路徑，探索新的選項，導致路徑比最佳路徑更長。

- **路徑長度突然增加**。我們在圖表中還觀察到某些時候路徑長度突然增加。這表明過程的隨機性，並且我們可能在某些時候通過覆寫新的值來“破壞”Q-Table的係數。理想情況下，這應該通過降低學習率來最小化（例如，在訓練的後期，我們僅以小幅度調整Q-Table的值）。

總體而言，重要的是要記住，學習過程的成功和質量在很大程度上取決於參數，例如學習率、學習率衰減和折扣因子。這些通常被稱為**超參數**，以區分於**參數**，後者是在訓練過程中優化的（例如Q-Table的係數）。尋找最佳超參數值的過程被稱為**超參數優化**，這是一個值得單獨討論的主題。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 作業 
[更真實的世界](assignment.md)

---

**免責聲明**：  
本文件已使用人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解讀概不負責。