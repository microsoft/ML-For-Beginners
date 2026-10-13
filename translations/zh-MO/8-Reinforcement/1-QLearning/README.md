# 強化學習與 Q-Learning 簡介

![機器學習中強化學習的概念速寫](../../../../translated_images/zh-MO/ml-reinforcement.94024374d63348db.webp)
> 速寫作者：[Tomomi Imura](https://www.twitter.com/girlie_mac)

強化學習涉及三個重要概念：代理人（agent）、一些狀態（states），以及每個狀態下的一組行動。透過在特定狀態執行一個行動，代理人會獲得一個獎勵。再次想像電腦遊戲《超級瑪利歐》。你是瑪利歐，你在一個遊戲關卡中，站在懸崖邊緣。上方有一個硬幣。你扮演的瑪利歐，在關卡內一個特定位置……這就是你的狀態。向右走一步（行動）會讓你掉下懸崖，這會給你一個低分數。然而，按下跳躍鍵會讓你獲得一分，並且你會繼續存活。這是一個正向的結果，應該給予你一個正的數值獎勵。

透過使用強化學習和模擬器（遊戲），你可以學習如何玩遊戲以最大化獎勵，也就是保持存活並盡可能多得分。

[![強化學習入門](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 點擊上方圖片收聽 Dmitry 討論強化學習

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 先決條件與設置

在本課程中，我們將用 Python 實驗一些代碼。你應該能在自己的電腦或雲端運行本課的 Jupyter Notebook 代碼。

你可以打開[課程筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb)並跟隨本課程建構。

> **注意：** 如果你打算從雲端打開這份代碼，也需要下載 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) 文件，此文件在筆記本代碼中會用到。將它放在與筆記本同一目錄下。

## 簡介

在本課，我們將探索<strong>[彼得與狼](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)</strong>的世界，靈感來自俄羅斯作曲家[謝爾蓋·普羅科菲耶夫](https://en.wikipedia.org/wiki/Sergei_Prokofiev)的音樂童話。我們將使用<strong>強化學習</strong>讓彼得探索環境，收集美味的蘋果，並避開狼。

<strong>強化學習</strong>（RL）是一種學習技術，允許我們通過大量實驗來學習代理人在某個環境中最佳行為。代理人在此環境中應有明確的<strong>目標</strong>，以<strong>獎勵函數</strong>表示。

## 環境

為簡化起見，我們假設彼得的世界是一個大小為 `寬度` x `高度` 的方形棋盤，如下所示：

![彼得的環境](../../../../translated_images/zh-MO/environment.40ba3cb66256c93f.webp)

棋盤中的每個格子可以是：

* <strong>地面</strong>，彼得和其他生物可以行走的地方。
* <strong>水域</strong>，你顯然不能走在水上。
* <strong>樹木</strong>或<strong>草地</strong>，可以休息的地方。
* <strong>蘋果</strong>，代表彼得會高興找到用來餵食自己的東西。
* <strong>狼</strong>，危險需避免。

Python 有一個獨立模組 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py)，包含用於此環境的代碼。因為這段代碼對理解本概念不重要，我們將導入模組並用它創建示例棋盤（程式碼塊 1）：

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

這段代碼會印出類似上圖的環境圖。

## 行動與策略

在例子中，彼得的目標是找到蘋果，並避開狼與其他障礙物。為此，他基本上可以四處走動直到找到蘋果。

因此，在任意位置，他可以選擇以下行動之一：上、下、左、右。

我們將定義這些行動為字典，並將它們映射到相應的座標變化對。例如，向右移動 (`R`) 對應於座標變化 `(1,0)`。（程式碼塊 2）：

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

總結來說，此場景的策略與目標如下：

- <strong>策略</strong>，代理人（彼得）的策略由所謂的<strong>策略函數（policy）</strong>定義。策略是一個函數，輸入任一狀態即返回該狀態下的行動。在此，我們的問題狀態由棋盤包括玩家當前位置表示。

- <strong>目標</strong>，強化學習的目標是最終學到一個優良策略，能有效解決問題。然而，作為基準，先考慮最簡單的策略稱為<strong>隨機漫步</strong>。

## 隨機漫步

首先，我們用實作隨機漫步策略解決問題。在隨機漫步中，我們將從允許的行動中隨機選擇下一步行動，直到找到蘋果（程式碼塊 3）。

1. 用以下代碼實作隨機漫步：

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # 步數
        # 設置初始位置
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # 成功！
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # 被狼吃掉或溺死
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # 進行實際移動
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    `walk` 函式的呼叫應返回相對應路徑長度，該長度可能每次運行都不同。 

1. 多次（例如 100 次）運行漫步實驗，並列印結果統計數據（程式碼塊 4）：

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

    注意平均路徑長度約為 30-40 步，考慮到最近蘋果平均距離約 5-6 步，這數量相當大。

    你還可以看到彼得隨機漫步的移動樣貌：

    ![彼得的隨機漫步](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## 獎勵函數

為了讓策略更聰明，我們需要了解哪些移動比其他「好」。為此，我們需要定義目標。

目標可用<strong>獎勵函數</strong>定義，該函數會對每個狀態回傳某分數值。數值越高，獎勵函數越好。（程式碼塊 5）

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

有趣的是，獎勵函數多數情況下，<em>我們只會在遊戲結束時得到顯著的獎勵</em>。這意味著我們的算法應該記憶導致結束時正面獎勵的「好」步驟，並加強它們的重要性。類似地，所有導致壞結果的移動應被削弱。

## Q-Learning

我們將介紹一個稱為 **Q-Learning** 的算法。在此算法中，策略由一個函數（或資料結構）稱為<strong>Q-表格</strong>定義。它記錄在給定狀態中各行動的「優劣」。

它稱為 Q-表，因為通常會以表格或多維陣列形式呈現。由於我們的棋盤是 `寬度` x `高度`，可用 numpy 陣列表示 Q-表，形狀為 `寬度` x `高度` x `len(actions)`：（程式碼塊 6）

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

注意我們以相同數值初始化 Q-表的所有值，在此為 0.25，對應「隨機漫步」策略，因為每狀態的行動同樣優劣。我們可以將 Q-表傳給 `plot` 函數，在棋盤中視覺化：`m.plot(Q)`。

![彼得的環境](../../../../translated_images/zh-MO/env_init.04e8f26d2d60089e.webp)

每格中心有一個「箭頭」指示偏好的移動方向。若所有方向相等，則顯示點。

現在我們要運行模擬，探索環境，學習較佳的 Q-表值分布，讓我們更快找到蘋果路徑。

## Q-Learning 的本質：Bellman 方程式

一旦開始移動，每個行動會有相應獎勵，即理論上可依最高即時獎勵選擇下一行動。但在多數狀態，行動無法馬上達成找蘋果目標，因此無法立即決定較佳方向。

> 請記住，不是即時結果重要，而是模擬結束時取得的最終結果。

為了考慮此延遲獎勵，我們需使用<strong>[動態規劃](https://en.wikipedia.org/wiki/Dynamic_programming)</strong>的原理，讓我們以遞迴思考問題。

假設我們處於狀態 *s*，想移動到下一狀態 *s'*；此時會得到即時獎勵 *r(s,a)* 為函數定義，再加上一些未來獎勵。若假設 Q-表正確反映每行動「吸引力」，則在狀態 *s'* 會選擇行動 *a*，使 *Q(s',a')* 達最大值。故狀態 *s* 可取得的最佳未來獎勵定義為 `max`<sub>a'</sub>*Q(s',a')*（此處最大化運算涵蓋狀態 *s'* 所有可能行動 *a'*）。

這給出在狀態 *s*、行動 *a* 下計算 Q-表值的 **Bellman 公式**：

<img src="../../../../translated_images/zh-MO/bellman-equation.7c0c4c722e5a6b7c.webp"/>

這裡的 γ 是所謂的<strong>折扣因子</strong>，決定你應多偏好當前獎勵還是未來獎勵。

## 學習算法

基於上述方程式，我們現在可以寫出學習算法的伪代碼：

* 用相同數值初始化 Q-表 Q，涵蓋所有狀態和行動
* 設定學習率 α ← 1
* 重複多次模擬
   1. 從隨機位置開始
   1. 重複
        1. 在狀態 *s* 選擇行動 *a*
        2. 執行行動，移動到新狀態 *s'*
        3. 若遇遊戲結束條件或總獎勵過小，退出模擬
        4. 計算新狀態的獎勵 *r*
        5. 根據 Bellman 方程更新 Q 函數：*Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. 更新總獎勵並降低 α。

## 利用（exploit）與探索（explore）

在上述算法中，我們未指定在 2.1 步驟如何選擇行動。若隨機選擇行動，我們會隨機<strong>探索</strong>環境，可能經常死亡且探索不常去的區域；另一方法是<strong>利用</strong>我們已知的 Q-表值，在狀態 *s* 選擇最佳行動（Q-表值高）。但此舉會阻礙探索其他狀態，可能無法找到最優解。

因此，最佳策略是在探索和利用間取得平衡。可根據 Q-表值按比例選擇狀態 *s* 下的行動。起初當 Q-表值均等時，行為視同隨機選擇；隨著我們對環境了解加深，代理人較可能沿最優路徑前進，且偶爾嘗試未探索路徑。

## Python 實作

現在我們準備實作學習算法。在此之前，我們還需要一個將 Q-表中任意數值轉換為相應行動概率向量的函數。

1. 創建 `probs()` 函數：

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

    我們在原始向量中加一些 `eps`，以避免初始所有分量相同時除以 0。

通過 5000 次實驗（稱為 **epoch**）執行學習算法：（程式碼塊 8）
```python
    for epoch in range(5000):
    
        # 選擇初始點
        m.random_start()
        
        # 開始移動
        n=0
        cum_reward = 0
        while True:
            x,y = m.human
            v = probs(Q[x,y])
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            m.move(dpos,check_correctness=False) # 我哋容許玩家移出棋盤，會終止回合
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

執行此算法後，Q-表將更新為定義各步驟不同行動吸引力的值。我們可以嘗試視覺化 Q-表，在每格畫出指向期望移動方向的向量。為簡便起見，改以小圓圈代替箭頭頭部。

<img src="../../../../translated_images/zh-MO/learned.ed28bcd8484b5287.webp"/>

## 檢查策略

由於 Q-表列出每狀態下行動的「吸引力」，我們很容易用它定義有效的導航。在最簡單情況下，我們可以選擇對應最高 Q-表值的行動：（程式碼塊 9）

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```


> 如果你多次嘗試上述代碼，你可能會注意到有時它會「卡住」，你需要按下筆記本中的停止按鈕來中斷它。這是因為可能存在兩個狀態在最佳 Q 值方面「互相指向」的情況，在這種情況下，代理會無限地在這些狀態之間移動。

## 🚀挑戰

> **任務 1：** 修改 `walk` 函數，使路徑的最大長度限制為一定步數（例如 100 步），並觀察上述代碼時不時返回該值。

> **任務 2：** 修改 `walk` 函數，使其不返回之前已經去過的地方。這將防止 `walk` 進入循環，但代理仍可能被困在一個無法逃脫的位置。

## 導航

一個更好的導航策略是我們在訓練時使用的，結合了利用和探索。在這個策略中，我們會以一定的概率選擇每個動作，概率與 Q 表中的值成比例。這種策略仍可能導致代理返回之前已經探索過的位置，但正如你從下面代碼中看到的，它能使到達目標位置的平均路徑非常短（記住 `print_statistics` 會運行模擬 100 次）：(代碼塊 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

執行此代碼後，你應該會得到比之前小得多的平均路徑長度，約在 3-6 步之間。

## 探討學習過程

如我們所述，學習過程是在問題空間結構的探索與利用間的平衡。我們已見到學習結果（幫助代理找到目標短路徑的能力）有所提升，但觀察平均路徑長度在學習過程中的變化同樣有趣：

<img src="../../../../translated_images/zh-MO/lpathlen1.0534784add58d4eb.webp"/>

學習總結如下：

- <strong>平均路徑長度增加</strong>。我們看到一開始平均路徑長度增加。這可能是因為當我們對環境一無所知時，容易陷入不良狀態，如水域或狼群。隨著學習更多並開始利用這些知識，我們能更長時間探索環境，但對蘋果的位置還不夠清楚。

- **隨著學習增多，路徑長度下降**。一旦學習足夠，代理達成目標變得更容易，路徑長逐漸縮短。然而，我們仍保持探索，因此常偏離最佳路徑，嘗試新選項，導致路徑比理想的更長。

- <strong>路徑長度突然增加</strong>。圖中也可觀察到某些時候路徑長度突然增加，這顯示出過程的隨機特性，某些時刻我們可能會透過新值覆寫而「破壞」Q 表係數。理想中應透過降低學習率（如訓練後期僅修正 Q 表的值很小）來減少這種情況。

總的來說，重要的是記住學習過程的成功與質量嚴重依賴於參數，比如學習率、學習率衰減和折扣因子。這些通常稱為 <strong>超參數</strong>，以區別於我們在訓練中優化的 <strong>參數</strong>（例如 Q 表係數）。尋找最佳超參數值的過程稱為 <strong>超參數優化</strong>，是個值得獨立討論的主題。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 作業
[一個更現實的世界](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->