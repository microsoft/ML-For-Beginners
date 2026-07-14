# 強化學習與 Q 學習簡介

![機器學習中強化學習的摘要示意圖](../../../../translated_images/zh-TW/ml-reinforcement.94024374d63348db.webp)
> 示意圖作者為 [Tomomi Imura](https://www.twitter.com/girlie_mac)

強化學習包含三個重要概念：代理人(agent)、一些狀態(states)，以及每個狀態下的一組動作(actions)。透過在指定狀態執行一個動作，代理人會獲得回饋(reward)。再次想像電腦遊戲超級瑪利歐(Super Mario)。你是瑪利歐，身處一個遊戲關卡，站在懸崖邊。你的上方有一個金幣。你身為瑪利歐，處於遊戲關卡中一個特定位置…這就是你的狀態(state)。向右走一步(一個動作)會讓你掉下懸崖，導致低分。但跳躍按鈕則會幫你得分且保持存活。這是一個正向結果，應該給予正向數值的分數。

利用強化學習與模擬器（遊戲），你可以學會怎麼玩遊戲以最大化獎勵，也就是保持存活並取得盡可能多的分數。

[![強化學習入門](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 點擊上方圖片聽 Dmitry 討論強化學習

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 前置條件與設定

在本課程中，我們會用 Python 實作一些程式碼。你應該能在你的電腦或雲端環境執行本課的 Jupyter Notebook 程式碼。

你可以打開[課程筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb)並跟著本課學習。

> **注意：** 如果你從雲端開啟這段程式碼，你還需下載 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) 檔案，該檔案被筆記本程式碼使用。請將它放在與筆記本相同的目錄下。

## 介紹

本課將探索 **[彼得與狼](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)** 的世界，靈感來自俄羅斯作曲家 [謝爾蓋·普羅科菲耶夫](https://en.wikipedia.org/wiki/Sergei_Prokofiev) 的音樂童話。我們將用 <strong>強化學習</strong> 讓彼得探索環境，收集美味的蘋果，並避免遇見狼。

<strong>強化學習</strong> (RL) 是一種學習技術，允許我們透過大量實驗學習代理人在某個<strong>環境</strong>中達成最佳行為。代理人在此環境應有一個<strong>目標</strong>，由一個<strong>回饋函數</strong>定義。

## 環境

為簡化起見，讓我們將彼得的世界視為一個 `width` x `height` 的方形棋盤，如下圖：

![彼得的環境](../../../../translated_images/zh-TW/environment.40ba3cb66256c93f.webp)

這張棋盤上的每一格可為：

* <strong>陸地</strong>，彼得與其他生物可行走的地方。
* <strong>水域</strong>，顯然無法行走的地方。
* <strong>樹木</strong>或<strong>草地</strong>，可供休息之處。
* <strong>蘋果</strong>，代表彼得會很高興找到的食物。
* <strong>狼</strong>，有危險應該避免。

有一個獨立的 Python 模組 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) 包含這個環境的相關程式碼。因為它對理解我們的概念不重要，我們將匯入此模組用來建立範例棋盤 (程式碼區塊 1)：

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

這段程式碼應該會列印出與上圖類似的環境圖形。

## 動作與策略

在本例中，彼得的目標是能找到蘋果，同時避開狼與其他障礙物。為達成目標，他可以四處走動直到找到蘋果。

因此，在任一位置，他能選擇向上、向下、向左或向右動作。

我們會將這些動作定義為字典，並將其映射為對應的座標變化。例如，向右移動(`R`) 對應座標變化對 `(1,0)`。(程式碼區塊 2)：

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

總結來說，本場景中策略與目標如下：

- <strong>策略</strong>：代理人(彼得)的策略由所謂的<strong>策略函數(policy)</strong>定義。策略是函數，輸入為任一狀態，輸出為該狀態下的動作。在我們的問題中，狀態由棋盤及玩家當前位置表示。

- <strong>目標</strong>：強化學習的目標是最終學習出一個能有效解決問題的良好策略。不過作為基線，我們先考慮最簡單的策略——**隨機行走(random walk)**。

## 隨機行走

讓我們先用隨機行走策略解決問題。隨機行走中，我們會從允許的動作中隨機選擇下一步，直到找到蘋果(程式碼區塊 3)。

1. 實作隨機行走的程式碼如下：

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # 步驟數量
        # 設定起始位置
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # 成功！
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # 被狼吃掉或溺水
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # 執行實際移動
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    `walk` 呼叫應回傳對應路徑長度，且此長度可能在不同次執行中有所不同。 

1. 執行多次走路實驗（例如 100 次），並列印結果統計數據 (程式碼區塊 4)：

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

    注意路徑的平均長度約為 30-40 步，相當多，考慮到離最近的蘋果平均距離約在 5-6 步左右。

    你也可以看到彼得在隨機行走時的移動樣貌：

    ![彼得的隨機行走](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## 獎勵函數

為了讓我們的策略更智慧，我們需了解哪些動作比其他更「好」。為此，我們必須定義目標。

目標可透過<strong>獎勵函數</strong>來定義，獎勵函數會對每個狀態回傳一個分數。數字愈大表示獎勵愈高。(程式碼區塊 5)

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

關於獎勵函數，有趣的是大多數情況下，<em>我們只有在遊戲結束時得到實質獎勵</em>。這表示我們的演算法應該記住導致正向獎勵的「好」步驟，並強化其重要性。反之，所有導致壞結果的動作應予以抑制。

## Q 學習

我們將討論一種稱為 **Q 學習** 的演算法。在此演算法中，策略由名為 **Q-表** 的函數(或資料結構)定義。Q-表紀錄每個狀態下各動作的「好壞」程度。

它稱為 Q-表，是因為常以表格或多維陣列形式呈現。由於我們的棋盤維度為 `width` x `height`，可用 shape 為 `width` x `height` x `len(actions)` 的 numpy 陣列來表示 Q-表：(程式碼區塊 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

注意，我們將 Q-表的初始值都設為相同的數值，本例為 0.25。這對應於「隨機行走」策略，因為在每個狀態下各動作的價值相等。我們可以將 Q-表傳給 `plot` 函數來在棋盤上視覺化此表：`m.plot(Q)`。

![彼得的環境](../../../../translated_images/zh-TW/env_init.04e8f26d2d60089e.webp)

在每一格中心都有一個「箭頭」指示偏好移動方向。因所有方向價值相同，所以會顯示一個點。

現在我們需要執行模擬，探索環境並學習更好的 Q-表分佈，以便更快速找到蘋果的路徑。

## Q 學習的核心：貝爾曼方程式

一旦開始移動，每個動作會有對應的回饋，也就是理論上可根據即時回饋選擇下一步動作。然而，在大多數狀態下，該動作不會立即達成我們想找蘋果的目標，因此無法馬上判斷哪個方向較好。

> 請記得，重要的不是即時結果，而是最後的結果，也就是模擬結束時獲得的結果。

為了因應這種延遲回饋，我們需要用到<strong>[動態規劃](https://en.wikipedia.org/wiki/Dynamic_programming)</strong> 的原理，讓我們能以遞迴方式思考問題。

假設我們目前處於狀態 *s*，想移動到下一狀態 *s'*。這麼做會拿到即時回饋 *r(s,a)*，由獎勵函數定義，外加未來的獎勵。如果假設 Q-表能正確反映每個動作的「誘因」，那麼在狀態 *s'* 我們會選擇動作 *a*，該動作使 *Q(s',a')* 最大。因此，在狀態 *s* 可取得的最佳未來回饋由 `max`<sub>a'</sub>*Q(s',a')* 定義（最大值基於狀態 *s'* 下的所有可能動作 *a'*）。

這樣我們得到對動作 *a* 在狀態 *s* 的 Q-表計算公式，即為以下 <strong>貝爾曼方程式</strong>：

<img src="../../../../translated_images/zh-TW/bellman-equation.7c0c4c722e5a6b7c.webp"/>

其中 γ 是所謂的<strong>折扣因子(discount factor)</strong>，決定你應多大程度偏好當前回饋而非未來回饋，反之亦然。

## 學習演算法

根據上述方程式，現在可以撰寫出學習演算法的偽代碼：

* 用相同值初始化所有狀態和動作的 Q-表 Q
* 設定學習率 α ← 1
* 重複多次模擬
   1. 從隨機位置開始
   1. 重複
        1. 在狀態 *s* 選擇行動 *a*
        2. 執行動作，移動到新狀態 *s'*
        3. 若達到遊戲結束條件，或總回饋太小則結束模擬  
        4. 在新狀態計算回饋 *r*
        5. 根據貝爾曼方程式更新 Q 函數：*Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. 更新總回饋並降低 α。

## 利用與探索

在以上演算法中，我們沒指定如何在步驟 2.1 選擇動作。若隨機選擇動作，會<strong>探索</strong>環境，也可能經常死亡並進入不常去的區域。另一種策略是<strong>利用</strong>既有的 Q-表數值，總是選擇狀態 *s* 下 Q 值最高的動作。但這會阻止探勘其它狀態，可能無法找到最佳解。

因此最佳方式是在探索與利用間取得平衡。這可作為以 Q-表值為比例機率選擇行動。起初所有 Q-表值相同，即相當於隨機選擇；隨著學習加深，較常走最優路徑，但仍偶爾嘗試未探索的路徑。

## Python 實作

現在準備實作學習演算法。前置工作是寫個函數將 Q-表任意數值轉成對應動作機率的向量。

1. 寫一個函數 `probs()`：

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

    我們在原向量加幾個 `eps` 以避免初期向量成分完全一樣時的除零錯誤。

執行 5000 次實驗（稱為 **epochs**）來跑學習演算法：(程式碼區塊 8)
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
            m.move(dpos,check_correctness=False) # 我們允許玩家移動到棋盤外，該動作將終止回合
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

執行完演算法後，Q-表應會更新，定義各步驟不同動作的誘因。我們試著將 Q-表視覺化：在每個格子畫出指向期望移動方向的向量。為簡便，我們用小圓圈代替箭頭。

<img src="../../../../translated_images/zh-TW/learned.ed28bcd8484b5287.webp"/>

## 檢查策略

由於 Q-表記錄了各狀態中動作的「誘因」，用它來決定我們世界中的有效導航相當簡單。最簡單方法是選擇具有最高 Q-值的動作：(程式碼區塊 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```


> 如果你嘗試執行上面的程式碼好幾次，你可能會注意到有時它會「卡住」，這時你需要按下筆記本中的停止按鈕以中斷它。這是因為可能出現兩個狀態在最佳 Q-值的意義上「互相指向」的情況，在這種情況下，代理會不斷在那兩個狀態之間移動而無限循環。

## 🚀挑戰

> **任務 1：** 修改 `walk` 函式，限制路徑的最大長度為一定的步數（例如，100步），並觀察上面程式碼不時返回這個值。

> **任務 2：** 修改 `walk` 函式，使其不會返回先前已經到過的地方。這將防止 `walk` 產生無限循環，但代理仍然可能會被困在一個無法逃脫的位置。

## 導航

一個更好的導航策略是我們訓練時使用的，結合了利用與探索。在這個策略中，我們會以與 Q-表中數值成比例的機率選擇每個動作。這個策略仍可能導致代理回到已探索過的位置，但正如以下程式碼所示，這會導致到達目標位置的平均路徑非常短（記得 `print_statistics` 會執行模擬100次）：(code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

執行此程式碼後，你應該會看到平均路徑長度比之前小得多，大約介於3到6之間。

## 探討學習過程

如我們所提，學習過程是在探索與利用獲得的問題空間結構知識之間取得平衡。我們已看到學習結果（幫助代理找到短路徑的能力）有所提升，但觀察平均路徑長度在學習過程中的行為也很有趣：

<img src="../../../../translated_images/zh-TW/lpathlen1.0534784add58d4eb.webp"/>

這些學習可以總結為：

- <strong>平均路徑長度先增加</strong>。我們看到的是，一開始平均路徑長度會增加。這大概是因為當我們對環境一無所知時，很可能會被困在壞狀態中，如水域或狼。當我們學到更多並開始利用這些知識時，可以探索環境更長時間，但仍不太了解蘋果的位置。

- <strong>隨著學習增加路徑長度減少</strong>。一旦學習足夠，代理達成目標變得更容易，路徑長度開始縮短。然而，我們仍然保持探索，因此常常偏離最佳路徑並嘗試新選項，使路徑比最優長。

- <strong>路徑長度突然增加</strong>。我們也從圖中觀察到，有時路徑長度會突然增加。這顯示過程的隨機性，以及在某些時候我們可能會用新值覆蓋 Q-表係數，導致「破壞」它們。在理想狀況下，這種情形應該透過降低學習率來減少（例如，在訓練末期，只以很小的值調整 Q-表）。

整體而言，重要的是要記住學習過程的成功與品質在很大程度上取決於參數，如學習率、學習率衰減及折扣因子。這些通常稱為<strong>超參數</strong>，用以區別於在訓練中優化的<strong>參數</strong>（例如 Q-表係數）。尋找最佳超參數的過程稱為<strong>超參數優化</strong>，這是值得另行討論的主題。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 作業
[一個更真實的世界](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力追求準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於關鍵資訊，建議採用專業人工翻譯。我們不對因使用此翻譯所產生的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->