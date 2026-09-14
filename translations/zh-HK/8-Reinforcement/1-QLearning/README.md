# 強化學習和Q學習簡介

![機器學習中強化學習的摘要手繪筆記](../../../../translated_images/zh-HK/ml-reinforcement.94024374d63348db.webp)
> 手繪筆記由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 製作

強化學習包含三個重要概念：代理人、一些狀態，以及每個狀態的動作集。通過在指定的狀態執行一個動作，代理人會獲得獎勵。再次想像電腦遊戲超級瑪利歐。你是瑪利歐，你在遊戲關卡中，站在懸崖邊緣。你上面有一個金幣。你作為瑪利歐，在遊戲關卡中，處於一個特定位置...這就是你的狀態。向右邁一步（一個動作）會讓你掉下懸崖，這會給你一個較低的分數。然而，按跳躍按鈕會讓你得分並存活。那是正面結果，應該給你一個正數分數。

使用強化學習和模擬器（遊戲），你可以學會如何玩這款遊戲，最大化獎勵，即保持存活並盡可能得分。

[![強化學習介紹](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 點擊上圖聽Dmitry討論強化學習

## [課前小測](https://ff-quizzes.netlify.app/en/ml/)

## 先決條件與設定

在本課程中，我們將用Python試驗一些程式碼。你應該能在自己的電腦或雲端環境中運行本課的Jupyter Notebook程式碼。

你可以打開[課程筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb)，跟著課程一起構建。

> **注意：** 如果你是從雲端開啟此程式碼，也需下載[`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py)檔案，此檔案用於筆記本程式碼中。把它放在與筆記本相同目錄。

## 介紹

本課將探討受俄羅斯作曲家[謝爾蓋·普羅科菲耶夫](https://en.wikipedia.org/wiki/Sergei_Prokofiev)的音樂童話故事<strong>[彼得與狼](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)</strong>啟發的世界。我們將使用<strong>強化學習</strong>讓彼得探索環境，收集美味蘋果，並避免遇見狼。

<strong>強化學習</strong>（RL）是一種學習技術，允許我們通過多次實驗學習一個<strong>代理人</strong>在某個<strong>環境</strong>中的最佳行為。代理人在此環境中應有某個由<strong>獎勵函數</strong>定義的<strong>目標</strong>。

## 環境

為簡化起見，將彼得的世界視為一個 `width` x `height` 的方格棋盤，如下所示：

![彼得的環境](../../../../translated_images/zh-HK/environment.40ba3cb66256c93f.webp)

這棋盤中的每個格子可以是：

* <strong>地面</strong>，彼得和其他生物可以行走。
* <strong>水域</strong>，顯然無法行走。
* <strong>樹</strong>或<strong>草地</strong>，可作休息之處。
* <strong>蘋果</strong>，代表彼得非常樂意找到來餵養自己。
* <strong>狼</strong>，危險且應避免。

有一個獨立的Python模組 [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py)，包含了操作此環境的程式碼。基於程式碼對理解概念非必須，我們將導入此模組並用它創建範例棋盤（代碼塊1）：

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

此程式碼應列印與上圖類似的環境圖片。

## 動作與策略

在我們的範例中，彼得的目標是找到蘋果，同時避開狼和其它障礙物。為此，他基本上可以四處走動，直到找到蘋果為止。

因此，在任何位置，他都能選擇以下動作之一：上、下、左、右。

我們會定義這些動作為字典，並將它們對應至相應座標變化對。例如，向右走（`R`）對應座標變化對 `(1,0)`。（代碼塊2）：

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

總結而言，此情境的策略和目標如下：

- <strong>策略</strong>，我們的代理人（彼得）由所謂的<strong>策略(policy)</strong>定義。策略是一個函數，對任一狀態給予所採取的動作。在此例中，問題的狀態由棋盤的整體情況表示，包括玩家當前位置。

- <strong>目標</strong>，強化學習的目標是最終學習出一套好的策略，使得問題能有效解決。然而，作為基線，讓我們考慮最簡單的策略，稱為<strong>隨機漫步</strong>。

## 隨機漫步

讓我們先用隨機漫步策略來解決問題。隨機漫步中，我們從允許動作中隨機選擇下一步，直到找到蘋果（代碼塊3）。

1. 用以下代碼實現隨機漫步：

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # 步數
        # 設定初始位置
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # 成功！
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # 被狼吃或溺斃
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # 執行實際移動
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    調用 `walk` 應回傳對應路徑長度，此長度會因運行而異。

1. 執行walk實驗若干次（例如100次），並列印結果統計數據（代碼塊4）：

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

    注意路徑長度平均約30至40步，考慮到到最近蘋果的平均距離約5至6步，數量相當多。

    你還可以看到彼得在隨機漫步時的移動模樣：

    ![彼得的隨機漫步](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## 獎勵函數

為了使策略更智慧，我們需要了解哪些動作比較“好”。為此，我們要定義目標。

目標可用<strong>獎勵函數</strong>來定義，它對每個狀態返回一個分數值。數字越高，獎勵函數越好。（代碼塊5）

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

獎勵函數的一個有趣特點是，在大多數情況下，<em>我們只在遊戲結束時得到重要獎勵</em>。這意味著演算法應記住那些導致最終正面獎勵的“好”步驟，並增強它們的重要性。相似地，所有導致負面結果的動作應該被抑制。

## Q學習

我們將討論的演算法叫做<strong>Q學習</strong>。在此演算法中，策略由一個名為<strong>Q表</strong>的函數（或資料結構）定義。它記錄給定狀態中每個動作的“好壞”。

它叫Q表，是因為通常將其表示成表格或多維陣列比較方便。由於我們的棋盤尺寸為 `width` x `height`，我們可以用形狀為 `width` x `height` x `len(actions)` 的numpy陣列表示Q表：（代碼塊6）

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

注意，我們用相同數值（此處為0.25）初始化Q表所有值。這代表“隨機漫步”策略，因為在每個狀態中所有動作的好壞相等。我們可以呼叫 `m.plot(Q)` 函數，視覺化棋盤上的Q表。

![彼得的環境](../../../../translated_images/zh-HK/env_init.04e8f26d2d60089e.webp)

在每個格子中央有個「箭頭」，顯示偏好的移動方向。由於所有方向相等，顯示為一個點。

現在我們需要運行模擬，探索環境，學習更佳的Q表分布，這將幫助我們更快找到蘋果的路徑。

## Q學習的本質：貝爾曼方程

一旦開始移動，每個動作會有對應的獎勵，即我們理論上可以根據最高即時獎勵選擇下一步動作。但在多數狀態下，移動不會立即達成找到蘋果的目標，因此我們不能立即判斷哪方向較好。

> 請記住，重要的不是立刻結果，而是我們在模擬結束時獲得的最終結果。

為了解決這個延遲獎勵問題，我們需使用<strong>[動態規劃](https://en.wikipedia.org/wiki/Dynamic_programming)</strong>原理，使我們能對問題進行遞歸思考。

假設我們當前在狀態 *s*，想移動到下一狀態 *s'*。如此，我們將獲得即時獎勵 *r(s,a)*，由獎勵函數定義，加上一些未來獎勵。如果Q表已正確反映每個動作的「吸引力」，在狀態 *s'* 我們會選擇使值 *Q(s',a')* 最大的動作 *a'*。因此，在狀態 *s* 所能獲得的最佳未來獎勵定義為對所有可能動作 *a'* 的 `max`<sub>a'</sub>*Q(s',a')*。

這便得到以貝爾曼公式計算狀態 *s*，行動 *a* 的Q表值：

<img src="../../../../translated_images/zh-HK/bellman-equation.7c0c4c722e5a6b7c.webp"/>

其中 γ 是所謂的<strong>折扣因子</strong>，決定你應偏好現有獎勵還是未來獎勵的程度。

## 學習演算法

根據上述公式，我們可寫出演算法偽碼：

* 初始化Q表Q，狀態和行動均設相同數值
* 設學習率α ← 1
* 重複多次模擬
   1. 從隨機位置開始
   1. 重複
        1. 在狀態<em>s</em>選擇動作<em>a</em>
        2. 執行動作，移至新狀態<em>s'</em>
        3. 若遇遊戲結束條件或總獎勵太低—結束模擬  
        4. 計算新狀態下的獎勵<em>r</em>
        5. 根據貝爾曼方程更新Q函數：*Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. 更新總獎勵並減少α。

## 利用 vs 探索

在上面演算法中，未具體說明第2.1步如何選擇動作。如果我們隨機選擇動作，會隨機<strong>探索</strong>環境，也很可能經常死亡並探索不常去的區域。另一方式是<strong>利用</strong>已知Q表值，選擇在狀態<em>s</em>中Q值最高的動作，然而這可能阻礙我們探索其他狀態，因而可能無法找到最佳解。

因此，最佳方案是在探索與利用間取得平衡。可透過依Q表數值比例選動作，在開始時所有Q表值相同，這等同於隨機選擇，隨著學習增加，你將傾向走最佳路徑，同時偶爾讓代理人選擇未探索路徑。

## Python實作

現在準備實作學習演算法。在此之前，我們也需要將Q表中的任意數字轉換成對應動作的機率向量的函數。

1. 創建函數 `probs()`：

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

    我們向原始向量添加少許 `eps`，避免初始全相同元素時除以0。

執行學習演算法5000次實驗，稱為<strong>epochs</strong>：（代碼塊8）
```python
    for epoch in range(5000):
    
        # 選擇初始點
        m.random_start()
        
        # 開始旅行
        n=0
        cum_reward = 0
        while True:
            x,y = m.human
            v = probs(Q[x,y])
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            m.move(dpos,check_correctness=False) # 我們允許玩家移動出棋盤範圍，這將結束本回合
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

執行此演算法後，Q表將被更新，用以定義不同步驟中動作的吸引力。我們可通過在每個格子繪製指向期望移動方向的向量來視覺化Q表。簡化起見，我們畫一個小圓圈代替箭頭。

<img src="../../../../translated_images/zh-HK/learned.ed28bcd8484b5287.webp"/>

## 檢查策略

因Q表列出了每狀態行動的“吸引力”，利用它定義高效導航很容易。最簡單情況下，我們選擇對應Q值最高的行動：（代碼塊9）

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```


> 如果你多次嘗試上面的程式碼，你可能會注意到有時它會「停頓」，你需要按筆記本上的停止按鈕來中斷它。這是因為可能存在兩個狀態在最佳 Q 值方面「互相指向」的情況，導致代理無限地在這些狀態之間移動。

## 🚀挑戰

> **任務 1：** 修改 `walk` 函數，以限制路徑的最大長度（例如 100 步），並觀察上面的程式碼不時返回該值。

> **任務 2：** 修改 `walk` 函數使其不會返回之前已經去過的地方。這會防止 `walk` 產生迴圈，但代理仍然有可能被「困」在無法逃脫的位置。

## 導航

一個更好的導航策略是我們訓練時使用的策略，結合了利用和探索。在此策略中，我們將以與 Q 表中的值成比例的某個概率選擇每個動作。這種策略仍可能導致代理返回已探索過的位置，但如下面的程式碼所示，它能產生到目標位置非常短的平均路徑長度（記住 `print_statistics` 是執行了 100 次模擬）：(程式碼區塊 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

執行此程式碼後，你應該會得到明顯比之前更短的平均路徑長度，約在 3-6 之間。

## 探究學習過程

如前所述，學習過程是在對問題空間結構已獲得知識的探索與利用之間取得平衡。我們看到學習結果（幫助代理找到目標的短路徑能力）有所改善，但觀察學習過程中平均路徑長度的變化也很有趣：

<img src="../../../../translated_images/zh-HK/lpathlen1.0534784add58d4eb.webp"/>

學習過程可總結為：

- <strong>平均路徑長度上升</strong>。我們看到一開始平均路徑長度會增加。這很可能是因為當我們對環境一無所知時，代理很可能卡在不利狀態，比如水域或狼身邊。隨著學習增多並開始利用這些知識，我們可以探索環境更久，但仍不完全知道蘋果在哪裡。

- <strong>學習更多後路徑長度下降</strong>。當學習足夠後，代理更容易達成目標，路徑長度開始減少。不過我們仍持續探索，因此經常會偏離最佳路徑，嘗試新選項，使路徑長度比理想狀況長。

- <strong>路徑長度突然增加</strong>。我們還看到圖中某些點路徑長度突然增加，這體現了過程的隨機性，意味著有時會用新值覆寫 Q 表係數，對它們造成「損害」。理想情況下可透過減小學習率（例如訓練後期只做小幅度調整）來降低此影響。

總體來說，要記住學習過程的成功與質量很大程度上取決於參數，例如學習率、學習率衰減和折扣因子。這些通常稱為<strong>超參數</strong>，以區別於訓練中優化的<strong>參數</strong>（例如 Q 表係數）。尋找最佳超參數值的過程稱為<strong>超參數優化</strong>，是值得獨立探討的主題。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 作業
[一個更現實的世界](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->