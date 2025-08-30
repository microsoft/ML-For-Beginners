<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9660fbd80845c59c15715cb418cd6e23",
  "translation_date": "2025-08-29T22:13:34+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "mo"
}
-->
## 先決條件

在本課中，我們將使用一個名為 **OpenAI Gym** 的庫來模擬不同的 **環境**。你可以在本地運行本課的代碼（例如，使用 Visual Studio Code），此時模擬將在新窗口中打開。如果在線運行代碼，可能需要對代碼進行一些調整，具體請參考[這裡](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7)。

## OpenAI Gym

在上一課中，遊戲的規則和狀態是由我們自己定義的 `Board` 類提供的。而在這裡，我們將使用一個特殊的 **模擬環境**，它將模擬平衡桿的物理行為。最受歡迎的強化學習算法模擬環境之一是 [Gym](https://gym.openai.com/)，由 [OpenAI](https://openai.com/) 維護。通過使用這個 Gym，我們可以創建不同的 **環境**，從平衡桿模擬到 Atari 遊戲。

> **注意**：你可以在 [這裡](https://gym.openai.com/envs/#classic_control) 查看 OpenAI Gym 提供的其他環境。

首先，讓我們安裝 gym 並導入所需的庫（代碼塊 1）：

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## 練習 - 初始化一個平衡桿環境

為了解決平衡桿問題，我們需要初始化相應的環境。每個環境都與以下內容相關聯：

- **觀察空間**：定義我們從環境中接收到的信息結構。對於平衡桿問題，我們接收到桿的位置、速度以及其他一些值。

- **行動空間**：定義可能的行動。在我們的例子中，行動空間是離散的，包括兩個行動——**左** 和 **右**。（代碼塊 2）

1. 要初始化，輸入以下代碼：

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

為了了解環境如何運作，讓我們運行一個 100 步的短模擬。在每一步中，我們提供一個行動——在這個模擬中，我們只是隨機從 `action_space` 中選擇一個行動。

1. 運行以下代碼，看看會發生什麼。

    ✅ 請記住，建議在本地 Python 安裝中運行此代碼！（代碼塊 3）

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    你應該會看到類似於這張圖片的效果：

    ![無法平衡的平衡桿](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. 在模擬過程中，我們需要獲取觀察值以決定如何行動。實際上，`step` 函數返回當前的觀察值、一個獎勵函數，以及一個表示是否應繼續模擬的 `done` 標誌：（代碼塊 4）

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    你會在筆記本輸出中看到類似以下的內容：

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

    在模擬的每一步中返回的觀察向量包含以下值：
    - 小車的位置
    - 小車的速度
    - 桿的角度
    - 桿的旋轉速率

1. 獲取這些數值的最小值和最大值：（代碼塊 5）

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    你可能還會注意到，在每一步模擬中，獎勵值始終為 1。這是因為我們的目標是盡可能長時間地保持桿在合理的垂直位置。

    ✅ 實際上，如果我們能在 100 次連續試驗中平均獲得 195 的獎勵值，就可以認為平衡桿問題已經解決。

## 狀態離散化

在 Q-Learning 中，我們需要構建 Q-表來定義在每個狀態下應該採取的行動。為了做到這一點，我們需要將狀態 **離散化**，更準確地說，它應該包含有限數量的離散值。因此，我們需要以某種方式將觀察值 **離散化**，將其映射到有限的狀態集合。

有幾種方法可以做到這一點：

- **分成區間**。如果我們知道某個值的範圍，我們可以將該範圍分成若干個 **區間**，然後用該值所屬的區間編號來替代原值。這可以使用 numpy 的 [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) 方法來完成。在這種情況下，我們可以精確地知道狀態的大小，因為它將取決於我們為離散化選擇的區間數量。

✅ 我們可以使用線性插值將值映射到某個有限範圍（例如，從 -20 到 20），然後通過四捨五入將數字轉換為整數。這種方法對狀態大小的控制稍弱，特別是當我們不知道輸入值的確切範圍時。例如，在我們的例子中，4 個值中的 2 個沒有上下界，這可能導致狀態數量無限。

在我們的例子中，我們將採用第二種方法。正如你稍後可能注意到的，儘管某些值沒有明確的上下界，但它們很少會超出某些有限範圍，因此具有極端值的狀態將非常罕見。

1. 以下是將模型的觀察值轉換為 4 個整數值元組的函數：（代碼塊 6）

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. 讓我們還探索另一種使用區間的離散化方法：（代碼塊 7）

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

1. 現在運行一個短模擬，觀察這些離散化的環境值。可以嘗試 `discretize` 和 `discretize_bins`，看看是否有差異。

    ✅ `discretize_bins` 返回的是區間編號，從 0 開始。因此，對於接近 0 的輸入變量值，它返回的是區間中間的數字（10）。在 `discretize` 中，我們不關心輸出值的範圍，允許它們為負數，因此狀態值未偏移，0 對應於 0。（代碼塊 8）

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

    ✅ 如果你想查看環境的執行情況，可以取消註釋以 `env.render` 開頭的行。否則，你可以在後台執行，這樣速度更快。在我們的 Q-Learning 過程中，我們將使用這種“隱形”執行方式。

## Q-表結構

在上一課中，狀態是一對從 0 到 8 的簡單數字，因此用形狀為 8x8x2 的 numpy 張量來表示 Q-表非常方便。如果我們使用區間離散化，狀態向量的大小也是已知的，因此我們可以使用相同的方法，將狀態表示為形狀為 20x20x10x10x2 的數組（其中 2 是行動空間的維度，前幾個維度對應於我們為觀察空間中每個參數選擇的區間數量）。

然而，有時觀察空間的精確維度是未知的。在使用 `discretize` 函數的情況下，我們無法確保狀態保持在某些限制範圍內，因為某些原始值是無界的。因此，我們將使用稍微不同的方法，通過字典來表示 Q-表。

1. 使用 *(state, action)* 作為字典鍵，值對應於 Q-表的條目值。（代碼塊 9）

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    在這裡，我們還定義了一個函數 `qvalues()`，它返回對應於給定狀態的所有可能行動的 Q-表值列表。如果 Q-表中沒有該條目，我們將返回默認值 0。

## 開始 Q-Learning

現在我們準備教 Peter 如何保持平衡了！

1. 首先，設置一些超參數：（代碼塊 10）

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    其中，`alpha` 是 **學習率**，定義了我們在每一步中應該在多大程度上調整 Q-表的當前值。在上一課中，我們從 1 開始，然後在訓練過程中將 `alpha` 降低到較小的值。在本例中，為了簡化，我們將保持其恆定，你可以稍後嘗試調整 `alpha` 值。

    `gamma` 是 **折扣因子**，表示我們應該在多大程度上優先考慮未來的獎勵而非當前的獎勵。

    `epsilon` 是 **探索/利用因子**，決定我們應該更傾向於探索還是利用。在我們的算法中，我們將在 `epsilon` 百分比的情況下根據 Q-表值選擇下一個行動，而在剩餘的情況下執行隨機行動。這將允許我們探索以前從未見過的搜索空間區域。

    ✅ 就平衡而言，選擇隨機行動（探索）就像是一個隨機的錯誤方向的推動，桿子需要學會如何從這些“錯誤”中恢復平衡。

### 改進算法

我們還可以對上一課的算法進行兩項改進：

- **計算平均累積獎勵**，在多次模擬中取平均值。我們將每 5000 次迭代打印一次進度，並將累積獎勵取平均值。如果我們獲得超過 195 分，就可以認為問題已經解決，並且質量甚至高於要求。

- **計算最大平均累積結果**，`Qmax`，並存儲對應於該結果的 Q-表值。在訓練過程中，你會注意到有時平均累積結果開始下降，我們希望保留對應於訓練過程中觀察到的最佳模型的 Q-表值。

1. 在每次模擬中將所有累積獎勵收集到 `rewards` 向量中，以便進一步繪圖。（代碼塊 11）

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

從這些結果中你可能會注意到：

- **接近目標**。我們非常接近實現目標，即在 100 次以上的連續模擬中獲得 195 的累積獎勵，或者我們可能已經實現了目標！即使我們獲得較小的數字，我們仍然無法確定，因為我們是對 5000 次運行取平均值，而正式標準只需要 100 次運行。

- **獎勵開始下降**。有時獎勵開始下降，這意味著我們可能會用更糟糕的值覆蓋 Q-表中已經學到的值。

如果我們繪製訓練進度，這一觀察會更加明顯。

## 繪製訓練進度

在訓練過程中，我們將每次迭代的累積獎勵值收集到 `rewards` 向量中。以下是將其與迭代次數繪製在一起的結果：

```python
plt.plot(rewards)
```

![原始進度](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.mo.png)

從這張圖中無法看出什麼，因為由於隨機訓練過程的特性，訓練會話的長度變化很大。為了讓這張圖更有意義，我們可以計算一系列實驗的 **移動平均值**，例如 100 次。這可以方便地使用 `np.convolve` 完成：（代碼塊 12）

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![訓練進度](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.mo.png)

## 調整超參數

為了使學習更加穩定，我們可以在訓練過程中調整一些超參數。特別是：

- **對於學習率**，`alpha`，我們可以從接近 1 的值開始，然後逐漸降低該參數。隨著時間的推移，我們會在 Q-表中獲得較好的概率值，因此我們應該稍微調整它們，而不是完全用新值覆蓋。

- **增加 epsilon**。我們可能希望慢慢增加 `epsilon`，以便減少探索，更多地進行利用。這可能意味著從較低的 `epsilon` 值開始，然後逐漸增加到接近 1。
> **任務 1**：嘗試調整超參數的值，看看是否能獲得更高的累積回報。你的回報是否超過 195？
> **任務 2**：為了正式解決這個問題，你需要在連續 100 次運行中獲得 195 的平均回報。在訓練過程中測量這一點，並確保你已經正式解決了這個問題！

## 查看結果的實際效果

實際觀察訓練後的模型行為會很有趣。我們來運行模擬，並遵循與訓練時相同的動作選擇策略，根據 Q-Table 中的概率分佈進行採樣：（代碼塊 13）

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

你應該會看到類似這樣的畫面：

![平衡的 CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀挑戰

> **任務 3**：在這裡，我們使用的是 Q-Table 的最終版本，但這可能不是表現最好的版本。記住，我們已經將表現最好的 Q-Table 存儲在 `Qbest` 變數中！嘗試將 `Qbest` 複製到 `Q` 中，並使用表現最好的 Q-Table 來運行相同的例子，看看是否能觀察到差異。

> **任務 4**：在這裡，我們並不是每一步都選擇最佳動作，而是根據相應的概率分佈進行採樣。是否更合理每次都選擇 Q-Table 值最高的最佳動作？這可以通過使用 `np.argmax` 函數來找到對應於最高 Q-Table 值的動作編號。實現這種策略，看看是否能改善平衡效果。

## [課後測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/)

## 作業
[訓練一輛山地車](assignment.md)

## 結論

我們現在已經學會如何通過提供定義遊戲期望狀態的回報函數，並讓代理智能地探索搜索空間，來訓練代理以獲得良好的結果。我們成功地在離散和連續環境（但動作是離散的情況下）中應用了 Q-Learning 演算法。

同樣重要的是研究動作狀態也是連續的情況，以及觀察空間更加複雜的情況，例如來自 Atari 遊戲畫面的圖像。在這些問題中，我們通常需要使用更強大的機器學習技術，例如神經網絡，來獲得良好的結果。這些更高級的主題將是我們即將推出的更高級 AI 課程的內容。

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵信息，建議尋求專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋不承擔責任。