# 賽車桿滑冰

我們在上一課解決的問題可能看起來像個玩具問題，似乎不太適用於現實生活場景。但事實並非如此，因為許多現實世界的問題也有類似的場景——包括下棋或圍棋。它們相似，因為我們也有一個有既定規則的棋盤和一個<strong>離散狀態</strong>。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

## 介紹

本課中，我們將把相同的 Q-Learning 原則應用於一個具有<strong>連續狀態</strong>的問題，也就是由一個或多個實數表示的狀態。我們將處理以下問題：

> <strong>問題</strong>：如果 Peter 想要逃離狼群，他需要能夠更快移動。我們將看看 Peter 如何通過 Q-Learning 學會滑冰，尤其是如何保持平衡。

![偉大的逃脫！](../../../../translated_images/zh-HK/escape.18862db9930337e3.webp)

> Peter 和朋友們發揮創意以逃離狼群！圖片來自 [Jen Looper](https://twitter.com/jenlooper)

我們將使用一個簡化版本的平衡問題，稱為 **賽車桿（CartPole）** 問題。在賽車桿世界中，我們有一個可左右移動的水平滑軌，目標是在滑軌上平衡一根垂直桿。

<img alt="a cartpole" src="../../../../translated_images/zh-HK/cartpole.b5609cc0494a14f7.webp" width="200"/>

## 前置知識

本課中，我們將使用一個叫做 **OpenAI Gym** 的庫來模擬不同的<strong>環境</strong>。你可以在本地執行本課代碼（例如，從 Visual Studio Code 中），此時模擬會在新視窗中打開。若在網上執行代碼，你可能需要對代碼做一些調整，具體說明見[這裡](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7)。

## OpenAI Gym

在上一課中，遊戲規則和狀態由我們自定義的 `Board` 類給出。這裡我們會使用一個特殊的<strong>模擬環境</strong>，模擬平衡桿背後的物理。OpenAI 維護的一個非常流行的強化學習訓練模擬環境叫做 [Gym](https://gym.openai.com/)，使用該 Gym，我們可以創建從賽車桿模擬到街機遊戲等不同的<strong>環境</strong>。

> <strong>注意</strong>：你可以在 [這裡](https://gym.openai.com/envs/#classic_control) 查看 OpenAI Gym 的其他可用環境。

首先，讓我們安裝 gym 並導入所需庫（代碼塊 1）：

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## 練習 - 初始化賽車桿環境

若要處理賽車桿平衡問題，我們需要初始化相應環境。每個環境都關聯著：

- <strong>觀察空間</strong>，定義了我們從環境中接收到的信息結構。對賽車桿問題而言，我們獲取桿的位置、速度及其他一些數值。

- <strong>動作空間</strong>，定義可能的動作。在我們的例子中，動作空間是離散的，由兩個動作組成——<strong>左</strong>和<strong>右</strong>。（代碼塊 2）

1. 初始化，輸入如下代碼：

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

看看環境如何工作，我們運行100步簡短模擬。每一步，我們提供要採取的一個動作——這個模擬中，我們只是從 `action_space` 隨機選擇一個動作。

1. 執行下面的代碼，看看結果如何。

    ✅ 請記住，最好在本地 Python 環境運行此代碼！（代碼塊 3）

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    你應該會看到類似下圖的畫面：

    ![未平衡的賽車桿](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. 在模擬過程中，我們需要獲取觀察值以決定如何行動。實際上，`step` 函數返回當前觀察、獎勵函數值以及完成標誌，用於指示是否應繼續模擬：（代碼塊 4）

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    你最終會在筆記本輸出中看到如下內容：

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

    每步模擬返回的觀察向量包含以下數值：
    - 賽車位置
    - 賽車速度
    - 桿的角度
    - 桿的旋轉速率

1. 獲取這些數值的最小和最大值：（代碼塊 5）

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    你可能還會注意到，每步模擬的獎勵值總是1。這是因為我們的目標是在最長時間內保持桿子相對垂直，也就是盡可能長時間生存。

    ✅ 事實上，當我們在100次連續試驗中平均獲得195分時，CartPole 模擬被認為已經解決。

## 狀態離散化

在 Q-Learning 中，我們需要構建 Q-Table，定義在每個狀態下該怎麼做。為此，我們需要狀態是<strong>離散的</strong>，更精確說，應包含有限數量的離散值。因此，我們需要將觀察值<strong>離散化</strong>，映射到一個有限的狀態集合。

有幾種方法可以做到這點：

- <strong>分箱</strong>。若知道某數值的區間，我們可以將該區間分成多個<strong>箱子</strong>，然後用該數值所在的箱位來替代原數值。這可以用 numpy 的 [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) 方法實現。如此一來，我們確切知道狀態大小，因為它取決於分箱數量。
  
✅ 另一種方法是利用線性插值將數值映射到有限區間（例如從 -20 到 20），然後通過四捨五入轉換為整數。這種方法對狀態大小的控制較少，尤其當我們不知道輸入數據的確切範圍時。比如本例中，4個數值中有2個沒有明確上限或下限，可能導致無限狀態數。

在本例中，我們會採用第二種方法。如你稍後可能會注意到，儘管沒有明確的界限，這些值很少超出某個有限區間，因此極端值的狀態會非常罕見。

1. 這是根據模擬觀察結果返回4個整數元組的函數：（代碼塊 6）

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. 我們也來看一下利用分箱的另一種離散化方法：（代碼塊 7）

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # 每個參數的數值區間
    nbins = [20,20,10,10] # 每個參數的分箱數量
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. 現在運行一段短模擬，觀察這些離散化的環境值。你可以嘗試 `discretize` 和 `discretize_bins`，看看有何不同。

    ✅ `discretize_bins` 返回的是箱號，從0開始。因此對接近0的輸入值，它返回中間箱號（10）。而 `discretize` 不限制輸出範圍，允許負值，狀態值未偏移，0 對應於 0。（代碼塊 8）

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

    ✅ 如果你想看到環境如何執行，可取消註釋以 `env.render` 開頭的行。否則你也可以讓它在後台執行，速度更快。我們在 Q-Learning 過程中將使用這種“隱形”執行方式。

## Q-Table 結構

在上一課中，狀態是從0到8的簡單數字對，因此我們用形狀為 8x8x2 的 numpy 張量方便地表示 Q-Table。如果我們使用分箱離散化，狀態向量大小也是已知的，因此我們可以用形狀為 20x20x10x10x2 的陣列表示狀態（其中 2 是動作空間維度，前幾維是對應每個觀察空間參數選定的分箱數）。

但有時候，觀察空間確切維度不明。對 `discretize` 函數而言，我們始終無法保證狀態會保持在一定限制內，因為原始值中有些沒有限定上下限。因此，我們將採用稍微不同的方法，用字典來表示 Q-Table。

1. 使用 *(state,action)* 對作為字典的鍵，對應值即 Q-Table 的條目值。（代碼塊 9）

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    這裡我們還定義了 `qvalues()` 函數，返回給定狀態在所有可能動作下的 Q-Table 值列表。若 Q-Table 中沒有該條目，預設回傳 0。

## 開始 Q-Learning

現在我們準備教 Peter 平衡了！

1. 首先，設置一些超參數：（代碼塊 10）

    ```python
    # 超參數
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    這裡，`alpha` 是<strong>學習率</strong>，定義每步調整 Q-Table 當前值的幅度。上一課我們從1開始，然後在訓練過程中將 `alpha` 降到較低值。這裡為簡便起見，我們保持其恒定，後續你可以試試調整 `alpha`。

    `gamma` 是<strong>折扣因子</strong>，表示我們對未來獎勵相比於當前獎勵的重視程度。

    `epsilon` 是<strong>探索/利用因子</strong>，決定我們應該優先探索還是利用。在算法中，我們會在 `epsilon` 百分比情況下根據 Q-Table 選擇下一步動作，剩餘情況隨機選擇動作。這可以讓我們探索搜索空間中未曾見過的區域。

    ✅ 對平衡問題而言，隨機選擇動作（探索）就像隨機向錯誤方向用力，桿子必須學會如何從這些“錯誤”中恢復平衡。

### 優化算法

我們可以對上一課的算法作兩個改進：

- <strong>計算平均累積獎勵</strong>，統計多次模擬。每5000次迭代我們輸出一次進度，並對累積獎勵取平均值。若平均超過 195 分，我們可認為問題已被解決，甚至優於要求質量。
  
- <strong>記錄最大平均累積結果</strong> `Qmax`，存儲對應該結果的 Q-Table。在訓練中你會發現，平均累積結果有時會下降，我們希望保留訓練中表現最好的 Q-Table。

1. 收集每次模擬的累積獎勵到 `rewards` 向量，便於繪圖分析。（代碼塊 11）

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
        # == 進行模擬 ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # 利用 - 根據 Q 表的機率選擇行動
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # 探索 - 隨機選擇行動
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == 定期列印結果及計算平均獎賞 ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

你可能會注意到：

- <strong>接近目標</strong>。我們很接近在100次連續模擬中獲得195累積獎勵的目標，甚至可能已經達成！即使數字較小，我們也無法判斷，因為取的是5000次的平均值，而正式判定只需100次。
  
- <strong>獎勵開始下降</strong>。有時獎勵開始下降，表示我們的 Q-Table 中的值開始被更差的值覆蓋，導致模型退化。

這一現象在繪製訓練過程中更為明顯。

## 繪製訓練進度

在訓練中，我們將每次迭代的累積獎勵值收集到了 `rewards` 向量中。當將其對迭代次數繪圖時：

```python
plt.plot(rewards)
```

![原始進度](../../../../translated_images/zh-HK/train_progress_raw.2adfdf2daea09c59.webp)

從該圖無法看出什麼，因為隨機訓練過程的長度大不相同。為讓圖形更有意義，我們計算一系列實驗的<strong>移動平均</strong>，譬如 100 次。可使用 `np.convolve` 方便地實現：（代碼塊 12）

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![訓練進度](../../../../translated_images/zh-HK/train_progress_runav.c71694a8fa9ab359.webp)

## 變化超參數

為使學習更穩定，調整訓練過程中的部分超參數是合理的。特別是：

- <strong>學習率</strong> `alpha`，我們可以先設為接近1的值，然後不斷減小。隨著時間推移，Q-Table 中概率值會越來越好，我們只需稍作調整而非完全更新。

- **增加 epsilon**。我們可能想慢慢增大 `epsilon`，逐步減少探索、增多利用。大概先用較低 `epsilon` 值，逐漸增大到接近1較為合理。

> **任務1**: 嘗試調整超參數值，看是否能達到更高累積獎勵。你能超過195嗎？


> **任務 2**：正式解決此問題，您需要在連續 100 次運行中取得平均獎勵 195。訓練期間測量該值，並確保您已正式解決此問題！

## 看到結果的實際表現

實際看看訓練好的模型如何表現會很有趣。讓我們運行模擬，並且遵循與訓練時相同的動作選擇策略，根據 Q-Table 中的機率分布進行抽樣：(程式碼區塊 13)

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

你應該會見到類似以下的畫面：

![一個平衡的倒立擺小車](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀挑戰

> **任務 3**：這裡我們使用的是最終版的 Q-Table，但這可能不是最佳版本。請記得我們已將表現最佳的 Q-Table 儲存在 `Qbest` 變數中！嘗試用這個表現最佳的 Q-Table 來做同樣的範例，把 `Qbest` 複製到 `Q`，並看看你是否能注意到不同。

> **任務 4**：這裡我們並非在每一步都選擇最佳動作，而是依照對應的機率分布進行抽樣。如果總是選擇 Q-Table 值最高的最佳動作會不會更合理？這可以透過 `np.argmax` 函數找出對應最高 Q-Table 值的動作編號來實現。請實作此策略，看看是否能改善平衡效果。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 作業
[訓練一輛山地車](assignment.md)

## 結論

我們現已學會如何透過提供定義遊戲目標狀態的獎勵函數，並讓智能體有機會聰明地探索搜尋空間，來訓練智能代理達成良好表現。我們成功地在離散及連續環境（但動作為離散）情境中應用了 Q-Learning 演算法。

同時也很重要去研究動作狀態也是連續的情況，還有當觀測空間更加複雜，例如來自 Atari 遊戲螢幕的影像。這些問題通常需要使用更強大的機器學習技術，如神經網絡，才能達到良好表現。這些更進階的主題將會是我們即將開設的進階 AI 課程內容。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->