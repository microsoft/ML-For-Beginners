# CartPole スケーティング

前回のレッスンで解いてきた問題は、おもちゃの問題のように見え、現実のシナリオにはあまり適用できないように思えるかもしれません。しかし、多くの実世界の問題もこのシナリオを共有しています。チェスや囲碁も含まれます。同様なのは、与えられたルールのあるボードと<strong>離散的な状態</strong>があるためです。

## [プレ講義クイズ](https://ff-quizzes.netlify.app/en/ml/)

## はじめに

このレッスンでは、Qラーニングの同じ原則を<strong>連続状態</strong>、すなわち1つ以上の実数で表される状態に適用します。取り扱うのは以下の問題です：

> <strong>問題</strong>: ピーターがオオカミから逃げるには、より速く動ける必要があります。ピーターがQラーニングを使ってスケートの仕方、特にバランスを保つ方法を学ぶ様子を見ていきましょう。

![華麗なる脱出！](../../../../translated_images/ja/escape.18862db9930337e3.webp)

> ピーターと友達が創意工夫してオオカミから逃げる！画像提供：[Jen Looper](https://twitter.com/jenlooper)

バランスを取る単純化されたバージョンとして、**CartPole** 問題を使います。カートポールの世界では、水平方向に左右に動けるスライダーがあり、その上に垂直な棒をバランスさせることがゴールです。

<img alt="カートポール" src="../../../../translated_images/ja/cartpole.b5609cc0494a14f7.webp" width="200"/>

## 前提条件

このレッスンでは、**OpenAI Gym** と呼ばれるライブラリを使い、異なる<strong>環境</strong>をシミュレートします。コードをローカル（例：Visual Studio Code）で実行する場合、シミュレーションは新しいウィンドウで開きます。オンラインで実行する場合は、[こちら](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7)の説明に従いコードを一部調整する必要があるかもしれません。

## OpenAI Gym

前回のレッスンでは、ゲームのルールと状態は自分で定義した `Board` クラスが担当していました。ここでは、バランス棒の物理をシミュレートする特別な<strong>シミュレーション環境</strong>を使います。強化学習アルゴリズムの訓練用シミュレーション環境で最も人気があるものの一つが[Gym](https://gym.openai.com/)で、[OpenAI](https://openai.com/)が管理しています。このGymを使うことで、カートポールシミュレーションからアタリのゲームまで様々な<strong>環境</strong>を作成できます。

> <strong>注意</strong>: OpenAI Gymで利用可能な他の環境は[こちら](https://gym.openai.com/envs/#classic_control)で確認できます。

まず、gymをインストールし必要なライブラリをインポートしましょう（コードブロック1）：

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## 演習 - カートポール環境の初期化

カートポール問題に取り組むには、対応する環境を初期化する必要があります。各環境には以下が関連しています：

- <strong>観測空間</strong>: 環境から受け取る情報の構造を定義します。カートポール問題では、棒の位置、速度などの値です。

- <strong>行動空間</strong>: 可能な行動を定義します。我々の場合、行動空間は離散的で、<strong>左</strong>か<strong>右</strong>の2つの行動からなります。（コードブロック2）

1. 初期化するには、以下のコードを入力してください：

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

環境がどう動くかを見るために、100ステップの短いシミュレーションをします。各ステップで実行する行動を1つ選択します。このシミュレーションでは `action_space` からランダムに選びます。

1. 下のコードを実行して結果を確認してください。

    ✅ このコードはローカルのPython環境で実行するのが推奨です！ （コードブロック3）

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    画面はこの画像に似たものが見えるはずです：

    ![バランスを取っていないカートポール](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. シミュレーション中、どのように行動するか決めるため観測情報を得る必要があります。`step`関数は現在の観測値、報酬値、そして続けるべきか示す`done`フラグを返します：（コードブロック4）

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    ノートブックの出力には以下のようなものが表示されるでしょう：

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

    シミュレーション各ステップで返される観測ベクトルには次の値が含まれます：
    - カートの位置
    - カートの速度
    - 棒の角度
    - 棒の回転速度

1. それらの値の最小・最大値を取得してみましょう：（コードブロック5）

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    シミュレーション各ステップの報酬値は常に1であることに気づくでしょう。これは目標ができるだけ長く生存し、棒を垂直に近い状態で保ち続けることだからです。

    ✅ 実際、CartPoleシミュレーションは連続して100回の試行で平均報酬が195以上になれば解決済みとされます。

## 状態の離散化

Qラーニングでは、状態ごとに何をすべきかを定義したQテーブルを作る必要があります。そのためには、状態を<strong>離散的</strong>に、つまり有限の離散値の数を持つようにしなければなりません。そこで観測を有限の状態集合に<strong>離散化</strong>してマッピングします。

いくつか方法があります：

- <strong>ビン分割</strong>：特定値の区間が分かっていれば、その区間をいくつかの<strong>ビン</strong>に分割し、値を属するビン番号で置き換えます。これは numpy の [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) メソッドを使って行うことができます。これにより状態サイズは正確にわかります。なぜなら選んだビンの数に依存するからです。
  
✅ 線形補間を使って値をある有限区間（例えば -20 から 20）におさめ、その後整数に丸める方法もあります。これにより状態のサイズは少し制御しづらくなります。特に入力値の範囲が正確にわからない場合に顕著です。例として私たちの場合、4つの値のうち2つは上下限がなく、無限の状態数が生じる可能性があります。

私たちの例では後者の方法を採用します。のちに気づくと思いますが、上限や下限が不定でも値はめったに一定の有限区間外に出ないため、極端な値の状態は稀です。

1. 観測から4つの整数値のタプルを返す関数はこちらです：（コードブロック6）

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. もう一つビン分割を使った離散化方法も試してみましょう：（コードブロック7）

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # 各パラメータの値の区間
    nbins = [20,20,10,10] # 各パラメータのビンの数
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. さて短いシミュレーションを実行し、この離散化された環境値を観察しましょう。`discretize` と `discretize_bins` の両方を試し、違いを見てみても構いません。

    ✅ discretize_bins はビン番号（0から始まる）を返します。0近辺の入力値には区間の中央付近の番号（10）を返します。discretizeは出力値の範囲を気にせず負の値も許すため状態値はシフトされておらず、0 は 0に対応します。（コードブロック8）

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

    ✅ 環境の動作を見たい場合は env.render で始まる行のコメントを外してください。そうでなければ背景で実行すれば速いです。Qラーニング過程ではこの「非表示」実行を使います。

## Qテーブルの構造

前回のレッスンでは状態が0〜8の単純な数字の組だったので、Qテーブルは8x8x2のnumpyテンソルで表せました。ビン分割離散化を使う場合も状態ベクトルのサイズは既知なので、同じやり方で状態を20x20x10x10x2の配列で表せます（ここで2は行動空間の次元、最初の4つは観測空間パラメータごとの選択したビンの数です）。

しかし観測空間の正確なサイズが不明なこともあります。`discretize` 関数の場合、状態が一定範囲に収まる保証はなく、元の値の一部に制約がありません。そこで少し違う方法として、Qテーブルを辞書で表現します。

1. *(状態, 行動)* のペアを辞書のキーにし、値がQテーブルのエントリ値となるようにします。（コードブロック9）

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    ここでは `qvalues()` 関数も定義しており、指定状態の全可能な行動に対応するQテーブルの値リストを返します。もしエントリがなければ0を返します。

## Qラーニングを始めよう

さあ、ピーターにバランスを教えます！

1. まずいくつかのハイパーパラメータを決めましょう：（コードブロック10）

    ```python
    # ハイパーパラメータ
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    ここで、`alpha` は<strong>学習率</strong>で、各ステップでQテーブルの値をどれだけ調整するかを決めます。前回は1から始め、訓練中に下げていきました。この例では単純さのために一定にしてありますが、後で調整してみてもいいです。

    `gamma` は<strong>割引率</strong>で、将来の報酬をどれだけ重視するかを示します。

    `epsilon` は<strong>探索/活用の係数</strong>で、探索と活用のどちらを優先するか決めます。アルゴリズムでは `epsilon` の割合でQテーブルに基づく行動を選び、残りはランダム行動を実行します。これにより未探索の領域も探索できます。

    ✅ バランスの観点ではランダム行動（探索）は不適切な方向への偶発的な衝撃であり、棒はこれらの「ミス」からバランスを取ることを学習します。

### アルゴリズムの改善

以前のレッスンから2つの改善が可能です：

- <strong>累積報酬の平均を計算</strong>し、複数のシミュレーションの間隔ごとに進捗を5000イテレーションごとに出力します。この期間の累積報酬の平均を出します。これにより195ポイント以上が得られれば問題解決と考えられます。
  
- <strong>最大平均累積結果</strong> `Qmax` を計算し、その結果に対応するQテーブルを保存します。学習中に平均値が下がり始めることがありますが、最高モデルの値を保持しておきたいからです。

1. シミュレーションごとの累積報酬を `rewards` ベクターに集め、後でプロットします。（コードブロック11）

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
        # == シミュレーションを行う ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # 利用 - Qテーブルの確率に従って行動を選択する
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # 探索 - ランダムに行動を選択する
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == 定期的に結果を表示し、平均報酬を計算する ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

これら結果から得られること：

- <strong>目標に近づいている</strong>。連続100回以上の試行で195の累積報酬を達成しつつあるか、既に達成している可能性もあります。小さい値もありえますが、5000試行に対する平均なので、公的な基準の100回の意味はまだ完全には反映されていません。
  
- <strong>報酬が下がり始めることも</strong>。報酬が下がることは、既に学習したQテーブルの値を悪化させる値で上書きしてしまっている可能性を示します。

この変化は学習の進行をグラフにするとより明確に分かります。

## 学習進行のグラフ化

学習中、各イテレーションで累積報酬を `rewards` ベクターに記録しました。イテレーション番号に対してプロットするとこうなります：

```python
plt.plot(rewards)
```

![生の進捗](../../../../translated_images/ja/train_progress_raw.2adfdf2daea09c59.webp)

このグラフからは何も読み取れません。学習の確率的な性質により学習セッションの長さが大きく変動するためです。そこで実験を連続100回分の<strong>移動平均</strong>をとると意味がわかりやすくなります。これは `np.convolve` を使うと便利です：（コードブロック12）

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![学習進行](../../../../translated_images/ja/train_progress_runav.c71694a8fa9ab359.webp)

## ハイパーパラメータの変化

学習を安定させるためには、訓練中にハイパーパラメータを調整すると良いです。特に：

- <strong>学習率</strong> `alpha` は初期に1に近い値を与え、その後徐々に減らしていきます。時間と共にQテーブル内には良い確率値が定まり、微調整をするべきであり、新しい値で完全に上書きすべきではありません。

- <strong>イプシロンを増加</strong>させる。初期は低い値で、徐々に1に近づけていくことで探索を減らし活用を増やすのが理にかなっています。

> **課題1**: ハイパーパラメータをいじって、より高い累積報酬を目指そう。195を超えられますか？


> **タスク 2**: 問題を正式に解決するには、100回連続の実行で195の平均報酬を得る必要があります。トレーニング中にこれを測定し、問題が正式に解決されたことを確認してください！

## 結果を実際に見る

トレーニングされたモデルがどのように動作するかを実際に見るのは興味深いでしょう。シミュレーションを実行し、トレーニング中と同じ行動選択戦略、つまりQテーブルの確率分布に従ってサンプリングする方法を試してみましょう：(コードブロック 13)

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

このような結果が得られるはずです：

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀チャレンジ

> **タスク 3**: ここでは最終版のQテーブルを使用しましたが、それが最良とは限りません。最も成績の良いQテーブルを `Qbest` 変数に保存していることを覚えておいてください！ `Qbest` を `Q` にコピーして同じ例を試し、違いが分かるか確かめてみましょう。

> **タスク 4**: ここでは各ステップで最良の行動を選択していませんでしたが、対応する確率分布でサンプリングしていました。常にQテーブルの値が最も高い最良の行動だけを選ぶ方が理にかなっているでしょうか？ `np.argmax` 関数を使って最も高いQテーブルの値に対応する行動番号を見つけることでこれを実装できます。この戦略を実装し、バランスの改善が見られるか試してみてください。

## [講義後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## 課題
[マウンテンカーを訓練する](assignment.md)

## 結論

我々は、望ましいゲーム状態を定義する報酬関数を提供し、探索空間を知的に探査する機会を与えるだけで、エージェントをうまく訓練する方法を学びました。Qラーニングアルゴリズムを離散および連続環境のケースに成功裏に適用しましたが、行動は離散的でした。

行動状態も連続的で、観測空間がAtariゲーム画面の画像のようにずっと複雑な状況も学ぶことが重要です。そのような問題では、良好な結果を得るためにニューラルネットワークのようなより強力な機械学習技術を使う必要があることが多いです。これらのより高度なトピックは、今後のより高度なAIコースの主題です。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責事項**：
本書類は AI 翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期していますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知おきください。原文の原語版が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や解釈違いについても、当方は責任を負いかねます。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->