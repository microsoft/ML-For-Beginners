<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-06T09:28:34+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "ja"
}
-->
# ARIMAによる時系列予測

前のレッスンでは、時系列予測について少し学び、一定期間にわたる電力負荷の変動を示すデータセットを読み込みました。

[![ARIMAの概要](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMAの概要")

> 🎥 上の画像をクリックすると動画が再生されます: ARIMAモデルの簡単な紹介。例はRで行われていますが、概念は普遍的です。

## [講義前のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## はじめに

このレッスンでは、[ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)を使用してモデルを構築する具体的な方法を学びます。ARIMAモデルは、[非定常性](https://wikipedia.org/wiki/Stationary_process)を示すデータに特に適しています。

## 基本概念

ARIMAを使用するためには、いくつかの重要な概念を理解しておく必要があります。

- 🎓 **定常性**: 統計的な観点から、定常性とは時間が経過しても分布が変化しないデータを指します。一方、非定常データはトレンドによる変動を示し、分析するためには変換が必要です。例えば季節性はデータに変動をもたらし、「季節差分」を行うことで排除できます。

- 🎓 **[差分](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**: 差分とは、非定常データを定常データに変換するプロセスを指します。これにより、データの非定常なトレンドが除去されます。「差分は時系列のレベル変化を除去し、トレンドや季節性を排除することで時系列の平均を安定化させます。」[Shixiongらの論文](https://arxiv.org/abs/1904.07632)

## 時系列におけるARIMAの役割

ARIMAの各部分を分解して、時系列データをモデル化し、予測を行う方法を理解しましょう。

- **AR - 自己回帰 (AutoRegressive)**: 自己回帰モデルは名前の通り、過去のデータを分析して仮定を立てます。これらの過去の値は「ラグ」と呼ばれます。例えば、鉛筆の月別販売データがある場合、各月の販売総数はデータセット内の「進化する変数」として扱われます。このモデルは「関心のある進化する変数が自身のラグ（つまり過去の値）に回帰する」として構築されます。[wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - 統合 (Integrated)**: ARIMAの「I」は、データが差分処理を通じて非定常性を排除することで「統合」されることを指します。これはARMAモデルとの違いです。

- **MA - 移動平均 (Moving Average)**: [移動平均](https://wikipedia.org/wiki/Moving-average_model)は、現在および過去のラグ値を観察することで出力変数を決定するモデルの側面を指します。

要するに、ARIMAは時系列データの特殊な形式にできるだけ近づけるようにモデルを構築するために使用されます。

## 演習 - ARIMAモデルを構築する

このレッスンの[_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working)フォルダーを開き、[_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb)ファイルを見つけてください。

1. ノートブックを実行してPythonライブラリ`statsmodels`を読み込みます。これはARIMAモデルに必要です。

1. 必要なライブラリを読み込む

1. 次に、データのプロットに便利なライブラリをいくつか読み込みます:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. `/data/energy.csv`ファイルからデータをPandasデータフレームに読み込み、内容を確認します:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. 2012年1月から2014年12月までのエネルギーデータをすべてプロットします。前のレッスンでこのデータを見たので驚きはないはずです:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    では、モデルを構築してみましょう！

### トレーニングとテストデータセットを作成する

データが読み込まれたので、トレーニングセットとテストセットに分割します。トレーニングセットでモデルを訓練し、訓練後にテストセットを使用してモデルの精度を評価します。トレーニングセットが未来の時間帯の情報を取得しないように、テストセットがトレーニングセットより後の期間をカバーする必要があります。

1. 2014年9月1日から10月31日までの2か月間をトレーニングセットに割り当てます。テストセットには2014年11月1日から12月31日までの2か月間を含めます:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    このデータはエネルギーの1日ごとの消費量を反映しているため、強い季節的パターンがありますが、消費量は最近の日々の消費量に最も類似しています。

1. 差異を視覚化します:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![トレーニングとテストデータ](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    したがって、データをトレーニングするために比較的小さな時間枠を使用するだけで十分です。

    > 注意: ARIMAモデルをフィットさせる関数はフィッティング中にインサンプル検証を使用するため、検証データは省略します。

### トレーニングのためのデータ準備

次に、データをフィルタリングしてスケーリングすることでトレーニングの準備をします。必要な期間と列のみを含むようにデータセットをフィルタリングし、データが0から1の範囲に投影されるようにスケーリングします。

1. 元のデータセットをフィルタリングして、前述の期間ごとのセットと必要な列「load」と日付のみを含むようにします:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    データの形状を確認できます:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. データを(0, 1)の範囲にスケーリングします。

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. 元のデータとスケーリングされたデータを視覚化します:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![元のデータ](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > 元のデータ

    ![スケーリングされたデータ](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > スケーリングされたデータ

1. スケーリングされたデータを調整したので、テストデータもスケーリングします:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMAの実装

いよいよARIMAを実装する時です！先ほどインストールした`statsmodels`ライブラリを使用します。

次の手順を実行します:

1. モデルを定義するために`SARIMAX()`を呼び出し、モデルパラメータ（p, d, qおよびP, D, Q）を渡します。
2. トレーニングデータに対してモデルを準備するために`fit()`関数を呼び出します。
3. `forecast()`関数を呼び出して予測を行い、予測するステップ数（ホライゾン）を指定します。

> 🎓 これらのパラメータは何のためにあるのでしょうか？ARIMAモデルには、時系列の主要な側面（季節性、トレンド、ノイズ）をモデル化するために使用される3つのパラメータがあります:

`p`: モデルの自己回帰部分に関連するパラメータで、過去の値を組み込みます。
`d`: モデルの統合部分に関連するパラメータで、時系列に適用する差分の量を決定します。
`q`: モデルの移動平均部分に関連するパラメータ。

> 注意: データに季節的な側面がある場合（このデータにはあります）、季節的ARIMAモデル（SARIMA）を使用します。その場合、`p`, `d`, `q`に対応する季節的なコンポーネントを記述するために`P`, `D`, `Q`という別のパラメータセットを使用します。

1. まず、ホライゾン値を設定します。3時間を試してみましょう:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    ARIMAモデルのパラメータの最適な値を選択するのは主観的で時間がかかる場合があります。[`pyramid`ライブラリ](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html)の`auto_arima()`関数を使用することを検討してもよいでしょう。

1. とりあえず手動でいくつかの選択を試して良いモデルを見つけてみましょう。

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    結果の表が表示されます。

最初のモデルを構築しました！次に、モデルを評価する方法を見つける必要があります。

### モデルの評価

モデルを評価するには、いわゆる`ウォークフォワード`検証を実行します。実際には、時系列モデルは新しいデータが利用可能になるたびに再トレーニングされます。これにより、各タイムステップで最適な予測を行うことができます。

この技術を使用して時系列の最初から始め、トレーニングデータセットでモデルを訓練します。その後、次のタイムステップで予測を行います。予測は既知の値と比較して評価されます。トレーニングセットは既知の値を含むように拡張され、このプロセスが繰り返されます。

> 注意: トレーニングセットのウィンドウを固定して効率的なトレーニングを行うことをお勧めします。新しい観測値をトレーニングセットに追加するたびに、セットの最初の観測値を削除します。

このプロセスは、モデルが実際にどのように機能するかをより堅牢に推定します。ただし、多くのモデルを作成する計算コストがかかります。データが小さい場合やモデルが単純な場合は許容されますが、大規模な場合は問題になる可能性があります。

ウォークフォワード検証は時系列モデル評価のゴールドスタンダードであり、独自のプロジェクトで推奨されます。

1. まず、各ホライゾンステップに対してテストデータポイントを作成します。

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    データはホライゾンポイントに応じて水平にシフトされます。

1. テストデータに対してスライディングウィンドウアプローチを使用して予測を行います。ループのサイズはテストデータの長さです:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    トレーニングが進行している様子を確認できます:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. 予測値を実際の負荷と比較します:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    出力
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    時間ごとのデータの予測値を実際の負荷と比較します。この精度はどの程度でしょうか？

### モデルの精度を確認する

モデルの精度を確認するには、すべての予測に対して平均絶対誤差率（MAPE）をテストします。
> **🧮 数学を見てみよう**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) は、上記の式で定義される比率として予測精度を示すために使用されます。実際の値と予測値の差を実際の値で割ります。
>
> 「この計算における絶対値は、予測されたすべての時点で合計され、フィットされた点の数 n で割られます。」 [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. コードで方程式を表現する:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. 1ステップのMAPEを計算する:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    1ステップ予測のMAPE:  0.5570581332313952 %

1. マルチステップ予測のMAPEを表示する:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    低い数値が理想的です: MAPEが10の場合、予測が10%ずれていることを意味します。

1. しかし、いつものように、このような精度の測定は視覚的に確認する方が簡単です。では、プロットしてみましょう:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![時系列モデル](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 とても良いプロットですね。精度の高いモデルを示しています。素晴らしい！

---

## 🚀チャレンジ

時系列モデルの精度をテストする方法を掘り下げてみましょう。このレッスンではMAPEについて触れましたが、他に使用できる方法はありますか？調査して注釈を付けてみてください。役立つドキュメントは[こちら](https://otexts.com/fpp2/accuracy.html)にあります。

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

このレッスンではARIMAを使用した時系列予測の基本のみを扱っています。時間をかけて知識を深め、[このリポジトリ](https://microsoft.github.io/forecasting/)とそのさまざまなモデルタイプを調べて、時系列モデルを構築する他の方法を学んでみてください。

## 課題

[新しいARIMAモデル](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当社は一切の責任を負いません。