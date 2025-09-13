<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-06T09:29:52+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "ja"
}
-->
# サポートベクター回帰による時系列予測

前回のレッスンでは、ARIMAモデルを使用して時系列予測を行う方法を学びました。今回は、連続データを予測するために使用される回帰モデルであるサポートベクター回帰（Support Vector Regressor）モデルについて学びます。

## [事前クイズ](https://ff-quizzes.netlify.app/en/ml/) 

## はじめに

このレッスンでは、回帰のための[**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine)を使用してモデルを構築する具体的な方法を学びます。これを**SVR: Support Vector Regressor**と呼びます。

### 時系列におけるSVR [^1]

時系列予測におけるSVRの重要性を理解する前に、以下の重要な概念を知っておく必要があります：

- **回帰:** 与えられた入力セットから連続値を予測する教師あり学習技術。特徴空間内で最大数のデータポイントを持つ曲線（または直線）をフィットさせることが目的です。[詳細はこちら](https://en.wikipedia.org/wiki/Regression_analysis)。
- **サポートベクターマシン (SVM):** 分類、回帰、外れ値検出に使用される教師あり機械学習モデルの一種。モデルは特徴空間内のハイパープレーンであり、分類の場合は境界として機能し、回帰の場合は最適なフィットラインとして機能します。SVMでは、カーネル関数を使用してデータセットをより高次元の空間に変換し、分離しやすくします。[詳細はこちら](https://en.wikipedia.org/wiki/Support-vector_machine)。
- **サポートベクター回帰 (SVR):** SVMの一種で、最大数のデータポイントを持つ最適なフィットライン（SVMの場合はハイパープレーン）を見つけることを目的としています。

### なぜSVRなのか？ [^1]

前回のレッスンでは、時系列データを予測するための非常に成功した統計的線形手法であるARIMAについて学びました。しかし、多くの場合、時系列データには*非線形性*が含まれており、線形モデルでは対応できません。そのような場合、回帰タスクにおいてデータの非線形性を考慮するSVMの能力が、時系列予測においてSVRを成功させる要因となります。

## 演習 - SVRモデルを構築する

データ準備の最初のステップは、前回の[ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)レッスンと同じです。

このレッスンの[_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working)フォルダーを開き、[_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb)ファイルを見つけてください。[^2]

1. ノートブックを実行し、必要なライブラリをインポートします: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. `/data/energy.csv`ファイルからデータをPandasデータフレームに読み込み、確認します: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. 2012年1月から2014年12月までの利用可能なエネルギーデータをプロットします: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![全データ](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   では、SVRモデルを構築しましょう。

### トレーニングとテストデータセットを作成する

データが読み込まれたので、トレーニングセットとテストセットに分割します。その後、SVRに必要なタイムステップベースのデータセットを作成するためにデータを再構成します。トレーニングセットでモデルをトレーニングします。トレーニングが完了したら、トレーニングセット、テストセット、そして全データセットでモデルの精度を評価し、全体的なパフォーマンスを確認します。テストセットがトレーニングセットより後の期間をカバーするようにして、モデルが将来の期間から情報を取得しないようにする必要があります[^2]（これを*過学習*と呼びます）。

1. 2014年9月1日から10月31日までの2か月間をトレーニングセットに割り当てます。テストセットには2014年11月1日から12月31日までの2か月間を含めます: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. 違いを可視化します: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![トレーニングとテストデータ](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### トレーニングのためのデータ準備

次に、データをフィルタリングしてスケーリングすることでトレーニングの準備をします。必要な期間と列のみを含むようにデータセットをフィルタリングし、データを0から1の範囲に投影するようにスケーリングします。

1. 元のデータセットをフィルタリングして、前述の期間ごとのセットと必要な列「load」と日付のみを含むようにします: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. トレーニングデータを(0, 1)の範囲にスケーリングします: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. 次に、テストデータをスケーリングします: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### タイムステップを使用したデータ作成 [^1]

SVRでは、入力データを`[batch, timesteps]`の形式に変換します。そのため、既存の`train_data`と`test_data`を再構成し、新しい次元（タイムステップ）を追加します。

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

この例では、`timesteps = 5`を使用します。つまり、モデルへの入力は最初の4つのタイムステップのデータであり、出力は5番目のタイムステップのデータになります。

```python
timesteps=5
```

ネストされたリスト内包表記を使用してトレーニングデータを2Dテンソルに変換します:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

テストデータを2Dテンソルに変換します:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

トレーニングデータとテストデータから入力と出力を選択します:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### SVRの実装 [^1]

次に、SVRを実装します。この実装について詳しく知りたい場合は、[こちらのドキュメント](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)を参照してください。今回の実装では以下の手順を行います：

1. `SVR()`を呼び出し、カーネル、ガンマ、C、イプシロンなどのモデルハイパーパラメータを渡してモデルを定義します。
2. `fit()`関数を呼び出してトレーニングデータにモデルを適用します。
3. `predict()`関数を呼び出して予測を行います。

ここでは[RBFカーネル](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel)を使用し、ハイパーパラメータのガンマ、C、イプシロンをそれぞれ0.5、10、0.05に設定します。

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### トレーニングデータにモデルを適用 [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### モデル予測を行う [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

SVRを構築しました！次に評価を行います。

### モデルを評価する [^1]

評価では、まずデータを元のスケールに戻します。その後、性能を確認するために、元の時系列データと予測データのプロットを作成し、MAPE結果を出力します。

予測値と元の出力をスケールバックします:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### トレーニングデータとテストデータでモデル性能を確認 [^1]

データセットからタイムスタンプを抽出し、プロットのx軸に表示します。最初の```timesteps-1```値を最初の出力の入力として使用しているため、出力のタイムスタンプはその後から始まります。

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

トレーニングデータの予測をプロットします:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![トレーニングデータ予測](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

トレーニングデータのMAPEを出力します:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

テストデータの予測をプロットします:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![テストデータ予測](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

テストデータのMAPEを出力します:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 テストデータセットで非常に良い結果が得られました！

### 全データセットでモデル性能を確認 [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![全データ予測](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 素晴らしいプロットで、精度の高いモデルを示しています。よくできました！

---

## 🚀チャレンジ

- モデルを作成する際にハイパーパラメータ（ガンマ、C、イプシロン）を調整し、テストデータで評価して、どのハイパーパラメータセットが最良の結果をもたらすか確認してください。これらのハイパーパラメータについて詳しく知りたい場合は、[こちらのドキュメント](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel)を参照してください。
- モデルに異なるカーネル関数を使用し、それらの性能をデータセットで分析してください。役立つドキュメントは[こちら](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)にあります。
- モデルが予測を行う際に参照する`timesteps`の値を変更してみてください。

## [事後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

このレッスンでは、時系列予測におけるSVRの応用を紹介しました。SVRについてさらに詳しく知りたい場合は、[このブログ](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/)を参照してください。この[scikit-learnのドキュメント](https://scikit-learn.org/stable/modules/svm.html)では、一般的なSVM、[SVR](https://scikit-learn.org/stable/modules/svm.html#regression)、および使用可能な異なる[カーネル関数](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)やそのパラメータなどの実装詳細について包括的な説明が提供されています。

## 課題

[新しいSVRモデル](assignment.md)

## クレジット

[^1]: このセクションのテキスト、コード、および出力は[@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)によって寄稿されました。
[^2]: このセクションのテキスト、コード、および出力は[ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)から引用されました。

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文を公式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当社は一切の責任を負いません。