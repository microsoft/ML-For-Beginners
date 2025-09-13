<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T09:37:28+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "ja"
}
-->
# 料理分類器 2

この第2回目の分類レッスンでは、数値データを分類するさまざまな方法を探ります。また、どの分類器を選ぶかによる影響についても学びます。

## [講義前のクイズ](https://ff-quizzes.netlify.app/en/ml/)

### 前提条件

前回のレッスンを完了し、`data`フォルダー内にクリーンアップされたデータセット _cleaned_cuisines.csv_ があることを前提としています。このフォルダーは4つのレッスンが含まれるルートフォルダーです。

### 準備

クリーンアップされたデータセットを使用して、_notebook.ipynb_ ファイルをロードしました。そして、データをXとyのデータフレームに分割し、モデル構築の準備を整えています。

## 分類マップ

以前、Microsoftのチートシートを使ってデータを分類する際のさまざまな選択肢について学びました。Scikit-learnは、さらに詳細なチートシートを提供しており、分類器（別名：推定器）を絞り込むのに役立ちます。

![Scikit-learnのMLマップ](../../../../4-Classification/3-Classifiers-2/images/map.png)
> ヒント: [このマップをオンラインで見る](https://scikit-learn.org/stable/tutorial/machine_learning_map/)と、ドキュメントをクリックして読むことができます。

### 計画

このマップはデータをしっかり理解した後に非常に役立ちます。マップの道筋をたどりながら決定を下すことができます。

- サンプル数が50以上
- カテゴリを予測したい
- ラベル付きデータがある
- サンプル数が10万未満
- ✨ Linear SVCを選択可能
- うまくいかない場合、数値データがあるので
    - ✨ KNeighbors Classifierを試す
      - それでもうまくいかない場合、✨ SVCや✨ Ensemble Classifiersを試す

この道筋は非常に役立ちます。

## 演習 - データを分割する

この道筋に従い、まず使用するライブラリをインポートするところから始めましょう。

1. 必要なライブラリをインポートする:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. トレーニングデータとテストデータを分割する:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC分類器

サポートベクトルクラスタリング（SVC）は、サポートベクトルマシン（SVM）技術の一部です。この方法では、ラベルをクラスタリングするための「カーネル」を選択できます。「C」パラメータは「正則化」を指し、パラメータの影響を調整します。カーネルは[いくつかの種類](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)がありますが、ここでは「線形」に設定してLinear SVCを活用します。確率はデフォルトで「false」ですが、ここでは確率推定を得るために「true」に設定します。ランダム状態を「0」に設定してデータをシャッフルし、確率を取得します。

### 演習 - Linear SVCを適用する

まず分類器の配列を作成します。テストを進めるにつれて、この配列に追加していきます。

1. Linear SVCを追加する:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVCを使用してモデルをトレーニングし、レポートを出力する:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    結果はかなり良好です:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## K-Neighbors分類器

K-Neighborsは、MLの「近傍」ファミリーの一部であり、教師あり学習と教師なし学習の両方に使用できます。この方法では、事前に定義されたポイントを作成し、その周囲にデータを集めて一般化されたラベルを予測します。

### 演習 - K-Neighbors分類器を適用する

前の分類器は良好で、データにうまく適合しましたが、精度をさらに向上させる可能性があります。K-Neighbors分類器を試してみましょう。

1. Linear SVC項目の後にコンマを追加し、次の行を追加する:

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    結果は少し悪化しました:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ [K-Neighborsについて学ぶ](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## サポートベクトル分類器

サポートベクトル分類器は、分類と回帰タスクに使用される[サポートベクトルマシン](https://wikipedia.org/wiki/Support-vector_machine)ファミリーの一部です。SVMは「トレーニング例を空間内のポイントにマッピング」し、2つのカテゴリ間の距離を最大化します。その後、データをこの空間にマッピングしてカテゴリを予測します。

### 演習 - サポートベクトル分類器を適用する

サポートベクトル分類器を使用して、さらに良い精度を目指しましょう。

1. K-Neighbors項目の後にコンマを追加し、次の行を追加する:

    ```python
    'SVC': SVC(),
    ```

    結果は非常に良好です！

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ [サポートベクトルについて学ぶ](https://scikit-learn.org/stable/modules/svm.html#svm)

## アンサンブル分類器

前回のテストは非常に良好でしたが、最後まで道筋をたどってみましょう。ランダムフォレストとAdaBoostという「アンサンブル分類器」を試してみます。

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

結果は非常に良好で、特にランダムフォレストが優れています:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ [アンサンブル分類器について学ぶ](https://scikit-learn.org/stable/modules/ensemble.html)

この機械学習の方法は「複数の基本推定器の予測を組み合わせる」ことでモデルの品質を向上させます。この例では、ランダムツリーとAdaBoostを使用しました。

- [ランダムフォレスト](https://scikit-learn.org/stable/modules/ensemble.html#forest)は平均化法であり、ランダム性を加えた「決定木」の「森」を構築して過学習を防ぎます。n_estimatorsパラメータは木の数を設定します。

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)はデータセットに分類器を適用し、その分類器のコピーを同じデータセットに適用します。誤分類された項目の重みを調整し、次の分類器の適合を改善します。

---

## 🚀チャレンジ

これらの技術には多くのパラメータがあり、それぞれを調整することができます。各技術のデフォルトパラメータを調査し、これらのパラメータを調整することでモデルの品質にどのような影響があるかを考えてみましょう。

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

これらのレッスンには専門用語が多く含まれているため、[この用語集](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott)を確認してみてください！

## 課題

[パラメータ調整](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当方は一切の責任を負いません。