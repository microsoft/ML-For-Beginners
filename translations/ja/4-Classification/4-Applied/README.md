<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ad2cf19d7490247558d20a6a59650d13",
  "translation_date": "2025-09-03T23:55:09+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "ja"
}
-->
# 料理推薦ウェブアプリを作成しよう

このレッスンでは、これまでのレッスンで学んだ技術を活用し、美味しい料理データセットを使用して分類モデルを構築します。また、保存したモデルを使用する小さなウェブアプリを作成し、Onnxのウェブランタイムを活用します。

機械学習の最も実用的な用途の1つは推薦システムの構築です。今日はその第一歩を踏み出しましょう！

[![このウェブアプリを紹介](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 上の画像をクリックすると動画が再生されます: Jen Looperが分類された料理データを使用してウェブアプリを構築します

## [講義前のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/)

このレッスンで学ぶこと:

- モデルを構築し、Onnxモデルとして保存する方法
- Netronを使用してモデルを確認する方法
- ウェブアプリでモデルを使用して推論を行う方法

## モデルを構築しよう

応用機械学習システムを構築することは、これらの技術をビジネスシステムに活用する重要な部分です。Onnxを使用することで、ウェブアプリケーション内でモデルを使用し、必要に応じてオフライン環境でも利用できます。

[前のレッスン](../../3-Web-App/1-Web-App/README.md)では、UFO目撃情報に関する回帰モデルを構築し、「ピクル化」してFlaskアプリで使用しました。このアーキテクチャは非常に有用ですが、フルスタックのPythonアプリであり、要件によってはJavaScriptアプリケーションを使用する必要がある場合があります。

このレッスンでは、推論のための基本的なJavaScriptベースのシステムを構築します。ただし、まずモデルをトレーニングし、Onnxで使用できるように変換する必要があります。

## 演習 - 分類モデルをトレーニングする

まず、以前使用したクリーンな料理データセットを使用して分類モデルをトレーニングします。

1. 便利なライブラリをインポートします:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    '[skl2onnx](https://onnx.ai/sklearn-onnx/)'を使用して、Scikit-learnモデルをOnnx形式に変換します。

1. 次に、以前のレッスンと同様に、`read_csv()`を使用してCSVファイルを読み込みます:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. 最初の2つの不要な列を削除し、残りのデータを'X'として保存します:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. ラベルを'y'として保存します:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### トレーニングルーチンを開始する

'SVC'ライブラリを使用します。このライブラリは精度が良好です。

1. Scikit-learnから適切なライブラリをインポートします:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. トレーニングセットとテストセットを分割します:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. 前のレッスンと同様にSVC分類モデルを構築します:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. 次に、`predict()`を呼び出してモデルをテストします:

    ```python
    y_pred = model.predict(X_test)
    ```

1. 分類レポートを出力してモデルの品質を確認します:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    以前見たように、精度は良好です:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### モデルをOnnxに変換する

適切なテンソル数で変換を行うことを確認してください。このデータセットには380の材料がリストされているため、`FloatTensorType`でその数を記載する必要があります。

1. 380のテンソル数を使用して変換します。

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. **model.onnx**というファイルとして保存します:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > 注意: 変換スクリプトで[オプション](https://onnx.ai/sklearn-onnx/parameterized.html)を渡すことができます。この場合、'nocl'をTrueに、'zipmap'をFalseに設定しました。このモデルは分類モデルであるため、辞書のリストを生成するZipMapを削除するオプションがあります（不要です）。`nocl`はモデルにクラス情報を含めるかどうかを指します。`nocl`をTrueに設定してモデルのサイズを縮小します。

ノートブック全体を実行すると、Onnxモデルが構築され、このフォルダに保存されます。

## モデルを確認する

OnnxモデルはVisual Studio Codeではあまり視覚的に確認できませんが、多くの研究者が使用している非常に優れた無料ソフトウェアがあります。 [Netron](https://github.com/lutzroeder/Netron)をダウンロードし、model.onnxファイルを開きます。380の入力と分類器がリストされたシンプルなモデルが視覚化されているのが確認できます:

![Netronの視覚化](../../../../translated_images/netron.a05f39410211915e0f95e2c0e8b88f41e7d13d725faf660188f3802ba5c9e831.ja.png)

Netronはモデルを確認するための便利なツールです。

これで、この便利なモデルをウェブアプリで使用する準備が整いました。冷蔵庫を見て、残りの材料の組み合わせでどの料理が作れるかをモデルで判断するアプリを作りましょう。

## 推薦ウェブアプリを構築する

モデルを直接ウェブアプリで使用できます。このアーキテクチャにより、ローカルで実行したり、必要に応じてオフラインでも使用できます。`model.onnx`ファイルを保存した同じフォルダに`index.html`ファイルを作成することから始めます。

1. このファイル _index.html_ に以下のマークアップを追加します:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. 次に、`body`タグ内で、材料を反映するチェックボックスのリストを表示するための少しのマークアップを追加します:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    各チェックボックスには値が設定されています。これはデータセットに基づいて材料が見つかるインデックスを反映しています。例えば、リンゴはこのアルファベット順のリストで5番目の列にあり、カウントが0から始まるため値は'4'です。[材料スプレッドシート](../../../../4-Classification/data/ingredient_indexes.csv)を参照して特定の材料のインデックスを確認できます。

    index.htmlファイルで作業を続け、最後の閉じタグ`</div>`の後にスクリプトブロックを追加します。

1. まず、[Onnx Runtime](https://www.onnxruntime.ai/)をインポートします:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtimeは、幅広いハードウェアプラットフォームでOnnxモデルを実行できるようにするために使用され、最適化やAPIを提供します。

1. Runtimeが設定されたら、呼び出します:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

このコードでは、以下のことが行われています:

1. 380の可能な値（1または0）の配列を作成し、選択された材料チェックボックスに応じてモデルに送信します。
2. チェックボックスの配列を作成し、アプリケーション開始時に呼び出される`init`関数でチェックされているかどうかを判断します。チェックボックスがチェックされると、`ingredients`配列が選択された材料を反映するように変更されます。
3. チェックボックスがチェックされているかどうかを確認する`testCheckboxes`関数を作成します。
4. ボタンが押されたときに`startInference`関数を使用し、チェックボックスがチェックされている場合は推論を開始します。
5. 推論ルーチンには以下が含まれます:
   1. モデルの非同期ロードを設定
   2. モデルに送信するテンソル構造を作成
   3. トレーニング時に作成した`float_input`入力を反映する'feeds'を作成（Netronで名前を確認できます）
   4. これらの'feeds'をモデルに送信し、応答を待つ

## アプリケーションをテストする

Visual Studio Codeで`index.html`ファイルがあるフォルダでターミナルセッションを開きます。[http-server](https://www.npmjs.com/package/http-server)がグローバルにインストールされていることを確認し、プロンプトで`http-server`と入力します。ローカルホストが開き、ウェブアプリを表示できます。さまざまな材料に基づいてどの料理が推薦されるかを確認してください:

![材料ウェブアプリ](../../../../translated_images/web-app.4c76450cabe20036f8ec6d5e05ccc0c1c064f0d8f2fe3304d3bcc0198f7dc139.ja.png)

おめでとうございます！いくつかのフィールドを持つ「推薦」ウェブアプリを作成しました。このシステムを構築する時間を取ってみてください！

## 🚀チャレンジ

ウェブアプリは非常にシンプルなので、[ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv)データの材料とそのインデックスを使用してさらに構築してください。どの味の組み合わせが特定の国の料理を作るのに適しているでしょうか？

## [講義後のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/)

## 復習と自己学習

このレッスンでは、食品材料の推薦システムを作成することの有用性に触れただけですが、この分野の機械学習アプリケーションには非常に多くの例があります。これらのシステムがどのように構築されるかについてさらに読んでみてください:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## 課題

[新しい推薦システムを構築する](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確さが含まれる可能性があります。元の言語で記載された原文が正式な情報源と見なされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤認について、当社は一切の責任を負いません。