<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-06T09:40:24+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "ja"
}
-->
# 自然言語処理の一般的なタスクと技術

ほとんどの*自然言語処理*タスクでは、処理対象のテキストを分解し、分析し、その結果をルールやデータセットと照合する必要があります。これらのタスクを通じて、プログラマーはテキスト内の単語やフレーズの_意味_や_意図_、または単に_頻度_を導き出すことができます。

## [講義前のクイズ](https://ff-quizzes.netlify.app/en/ml/)

テキストを処理する際に使用される一般的な技術を見てみましょう。これらの技術は機械学習と組み合わせることで、大量のテキストを効率的に分析するのに役立ちます。ただし、これらのタスクに機械学習を適用する前に、自然言語処理の専門家が直面する問題を理解する必要があります。

## 自然言語処理に共通するタスク

テキストを分析する方法はさまざまです。実行できるタスクがあり、それらを通じてテキストを理解し、結論を導き出すことができます。通常、これらのタスクは順序立てて実行されます。

### トークン化

おそらくほとんどの自然言語処理アルゴリズムが最初に行うことは、テキストをトークン、つまり単語に分割することです。一見簡単そうに思えますが、句読点や異なる言語の単語や文の区切りを考慮する必要があるため、複雑になることがあります。区切りを決定するためにさまざまな方法を使用する必要があるかもしれません。

![トークン化](../../../../6-NLP/2-Tasks/images/tokenization.png)
> **Pride and Prejudice**の文をトークン化する様子。インフォグラフィック作成者：[Jen Looper](https://twitter.com/jenlooper)

### 埋め込み

[単語埋め込み](https://wikipedia.org/wiki/Word_embedding)は、テキストデータを数値に変換する方法です。埋め込みは、意味が似ている単語や一緒に使われる単語がクラスター化されるように行われます。

![単語埋め込み](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - **Pride and Prejudice**の文の単語埋め込み。インフォグラフィック作成者：[Jen Looper](https://twitter.com/jenlooper)

✅ [この興味深いツール](https://projector.tensorflow.org/)を試して単語埋め込みを実験してみましょう。単語をクリックすると、似た単語のクラスターが表示されます。例えば、'toy'は'disney', 'lego', 'playstation', 'console'とクラスター化されます。

### 構文解析と品詞タグ付け

トークン化されたすべての単語は、品詞としてタグ付けすることができます。例えば、名詞、動詞、形容詞などです。文`the quick red fox jumped over the lazy brown dog`は、品詞タグ付けされると、fox = 名詞、jumped = 動詞となります。

![構文解析](../../../../6-NLP/2-Tasks/images/parse.png)

> **Pride and Prejudice**の文を解析する様子。インフォグラフィック作成者：[Jen Looper](https://twitter.com/jenlooper)

構文解析は、文中でどの単語が互いに関連しているかを認識することです。例えば、`the quick red fox jumped`は形容詞-名詞-動詞のシーケンスであり、`lazy brown dog`のシーケンスとは別です。

### 単語とフレーズの頻度

大量のテキストを分析する際に役立つ手法の一つは、興味のあるすべての単語やフレーズの辞書を作成し、それがどれだけ頻繁に出現するかを記録することです。例えば、`the quick red fox jumped over the lazy brown dog`というフレーズでは、`the`の頻度は2です。

以下は、単語の頻度を数える例です。ラドヤード・キップリングの詩「The Winners」には次のような一節があります：

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

フレーズの頻度は必要に応じて大文字小文字を区別することもできます。例えば、`a friend`の頻度は2、`the`の頻度は6、`travels`の頻度は2です。

### N-グラム

テキストを一定の長さの単語のシーケンスに分割することができます。1単語（ユニグラム）、2単語（バイグラム）、3単語（トライグラム）、または任意の数の単語（N-グラム）です。

例えば、`the quick red fox jumped over the lazy brown dog`をN-グラムスコア2で分割すると、以下のN-グラムが生成されます：

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

これをスライディングボックスとして文に適用すると視覚的に理解しやすくなります。以下は3単語のN-グラムの場合です。各文でN-グラムが太字で示されています：

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![N-グラムスライディングウィンドウ](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-グラム値3：インフォグラフィック作成者：[Jen Looper](https://twitter.com/jenlooper)

### 名詞句抽出

ほとんどの文には、主語または目的語となる名詞があります。英語では、しばしば`a`、`an`、`the`がその前に付くことで識別できます。文の意味を理解しようとする際に、名詞句を抽出することは自然言語処理で一般的なタスクです。

✅ 文「I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.」では、名詞句を特定できますか？

文`the quick red fox jumped over the lazy brown dog`には2つの名詞句があります：**quick red fox**と**lazy brown dog**。

### 感情分析

文やテキストは、*ポジティブ*または*ネガティブ*な感情を分析することができます。感情は*極性*と*客観性/主観性*で測定されます。極性は-1.0から1.0（ネガティブからポジティブ）、客観性は0.0から1.0（最も客観的から最も主観的）で測定されます。

✅ 後で学ぶように、機械学習を使用して感情を判断する方法はさまざまですが、一つの方法として、人間の専門家がポジティブまたはネガティブに分類した単語やフレーズのリストを使用し、そのモデルをテキストに適用して極性スコアを計算する方法があります。この方法がうまく機能する場合とそうでない場合がある理由を考えてみてください。

### 語形変化

語形変化を使用すると、単語の単数形または複数形を取得することができます。

### レンマ化

*レンマ*は、一連の単語の根本的な単語または見出し語です。例えば、*flew*、*flies*、*flying*のレンマは動詞*fly*です。

自然言語処理研究者にとって便利なデータベースもあります。特に以下が挙げられます：

### WordNet

[WordNet](https://wordnet.princeton.edu/)は、単語、同義語、反意語、その他多くの詳細をさまざまな言語で提供するデータベースです。翻訳、スペルチェッカー、またはあらゆる種類の言語ツールを構築する際に非常に役立ちます。

## 自然言語処理ライブラリ

幸いなことに、これらの技術をすべて自分で構築する必要はありません。自然言語処理や機械学習に特化していない開発者でも利用しやすい優れたPythonライブラリが利用可能です。次のレッスンではこれらの例をさらに詳しく学びますが、ここでは次のタスクに役立ついくつかの便利な例を紹介します。

### 演習 - `TextBlob`ライブラリの使用

TextBlobというライブラリを使用してみましょう。このライブラリには、これらの種類のタスクに取り組むための便利なAPIが含まれています。TextBlobは「[NLTK](https://nltk.org)と[pattern](https://github.com/clips/pattern)の巨人の肩の上に立ち、両者とうまく連携します。」そのAPIにはかなりの量の機械学習が組み込まれています。

> 注：[Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart)ガイドは、経験豊富なPython開発者に推奨されます。

名詞句を特定しようとする際、TextBlobは名詞句を見つけるためのいくつかの抽出オプションを提供しています。

1. `ConllExtractor`を見てみましょう。

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > ここで何が起こっているのでしょうか？[ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor)は「ConLL-2000トレーニングコーパスで学習されたチャンク解析を使用する名詞句抽出器」です。ConLL-2000は、2000年の計算自然言語学習会議を指します。この会議では毎年、難しい自然言語処理の問題に取り組むワークショップが開催され、2000年には名詞チャンク化がテーマでした。モデルはWall Street Journalを基に学習され、「セクション15-18をトレーニングデータ（211727トークン）として使用し、セクション20をテストデータ（47377トークン）として使用」しました。使用された手順は[こちら](https://www.clips.uantwerpen.be/conll2000/chunking/)で確認でき、[結果](https://ifarm.nl/erikt/research/np-chunking.html)も参照できます。

### チャレンジ - NLPを活用してボットを改善する

前のレッスンでは非常にシンプルなQ&Aボットを作成しました。今回は、入力を感情分析してその感情に応じた応答を出力することで、マーヴィンを少し共感的にします。また、`noun_phrase`を特定してそのトピックについてさらに質問する必要があります。

より良い会話型ボットを構築する際の手順：

1. ユーザーにボットとの対話方法を説明する指示を表示
2. ループ開始 
   1. ユーザー入力を受け取る
   2. ユーザーが終了を要求した場合は終了
   3. ユーザー入力を処理し、適切な感情応答を決定
   4. 感情に名詞句が含まれている場合は、それを複数形にしてそのトピックについてさらに質問
   5. 応答を表示
3. ステップ2に戻る

以下はTextBlobを使用して感情を判断するコードスニペットです。感情応答の*グラデーション*は4つしかありません（必要に応じて増やすこともできます）：

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

以下はサンプル出力の例です（ユーザー入力は`>`で始まる行にあります）：

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

タスクの一つの解決策は[こちら](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)にあります。

✅ 知識チェック

1. 共感的な応答は、ボットが実際にユーザーを理解していると思わせることができるでしょうか？
2. 名詞句を特定することで、ボットはより「信じられる」ものになるでしょうか？
3. 文から名詞句を抽出することは、なぜ有用なのでしょうか？

---

前の知識チェックでのボットを実装し、友人にテストしてもらいましょう。それが友人を騙せるかどうか試してみてください。ボットをより「信じられる」ものにすることはできますか？

## 🚀チャレンジ

前の知識チェックでのタスクを実装してみましょう。ボットを友人にテストしてもらいましょう。それが友人を騙せるかどうか試してみてください。ボットをより「信じられる」ものにすることはできますか？

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

次のいくつかのレッスンでは感情分析についてさらに学びます。この興味深い技術について[KDNuggets](https://www.kdnuggets.com/tag/nlp)の記事などを調べてみてください。

## 課題 

[ボットに応答させる](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確さが含まれる可能性があります。元の言語で記載された原文が正式な情報源と見なされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤認について、当社は一切の責任を負いません。