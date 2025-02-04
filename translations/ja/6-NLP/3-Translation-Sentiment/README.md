# 機械学習を使った翻訳と感情分析

前回のレッスンでは、`TextBlob` を使って基本的なボットを構築する方法を学びました。このライブラリは、名詞句の抽出などの基本的な自然言語処理タスクを実行するために、裏で機械学習を利用しています。計算言語学におけるもう一つの重要な課題は、ある言語から別の言語への正確な _翻訳_ です。

## [講義前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

翻訳は非常に難しい問題であり、これは数千もの言語が存在し、それぞれが非常に異なる文法規則を持つためです。一つのアプローチは、英語のような言語の正式な文法規則を非言語依存の構造に変換し、次に別の言語に変換することです。このアプローチでは、次のステップを踏むことになります：

1. **識別**。入力言語の単語を名詞、動詞などにタグ付けする。
2. **翻訳の作成**。ターゲット言語の形式で各単語の直接的な翻訳を生成する。

### 英語からアイルランド語への例文

英語では、_I feel happy_ という文は次の3つの単語で構成されています：

- **主語** (I)
- **動詞** (feel)
- **形容詞** (happy)

しかし、アイルランド語では、同じ文は非常に異なる文法構造を持っています。感情は「*あなたの上に*」あると表現されます。

英語のフレーズ `I feel happy` をアイルランド語にすると `Tá athas orm` となります。*直訳* すると `Happy is upon me` です。

アイルランド語の話者が英語に翻訳する場合、`I feel happy` と言うでしょう。`Happy is upon me` とは言いません。なぜなら、彼らは文の意味を理解しているからです。単語や文の構造が異なっていてもです。

アイルランド語の文の正式な順序は：

- **動詞** (Tá または is)
- **形容詞** (athas または happy)
- **主語** (orm または upon me)

## 翻訳

単純な翻訳プログラムは、文の構造を無視して単語だけを翻訳するかもしれません。

✅ 第二言語（または第三言語以上）を大人になってから学んだことがある場合、最初は母国語で考え、概念を頭の中で一語ずつ第二言語に翻訳し、それを話すことから始めたかもしれません。これは単純な翻訳コンピュータープログラムが行っていることと似ています。この段階を超えて流暢さを達成することが重要です！

単純な翻訳は、悪い（時には面白い）誤訳を引き起こします。`I feel happy` はアイルランド語では `Mise bhraitheann athas` と直訳されます。これは（直訳すると）`me feel happy` という意味で、有効なアイルランド語の文ではありません。英語とアイルランド語は隣接する島で話される言語ですが、非常に異なる文法構造を持っています。

> アイルランドの言語伝統についてのビデオをいくつか見ることができます。例えば [こちら](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### 機械学習のアプローチ

これまでのところ、自然言語処理の形式的な規則アプローチについて学びました。もう一つのアプローチは、単語の意味を無視し、_代わりにパターンを検出するために機械学習を使用する_ ことです。これには、元の言語とターゲット言語の両方で大量のテキスト（*コーパス*）またはテキスト（*コーパス*）が必要です。

例えば、1813年にジェーン・オースティンによって書かれた有名な英語の小説『高慢と偏見』を考えてみましょう。英語の本とその*フランス語*の人間による翻訳を参照すると、ある言語のフレーズが他の言語に*イディオム的に*翻訳されていることがわかります。すぐにそれを行います。

例えば、英語のフレーズ `I have no money` をフランス語に直訳すると、`Je n'ai pas de monnaie` になるかもしれません。"Monnaie" はフランス語の 'false cognate' で、'money' と 'monnaie' は同義ではありません。人間が行うより良い翻訳は `Je n'ai pas d'argent` で、これはお金がないという意味をよりよく伝えます（'monnaie' の意味は '小銭' です）。

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.ja.png)

> 画像提供 [Jen Looper](https://twitter.com/jenlooper)

十分な人間による翻訳がある場合、MLモデルは以前に専門家の人間が翻訳したテキストの一般的なパターンを特定することにより、翻訳の精度を向上させることができます。

### 演習 - 翻訳

`TextBlob` を使用して文章を翻訳できます。**高慢と偏見**の有名な最初の一文を試してみてください：

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` は翻訳でかなり良い仕事をします："C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!"。

`TextBlob` の翻訳は、実際には1932年にV. LeconteとCh. Pressoirによって行われた本のフランス語翻訳よりもはるかに正確であると言えます：

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

この場合、機械学習による翻訳は、原作者の言葉を不必要に追加している人間の翻訳者よりも良い仕事をしています。

> ここで何が起こっているのでしょうか？そしてなぜ`TextBlob`は翻訳がこんなに上手いのでしょうか？実は、背後ではGoogle翻訳を使用しており、何百万ものフレーズを解析して最適な文字列を予測する高度なAIが動作しています。ここでは手動の操作は一切行われておらず、`blob.translate` を使用するにはインターネット接続が必要です。

## 感情分析

次に、機械学習を使用してテキストの感情を分析する方法を見てみましょう。

> **例**: "Great, that was a wonderful waste of time, I'm glad we are lost on this dark road" は皮肉で否定的な感情の文ですが、単純なアルゴリズムは 'great', 'wonderful', 'glad' を肯定的として検出し、'waste', 'lost' および 'dark' を否定的として検出します。全体の感情はこれらの相反する単語によって揺れ動きます。

✅ 人間の話者として皮肉をどのように伝えるかについて少し考えてみてください。声のイントネーションが大きな役割を果たします。"Well, that film was awesome" というフレーズを異なる方法で言ってみて、声がどのように意味を伝えるかを発見してみてください。

### 機械学習のアプローチ

機械学習のアプローチは、否定的および肯定的なテキストのコーパスを手動で収集することです。ツイート、映画のレビュー、または人間がスコアと意見を書いたものなら何でも構いません。その後、意見とスコアにNLP技術を適用し、パターンを見つけます（例えば、肯定的な映画レビューには 'Oscar worthy' というフレーズが否定的な映画レビューよりも多く含まれる傾向がある、または肯定的なレストランレビューには 'gourmet' という言葉が 'disgusting' よりも多く含まれる）。

> ⚖️ **例**: 政治家のオフィスで働いていて、新しい法律が議論されている場合、有権者はその特定の新しい法律を支持するメールや反対するメールをオフィスに送るかもしれません。あなたがそのメールを読んで、*賛成* と *反対* に分けるように任命されたとしましょう。メールがたくさんあれば、すべてを読むのは大変です。ボットがすべてのメールを読んで、理解し、どの山に属するかを教えてくれたら素晴らしいと思いませんか？
>
> これを実現する一つの方法は、機械学習を使用することです。モデルを*反対*のメールの一部と*賛成*のメールの一部で訓練します。モデルは、反対側または賛成側のメールに特定のフレーズや単語が現れる可能性が高いことを関連付ける傾向がありますが、*内容を理解することはありません*。モデルを訓練に使用していないメールでテストし、同じ結論に達するかどうかを確認できます。モデルの精度に満足したら、今後のメールを読むことなく処理できます。

✅ 以前のレッスンで使用したプロセスと似ていると思いますか？

## 演習 - 感情的な文章

感情は -1 から 1 の*極性*で測定されます。-1 は最も否定的な感情を示し、1 は最も肯定的な感情を示します。また、感情は客観性 (0) と主観性 (1) のスコアで測定されます。

ジェーン・オースティンの『高慢と偏見』をもう一度見てみましょう。テキストは [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) で利用可能です。以下のサンプルは、本の最初と最後の文章の感情を分析し、その感情の極性と主観性/客観性のスコアを表示する短いプログラムを示しています。

以下のタスクでは、`sentiment` を決定するために `TextBlob` ライブラリ（上記で説明）を使用する必要があります（独自の感情計算機を書く必要はありません）。

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

次のような出力が表示されます：

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## チャレンジ - 感情の極性をチェック

あなたのタスクは、感情の極性を使用して、『高慢と偏見』が絶対的に肯定的な文章が絶対的に否定的な文章より多いかどうかを判断することです。このタスクでは、極性スコアが 1 または -1 である場合、それぞれ絶対的に肯定的または否定的であると仮定できます。

**ステップ:**

1. Project Gutenberg から [高慢と偏見のコピー](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) を .txt ファイルとしてダウンロードします。ファイルの最初と最後のメタデータを削除し、元のテキストのみを残します。
2. Pythonでファイルを開き、内容を文字列として抽出します。
3. 本の文字列を使用して TextBlob を作成します。
4. 本の各文章をループで分析します。
   1. 極性が 1 または -1 の場合、文章を肯定的または否定的なメッセージの配列またはリストに保存します。
5. 最後に、肯定的な文章と否定的な文章（別々に）およびそれぞれの数を出力します。

サンプルの[解決策](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb)はこちらです。

✅ 知識チェック

1. 感情は文中で使用される単語に基づいていますが、コードは単語を*理解*していますか？
2. 感情の極性が正確だと思いますか、つまり、スコアに*同意*しますか？
   1. 特に、次の文章の絶対的な**肯定的**極性に同意しますか、それとも反対しますか？
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. 次の3つの文章は絶対的に肯定的な感情でスコアリングされましたが、よく読むと肯定的な文章ではありません。なぜ感情分析はそれらを肯定的な文章だと思ったのでしょうか？
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. 次の文章の絶対的な**否定的**極性に同意しますか、それとも反対しますか？
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ ジェーン・オースティンのファンなら、彼女がしばしば自分の本を使ってイギリスのリージェンシー社会のより滑稽な側面を批判していることを理解しているでしょう。『高慢と偏見』の主人公であるエリザベス・ベネットは、鋭い社会観察者であり（著者と同様）、彼女の言葉はしばしば非常に微妙です。物語のラブインタレストであるダルシー氏でさえ、エリザベスの遊び心とからかいの言葉の使い方に気づいています。「あなたが時折、自分の意見ではないことを表明することを楽しんでいることを知っています。」

---

## 🚀チャレンジ

ユーザー入力から他の特徴を抽出して、Marvinをさらに改善できますか？

## [講義後クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## 復習 & 自習

テキストから感情を抽出する方法はたくさんあります。この技術を利用するビジネスアプリケーションについて考えてみてください。また、どのように誤って使用される可能性があるかについても考えてみてください。感情を分析する洗練されたエンタープライズ対応のシステムについてさらに詳しく読みましょう。例えば、[Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott) などです。上記の『高慢と偏見』の文章のいくつかをテストして、ニュアンスを検出できるかどうかを確認してみてください。

## 課題

[Poetic license](assignment.md)

**免責事項**:
この文書は機械翻訳AIサービスを使用して翻訳されています。正確さを期すよう努めていますが、自動翻訳には誤りや不正確さが含まれる可能性があります。元の言語で書かれた原文を信頼できる情報源としてください。重要な情報については、専門の人間による翻訳をお勧めします。この翻訳の使用により生じた誤解や誤認について、当方は一切の責任を負いません。