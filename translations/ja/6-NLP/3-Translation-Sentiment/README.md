<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-06T09:42:39+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "ja"
}
-->
# 翻訳と感情分析をMLで行う

前のレッスンでは、`TextBlob`を使用して基本的なボットを構築する方法を学びました。このライブラリは、名詞句の抽出などの基本的な自然言語処理（NLP）タスクを実行するために、裏で機械学習（ML）を組み込んでいます。計算言語学におけるもう一つの重要な課題は、ある話し言葉や書き言葉の言語から別の言語への正確な「翻訳」です。

## [講義前のクイズ](https://ff-quizzes.netlify.app/en/ml/)

翻訳は非常に難しい問題であり、何千もの言語が存在し、それぞれが非常に異なる文法規則を持つ可能性があるため、さらに複雑になります。一つのアプローチは、英語のような言語の正式な文法規則を非言語依存の構造に変換し、それを別の言語に変換する方法です。このアプローチでは、以下の手順を取ることになります：

1. **識別**: 入力言語の単語を名詞、動詞などにタグ付けして識別する。
2. **翻訳の作成**: ターゲット言語形式で各単語を直接翻訳する。

### 英語からアイルランド語への例文

英語では、_I feel happy_ という文は以下の順序で3つの単語から成り立っています：

- **主語** (I)
- **動詞** (feel)
- **形容詞** (happy)

しかし、アイルランド語では、同じ文は非常に異なる文法構造を持っています。「happy」や「sad」のような感情は、あなたの上に「ある」と表現されます。

英語のフレーズ `I feel happy` はアイルランド語では `Tá athas orm` となります。直訳すると `Happy is upon me` となります。

アイルランド語話者が英語に翻訳する場合、`Happy is upon me` ではなく `I feel happy` と言います。これは、単語や文の構造が異なっていても、文の意味を理解しているからです。

アイルランド語の文の正式な順序は以下の通りです：

- **動詞** (Tá または is)
- **形容詞** (athas または happy)
- **主語** (orm または upon me)

## 翻訳

単純な翻訳プログラムは、文の構造を無視して単語だけを翻訳するかもしれません。

✅ 第二言語（または第三言語以上）を成人してから学んだことがある場合、母国語で考え、概念を頭の中で単語ごとに第二言語に翻訳し、それを話すことから始めたかもしれません。これは、単純な翻訳コンピュータプログラムが行っていることと似ています。この段階を超えて流暢さを達成することが重要です！

単純な翻訳は、悪い（時には面白い）誤訳を引き起こします。例えば、`I feel happy` を直訳するとアイルランド語では `Mise bhraitheann athas` となります。これは文字通り `me feel happy` を意味し、有効なアイルランド語の文ではありません。英語とアイルランド語は、隣接する2つの島で話されている言語ですが、文法構造が非常に異なる言語です。

> アイルランド語の言語的伝統についてのビデオをいくつか見ることができます。例えば [こちら](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### 機械学習アプローチ

これまで、自然言語処理における正式な規則アプローチについて学びました。もう一つのアプローチは、単語の意味を無視し、代わりに機械学習を使用してパターンを検出することです。これは、元の言語とターゲット言語の両方で大量のテキスト（*コーパス*）またはテキスト群（*コーパラ*）がある場合に翻訳で機能します。

例えば、ジェーン・オースティンが1813年に書いた有名な英語の小説『高慢と偏見』を考えてみましょう。この本を英語で読み、*フランス語*の人間による翻訳を参照すると、一方の言語でのフレーズが他方の言語に*慣用的に*翻訳されていることを検出できます。これをすぐに試してみましょう。

例えば、英語のフレーズ `I have no money` をフランス語に直訳すると、`Je n'ai pas de monnaie` になるかもしれません。「Monnaie」はフランス語の微妙な「偽の同義語」であり、「money」と「monnaie」は同義ではありません。人間が行うより良い翻訳は `Je n'ai pas d'argent` であり、これは「お金がない」という意味をよりよく伝えます（「monnaie」は「小銭」という意味です）。

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> 画像提供：[Jen Looper](https://twitter.com/jenlooper)

十分な人間による翻訳があれば、MLモデルは以前に専門の人間話者によって翻訳されたテキストの一般的なパターンを特定することで、翻訳の精度を向上させることができます。

### 演習 - 翻訳

`TextBlob` を使用して文を翻訳できます。有名な『高慢と偏見』の冒頭の一文を試してみましょう：

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` は翻訳をかなりうまく行います："C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!"。

実際、TextBlobの翻訳は、1932年にV. LeconteとCh. Pressoirによるフランス語翻訳よりもはるかに正確であると言えます：

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

この場合、MLによる翻訳は、原作者の意図を明確にするために不必要に言葉を追加している人間の翻訳者よりも良い仕事をしています。

> ここで何が起こっているのでしょうか？そして、なぜTextBlobは翻訳が得意なのでしょうか？実は、裏でGoogle翻訳を使用しており、数百万のフレーズを解析してタスクに最適な文字列を予測する高度なAIを活用しています。ここでは手動の作業は一切行われておらず、`blob.translate` を使用するにはインターネット接続が必要です。

✅ 他の文も試してみましょう。MLと人間の翻訳のどちらが優れているでしょうか？どのような場合に違いが出るでしょうか？

## 感情分析

機械学習が非常にうまく機能するもう一つの分野は感情分析です。非MLアプローチでは、ポジティブな単語とネガティブな単語を識別し、新しいテキストが与えられた場合、ポジティブ、ネガティブ、中立の単語の総価値を計算して全体的な感情を特定します。

このアプローチは、マーヴィンのタスクで見たように簡単に騙される可能性があります。例えば、`Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` という文は皮肉的でネガティブな感情の文ですが、単純なアルゴリズムは `great`、`wonderful`、`glad` をポジティブとして検出し、`waste`、`lost`、`dark` をネガティブとして検出します。これらの矛盾する単語によって全体的な感情が揺らぎます。

✅ 人間の話者として皮肉をどのように伝えるか考えてみてください。声の抑揚が大きな役割を果たします。"Well, that film was awesome" というフレーズを異なる方法で言ってみて、声が意味をどのように伝えるかを発見してください。

### MLアプローチ

MLアプローチでは、ネガティブとポジティブなテキスト群を手動で収集します。例えば、ツイートや映画レビューなど、人間がスコアと意見を書いたものです。その後、NLP技術を意見とスコアに適用し、パターンが浮かび上がります（例：ポジティブな映画レビューでは「Oscar worthy」というフレーズがネガティブな映画レビューよりも頻繁に使用される、またはポジティブなレストランレビューでは「gourmet」という言葉が「disgusting」よりも頻繁に使用される）。

> ⚖️ **例**: 政治家のオフィスで働いていて、新しい法律が議論されている場合、支持するメールや反対するメールがオフィスに届くかもしれません。これらのメールを読んで、*支持*と*反対*の2つの山に分類するように指示されたとします。メールが多い場合、すべてを読むのは圧倒されるかもしれません。ボットがすべてのメールを読み、理解して、それぞれのメールがどの山に属するかを教えてくれるとしたら便利ではないでしょうか？
> 
> これを実現する一つの方法は機械学習を使用することです。モデルを*反対*メールの一部と*支持*メールの一部でトレーニングします。モデルは、反対側または支持側のメールに特定のフレーズや単語が出現する可能性が高いことを関連付けますが、*内容を理解することはありません*。その後、トレーニングに使用していないメールでテストし、モデルが自分と同じ結論に達するかどうかを確認します。モデルの精度に満足したら、将来のメールを読むことなく処理できるようになります。

✅ このプロセスは、以前のレッスンで使用したプロセスに似ていますか？

## 演習 - 感情的な文

感情は、-1から1の範囲で*極性*として測定されます。-1は最もネガティブな感情を意味し、1は最もポジティブな感情を意味します。また、感情は0から1のスコアで客観性（0）と主観性（1）として測定されます。

ジェーン・オースティンの『高慢と偏見』をもう一度見てみましょう。このテキストは[Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm)で利用可能です。以下のサンプルは、書籍の最初と最後の文の感情を分析し、その感情の極性と主観性/客観性スコアを表示する短いプログラムを示しています。

以下のタスクでは、`TextBlob`ライブラリ（上記で説明）を使用して`sentiment`を決定してください（独自の感情計算機を作成する必要はありません）。

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

以下の出力が表示されます：

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## チャレンジ - 感情の極性を確認する

あなたのタスクは、感情の極性を使用して、『高慢と偏見』に絶対的にポジティブな文が絶対的にネガティブな文よりも多いかどうかを判断することです。このタスクでは、極性スコアが1または-1の場合、絶対的にポジティブまたはネガティブであると仮定できます。

**手順:**

1. [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm)から『高慢と偏見』のコピーを.txtファイルとしてダウンロードします。ファイルの冒頭と末尾のメタデータを削除し、元のテキストのみを残します。
2. Pythonでファイルを開き、内容を文字列として抽出します。
3. 書籍文字列を使用してTextBlobを作成します。
4. 書籍内の各文をループで分析します。
   1. 極性が1または-1の場合、その文をポジティブまたはネガティブなメッセージの配列またはリストに保存します。
5. 最後に、ポジティブな文とネガティブな文（別々に）をすべて印刷し、それぞれの数を表示します。

こちらにサンプル[解答](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb)があります。

✅ 知識チェック

1. 感情は文で使用される単語に基づいていますが、コードは単語を*理解*していますか？
2. 感情の極性が正確だと思いますか？つまり、スコアに*同意*しますか？
   1. 特に、以下の文の絶対的な**ポジティブ**極性に同意しますか？
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. 次の3つの文は絶対的なポジティブな感情としてスコア付けされましたが、よく読むとポジティブな文ではありません。なぜ感情分析はこれらをポジティブな文だと判断したのでしょうか？
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. 以下の文の絶対的な**ネガティブ**極性に同意しますか？
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ ジェーン・オースティンの愛好家なら、彼女がしばしば英語リージェンシー社会のより滑稽な側面を批判するために彼女の本を使用していることを理解しているでしょう。『高慢と偏見』の主人公であるエリザベス・ベネットは鋭い社会観察者（著者と同様）であり、彼女の言葉はしばしば非常にニュアンスに富んでいます。物語の恋愛対象であるダーシー氏でさえ、エリザベスの遊び心とからかいの言葉遣いに気づきます："I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀チャレンジ

ユーザー入力から他の特徴を抽出することで、マーヴィンをさらに改善できますか？

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## レビューと自己学習
テキストから感情を抽出する方法はたくさんあります。この技術を活用する可能性のあるビジネスアプリケーションについて考えてみてください。また、この技術がどのように誤作動する可能性があるかについても考えてみましょう。感情を分析する高度で企業向けのシステムについてさらに詳しく知りたい場合は、[Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott)をご覧ください。上記の『高慢と偏見』の文章をいくつか試してみて、ニュアンスを検出できるかどうか確認してみてください。

## 課題

[Poetic license](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文を公式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当社は一切の責任を負いません。