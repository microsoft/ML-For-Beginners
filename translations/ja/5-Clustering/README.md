# 機械学習のクラスタリングモデル

クラスタリングは、互いに似ているオブジェクトを見つけ出し、それらをクラスタと呼ばれるグループにまとめる機械学習のタスクです。クラスタリングが他の機械学習手法と異なるのは、この処理が自動的に行われる点で、実際には教師あり学習の逆とも言えます。

## 地域テーマ：ナイジェリアの聴衆の音楽嗜好向けクラスタリングモデル 🎧

ナイジェリアの多様な聴衆は多様な音楽の嗜好を持っています。[この記事](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)に触発されてSpotifyからスクレイプしたデータを使い、ナイジェリアで人気のある音楽を見てみましょう。このデータセットにはさまざまな曲の「ダンス性」スコア、「アコースティック度」、ラウドネス、「スピーチ度」、人気度、エネルギーが含まれています。このデータのパターンを発見するのは興味深いでしょう！

![ターンテーブル](../../../translated_images/ja/turntable.f2b86b13c53302dc.webp)

> 写真提供 <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> via <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
このシリーズのレッスンでは、クラスタリング技術を使った新しいデータ分析の方法を発見できます。ラベルのないデータセットの場合、クラスタリングは特に有用です。ラベルがある場合は、前のレッスンで学んだ分類技術の方が役立つかもしれません。しかし、ラベルなしデータをグループ化したい場合、クラスタリングはパターンを発見する素晴らしい方法です。

> クラスタリングモデルの操作方法を学ぶのに役立つローコードツールがあります。このタスクには [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)を試してみてください

## レッスン

1. [クラスタリング入門](1-Visualize/README.md)
2. [K-平均クラスタリング](2-K-Means/README.md)

## クレジット

これらのレッスンは [Jen Looper](https://www.twitter.com/jenlooper) が🎶を込めて執筆し、[Rishit Dagli](https://rishit_dagli/) と [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) の有益なレビューを受けています。

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) のデータセットはKaggleから提供され、Spotifyからスクレイプされたものです。

このレッスン作成に役立った有用なK-平均の例には、[アイリス探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、[入門ノートブック](https://www.kaggle.com/prashant111/k-means-clustering-with-python)、[仮想NGOの例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)があります。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責事項**：
本書類は AI 翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期していますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知おきください。原文の原語版が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や解釈違いについても、当方は責任を負いかねます。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->