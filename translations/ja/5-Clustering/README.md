<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T22:55:43+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "ja"
}
-->
# 機械学習のためのクラスタリングモデル

クラスタリングは、互いに似ているオブジェクトを見つけ、それらをクラスタと呼ばれるグループにまとめる機械学習のタスクです。他の機械学習アプローチとクラスタリングの違いは、プロセスが自動的に進む点です。実際、教師あり学習とは正反対と言っても良いでしょう。

## 地域トピック: ナイジェリアの聴衆の音楽嗜好に基づくクラスタリングモデル 🎧

ナイジェリアの多様な聴衆は、多様な音楽嗜好を持っています。Spotifyから収集したデータを使用して（[この記事](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)に触発されました）、ナイジェリアで人気の音楽を見てみましょう。このデータセットには、曲の「ダンサビリティ」スコア、「アコースティック性」、音量、「スピーチ性」、人気度、エネルギーに関するデータが含まれています。このデータからパターンを発見するのは興味深いでしょう！

![ターンテーブル](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.ja.jpg)

> 写真提供: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> on <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
このレッスンシリーズでは、クラスタリング技術を使用してデータを分析する新しい方法を学びます。クラスタリングは、データセットにラベルがない場合に特に役立ちます。ラベルがある場合は、以前のレッスンで学んだ分類技術の方が役立つかもしれません。しかし、ラベルのないデータをグループ化したい場合、クラスタリングはパターンを発見する素晴らしい方法です。

> クラスタリングモデルを扱う方法を学ぶのに役立つローコードツールがあります。[Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)を試してみてください。

## レッスン

1. [クラスタリングの概要](1-Visualize/README.md)
2. [K-Meansクラスタリング](2-K-Means/README.md)

## クレジット

これらのレッスンは🎶を込めて[Jen Looper](https://www.twitter.com/jenlooper)によって書かれ、[Rishit Dagli](https://rishit_dagli)と[Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)による有益なレビューが加えられました。

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify)データセットは、Spotifyから収集されたものとしてKaggleから提供されました。

このレッスンの作成に役立った有用なK-Meansの例には、[アイリスの探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、[入門ノートブック](https://www.kaggle.com/prashant111/k-means-clustering-with-python)、および[仮想NGOの例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)があります。

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文を公式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当社は一切の責任を負いません。