# 機械学習のための回帰モデル
## トピック: 北米のカボチャ価格に関する回帰モデル 🎃

北米では、ハロウィンのためにカボチャはよく怖い顔に彫られています。そんな魅力的な野菜についてもっと知りましょう！

![jack-o-lanterns](../images/jack-o-lanterns.jpg)
> <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a>によって<a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>に投稿された写真

## 今回学ぶこと
この章のレッスンでは、機械学習の文脈における回帰の種類について説明します。回帰モデルは変数間の"関係"を決定するのに役立ちます。このタイプのモデルは、長さ、温度、年齢などの値を予測し、データポイントの分析をすることで変数間の関係性を明らかにします。

今回のレッスンでは、線形回帰とロジスティック回帰の違いやどのように使い分けるかを説明します。

データサイエンティストの共通開発環境であるノートブックを管理するためのVisual Studio Codeの構成や機械学習のタスクを開始するための準備を行います。また、機械学習用のライブラリであるScikit-learnを利用し最初のモデルを構築します。この章では回帰モデルに焦点を当てます。

> 回帰モデルを学習するのに役立つローコードツールがあります。ぜひ[Azure ML for this task](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)を使ってみてください。

### レッスン

1. [商売道具](../1-Tools/translations/README.ja.md)
2. [データ管理](../2-Data/translations/README.ja.md)
3. [線形回帰と多項式回帰](../3-Linear/translations/README.ja.md)
4. [ロジスティック回帰](../4-Logistic/translations/README.ja.md)

---
### クレジット

"機械学習と回帰"は、[Jen Looper](https://twitter.com/jenlooper)によって制作されました。

クイズの貢献者: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)と[Ornella Altunyan](https://twitter.com/ornelladotcom)

pumpkin datasetは、[こちらのKaggleプロジェクト](https://www.kaggle.com/usda/a-year-of-pumpkin-prices)で提案されています。このデータは、アメリカ合衆国農務省が配布している[Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice)が元になっています。私たちは、分布を正規化するために多様性を元に色についていくつか追加を行っています。このデータはパブリックドメインです。
