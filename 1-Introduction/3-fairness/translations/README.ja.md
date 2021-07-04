# 機械学習における公平さ
 
![機械学習における公平性をまとめたスケッチ](../../../sketchnotes/ml-fairness.png)
> [Tomomi Imura](https://www.twitter.com/girlie_mac)によるスケッチ

## [Pre-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/5/)
 
## イントロダクション

このカリキュラムでは、機械学習が私たちの日常生活にどのような影響を与えているかを知ることができます。たった今、医療の診断や不正の検出など、日常の意思決定にシステムやモデルが関わっています。そのため、誰もが公平な結果を得られるようにするためには、これらのモデルがうまく機能することが重要です。

しかし、これらのモデルを構築するために使用しているデータに、人種、性別、政治的見解、宗教などの特定の属性が欠けていたり、そのような属性が偏っていたりすると、何が起こるか想像してみてください。また、モデルの出力が特定の層に有利になるように解釈された場合はどうでしょうか。その結果、アプリケーションはどのような影響を受けるのでしょうか？

このレッスンでは、以下のことを行います:

- 機械学習における公平性の重要性に対する意識を高める。
- 公平性に関連する問題について学ぶ。
- 公平性の評価と緩和について学ぶ。

## 前提条件
前提条件として、"Responsible AI Principles"のLearn Pathを受講し、このトピックに関する以下のビデオを視聴してください。

こちらの[Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-15963-cxa)より、責任のあるAIについて学ぶ。

[![Microsoftの責任あるAIに対する取り組み](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoftの責任あるAIに対する取り組み")

> 🎥 上の画像をクリックすると動画が表示されます：Microsoftの責任あるAIに対する取り組み

## データやアルゴリズムの不公平さ

> 「データを長く拷問すれば、何でも自白するようになる」 - Ronald Coase

この言葉は極端に聞こえますが、データがどんな結論をも裏付けるように操作できることは事実です。しかし、そのような操作は、時に意図せずに行われることがあります。人間は誰でもバイアスを持っており、自分がいつデータにバイアスを導入しているかを意識的に知ることは難しいことが多いのです。

AIや機械学習における公平性の保証は、依然として複雑な社会技術的課題です。つまり、純粋に社会的な視点や技術的な視点のどちらからも対処できないということです。

### 公平性に関連した問題

不公平とはどういう意味ですか？不公平とは、人種、性別、年齢、障害の有無などで定義された人々のグループに悪影響を与えること、あるいは、被害を与えることです。

主な不公平に関連する問題は以下のように分類されます。:

- **アロケーション**。ある性別や民族が他の性別や民族よりも優遇されている場合。
- **サービスの質**。ある特定のシナリオのためにデータを訓練しても、現実がより複雑な場合にはサービスの質の低下につながります。
- **固定観念**。特定のグループにあらかじめ割り当てられた属性を関連させること。
- **誹謗中傷**。何かや誰かを不当に批判したり、レッテルを貼ること。
- **過剰表現または過小表現**。特定のグループが特定の職業に就いている姿が見られず、それを宣伝し続けるサービスや機能は被害を助長しているという考え。

それでは、いくつか例を見ていきましょう。

### アロケーション

ローン申請を審査する仮想的なシステムを考えてみましょう。このシステムでは、他のグループよりも白人男性を優秀な候補者として選ぶ傾向があります。その結果、特定の申請者にはローンが提供されませんでした。

もう一つの例は、大企業が候補者を審査するために開発した実験的な採用ツールです。このツールは、ある性別に関連する言葉を好むように訓練されたモデルを使って、ある性別をシステム的に差別していました。その結果、履歴書に「女子ラグビーチーム」などの単語が含まれている候補者にペナルティを課すものとなっていました。

✅ ここで、上記のような実例を少し調べてみてください。

### サービスの質

研究者は、いくつかの市販のジェンダー分類法は、明るい肌色の男性の画像と比較して、暗い肌色の女性の画像では高い不正解率を示したことを発見した。[参照](https://www.media.mit.edu/publications/gender-shades-intersectional-accuracy-disparities-in-commercial-gender-classification/) 

また、肌の色が暗い人を感知できなかったハンドソープディスペンサーの例も悪い意味で有名です。[参照](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)

### 固定観念

機械翻訳には、ステレオタイプな性別観が見られます。「彼はナースで、彼女は医者です。(“he is a nurse and she is a doctor”)」という文をトルコ語に翻訳する際、問題が発生しました。トルコ語は単数の三人称を表す代名詞「o」が1つあるのみで、性別の区別のない言語で、この文章をトルコ語から英語に翻訳し直すと、「彼女はナースで、彼は医者です。(“she is a nurse and he is a doctor”)」というステレオタイプによる正しくない文章になってしまいます。

![トルコ語に対する翻訳](../images/gender-bias-translate-en-tr.png)

![英語に復元する翻訳](../images/gender-bias-translate-tr-en.png)

### 誹謗中傷

画像ラベリング技術により、肌の色が黒い人の画像をゴリラと誤表示したことが有名です。誤表示は、システムが単に間違いをしたというだけでなく、黒人を誹謗中傷するためにこの表現が意図的に使われてきた長い歴史を持っていたため、有害である。

[![AI: 自分は女性ではないの？](https://img.youtube.com/vi/QxuyfWoVV98/0.jpg)](https://www.youtube.com/watch?v=QxuyfWoVV98 "AI: 自分は女性ではないの？")
> 🎥 上の画像をクリックすると動画が表示されます: AI: 自分は女性ではないの？ - AIによる人種差別的な誹謗中傷による被害を示すパフォーマンス

### 過剰表現または過小表現
 
異常な画像検索の結果はこの問題の良い例です。エンジニアやCEOなど、男性と女性の割合が同じかそれ以上の職業の画像を検索すると、どちらかの性別に大きく偏った結果が表示されるので注意が必要です。

![BingでCEOと検索](../images/ceos.png)
> This search on Bing for 'CEO' produces pretty inclusive results

これらの5つの主要なタイプの問題は、相互に排他的なものではなく、1つのシステムが複数のタイプの害を示すこともあります。さらに、それぞれのケースでは、その重大性が異なります。例えば、ある人に不当に犯罪者のレッテルを貼ることは、画像を誤って表示することよりもはるかに深刻な問題です。しかし、比較的深刻ではない被害であっても、人々が疎外感を感じたり、特別視されていると感じたりすることがあり、その累積的な影響は非常に抑圧的なものになりうることを覚えておくことは重要でしょう。
 
✅ **ディスカッション**: いくつかの例を再検討し、異なる害を示しているかどうかを確認してください。 

|                         | アロケーション | サービスの質 | 固定観念 | 誹謗中傷 | 過剰表現/過小表現 |
| ----------------------- | :--------: | :----------------: | :----------: | :---------: | :----------------------------: |
| 採用システムの自動化 |     x      |         x          |      x       |             |               x                |
| 機械翻訳     |            |                    |              |             |                                |
| 写真のラベリング          |            |                    |              |             |                                |


## 不公平の検出

あるシステムが不公平な動作をする理由はさまざまです。例えば、社会的なバイアスが、学習に使われたデータセットに反映されているかもしれないですし、過去のデータに頼りすぎたために、採用の不公平が悪化したかもしれません。あるモデルは、10年間に会社に提出された履歴書のパターンを利用して、男性からの履歴書が大半を占めていたことから、男性の方が適格であると判断しました。

特定のグループに関するデータが不十分であることも、不公平の原因となります。例えば、肌の色が濃い人のデータが少ないために、画像分類において肌の色が濃い人の画像のエラー率が高くなります。

また、開発時の誤った仮定も不公平の原因となります。例えば、人の顔の画像から犯罪を犯す人を予測することを目的とした顔分析システムでは、有害な推測をしてしまうことがあります。その結果、誤った分類をされた人が大きな被害を受けることになりかねません。

## モデルを理解し、公平性を構築する
 
Although many aspects of fairness are not captured in quantitative fairness metrics, and it is not possible to fully remove bias from a system to guarantee fairness, you are still responsible to detect and to mitigate fairness issues as much as possible. 

When you are working with machine learning models, it is important to understand your models by means of assuring their interpretability and by assessing and mitigating unfairness.

Let’s use the loan selection example to isolate the case to figure out each factor's level of impact on the prediction.

## Assessment methods

1. **Identify harms (and benefits)**. The first step is to identify harms and benefits. Think about how actions and decisions can affect both potential customers and a business itself.
  
1. **Identify the affected groups**. Once you understand what kind of harms or benefits that can occur, identify the groups that may be affected. Are these groups defined by gender, ethnicity, or social group?

1. **Define fairness metrics**. Finally, define a metric so you have something to measure against in your work to improve the situation.

### Identify harms (and benefits)

What are the harms and benefits associated with lending? Think about false negatives and false positive scenarios: 

**False negatives** (reject, but Y=1) - in this case, an applicant who will be capable of repaying a loan is rejected. This is an adverse event because the resources of the loans are withheld from qualified applicants.

**False positives** (accept, but Y=0) - in this case, the applicant does get a loan but eventually defaults. As a result, the applicant's case will be sent to a debt collection agency which can affect their future loan applications.

### Identify affected groups

The next step is to determine which groups are likely to be affected. For example, in case of a credit card application, a model might determine that women should receive much lower credit limits compared with their spouses who share household assets. An entire demographic, defined by gender, is thereby affected.

### Define fairness metrics
 
You have identified harms and an affected group, in this case, delineated by gender. Now, use the quantified factors to disaggregate their metrics. For example, using the data below, you can see that women have the largest false positive rate and men have the smallest, and that the opposite is true for false negatives.

✅ In a future lesson on Clustering, you will see how to build this 'confusion matrix' in code

|            | False positive rate | False negative rate | count |
| ---------- | ------------------- | ------------------- | ----- |
| Women      | 0.37                | 0.27                | 54032 |
| Men        | 0.31                | 0.35                | 28620 |
| Non-binary | 0.33                | 0.31                | 1266  |

 
This table tells us several things. First, we note that there are comparatively few non-binary people in the data. The data is skewed, so you need to be careful how you interpret these numbers.

In this case, we have 3 groups and 2 metrics. When we are thinking about how our system affects the group of customers with their loan applicants, this may be sufficient, but when you want to define larger number of groups, you may want to distill this to smaller sets of summaries. To do that, you can add more metrics, such as the largest difference or smallest ratio of each false negative and false positive. 
 
✅ Stop and Think: What other groups are likely to be affected for loan application? 
 
## Mitigating unfairness 
 
To mitigate unfairness, explore the model to generate various mitigated models and compare the tradeoffs it makes between accuracy and fairness to select the most fair model. 

This introductory lesson does not dive deeply into the details of algorithmic unfairness mitigation, such as post-processing and reductions approach, but here is a tool that you may want to try. 

### Fairlearn 
 
[Fairlearn](https://fairlearn.github.io/) is an open-source Python package that allows you to assess your systems' fairness and mitigate unfairness.  

The tool helps you to assesses how a model's predictions affect different groups, enabling you to compare multiple models by using fairness and performance metrics, and supplying a set of algorithms to mitigate unfairness in binary classification and regression. 

- Learn how to use the different components by checking out the Fairlearn's [GitHub](https://github.com/fairlearn/fairlearn/)

- Explore the [user guide](https://fairlearn.github.io/main/user_guide/index.html), [examples](https://fairlearn.github.io/main/auto_examples/index.html)

- Try some [sample notebooks](https://github.com/fairlearn/fairlearn/tree/master/notebooks). 
  
- Learn [how to enable fairness assessments](https://docs.microsoft.com/azure/machine-learning/how-to-machine-learning-fairness-aml?WT.mc_id=academic-15963-cxa) of machine learning models in Azure Machine Learning. 
  
- Check out these [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/contrib/fairness) for more fairness assessment scenarios in Azure Machine Learning. 

---
## 🚀 Challenge 
 
To prevent biases from being introduced in the first place, we should: 

- have a diversity of backgrounds and perspectives among the people working on systems 
- invest in datasets that reflect the diversity of our society 
- develop better methods for detecting and correcting bias when it occurs 

Think about real-life scenarios where unfairness is evident in model-building and usage. What else should we consider? 

## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/6/)
## Review & Self Study 
 
In this lesson, you have learned some basics of the concepts of fairness and unfairness in machine learning.  
 
Watch this workshop to dive deeper into the topics: 

- YouTube: Fairness-related harms in AI systems: Examples, assessment, and mitigation by Hanna Wallach and Miro Dudik [Fairness-related harms in AI systems: Examples, assessment, and mitigation - YouTube](https://www.youtube.com/watch?v=1RptHwfkx_k) 

Also, read: 

- Microsoft’s RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoft’s FATE research group: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

Explore the Fairlearn toolkit

[Fairlearn](https://fairlearn.org/)

Read about Azure Machine Learning's tools to ensure fairness

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-15963-cxa) 

## Assignment

[Explore Fairlearn](assignment.md) 
