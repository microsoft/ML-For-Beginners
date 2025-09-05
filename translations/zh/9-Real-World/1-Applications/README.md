<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T09:02:25+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "zh"
}
-->
# 后记：机器学习在现实世界中的应用

![现实世界中机器学习的总结图](../../../../sketchnotes/ml-realworld.png)
> 由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 绘制的手绘笔记

在本课程中，你学习了许多准备数据进行训练和创建机器学习模型的方法。你构建了一系列经典的回归、聚类、分类、自然语言处理和时间序列模型。恭喜你！现在，你可能会好奇这些模型的实际用途是什么……它们在现实世界中的应用是什么？

尽管深度学习驱动的人工智能在工业界引起了广泛关注，但经典机器学习模型仍然有其重要的应用价值。事实上，你可能已经在日常生活中使用了其中的一些应用！在本课中，你将探索八个不同的行业和领域如何利用这些模型来使其应用更加高效、可靠、智能，并为用户创造更大的价值。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

## 💰 金融

金融领域为机器学习提供了许多机会。该领域的许多问题都可以通过机器学习建模和解决。

### 信用卡欺诈检测

我们在课程中学习了 [k-means 聚类](../../5-Clustering/2-K-Means/README.md)，但它如何用于解决信用卡欺诈相关问题呢？

k-means 聚类在一种称为**异常值检测**的信用卡欺诈检测技术中非常有用。异常值，即数据集中的偏离观测值，可以帮助我们判断信用卡的使用是否正常或是否存在异常情况。正如以下论文所述，你可以使用 k-means 聚类算法对信用卡数据进行分类，并根据每笔交易的异常程度将其分配到一个聚类中。然后，你可以评估最具风险的聚类以区分欺诈交易和合法交易。
[参考](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### 财富管理

在财富管理中，个人或公司代表客户管理投资。他们的工作是长期维持和增长财富，因此选择表现良好的投资至关重要。

评估某项投资表现的一种方法是通过统计回归。[线性回归](../../2-Regression/1-Tools/README.md)是理解基金相对于某个基准表现的有力工具。我们还可以推断回归结果是否具有统计显著性，以及它们对客户投资的影响程度。你甚至可以进一步扩展分析，使用多元回归来考虑额外的风险因素。以下论文展示了如何使用回归评估特定基金的表现。
[参考](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 教育

教育领域也是机器学习可以应用的一个非常有趣的领域。这里有许多有趣的问题需要解决，例如检测考试或论文中的作弊行为，或管理纠正过程中的偏见（无论是有意还是无意）。

### 预测学生行为

[Coursera](https://coursera.com)，一个在线开放课程提供商，在其技术博客中讨论了许多工程决策。在这个案例研究中，他们绘制了一条回归线，试图探索低 NPS（净推荐值）评分与课程保留或退课之间的相关性。
[参考](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### 减少偏见

[Grammarly](https://grammarly.com)，一个检查拼写和语法错误的写作助手，在其产品中使用了复杂的[自然语言处理系统](../../6-NLP/README.md)。他们在技术博客中发布了一篇有趣的案例研究，讨论了如何处理机器学习中的性别偏见问题，这也是你在我们的[公平性入门课程](../../1-Introduction/3-fairness/README.md)中学习过的内容。
[参考](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 零售

零售行业可以通过机器学习受益，从优化客户体验到优化库存管理。

### 个性化客户体验

在 Wayfair，一家销售家具等家居用品的公司，帮助客户找到符合他们品味和需求的产品至关重要。在这篇文章中，该公司的工程师描述了他们如何使用机器学习和自然语言处理来“为客户提供合适的搜索结果”。特别是，他们的查询意图引擎通过实体提取、分类器训练、资产和意见提取以及客户评论的情感标记来实现。这是 NLP 在在线零售中的经典应用案例。
[参考](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### 库存管理

像 [StitchFix](https://stitchfix.com) 这样的创新型公司，一个向消费者发送服装盒的服务，严重依赖机器学习进行推荐和库存管理。他们的造型团队与商品团队紧密合作：“我们的数据科学家使用遗传算法并将其应用于服装，以预测哪些尚不存在的服装可能会成功。我们将这一工具提供给商品团队，现在他们可以将其作为工具使用。”
[参考](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 医疗保健

医疗保健领域可以利用机器学习优化研究任务以及物流问题，例如患者再入院管理或疾病传播控制。

### 临床试验管理

临床试验中的毒性是药物制造商的主要关注点。多少毒性是可以接受的？在这项研究中，分析各种临床试验方法导致了一种预测临床试验结果概率的新方法的开发。具体来说，他们使用随机森林生成了一个[分类器](../../4-Classification/README.md)，能够区分药物组。
[参考](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### 医院再入院管理

医院护理成本高昂，尤其是当患者需要再次入院时。这篇论文讨论了一家公司如何使用机器学习通过[聚类](../../5-Clustering/README.md)算法预测再入院的可能性。这些聚类帮助分析师“发现可能具有共同原因的再入院群体”。
[参考](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### 疾病管理

最近的疫情突显了机器学习在阻止疾病传播方面的作用。在这篇文章中，你会看到 ARIMA、逻辑曲线、线性回归和 SARIMA 的应用。“这项工作试图计算病毒的传播率，从而预测死亡、康复和确诊病例，以帮助我们更好地准备和应对。”
[参考](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 生态与绿色科技

自然和生态由许多敏感系统组成，动物与自然之间的相互作用尤为重要。准确测量这些系统并在发生问题时采取适当行动（例如森林火灾或动物数量下降）非常重要。

### 森林管理

你在之前的课程中学习了[强化学习](../../8-Reinforcement/README.md)。它在预测自然模式时非常有用。特别是，它可以用于跟踪生态问题，例如森林火灾和入侵物种的传播。在加拿大，一组研究人员使用强化学习从卫星图像中构建了森林火灾动态模型。通过创新的“空间传播过程（SSP）”，他们将森林火灾视为“景观中任何单元格的代理”。“火灾在任何时间点可以采取的行动包括向北、南、东或西传播或不传播。”

这种方法颠覆了通常的强化学习设置，因为相应马尔可夫决策过程（MDP）的动态是已知的即时火灾传播函数。阅读以下链接了解该团队使用的经典算法。
[参考](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### 动物运动感知

虽然深度学习在视觉跟踪动物运动方面带来了革命性变化（你可以在这里构建自己的[北极熊追踪器](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott)），但经典机器学习在这一任务中仍然有其作用。

用于跟踪农场动物运动的传感器和物联网利用了这种视觉处理，但更基本的机器学习技术在数据预处理方面非常有用。例如，在这篇论文中，使用各种分类器算法监测和分析了羊的姿势。你可能会在第 335 页看到 ROC 曲线。
[参考](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ 能源管理

在我们关于[时间序列预测](../../7-TimeSeries/README.md)的课程中，我们提到了智能停车计时器的概念，通过理解供需关系为一个城镇创造收入。这篇文章详细讨论了聚类、回归和时间序列预测如何结合起来帮助预测爱尔兰未来的能源使用，基于智能计量。
[参考](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 保险

保险行业是另一个使用机器学习构建和优化可行财务和精算模型的领域。

### 波动性管理

MetLife，一家人寿保险提供商，公开了他们分析和缓解财务模型波动性的方法。在这篇文章中，你会看到二元和序列分类的可视化图表，还会发现预测的可视化图表。
[参考](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 艺术、文化与文学

在艺术领域，例如新闻业，有许多有趣的问题。检测假新闻是一个巨大的挑战，因为它已被证明会影响人们的观点，甚至颠覆民主。博物馆也可以通过机器学习受益，从发现文物之间的联系到资源规划。

### 假新闻检测

在当今媒体中，检测假新闻已成为一场猫捉老鼠的游戏。在这篇文章中，研究人员建议测试结合我们学习过的多种机器学习技术的系统，并部署最佳模型：“该系统基于自然语言处理从数据中提取特征，然后使用这些特征训练机器学习分类器，例如朴素贝叶斯、支持向量机（SVM）、随机森林（RF）、随机梯度下降（SGD）和逻辑回归（LR）。”
[参考](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

这篇文章展示了如何结合不同的机器学习领域来产生有趣的结果，从而帮助阻止假新闻的传播和造成的实际损害；在这种情况下，动机是关于 COVID 治疗的谣言传播引发的暴力事件。

### 博物馆机器学习

博物馆正处于人工智能革命的前沿，随着技术的进步，编目和数字化收藏以及发现文物之间的联系变得更加容易。像 [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) 这样的项目正在帮助解锁难以接触的收藏，例如梵蒂冈档案。但博物馆的商业方面也从机器学习模型中受益。

例如，芝加哥艺术学院构建了模型来预测观众的兴趣以及他们参观展览的时间。目标是每次用户参观博物馆时都能创造个性化和优化的体验。“在 2017 财年，该模型预测的参观人数和门票收入的准确率达到了 1%，”芝加哥艺术学院高级副总裁 Andrew Simnick 说道。
[参考](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 营销

### 客户细分

最有效的营销策略根据不同的分组以不同方式定位客户。在这篇文章中，讨论了聚类算法在支持差异化营销中的应用。差异化营销帮助公司提高品牌认知度、接触更多客户并赚取更多利润。
[参考](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 挑战

找出另一个受益于本课程中所学技术的领域，并探索它如何使用机器学习。
## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

Wayfair的数据科学团队制作了几段有趣的视频，介绍他们如何在公司中应用机器学习。值得[看看](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)！

## 作业

[机器学习寻宝游戏](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们不对因使用此翻译而产生的任何误解或误读承担责任。