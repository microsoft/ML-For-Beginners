# 附录：机器学习在现实世界中的应用


![机器学习在现实世界中的总结：素描笔记](../../../sketchnotes/ml-realworld.png)
> 素描为 [Tomomi Imura](https://www.twitter.com/girlie_mac) 作品

在这个课程中，您学会了许多准备数据以进行训练和创建机器学习模型的方法。您建立了一系列经典回归、聚类、分类、自然语言处理和时间序列模型。恭喜你！现在，您可能会想知道所有这些模型的真实世界应用是什么？

尽管AI通常利用深度学习，吸引了许多行业的兴趣，但经典机器学习模型仍有宝贵的应用。你甚至可能今天就在使用这些应用！在本课程中，你将探索八个不同行业和学科领域如何使用这些类型的模型，以使它们的应用程序更加高效、可靠、智能和有价值。

## [课前小测](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/)

## 💰 经济

机器学习在经济部分有很多的应用，许多这个领域的问题都可以通过使用机器学习来建模和解决。

### 信用卡欺诈检测

我们已经在这门课程的早些时候学习过 [K-means聚类算法](../../../5-Clustering/2-K-Means/README.md)，但是这个算法是如何解决信息卡欺诈的问题的？


K-means聚类算法在信用卡欺诈检测技术中的**异常检测**中非常有用。异常值或数据集中的观测偏差可以告诉我们信用卡是否被正常使用或是否存在异常情况。如下文链接的论文所示，可以使用k-means聚类算法对信用卡数据进行排序，并根据每个交易的异常值大小将其分配到一个集群中。然后，可以评估最危险的集群是诈骗性还是合法交易。


https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf

### 财富管理

在财富管理方面，一个人或公司代表他们的客户处理投资。他们的工作是在长期内维持和增长财富，因此选择表现良好的投资是至关重要的。

一个评估投资表现的方法是通过统计回归。[线性回归](../../../2-Regression/1-Tools/README.md)是理解基金相对于某个基准表现的宝贵工具。我们还可以推断回归的结果是否具有统计意义，或者它们会如何影响客户的投资。您甚至可以使用多重回归进一步扩展分析，其中可以考虑其他风险因素。有关如何为特定基金工作的示例，请查看下面的论文，该论文使用回归评估基金的表现。

http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/

## 🎓 教育

The educational sector is also a very interesting area where ML can be applied. There are interesting problems to be tackled such as detecting cheating on tests or essays or managing bias, unintentional or not, in the correction process.
教育部分也是一个尚待机器学习应用的非常有趣的领域。有很多有趣的问题可以解决，例如检测考试或论文的作弊行为，或者在纠正学生错误过程中的管理偏见，不论是有意还是无意的。

### 预测学生行为

[Coursera](https://coursera.com)是一个在线开放课程提供商，他们有一个很棒的技术博客，他们在那里讨论了许多工程决策。在这个案例研究中，他们绘制了一条回归线，试图探索NPS（净推荐指数）评分低与课程保留或退出之间的任何相关性。

https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a

### 缓和偏见

[Grammarly](https://grammarly.com)是一个写作助手，用于检查拼写和语法错误。它在整个产品中使用了复杂的[自然语言处理系统](../../../6-NLP/README.md)。他们在技术博客中发布了一个有趣的案例研究，讨论了他们如何处理机器学习中的性别偏见，这是您在我们的[简介公平课程](../../../1-Introduction/3-fairness/README.md)中学到的。

https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/

## 👜 零售行业

零售部分可以从机器学习的使用中获益，从创建更好的客户体验到以最佳方式存货。

### 个性化顾客的体验

在Wayfair，一家销售家具等家居用品的公司，帮助客户找到适合他们口味和需求的产品是至关重要的。在这篇文章中，公司的工程师描述了他们如何使用机器学习和自然语言处理来“为客户提供正确的结果”。值得注意的是，他们的查询意图引擎已经建立起来，可以使用实体提取，分类器训练，资产和意见提取以及客户评论的情感标记。这是在线零售中自然语言处理如何工作的典型用例。

https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search

### 进货管理

创新的，灵活的公司，如[StitchFix](https://stitchfix.com)，一家向消费者运送服装的盒子服务，对推荐和库存管理依赖于机器学习。他们的风格团队与他们的商品团队一起工作，事实上：“我们的一位数据科学家使用遗传算法对服装进行了调整，并将其应用于服装，以预测今天不存在的成功服装。我们将其带给商品团队，现在他们可以将其作为工具使用。”

https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/

## 🏥 医疗保健

医院可以利用机器学习来优化研究任务，也可以解决诸如重新入院患者或阻止疾病传播之类的后勤问题。

### 管理临床试验

临床试验中的毒性是药物制造商的主要关注点。可以容忍多少毒性？在这项研究中，分析各种临床试验方法导致了开发一种新方法来预测临床试验结果的可能性。具体来说，他们能够使用随机森林来产生一个[分类器](../../../4-Classification/README.md)，该分类器能够区分药物组。

https://www.sciencedirect.com/science/article/pii/S2451945616302914

### 医院再入院管理

医院护理成本高昂，尤其是当患者需要重新入院时。本文讨论了一家公司，使用[聚类](../../../5-Clustering/README.md)算法预测再入院潜力。这些聚类有助于分析师“发现可能共享共同原因的再入院群”。

https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning

### 疾病管理

最近的大流行病照亮了机器学习如何帮助阻止疾病传播的方式。在以下的文章中，您将认识到ARIMA，逻辑曲线，线性回归和SARIMA的使用。“这项工作旨在计算这种病毒的传播速率，从而预测死亡，恢复和确诊病例，以便我们可以更好地准备和生存。”

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/

## 🌲 生态和绿色科技

自然和生态由许多敏感的系统组成，其中动物和自然之间的相互作用成为焦点。如果发生了一些事情，如森林大火或动物人口的下降，那么能够准确地测量这些系统并采取适当的措施是非常重要的。

### 森林管理

在前面的课程中，你已经学习了[强化学习](../../../8-Reinforcement/README.md)。当试图预测自然中的模式时，它非常有用。尤其是它可以用来跟踪森林大火和入侵物种的生态问题。在加拿大，一组研究人员使用强化学习从卫星图像中构建森林大火动力学模型。使用创新的“空间扩散过程（SSP）”，他们将森林大火视为“任何地形单元中的代理”。“从任何时间点的位置开始，火可以采取的动作包括向北，南，东或西扩散或不扩散”。

这个方法反转了通常的RL设置，因为相应的马尔可夫决策过程（MDP）的动力学是立即野火扩散的已知函数。在下面的链接中，了解这组人员使用的经典算法的更多信息。

https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full

### 动物的运动传感

尽管深度学习在视觉上跟踪动物运动方面引起了革命（您可以在这里构建自己的[北极熊跟踪器](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott)），但经典的ML仍然在这项任务中有着重要的地位。

追踪农场动物运动的传感器和物联网利用了这种类型的视觉处理，但更基本的ML技术对预处理数据也很有用。例如，在本文中，使用各种分类器算法监视和分析了羊的姿势。您可能会在第335页上看到ROC曲线。

https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf

### ⚡️ 能源管理
  
在我们的课程中，我们调用了[智能停车计时器](../../../7-TimeSeries/README.md)的概念，根据供应和需求来为一个城镇产生收入。这篇文章详细讨论了如何将聚类，回归和时间序列预测相结合，以帮助预测基于智能计量的爱尔兰未来的能源使用情况。

https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf

## 💼 保险

保险行业是另一个使用ML构建和优化可行的金融和精算模型的行业。

### 波动性管理

MetLife是一家人寿保险提供商，他们对分析和减轻他们的金融模型中的波动性非常开放。在这篇文章中，您将注意到二元和序数分类可视化。您还将发现预测可视化。

https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf

## 🎨 艺术，文化和文学

在艺术领域，例如新闻报道，有许多有趣的问题。检测假新闻是一个巨大的问题，因为它已被证明可以影响人们的意见，甚至推翻民主制度。博物馆也可以从在寻找文物之间的联系到资源规划方面使用ML中受益。

### 假新闻检测

检测假新闻已经成为当今媒体中猫和老鼠的游戏。在本文中，研究人员建议可以测试并部署最佳模型的系统结合了我们学习的几种ML技术：“该系统基于自然语言处理从数据中提取特征，然后这些特征用于训练机器学习分类器，如朴素贝叶斯，支持向量机（SVM），随机森林（RF），随机梯度下降（SGD）和逻辑回归（LR）。”

https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf

这篇文章展示了如何将不同的ML领域结合起来，可以产生有趣的结果，可以帮助阻止假新闻的传播和造成真正的伤害：在这个案例中，关于COVID治疗的谣言的传播导致了暴力事件。

### 博物馆

博物馆是AI革命的前沿，其中目录和数字化收藏品以及在技术不断发展的情况下找到文物之间的联系变得更加容易。诸如[In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.)之类的项目正在帮助解锁梵蒂冈档案等无法访问的收藏品的神秘之处。同时，博物馆的商业方面同样受益于ML模型。

例如，芝加哥艺术学院建立了模型来预测观众对什么感兴趣以及何时参加展览。模型的预期目标是在每次用户访问博物馆时都创造个性化和优化的访客体验。芝加哥艺术学院的高级副总裁安德鲁·西姆尼克（Andrew Simnick）说：“在财政2017年期间，该模型预测了出席率和入场率，准确率为1％。”

https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices

## 🏷 市场

### 顾客分组

最高效的营销策略是根据不同的分组针对不同的客户。这篇文章讨论了聚类算法的用途，以支持差异化营销。差异化营销有助于公司提高品牌认知度，吸引更多客户，并获取更多利润。

https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/

## 🚀 挑战

识别另一个受益于本课程中某些技术的部门，并发现它如何使用ML。

## [课后小测](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/)

## 复习 & 自学

The Wayfair data science team has several interesting videos on how they use ML at their company. It's worth [taking a look](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!
Wayfair的数据科学团队有几个有趣的视频，介绍了他们在公司中如何使用ML。这些内容很值得[看一看](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)！

## 作业

[A ML scavenger hunt](assignment.md)
