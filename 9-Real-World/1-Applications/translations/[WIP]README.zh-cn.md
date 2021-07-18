# 附录：机器学习的实际应用


![Summary of Machine learning in the real world in a sketchnote](../../../sketchnotes/ml-realworld.png)
> 绘图笔记由[Tomomi Imura](https://www.twitter.com/girlie_mac)绘制

在本课程中，您学到了很多种为训练准备数据和创建机器学习模型的方法。您构建了一系列经典回归、聚类、分类、自然语言处理和时间序列模型。恭喜！现在，您可能想知道这一切是为了什么？这些模型如何应用于实际生活中？

虽然利用深度学习的AI已经引起了工业界的极大兴趣，但经典机器学习模型仍然有很多可取之处。甚至，可能您今天都使用了其中的一些应用程序！在本课中，您将探索八个不同的行业和领域如何使用这些模型来提高其应用程序的性能、可靠性、智能性和对用户的价值。

## [课前测验](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/49/)

## 💰 金融

金融领域为机器学习提供了许多机会。该领域的许多问题都可以通过使用机器学习来建模解决。

### 信用卡欺诈检测

我们在之前的课程里学习了[k-means clustering](../../../5-Clustering/2-K-Means/translations/README.zh-cn.md)，但它如何用于解决与信用卡欺诈相关的问题？

K-means 聚类在称为**异常值检测**的信用卡欺诈检测技术中十分有用。 异常值（或一组数据的观察偏差）可以告诉我们信用卡是否正常使用，或是否发生了异常情况。 如下面链接的论文所示，您可以使用k-means clustering算法对信用卡数据进行排序，并根据每笔交易出现的异常值将其分配到一个集群中。 然后，您可以评估欺诈交易与合法交易的风险最高的集群。

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf

### 财富管理

在财富管理中，个人或公司代表其客户处理投资。 他们的工作是长期维持并增长财富，因此选择表现良好的投资至关重要。

评估特定投资收益的一种方法是统计回归。其中，[线性回归]((../../../2-Regression/1-Tools/translations/README.zh-cn.md))是了解基金相对于某些基准表现的重要工具。我们还可以推断回归结果在统计上是否显着，或者它们会在多大程度上影响客户的投资结果。您甚至可以考虑其他风险因素，使用多元回归来进一步扩展您的分析。 有关如何适用于特定基金的示例，请查看以下有关使用回归工具评估基金业绩的论文。

http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/

## 🎓 教育

教育领域也是一个机器学习很引人注意的领域。 有一些令人瞩目的问题亟待解决，例如检测考试或论文中的作弊行为，或在评估过程中管理有意或无意的偏见。

### 预测学生行为

[Coursera](https://coursera.com)是一家在线公开课提供商。他们有一个很棒的技术博客讨论了许多工程决策。在这个下方的参考案例中，他们绘制了一条回归线，试图探究低 NPS（净推荐值）评级与课程保留或降级之间的相关性。

https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a

### 降低偏见

[Grammarly](https://grammarly.com)是一种检查拼写和语法错误的写作助手，在其整个产品中使用了复杂的[自然语言处理系统](../../../6-NLP/translations/README.zh-cn.md)。 他们在他们的技术博客中发布了一个有趣的案例研究，内容涉及他们如何处理机器学习中的性别偏见，您在我们的[introductory fairness lesson](../../../1-Introduction/3-fairness/translations/README.zh-cn.md)中了解到。

https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/

## 👜 零售

无论是创造更好的客户体验，还是以最佳方式管理库存，零售业绝对可以从机器学习的使用中受益匪浅。

### 个性化的客户体验

Wayfair是一家销售家具等家居用品的公司，对于它们来说，帮助客户找到适合他们口味和需求的产品至关重要。在下方文章中，该公司的工程师描述了他们如何使用机器学习和NLP来“为客户呈现正确的结果”。值得注意的是，他们构建的的意向引擎（Query Intent Engine）已经利用了实体提取、分类器训练、反馈意见提取，以及客户评论的情感标签。这是NLP如何在在线零售中发挥作用的经典案例。

https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search

### 库存管理

像 [StitchFix](https://stitchfix.com) （一家售卖服饰的造型公司）这样的创新、灵活的公司，非常依赖机器学习进行推荐和库存管理。 事实上，他们的造型团队与他们的营销团队合作：“我们的一位数据科学家修改了遗传算法并将其应用于服装，以预测今后可能大受欢迎的服装。我们将其引入到商品团队，现在他们可以将其用作工具。”

https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/

## 🏥 保健

The health care sector can leverage ML to optimize research tasks and also logistic problems like readmitting patients or stopping diseases from spreading.

### Managing clinical trials

Toxicity in clinical trials is a major concern to drug makers. How much toxicity is tolerable? In this study, analyzing various clinical trial methods led to the development of a new approach for predicting the odds of clinical trial outcomes. Specifically, they were able to use random forest to produce a [classifier](../../4-Classification/README.md) that is able to distinguish between groups of drugs.

https://www.sciencedirect.com/science/article/pii/S2451945616302914

### Hospital readmission management

Hospital care is costly, especially when patients have to be readmitted. This paper discusses a company that uses ML to predict readmission potential using [clustering](../../5-Clustering/README.md) algorithms. These clusters help analysts to "discover groups of readmissions that may share a common cause".

https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning

### Disease management

The recent pandemic has shone a bright light on the ways that machine learning can aid in stopping the spread of disease. In this article, you'll recognize the use of ARIMA, logistic curves, linear regression, and SARIMA. "This work is an attempt to calculate the rate of spread of this virus and thus to predict the deaths, recoveries, and confirmed cases, so that it may help us to prepare better and survive."

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/

## 🌲 生态与环保科技 

Nature and ecology consists of many sensitive systems where the interplay between animals and nature come into focus. It's important to be able to measure these systems accurately and act appropriately if something happens, like a forest fire or a drop in the animal population.

### Forest management

You learned about [Reinforcement Learning](../../8-Reinforcement/README.md) in previous lessons. It can be very useful when trying to predict patterns in nature. In particular, it can be used to track ecological problems like forest fires and the spread of invasive species. In Canada, a group of researchers used Reinforcement Learning to build forest wildfire dynamics models from satellite images. Using an innovative "spatially spreading process (SSP)", they envisioned a forest fire as "the agent at any cell in the landscape." "The set of actions the fire can take from a location at any point in time includes spreading north, south, east, or west or not spreading.

This approach inverts the usual RL setup since the dynamics of the corresponding Markov Decision Process (MDP) is a known function for immediate wildfire spread." Read more about the classic algorithms used by this group at the link below.

https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full

### Motion sensing of animals

While deep learning has created a revolution in visually-tracking animal movements (you can build your own [polar bear tracker](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-15963-cxa) here), classic ML still has a place in this task.

Sensors to track movements of farm animals and IoT makes use of this type of visual processing, but more basic ML techniques are useful to preprocess data. For example, in this paper, sheep postures were monitored and analyzed using various classifier algorithms. You might recognize the ROC curve on page 335.

https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf

### ⚡️ Energy Management
  
In our lessons on [time series forecasting](../../7-TimeSeries/README.md), we invoked the concept of smart parking meters to generate revenue for a town based on understanding supply and demand. This article discusses in detail how clustering, regression and time series forecasting combined to help predict future energy use in Ireland, based off of smart metering.

https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf

## 💼 保险行业

The insurance sector is another sector that uses ML to construct and optimize viable financial and actuarial models. 

### Volatility Management

MetLife, a life insurance provider, is forthcoming with the way they analyze and mitigate volatility in their financial models. In this article you'll notice binary and ordinal classification visualizations. You'll also discover forecasting visualizations.

https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf

## 🎨 Arts, Culture, and Literature

In the arts, for example in journalism, there are many interesting problems. Detecting fake news is huge problem as it has been proven to influence the opinion of people and even to topple democracies. Museums can also benefit from using ML in everything from finding links between artifacts to resource planning.

### Fake news detection

Detecting fake news has become a game of cat and mouse in today's media. In this article, researchers suggest that a system combining several of the ML techniques we have studied can be tested and the best model deployed: "This system is based on natural language processing to extract features from the data and then these features are used for the training of machine learning classifiers such as Naive Bayes,  Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), and Logistic Regression(LR)."

https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf

This article shows how combining different ML domains can produce interesting results that can help stop fake news from spreading and creating real damage; in this case, the impetus was the spread of rumors about COVID treatments that incited mob violence.

### Museum ML

Museums are at the cusp of an AI revolution in which cataloging and digitizing collections and finding links between artifacts is becoming easier as technology advances. Projects such as [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) are helping unlock the mysteries of inaccessible collections such as the Vatican Archives. But, the business aspect of museums benefits from ML models as well.

For example, the Art Institute of Chicago built models to predict what audiences are interested in and when they will attend expositions. The goals is to create individualized and optimized visitor experiences each time the user visit the museum. "During fiscal 2017, the model predicted attendance and admissions within 1 percent of accuracy, says Andrew Simnick, senior vice president at the Art Institute."

https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices

## 🏷 市场营销

### Customer segmentation

The most effective marketing strategies target customers in different ways based on various groupings. In this article, the uses of Clustering algorithms are discussed to support differentiated marketing. Differentiated marketing helps companies improve brand recognition, reach more customers, and make more money.

https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/

## 🚀 挑战

Identify another sector that benefits from some of the techniques you learned in this curriculum, and discover how it uses ML.

## [课后测试](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/50/)

## 复习&自学

The Wayfair data science team has several interesting videos on how they use ML at their company. It's worth [taking a look](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## 课后作业

[A ML scavenger hunt](assignment.md)
