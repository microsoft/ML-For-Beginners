<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-06T10:51:39+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "en"
}
-->
# Postscript: Machine Learning in the Real World

![Summary of machine learning in the real world in a sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

In this curriculum, you’ve learned various ways to prepare data for training and create machine learning models. You’ve built a series of classic regression, clustering, classification, natural language processing, and time series models. Congratulations! Now, you might be wondering what it’s all for... what are the real-world applications for these models?

While AI, often powered by deep learning, has captured much of the industry’s attention, classical machine learning models still have valuable applications. In fact, you might already be using some of these applications today! In this lesson, you’ll explore how eight different industries and domains use these types of models to make their applications more efficient, reliable, intelligent, and valuable to users.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finance

The finance sector offers numerous opportunities for machine learning. Many problems in this field are well-suited to being modeled and solved using ML.

### Credit Card Fraud Detection

We learned about [k-means clustering](../../5-Clustering/2-K-Means/README.md) earlier in the course, but how can it be applied to credit card fraud detection?

K-means clustering is useful in a fraud detection technique called **outlier detection**. Outliers, or deviations in data patterns, can indicate whether a credit card is being used normally or if something suspicious is happening. As detailed in the paper linked below, you can use k-means clustering to group credit card transactions and identify clusters with unusual patterns. These clusters can then be analyzed to distinguish fraudulent transactions from legitimate ones.  
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Wealth Management

In wealth management, individuals or firms manage investments on behalf of clients, aiming to sustain and grow wealth over time. Choosing high-performing investments is critical.

Statistical regression is a valuable tool for evaluating investment performance. [Linear regression](../../2-Regression/1-Tools/README.md) can help assess how a fund performs relative to a benchmark. It can also determine whether the results are statistically significant and how they might impact a client’s portfolio. Multiple regression can further enhance the analysis by accounting for additional risk factors. For an example of how regression can be used to evaluate fund performance, see the paper linked below.  
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Education

The education sector is another fascinating area for ML applications. Challenges include detecting cheating on tests or essays and addressing bias, whether intentional or unintentional, in grading.

### Predicting Student Behavior

[Coursera](https://coursera.com), an online course provider, has a tech blog where they share insights into their engineering decisions. In one case study, they used regression to explore correlations between low NPS (Net Promoter Score) ratings and course retention or drop-off.  
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigating Bias

[Grammarly](https://grammarly.com), a writing assistant that checks for spelling and grammar errors, uses advanced [natural language processing systems](../../6-NLP/README.md) in its products. Their tech blog features a case study on addressing gender bias in machine learning, a topic you explored in our [introductory fairness lesson](../../1-Introduction/3-fairness/README.md).  
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retail

The retail sector can greatly benefit from ML, from enhancing the customer journey to optimizing inventory management.

### Personalizing the Customer Journey

Wayfair, a company specializing in home goods, prioritizes helping customers find products that suit their tastes and needs. In this article, their engineers explain how they use ML and NLP to deliver relevant search results. Their Query Intent Engine employs techniques like entity extraction, classifier training, opinion extraction, and sentiment tagging on customer reviews. This is a classic example of NLP in online retail.  
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Inventory Management

Innovative companies like [StitchFix](https://stitchfix.com), a clothing subscription service, rely heavily on ML for recommendations and inventory management. Their data scientists collaborate with merchandising teams to predict successful clothing designs using genetic algorithms.  
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Health Care

The health care sector can use ML to optimize research and logistics, such as managing patient readmissions or preventing disease spread.

### Managing Clinical Trials

Toxicity in clinical trials is a significant concern for drug developers. This study used random forest to create a [classifier](../../4-Classification/README.md) that distinguishes between groups of drugs based on clinical trial outcomes.  
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Hospital Readmission Management

Hospital readmissions are costly. This paper describes how ML clustering algorithms can predict readmission potential and identify groups of readmissions with common causes.  
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Disease Management

The recent pandemic highlighted how ML can help combat disease spread. This article discusses using ARIMA, logistic curves, linear regression, and SARIMA to predict the spread of a virus and its outcomes, aiding in preparation and response.  
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecology and Green Tech

Nature and ecology involve sensitive systems where accurate measurement and timely action are crucial, such as in forest fire management or monitoring animal populations.

### Forest Management

[Reinforcement Learning](../../8-Reinforcement/README.md) can predict patterns in nature, such as forest fires. Researchers in Canada used RL to model wildfire dynamics from satellite images, treating the fire as an agent in a Markov Decision Process.  
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Motion Sensing of Animals

While deep learning is often used for tracking animal movements, classical ML techniques are still valuable for preprocessing data. For example, this paper analyzed sheep postures using various classifier algorithms.  
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energy Management

In our lessons on [time series forecasting](../../7-TimeSeries/README.md), we discussed smart parking meters. This article explores how clustering, regression, and time series forecasting were used to predict energy usage in Ireland based on smart metering.  
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Insurance

The insurance sector uses ML to build and optimize financial and actuarial models.

### Volatility Management

MetLife, a life insurance provider, uses ML to analyze and mitigate volatility in financial models. This article includes examples of binary and ordinal classification visualizations, as well as forecasting visualizations.  
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Arts, Culture, and Literature

In the arts, challenges like detecting fake news or optimizing museum operations can benefit from ML.

### Fake News Detection

Detecting fake news is a pressing issue. This article describes a system that combines ML techniques like Naive Bayes, SVM, Random Forest, SGD, and Logistic Regression to combat misinformation.  
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

### Museum ML

Museums are leveraging ML to catalog collections and predict visitor interests. For example, the Art Institute of Chicago uses ML to optimize visitor experiences and predict attendance with high accuracy.  
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Customer Segmentation

Effective marketing strategies often rely on customer segmentation. This article discusses how clustering algorithms can support differentiated marketing, improving brand recognition and revenue.  
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Challenge

Identify another sector that benefits from some of the techniques you learned in this curriculum, and discover how it uses ML.
## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

The Wayfair data science team has some fascinating videos about how they apply ML in their organization. It's definitely [worth checking out](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Assignment

[A ML scavenger hunt](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.