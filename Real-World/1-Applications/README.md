# Machine Learning in the Real World

In this curriculum, you have learned many ways to prepare data for training and create machine learning models. You built a series of classic Regression, Clustering, Classification, Natural Language Processing, and Time Series models. Congratulations! Now, you might be wondering what it's all for...what are the real world applications for these models?

While a lot of interest in industry has been garnered by AI, which usually leverages Deep Learning, there are still valuable applications for classical machine learning models, some of which you use today, although you might not be aware of it. In this lesson, you'll explore how ten different industries and subject-matter domains use these types of models to make their applications more performant, reliable, intelligent, and thus more valuable to users.
## [Pre-lecture quiz](link-to-quiz-app)

## 💰 Finance

One of the major consumers of classical machine learning models is the finance industry. Two specific examples we cover here are **credit card fraud detection** and **wealth management**. 

### Credit card fraud detection

We learned about [k-means clustering](Clustering/2-K-Means/README.md) earlier in the course, but how can it be used to solve problems related to credit card fraud?

K-means clustering comes in handy during a credit card fraud detection technique called **outlier detection**. Outliers, or deviations in observations about a set of data, can tell us if a credit card is being used in a normal capacity, or if something unusual is going on. As shown in [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf), you can sort credit card data using a k-means clustering algorithm and assign each transaction to a cluster based on how much of an outlier it appears to be. Then, you can evaluate for riskiest cluster for fraudulent versus legitimate transactions.

### Wealth management

In wealth management, an individual or firm handles investments on behalf of their clients. Their job is to sustain and grow wealth in the long-term, so it is essential to choose investments that perform well.

One way to evaluate how a particular investment performs is through statistical regression. [Linear regression](Regression/1-Tools/README.md) is a valuable tool for understanding how a fund performs relative to some benchmark. We can also deduce whether or not the results of the regression are statistically significant, or how much they would affect a client's investments. You could even further expand your analysis using multiple regression, where additional risk factors can be taken into account. For an example of how this would work for a specific fund, check out [this paper](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/) on evaluating fund performance using regression.

## 🎓 Education

### Predicting student behavior
### Preventing plagiarism
### Course recommendations

## 👜 Retail

### Personalizing the customer journey

At Wayfair, a company that sells home goods like furniture, helping customers find the right products for their taste and needs is paramount. In this article, engineers from the company describe how they use ML and NLP to "surface the right results for customers". Notably, their Query Intent Engine has been built to use entity extraction, classifier training, asset and opinion extraction, and sentiment tagging on customer reviews. This is a classic use case of how NLP works in online retail. 

https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search

### Inventory management

Innovative, nimble companies like [StitchFix](https://stitchfix.com), a box service that ships clothing to consumers, rely heavily on ML for recommendations and inventory management. Their styling teams work together with their merchandising teams, in fact: "one of our data scientists tinkered with a genetic algorithm and applied it to apparel to predict what would be a successful piece of clothing that doesn't exist today. We brought that to the merchandise team and now they can use that as a tool."

https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/

## 🏥 Health Care

### Managing clinical trials

Toxicity in clinical trials is a major concern to drug makers. How much toxicity is tolerable? In this study, analyzing various clinical trial methods led to the development of a new approach for predicting the odds of clinical trial outcomes using random forest to produce a classifier that is able to distinguish between groups of drugs.

https://www.sciencedirect.com/science/article/pii/S2451945616302914

### Hospital readmission management

Hospital care is costly, especially when patients have to be readmitted. This paper  discusses a company that uses ML to predict readmission potential by means of Clustering algorithms. These clusters help analysts to "discover groups of readmissions that may share a common cause".

https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning
### Disease management

The recent pandemic has shone a bright light on the ways that machine learning can aid in stopping the spread of disease. In this article, you'll recognise the use of ARIMA, logistic curves, linear regression, and SARIMA. "This work is an attempt to calculate the rate of spread of this virus and thus to predict the deaths, recoveries, and confirmed cases, so that it may help us to prepare better and survive."

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/

## 🌲 Ecology and Green Tech

### Forest management
You learned about Reinforcement Learning in previous lessons. It can be very useful when trying to predict patterns in nature. In particular, it could be used to track ecological problems like forest fires and the spread of invasive species. In Canada, a group of researchers used Reinforcement Learning to build forest wildfire dynamics models from satellite images. Using an innovative "spatially spreading process (SSP)", they envisioned a forest fire as "the agent at any cell in the landscape". "The set of actions the fire can take from a location at any point in time includes spreading north, south, east, or west or not spreading. 

This approach inverts the usual RL setup since the dynamics of the corresponding Markov Decision Process (MDP) is a known function for immediate wildfire spread." Read more about the classic algorithms used by this group in this article: https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full

### Motion sensing of animals

While deep learning has created a revolution in visually-tracking animal movements (you can build your own [polar bear tracker](https://docs.microsoft.com/en-us/learn/modules/build-ml-model-with-azure-stream-analytics/) here), classic ML still has a place in this task.

Sensors to track movements of farm animals and IoT makes use of this type of visual processing, but more basic ML techniques are useful to preprocess data. For example, in this paper, sheep postures were monitored and analyzed using various classifier algorithms. You will recocgnize the ROC curve on p. 335: https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf

### ⚡️ Energy Management
  
In our lesson on Time Series, we invoked the concept of smart parking meters to generate revenue for a town based on understanding supply and demand. This article discusses in detail how clustering, regression and time series forecasting combined to help predict future energy use in Ireland, based off of smart metering: https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf

## 💼 Insurance

### Volatility Management

MetLife is forthcoming with the way they analyze and mitigate volatility in their financial models. In this article you'll notice binary and ordinal classification visualizations. You'll also discover forecasting visualizations.

https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf

## 📱 Consumer Electronics

### Motion sensing
## 💻 Software

### UI regression
### Document search

### Recommendation engines

## 🎨 Arts, Culture, and Literature

### Fake news detection

Detecting fake news has become a game of cat and mouse in today's media. In this article, researchers suggest that a system combining several of the ML techniques we have studied can be tested and the best model deployed: "This system is based on natural language processing to extract features from the data and then these features are used for the training of machine learning classifiers such as Naive Bayes,  Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), and Logistic Regression(LR)." 

https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf

This article shows how combining different ML domains can produce interesting results that can help stop fake news from spreading and creating real damage; in this case, the impetus was the spread of rumors about COVID treatments that incited mob violence.
### Museum ML

Museums are at the cusp of an AI revolution in which cataloging and digitizing collections and finding links between artifacts is becoming easier as technology becomes more easy to use. Projects such as In Codice Ratio are helping unlock the mysteries of inaccessible collections such as the Vatican Archives. But the business aspect of museums benefits from ML models as well. 

For example, the Art Institute of Chicago built models to predict what audiences are interested in and when they will attend expositions. The goals is to create individualized and optimized visitor experiences each time the user visit the museum. "During fiscal 2017, the model predicted attendance and admissions within 1 percent of accuracy, says Andrew Simnick, senior vice president at the Art Institute."

https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices

## 🏷 Marketing

### 'Ad words'
### Customer segmentation

## 🚀Challenge

Discover one more sector that benefits from some of the techniques you learned in this curriculum.
## [Post-lecture quiz](link-to-quiz-app)

## Review & Self Study

Wayfair Data Science group has several interesting videos on how they use ML in their company. It's worth [taking a look](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

**Assignment**: [A ML scavenger hunt](assignment.md)
