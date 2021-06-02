# Machine Learning in the Real World

In this curriculum, you have learned many ways to prepare data for training and create machine learning models. You built a series of classic Regression, Clustering, Classification, Natural Language Processing, and Time Series models. Congratulations! Now, you might be wondering what it's all for...what are the real world applications for these models?

While a lot of interest in industry has been garnered by AI, which usually leverages Deep Learning, there are still valuable applications for classical machine learning models, some of which you use today, although you might not be aware of it. In this lesson, you'll explore how ten different industries and subject-matter domains use these types of models to make their applications more performant, reliable, intelligent, and thus more valuable to users.
## [Pre-lecture quiz](link-to-quiz-app)

## Finance

One of the major consumers of classical machine learning models is the finance industry. Two specific examples we cover here are **credit card fraud detection** and **wealth management**. 

### Credit card fraud detection

We learned about [k-means clustering](Clustering/2-K-Means/README.md) earlier in the course, but how can it be used to solve problems related to credit card fraud?

K-means clustering comes in handy during a credit card fraud detection technique called **outlier detection**. Outliers, or deviations in observations about a set of data, can tell us if a credit card is being used in a normal capacity, or if something unusual is going on. As shown in [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf), you can sort credit card data using a k-means clustering algorithm and assign each transaction to a cluster based on how much of an outlier it appears to be. Then, you can evaluate for riskiest cluster for fraudulent versus legitimate transactions.

### Wealth management

In wealth management, an individual or firm handles investments on behalf of their clients. Their job is to sustain and grow wealth in the long-term, so it is essential to choose investments that perform well.

One way to evaluate how a particular investment performs is through statistical regression. [Linear regression](Regression/1-Tools/README.md) is a valuable tool for understanding how a fund performs relative to some benchmark. We can also deduce whether or not the results of the regression are statistically significant, or how much they would affect a client's investments. You could even further expand your analysis using multiple regression, where additional risk factors can be taken into account. For an example of how this would work for a specific fund, check out [this paper](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/) on evaluating fund performance using regression.

## Education

### Predicting student behavior
### Preventing plagiarism
### Course recommendations

## Retail

### Personalizing the customer journey

### Inventory management

## Health Care

### Optimizing drug delivery
### Hospital re-entry management
### Disease management

## Ecology and Green Tech

### Forest management
You learned about Reinforcement Learning in previous lessons. It can be very useful when trying to predict patterns in nature. In particular, it could be used to track ecological problems like forest fires and the spread of invasive species. In Canada, a group of researchers used Reinforcement Learning to build forest wildfire dynamics models from satellite images. Using an innovative "spatially spreading process (SSP)", they envisioned a forest fire as "the agent at any cell in the landscape". "The set of actions the fire can take from a location at any point in time includes spreading north, south, east, or west or not spreading. 

This approach inverts the usual RL setup since the dynamics of the corresponding Markov Decision Process (MDP) is a known function for immediate wildfire spread." Read more about the classic algorithms used by this group in this article: https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full

### Motion sensing of animals

While deep learning has created a revolution in visually-tracking animal movements (you can build your own [polar bear tracker](https://docs.microsoft.com/en-us/learn/modules/build-ml-model-with-azure-stream-analytics/) here), classic ML still has a place in this task.

Sensors to track movements of farm animals and IoT makes use of this type of visual processing, but more basic ML techniques are useful to preprocess data. For example, in this paper, sheep postures were monitored and analyzed using various classifier algorithms. You will recocgnize the ROC curve on p. 335: https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf

### Energy Management
  
In our lesson on Time Series, we invoked the concept of smart parking meters to generate revenue for a town based on understanding supply and demand. This article discusses in detail how clustering, regression and time series forecasting combined to help predict future energy use in Ireland, based off of smart metering: https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf

## Insurance

### Actuarial tasks

## Consumer Electronics

### Motion sensing
## Software

### UI regression
### Document search

### Recommendation engines

## Arts, Culture, and Literature

### Fake news detection

Detecting fake news has become a game of cat and mouse in today's media. In this article, researchers suggest that a system combining several of the ML techniques we have studied can be tested and the best model deployed: "This system is based on natural language processing to extract features from the data and then these features are used for the training of machine learning classifiers such as Naive Bayes,  Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), and Logistic Regression(LR)." 

https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf

This article shows how combining different ML domains can produce interesting results that can help stop fake news from spreading and creating real damage; in this case, the impetus was the spread of rumors about COVID treatments that incited mob violence.

### Museum ML

Museums are at the cusp of an AI revolution in which cataloging and digitizing collections and finding links between artifacts is becoming easier as technology becomes more easy to use. Projects such as In Codice Ratio are helping unlock the mysteries of inaccessible collections such as the Vatican Archives. But the business aspect of museums benefits from ML models as well. For example, the Art Institute of Chicago built models to predict what audiences are interested in and when they will attend expositions. The goals is to create individualized and optimized visitor experiences each time the user visit the museum. "During fiscal 2017, the model predicted attendance and admissions within 1 percent of accuracy, says Andrew Simnick, senior vice president at the Art Institute."

https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices

## Marketing

### 'Ad words'
### Customer segmentation




✅ Knowledge Check - use this moment to stretch students' knowledge with open questions


## 🚀Challenge

Add a challenge for students to work on collaboratively in class to enhance the project

## [Post-lecture quiz](link-to-quiz-app)

## Review & Self Study

**Assignment**: [A ML scavenger hunt](assignment.md)
