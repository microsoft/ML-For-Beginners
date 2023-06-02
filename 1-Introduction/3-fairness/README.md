# Practicing responsible AI in Machine Learning  
 
![Summary of responsible AI in Machine Learning in a sketchnote](../../sketchnotes/ml-fairness.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)
 
## Introduction

In this curriculum, you will start to discover how machine learning can and is impacting our everyday lives. Even now, systems and models are involved in daily decision-making tasks, such as health care diagnoses, loan approvals or detecting fraud. So, it is important that these models work well to provide outcomes that are trustworthy. Just as any software application, AI systems are going to miss expectations or have an undesirable outcome. That is why it is essential to be about to understand and explain the behavior of an AI model. 

Imagine what can happen when the data you are using to build these models lacks certain demographics, such as race, gender, political view, religion, or disproportionally represents such demographics. What about when the model’s output is interpreted to favor some demographic? What is the consequence for the application? In addition, what happens when the model has an adverse outcome and is harmful to people? Who is accountable for the AI systems behavior? These are some questions we will explore in this curriculum. 

In this lesson, you will: 

- Raise your awareness of the importance of fairness in machine learning.
- Learn about fairness-related harms.
- Learn about unfairness assessment and mitigation.

## Prerequisite

As a prerequisite, please take the "Responsible AI Principles" Learn Path and watch the video below on the topic:

Learn more about Responsible AI by following this [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Microsoft's Approach to Responsible AI

## Fairness

AI systems should treat everyone fairly and avoid affecting similar groups of people in different ways. For example, when AI systems provide guidance on medical treatment, loan applications, or employment, they should make the same recommendations to everyone with similar symptoms, financial circumstances, or professional qualifications. Each of us as humans carries around inherited biases that affect our decisions and actions. These biases can be evident in the data that we use to train AI systems. Such manipulation can sometimes happen unintentionally. It is often difficult to consciously know when you are introducing bias in data. 

**“Unfairness”** encompasses negative impacts, or “harms”, for a group of people, such as those defined in terms of race, gender, age, or disability status. The main fairness-related harms can be classified as: 

- **Allocation**, if a gender or ethnicity for example is favored over another.
- **Quality of service**. If you train the data for one specific scenario but reality is much more complex, it leads to a poor performing service.
- **Stereotyping**. Associating a given group with pre-assigned attributes.
- **Denigration**. To unfairly criticize and label something or someone.
- **Over- or under- representation**. The idea is that a certain group is not seen in a certain profession, and any service or function that keeps promoting that is contributing to harm.

When designing and testing AI systems, we need to ensure that AI is fair and not programmed to make biased or discriminatory decisions, which human beings are also prohibited from making. Guaranteeing fairness in AI and machine learning remains a complex sociotechnical challenge. 

### Reliability and safety

To build trust, AI systems need to be reliable, safe, and consistent under normal and unexpected conditions. It is important to know how AI systems will behavior in a variety of situations, especially when they are outliers. When building AI solutions, there needs to be a substantial amount of focus on how to handle a wide variety of circumstances that the AI solutions would encounter. 

For example, a self-driving car needs to put people's safety as a top priority. As a result, the AI powering the car need to consider all the possible scenarios that the car could come across such as night, thunderstorms or blizzards, kids running across the street, pets, road constructions etc. How well an AI system can handle a wild range of conditions reliably and safely reflects the level of anticipation the data scientist or AI developer considered during the design or testing of the system.  

<!-- [![Implementing reliability & safety in AI ](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Ensure reliability and safety in AI -->

### Inclusiveness

AI systems should be designed to engage and empower everyone. When designing and implementing AI systems data scientists and AI developers identify and address potential barriers in the system that could unintentionally exclude people. For example, there are 1 billion people with disabilities around the world. With the advancement of AI, they can access a wide range of information and opportunities more easily in their daily lives. By addressing the barriers, it creates opportunities to innovate and develop AI products with better experiences that benefit everyone. 

![Inclusive systems for accessibility](images/accessibility.png)
> Inclusive systems for accessibility 

### Security and privacy 

AI systems should be safe and respect people’s privacy. People have less trust in systems that put their privacy, information, or lives at risk. When training machine learning models, we rely on data to produce the best results. In doing so, the origin of the data and integrity must be considered. For example, was the data user submitted or publicly available? 

Next, while working with the data, it is crucial to develop AI systems that can protect confidential information and resist attacks. As AI becomes more prevalent, protecting privacy and securing important personal and business information is becoming more critical and complex. Privacy and data security issues require especially close attention for AI because access to data is essential for AI systems to make accurate and informed predictions and decisions about people. 

- As an industry we have made significant advancements in Privacy & security, fueled significantly by regulations like the GDPR (General Data Protection Regulation). 
- Yet with AI systems we must acknowledge the tension between the need for more personal data to make systems more personal and effective – and privacy. 
- Just like with the birth of connected computers with the internet, we are also seeing a huge uptick in the number of security issues related to AI. 
- At the same time, we have seen AI being used to improve security. As an example, most modern anti-virus scanners are driven by AI heuristics today. 
- We need to ensure that our Data Science processes blend harmoniously with the latest privacy and security practices. 


### Transparency
AI systems should be understandable. A crucial part of transparency is explaining the behavior of AI systems and their components. Improving the understanding of AI systems requires that stakeholders comprehend how and why they function so that they can identify potential performance issues, safety and privacy concerns, biases, exclusionary practices, or unintended outcomes. We also believe that those who use AI systems should be honest and forthcoming about when, why, and how they choose to deploy them. As well as the limitations of the systems they use. 

For example, if a bank uses an AI system to support its consumer lending decisions, it is important to examine the outcomes and understand which data influences the system’s recommendations. Governments are starting to regulate AI across industries, so data scientists and organizations must explain if an AI system meets regulatory requirements, especially when there is an undesirable outcome. 

- Because AI systems are so complex, it is hard to understand how they work and interpret the results. 
- This lack of understanding affects the way these systems are managed, operationalized, and documented. 
- This lack of understanding more importantly affects the decisions made using the results these systems produce. 

### Accountability 
 
The people who design and deploy AI systems must be accountable for how their systems operate. The need for accountability is particularly crucial with sensitive use technologies like facial recognition. Recently, there has been a growing demand for facial recognition technology, especially from law enforcement organizations who see the potential of the technology in uses like finding missing children. However, these technologies could potentially be used by a government to put their citizens’ fundamental freedoms at risk by, for example, enabling continuous surveillance of specific individuals. Hence, data scientists and organizations need to be responsible for how their AI system impacts individuals or society.

[![Leading AI Researcher Warns of Mass Surveillance Through Facial Recognition](images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Warnings of Mass Surveillance Through Facial Recognition 

One of the biggest questions for our generation, as the first generation that is bringing AI to society, is how to ensure that computers will remain accountable to people and how to ensure that the people that design computers remain accountable to everyone else. 

Let us look at the examples. 

#### Allocation
Consider a hypothetical system for screening loan applications. The system tends to pick white men as better candidates over other groups. As a result, loans are withheld from certain applicants. 

Another example would be an experimental hiring tool developed by a large corporation to screen candidates. The tool systemically discriminated against one gender by using the models were trained to prefer words associated with another. It resulted in penalizing candidates whose resumes contain words such as “women’s rugby team”. 

✅ Do a little research to find a real-world example of something like this.

#### Quality of Service 
Researchers found that several commercial gender classifiers had higher error rates around images of women with darker skin tones as opposed to images of men with lighter skin tones. [Reference](https://www.media.mit.edu/publications/gender-shades-intersectional-accuracy-disparities-in-commercial-gender-classification/) 

Another infamous example is a hand soap dispenser that could not seem to be able to sense people with dark skin. [Reference](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773) 

#### Stereotyping
A stereotypical gender view was found in machine translation. When translating “he is a nurse and she is a doctor” into Turkish, problems were encountered. Turkish is a genderless language which has one pronoun, “o” to convey a singular third person, but translating the sentence back from Turkish to English yields the stereotypical and incorrect as “she is a nurse, and he is a doctor.” 

![translation to Turkish](images/gender-bias-translate-en-tr.png)
> translation to Turkish

![translation back to English](images/gender-bias-translate-tr-en.png)
> translation back to English

#### Denigration
 An image labeling technology infamously mislabeled images of dark-skinned people as gorillas. Mislabeling is harmful not just because the system made a mistake because it specifically applied a label that has a long history of being purposefully used to denigrate Black people. 

 [![AI: Ain't I a Woman?](https://img.youtube.com/vi/QxuyfWoVV98/0.jpg)](https://www.youtube.com/watch?v=QxuyfWoVV98 "AI, Ain't I a Woman?")
> 🎥 Click the image above for a video: AI, Ain't I a Woman - a performance showing the harm caused by racist denigration by AI

#### Over-representation or under-representation
Skewed image search results can be a good example of this harm. When searching images of professions with an equal or higher percentage of men than women, such as engineering, or CEO, watch for results that are more heavily skewed towards a given gender. 

![Bing search for 'CEO'](images/ceos.png)
> This search on Bing for ‘CEO’ produces inclusive results 

These five main types of harm are not mutually exclusive, and a single system can exhibit more than one type of harm. In addition, each case varies in its severity. For instance, unfairly labeling someone as a criminal is a much more severe harm than mislabeling an image. It is important, however, to remember that even relatively non-severe harms can make people feel alienated or singled out and the cumulative impact can be extremely oppressive. 

✅ **Discussion**: Revisit some of the examples and see if they show different harms.  

|                         | Allocation | Quality of service | Stereotyping | Denigration | Over- or under- representation |
| ----------------------- | :--------: | :----------------: | :----------: | :---------: | :----------------------------: |
| Automated hiring system |     x      |         x          |      x       |             |               x                |
| Machine translation     |            |                    |              |             |                                |
| Photo labeling          |            |                    |              |             |                                |


## Detecting unfairness 

There are many reasons why a given system behaves unfairly. Social biases, for example, might be reflected in the datasets used to train them. For example, hiring unfairness might have been exacerbated by over reliance on historical data. By using the patterns in resumes submitted to the company over a 10-year period, the model determined that men were more qualified because many resumes came from men, a reflection of past male dominance across the tech industry. 

Inadequate data about a certain group of people can be the reason for unfairness. For example, image classifiers have a higher rate of error for images of dark-skinned people because darker skin tones were underrepresented in the data. 

Wrong assumptions made during development cause unfairness too. For example, a facial analysis system intended to predict who is going to commit a crime based on images of people’s faces can lead to damaging assumptions. This could lead to substantial harm for people who are misclassified. 

## Understand your models and build in fairness
 
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
  
- Learn [how to enable fairness assessments](https://docs.microsoft.com/azure/machine-learning/how-to-machine-learning-fairness-aml?WT.mc_id=academic-77952-leestott) of machine learning models in Azure Machine Learning. 
  
- Check out these [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/contrib/fairness) for more fairness assessment scenarios in Azure Machine Learning. 

---
## 🚀 Challenge 
 
To prevent biases from being introduced in the first place, we should: 

- have a diversity of backgrounds and perspectives among the people working on systems 
- invest in datasets that reflect the diversity of our society 
- develop better methods for detecting and correcting bias when it occurs 

Think about real-life scenarios where unfairness is evident in model-building and usage. What else should we consider? 

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Review & Self Study 
 
In this lesson, you have learned some basics of the concepts of fairness and unfairness in machine learning.  
 
Watch this workshop to dive deeper into the topics: 

- Fairness-related harms in AI systems: Examples, assessment, and mitigation by Hanna Wallach and Miro Dudik 

[![Fairness-related harms in AI systems: Examples, assessment, and mitigation](https://img.youtube.com/vi/1RptHwfkx_k/0.jpg)](https://www.youtube.com/watch?v=1RptHwfkx_k "Fairness-related harms in AI systems: Examples, assessment, and mitigation")
> 🎥 Click the image above for a video: Fairness-related harms in AI systems: Examples, assessment, and mitigation by Hanna Wallach and Miro Dudik

Also, read: 

- Microsoft’s RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoft’s FATE research group: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

Explore the Fairlearn toolkit:

- [Fairlearn](https://fairlearn.org/)

Read about Azure Machine Learning's tools to ensure fairness:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Assignment

[Explore Fairlearn](assignment.md) 
