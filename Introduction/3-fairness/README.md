# Fairness in Machine Learning 
 
TODO: Illustration “black box” here 

## [Pre-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/5/)
 
## Introduction 
 
In this curriculum, you will start to discover how machine learning can and is impacting our everyday lives. Even now, systems and models are involved in daily decision-making tasks, such as health care diagnoses or detecting fraud. So it is important that these models work well in order to provide fair outcomes for everyone.

Imagine what can happen when the data you are using to build these models lacks certain demographics, such as race, gender, political view, religion, or disproportionally represents such demographics. What about when the model's output is interpreted to favor some demographic? What is the consequence for the application? 

In this lesson, you will: 

- Raise your awareness of importance of fairness in machine learning
- Learn about fairness-related harms
- Learn about unfairness assessment and mitigation

## Prerequisite

As a prerequisite, please take the "Responsible AI Principles" Learn Path and watch the video below on the topic:

Learn more about Responsible AI by following this [Learning Path](https://docs.microsoft.com/en-us/learn/modules/responsible-ai-principles/?WT.mc_id=academic-15963-cxa)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")
> Video: Microsoft's Approach to Responsible AI
 
## Unfairness in data and algorithms 
 
> "If you torture the data long enough, it will confess to anything." - Ronald Coase

This sounds extreme but it is true that data can be manipulated to support any conclusion. Such manipulation can sometimes happen unintentionally. As humans, we all have bias, and you it is often difficult to consciously know when you are introducing bias in data. 

Guaranteeing fairness in AI and machine learning remains a complex sociotechnical challenge. This means that it cannot be addressed from either purely social or technical perspectives.



### Fairness-related harms 
 
What do you mean by unfairness? "Unfairness" encompasses negative impacts, or "harms", for a group of people, such as those defined in terms of race, gender, age, or disability status.  

The main fairness-related harms can be classified as: 

- Allocation 
- Quality of service 
- Stereotyping 
- Denigration 
- Over- or under- representation 

Let’s take a look at the examples— 
 
#### Allocation 

Consider a given system for screening loan applications. The system tends to pick white men as better candidates over other groups. As a result, loans are withheld from certain applicants. 

Another example would be an experimental hiring tool developed by a large corporation to screen candidates. The tool systemically discriminated against one gender by using the models were trained to prefer words associated with another. It resulted in penalizing candidates whose resumes contain words such as "women’s rugby team". 

✅ Do a little research to find a real-world example of something like this
 
#### Quality of service 
 
Researchers found that several commercial gender classifiers had higher error rates that images of women with darker skin tones than the images of men with lighter skin tones.  

#### Stereotyping 
 
Stereotypical gender view was found in machine translation. When translating “he is a nurse and she is a doctor” into Turkish, a genderless language, which has one pronoun, “o” to convey a singular third person, then back into English yields the stereotypical and incorrect as “she is a nurse and he is a doctor”. 

!["He's a nurse. She's a doctor." machine-translated in Turkish: "O bir hemşire. O bir doktor."](images/gender-bias-translate-en-tr.png) 
!["O bir hemşire. O bir doktor." machine-translated in English: "She's a nurse. He's a doctor."](images/gender-bias-translate-tr-en.png)

#### Denigration 
 
An image labeling technology infamously mislabeled images of dark-skinned people as gorillas. Mislabeling is harmful not just because the system made a mistake because it specifically applied a label that has a long history of being purposefully used to denigrate demean Black people. 
 
#### Over- or under- representation 
 
Skewed image search results can be a good example of this harm. When searching images of professions with an equal or higher percentage of men than women, such as engineering, or CEO results heavily skewed toward images of men than reality. 
 
There five main types of harms are not mutually exclusive, and a single system can exhibit more than one type of harms. 
Each case varies severities, for instance, unfairly labeling someone as a criminal is a much more severe harm than mislabeling an image but it's important to remember that even relatively non severe harms can make people feel alienated or singled out and the cumulative impact can be extremely oppressive. 
 
✅ **Discussion**: Revisit some of the examples and see if they show different harms.  

|                         | Allocation | Quality of service | Stereotyping | Denigration | Over- or under- representation |
| ----------------------- | :--------: | :----------------: | :----------: | :---------: | :----------------------------: |
| Automated hiring system | x          | x                  | x            |             | x                              |
| Machine translation     |            |                    |              |             |                                |
| Photo labeling          |            |                    |              |             |                                |

 
## Detecting unfairness 
 
There are many reasons why the system behaves unfairly— the reasons include societal biases reflected of the datasets used to train them. For example, the hiring unfairness was caused by the historical data, by using the patterns in resumes submitted to the company over a 10-year period, and the problem was that the majority came from men, a reflection of male dominance across the tech industry. 

Inadequate data points about a certain group of people can be the reason. For example, image classifiers have higher rate error for images of dark-skinned women because there aren’t enough dataset with darker skin tones to train.  

Wrong assumptions made during the development cases unfairness too. For example, facial analysis system to predict who is going to commit a crime based on images of people’s faces. The assumption to make it believe this system is capable of doing this could lead substantial harms for people who are misclassified. 
 
### Understand your models and build fairness 
 
Although many aspects of fairness are not captured in quantitative fairness metrics, and it is not possible to fully remove bias from a system to guarantee fairness, you are still responsible to detect and to mitigate fairness issues as much as possible. 
When you are working with machine learning models, it is important to understand your models with interpretability and assess and mitigate unfairness. 
Let’s use the loan selection example to isolate the case to figure out each factors level of impact on the prediction.  
 
### Assessment methods 
 

1. Identify the harms (and benefits)  
2. Identify the affected groups 
3. Define fairness metrics 

#### Identify the harms (and benefits) 
 
What are the harms and benefits associated with lending? Think false negatives and false positive scenarios: 

**False negatives** (reject, but Y=1) - when an applicant will be capable of repaying loan is rejected. This is an adverse event because the resources of the loans are withheld from qualified applicants. 

**False positives** (accept, but Y=0) - when the applicant does get the loan but eventually defaults. As the result, the applicant will be sent to the debt collection agencies, and possibly affects their future loan applications. 
Identify the affected groups 

#### Identify the affected groups 

The next step is to determine which groups are likely to be affected. 

For example, in case of a credit card application, where you see women are receiving much lower credit limits compared with their spouses who shares assets, the affected groups can be defined by the gender identity. 
 
#### Define fairness metrics 
 
You have identified harms and affected groups, in this case, gender. Now, use the quantified factors to disaggregate metrics. 

For example, when you have the data below, by examining this table, we see the women has the largest false positive rate and men has the smallest, and the opposite for false negatives. 

|            | False positive rate | False negative rate | count |
| ---------- | ------------------- | ------------------- | ----- |
| women      | 0.37                | 0.27                | 54032 |
| men        | 0.31                | 0.35                | 28620 |
| Non-binary | 0.33                | 0.31                | 1266  |

 
Also, note that this table also tells us that the non-binary people have much smaller count. It means the data is less certain and possibly has a larger error bars, so you need to be more careful how you interpret these numbers. 

So, in this case, we have 3 groups and 2 metrics. When we are thinking about how our system affects the group of customers/loan applicants, this may be sufficient, but when you want to define larger number of groups, you may want to distill this to smaller sets of summaries. To do that, you can add more metrics, such as largest difference, and smallest ratio of each false negative/positive rates. 
 
✅ **Discussion**: What other groups are likely to be affected for loan application? 
 
## Mitigating unfairness 
 
To mitigate the fairness issue, explorer the model to generate various mitigated models and compare them navigate tradeoffs between accuracy and fairness to select the model with the desired trade off. 

This intro lesson does not dive deeply into the details of algorithmic unfairness mitigation, such as post-processing and reductions approach, but introducing a tool that you may want to try. 

### Fairlearn 
 
\[Fairlearn\](https://fairlearn.github.io/) is an open-source Python package that allows you to assess your systems' fairness and mitigate unfairness.  
The tool may help you to assesses how a model's predictions affect different groups, enables comparing multiple models by using fairness and performance metrics, and supply a set of algorithms to mitigate unfairness in binary classification and regression. 

- Learn how to use the different components by checking out the Fairlearn's [GitHub](https://github.com/fairlearn/fairlearn/), [user guide](https://fairlearn.github.io/main/user_guide/index.html), [examples](https://fairlearn.github.io/main/auto_examples/index.html), and [sample notebooks](https://github.com/fairlearn/fairlearn/tree/master/notebooks). 
  
- Learn how to enable fairness assessment of machine learning models in [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-fairness-aml). 
  
- See the [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/contrib/fairness) for more fairness assessment scenarios in Azure Machine Learning. 


## 🚀 Challenge 
 
To avoid biases to be introduced in the first place, we should: 

- have a diversity of backgrounds and perspectives among the people working on systems 
- invest in datasets that reflect the diversity in our society 
- develop better methods for detecting and correcting bias when it occurs 

What else should we consider? 
 
## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/6/)

## Review & Self Study 
 
Watch this workshop to dive deeper into the topics:

📺 **Fairness-related harms in AI systems: Examples, assessment, and mitigation** by Hanna Wallach and Miro Dudik

[![Fairness-related harms in AI systems: Examples, assessment, and mitigation](https://img.youtube.com/vi/1RptHwfkx_k/0.jpg)](https://youtu.be/1RptHwfkx_k "Microsoft researchers Hanna Wallach and Miroslav Dudík will guide you through how AI systems can lead to a variety of fairness-related harms. ")
> Video: Microsoft researchers Hanna Wallach and Miroslav Dudík will guide you through how AI systems can lead to a variety of fairness-related harms.

📖 Also, read: 

- [Responsible AI Resources](https://www.microsoft.com/en-us/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) – Microsoft AI
- [FATE: Fairness, Accountability, Transparency, and Ethics in AI](https://www.microsoft.com/en-us/research/theme/fate/) - Microsoft Research
- [Fairlearn toolkit](https://fairlearn.org/)
- Azure Machine Learning [Machine learning fairness](https://docs.microsoft.com/en-us/azure/machine-learning/concept-fairness-ml)  

📺 Tech talk

[![Eric Horvitz discusses Ethical AI](https://img.youtube.com/vi/tL7t2O5Iu8E/0.jpg)](https://youtu.be/tL7t2O5Iu8E "Eric Horvitz, Technical Fellow and Director of Microsoft Research Labs, talks about some of the benefits AI and machine learning are bringing and why it is essential for companies to establish ethical principles to make sure AI is properly governed.")
> Video: Eric Horvitz, Technical Fellow and Director of Microsoft Research Labs, talks about some of the benefits AI and machine learning are bringing and why it is essential for companies to establish ethical principles to make sure AI is properly governed.


## [Assignment Name](assignment.md) 
