# Building Machine Learning solutions with responsible AI
 
![Summary of responsible AI in Machine Learning in a sketchnote](../../sketchnotes/ml-fairness.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)
 
## Introduction

In this curriculum, you will start to discover how machine learning can and is impacting our everyday lives. Even now, systems and models are involved in daily decision-making tasks, such as health care diagnoses, loan approvals or detecting fraud. So, it is important that these models work well to provide outcomes that are trustworthy. Just as any software application, AI systems are going to miss expectations or have an undesirable outcome. That is why it is essential to be about to understand and explain the behavior of an AI model. 

Imagine what can happen when the data you are using to build these models lacks certain demographics, such as race, gender, political view, religion, or disproportionally represents such demographics. What about when the model’s output is interpreted to favor some demographic? What is the consequence for the application? In addition, what happens when the model has an adverse outcome and is harmful to people? Who is accountable for the AI systems behavior? These are some questions we will explore in this curriculum. 

In this lesson, you will: 

- Raise your awareness of the importance of fairness in machine learning and fairness-related harms.
- Become familiar with the practice of exploring outliers and unusual scenarios to ensure reliability and safety
- Gain understanding on the need to empower everyone by designing inclusive systems
- Explore how vital it is to protect privacy and security of data and people
- See the importance of having a glass box approach to explain the behavior of AI models
- Be mindful of how accountability is essential to build trust in AI systems

## Prerequisite

As a prerequisite, please take the "Responsible AI Principles" Learn Path and watch the video below on the topic:

Learn more about Responsible AI by following this [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Microsoft's Approach to Responsible AI

## Fairness

AI systems should treat everyone fairly and avoid affecting similar groups of people in different ways. For example, when AI systems provide guidance on medical treatment, loan applications, or employment, they should make the same recommendations to everyone with similar symptoms, financial circumstances, or professional qualifications. Each of us as humans carries around inherited biases that affect our decisions and actions. These biases can be evident in the data that we use to train AI systems. Such manipulation can sometimes happen unintentionally. It is often difficult to consciously know when you are introducing bias in data. 

**“Unfairness”** encompasses negative impacts, or “harms”, for a group of people, such as those defined in terms of race, gender, age, or disability status. The main fairness-related harms can be classified as: 

- **Allocation**, if a gender or ethnicity for example is favored over another.
- **Quality of service**. If you train the data for one specific scenario but reality is much more complex, it leads to a poor performing service.  For instance, a hand soap dispenser that could not seem to be able to sense people with dark skin. [Reference](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigration**. To unfairly criticize and label something or someone. For example, an image labeling technology infamously mislabeled images of dark-skinned people as gorillas.
- **Over- or under- representation**. The idea is that a certain group is not seen in a certain profession, and any service or function that keeps promoting that is contributing to harm.
- **Stereotyping**. Associating a given group with pre-assigned attributes.  For example, a language translation system betweem English and Turkish may have inaccuraces due to words with stereotypical associations to gender.

![translation to Turkish](images/gender-bias-translate-en-tr.png)
> translation to Turkish

![translation back to English](images/gender-bias-translate-tr-en.png)
> translation back to English

When designing and testing AI systems, we need to ensure that AI is fair and not programmed to make biased or discriminatory decisions, which human beings are also prohibited from making. Guaranteeing fairness in AI and machine learning remains a complex sociotechnical challenge. 

### Reliability and safety

To build trust, AI systems need to be reliable, safe, and consistent under normal and unexpected conditions. It is important to know how AI systems will behavior in a variety of situations, especially when they are outliers. When building AI solutions, there needs to be a substantial amount of focus on how to handle a wide variety of circumstances that the AI solutions would encounter. For example, a self-driving car needs to put people's safety as a top priority. As a result, the AI powering the car need to consider all the possible scenarios that the car could come across such as night, thunderstorms or blizzards, kids running across the street, pets, road constructions etc. How well an AI system can handle a wild range of conditions reliably and safely reflects the level of anticipation the data scientist or AI developer considered during the design or testing of the system.  

> [🎥 Click the here for a video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusiveness

AI systems should be designed to engage and empower everyone. When designing and implementing AI systems data scientists and AI developers identify and address potential barriers in the system that could unintentionally exclude people. For example, there are 1 billion people with disabilities around the world. With the advancement of AI, they can access a wide range of information and opportunities more easily in their daily lives. By addressing the barriers, it creates opportunities to innovate and develop AI products with better experiences that benefit everyone. 

> [🎥 Click the here for a video: inclusiveness in AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Security and privacy 

AI systems should be safe and respect people’s privacy. People have less trust in systems that put their privacy, information, or lives at risk. When training machine learning models, we rely on data to produce the best results. In doing so, the origin of the data and integrity must be considered. For example, was the data user submitted or publicly available? Next, while working with the data, it is crucial to develop AI systems that can protect confidential information and resist attacks. As AI becomes more prevalent, protecting privacy and securing important personal and business information is becoming more critical and complex. Privacy and data security issues require especially close attention for AI because access to data is essential for AI systems to make accurate and informed predictions and decisions about people. 

> [🎥 Click the here for a video: security in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- As an industry we have made significant advancements in Privacy & security, fueled significantly by regulations like the GDPR (General Data Protection Regulation). 
- Yet with AI systems we must acknowledge the tension between the need for more personal data to make systems more personal and effective – and privacy. 
- Just like with the birth of connected computers with the internet, we are also seeing a huge uptick in the number of security issues related to AI. 
- At the same time, we have seen AI being used to improve security. As an example, most modern anti-virus scanners are driven by AI heuristics today. 
- We need to ensure that our Data Science processes blend harmoniously with the latest privacy and security practices. 


### Transparency
AI systems should be understandable. A crucial part of transparency is explaining the behavior of AI systems and their components. Improving the understanding of AI systems requires that stakeholders comprehend how and why they function so that they can identify potential performance issues, safety and privacy concerns, biases, exclusionary practices, or unintended outcomes. We also believe that those who use AI systems should be honest and forthcoming about when, why, and how they choose to deploy them. As well as the limitations of the systems they use. For example, if a bank uses an AI system to support its consumer lending decisions, it is important to examine the outcomes and understand which data influences the system’s recommendations. Governments are starting to regulate AI across industries, so data scientists and organizations must explain if an AI system meets regulatory requirements, especially when there is an undesirable outcome. 

> [🎥 Click the here for a video: transparency in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Because AI systems are so complex, it is hard to understand how they work and interpret the results. 
- This lack of understanding affects the way these systems are managed, operationalized, and documented. 
- This lack of understanding more importantly affects the decisions made using the results these systems produce. 

### Accountability 
 
The people who design and deploy AI systems must be accountable for how their systems operate. The need for accountability is particularly crucial with sensitive use technologies like facial recognition. Recently, there has been a growing demand for facial recognition technology, especially from law enforcement organizations who see the potential of the technology in uses like finding missing children. However, these technologies could potentially be used by a government to put their citizens’ fundamental freedoms at risk by, for example, enabling continuous surveillance of specific individuals. Hence, data scientists and organizations need to be responsible for how their AI system impacts individuals or society.

[![Leading AI Researcher Warns of Mass Surveillance Through Facial Recognition](images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Warnings of Mass Surveillance Through Facial Recognition 

Ultimately one of the biggest questions for our generation, as the first generation that is bringing AI to society, is how to ensure that computers will remain accountable to people and how to ensure that the people that design computers remain accountable to everyone else.

## Impact assessment 

Before training a machine learning model, it is important to conduct an impact assessmet to understand the purpose of the AI system; what the intended use is; where it will be deployed; and who will be interacting with the system.  These are helpful for reviewer(s) or testers evaluating the system to know what factors to take into consideration when identifying potential risks and expected consequences.

The following are areas of focus when conducting an impact assessment:

* **Adverse impact on individuals**.  Being aware of any restriction or requirements, unsupported use or any known limitations hindering the system's performance is vital to ensure that the system is not used in a way that could cause harm to individuals.
* **Data requirements**.  Gaining an understanding of how and where the system will use data enables reviewers to explore any data requirements you would need to be mindful of (e.g., GDPR or HIPPA data regulations).  In addition, examine whether the source or quantity of data is substantial for training.
* **Summary of impact**.  Gather a list of potential harms that could  arise from using the system.  Throughout the ML lifecycle, review if the issues identified are mitigated or addressed.
* **Applicable goals** for each of the six core principles.  Assess if the goals from each of the principles are met and if there are any gaps.


## Debugging with responsible AI  

Similar to debugging a software application, debugging an AI system is a necessary process of identifying and resolving issues in the system.  There are many factors that would affect a model not performing as expected or responsibly.  Most traditional model performance metrics are quantitative aggregates of a model's performance, which are not sufficient to analyze how a model violates the responsible AI principles. Furthermore, a machine learning model is a black box that makes it difficult to understand what drives its outcome or provide explanation when it makes a mistake.  Later in this course, we will learn how to use the Responsible AI dashboard to help debug AI systems.  The dashboard provides a holistic tool for data scientists and AI developers to perform:

* **Error analysis**.  To identify the error distribution of the model that can affect the system's fairness or reliability.
* **Model overview**. To discover where there are disparities in the model's performance across data cohorts.
* **Data analysis**.  To understand the data distribution and identify any potential bias in the data that could lead to fairness, inclusiveness, and reliability issues.
* **Model interpretability**. To understand what affects or influences the model's predictions. This helps in explaining the model's behavior, which is important for transparency and accountability.


## 🚀 Challenge 
 
To prevent harms from being introduced in the first place, we should: 

- have a diversity of backgrounds and perspectives among the people working on systems 
- invest in datasets that reflect the diversity of our society 
- develop better methods throughout the machine learning lifecycle for detecting and correcting responible AI when it occurs 

Think about real-life scenarios where a model's untrustworthiness is evident in model-building and usage. What else should we consider? 

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Review & Self Study 
 
In this lesson, you have learned some basics of the concepts of fairness and unfairness in machine learning.  
 
Watch this workshop to dive deeper into the topics: 

- In pursuit of responsible AI: Bringing principles to practice by Besmira Nushi, Mehrnoosh Sameki and Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Click the image above for a video: RAI Toolbox: An open-source framework for building responsible AI by Besmira Nushi, Mehrnoosh Sameki, and Amit Sharma

Also, read: 

- Microsoft’s RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoft’s FATE research group: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Read about Azure Machine Learning's tools to ensure fairness:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Assignment

[Explore RAI Toolbox](assignment.md) 
