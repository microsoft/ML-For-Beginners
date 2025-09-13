<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-06T10:53:05+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "en"
}
-->
# Building Machine Learning solutions with responsible AI

![Summary of responsible AI in Machine Learning in a sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

In this curriculum, you will begin to explore how machine learning is shaping and influencing our daily lives. Today, systems and models play a role in decision-making processes such as healthcare diagnoses, loan approvals, or fraud detection. It is crucial that these models perform well and deliver trustworthy outcomes. Like any software application, AI systems can fail to meet expectations or produce undesirable results. This is why it is essential to understand and explain the behavior of AI models.

Consider what might happen if the data used to build these models lacks representation of certain demographics, such as race, gender, political views, or religion, or if it disproportionately represents some groups. What if the model’s output favors one demographic over another? What are the consequences for the application? Furthermore, what happens when the model causes harm? Who is accountable for the behavior of AI systems? These are some of the questions we will explore in this curriculum.

In this lesson, you will:

- Learn about the importance of fairness in machine learning and the harms related to unfairness.
- Understand the practice of exploring outliers and unusual scenarios to ensure reliability and safety.
- Recognize the need to design inclusive systems that empower everyone.
- Explore the importance of protecting privacy and security for both data and individuals.
- Understand the value of a transparent approach to explain AI model behavior.
- Appreciate how accountability is key to building trust in AI systems.

## Prerequisite

Before starting, please complete the "Responsible AI Principles" Learn Path and watch the video below on the topic:

Learn more about Responsible AI by following this [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Microsoft's Approach to Responsible AI

## Fairness

AI systems should treat everyone fairly and avoid disadvantaging similar groups of people. For example, when AI systems provide recommendations for medical treatment, loan applications, or employment, they should offer the same guidance to individuals with similar symptoms, financial situations, or qualifications. As humans, we all carry biases that influence our decisions and actions. These biases can also appear in the data used to train AI systems, sometimes unintentionally. It can be challenging to recognize when bias is being introduced into data.

**“Unfairness”** refers to negative impacts, or “harms,” experienced by a group of people, such as those defined by race, gender, age, or disability. The main types of fairness-related harms include:

- **Allocation**: Favoring one gender, ethnicity, or group over another.
- **Quality of service**: Training data for a specific scenario while ignoring the complexity of real-world situations, leading to poor performance. For example, a soap dispenser that fails to detect people with dark skin. [Reference](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigration**: Unfairly criticizing or labeling someone or something. For instance, an image labeling system that mislabeled dark-skinned individuals as gorillas.
- **Over- or under-representation**: When a group is absent or underrepresented in a profession, and systems perpetuate this imbalance.
- **Stereotyping**: Associating a group with predefined attributes. For example, a language translation system between English and Turkish may produce errors due to gender-based stereotypes.

![translation to Turkish](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> translation to Turkish

![translation back to English](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> translation back to English

When designing and testing AI systems, it is essential to ensure that AI is fair and does not make biased or discriminatory decisions—just as humans are prohibited from doing. Achieving fairness in AI and machine learning is a complex sociotechnical challenge.

### Reliability and safety

To build trust, AI systems must be reliable, safe, and consistent under both normal and unexpected conditions. It is important to understand how AI systems behave in various situations, especially outliers. When developing AI solutions, significant attention must be given to handling a wide range of scenarios the system might encounter. For example, a self-driving car must prioritize people’s safety. The AI powering the car must account for scenarios like nighttime driving, thunderstorms, blizzards, children running into the street, pets, road construction, and more. The reliability and safety of an AI system reflect the level of foresight and preparation by the data scientist or AI developer during design and testing.

> [🎥 Click here for a video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusiveness

AI systems should be designed to engage and empower everyone. Data scientists and AI developers must identify and address potential barriers in the system that could unintentionally exclude people. For example, there are 1 billion people with disabilities worldwide. Advances in AI can help them access information and opportunities more easily in their daily lives. Addressing barriers creates opportunities to innovate and develop AI products that provide better experiences for everyone.

> [🎥 Click here for a video: inclusiveness in AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Security and privacy

AI systems must be safe and respect people’s privacy. People are less likely to trust systems that put their privacy, information, or lives at risk. When training machine learning models, data is essential for producing accurate results. However, the origin and integrity of the data must be considered. For example, was the data user-submitted or publicly available? Additionally, AI systems must protect confidential information and resist attacks. As AI becomes more widespread, safeguarding privacy and securing personal and business information are increasingly critical and complex. Privacy and data security require special attention because AI systems rely on data to make accurate predictions and decisions.

> [🎥 Click here for a video: security in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- The industry has made significant progress in privacy and security, driven by regulations like GDPR (General Data Protection Regulation).
- However, AI systems face a tension between the need for personal data to improve effectiveness and the need to protect privacy.
- Just as the internet brought new security challenges, AI has led to a rise in security issues.
- At the same time, AI is being used to enhance security, such as in modern antivirus scanners powered by AI heuristics.
- Data science processes must align with the latest privacy and security practices.

### Transparency

AI systems should be understandable. Transparency involves explaining the behavior of AI systems and their components. Stakeholders need to understand how and why AI systems function to identify potential performance issues, safety and privacy concerns, biases, exclusionary practices, or unintended outcomes. Those who use AI systems should also be transparent about when, why, and how they deploy them, as well as the systems’ limitations. For example, if a bank uses AI for lending decisions, it is important to examine the outcomes and understand which data influences the system’s recommendations. Governments are beginning to regulate AI across industries, so data scientists and organizations must ensure their systems meet regulatory requirements, especially in cases of undesirable outcomes.

> [🎥 Click here for a video: transparency in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- AI systems are complex, making it difficult to understand how they work and interpret their results.
- This lack of understanding affects how systems are managed, operationalized, and documented.
- More importantly, it impacts the decisions made based on the systems’ results.

### Accountability

The people who design and deploy AI systems must be accountable for their operation. Accountability is especially critical for sensitive technologies like facial recognition. For example, law enforcement agencies may use facial recognition to find missing children, but the same technology could enable governments to infringe on citizens’ freedoms through continuous surveillance. Data scientists and organizations must take responsibility for how their AI systems impact individuals and society.

[![Leading AI Researcher Warns of Mass Surveillance Through Facial Recognition](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Warnings of Mass Surveillance Through Facial Recognition

One of the most significant questions for our generation, as the first to bring AI to society, is how to ensure that computers remain accountable to people and that the people designing these systems remain accountable to everyone else.

## Impact assessment

Before training a machine learning model, it is important to conduct an impact assessment to understand the purpose of the AI system, its intended use, where it will be deployed, and who will interact with it. This helps reviewers or testers identify potential risks and expected consequences.

Key areas to focus on during an impact assessment include:

- **Adverse impact on individuals**: Be aware of any restrictions, requirements, unsupported uses, or known limitations that could hinder the system’s performance and cause harm.
- **Data requirements**: Understand how and where the system will use data to identify any regulatory requirements (e.g., GDPR or HIPAA) and ensure the data source and quantity are sufficient for training.
- **Summary of impact**: Identify potential harms that could arise from using the system and review whether these issues are addressed throughout the ML lifecycle.
- **Applicable goals** for each of the six core principles: Assess whether the goals of each principle are met and identify any gaps.

## Debugging with responsible AI

Debugging an AI system is similar to debugging a software application—it involves identifying and resolving issues. Many factors can cause a model to perform unexpectedly or irresponsibly. Traditional model performance metrics, which are often quantitative aggregates, are insufficient for analyzing how a model violates responsible AI principles. Additionally, machine learning models are often black boxes, making it difficult to understand their outcomes or explain their mistakes. Later in this course, we will learn how to use the Responsible AI dashboard to debug AI systems. This dashboard provides a comprehensive tool for data scientists and AI developers to:

- **Perform error analysis**: Identify error distributions that affect fairness or reliability.
- **Review the model overview**: Discover disparities in the model’s performance across data cohorts.
- **Analyze data**: Understand data distribution and identify potential biases that could impact fairness, inclusiveness, and reliability.
- **Interpret the model**: Understand what influences the model’s predictions, which is essential for transparency and accountability.

## 🚀 Challenge

To prevent harms from being introduced in the first place, we should:

- Ensure diversity in the backgrounds and perspectives of the people working on AI systems.
- Invest in datasets that reflect the diversity of society.
- Develop better methods throughout the machine learning lifecycle to detect and address responsible AI issues.

Think about real-life scenarios where a model’s lack of trustworthiness is evident during development or use. What else should we consider?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

In this lesson, you have learned the basics of fairness and unfairness in machine learning.
Watch this workshop to explore the topics in more detail:

- In pursuit of responsible AI: Applying principles in practice by Besmira Nushi, Mehrnoosh Sameki, and Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Click the image above to watch the video: RAI Toolbox: An open-source framework for building responsible AI by Besmira Nushi, Mehrnoosh Sameki, and Amit Sharma

Additionally, check out:

- Microsoft’s RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft’s FATE research group: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Learn about Azure Machine Learning's tools for ensuring fairness:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Assignment

[Explore RAI Toolbox](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.