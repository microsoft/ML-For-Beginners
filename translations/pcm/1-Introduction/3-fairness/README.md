# Building Machine Learning solutions with responsible AI
 
![Summary of responsible AI in Machine Learning in a sketchnote](../../../../translated_images/pcm/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
 
## Introduction

For dis curriculum, you go start to discover how machine learning fit dey affect our everyday life. Even now, system and model dey involved for daily decision-making work, like health care diagnoses, loan approvals or detecting fraud. So e make sense say dis models go work well to give outcomes wey people fit trust. Like any software app, AI systems fit miss expectations or give bad result. Na why e important to fit sabi and explain how AI model dey behave.

Imagine wetin fit happen if the data wey you dey use build this kind model no get some demographics, like race, gender, political view, religion, or e just too much for some demographics. Wetin you go do if the model output show say e favor some kind demographic? Wetin fit be the consequence for the app? Plus, wetin fit happen if the model get bad outcome wey fit harm people? Who dey responsible for as the AI system dey act? These na some questions we go reason for this curriculum.

For dis lesson, you go:

- Raise your awareness of the importance of fairness in machine learning and fairness-related harms.
- Become familiar with the practice of exploring outliers and unusual scenarios to ensure reliability and safety
- Gain understanding on the need to empower everyone by designing inclusive systems
- Explore how vital it is to protect privacy and security of data and people
- See the importance of having a glass box approach to explain the behavior of AI models
- Be mindful of how accountability is essential to build trust in AI systems

## Prerequisite

As prerequisite, abeg take the "Responsible AI Principles" Learn Path and watch the video wey dey below on top the topic:

Learn more about Responsible AI by following this [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Microsoft's Approach to Responsible AI

## Fairness

AI systems suppose make everybody fair and no treat similar groups of people different. For example, if AI systems dey give advice on medical treatment, loan applications, or work, dem suppose give same recommendation to everybody wey get similar symptoms, money situation, or professional qualification. Each of us as human carry inherited biases wey fit affect our decisions and actions. This kind biases fit show for the data wey we dey use train AI systems. Sometimes this kin thing fit happen without person intention. E hard to consciously sabi when you dey bring bias into data.

**“Unfairness”** na to cause bad impacts, or “harms”, to group of people, like those wey dem define based on race, gender, age, or disability status. The main fairness-related harms fit be:

- **Allocation**, if dem favor one gender or ethnicity pass another.
- **Quality of service**. If you train data for one kind scenario but reality na much more complex, e go lead to poor service. For example, hand soap dispenser wey no fit detect people with dark skin. [Reference](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigration**. To unfairly criticize or label person or thing. For example, image labeling technology wey wrongly label pictures of dark-skinned people as gorillas.
- **Over- or under- representation**. This na the idea say some group no dey for some job, and any service or function wey still dey promote that dey add to harm.
- **Stereotyping**. To put one group with pre-assigned attributes. For example, language translation system between English and Turkish fit get mistakes because of stereotypical links between words and gender.

![translation to Turkish](../../../../translated_images/pcm/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> translation to Turkish

![translation back to English](../../../../translated_images/pcm/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> translation back to English

When you dey design and test AI systems, you need make sure AI fair and no dey program to make biased or discriminatory decisions, the kain decision way human being self no suppose make. Guaranteeing fairness for AI and machine learning still na serious social and technical wahala.

### Reliability and safety

To build trust, AI systems suppose reliable, safe, and consistent for normal and unexpected situations. E important to sabi how AI systems go behave for many kinds situation, especially those wey unusual. When you dey build AI solutions, you need put plenty eye for how you go handle plenty different scenarios wey AI solutions fit face. For example, self-driving car suppose put people safety first. So, the AI wey dey power the car need consider all possible scenario like night, thunderstorm, blizzard, pikin wey dey run cross road, pets, road work and so on. How well AI system fit handle different condition well and safe show how data scientist or AI developer take think am when dem design or test the system.

> [🎥 Click here for a video: Reliability and safety in AI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusiveness

AI systems suppose design make everybody fit join and get power. When data scientists and AI developers dey design and implement AI, dem go find and solve any barrier wey fit exclude people without intention. For example, 1 billion people get disability worldwide. With AI progress, dem fit get access to many information and chance easier for their daily life. When you solve these barriers, e go open door to innovate and build AI products wey get better experience wey everybody go benefit.

> [🎥 Click here for a video: Inclusiveness in AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Security and privacy 

AI systems suppose dey safe and respect people privacy. People no too trust systems wey fit put their privacy, info, or life for risk. When you dey train machine learning models, you rely on data to get best result. For that reason, e good to consider where data come from and if e pure. For example, na user submit the data or e public? Next, as you dey work with data, e important to develop AI systems capable to protect confidential info and resist attack. As AI dey grow for everywhere, protecting privacy and securing personal and business info na big matter wey dey become more complex. Privacy and data security issues especially need attention for AI because AI systems rely on data to make correct and informed predictions and decisions about people.

> [🎥 Click here for a video: Security in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- As industry, we don make better progress for Privacy & security, especially because of regulations like GDPR (General Data Protection Regulation). 
- But for AI systems, we gerr acknowledge say tension dey between need for more personal data to make system personal and effective – as well as privacy. 
- Just like how connected computers and internet start, we still dey see increase for security issues wey concern AI. 
- Meanwhile, we see AI dey use to improve security. For example, most modern anti-virus scanners today dey driven by AI heuristics. 
- We need make sure say our Data Science process blend well with latest privacy and security practice.


### Transparency
AI systems suppose make sense. One important part of transparency na to explain how AI systems and their parts dey behave. Make people sabi how and why dem dey function so that dem fit identify any potential problem for performance, safety, privacy, bias, exclusion, or unintended result. Also, those wey dey use AI systems suppose dey honest and open about when, why, and how dem wan deploy dem. Plus, the limitations of the system dem dey use. For example, if bank dey use AI system for consumer loan decision, e important to check the outcome and why data fit influence system’s advice. Governments don start to regulate AI for different industries, so data scientists and organizations must fit explain if AI system meet law, especially when outcome bad.

> [🎥 Click here for a video: Transparency in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Because AI systems complex wella, e hard to understand how dem work and interpret the results. 
- This lack of understanding affect how people dey manage, use and document these systems. 
- More important, this lack of understanding affect decisions wey people dey make using results from these systems. 

### Accountability 
 
People wey design and run AI systems suppose dey responsible for how their system dey operate. This accountability na especially important for sensitive tech like facial recognition. Recently, demand don grow for facial recognition, especially from law enforcement wey see the tech get potential for things like finding missing children. But, this tech fit also fit be use by government to put people fundamental rights for risk by, for example, to dey watch some people nonstop. So, data scientists and organizations gerr responsible for how their AI system fit affect people or society.

[![Leading AI Researcher Warns of Mass Surveillance Through Facial Recognition](../../../../translated_images/pcm/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Warnings of Mass Surveillance Through Facial Recognition 

For finally, one big question for our generation, as the first wey dey bring AI to society, na how to make sure computers go still dey responsible to people and how to make sure people wey design computers go still dey accountable to everybody else.

## Impact assessment 

Before you train machine learning model, e important to do impact assessment to sabi the purpose of AI system; wetin e go use for; where e go deploy; and who go dey use am. This one go help reviewers or testers wey dey check the system to know wetin dem suppose consider as potential risk and wetin the consequences fit be.

The areas wey you gerr focus when you dey do impact assessment include:

* **Adverse impact on individuals**. Sabi any restriction or requirement, unsupported use or any known limitation wey fit disturb how the system dey work go help make sure say the system no go dey waka for way wey fit harm people.
* **Data requirements**. To sabi how and where the system go use data go help reviewers check any data rules wey you gerr watch like GDPR or HIPAA. Plus, check if the origin or amount of data enough to train.
* **Summary of impact**. Gather list of possible harms wey fit show from using the system. For the machine learning lifecycle, dey check if the issues you find don solve or address.
* **Applicable goals** for each six core principles. See if goals for every principle meet and if gaps dey.


## Debugging with responsible AI  

Like how you dey debug software app, debugging AI system na important work for find and fix issues wey dey system. Plenty things fit cause model no to work as expected or responsible. Most normal model performance metrics na quantitative total of model’s performance, but dem no complete for check how model fit violate responsible AI principles. Also, machine learning model na black box wey make am hard to understand wetin dey push the outcome or give explanation when e do error. Later for this course, we go learn how to use Responsible AI dashboard to help debug AI systems. The dashboard dey give data scientists and AI developers full tool to do:

* **Error analysis**. To check the error distribution for model wey fit affect system fairness or reliability.
* **Model overview**. To find where model get performance differences for data groups.
* **Data analysis**. To understand data distribution and find any bias for data wey fit cause fairness, inclusiveness, and reliability problem.
* **Model interpretability**. To understand wetin dey affect or influence model’s prediction. This one dey help explain model behavior, wey important for transparency and accountability.


## 🚀 Challenge 
 
To stop harms from show, we gerr:

- get diversity of backgrounds and ideas among the people wey dey work on systems 
- invest for datasets wey reflect the diversity of our society 
- develop better ways for all machine learning lifecycle to find and fix irresponsible AI when e show 

Think of real-life situation where model no fit trust clear for building and use am. Wetin else we gerr think about? 

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study 
 
For this lesson, you don learn some basics about fairness and unfairness concepts for machine learning.  
 
Watch this workshop to learn more on top the topics: 

- In pursuit of responsible AI: Bringing principles to practice by Besmira Nushi, Mehrnoosh Sameki and Amit Sharma

[![Responsible AI Toolbox: Na open-source framework wey dem dey use build responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Na open-source framework wey dem dey use build responsible AI")

> 🎥 Klik di pikcha wey dey up for one video: RAI Toolbox: Na open-source framework wey dem dey use build responsible AI by Besmira Nushi, Mehrnoosh Sameki, and Amit Sharma

Also, read: 

- Microsoft RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoft FATE research group: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Read about Azure Machine Learning tools to make sure say fairness dey:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Assignment

[Explore RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Dis document don translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even tho we dey try make am correct, abeg make you know say automated translation fit get errors or mistakes. Di original document for dia own language na im be di correct source. For important info, make person wey sabi human translation do am. We no go responsible for any misunderstanding or wrong understanding wey fit happen because of dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->