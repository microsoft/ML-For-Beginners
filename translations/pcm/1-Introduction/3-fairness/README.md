<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-11-18T18:22:42+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "pcm"
}
-->
# How to Build Machine Learning Solutions Wey Get Responsible AI

![Summary of responsible AI in Machine Learning in a sketchnote](../../../../translated_images/ml-fairness.ef296ebec6afc98a.pcm.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

For dis curriculum, you go start to sabi how machine learning dey affect our everyday life. Even now, systems and models dey involved for daily decision-making tasks, like health care diagnosis, loan approval or to catch fraud. So, e dey important say these models go work well to give results wey people fit trust. Just like any software application, AI systems fit fail or give result wey no dey okay. Na why e dey important to sabi and fit explain how AI model dey behave.

Imagine wetin fit happen if the data wey you dey use to build these models no get some kind people, like race, gender, political view, religion, or e dey represent some people too much. Wetin go happen if the model dey favor one group over another? Wetin be the result for the application? Plus, wetin go happen if the model give bad result wey go harm people? Who go take responsibility for how the AI system dey behave? Na these kind questions we go look for dis curriculum.

For dis lesson, you go:

- Learn why fairness for machine learning and fairness-related wahala dey important.
- Sabi how to check outliers and unusual situations to make sure say e dey reliable and safe.
- Understand why e dey important to design systems wey go include everybody.
- See why e dey necessary to protect privacy and security of data and people.
- Understand why e dey good to explain how AI models dey behave (glass box approach).
- Know why accountability dey important to build trust for AI systems.

## Prerequisite

Before you start, make sure say you don take the "Responsible AI Principles" Learn Path and watch the video below about the topic:

Learn more about Responsible AI by following this [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Microsoft's Approach to Responsible AI

## Fairness

AI systems suppose treat everybody fairly and no dey affect similar groups differently. For example, if AI systems dey give advice for medical treatment, loan application, or employment, e suppose give the same advice to people wey get similar symptoms, financial situation, or qualifications. As humans, we dey carry bias wey dey affect our decisions and actions. These biases fit show for the data wey we dey use to train AI systems. Sometimes, e fit happen by mistake. E dey hard to sabi when you dey add bias for data.

**“Unfairness”** mean negative impact or “harms” for one group of people, like race, gender, age, or disability. The main fairness-related harms fit be:

- **Allocation**, if one gender or ethnicity dey favored over another.
- **Quality of service**. If you train data for one specific situation but reality dey more complex, e go make the service no perform well. For example, hand soap dispenser wey no fit sense people wey get dark skin. [Reference](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigration**. To criticize or label something or someone unfairly. For example, image labeling technology wey mistakenly call dark-skinned people gorillas.
- **Over- or under-representation**. When one group no dey seen for one profession, and any service wey dey promote that na harm.
- **Stereotyping**. To connect one group with pre-assigned attributes. For example, language translation system between English and Turkish fit get mistake because of words wey dey stereotype gender.

![translation to Turkish](../../../../translated_images/gender-bias-translate-en-tr.f185fd8822c2d437.pcm.png)
> translation to Turkish

![translation back to English](../../../../translated_images/gender-bias-translate-tr-en.4eee7e3cecb8c70e.pcm.png)
> translation back to English

When we dey design and test AI systems, we need to make sure say AI dey fair and no dey programmed to make biased or discriminatory decisions, wey humans no suppose make. To make AI and machine learning fair na one big sociotechnical challenge.

### Reliability and safety

To build trust, AI systems suppose dey reliable, safe, and consistent for normal and unexpected conditions. E dey important to sabi how AI systems go behave for different situations, especially outliers. When we dey build AI solutions, we need to focus well on how to handle different situations wey the AI go face. For example, self-driving car suppose put people safety first. The AI wey dey power the car suppose think about all the possible situations wey the car fit face like night, thunderstorms, blizzards, kids wey dey run cross road, pets, road construction, etc. How well AI system fit handle different conditions reliably and safely go show how the data scientist or AI developer take plan and test the system.

> [🎥 Click the here for a video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusiveness

AI systems suppose dey designed to involve and empower everybody. When data scientists and AI developers dey design AI systems, dem suppose identify and fix barriers wey fit exclude people by mistake. For example, 1 billion people wey get disabilities dey for the world. With AI, dem fit access plenty information and opportunities easily for their daily life. If we fix these barriers, e go create chance to innovate and make AI products wey go give better experience wey go benefit everybody.

> [🎥 Click the here for a video: inclusiveness in AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Security and privacy

AI systems suppose dey safe and respect people privacy. People no go trust systems wey dey put their privacy, information, or life at risk. When we dey train machine learning models, we dey depend on data to get better results. So, we need to check where the data come from and if e dey okay. For example, na user submit the data or e dey public? Next, as we dey work with the data, e dey important to develop AI systems wey fit protect confidential information and resist attacks. As AI dey grow, to protect privacy and secure personal and business information dey more critical and complex. Privacy and data security matter need special attention for AI because data na the main thing wey AI systems dey use to make correct predictions and decisions about people.

> [🎥 Click the here for a video: security in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- As industry, we don make big progress for Privacy & security, especially because of regulations like GDPR (General Data Protection Regulation).
- But for AI systems, we need to balance the need for more personal data to make systems better – and privacy.
- Just like when internet start, we dey see plenty security wahala related to AI.
- At the same time, AI dey help improve security. For example, most modern anti-virus scanners dey use AI today.
- We need to make sure say our Data Science processes dey work well with the latest privacy and security practices.

### Transparency

AI systems suppose dey easy to understand. Transparency mean to explain how AI systems and their parts dey behave. To improve understanding of AI systems, stakeholders need to sabi how and why dem dey work so dem fit identify performance wahala, safety and privacy concerns, bias, exclusionary practices, or mistakes. People wey dey use AI systems suppose dey honest about when, why, and how dem dey use am. Plus, dem suppose talk about the limitations of the systems. For example, if bank dey use AI system to help with lending decisions, e dey important to check the results and sabi which data dey influence the system recommendations. Governments don dey regulate AI for different industries, so data scientists and organizations suppose explain if AI system meet regulatory requirements, especially when e give bad result.

> [🎥 Click the here for a video: transparency in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Because AI systems dey complex, e dey hard to understand how dem dey work and interpret the results.
- This lack of understanding dey affect how dem dey manage, use, and document the systems.
- This lack of understanding dey also affect the decisions wey people dey make with the results wey the systems dey give.

### Accountability

The people wey dey design and use AI systems suppose take responsibility for how their systems dey work. Accountability dey very important for sensitive technologies like facial recognition. Recently, demand for facial recognition technology don dey grow, especially from law enforcement wey wan use am to find missing children. But these technologies fit make government use am to take away people freedom, like to dey monitor specific people every time. So, data scientists and organizations need to take responsibility for how their AI system dey affect people or society.

[![Leading AI Researcher Warns of Mass Surveillance Through Facial Recognition](../../../../translated_images/accountability.41d8c0f4b85b6231.pcm.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Warnings of Mass Surveillance Through Facial Recognition

At the end, one big question for our generation, as the first generation wey dey bring AI to society, na how we go make sure say computers go dey accountable to people and how we go make sure say the people wey dey design computers go dey accountable to everybody.

## Impact assessment

Before you train machine learning model, e dey important to do impact assessment to sabi the purpose of the AI system; wetin dem wan use am do; where dem go use am; and who go dey interact with the system. Dis go help reviewer(s) or testers to sabi wetin to check when dem dey look for risks and expected results.

The following na areas to focus on when you dey do impact assessment:

* **Adverse impact on individuals**. Sabi any restriction or requirements, unsupported use or any known limitations wey fit make the system no perform well. This go help make sure say the system no go harm people.
* **Data requirements**. Sabi how and where the system go use data so reviewers fit check any data requirements wey you need to follow (e.g., GDPR or HIPPA data regulations). Plus, check if the source or quantity of data dey enough for training.
* **Summary of impact**. Gather list of potential harms wey fit happen if you use the system. For the ML lifecycle, check if the issues wey you identify don dey fixed or addressed.
* **Applicable goals** for each of the six core principles. Check if the goals for each principle don dey met and if gaps dey.

## Debugging with responsible AI

Just like how we dey debug software application, debugging AI system na process to find and fix issues for the system. Plenty things fit make model no perform as e suppose or no dey responsible. Most traditional model performance metrics dey give numbers wey no dey enough to check how model dey break responsible AI principles. Plus, machine learning model na black box wey dey hard to understand wetin dey drive the result or explain mistake. Later for this course, we go learn how to use Responsible AI dashboard to debug AI systems. The dashboard dey give data scientists and AI developers tool to do:

* **Error analysis**. To check the error distribution of the model wey fit affect fairness or reliability.
* **Model overview**. To find where the model performance dey different across data groups.
* **Data analysis**. To check the data distribution and find any bias for the data wey fit cause fairness, inclusiveness, and reliability wahala.
* **Model interpretability**. To sabi wetin dey affect or influence the model predictions. This go help explain the model behavior, wey dey important for transparency and accountability.

## 🚀 Challenge

To stop harms from happening, we suppose:

- Get people wey get different backgrounds and perspectives to work on systems.
- Invest for datasets wey show the diversity of our society.
- Develop better methods for the machine learning lifecycle to detect and fix responsible AI issues when dem happen.

Think about real-life situations wey show say model no dey trustworthy for model-building and usage. Wetin else we suppose consider?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
## Review & Self Study

For dis lesson, you don learn some basic tins about fairness and unfairness for machine learning.

Watch dis workshop to sabi more about di topics:

- How to pursue responsible AI: How to carry principles enter practice by Besmira Nushi, Mehrnoosh Sameki and Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Click di image wey dey up for video: RAI Toolbox: An open-source framework for building responsible AI by Besmira Nushi, Mehrnoosh Sameki, and Amit Sharma

Still read:

- Microsoft RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft FATE research group: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Read about Azure Machine Learning tools wey fit help make sure fairness dey:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Assignment

[Explore RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI transleshion service [Co-op Translator](https://github.com/Azure/co-op-translator) do di transleshion. Even as we dey try make am accurate, abeg make you sabi say automatik transleshion fit get mistake or no dey correct well. Di original dokyument for im native language na di one wey you go take as di correct source. For important informashon, e good make professional human transleshion dey use. We no go fit take blame for any misunderstanding or wrong interpretation wey go happen because you use dis transleshion.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->