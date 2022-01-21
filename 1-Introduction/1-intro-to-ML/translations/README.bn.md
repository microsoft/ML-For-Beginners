# মেশিন লার্নিং এর সূচনা 
<!--
Watch the video, then take the pre-lesson quiz
-->

[![ML, AI, deep learning - What's the difference?](https://img.youtube.com/vi/lTd9RSxS9ZE/0.jpg)](https://youtu.be/lTd9RSxS9ZE "ML, AI, deep learning - What's the difference?")

> 🎥 মেশিন লার্নিং, এআই(আর্টিফিশিয়াল ইন্টিলিজেন্স) এবং ডিপ লার্নিং এর মধ্যে পার্থক্য এর আলোচনা জানতে উপরের ছবিটিতে ক্লিক করে ভিডিওটি দেখুন। 

## [প্রি-লেকচার-কুইজ](https://white-water-09ec41f0f.azurestaticapps.net/quiz/1/)

---
বিগিনারদের জন্য ক্লাসিক্যাল মেশিন লার্নিং কোর্স এ আপনাকে স্বাগতম!আপনি হয় এই বিষয়ে সম্পূর্ণ নতুন অথবা মেশিন লার্নিং এ নিজের অনুশীলনকে আরও উন্নত করতে চান, আপনি আমাদের সাথে যোগদান করতে পেরে আমরা খুশি! আমরা আপনার ML অধ্যয়নের জন্য একটি বন্ধুত্বপূর্ণ লঞ্চিং স্পট তৈরি করতে চাই এবং আপনার মূল্যায়ন, প্রতিক্রিয়া,[ফিডব্যাক](https://github.com/microsoft/ML-For-Beginners/discussions). জানাতে এবং অন্তর্ভুক্ত করতে পেরে খুশি হব । 


[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")


> 🎥 ভিডিওটি দেখার জন্য উপরের ছবিতে ক্লিক করুন
MIT এর জন গাটেং মেশিন লার্নিং এর পরিচিতি করাচ্ছেন। 

---
## মেশিন লার্নিং এর শুরু 

এই ক্যারিকুলাম শুরুর করার পূর্বে, নোটবুক রান করার জন্য নোটবুক সেটআপ থাকতে হবে। 


- **আপনার মেশিন কে কনফিগার করুন এই ভিডিও দেখে**. শিখার জন্য এই লিংকটি ব্যবহার করুন [কিভাবে পাইথন ইন্সটল করতে হয়](https://youtu.be/CXZYvNRIAKM) এবং [সেটআপ এ ইডিটর](https://youtu.be/EU8eayHWoZg) .
- **পাইথন শিখুন**. [পাইথন](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-15963-cxa) এর ব্যাসিক নলেজ জানা থাকা জরুরী। এই কোর্সের প্রোগ্রামিং ল্যাঙ্গুয়েজ ডেটা সাইন্সটিস্ট এর জন্য খুবই গুরুত্বপূর্ণ। 
- **Node.js এবং JavaScript শিখুন**.ওয়েব অ্যাপস তৈরির জন্য এই কোর্সে আমরা জাবাস্ক্রিপট ব্যাবহার করব। তাই, আপনার [নোড](https://nodejs.org) এবং [npm](https://www.npmjs.com/) ইন্সটল থাকতে হবে। অন্যদিকে, পাইথন এবং জাভাস্ক্রিপট ডেভেলাপমেন্টের জন্য [ভিজুয়াল স্টুডিও](https://code.visualstudio.com/) কোড এ দুটুই আছে। 
- **একটি গিটহাব অ্যাকাউন্ট তৈরি করুন**. যেহেতু আপনি আমাদের কে [গিটহাব](https://github.com) এ পেয়েছেন, তারমানে আপনার ইতিমধ্যেই একাউন্ট আছে। তবে যদি না থাকে, একটি একাউন্ট তৈরি করুন এবং পরে ফর্ক করে আপনার বানিয়ে নিন। (স্টার দিতে ভুলে যাবেন না,😊 )
- **ঘুরিয়ে আসেন Scikit-learn**. নিজেকে পরিচিত করুন [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) এর সাথে, মেশিন লার্নিং লাইব্রেরি সেট যা আমরা এই কোর্সে উল্লেখ করে থাকব

---
## মেশিন লার্নিং কি?
'মেশিন লার্নিং' শব্দটি বর্তমান সময়ের সবচেয়ে জনপ্রিয় এবং প্রায়ই ব্যবহৃত একটি শব্দ। আপনি যে ডোমেইনে কাজ করেন না কেন প্রযুক্তির সাথে আপনার পরিচিতি থাকলে অন্তত একবার এই শব্দটি শুনেছেন এমন একটি অপ্রয়োজনীয় সম্ভাবনা রয়েছে। মেশিন লার্নিং এর মেকানিক্স, যাইহোক, বেশিরভাগ মানুষের কাছে  এটি একটি রহস্য। একজন মেশিন লার্নিং নতুনদের জন্য, বিষয়টি কখনও কখনও অপ্রতিরোধ্য মনে হতে পারে। অতএব, মেশিন লার্নিং আসলে কী তা বোঝা গুরুত্বপূর্ণ এবং বাস্তব উদাহরণের মাধ্যমে ধাপে ধাপে এটি সম্পর্কে শিখতে হবে।

---
## হাইফ কার্ভ

![ml hype curve](images/hype.png)

> Google Trends এ 'মেশিন লার্নিং' শব্দটির সাম্প্রতিক 'হাইপ কার্ভ'।

---
## এক রহস্যময় মহাবিশ্ব

আমরা রহস্যে ভরপুর একটি আকর্ষনীয় মহাবিশ্বে বাস করি। স্টিফেন হকিং, আলবার্ট আইনস্টাইন এবং আরও অনেকের মতো মহান বিজ্ঞানীরা আমাদের চারপাশের বিশ্বের রহস্য উন্মোচন করে এমন অর্থপূর্ণ তথ্য অনুসন্ধানে তাদের জীবন উৎসর্গ করেছেন।এটি মানুষের শেখার একটি অবস্থা: একটি মানব শিশু নতুন জিনিস শিখে এবং বছরের পর বছর তাদের বিশ্বের গঠন উন্মোচন করে যখন তারা প্রাপ্তবয়স্ক হয়ে ওঠে।

---
## The child's brain

A child's brain and senses perceive the facts of their surroundings and gradually learn the hidden patterns of life which help the child to craft logical rules to identify learned patterns. The learning process of the human brain makes humans the most sophisticated living creature of this world. Learning continuously by discovering hidden patterns and then innovating on those patterns enables us to make ourselves better and better throughout our lifetime. This learning capacity and evolving capability is related to a concept called [brain plasticity](https://www.simplypsychology.org/brain-plasticity.html). Superficially, we can draw some motivational similarities between the learning process of the human brain and the concepts of machine learning.

---
## The human brain

The [human brain](https://www.livescience.com/29365-human-brain.html) perceives things from the real world, processes the perceived information, makes rational decisions, and performs certain actions based on circumstances. This is what we called behaving intelligently. When we program a facsimile of the intelligent behavioral process to a machine, it is called artificial intelligence (AI).

---
## Some terminology

Although the terms can be confused, machine learning (ML) is an important subset of artificial intelligence. **ML is concerned with using specialized algorithms to uncover meaningful information and find hidden patterns from perceived data to corroborate the rational decision-making process**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](images/ai-ml-ds.png)

> A diagram showing the relationships between AI, ML, deep learning, and data science. Infographic by [Jen Looper](https://twitter.com/jenlooper) inspired by [this graphic](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepts to cover

In this curriculum, we are going to cover only the core concepts of machine learning that a beginner must know. We cover what we call 'classical machine learning' primarily using Scikit-learn, an excellent library many students use to learn the basics.  To understand broader concepts of artificial intelligence or deep learning, a strong fundamental knowledge of machine learning is indispensable, and so we would like to offer it here.

---
## In this course you will learn:

- core concepts of machine learning
- the history of ML
- ML and fairness
- regression ML techniques
- classification ML techniques
- clustering ML techniques
- natural language processing ML techniques
- time series forecasting ML techniques
- reinforcement learning
- real-world applications for ML

---
## What we will not cover

- deep learning
- neural networks
- AI

To make for a better learning experience, we will avoid the complexities of neural networks, 'deep learning' - many-layered model-building using neural networks - and AI, which we will discuss in a different curriculum. We also will offer a forthcoming data science curriculum to focus on that aspect of this larger field.

---
## Why study machine learning?

Machine learning, from a systems perspective, is defined as the creation of automated systems that can learn hidden patterns from data to aid in making intelligent decisions.

This motivation is loosely inspired by how the human brain learns certain things based on the data it perceives from the outside world.

✅ Think for a minute why a business would want to try to use machine learning strategies vs. creating a hard-coded rules-based engine.

---
## Applications of machine learning

Applications of machine learning are now almost everywhere, and are as ubiquitous as the data that is flowing around our societies, generated by our smart phones, connected devices, and other systems. Considering the immense potential of state-of-the-art machine learning algorithms, researchers have been exploring their capability to solve multi-dimensional and multi-disciplinary real-life problems with great positive outcomes.

---
## Examples of applied ML

**You can use machine learning in many ways**:

- To predict the likelihood of disease from a patient's medical history or reports.
- To leverage weather data to predict weather events.
- To understand the sentiment of a text.
- To detect fake news to stop the spread of propaganda.

Finance, economics, earth science, space exploration, biomedical engineering, cognitive science, and even fields in the humanities have adapted machine learning to solve the arduous, data-processing heavy problems of their domain.

---
## Conclusion

Machine learning automates the process of pattern-discovery by finding meaningful insights from real-world or generated data. It has proven itself to be highly valuable in business, health, and financial applications, among others.

In the near future, understanding the basics of machine learning is going to be a must for people from any domain due to its widespread adoption.

---
# 🚀 Challenge

Sketch, on paper or using an online app like [Excalidraw](https://excalidraw.com/), your understanding of the differences between AI, ML, deep learning, and data science. Add some ideas of problems that each of these techniques are good at solving.

# [Post-lecture quiz](https://white-water-09ec41f0f.azurestaticapps.net/quiz/2/)

---
# Review & Self Study

To learn more about how you can work with ML algorithms in the cloud, follow this [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-15963-cxa).

Take a [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-15963-cxa) about the basics of ML.

---
# Assignment

[Get up and running](assignment.md)