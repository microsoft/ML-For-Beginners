<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T09:05:11+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "zh"
}
-->
# 机器学习简介

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

---

[![初学者的机器学习 - 机器学习入门](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "初学者的机器学习 - 机器学习入门")

> 🎥 点击上方图片观看本课相关的短视频。

欢迎来到这门面向初学者的经典机器学习课程！无论你是完全新手，还是一位希望复习某些领域的经验丰富的机器学习从业者，我们都很高兴你能加入我们！我们希望为你的机器学习学习提供一个友好的起点，并欢迎你提供[反馈](https://github.com/microsoft/ML-For-Beginners/discussions)，我们会评估、回应并融入你的建议。

[![机器学习简介](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "机器学习简介")

> 🎥 点击上方图片观看视频：麻省理工学院的 John Guttag 介绍机器学习

---
## 开始学习机器学习

在开始学习本课程之前，你需要确保你的电脑已经设置好并可以本地运行笔记本。

- **通过以下视频配置你的电脑**。使用以下链接学习[如何安装 Python](https://youtu.be/CXZYvNRIAKM)以及[设置文本编辑器](https://youtu.be/EU8eayHWoZg)进行开发。
- **学习 Python**。建议你对[Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott)有基本的了解，这是一种对数据科学家非常有用的编程语言，我们将在课程中使用它。
- **学习 Node.js 和 JavaScript**。我们在课程中会使用 JavaScript 构建一些网页应用，因此你需要安装 [node](https://nodejs.org) 和 [npm](https://www.npmjs.com/)，以及为 Python 和 JavaScript 开发准备好 [Visual Studio Code](https://code.visualstudio.com/)。
- **创建 GitHub 账户**。既然你在 [GitHub](https://github.com) 找到了我们，你可能已经有一个账户了，但如果没有，请创建一个账户，然后 fork 本课程以供自己使用。（也可以给我们点个星星 😊）
- **探索 Scikit-learn**。熟悉 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)，这是我们在课程中参考的一组机器学习库。

---
## 什么是机器学习？

“机器学习”是当今最流行和最常用的术语之一。如果你对技术有一定的了解，无论你从事哪个领域，都有很大可能至少听过一次这个术语。然而，机器学习的运作机制对大多数人来说仍然是一个谜。对于机器学习初学者来说，这个主题有时可能会让人感到不知所措。因此，了解机器学习的真正含义，并通过实际例子一步步学习它是非常重要的。

---
## 热度曲线

![机器学习热度曲线](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends 显示了“机器学习”这一术语的近期热度曲线

---
## 神秘的宇宙

我们生活在一个充满迷人奥秘的宇宙中。像斯蒂芬·霍金、阿尔伯特·爱因斯坦等伟大的科学家们，毕生致力于寻找有意义的信息，以揭示我们周围世界的奥秘。这是人类学习的本质：一个孩子通过逐年成长，学习新事物并揭示其世界的结构。

---
## 孩子的大脑

孩子的大脑和感官感知周围环境的事实，并逐渐学习生活中隐藏的模式，这些模式帮助孩子制定逻辑规则以识别已学到的模式。人类大脑的学习过程使人类成为这个世界上最复杂的生物。通过发现隐藏模式并不断创新，我们能够在一生中不断提升自己。这种学习能力和进化能力与一个叫做[脑可塑性](https://www.simplypsychology.org/brain-plasticity.html)的概念有关。从表面上看，我们可以将人类大脑的学习过程与机器学习的概念进行一些激励性的类比。

---
## 人类大脑

[人类大脑](https://www.livescience.com/29365-human-brain.html)从现实世界中感知事物，处理感知到的信息，做出理性决策，并根据情况采取某些行动。这就是我们所说的智能行为。当我们将智能行为过程的模拟编程到机器中时，这就被称为人工智能（AI）。

---
## 一些术语

尽管这些术语可能会混淆，但机器学习（ML）是人工智能的重要子集。**机器学习关注的是使用专门的算法从感知到的数据中发现有意义的信息和隐藏模式，以支持理性决策过程**。

---
## AI、ML、深度学习

![AI、ML、深度学习、数据科学](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> 一张展示 AI、ML、深度学习和数据科学之间关系的图表。信息图由 [Jen Looper](https://twitter.com/jenlooper) 制作，灵感来源于[这张图](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## 涵盖的概念

在本课程中，我们将仅涵盖机器学习的核心概念，这些是初学者必须了解的内容。我们主要使用 Scikit-learn，这是一款许多学生用来学习基础知识的优秀库，来讲解我们称之为“经典机器学习”的内容。要理解人工智能或深度学习的更广泛概念，扎实的机器学习基础知识是不可或缺的，因此我们希望在这里提供这些知识。

---
## 在本课程中你将学习：

- 机器学习的核心概念
- 机器学习的历史
- 机器学习与公平性
- 回归机器学习技术
- 分类机器学习技术
- 聚类机器学习技术
- 自然语言处理机器学习技术
- 时间序列预测机器学习技术
- 强化学习
- 机器学习的实际应用

---
## 我们不会涵盖的内容

- 深度学习
- 神经网络
- 人工智能

为了提供更好的学习体验，我们将避免涉及神经网络的复杂性、“深度学习”（使用神经网络构建多层模型）以及人工智能，这些内容将在另一门课程中讨论。我们还将提供即将推出的数据科学课程，以专注于这一更广泛领域的相关内容。

---
## 为什么学习机器学习？

从系统的角度来看，机器学习被定义为创建能够从数据中学习隐藏模式以帮助做出智能决策的自动化系统。

这种动机在一定程度上受到人类大脑如何根据外界感知的数据学习某些事物的启发。

✅ 思考一下，为什么企业会选择使用机器学习策略，而不是创建一个基于硬编码规则的引擎？

---
## 机器学习的应用

机器学习的应用几乎无处不在，就像我们社会中流动的数据一样，这些数据由智能手机、连接设备和其他系统生成。考虑到最先进的机器学习算法的巨大潜力，研究人员一直在探索其解决多维度和多学科现实问题的能力，并取得了非常积极的成果。

---
## 应用机器学习的例子

**机器学习有许多用途**：

- 根据患者的病史或报告预测疾病的可能性。
- 利用天气数据预测天气事件。
- 理解文本的情感。
- 检测虚假新闻以阻止宣传的传播。

金融、经济、地球科学、太空探索、生物医学工程、认知科学，甚至人文学科都已经适应了机器学习，以解决其领域中繁重的数据处理问题。

---
## 结论

机器学习通过从现实世界或生成的数据中发现有意义的洞察来自动化模式发现的过程。它已在商业、健康和金融等领域证明了其高度价值。

在不久的将来，由于机器学习的广泛应用，了解机器学习的基础知识将成为任何领域人士的必备技能。

---
# 🚀 挑战

用纸或在线应用（如 [Excalidraw](https://excalidraw.com/)）绘制你对 AI、ML、深度学习和数据科学之间差异的理解。添加一些关于每种技术擅长解决的问题的想法。

# [课后测验](https://ff-quizzes.netlify.app/en/ml/)

---
# 复习与自学

要了解如何在云端使用机器学习算法，请参考此[学习路径](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)。

学习机器学习基础知识，请参考此[学习路径](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)。

---
# 作业

[开始学习](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。因使用本翻译而导致的任何误解或误读，我们概不负责。