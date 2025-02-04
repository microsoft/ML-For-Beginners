# 机器学习简介

## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1/)

---

[![初学者的机器学习 - 初学者的机器学习简介](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "初学者的机器学习 - 初学者的机器学习简介")

> 🎥 点击上面的图片观看一个简短的视频，了解本课内容。

欢迎来到这个面向初学者的经典机器学习课程！无论你是完全不了解这个话题，还是一个有经验的ML从业者想要复习某个领域，我们都很高兴你能加入我们！我们希望为你的ML学习创造一个友好的起点，并乐于评估、回应并采纳你的[反馈](https://github.com/microsoft/ML-For-Beginners/discussions)。

[![ML简介](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "ML简介")

> 🎥 点击上面的图片观看视频：MIT的John Guttag介绍机器学习

---
## 机器学习入门

在开始这个课程之前，你需要将你的计算机设置好，并准备好在本地运行笔记本。

- **通过这些视频配置你的机器**。使用以下链接了解[如何在系统中安装Python](https://youtu.be/CXZYvNRIAKM)和[设置开发用的文本编辑器](https://youtu.be/EU8eayHWoZg)。
- **学习Python**。还建议你对[Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott)有基本的了解，这是一门对数据科学家很有用的编程语言，我们在本课程中会使用它。
- **学习Node.js和JavaScript**。我们在构建Web应用程序时也会多次使用JavaScript，所以你需要安装[node](https://nodejs.org)和[npm](https://www.npmjs.com/)，以及用于Python和JavaScript开发的[Visual Studio Code](https://code.visualstudio.com/)。
- **创建一个GitHub账户**。既然你在[GitHub](https://github.com)上找到了我们，你可能已经有一个账户了，如果没有，创建一个，然后fork这个课程以便自己使用。（也可以给我们一个star 😊）
- **探索Scikit-learn**。熟悉一下[Scikit-learn](https://scikit-learn.org/stable/user_guide.html)，这是我们在这些课程中引用的一组ML库。

---
## 什么是机器学习？

“机器学习”这个术语是当今最流行和经常使用的术语之一。如果你对技术有一定了解，无论你从事什么领域，你都有很大可能至少听过一次这个术语。然而，机器学习的机制对大多数人来说仍然是一个谜。对于机器学习初学者来说，这个主题有时会让人感到不知所措。因此，理解机器学习的真正含义，并通过实际例子一步步学习它，是很重要的。

---
## 热潮曲线

![ml hype curve](../../../../translated_images/hype.07183d711a17aafe70915909a0e45aa286ede136ee9424d418026ab00fec344c.zh.png)

> 谷歌趋势显示了最近“机器学习”一词的“热潮曲线”

---
## 神秘的宇宙

我们生活在一个充满迷人谜团的宇宙中。伟大的科学家如斯蒂芬·霍金、阿尔伯特·爱因斯坦等，终其一生致力于寻找揭示我们周围世界谜团的有意义的信息。这是人类学习的本质：一个人类孩子通过感知周围环境的事实，逐年揭示世界的结构，直到成年。

---
## 孩子的脑袋

孩子的脑袋和感官感知周围环境的事实，并逐渐学习生活中的隐藏模式，这帮助孩子制定逻辑规则来识别学到的模式。人类大脑的学习过程使人类成为这个世界上最复杂的生物。通过不断发现隐藏的模式并在这些模式上进行创新，使我们在一生中变得越来越好。这种学习能力和进化能力与一个叫做[脑可塑性](https://www.simplypsychology.org/brain-plasticity.html)的概念有关。从表面上看，我们可以在一定程度上将人类大脑的学习过程与机器学习的概念联系起来。

---
## 人类大脑

[人类大脑](https://www.livescience.com/29365-human-brain.html)从现实世界中感知事物，处理感知到的信息，做出理性决策，并根据情况执行某些行为。这就是我们所说的智能行为。当我们将这种智能行为过程的仿真程序化到机器上时，这就是人工智能（AI）。

---
## 一些术语

虽然这些术语可能会混淆，但机器学习（ML）是人工智能的一个重要子集。**ML关注的是使用专门的算法从感知到的数据中发现有意义的信息和隐藏的模式，以支持理性决策过程**。

---
## AI, ML, 深度学习

![AI, ML, deep learning, data science](../../../../translated_images/ai-ml-ds.537ea441b124ebf69c144a52c0eb13a7af63c4355c2f92f440979380a2fb08b8.zh.png)

> 一张展示AI、ML、深度学习和数据科学之间关系的图表。由[Jen Looper](https://twitter.com/jenlooper)制作，灵感来自[这张图](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## 涵盖的概念

在这个课程中，我们将只涵盖机器学习的核心概念，这是初学者必须了解的。我们主要使用Scikit-learn，一个许多学生用来学习基础知识的优秀库，来讲解我们所谓的“经典机器学习”。要理解人工智能或深度学习的更广泛概念，扎实的机器学习基础知识是不可或缺的，因此我们希望在这里提供这些知识。

---
## 在本课程中你将学习：

- 机器学习的核心概念
- ML的历史
- ML与公平性
- 回归ML技术
- 分类ML技术
- 聚类ML技术
- 自然语言处理ML技术
- 时间序列预测ML技术
- 强化学习
- ML的实际应用

---
## 我们不会涵盖的内容

- 深度学习
- 神经网络
- AI

为了提供更好的学习体验，我们将避免神经网络的复杂性、“深度学习”——使用神经网络构建多层模型——和AI，我们将在不同的课程中讨论这些内容。我们还将提供即将推出的数据科学课程，以专注于这个更大领域的这一方面。

---
## 为什么要学习机器学习？

从系统的角度来看，机器学习被定义为创建能够从数据中学习隐藏模式以辅助智能决策的自动化系统。

这种动机在某种程度上是受人类大脑如何根据从外部世界感知的数据学习某些事物的启发。

✅ 想一想为什么企业会想尝试使用机器学习策略，而不是创建一个基于硬编码规则的引擎。

---
## 机器学习的应用

机器学习的应用现在几乎无处不在，就像由我们的智能手机、连接设备和其他系统生成的数据在我们的社会中流动一样无处不在。考虑到最先进的机器学习算法的巨大潜力，研究人员一直在探索其解决多维和多学科现实生活问题的能力，并取得了很好的成果。

---
## 应用ML的例子

**你可以通过多种方式使用机器学习**：

- 从患者的病史或报告中预测疾病的可能性。
- 利用天气数据预测天气事件。
- 理解文本的情感。
- 识别假新闻以阻止宣传的传播。

金融、经济学、地球科学、太空探索、生物医学工程、认知科学，甚至人文学科领域都采用机器学习来解决它们领域中繁重的数据处理问题。

---
## 结论

机器学习通过从现实世界或生成的数据中发现有意义的见解来自动化模式发现过程。它在商业、健康和金融应用等领域中已经证明了自己的高度价值。

在不久的将来，了解机器学习的基础知识将成为任何领域的人们必须掌握的技能，因为它被广泛采用。

---
# 🚀 挑战

在纸上或使用[Excalidraw](https://excalidraw.com/)等在线应用程序，画出你对AI、ML、深度学习和数据科学之间区别的理解。添加一些这些技术擅长解决的问题的想法。

# [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2/)

---
# 复习与自学

要了解更多关于如何在云中使用ML算法的信息，请关注这个[学习路径](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)。

参加一个关于ML基础知识的[学习路径](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)。

---
# 作业

[开始运行](assignment.md)

**免责声明**：
本文件是使用机器翻译服务翻译的。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原文档的母语版本为权威来源。对于关键信息，建议使用专业的人类翻译。我们不对因使用本翻译而产生的任何误解或误读承担责任。