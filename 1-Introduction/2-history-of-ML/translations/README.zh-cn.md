# 机器学习的历史 

![机器学习历史概述](../../../sketchnotes/ml-history.png)
> 作者 [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/3/)

在本课中，我们将走过机器学习和人工智能历史上的主要里程碑。 

人工智能（AI）作为一个领域的历史与机器学习的历史交织在一起，因为支持机器学习的算法和计算能力的进步推动了AI的发展。记住，虽然这些领域作为不同研究领域在 20 世纪 50 年代才开始具体化，但重要的[算法、统计、数学、计算和技术发现](https://wikipedia.org/wiki/Timeline_of_machine_learning) 要早于和重叠了这个时代。 事实上，[数百年来](https://wikipedia.org/wiki/History_of_artificial_intelligence)人们一直在思考这些问题：本文讨论了“思维机器”这一概念的历史知识基础。 

## 主要发现

- 1763, 1812 [贝叶斯定理](https://wikipedia.org/wiki/Bayes%27_theorem) 及其前身。该定理及其应用是推理的基础，描述了基于先验知识的事件发生的概率。
- 1805 [最小二乘理论](https://wikipedia.org/wiki/Least_squares)由法国数学家 Adrien-Marie Legendre 提出。 你将在我们的回归单元中了解这一理论，它有助于数据拟合。
- 1913 [马尔可夫链](https://wikipedia.org/wiki/Markov_chain)以俄罗斯数学家 Andrey Markov 的名字命名，用于描述基于先前状态的一系列可能事件。
- 1957 [感知器](https://wikipedia.org/wiki/Perceptron)是美国心理学家 Frank Rosenblatt 发明的一种线性分类器，是深度学习发展的基础。
- 1967 [最近邻](https://wikipedia.org/wiki/Nearest_neighbor)是一种最初设计用于映射路线的算法。 在 ML 中，它用于检测模式。
- 1970 [反向传播](https://wikipedia.org/wiki/Backpropagation)用于训练[前馈神经网络](https://wikipedia.org/wiki/Feedforward_neural_network)。
- 1982 [循环神经网络](https://wikipedia.org/wiki/Recurrent_neural_network) 是源自产生时间图的前馈神经网络的人工神经网络。

✅ 做点调查。在 ML 和 AI 的历史上，还有哪些日期是重要的？
## 1950: 会思考的机器 

Alan Turing，一个真正杰出的人，[在 2019 年被公众投票选出](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) 作为 20 世纪最伟大的科学家，他认为有助于为“会思考的机器”的概念打下基础。他通过创建 [图灵测试](https://www.bbc.com/news/technology-18475646)来解决反对者和他自己对这一概念的经验证据的需求，你将在我们的 NLP 课程中进行探索。

## 1956: 达特茅斯夏季研究项目

“达特茅斯夏季人工智能研究项目是人工智能领域的一个开创性事件，”正是在这里，人们创造了“人工智能”一词（[来源](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)）。

> 原则上，学习的每个方面或智能的任何其他特征都可以被精确地描述，以至于可以用机器来模拟它。 

首席研究员、数学教授 John McCarthy 希望“基于这样一种猜想，即学习的每个方面或智能的任何其他特征原则上都可以如此精确地描述，以至于可以制造出一台机器来模拟它。” 参与者包括该领域的另一位杰出人物 Marvin Minsky。

研讨会被认为发起并鼓励了一些讨论，包括“符号方法的兴起、专注于有限领域的系统（早期专家系统），以及演绎系统与归纳系统的对比。”（[来源](https://wikipedia.org/wiki/Dartmouth_workshop)）。

## 1956 - 1974: “黄金岁月”

从 20 世纪 50 年代到 70 年代中期，乐观情绪高涨，希望人工智能能够解决许多问题。1967 年，Marvin Minsky 自信地说，“一代人之内...创造‘人工智能’的问题将得到实质性的解决。”（Minsky，Marvin（1967），《计算：有限和无限机器》，新泽西州恩格伍德克利夫斯：Prentice Hall）

自然语言处理研究蓬勃发展，搜索被提炼并变得更加强大，创造了“微观世界”的概念，在这个概念中，简单的任务是用简单的语言指令完成的。

这项研究得到了政府机构的充分资助，在计算和算法方面取得了进展，并建造了智能机器的原型。其中一些机器包括：

* [机器人 Shakey](https://wikipedia.org/wiki/Shakey_the_robot)，他们可以“聪明地”操纵和决定如何执行任务。

    ![Shakey, 智能机器人](../images/shakey.jpg)
    > 1972 年的 Shakey

* Eliza，一个早期的“聊天机器人”，可以与人交谈并充当原始的“治疗师”。 你将在 NLP 课程中了解有关 Eliza 的更多信息。 

    ![Eliza, 机器人](../images/eliza.png)
    > Eliza 的一个版本，一个聊天机器人 

* “积木世界”是一个微观世界的例子，在那里积木可以堆叠和分类，并且可以测试教机器做出决策的实验。 使用 [SHRDLU](https://wikipedia.org/wiki/SHRDLU) 等库构建的高级功能有助于推动语言处理向前发展。

    [![积木世界与 SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "积木世界与SHRDLU")
    
    > 🎥 点击上图观看视频： 积木世界与 SHRDLU

## 1974 - 1980: AI 的寒冬

到了 20 世纪 70 年代中期，很明显制造“智能机器”的复杂性被低估了，而且考虑到可用的计算能力，它的前景被夸大了。资金枯竭，市场信心放缓。影响信心的一些问题包括：

- **限制**。计算能力太有限了
- **组合爆炸**。随着对计算机的要求越来越高，需要训练的参数数量呈指数级增长，而计算能力却没有平行发展。
- **缺乏数据**。 缺乏数据阻碍了测试、开发和改进算法的过程。 
- **我们是否在问正确的问题？**。 被问到的问题也开始受到质疑。 研究人员开始对他们的方法提出批评： 
  - 图灵测试受到质疑的方法之一是“中国房间理论”，该理论认为，“对数字计算机进行编程可能使其看起来能理解语言，但不能产生真正的理解。” ([来源](https://plato.stanford.edu/entries/chinese-room/))
  - 将“治疗师”ELIZA 这样的人工智能引入社会的伦理受到了挑战。

与此同时，各种人工智能学派开始形成。 在 [“scruffy” 与 “neat AI”](https://wikipedia.org/wiki/Neats_and_scruffies) 之间建立了二分法。 _Scruffy_ 实验室对程序进行了数小时的调整，直到获得所需的结果。 _Neat_ 实验室“专注于逻辑和形式问题的解决”。 ELIZA 和 SHRDLU 是众所周知的 _scruffy_ 系统。 在 1980 年代，随着使 ML 系统可重现的需求出现，_neat_ 方法逐渐走上前沿，因为其结果更易于解释。

## 1980s 专家系统

随着这个领域的发展，它对商业的好处变得越来越明显，在 20 世纪 80 年代，‘专家系统’也开始广泛流行起来。“专家系统是首批真正成功的人工智能 (AI) 软件形式之一。” （[来源](https://wikipedia.org/wiki/Expert_system)）。

这种类型的系统实际上是混合系统，部分由定义业务需求的规则引擎和利用规则系统推断新事实的推理引擎组成。

在这个时代，神经网络也越来越受到重视。

## 1987 - 1993: AI 的冷静期

专业的专家系统硬件的激增造成了过于专业化的不幸后果。个人电脑的兴起也与这些大型、专业化、集中化系统展开了竞争。计算机的平民化已经开始，它最终为大数据的现代爆炸铺平了道路。

## 1993 - 2011

这个时代见证了一个新的时代，ML 和 AI 能够解决早期由于缺乏数据和计算能力而导致的一些问题。数据量开始迅速增加，变得越来越广泛，无论好坏，尤其是 2007 年左右智能手机的出现，计算能力呈指数级增长，算法也随之发展。这个领域开始变得成熟，因为过去那些随心所欲的日子开始具体化为一种真正的纪律。

## 现在

今天，机器学习和人工智能几乎触及我们生活的每一个部分。这个时代要求仔细了解这些算法对人类生活的风险和潜在影响。正如微软的 Brad Smith 所言，“信息技术引发的问题触及隐私和言论自由等基本人权保护的核心。这些问题加重了制造这些产品的科技公司的责任。在我们看来，它们还呼吁政府进行深思熟虑的监管，并围绕可接受的用途制定规范”（[来源](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)）。

未来的情况还有待观察，但了解这些计算机系统以及它们运行的软件和算法是很重要的。我们希望这门课程能帮助你更好的理解，以便你自己决定。

[![深度学习的历史](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "深度学习的历史")
> 🎥 点击上图观看视频：Yann LeCun 在本次讲座中讨论深度学习的历史 

---
## 🚀挑战

深入了解这些历史时刻之一，并更多地了解它们背后的人。这里有许多引人入胜的人物，没有一项科学发现是在文化真空中创造出来的。你发现了什么？

## [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/4/)

## 复习与自学

以下是要观看和收听的节目：

[这是 Amy Boyd 讨论人工智能进化的播客](http://runasradio.com/Shows/Show/739)

[![Amy Boyd的《人工智能史》](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Amy Boyd的《人工智能史》")

## 任务

[创建时间线](assignment.zh-cn.md)
