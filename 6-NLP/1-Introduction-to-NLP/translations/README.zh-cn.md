# 自然语言处理介绍
这节课讲解了*自然语言处理*简要历史和重要概念，*自然语言处理*是计算语言学的一个子领域。

## [课前测验](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/31/)

## 介绍
众所周知，自然语言处理(Natural Language Processing, NLP)是机器学习在生产软件中应用最广泛的领域之一。

✅你能想到哪些你日常生活中使用的软件嵌入了某些自然语言处理技术呢？你经常使用的文字处理程序或移动应用程序是否嵌入了自然语言处理技术呢？

你将会学习到：

- **语言的思想**. 语言的发展历程及主要研究领域.
- **定义和概念**. 你还将学习到有关计算机如何处理文本的定义和概念，包括解析、语法以及名词和动词的识别。本节课程包含一些编码任务并介绍了几个重要的概念，你将在下一节课中学习编码实现这些概念。
## 计算语言学

计算语言学是一个经过几十年研究和发展的领域，它研究计算机如何使用语言、理解语言、翻译语言及使用语言交流。自然语言处理(NLP)是计算语言学中一个专注于计算机如何处理“自然”或人类语言的相关领域，
### 例子 - 电话号码识别

如果你曾经在手机上使用语音输入替代键盘输入或者向语音助手小娜提问，那么你的语音将被转录为文本形式后进行处理或者叫*解析*。被检测到的关键字最后将被处理成手机或语音助手可以理解并采取行动的格式。

![comprehension](../images/comprehension.png)
> 真实的语言理解十分困难！图源：[Jen Looper](https://twitter.com/jenlooper)
### 这项技术是如何实现的？

有人编写了一个计算机程序来实现这项技术。几十年前，一些科幻作家预测人类很大可能会和他们的电脑对话，而电脑总是能准确地理解人类的意思。可惜的是，事实证明这是一个比许多人想象中更难实现的问题，虽然今天这个问题已经被初步解决，但在理解句子的含义时，要实现“完美”的自然语言处理仍然存在重大挑战。句子中的幽默理解或讽刺等情绪的检测是一个特别困难的问题。

此时，你可能会想起学校课堂上老师讲解的部分句子语法。在某些国家/地区，语法和语言学知识是学生的专题课内容。但在另一些国家/地区，不管是在小学时的第一语言（学习阅读和写作），或者在高年级及高中时学习的第二语言中，语法及语言学知识是作为学习语言的一部分教学的。如果你不能很好地区分名词与动词或者区分副词与形容词，请不要担心！

如果你还为区分*一般现在时*与*现在进行时*而烦恼，你并不是一个人。即使是对以这门语言为母语的人在内的很多人来说这都是一项有挑战性的任务。好消息是，计算机非常善于应用标准的规则，你将学会编写可以像人一样"解析"句子的代码。稍后你将面对的更大挑战是理解句子的*语义*和*情绪*。
## 前提

本节教程的主要先决条件是能够阅读和理解本节教程的语言。本节中没有数学问题或方程需要解决。虽然原作者用英文写了这教程，但它也被翻译成其他语言，所以你可能在阅读翻译内容。有使用多种不同语言的示例（以比较不同语言的不同语法规则）。这些是*未*翻译的，但解释性文本是翻译内容，所以表义应当是清晰的。

编程任务中，你将会使用Python语言，示例使用的是Python 3.8版本。

在本节中你将需要并使用：

- **Python 3 理解**.  Python 3中的编程语言理解，本课使用输入、循环、文件读取、数组。
- **Visual Studio Code + 扩展**. 我们将使用 Visual Studio Code 及其 Python 扩展。你还可以使用你选择的 Python IDE。
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob)是一个简化的 Python 文本处理库。按照 TextBlob 网站上的说明在您的系统上安装它（也安装语料库，如下所示）：
- 
   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 提示：可以在 VS Code 环境中直接运行 Python。 点击[docs](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-15963-cxa)查看更多信息。

## 与机器对话

试图让计算机理解人类语言的历史可以追溯到几十年前，最早考虑自然语言处理的科学家之一是 *Alan Turing*。
### 图灵测试

当图灵在1950年代研究*人工智能*时，他考虑是否可以对人和计算机进行对话测试（通过打字对应），其中对话中的人不确定他们是在与另一个人交谈还是与计算机交谈.

如果经过一定时间的交谈，人类无法确定答案是否来自计算机，那么是否可以说计算机正在“思考”？
### 灵感 - “模仿游戏”

这个想法来自一个名为 *模仿游戏* 的派对游戏，其中一名审讯者独自一人在一个房间里，负责确定两个人（在另一个房间里）是男性还是女性。审讯者可以传递笔记，并且需要想出能够揭示神秘人性别的问题。当然，另一个房间的玩家试图通过回答问题的方式来欺骗审讯者，例如误导或迷惑审讯者，同时表现出诚实回答的样子。

### Eliza的研发

在 1960 年代，一位名叫 *Joseph Weizenbaum* 的麻省理工学院科学家开发了[*Eliza*](https:/wikipedia.org/wiki/ELIZA)，Eliza是一位计算机“治疗师”，它可以向人类提出问题并表现出理解他们的答案。然而，虽然 Eliza 可以解析句子并识别某些语法结构和关键字以给出合理的答案，但不能说它*理解*了句子。如果 Eliza 看到的句子格式为“**I am** <u>sad</u>”，它可能会重新排列并替换句子中的单词以形成响应“How long have **you been** <u>sad</u>"。

这给人的印象是伊丽莎理解了这句话，并在问一个后续问题，而实际上，它是在改变时态并添加一些词。如果 Eliza 无法识别它有响应的关键字，它会给出一个随机响应，该响应可以适用于许多不同的语句。 Eliza 很容易被欺骗，例如，如果用户写了**You are** a <u>bicycle</u>"，它可能会回复"How long have **I been** a <u>bicycle</u>?"，而不是更合理的回答。

[![Chatting with Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 点击上方的图片查看真实的ELIZA程序视频

> 注意:如果你拥有ACM账户，你可以阅读1996年发表的[Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract)的原始介绍。或者，在[wikipedia](https://wikipedia.org/wiki/ELIZA)阅读有关 Eliza 的信息 

## 联系 - 编码实现一个基础的对话机器人

像 Eliza 一样的对话机器人是一个似乎可以智能地理解和响应用户输入的程序。与 Eliza 不同的是，我们的机器人不会用规则让它看起来像是在进行智能对话。取而代之的是，我们的对话机器人将只有一种能力，通过几乎在所有琐碎对话中都适用的随机响应保持对话的进行。

### 计划

搭建聊天机器人的步骤

1. 打印指导用户如何与机器人交互的说明
2. 开启循环
   1. 获取用户输入
   2. 如果用户要求退出，就退出
   3. 处理用户输入并选择一个回答（在这个例子中，回答从一个可能的通用回答列表中随机选择）
   4. 打印回答
3. 重复步骤2

### 构建聊天机器人

接下来让我们构建聊天机器人。我们将从定义一些短语开始。

1. 使用以下随机响应在 Python 中自己创建此机器人：

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    以下是一些指导你的示例输出（用户输入位于以 `>` 开头的行上）：

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    该任务的一种可能解决方案在[这里](../solution/bot.py)

    ✅ 停止并思考
   
    1. 你认为随机响应会“欺骗”某人认为机器人实际上理解他们吗？
    2. 机器人需要哪些功能才能更有效？
    3. 如果机器人真的可以“理解”一个句子的意思，它是否也需要“记住”对话中前面句子的意思？

---
## 🚀挑战

选择上面的“停止并思考”元素之一，然后尝试在代码中实现它们或使用伪代码在纸上编写解决方案。

在下一课中，您将了解解析自然语言和机器学习的许多其他方法。

## [课后测验](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/32/)

## 复习与自学

看看下面的参考资料作为进一步的阅读机会。
### 参考

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 任务

[查找一个机器人](../assignment.md)
