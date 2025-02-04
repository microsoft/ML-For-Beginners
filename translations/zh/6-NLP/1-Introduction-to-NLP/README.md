# 自然语言处理简介

本课涵盖了*自然语言处理*（NLP）这一*计算语言学*的子领域的简史和重要概念。

## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## 介绍

NLP是机器学习应用和生产软件中最知名的领域之一。

✅ 你能想到每天使用的软件中可能嵌入了一些NLP吗？你经常使用的文字处理程序或移动应用程序呢？

你将学习：

- **语言的概念**。语言是如何发展的，主要的研究领域是什么。
- **定义和概念**。你还将学习计算机如何处理文本的定义和概念，包括解析、语法以及识别名词和动词。本课中有一些编码任务，并引入了一些重要概念，你将在接下来的课程中学习如何编码这些概念。

## 计算语言学

计算语言学是一个研究和开发领域，研究计算机如何处理、理解、翻译和与语言交流。自然语言处理（NLP）是一个相关领域，专注于计算机如何处理“自然”或人类语言。

### 示例 - 手机语音输入

如果你曾经使用手机语音输入而不是打字，或者向虚拟助手提问，你的语音会被转换为文本形式，然后进行处理或*解析*。检测到的关键词会被处理成手机或助手能够理解和执行的格式。

![理解](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.zh.png)
> 真正的语言理解很难！图片由[Jen Looper](https://twitter.com/jenlooper)提供

### 这种技术是如何实现的？

这是因为有人编写了一个计算机程序来实现这一点。几年前，一些科幻作家预测人们主要会与计算机对话，计算机会总是准确理解他们的意思。遗憾的是，这个问题比许多人想象的要难得多，虽然今天我们对这个问题有了更好的理解，但在实现“完美”的自然语言处理方面仍然面临重大挑战，特别是在理解句子的意义时。这在理解幽默或检测句子中的情感（如讽刺）时尤其困难。

此时，你可能会回想起学校课堂上老师讲解句子语法部分的情景。在一些国家，学生会专门学习语法和语言学，但在许多国家，这些主题是作为学习语言的一部分：在小学学习母语（学习阅读和写作），可能在中学学习第二语言。如果你不擅长区分名词和动词或副词和形容词，也不用担心！

如果你在区分*简单现在时*和*现在进行时*方面有困难，你并不孤单。这对许多人来说是一个挑战，即使是某种语言的母语者。好消息是，计算机非常擅长应用正式规则，你将学习编写代码，能够像人类一样*解析*句子。更大的挑战是理解句子的*意义*和*情感*。

## 前提条件

本课的主要前提条件是能够阅读和理解本课的语言。本课没有数学问题或方程需要解决。虽然原作者用英语写了本课，但它也被翻译成其他语言，所以你可能在阅读翻译版本。有些例子使用了不同的语言（以比较不同语言的语法规则）。这些例子*没有*翻译，但解释性文本是翻译的，所以意思应该是清楚的。

对于编码任务，你将使用Python，例子使用的是Python 3.8。

在本节中，你将需要并使用：

- **Python 3 理解**。编程语言理解Python 3，本课使用输入、循环、文件读取、数组。
- **Visual Studio Code + 扩展**。我们将使用Visual Studio Code及其Python扩展。你也可以使用你喜欢的Python IDE。
- **TextBlob**。[TextBlob](https://github.com/sloria/TextBlob)是一个简化的Python文本处理库。按照TextBlob网站上的说明将其安装到你的系统中（同时安装语料库，如下所示）：

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 提示：你可以直接在VS Code环境中运行Python。查看[文档](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott)以获取更多信息。

## 与机器对话

让计算机理解人类语言的历史可以追溯到几十年前，最早考虑自然语言处理的科学家之一是*阿兰·图灵*。

### '图灵测试'

当图灵在20世纪50年代研究*人工智能*时，他考虑是否可以给人类和计算机（通过打字通信）进行一个对话测试，让人类在对话中无法确定他们是在与另一个人还是计算机对话。

如果在一定长度的对话后，人类无法确定回答是否来自计算机，那么是否可以说计算机在*思考*？

### 灵感来源 - '模仿游戏'

这个想法来自一个叫做*模仿游戏*的聚会游戏，审问者独自在一个房间里，任务是确定另一个房间里的两个人分别是男性和女性。审问者可以发送纸条，并且必须尝试提出问题，通过书面回答来揭示神秘人物的性别。当然，另一个房间里的玩家试图通过回答问题来误导或困惑审问者，同时给出看似诚实的回答。

### 开发Eliza

在20世纪60年代，一位MIT科学家*约瑟夫·魏岑鲍姆*开发了[*Eliza*](https://wikipedia.org/wiki/ELIZA)，一个计算机“治疗师”，会向人类提问并给出理解他们答案的假象。然而，虽然Eliza可以解析句子并识别某些语法结构和关键词，从而给出合理的回答，但不能说它*理解*句子。如果Eliza遇到格式为“**我很** <u>难过</u>”的句子，它可能会重新排列并替换句子中的单词，形成“你**一直** <u>难过</u>多久了”的回答。

这给人一种Eliza理解了陈述并在问后续问题的印象，而实际上，它只是改变了时态并添加了一些单词。如果Eliza无法识别出有响应的关键词，它会给出一个随机的回答，这个回答可以适用于许多不同的陈述。例如，如果用户写“**你是** <u>自行车</u>”，它可能会回答“我**一直**是 <u>自行车</u>多久了？”，而不是一个更合理的回答。

[![与Eliza聊天](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "与Eliza聊天")

> 🎥 点击上方图片观看关于原始ELIZA程序的视频

> 注意：如果你有ACM账户，可以阅读1966年发表的[Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract)原始描述。或者，可以在[wikipedia](https://wikipedia.org/wiki/ELIZA)上了解Eliza

## 练习 - 编写一个基本的对话机器人

一个对话机器人，如Eliza，是一个引导用户输入并似乎能够理解和智能回应的程序。与Eliza不同，我们的机器人不会有多个规则来让它看起来像是在进行智能对话。相反，我们的机器人只有一个功能，就是通过随机回应来保持对话，这些回应在几乎任何琐碎的对话中都可能有效。

### 计划

构建对话机器人的步骤：

1. 打印指示，告知用户如何与机器人互动
2. 开始一个循环
   1. 接受用户输入
   2. 如果用户要求退出，则退出
   3. 处理用户输入并确定回应（在本例中，回应是从可能的通用回应列表中随机选择的）
   4. 打印回应
3. 返回第2步循环

### 构建机器人

接下来让我们创建机器人。我们将从定义一些短语开始。

1. 使用以下随机回应在Python中自己创建这个机器人：

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    这里有一些示例输出供你参考（用户输入在以`>`开头的行上）：

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

    任务的一个可能解决方案在[这里](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ 停下来思考

    1. 你认为随机回应会“骗”某人认为机器人真的理解他们吗？
    2. 机器人需要哪些功能才能更有效？
    3. 如果一个机器人真的能“理解”句子的意义，它是否需要“记住”对话中前面句子的意义？

---

## 🚀挑战

选择上面的一个“停下来思考”元素，尝试在代码中实现它们，或者用伪代码在纸上写出解决方案。

在下一课中，你将学习一些其他解析自然语言和机器学习的方法。

## [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## 复习与自学

查看下面的参考资料，作为进一步阅读的机会。

### 参考资料

1. Schubert, Lenhart, "Computational Linguistics", *斯坦福哲学百科全书* (2020年春季版), Edward N. Zalta (编), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 作业 

[搜索一个机器人](assignment.md)

**免责声明**：
本文件使用基于机器的人工智能翻译服务进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应将原始语言的文件视为权威来源。对于关键信息，建议进行专业的人类翻译。我们对使用本翻译所产生的任何误解或误读不承担责任。