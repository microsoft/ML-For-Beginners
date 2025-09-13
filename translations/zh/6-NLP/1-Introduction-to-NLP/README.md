<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T09:11:59+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "zh"
}
-->
# 自然语言处理简介

本课程涵盖了*自然语言处理*（NLP）的简要历史和重要概念，这是*计算语言学*的一个分支领域。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

## 简介

NLP是机器学习应用最广泛的领域之一，并已被用于生产软件中。

✅ 你能想到每天使用的软件中可能嵌入了NLP吗？比如你经常使用的文字处理程序或手机应用？

你将学习以下内容：

- **语言的概念**。了解语言的发展以及主要研究领域。
- **定义和概念**。你还将学习计算机如何处理文本的定义和概念，包括解析、语法以及识别名词和动词。本课程中有一些编码任务，并引入了几个重要概念，这些概念将在后续课程中学习如何编写代码。

## 计算语言学

计算语言学是一个研究和开发领域，已有数十年的历史，研究计算机如何与语言协作，甚至理解、翻译和与语言进行交流。自然语言处理（NLP）是一个相关领域，专注于计算机如何处理“自然”或人类语言。

### 示例 - 手机语音输入

如果你曾经对手机进行语音输入而不是打字，或者向虚拟助手提问，你的语音会被转换为文本形式，然后被处理或*解析*成你所说的语言。检测到的关键词随后会被处理成手机或助手可以理解并执行的格式。

![理解](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> 真正的语言理解很难！图片来源：[Jen Looper](https://twitter.com/jenlooper)

### 这项技术是如何实现的？

这项技术的实现是因为有人编写了一个计算机程序来完成这些任务。几十年前，一些科幻作家预测人们将主要通过语音与计算机交流，而计算机将始终准确理解他们的意思。然而，事实证明这是一个比许多人想象的更难的问题。尽管今天对这个问题的理解已经大大提高，但在实现“完美”的自然语言处理以理解句子的意义时仍然存在重大挑战。尤其是在理解幽默或检测句子中的情感（如讽刺）时，这个问题尤为困难。

此时，你可能会回忆起学校课堂上老师讲解句子语法部分的情景。在一些国家，学生会专门学习语法和语言学，而在许多国家，这些主题是语言学习的一部分：小学学习母语（学习阅读和写作），高中可能学习第二语言。如果你不擅长区分名词和动词或副词和形容词，不用担心！

如果你对区分*一般现在时*和*现在进行时*感到困难，你并不孤单。这对许多人来说是一个挑战，即使是母语使用者。好消息是，计算机非常擅长应用正式规则，你将学习编写代码来像人类一样*解析*句子。更大的挑战是理解句子的*意义*和*情感*。

## 前置知识

本课程的主要前置知识是能够阅读和理解本课程的语言。本课程没有数学问题或需要解决的方程。虽然课程的原作者是用英语编写的，但它也被翻译成其他语言，因此你可能正在阅读翻译版本。本课程中有一些例子使用了多种语言（用于比较不同语言的语法规则）。这些例子*没有*被翻译，但解释性文本是翻译过的，因此意义应该是清晰的。

对于编码任务，你将使用Python，示例使用的是Python 3.8。

在本节中，你将需要并使用以下内容：

- **Python 3理解能力**。理解Python 3编程语言，本课程使用输入、循环、文件读取、数组。
- **Visual Studio Code + 扩展**。我们将使用Visual Studio Code及其Python扩展。你也可以使用自己选择的Python IDE。
- **TextBlob**。 [TextBlob](https://github.com/sloria/TextBlob) 是一个简化的Python文本处理库。按照TextBlob网站上的说明将其安装到你的系统中（同时安装语料库，如下所示）：

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 提示：你可以直接在VS Code环境中运行Python。查看[文档](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott)了解更多信息。

## 与机器对话

让计算机理解人类语言的历史可以追溯到几十年前，最早考虑自然语言处理的科学家之一是*艾伦·图灵*。

### 图灵测试

当图灵在20世纪50年代研究*人工智能*时，他提出了一个对话测试：通过打字交流，让人类和计算机进行对话，而人类无法确定自己是在与另一个人还是计算机交流。

如果在一定时间的对话后，人类无法判断回答是来自计算机还是人类，那么是否可以说计算机在“思考”？

### 灵感来源 - 模仿游戏

这个想法来源于一个叫*模仿游戏*的派对游戏，游戏中一个审问者独自待在一个房间里，任务是判断另一个房间里的两个人分别是男性还是女性。审问者可以发送纸条，并试图提出问题，通过书面回答来揭示神秘人物的性别。当然，另一个房间里的玩家会试图通过回答问题来误导或迷惑审问者，同时也要表现得像是在诚实回答。

### 开发Eliza

在20世纪60年代，麻省理工学院的科学家*约瑟夫·魏岑鲍姆*开发了[*Eliza*](https://wikipedia.org/wiki/ELIZA)，一个计算机“治疗师”，它会向人类提问并表现出理解他们的回答。然而，虽然Eliza可以解析句子并识别某些语法结构和关键词以给出合理的回答，但它不能说是*理解*句子。如果Eliza收到一个格式为“**我很**<u>难过</u>”的句子，它可能会重新排列并替换句子中的单词，形成“你**已经**<u>难过</u>多久了”的回答。

这给人一种Eliza理解了陈述并提出了后续问题的印象，而实际上它只是改变了时态并添加了一些单词。如果Eliza无法识别一个关键词，它会给出一个随机回答，这可能适用于许多不同的陈述。例如，如果用户写“**你是**一辆<u>自行车</u>”，它可能会回答“我**已经**是一辆<u>自行车</u>多久了？”，而不是一个更合理的回答。

[![与Eliza聊天](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "与Eliza聊天")

> 🎥 点击上方图片观看关于原始ELIZA程序的视频

> 注意：如果你有ACM账户，可以阅读1966年发表的[Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract)原始描述。或者，可以在[wikipedia](https://wikipedia.org/wiki/ELIZA)上了解Eliza。

## 练习 - 编写一个基础对话机器人

一个对话机器人，比如Eliza，是一个能够引导用户输入并表现出理解和智能回应的程序。与Eliza不同，我们的机器人不会有多个规则来表现出智能对话。相反，我们的机器人只有一个功能，即通过随机回应来保持对话，这些回应可能适用于几乎任何简单对话。

### 计划

构建对话机器人的步骤：

1. 打印说明，告知用户如何与机器人互动
2. 开始一个循环
   1. 接收用户输入
   2. 如果用户要求退出，则退出
   3. 处理用户输入并确定回应（在本例中，回应是从可能的通用回应列表中随机选择）
   4. 打印回应
3. 返回步骤2

### 构建机器人

接下来我们来创建机器人。首先定义一些短语。

1. 使用以下随机回应在Python中创建这个机器人：

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    以下是一些示例输出（用户输入以`>`开头的行）：

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

    1. 你认为随机回应能否“欺骗”某人，让他们认为机器人真的理解了他们？
    2. 机器人需要哪些功能才能更有效？
    3. 如果一个机器人真的能“理解”句子的意义，它是否需要“记住”对话中前几句的意义？

---

## 🚀挑战

选择上述“停下来思考”中的一个元素，尝试用代码实现它，或者用伪代码在纸上写出解决方案。

在下一节课中，你将学习其他解析自然语言和机器学习的方法。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

查看以下参考资料，作为进一步阅读的机会。

### 参考资料

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 作业 

[寻找一个机器人](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们不对因使用此翻译而产生的任何误解或误读承担责任。