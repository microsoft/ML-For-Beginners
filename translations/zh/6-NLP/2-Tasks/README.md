<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T09:10:13+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "zh"
}
-->
# 常见的自然语言处理任务和技术

对于大多数*自然语言处理*任务，需要将待处理的文本分解、分析，并将结果存储或与规则和数据集进行交叉引用。这些任务使程序员能够推导出文本中的_意义_、_意图_或仅仅是_词语和术语的频率_。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

让我们来探索处理文本时常用的技术。这些技术结合机器学习，可以帮助你高效地分析大量文本。然而，在将机器学习应用于这些任务之前，我们需要了解自然语言处理专家可能遇到的问题。

## 自然语言处理的常见任务

分析文本有多种方法。通过执行不同的任务，你可以理解文本并得出结论。这些任务通常按顺序进行。

### 分词

大多数自然语言处理算法的第一步可能是将文本分解为词或标记。虽然这听起来很简单，但考虑到标点符号以及不同语言的词和句子的分隔符，这可能会变得复杂。你可能需要使用多种方法来确定分界点。

![分词](../../../../6-NLP/2-Tasks/images/tokenization.png)
> 从**傲慢与偏见**中分词的示例。信息图由 [Jen Looper](https://twitter.com/jenlooper) 制作

### 嵌入

[词嵌入](https://wikipedia.org/wiki/Word_embedding)是一种将文本数据转换为数值的方式。嵌入的方式使得具有相似意义或经常一起使用的词汇聚集在一起。

![词嵌入](../../../../6-NLP/2-Tasks/images/embedding.png)
> “我对你的神经非常尊重，它们是我的老朋友。” - **傲慢与偏见**中的一句话的词嵌入。信息图由 [Jen Looper](https://twitter.com/jenlooper) 制作

✅ 尝试[这个有趣的工具](https://projector.tensorflow.org/)来实验词嵌入。点击一个词可以显示类似词的聚类，例如“toy”与“disney”、“lego”、“playstation”和“console”聚类在一起。

### 解析与词性标注

每个被分词的词都可以标注为词性，例如名词、动词或形容词。句子`the quick red fox jumped over the lazy brown dog`可能会被词性标注为：fox = 名词，jumped = 动词。

![解析](../../../../6-NLP/2-Tasks/images/parse.png)

> **傲慢与偏见**中的一句话解析示例。信息图由 [Jen Looper](https://twitter.com/jenlooper) 制作

解析是识别句子中哪些词彼此相关，例如`the quick red fox jumped`是一个形容词-名词-动词序列，与`lazy brown dog`序列分开。

### 词和短语频率

分析大量文本时，一个有用的步骤是构建一个字典，记录每个感兴趣的词或短语及其出现的频率。短语`the quick red fox jumped over the lazy brown dog`中，词`the`的频率为2。

让我们看一个示例文本，统计词频。拉迪亚德·吉卜林的诗《胜利者》中有以下诗句：

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

由于短语频率可以根据需要区分大小写，短语`a friend`的频率为2，`the`的频率为6，`travels`的频率为2。

### N-grams

文本可以分解为固定长度的词序列，例如单词（unigram）、两个词（bigram）、三个词（trigram）或任意数量的词（n-grams）。

例如，`the quick red fox jumped over the lazy brown dog`的n-gram长度为2，生成以下n-grams：

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

可以将其想象为一个滑动窗口在句子上移动。以下是长度为3的n-grams，每个句子中的n-gram用加粗表示：

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams滑动窗口](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram值为3：信息图由 [Jen Looper](https://twitter.com/jenlooper) 制作

### 名词短语提取

在大多数句子中，有一个名词是句子的主语或宾语。在英语中，通常可以通过前面的`a`、`an`或`the`来识别。通过“提取名词短语”来识别句子的主语或宾语是自然语言处理中理解句子意义的常见任务。

✅ 在句子“我无法确定时间、地点、表情或语言，这些构成了基础。这太久远了。我在不知不觉中已经开始了。”中，你能识别出名词短语吗？

在句子`the quick red fox jumped over the lazy brown dog`中，有两个名词短语：**quick red fox**和**lazy brown dog**。

### 情感分析

可以分析句子或文本的情感，即其*积极性*或*消极性*。情感通过*极性*和*客观性/主观性*来衡量。极性范围从-1.0到1.0（消极到积极），客观性范围从0.0到1.0（最客观到最主观）。

✅ 稍后你会学习使用机器学习确定情感的不同方法，但一种方法是由人工专家将词和短语分类为积极或消极，并将该模型应用于文本以计算极性分数。你能看到这种方法在某些情况下有效，而在其他情况下效果较差吗？

### 词形变化

词形变化使你能够获取一个词的单数或复数形式。

### 词形还原

*词形还原*是指获取一组词的词根或主词，例如*flew*、*flies*、*flying*的词形还原为动词*fly*。

还有一些对自然语言处理研究人员非常有用的数据库，例如：

### WordNet

[WordNet](https://wordnet.princeton.edu/)是一个包含词汇、同义词、反义词以及许多其他细节的数据库，涵盖多种语言中的每个词汇。在构建翻译、拼写检查器或任何类型的语言工具时，它非常有用。

## 自然语言处理库

幸运的是，你不需要自己构建所有这些技术，因为有许多优秀的Python库可以让非自然语言处理或机器学习专家的开发者更容易使用。在接下来的课程中会有更多示例，但这里你将学习一些有用的示例来帮助你完成下一项任务。

### 练习 - 使用`TextBlob`库

让我们使用一个名为TextBlob的库，它包含处理这些任务的有用API。TextBlob“基于[NLTK](https://nltk.org)和[pattern](https://github.com/clips/pattern)，并与它们很好地协作。”它的API中嵌入了大量机器学习功能。

> 注意：推荐给有经验的Python开发者的TextBlob[快速入门指南](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart)

在尝试识别*名词短语*时，TextBlob提供了几种提取器选项来找到名词短语。

1. 看看`ConllExtractor`。

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > 这里发生了什么？[ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor)是“一个使用ConLL-2000训练语料库进行块解析的名词短语提取器。”ConLL-2000指的是2000年计算自然语言学习会议。每年会议都会举办一个研讨会来解决一个棘手的自然语言处理问题，2000年的主题是名词块解析。一个模型在《华尔街日报》上进行了训练，“使用第15-18节作为训练数据（211727个标记），第20节作为测试数据（47377个标记）”。你可以查看使用的程序[这里](https://www.clips.uantwerpen.be/conll2000/chunking/)以及[结果](https://ifarm.nl/erikt/research/np-chunking.html)。

### 挑战 - 使用自然语言处理改进你的机器人

在上一课中，你构建了一个非常简单的问答机器人。现在，你将通过分析用户输入的情感并打印出匹配情感的响应，使Marvin更加富有同情心。你还需要识别一个`noun_phrase`并围绕它提出更多问题。

构建更好的对话机器人的步骤：

1. 打印说明，指导用户如何与机器人互动
2. 开始循环 
   1. 接收用户输入
   2. 如果用户要求退出，则退出
   3. 处理用户输入并确定适当的情感响应
   4. 如果在情感中检测到名词短语，将其复数化并围绕该主题提出更多问题
   5. 打印响应
3. 返回步骤2

以下是使用TextBlob确定情感的代码片段。注意，这里只有四种*情感响应梯度*（如果你愿意，可以增加更多）：

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

以下是一些示例输出以供参考（用户输入以`>`开头的行）：

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

任务的一个可能解决方案在[这里](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ 知识检查

1. 你认为这些富有同情心的响应能否“欺骗”某人，让他们认为机器人真的理解他们？
2. 识别名词短语是否让机器人更“可信”？
3. 为什么从句子中提取“名词短语”是一件有用的事情？

---

实现上述知识检查中的机器人并测试它。它能否欺骗你的朋友？你能让你的机器人更“可信”吗？

## 🚀挑战

尝试实现上述知识检查中的任务并测试机器人。它能否欺骗你的朋友？你能让你的机器人更“可信”吗？

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

在接下来的几节课中，你将学习更多关于情感分析的内容。通过阅读像[KDNuggets](https://www.kdnuggets.com/tag/nlp)上的文章来研究这一有趣的技术。

## 作业 

[让机器人回复](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。因使用本翻译而导致的任何误解或误读，我们概不负责。