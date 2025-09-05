<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T09:13:02+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "zh"
}
-->
# 使用酒店评论进行情感分析

现在您已经详细探索了数据集，是时候筛选列并对数据集应用NLP技术，以便获得关于酒店的新见解。

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

### 筛选与情感分析操作

正如您可能已经注意到的，数据集存在一些问题。一些列充满了无用的信息，另一些列看起来不正确。即使它们是正确的，也不清楚它们是如何计算的，您无法通过自己的计算独立验证答案。

## 练习：进一步处理数据

对数据进行更多清理。添加一些后续会用到的列，修改其他列中的值，并完全删除某些列。

1. 初步列处理

   1. 删除 `lat` 和 `lng`

   2. 将 `Hotel_Address` 的值替换为以下值（如果地址中包含城市和国家的名称，则将其更改为仅包含城市和国家）。

      数据集中仅包含以下城市和国家：

      阿姆斯特丹，荷兰  
      巴塞罗那，西班牙  
      伦敦，英国  
      米兰，意大利  
      巴黎，法国  
      维也纳，奥地利  

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      现在您可以查询国家级别的数据：

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | 阿姆斯特丹，荷兰       |    105     |
      | 巴塞罗那，西班牙       |    211     |
      | 伦敦，英国             |    400     |
      | 米兰，意大利           |    162     |
      | 巴黎，法国             |    458     |
      | 维也纳，奥地利         |    158     |

2. 处理酒店元评论列

   1. 删除 `Additional_Number_of_Scoring`

   2. 将 `Total_Number_of_Reviews` 替换为数据集中该酒店实际的评论总数

   3. 用我们自己计算的分数替换 `Average_Score`

      ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. 处理评论列

   1. 删除 `Review_Total_Negative_Word_Counts`、`Review_Total_Positive_Word_Counts`、`Review_Date` 和 `days_since_review`

   2. 保留 `Reviewer_Score`、`Negative_Review` 和 `Positive_Review` 不变

   3. 暂时保留 `Tags`

      - 我们将在下一部分对标签进行一些额外的筛选操作，然后再删除标签

4. 处理评论者列

   1. 删除 `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. 保留 `Reviewer_Nationality`

### 标签列

`Tag` 列是一个问题，因为它是一个以文本形式存储的列表。不幸的是，该列中的子部分顺序和数量并不总是相同的。由于数据集有515,000行和1427家酒店，每个评论者可以选择的选项略有不同，因此人类很难识别出需要关注的正确短语。这正是NLP的优势所在。您可以扫描文本，找到最常见的短语并统计它们的数量。

不幸的是，我们对单个单词不感兴趣，而是对多词短语（例如 *商务旅行*）感兴趣。在如此庞大的数据（6762646个单词）上运行多词频率分布算法可能需要极长的时间，但在不了解数据的情况下，这似乎是必要的开销。这时，探索性数据分析就派上用场了，因为您已经看到了标签的样本，例如 `[' 商务旅行 ', ' 独自旅行者 ', ' 单人房 ', ' 住了5晚 ', ' 从移动设备提交 ']`，您可以开始思考是否有可能大幅减少需要处理的数据量。幸运的是，这是可能的——但首先您需要遵循一些步骤来确定感兴趣的标签。

### 筛选标签

记住，数据集的目标是添加情感和列，以帮助您选择最佳酒店（无论是为自己还是为客户创建一个酒店推荐机器人）。您需要问自己，这些标签在最终数据集中是否有用。以下是一个解释（如果您出于其他原因需要数据集，不同的标签可能会被保留或删除）：

1. 旅行类型是相关的，应该保留
2. 客人群体类型是重要的，应该保留
3. 客人入住的房间、套房或工作室类型是无关的（所有酒店基本上都有相同的房间）
4. 提交评论的设备是无关的
5. 评论者入住的晚数*可能*相关，如果您认为更长的入住时间意味着他们更喜欢酒店，但这有点牵强，可能无关

总之，**保留两类标签，删除其他标签**。

首先，您不想在标签格式更好之前统计它们，因此需要移除方括号和引号。您可以通过多种方式完成此操作，但您需要最快的方法，因为处理大量数据可能需要很长时间。幸运的是，pandas 提供了一种简单的方法来完成这些步骤。

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

每个标签变成类似于：`商务旅行, 独自旅行者, 单人房, 住了5晚, 从移动设备提交`。

接下来我们发现一个问题。一些评论（或行）有5列，一些有3列，一些有6列。这是数据集创建方式的结果，很难修复。您希望统计每个短语的频率，但它们在每条评论中的顺序不同，因此统计可能会出错，某些酒店可能没有被分配到它应得的标签。

相反，您可以利用不同的顺序，因为每个标签是多词的，但也用逗号分隔！最简单的方法是创建6个临时列，将每个标签插入到对应顺序的列中。然后，您可以将这6列合并为一个大列，并对结果列运行 `value_counts()` 方法。打印出来后，您会看到有2428个唯一标签。以下是一个小样本：

| 标签                              | 计数   |
| --------------------------------- | ------ |
| 休闲旅行                         | 417778 |
| 从移动设备提交                   | 307640 |
| 夫妻                             | 252294 |
| 住了1晚                          | 193645 |
| 住了2晚                          | 133937 |
| 独自旅行者                       | 108545 |
| 住了3晚                          | 95821  |
| 商务旅行                         | 82939  |
| 团体                             | 65392  |
| 带小孩的家庭                     | 61015  |
| 住了4晚                          | 47817  |
| 双人房                           | 35207  |
| 标准双人房                       | 32248  |
| 高级双人房                       | 31393  |
| 带大孩的家庭                     | 26349  |
| 豪华双人房                       | 24823  |
| 双人或双床房                     | 22393  |
| 住了5晚                          | 20845  |
| 标准双人或双床房                 | 17483  |
| 经典双人房                       | 16989  |
| 高级双人或双床房                 | 13570  |
| 2间房                            | 12393  |

一些常见标签如 `从移动设备提交` 对我们没有用，因此在统计短语出现次数之前删除它们可能是明智的，但由于这是一个非常快速的操作，您可以将它们保留并忽略它们。

### 删除入住时长标签

删除这些标签是第一步，这稍微减少了需要考虑的标签总数。注意，您并没有从数据集中删除它们，只是选择不将它们作为评论数据集中需要统计/保留的值。

| 入住时长       | 计数   |
| -------------- | ------ |
| 住了1晚        | 193645 |
| 住了2晚        | 133937 |
| 住了3晚        | 95821  |
| 住了4晚        | 47817  |
| 住了5晚        | 20845  |
| 住了6晚        | 9776   |
| 住了7晚        | 7399   |
| 住了8晚        | 2502   |
| 住了9晚        | 1293   |
| ...            | ...    |

房间、套房、工作室、公寓等类型种类繁多。它们的意义大致相同，对您来说并不重要，因此从考虑中删除它们。

| 房间类型                     | 计数  |
| ---------------------------- | ----- |
| 双人房                       | 35207 |
| 标准双人房                   | 32248 |
| 高级双人房                   | 31393 |
| 豪华双人房                   | 24823 |
| 双人或双床房                 | 22393 |
| 标准双人或双床房             | 17483 |
| 经典双人房                   | 16989 |
| 高级双人或双床房             | 13570 |

最后，令人欣喜的是（因为几乎不需要处理），您将剩下以下**有用**的标签：

| 标签                                           | 计数   |
| --------------------------------------------- | ------ |
| 休闲旅行                                      | 417778 |
| 夫妻                                         | 252294 |
| 独自旅行者                                   | 108545 |
| 商务旅行                                     | 82939  |
| 团体（与朋友旅行者合并）                     | 67535  |
| 带小孩的家庭                                 | 61015  |
| 带大孩的家庭                                 | 26349  |
| 带宠物                                       | 1405   |

您可以认为 `与朋友旅行者` 与 `团体` 基本相同，将两者合并是合理的，如上所示。识别正确标签的代码在 [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) 中。

最后一步是为每个这些标签创建新列。然后，对于每条评论行，如果 `Tag` 列与新列之一匹配，则添加1，否则添加0。最终结果将是一个统计数据，显示有多少评论者选择了这家酒店（总体上）用于商务、休闲或带宠物入住，这在推荐酒店时是有用的信息。

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### 保存文件

最后，将当前数据集保存为一个新名称。

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## 情感分析操作

在最后一部分中，您将对评论列应用情感分析，并将结果保存到数据集中。

## 练习：加载并保存筛选后的数据

注意，现在您加载的是上一部分保存的筛选后的数据集，而**不是**原始数据集。

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### 删除停用词

如果您对负面和正面评论列运行情感分析，可能需要很长时间。在一台性能强劲的测试笔记本电脑上测试时，根据使用的情感分析库不同，耗时为12到14分钟。这是一个（相对）较长的时间，因此值得研究是否可以加快速度。

删除停用词（即不会改变句子情感的常见英语单词）是第一步。通过删除它们，情感分析应该会运行得更快，但不会降低准确性（因为停用词不会影响情感，但会减慢分析速度）。

最长的负面评论有395个单词，但删除停用词后仅剩195个单词。

删除停用词也是一个快速操作，在测试设备上，从2个评论列中删除515,000行的停用词耗时3.3秒。根据您的设备CPU速度、内存、是否有SSD以及其他一些因素，这个时间可能略长或略短。操作相对较短，这意味着如果它能提高情感分析速度，那么值得一试。

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### 执行情感分析

现在，您应该计算负面和正面评论列的情感分析，并将结果存储在2个新列中。情感分析的测试是将其与同一评论的评论者评分进行比较。例如，如果情感分析认为负面评论的情感为1（极其正面的情感），正面评论的情感也为1，但评论者给酒店的评分是最低分，那么要么评论文本与评分不匹配，要么情感分析器无法正确识别情感。您应该预期某些情感评分完全错误，这通常是可以解释的，例如评论可能极具讽刺意味，“当然，我*喜欢*住在没有暖气的房间里”，情感分析器可能认为这是正面情感，但人类阅读时会知道这是讽刺。
NLTK 提供了不同的情感分析器供学习使用，您可以替换它们并查看情感分析的准确性是否有所不同。这里使用的是 VADER 情感分析。

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: 一种简洁的基于规则的社交媒体文本情感分析模型。第八届国际博客与社交媒体会议 (ICWSM-14)。美国密歇根州安娜堡，2014年6月。

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

在程序中，当您准备计算情感时，可以将其应用到每条评论，如下所示：

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

在我的电脑上大约需要 120 秒，但每台电脑的运行时间会有所不同。如果您想打印结果并查看情感是否与评论匹配：

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

在挑战中使用文件之前，最后要做的事情就是保存它！您还应该考虑重新排列所有新列，使其更易于操作（对人类来说，这只是一个外观上的调整）。

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

您应该运行 [分析笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) 的完整代码（在运行 [过滤笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) 生成 Hotel_Reviews_Filtered.csv 文件之后）。

回顾一下，步骤如下：

1. 原始数据集文件 **Hotel_Reviews.csv** 在上一课中通过 [探索笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) 进行了探索。
2. Hotel_Reviews.csv 通过 [过滤笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) 过滤，生成 **Hotel_Reviews_Filtered.csv**。
3. Hotel_Reviews_Filtered.csv 通过 [情感分析笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) 处理，生成 **Hotel_Reviews_NLP.csv**。
4. 在下面的 NLP 挑战中使用 Hotel_Reviews_NLP.csv。

### 结论

在开始时，您有一个包含列和数据的数据集，但并非所有数据都可以验证或使用。您已经探索了数据，过滤掉了不需要的部分，将标签转换为有用的内容，计算了自己的平均值，添加了一些情感列，并希望学到了一些关于处理自然文本的有趣知识。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 挑战

现在您已经对数据集进行了情感分析，试着使用您在本课程中学到的策略（例如聚类）来确定情感的模式。

## 复习与自学

学习 [这个模块](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)，了解更多内容并使用不同的工具探索文本中的情感。

## 作业

[尝试一个不同的数据集](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。对于因使用本翻译而引起的任何误解或误读，我们概不负责。