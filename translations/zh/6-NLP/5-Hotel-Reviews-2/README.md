<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-03T19:11:43+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "zh"
}
-->
# 使用酒店评论进行情感分析

现在您已经详细探索了数据集，是时候过滤列并使用 NLP 技术对数据集进行分析，以获得关于酒店的新见解了。
## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### 过滤与情感分析操作

正如您可能已经注意到的，数据集存在一些问题。一些列充满了无用的信息，另一些列似乎不正确。如果它们是正确的，也不清楚它们是如何计算的，答案无法通过您自己的计算独立验证。

## 练习：进一步处理数据

对数据进行更多清理。添加以后有用的列，更改其他列中的值，并完全删除某些列。

1. 初步列处理

   1. 删除 `lat` 和 `lng`

   2. 将 `Hotel_Address` 的值替换为以下值（如果地址包含城市和国家的名称，则将其更改为仅城市和国家）。

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

      现在您可以查询国家级数据：

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | 阿姆斯特丹，荷兰         |    105     |
      | 巴塞罗那，西班牙         |    211     |
      | 伦敦，英国               |    400     |
      | 米兰，意大利             |    162     |
      | 巴黎，法国               |    458     |
      | 维也纳，奥地利           |    158     |

2. 处理酒店元评论列

  1. 删除 `Additional_Number_of_Scoring`

  1. 将 `Total_Number_of_Reviews` 替换为数据集中该酒店实际的评论总数

  1. 用我们自己计算的分数替换 `Average_Score`

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

     - 我们将在下一部分对标签进行一些额外的过滤操作，然后删除标签

4. 处理评论者列

  1. 删除 `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. 保留 `Reviewer_Nationality`

### 标签列

`Tag` 列是一个问题，因为它是一个以文本形式存储在列中的列表。不幸的是，这列中的子部分的顺序和数量并不总是相同。由于有 515,000 行和 1427 家酒店，每个评论者可以选择的选项略有不同，因此人类很难识别出正确的短语。这正是 NLP 的优势所在。您可以扫描文本，找到最常见的短语并统计它们。

不幸的是，我们对单词不感兴趣，而是对多词短语（例如 *商务旅行*）感兴趣。对如此多的数据（6762646 个单词）运行多词频率分布算法可能需要非常长的时间，但如果不查看数据，这似乎是必要的开销。这时探索性数据分析就派上了用场，因为您已经看到了标签的样本，例如 `[' 商务旅行 ', ' 独行旅客 ', ' 单人房 ', ' 住了 5 晚 ', ' 从移动设备提交 ']`，您可以开始询问是否有可能大大减少您需要处理的数据量。幸运的是，可以做到这一点——但首先您需要遵循一些步骤来确定感兴趣的标签。

### 过滤标签

请记住，数据集的目标是添加情感和列，以帮助您选择最佳酒店（为自己或可能是委托您制作酒店推荐机器人的客户）。您需要问自己这些标签在最终数据集中是否有用。以下是一个解释（如果您需要数据集用于其他目的，不同的标签可能会保留或删除）：

1. 旅行类型是相关的，应该保留
2. 客人群体类型很重要，应该保留
3. 客人入住的房间、套房或工作室类型无关紧要（所有酒店基本上都有相同的房间）
4. 评论提交设备无关紧要
5. 评论者入住的夜晚数量*可能*相关，如果您将较长的入住时间与他们更喜欢酒店联系起来，但这只是一个假设，可能无关紧要

总而言之，**保留两种标签并删除其他标签**。

首先，您不想在标签格式更好之前统计标签，这意味着删除方括号和引号。您可以通过多种方式执行此操作，但您需要最快的方法，因为处理大量数据可能需要很长时间。幸运的是，pandas 提供了一个简单的方法来完成每个步骤。

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

每个标签变成类似于：`商务旅行, 独行旅客, 单人房, 住了 5 晚, 从移动设备提交`。

接下来我们发现一个问题。一些评论或行有 5 列，一些有 3 列，一些有 6 列。这是数据集创建方式的结果，很难修复。您希望获得每个短语的频率计数，但它们在每个评论中的顺序不同，因此计数可能不准确，酒店可能没有获得它应得的标签。

相反，您将利用不同的顺序，因为每个标签都是多词的，但也由逗号分隔！最简单的方法是创建 6 个临时列，每个标签插入到与其在标签中的顺序对应的列中。然后，您可以将 6 列合并为一个大列，并对生成的列运行 `value_counts()` 方法。打印出来后，您会看到有 2428 个唯一标签。以下是一个小样本：

| Tag                            | Count  |
| ------------------------------ | ------ |
| 休闲旅行                       | 417778 |
| 从移动设备提交                 | 307640 |
| 夫妻                           | 252294 |
| 住了 1 晚                      | 193645 |
| 住了 2 晚                      | 133937 |
| 独行旅客                       | 108545 |
| 住了 3 晚                      | 95821  |
| 商务旅行                       | 82939  |
| 团体                           | 65392  |
| 带小孩的家庭                   | 61015  |
| 住了 4 晚                      | 47817  |
| 双人房                         | 35207  |
| 标准双人房                     | 32248  |
| 高级双人房                     | 31393  |
| 带大孩的家庭                   | 26349  |
| 豪华双人房                     | 24823  |
| 双人或双床房                   | 22393  |
| 住了 5 晚                      | 20845  |
| 标准双人或双床房               | 17483  |
| 经典双人房                     | 16989  |
| 高级双人或双床房               | 13570  |
| 2 间房                         | 12393  |

一些常见标签如 `从移动设备提交` 对我们没有用，因此在统计短语出现次数之前删除它们可能是明智的，但这是一个非常快速的操作，您可以将它们保留并忽略它们。

### 删除入住时长标签

删除这些标签是第一步，它稍微减少了需要考虑的标签总数。注意，您并没有从数据集中删除它们，只是选择不将它们作为评论数据集中的值进行统计或保留。

| 入住时长       | Count  |
| -------------- | ------ |
| 住了 1 晚      | 193645 |
| 住了 2 晚      | 133937 |
| 住了 3 晚      | 95821  |
| 住了 4 晚      | 47817  |
| 住了 5 晚      | 20845  |
| 住了 6 晚      | 9776   |
| 住了 7 晚      | 7399   |
| 住了 8 晚      | 2502   |
| 住了 9 晚      | 1293   |
| ...            | ...    |

有各种各样的房间、套房、工作室、公寓等等。它们的意义大致相同，对您来说并不重要，因此从考虑中删除它们。

| 房间类型                  | Count |
| ------------------------- | ----- |
| 双人房                   | 35207 |
| 标准双人房               | 32248 |
| 高级双人房               | 31393 |
| 豪华双人房               | 24823 |
| 双人或双床房             | 22393 |
| 标准双人或双床房         | 17483 |
| 经典双人房               | 16989 |
| 高级双人或双床房         | 13570 |

最后，这令人欣喜（因为几乎不需要处理），您将只剩下以下**有用**的标签：

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| 休闲旅行                                      | 417778 |
| 夫妻                                          | 252294 |
| 独行旅客                                      | 108545 |
| 商务旅行                                      | 82939  |
| 团体（与朋友旅行者合并）                      | 67535  |
| 带小孩的家庭                                  | 61015  |
| 带大孩的家庭                                  | 26349  |
| 带宠物                                        | 1405   |

您可以认为 `与朋友旅行者` 与 `团体` 基本相同，这样合并是合理的，如上所示。识别正确标签的代码在 [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)。

最后一步是为每个标签创建新列。然后，对于每个评论行，如果 `Tag` 列与新列之一匹配，则添加 1，否则添加 0。最终结果将是统计有多少评论者选择了这家酒店（总计）用于例如商务旅行还是休闲旅行，或者是否带宠物，这在推荐酒店时是有用的信息。

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

在最后一部分中，您将对评论列应用情感分析，并将结果保存在数据集中。

## 练习：加载并保存过滤后的数据

请注意，现在您加载的是上一部分保存的过滤后的数据集，而**不是**原始数据集。

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

如果您对负面和正面评论列运行情感分析，可能需要很长时间。在一台强大的测试笔记本电脑上测试，使用快速 CPU，运行时间为 12 - 14 分钟，具体取决于使用的情感库。这是一个（相对）较长的时间，因此值得研究是否可以加快速度。

删除停用词，即不会改变句子情感的常见英语词，是第一步。通过删除它们，情感分析应该运行得更快，但不会降低准确性（因为停用词不会影响情感，但它们确实会减慢分析速度）。

最长的负面评论有 395 个单词，但删除停用词后只有 195 个单词。

删除停用词也是一个快速操作，在测试设备上从 2 个评论列的 515,000 行中删除停用词只用了 3.3 秒。根据您的设备 CPU 速度、RAM、是否有 SSD 以及其他一些因素，可能需要稍多或稍少的时间。操作相对较短，这意味着如果它能改善情感分析时间，那么值得进行。

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
现在，您应该计算负面评论和正面评论列的情感分析，并将结果存储在两个新列中。情感测试的方式是将其与评论者对同一评论的评分进行比较。例如，如果情感分析认为负面评论的情感得分为1（极度正面情感），正面评论的情感得分也为1，但评论者给酒店打了最低分，那么要么评论文本与评分不匹配，要么情感分析器无法正确识别情感。您应该预期某些情感得分完全错误，这通常是可以解释的，例如评论可能极具讽刺意味：“当然，我喜欢住在一个没有暖气的房间里”，情感分析器可能认为这是正面情感，但人类阅读时会知道这是讽刺。

NLTK提供了不同的情感分析器供学习使用，您可以替换它们，看看情感分析是否更准确。这里使用的是VADER情感分析。

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: 一种基于规则的简约模型，用于社交媒体文本的情感分析。第八届国际博客与社交媒体会议（ICWSM-14）。美国密歇根州安娜堡，2014年6月。

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

在程序中，当您准备好计算情感时，可以将其应用于每条评论，如下所示：

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

在我的电脑上大约需要120秒，但每台电脑的时间会有所不同。如果您想打印结果并查看情感是否与评论匹配：

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

在挑战中使用文件之前，最后要做的事情是保存它！您还应该考虑重新排列所有新列，使其更易于操作（对人类来说，这只是一个外观上的调整）。

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

您应该运行[分析笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)的完整代码（在运行[过滤笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)以生成Hotel_Reviews_Filtered.csv文件之后）。

回顾一下，步骤如下：

1. 原始数据集文件**Hotel_Reviews.csv**在上一课中通过[探索笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)进行了探索。
2. **Hotel_Reviews.csv**通过[过滤笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)过滤，生成**Hotel_Reviews_Filtered.csv**。
3. **Hotel_Reviews_Filtered.csv**通过[情感分析笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)处理，生成**Hotel_Reviews_NLP.csv**。
4. 在下面的NLP挑战中使用**Hotel_Reviews_NLP.csv**。

### 结论

当您开始时，您有一个包含列和数据的数据集，但并非所有数据都可以验证或使用。您已经探索了数据，过滤掉了不需要的部分，将标签转换为有用的内容，计算了自己的平均值，添加了一些情感列，并希望学到了一些关于处理自然文本的有趣知识。

## [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## 挑战

现在您已经对数据集进行了情感分析，试着使用您在本课程中学到的策略（例如聚类）来确定情感模式。

## 复习与自学

学习[这个模块](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)，了解更多内容并使用不同工具探索文本中的情感。

## 作业

[尝试一个不同的数据集](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。因使用本翻译而导致的任何误解或误读，我们概不负责。