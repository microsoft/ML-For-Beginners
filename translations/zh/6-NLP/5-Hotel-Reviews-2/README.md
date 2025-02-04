# 酒店评论的情感分析

现在你已经详细探索了数据集，是时候过滤列并使用NLP技术在数据集上获得关于酒店的新见解了。
## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### 过滤和情感分析操作

正如你可能已经注意到的，数据集存在一些问题。有些列充满了无用的信息，其他一些似乎不正确。如果它们是正确的，也不清楚它们是如何计算的，并且无法通过你自己的计算独立验证答案。

## 练习：更多数据处理

进一步清理数据。添加以后有用的列，改变其他列中的值，并完全删除某些列。

1. 初步列处理

   1. 删除`lat`和`lng`

   2. 将`Hotel_Address`的值替换为以下值（如果地址包含城市和国家的名称，将其更改为仅包含城市和国家）。

      数据集中只有以下城市和国家：

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

      现在你可以查询国家级别的数据：

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | 酒店地址            | 酒店名称 |
      | :------------------ | :------: |
      | 阿姆斯特丹，荷兰     |   105    |
      | 巴塞罗那，西班牙     |   211    |
      | 伦敦，英国           |   400    |
      | 米兰，意大利         |   162    |
      | 巴黎，法国           |   458    |
      | 维也纳，奥地利       |   158    |

2. 处理酒店元评论列

  1. 删除`Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score`，用我们自己计算的分数替代

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. 处理评论列

   1. 删除`Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`，你可以开始问是否有可能大大减少你必须做的处理。幸运的是，这是可能的，但首先你需要遵循几个步骤来确定感兴趣的标签。

### 过滤标签

记住数据集的目标是添加情感和列，以帮助你选择最佳酒店（为自己或客户要求你制作一个酒店推荐机器人）。你需要问自己这些标签在最终数据集中是否有用。这里有一个解释（如果你出于其他原因需要数据集，不同的标签可能会被保留/排除在选择之外）：

1. 旅行类型是相关的，应该保留
2. 客人群体类型是重要的，应该保留
3. 客人入住的房间、套房或工作室类型是无关紧要的（所有酒店基本上都有相同的房间）
4. 提交评论的设备是无关紧要的
5. 评论者入住的夜晚数量*可能*是相关的，如果你将较长的入住时间与他们更喜欢酒店联系起来，但这有点牵强，可能是无关紧要的

总之，**保留两种标签，删除其他的**。

首先，你不想计算标签，直到它们处于更好的格式，这意味着删除方括号和引号。你可以通过多种方式来做这件事，但你想要最快的方法，因为处理大量数据可能需要很长时间。幸运的是，pandas有一种简单的方法来完成每个步骤。

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

每个标签变成类似这样的：`Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag`列匹配其中一个新列，添加1，否则添加0。最终结果将是一个计数，显示有多少评论者选择了这家酒店（总体上）用于，例如，商务旅行还是休闲旅行，或者是否带宠物，这在推荐酒店时是有用的信息。

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

最后，以新名称保存现在的数据集。

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## 情感分析操作

在这一最后部分，你将对评论列应用情感分析，并将结果保存在数据集中。

## 练习：加载和保存过滤后的数据

请注意，现在你加载的是在上一部分保存的过滤后的数据集，而不是原始数据集。

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

### 移除停用词

如果你在负面和正面评论列上运行情感分析，可能需要很长时间。测试在一台强大的测试笔记本电脑上，使用快速CPU，耗时12-14分钟，具体取决于使用的情感库。这是一个（相对）较长的时间，所以值得调查是否可以加快速度。

移除停用词，即不改变句子情感的常见英语词汇，是第一步。通过移除它们，情感分析应该运行得更快，但不会降低准确性（因为停用词不会影响情感，但会减慢分析速度）。

最长的负面评论是395个词，但在移除停用词后是195个词。

移除停用词也是一个快速操作，在测试设备上从515,000行的两个评论列中移除停用词耗时3.3秒。根据你的设备CPU速度、RAM、是否有SSD等因素，这可能会稍微多一点或少一点时间。操作的相对短暂性意味着如果它能改善情感分析时间，那么这是值得做的。

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

现在你应该计算负面和正面评论列的情感分析，并将结果存储在两个新列中。情感测试将与同一评论的评论者评分进行比较。例如，如果情感认为负面评论的情感是1（极其正面的情感）而正面评论的情感也是1，但评论者给酒店的评分是最低的，那么要么评论文本与评分不匹配，要么情感分析器无法正确识别情感。你应该预期一些情感评分是完全错误的，通常这可以解释，例如评论可能是极其讽刺的“当然，我喜欢在没有暖气的房间里睡觉”，而情感分析器认为这是正面的情感，即使人类阅读它会知道这是讽刺。

NLTK提供了不同的情感分析器供学习，你可以替换它们，看看情感是否更准确。这里使用的是VADER情感分析。

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

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

在你的程序中，当你准备好计算情感时，可以将其应用于每个评论，如下所示：

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

这在我的电脑上大约需要120秒，但在每台电脑上都会有所不同。如果你想打印结果并查看情感是否与评论匹配：

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

在使用文件之前要做的最后一件事是保存它！你还应该考虑重新排序所有新列，以便于使用（对人类来说，这是一种外观上的变化）。

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

你应该运行整个[分析笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)的代码（在你运行[过滤笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)以生成Hotel_Reviews_Filtered.csv文件之后）。

回顾一下，步骤是：

1. 原始数据集文件**Hotel_Reviews.csv**在上一课中通过[探索笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)进行了探索
2. 通过[过滤笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)过滤Hotel_Reviews.csv，生成**Hotel_Reviews_Filtered.csv**
3. 通过[情感分析笔记本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)处理Hotel_Reviews_Filtered.csv，生成**Hotel_Reviews_NLP.csv**
4. 在下面的NLP挑战中使用Hotel_Reviews_NLP.csv

### 结论

当你开始时，你有一个包含列和数据的数据集，但并非所有数据都可以验证或使用。你已经探索了数据，过滤了不需要的内容，将标签转换为有用的东西，计算了自己的平均值，添加了一些情感列，并希望学习了一些关于处理自然文本的有趣知识。

## [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## 挑战

现在你已经对数据集进行了情感分析，看看你是否可以使用本课程中学到的策略（例如聚类）来确定情感模式。

## 复习与自学

参加[这个学习模块](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)以了解更多并使用不同的工具探索文本中的情感。
## 作业 

[尝试不同的数据集](assignment.md)

**免责声明**：
本文档是使用基于机器的人工智能翻译服务翻译的。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应将原文档的母语版本视为权威来源。对于关键信息，建议进行专业人工翻译。对于因使用本翻译而引起的任何误解或误读，我们不承担任何责任。