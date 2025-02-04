# 使用酒店评论进行情感分析 - 处理数据

在本节中，你将使用前几节课中学到的技术对一个大型数据集进行一些探索性数据分析。一旦你对各列的实用性有了较好的理解，你将学到：

- 如何删除不必要的列
- 如何基于现有列计算一些新数据
- 如何保存结果数据集以便在最终挑战中使用

## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### 介绍

到目前为止，你已经了解了文本数据与数值数据类型有很大的不同。如果是人类书写或说出的文本，可以通过分析找到模式和频率、情感和意义。本课将带你进入一个真实的数据集，面临一个真实的挑战：**[欧洲515K酒店评论数据](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**，并包含一个[CC0：公共领域许可](https://creativecommons.org/publicdomain/zero/1.0/)。它是从Booking.com的公共资源中抓取的。数据集的创建者是Jiashen Liu。

### 准备

你将需要：

* 能够使用Python 3运行.ipynb笔记本
* pandas
* NLTK，[你应该在本地安装](https://www.nltk.org/install.html)
* 数据集可在Kaggle上找到[欧洲515K酒店评论数据](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)。解压后大约230 MB。下载到与这些NLP课程相关的根`/data`文件夹中。

## 探索性数据分析

这个挑战假设你正在使用情感分析和客人评论评分来构建一个酒店推荐机器人。你将使用的数据集包括6个城市中1493家不同酒店的评论。

使用Python、一个酒店评论数据集和NLTK的情感分析，你可以找出：

* 评论中最常用的词和短语是什么？
* 描述酒店的官方标签与评论评分是否相关（例如，某个酒店的“有小孩的家庭”标签的负面评论是否多于“单人旅行者”的负面评论，这可能表明它更适合单人旅行者？）
* NLTK情感评分是否与酒店评论者的数值评分一致？

#### 数据集

让我们探索你已下载并保存在本地的数据集。用VS Code或Excel等编辑器打开文件。

数据集的标题如下：

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

以下是按便于检查的方式分组的标题：
##### 酒店列

* `Hotel_Name`, `Hotel_Address`, `lat` (纬度), `lng` (经度)
  * 使用*lat*和*lng*你可以用Python绘制一张地图，显示酒店位置（或许用颜色编码表示负面和正面评论）
  * Hotel_Address对我们来说不是显而易见的有用，我们可能会用国家替换它，以便更容易排序和搜索

**酒店元评论列**

* `Average_Score`
  * 根据数据集创建者，这一列是*酒店的平均评分，根据去年最新评论计算*。这种计算评分的方式似乎有些不寻常，但这是抓取的数据，所以我们暂且接受。

  ✅ 基于数据中的其他列，你能想到另一种计算平均评分的方法吗？

* `Total_Number_of_Reviews`
  * 该酒店收到的总评论数 - 不清楚（不写代码的话）这是否指的是数据集中的评论数。
* `Additional_Number_of_Scoring`
  * 这意味着给出了评论评分，但评论者没有写正面或负面评论

**评论列**

- `Reviewer_Score`
  - 这是一个最多有一位小数的数值，最小值和最大值在2.5到10之间
  - 没有解释为什么2.5是最低可能的评分
- `Negative_Review`
  - 如果评论者什么都没写，这一栏将显示“**No Negative**”
  - 注意评论者可能会在负面评论栏写正面评论（例如，“这家酒店没有什么不好的地方”）
- `Review_Total_Negative_Word_Counts`
  - 更高的负面词汇数量表示较低的评分（不检查情感性）
- `Positive_Review`
  - 如果评论者什么都没写，这一栏将显示“**No Positive**”
  - 注意评论者可能会在正面评论栏写负面评论（例如，“这家酒店根本没有什么好的地方”）
- `Review_Total_Positive_Word_Counts`
  - 更高的正面词汇数量表示较高的评分（不检查情感性）
- `Review_Date`和`days_since_review`
  - 可以对评论应用新鲜度或陈旧度的衡量（较旧的评论可能不如较新的评论准确，因为酒店管理变了，或进行了装修，或添加了游泳池等）
- `Tags`
  - 这些是评论者可能选择用来描述他们是何种客人的简短描述（例如，单人或家庭），他们住的房间类型，停留时间以及评论提交的方式。
  - 不幸的是，使用这些标签是有问题的，请查看下面讨论它们实用性的部分

**评论者列**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 这可能是推荐模型中的一个因素，例如，如果你能确定更多的多产评论者（有数百条评论）更可能给出负面而不是正面的评论。然而，任何特定评论的评论者没有用唯一代码标识，因此无法链接到一组评论。有30位评论者有100条或更多评论，但很难看出这如何有助于推荐模型。
- `Reviewer_Nationality`
  - 有些人可能认为某些国籍的人更可能给出正面或负面评论，因为有国家倾向。小心将这种轶事观点构建到你的模型中。这些是国家（有时是种族）刻板印象，每个评论者都是基于他们的经验写评论的个体。可能通过多种镜头过滤，如他们以前的酒店住宿、旅行距离和个人性格。认为他们的国籍是评论评分的原因是难以证明的。

##### 示例

| 平均评分 | 总评论数 | 评论者评分 | 负面评论                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 正面评论                 | 标签                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | 这目前不是一家酒店，而是一个施工现场。我从早上到全天都被不可接受的建筑噪音吓坏了，无法在长途旅行后休息和在房间工作。人们整天都在用凿岩机在相邻房间工作。我要求换房，但没有安静的房间。更糟的是，我被多收了费用。我在晚上退房，因为我有很早的航班，并收到了一张合适的账单。一天后，酒店未经我同意再次收取超出预订价格的费用。这是一个可怕的地方，不要通过预订来惩罚自己。 | 没有任何好地方，远离这里 | 商务旅行，夫妻标准双人房，住了2晚 |

如你所见，这位客人在这家酒店住得并不愉快。这家酒店有7.8的良好平均评分和1945条评论，但这位评论者给了2.5分，并写了115个字描述他们的负面经历。如果他们在Positive_Review栏中什么都没写，你可能会推测没有什么正面的，但他们写了7个警告词。如果我们只是数词而不是词的意义或情感，我们可能会对评论者的意图有偏差的看法。奇怪的是，他们的2.5分令人困惑，因为如果酒店住宿如此糟糕，为什么还给它任何分数？仔细调查数据集，你会发现最低可能的分数是2.5，而不是0。最高可能的分数是10。

##### 标签

如上所述，乍一看，使用`Tags`来分类数据的想法是有道理的。不幸的是，这些标签没有标准化，这意味着在给定的酒店中，选项可能是*单人房*、*双人房*和*双人房*，但在下一家酒店中，它们是*豪华单人房*、*经典大床房*和*行政大床房*。这些可能是相同的东西，但有太多的变体，选择变得：

1. 尝试将所有术语更改为单一标准，这非常困难，因为不清楚每种情况下的转换路径是什么（例如，*经典单人房*映射到*单人房*，但*带庭院花园或城市景观的高级大床房*更难映射）

1. 我们可以采取NLP方法，测量某些术语如*单人旅行者*、*商务旅行者*或*带小孩的家庭*在每家酒店中的频率，并将其纳入推荐

标签通常（但不总是）是一个包含5到6个逗号分隔值的字段，对应于*旅行类型*、*客人类型*、*房间类型*、*夜晚数*和*提交评论的设备类型*。然而，因为有些评论者没有填写每个字段（他们可能留一个空白），值并不总是按相同顺序排列。

例如，取*组类型*。在`Tags`列中有1025种独特的可能性，不幸的是，只有一部分指的是组（有些是房间类型等）。如果你只过滤提到家庭的那些，结果包含许多*家庭房*类型的结果。如果你包括*带*这个词，即计算*带*值的*家庭*，结果更好，在515,000个结果中有超过80,000个包含“带小孩的家庭”或“带大孩子的家庭”短语。

这意味着标签列对我们来说并非完全无用，但需要一些工作使其有用。

##### 酒店平均评分

数据集中有一些奇怪之处或差异，我无法弄清楚，但在此说明以便你在构建模型时注意到它们。如果你弄清楚了，请在讨论区告诉我们！

数据集有以下列与平均评分和评论数量相关：

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

数据集中评论最多的单个酒店是*Britannia International Hotel Canary Wharf*，有4789条评论中的515,000条。但如果我们看`Total_Number_of_Reviews`值，这家酒店是9086。你可能推测有更多没有评论的评分，所以也许我们应该加上`Additional_Number_of_Scoring`列值。该值是2682，加上4789得到7471，仍然比`Total_Number_of_Reviews`少1615。

如果你取`Average_Score`列，你可能推测它是数据集中评论的平均值，但Kaggle的描述是“*根据去年最新评论计算的酒店平均评分*”。这似乎没有那么有用，但我们可以根据数据集中的评论评分计算自己的平均值。以同一家酒店为例，给出的酒店平均评分是7.1，但计算得出的评分（数据集中评论者的平均评分）是6.8。这接近但不是相同的值，我们只能猜测`Additional_Number_of_Scoring`评论中的评分将平均值提高到7.1。不幸的是，没有办法测试或证明这一断言，因此很难使用或信任`Average_Score`、`Additional_Number_of_Scoring`和`Total_Number_of_Reviews`，因为它们基于或引用了我们没有的数据。

更复杂的是，评论第二多的酒店的计算平均评分为8.12，而数据集`Average_Score`是8.1。这是正确的评分巧合还是第一家酒店的差异？

考虑到这些酒店可能是异常值，也许大多数值匹配（但某些原因导致部分不匹配），我们将在下一步编写一个短程序来探索数据集中的值，并确定这些值的正确使用（或不使用）。

> 🚨 注意
>
> 使用此数据集时，你将编写代码从文本中计算某些内容，而不必自己阅读或分析文本。这是NLP的本质，解释意义或情感而不需要人类来做。然而，你可能会阅读一些负面评论。我建议你不要这样做，因为你不需要这样做。有些是愚蠢的或无关紧要的负面酒店评论，如“天气不好”，这是酒店或任何人无法控制的。但也有一些负面评论是种族主义、性别歧视或年龄歧视的。这是从公共网站抓取的数据集中不可避免的。某些评论者留下的评论会让你觉得令人反感、不舒服或不安。最好让代码测量情感，而不是自己阅读它们并感到不安。尽管如此，写这种东西的人是少数，但它们依然存在。

## 练习 - 数据探索
### 加载数据

视觉检查数据已经够多了，现在你将编写一些代码并得到一些答案！本节使用pandas库。你的第一个任务是确保你能加载和读取CSV数据。pandas库有一个快速的CSV加载器，结果放在一个数据框中，如前几课所示。我们加载的CSV有超过50万行，但只有17列。pandas为你提供了许多强大的方法与数据框交互，包括对每一行执行操作的能力。

从这里开始，将有代码片段和一些代码解释以及一些关于结果含义的讨论。使用包含的_notebook.ipynb_进行你的代码编写。

让我们从加载你将使用的数据文件开始：

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

现在数据已加载，我们可以对其进行一些操作。将此代码保留在程序顶部以便下一部分使用。

## 探索数据

在这种情况下，数据已经是*干净的*，这意味着它已经准备好使用，并且没有可能使只期望英文字符的算法出错的其他语言字符。

✅ 你可能需要处理需要一些初步处理才能格式化的数据，然后应用NLP技术，但这次不需要。如果需要，你将如何处理非英文字符？

花点时间确保数据加载后，你可以用代码探索它。非常容易想要专注于`Negative_Review`和`Positive_Review`列。它们充满了你的NLP算法要处理的自然文本。但等等！在你跳入NLP和情感分析之前，你应该按照下面的代码确定数据集中给定的值是否与你用pandas计算的值匹配。

## 数据框操作

本课的第一个任务是编写一些代码检查以下断言是否正确（不更改数据框）。

> 像许多编程任务一样，有几种方法可以完成它，但好的建议是以最简单、最容易理解的方式完成，特别是当你将来回到这段代码时更容易理解。对于数据框，有一个全面的API，通常有一种方法可以高效地完成你想要的操作。
将以下问题视为编码任务，尝试在不查看解决方案的情况下回答它们。
1. 打印出你刚加载的数据框的*形状*（形状是行数和列数）
2. 计算评论者国籍的频率计数：
  1. 列`Reviewer_Nationality`有多少不同的值，它们是什么？
  2. 数据集中最常见的评论者国籍是什么（打印国家和评论数）？
  3. 下一个最常见的10个国籍及其频率计数是什么？
3. 每个最常见的10个评论者国籍中评论最多的酒店是什么？
4. 数据集中每家酒店有多少评论（酒店的频率计数）？
5. 虽然数据集中每家酒店都有`Average_Score`列，但你也可以计算一个平均评分（获取数据集中每家酒店的所有评论者评分的平均值）。在你的数据框中添加一个新列，标题为`Calc_Average_Score`，包含计算的平均值。
6. 是否有任何酒店的`Average_Score`和`Calc_Average_Score`（四舍五入到一位小数）相同？使用`?
   1. Try writing a Python function that takes a Series (row) as an argument and compares the values, printing out a message when the values are not equal. Then use the `.apply()方法处理每一行。
7. 计算并打印出`Negative_Review`列值为“No Negative”的行数
8. 计算并打印出`Positive_Review`列值为“No Positive”的行数
rows have column `Positive_Review` values of "No Positive" 9. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ### Code answers 1. Print out the *shape* of the data frame you have just loaded (the shape is the number of rows and columns) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Calculate the frequency count for reviewer nationalities: 1. How many distinct values are there for the column `Reviewer_Nationality` and what are they? 2. What reviewer nationality is the most common in the dataset (print country and number of reviews)? ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ``` 3. What are the next top 10 most frequently found nationalities, and their frequency count? ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ``` 3. What was the most frequently reviewed hotel for each of the top 10 most reviewer nationalities? ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ``` 4. How many reviews are there per hotel (frequency count of hotel) in the dataset? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | You may notice that the *counted in the dataset* results do not match the value in `Total_Number_of_Reviews`. It is unclear if this value in the dataset represented the total number of reviews the hotel had, but not all were scraped, or some other calculation. `Total_Number_of_Reviews` is not used in the model because of this unclarity. 5. While there is an `Average_Score` column for each hotel in the dataset, you can also calculate an average score (getting the average of all reviewer scores in the dataset for each hotel). Add a new column to your dataframe with the column header `Calc_Average_Score` that contains that calculated average. Print out the columns `Hotel_Name`, `Average_Score`, and `Calc_Average_Score`. ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ``` You may also wonder about the `Average_Score` value and why it is sometimes different from the calculated average score. As we can't know why some of the values match, but others have a difference, it's safest in this case to use the review scores that we have to calculate the average ourselves. That said, the differences are usually very small, here are the hotels with the greatest deviation from the dataset average and the calculated average: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | With only 1 hotel having a difference of score greater than 1, it means we can probably ignore the difference and use the calculated average score. 6. Calculate and print out how many rows have column `Negative_Review` values of "No Negative" 7. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" 8. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ``` ## Another way Another way count items without Lambdas, and use sum to count the rows: ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ``` You may have noticed that there are 127 rows that have both "No Negative" and "No Positive" values for the columns `Negative_Review` and `Positive_Review` respectively. That means that the reviewer gave the hotel a numerical score, but declined to write either a positive or negative review. Luckily this is a small amount of rows (127 out of 515738, or 0.02%), so it probably won't skew our model or results in any particular direction, but you might not have expected a data set of reviews to have rows with no reviews, so it's worth exploring the data to discover rows like this. Now that you have explored the dataset, in the next lesson you will filter the data and add some sentiment analysis. --- ## 🚀Challenge This lesson demonstrates, as we saw in previous lessons, how critically important it is to understand your data and its foibles before performing operations on it. Text-based data, in particular, bears careful scrutiny. Dig through various text-heavy datasets and see if you can discover areas that could introduce bias or skewed sentiment into a model. ## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Review & Self Study Take [this Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) to discover tools to try when building speech and text-heavy models. ## Assignment [NLTK](assignment.md)

**免责声明**：
本文件使用基于机器的人工智能翻译服务进行翻译。虽然我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应将原始语言的文件视为权威来源。对于关键信息，建议使用专业的人类翻译。对于因使用本翻译而引起的任何误解或误读，我们概不负责。