<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3c4738bb0836dd838c552ab9cab7e09d",
  "translation_date": "2025-09-03T18:54:16+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "zh"
}
-->
# 使用酒店评论进行情感分析 - 数据处理

在本节中，您将使用前几课中学到的技术对一个大型数据集进行一些探索性数据分析。当您对各列的实用性有了较好的理解后，您将学习：

- 如何删除不必要的列
- 如何基于现有列计算一些新的数据
- 如何保存处理后的数据集以用于最终的挑战

## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### 介绍

到目前为止，您已经了解了文本数据与数值数据类型有很大的不同。如果是人类书写或口述的文本数据，可以通过分析发现模式、频率、情感和含义。本课将带您进入一个真实的数据集并面对一个实际挑战：**[欧洲515K酒店评论数据](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**，该数据集包含[CC0: 公共领域许可](https://creativecommons.org/publicdomain/zero/1.0/)。数据来自Booking.com的公开来源，由Jiashen Liu创建。

### 准备工作

您需要：

* 能够使用Python 3运行.ipynb笔记本
* pandas库
* NLTK库，[您需要在本地安装](https://www.nltk.org/install.html)
* 数据集可从Kaggle下载：[欧洲515K酒店评论数据](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)。解压后约230 MB。将其下载到与这些NLP课程相关的根目录`/data`文件夹中。

## 探索性数据分析

本次挑战假设您正在使用情感分析和客人评论评分构建一个酒店推荐机器人。您将使用的数据集包含6个城市中1493家不同酒店的评论。

通过使用Python、酒店评论数据集和NLTK的情感分析，您可以发现：

* 评论中最常用的单词和短语是什么？
* 描述酒店的官方*标签*是否与评论评分相关（例如，某个酒店的*家庭带小孩*标签是否比*独自旅行者*标签更容易收到负面评论，这可能表明该酒店更适合*独自旅行者*）？
* NLTK的情感评分是否与酒店评论者的数值评分一致？

#### 数据集

让我们来探索您已下载并保存到本地的数据集。可以在VS Code或Excel等编辑器中打开文件。

数据集的标题如下：

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

以下是按类别分组的方式，可能更便于查看：
##### 酒店相关列

* `Hotel_Name`, `Hotel_Address`, `lat`（纬度）, `lng`（经度）
  * 使用*lat*和*lng*，您可以使用Python绘制一张地图，显示酒店位置（可能用颜色区分正面和负面评论）
  * `Hotel_Address`对我们来说似乎没有明显的用处，我们可能会用国家名称替代它以便更容易排序和搜索

**酒店元评论列**

* `Average_Score`
  * 根据数据集创建者的描述，这一列是*酒店的平均评分，基于过去一年内的最新评论计算*。这种计算方式似乎有些不寻常，但由于数据是抓取的，我们暂时按字面意思接受。
  
  ✅ 根据数据中的其他列，您能想到另一种计算平均评分的方法吗？

* `Total_Number_of_Reviews`
  * 该酒店收到的评论总数——但尚不清楚（除非编写代码验证）这是否指数据集中包含的评论。
* `Additional_Number_of_Scoring`
  * 表示评论者给出了评分，但没有写正面或负面的评论。

**评论相关列**

- `Reviewer_Score`
  - 这是一个数值，最多有1位小数，范围在2.5到10之间
  - 不清楚为什么最低分是2.5而不是0
- `Negative_Review`
  - 如果评论者没有写任何内容，这一字段会显示“**No Negative**”
  - 请注意，评论者可能会在负面评论列中写正面评论（例如，“这家酒店没有任何不好的地方”）
- `Review_Total_Negative_Word_Counts`
  - 较高的负面词汇计数表明评分较低（不考虑情感分析）
- `Positive_Review`
  - 如果评论者没有写任何内容，这一字段会显示“**No Positive**”
  - 请注意，评论者可能会在正面评论列中写负面评论（例如，“这家酒店完全没有任何优点”）
- `Review_Total_Positive_Word_Counts`
  - 较高的正面词汇计数表明评分较高（不考虑情感分析）
- `Review_Date` 和 `days_since_review`
  - 可以对评论的新鲜度或陈旧度进行衡量（例如，较旧的评论可能不如较新的评论准确，因为酒店管理可能发生了变化，或者进行了装修，或者新增了游泳池等）
- `Tags`
  - 这些是评论者可能选择的简短描述，用于描述他们的客人类型（例如独自旅行或家庭）、房间类型、停留时间以及评论提交方式。
  - 不幸的是，使用这些标签存在问题，请查看下面关于其实用性的讨论部分。

**评论者相关列**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 这可能是推荐模型中的一个因素，例如，如果您能够确定评论数量较多的评论者（有数百条评论）更倾向于给出负面评论而不是正面评论。然而，任何特定评论的评论者并没有唯一的标识符，因此无法将其与一组评论关联起来。有30位评论者的评论数量达到或超过100条，但很难看出这如何有助于推荐模型。
- `Reviewer_Nationality`
  - 有些人可能认为某些国籍的评论者更倾向于给出正面或负面的评论，这可能是由于某种国家倾向。然而，在模型中构建这样的观点需要谨慎。这些是国家（有时是种族）刻板印象，每位评论者都是基于自己的体验写下评论的个体。评论可能受到许多因素的影响，例如他们之前的酒店住宿经历、旅行距离以及个人性格。仅仅因为评论者的国籍而推断评论分数的原因是很难站得住脚的。

##### 示例

| 平均评分 | 评论总数 | 评论者评分 | 负面评论                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 正面评论                 | 标签                                                                                      |
| -------- | -------- | ---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------- |
| 7.8      | 1945     | 2.5        | 这目前不是一家酒店，而是一个施工现场。我在长途旅行后休息时，从早到晚都被无法接受的施工噪音折磨。人们整天在隔壁房间用凿岩机工作。我要求换房，但没有安静的房间可用。更糟糕的是，我被多收了费用。我在晚上退房，因为我需要赶早班飞机，并收到了一张合适的账单。一天后，酒店未经我同意又多收了一笔费用，超出了预订价格。这是一个可怕的地方。不要惩罚自己，千万别预订这里。 | 没有任何优点。可怕的地方，远离它。 | 商务旅行，情侣，标准双人房，住了2晚 |

如您所见，这位客人在这家酒店的住宿体验非常糟糕。尽管酒店的平均评分为7.8，总共有1945条评论，但这位评论者给出了2.5分，并写了115个字描述他们的负面经历。如果他们在`Positive_Review`列中什么都没写，您可能会推测没有任何优点，但实际上他们写了7个字的警告。如果我们仅仅统计单词数量，而不是单词的含义或情感，我们可能会对评论者的意图产生偏差。奇怪的是，他们的评分2.5令人困惑，因为如果这家酒店的住宿体验如此糟糕，为什么还要给任何分数？仔细研究数据集，您会发现最低可能的评分是2.5，而不是0。最高可能的评分是10。

##### 标签

如上所述，乍一看，使用`Tags`列对数据进行分类似乎是个好主意。不幸的是，这些标签并不标准化，这意味着在某家酒店中，选项可能是*单人房*、*双床房*和*双人房*，但在另一家酒店中，它们可能是*豪华单人房*、*经典大床房*和*行政特大床房*。这些可能是相同的房型，但由于存在太多变体，选择变得复杂：

1. 尝试将所有术语更改为单一标准，这非常困难，因为在每种情况下转换路径并不明确（例如，*经典单人房*映射到*单人房*，但*带庭院花园或城市景观的高级大床房*则更难映射）

2. 我们可以采用NLP方法，测量某些术语（如*独自旅行*、*商务旅行者*或*带小孩的家庭*）在每家酒店中的频率，并将其纳入推荐模型中

标签通常（但并非总是）是一个包含5到6个逗号分隔值的字段，对应于*旅行类型*、*客人类型*、*房间类型*、*住宿天数*以及*评论提交设备类型*。然而，由于某些评论者没有填写每个字段（可能留空一个字段），这些值并不总是按相同顺序排列。

例如，考虑*群体类型*。在`Tags`列中，这一字段有1025种唯一可能性，不幸的是，只有部分值提到群体（有些是房间类型等）。如果您仅过滤提到家庭的值，结果包含许多*家庭房*类型的结果。如果您包括术语*with*，即统计*Family with*的值，结果会更好，在515,000条结果中，有超过80,000条包含“Family with young children”或“Family with older children”短语。

这意味着`Tags`列对我们来说并非完全无用，但需要一些工作才能使其变得有用。

##### 酒店平均评分

数据集中有一些奇怪或不一致的地方，我无法完全弄清楚，但在这里列出以便您在构建模型时注意。如果您能弄明白，请在讨论区告诉我们！

数据集包含以下与平均评分和评论数量相关的列：

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

数据集中评论数量最多的酒店是*Britannia International Hotel Canary Wharf*，共有4789条评论。然而，如果我们查看该酒店的`Total_Number_of_Reviews`值，它是9086。您可能会推测有更多评分没有评论，因此我们可以将`Additional_Number_of_Scoring`列的值加进去。该值为2682，将其与4789相加得到7471，但仍比`Total_Number_of_Reviews`少1615。

如果您查看`Average_Score`列，您可能会推测它是数据集中评论的平均值，但Kaggle的描述是“*酒店的平均评分，基于过去一年内的最新评论计算*”。这似乎不太有用，但我们可以根据数据集中的评论评分计算自己的平均值。以同一家酒店为例，给出的平均评分是7.1，但根据数据集中的评论者评分计算的平均值是6.8。这很接近，但不是相同的值，我们只能猜测`Additional_Number_of_Scoring`列中的评分将平均值提高到了7.1。不幸的是，由于无法验证或证明这一推测，使用或信任`Average_Score`、`Additional_Number_of_Scoring`和`Total_Number_of_Reviews`列变得困难，因为它们基于或引用了我们没有的数据。

更复杂的是，评论数量第二多的酒店的计算平均评分是8.12，而数据集中的`Average_Score`是8.1。这是否是正确的评分巧合，还是第一家酒店的数据存在问题？
关于这些酒店可能是异常值的可能性，以及大多数值可能一致（但由于某些原因有些不一致），我们将编写一个简短的程序来探索数据集中的值，并确定这些值的正确使用（或不使用）。

> 🚨 注意事项
>
> 在处理这个数据集时，你将编写代码从文本中计算某些内容，而无需自己阅读或分析文本。这是自然语言处理（NLP）的核心：无需人工参与即可解释意义或情感。然而，你可能会读到一些负面评论。我建议你不要这样做，因为没有必要。有些评论很荒谬或与酒店无关，例如“天气不好”，这是酒店或任何人都无法控制的事情。但有些评论也有阴暗的一面。有时负面评论带有种族歧视、性别歧视或年龄歧视。这种情况虽然令人遗憾，但在从公共网站抓取的数据集中是可以预料的。一些评论者会留下让人觉得不愉快、不舒服或令人心烦的评论。最好让代码来衡量情感，而不是自己阅读这些评论并感到不安。话虽如此，这种情况只占少数，但它确实存在。

## 练习 - 数据探索
### 加载数据

视觉检查数据已经够了，现在你将编写一些代码来获取答案！本节使用了 pandas 库。你的第一个任务是确保你能够加载并读取 CSV 数据。pandas 库有一个快速的 CSV 加载器，结果会像之前的课程一样存储在一个数据框中。我们加载的 CSV 文件有超过五十万行，但只有 17 列。pandas 提供了许多强大的方法来与数据框交互，包括对每一行执行操作的能力。

从现在开始，这节课将包含代码片段、代码解释以及结果讨论。请使用提供的 _notebook.ipynb_ 来编写代码。

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

数据加载完成后，我们可以对其进行一些操作。将这段代码保留在程序顶部，以便进行下一部分。

## 数据探索

在这个案例中，数据已经是*干净的*，这意味着它已经可以直接使用，并且没有其他语言的字符，这些字符可能会让只接受英文字符的算法出错。

✅ 你可能需要处理需要初步格式化的数据，以便应用 NLP 技术，但这次不需要。如果需要，你会如何处理非英文字符？

确保数据加载后，你可以通过代码进行探索。很容易想要专注于 `Negative_Review` 和 `Positive_Review` 列。这些列中填充了自然文本，供你的 NLP 算法处理。但等等！在跳入 NLP 和情感分析之前，你应该按照下面的代码来确定数据集中给出的值是否与通过 pandas 计算的值一致。

## 数据框操作

本课的第一个任务是通过编写代码检查以下断言是否正确（无需更改数据框）。

> 像许多编程任务一样，有多种方法可以完成，但好的建议是尽可能简单、容易理解，尤其是当你以后回顾代码时。对于数据框，pandas 提供了一个全面的 API，通常可以高效地完成你想要的操作。

将以下问题视为编码任务，尝试在不查看解决方案的情况下回答它们。

1. 打印出刚加载的数据框的*形状*（形状是行数和列数）
2. 计算评论者国籍的频率计数：
   1. `Reviewer_Nationality` 列中有多少个不同的值？它们是什么？
   2. 数据集中最常见的评论者国籍是什么（打印国家和评论数量）？
   3. 接下来最常见的 10 个国籍及其频率计数是什么？
3. 对于评论者国籍排名前 10 的国家，每个国家评论最多的酒店是什么？
4. 数据集中每个酒店有多少条评论（酒店的频率计数）？
5. 数据集中每个酒店都有一个 `Average_Score` 列，你也可以计算平均分（获取数据集中每个酒店所有评论者评分的平均值）。向数据框添加一个新列，列标题为 `Calc_Average_Score`，其中包含计算出的平均值。
6. 是否有酒店的 `Average_Score` 和 `Calc_Average_Score`（四舍五入到小数点后一位）相同？
   1. 尝试编写一个 Python 函数，该函数接受一个 Series（行）作为参数并比较值，当值不相等时打印消息。然后使用 `.apply()` 方法处理每一行。
7. 计算并打印 `Negative_Review` 列值为 "No Negative" 的行数
8. 计算并打印 `Positive_Review` 列值为 "No Positive" 的行数
9. 计算并打印 `Positive_Review` 列值为 "No Positive" **且** `Negative_Review` 列值为 "No Negative" 的行数

### 代码答案

1. 打印出刚加载的数据框的*形状*（形状是行数和列数）

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. 计算评论者国籍的频率计数：

   1. `Reviewer_Nationality` 列中有多少个不同的值？它们是什么？
   2. 数据集中最常见的评论者国籍是什么（打印国家和评论数量）？

   ```python
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
   ```

   3. 接下来最常见的 10 个国籍及其频率计数是什么？

      ```python
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
      ```

3. 对于评论者国籍排名前 10 的国家，每个国家评论最多的酒店是什么？

   ```python
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
   ```

4. 数据集中每个酒店有多少条评论（酒店的频率计数）？

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   你可能会注意到，*数据集中统计的*结果与 `Total_Number_of_Reviews` 中的值不匹配。目前尚不清楚数据集中该值是否表示酒店的总评论数，但并非所有评论都被抓取，或者是其他计算。由于这种不明确性，`Total_Number_of_Reviews` 未被用于模型。

5. 数据集中每个酒店都有一个 `Average_Score` 列，你也可以计算平均分（获取数据集中每个酒店所有评论者评分的平均值）。向数据框添加一个新列，列标题为 `Calc_Average_Score`，其中包含计算出的平均值。打印出列 `Hotel_Name`、`Average_Score` 和 `Calc_Average_Score`。

   ```python
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
   ```

   你可能也会对 `Average_Score` 值感到疑惑，为什么它有时与计算出的平均分不同。由于我们无法知道为什么有些值匹配，而其他值存在差异，在这种情况下，最安全的做法是使用我们拥有的评论分数自己计算平均值。话虽如此，差异通常非常小，以下是数据集中平均分与计算出的平均分差异最大的酒店：

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   只有 1 家酒店的分数差异超过 1，这意味着我们可能可以忽略差异并使用计算出的平均分。

6. 计算并打印 `Negative_Review` 列值为 "No Negative" 的行数 

7. 计算并打印 `Positive_Review` 列值为 "No Positive" 的行数

8. 计算并打印 `Positive_Review` 列值为 "No Positive" **且** `Negative_Review` 列值为 "No Negative" 的行数

   ```python
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
   ```

## 另一种方法

另一种方法是不用 Lambdas，而是使用 sum 来统计行数：

   ```python
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
   ```

   你可能注意到有 127 行同时在 `Negative_Review` 和 `Positive_Review` 列中分别具有 "No Negative" 和 "No Positive" 的值。这意味着评论者给酒店打了一个数字评分，但没有写任何正面或负面评论。幸运的是，这只是少量行（127 行中的 515738 行，占 0.02%），所以它可能不会对我们的模型或结果产生任何特定方向的偏差，但你可能没有预料到一个评论数据集会有没有评论的行，因此值得探索数据以发现这样的行。

现在你已经探索了数据集，在下一课中你将过滤数据并添加一些情感分析。

---
## 🚀挑战

本课展示了（正如我们在之前的课程中看到的）在对数据进行操作之前，了解数据及其缺陷是多么重要。特别是基于文本的数据需要仔细审查。挖掘各种文本密集型数据集，看看你是否能发现可能引入偏差或情感倾斜的领域。

## [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## 复习与自学

参加 [这个 NLP 学习路径](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott)，了解在构建语音和文本密集型模型时可以尝试的工具。

## 作业 

[NLTK](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。对于因使用本翻译而引起的任何误解或误读，我们概不负责。