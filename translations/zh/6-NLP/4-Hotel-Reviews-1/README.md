<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T09:10:44+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "zh"
}
-->
# 使用酒店评论进行情感分析 - 数据处理

在本节中，您将使用前几课中的技术对一个大型数据集进行一些探索性数据分析。一旦您对各列的实用性有了良好的理解，您将学习：

- 如何删除不必要的列
- 如何基于现有列计算一些新数据
- 如何保存处理后的数据集以用于最终挑战

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

### 简介

到目前为止，您已经了解了文本数据与数值数据类型的不同。如果文本是由人类书写或口述的，它可以被分析以发现模式和频率、情感和意义。本课将带您进入一个真实的数据集并面对一个真实的挑战：**[欧洲515K酒店评论数据](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**，并包含一个[CC0: 公共领域许可](https://creativecommons.org/publicdomain/zero/1.0/)。该数据集是从Booking.com的公共来源抓取的，数据集的创建者是Jiashen Liu。

### 准备工作

您需要：

* 能够使用Python 3运行.ipynb笔记本
* pandas
* NLTK，[您需要在本地安装](https://www.nltk.org/install.html)
* 数据集可从Kaggle下载：[欧洲515K酒店评论数据](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)。解压后约230 MB。将其下载到与这些NLP课程相关的根目录`/data`文件夹中。

## 探索性数据分析

本次挑战假设您正在构建一个使用情感分析和客人评论评分的酒店推荐机器人。您将使用的数据集包括6个城市中1493家不同酒店的评论。

使用Python、酒店评论数据集和NLTK的情感分析，您可以发现：

* 评论中最常用的词汇和短语是什么？
* 描述酒店的官方*标签*是否与评论评分相关（例如，某个酒店的*家庭带小孩*标签是否比*独行旅客*标签有更多负面评论，这可能表明该酒店更适合*独行旅客*？）
* NLTK的情感评分是否与酒店评论者的数值评分“吻合”？

#### 数据集

让我们探索您已下载并保存到本地的数据集。使用VS Code或Excel等编辑器打开文件。

数据集的标题如下：

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

以下是按类别分组的标题，可能更容易检查：
##### 酒店相关列

* `Hotel_Name`, `Hotel_Address`, `lat`（纬度）, `lng`（经度）
  * 使用*lat*和*lng*，您可以使用Python绘制一张地图，显示酒店位置（或许可以根据正面和负面评论进行颜色编码）
  * Hotel_Address对我们来说似乎没有明显的用处，我们可能会将其替换为国家名称以便更容易排序和搜索

**酒店元评论列**

* `Average_Score`
  * 根据数据集创建者的说法，此列是*酒店的平均评分，基于过去一年内的最新评论计算*。这似乎是一种不寻常的评分计算方式，但由于数据是抓取的，我们暂时接受这一点。
  
  ✅ 根据此数据中的其他列，您能想到另一种计算平均评分的方法吗？

* `Total_Number_of_Reviews`
  * 此酒店收到的评论总数——尚不清楚（需要编写一些代码）这是否指数据集中的评论。
* `Additional_Number_of_Scoring`
  * 表示评论者给出了评分但没有写正面或负面评论

**评论相关列**

- `Reviewer_Score`
  - 这是一个数值，最多有1位小数，范围在2.5到10之间
  - 未解释为何最低评分为2.5
- `Negative_Review`
  - 如果评论者未写任何内容，此字段将显示“**No Negative**”
  - 请注意，评论者可能会在负面评论列中写正面评论（例如，“这家酒店没有任何不好的地方”）
- `Review_Total_Negative_Word_Counts`
  - 较高的负面词汇计数表明评分较低（无需检查情感性）
- `Positive_Review`
  - 如果评论者未写任何内容，此字段将显示“**No Positive**”
  - 请注意，评论者可能会在正面评论列中写负面评论（例如，“这家酒店完全没有任何好的地方”）
- `Review_Total_Positive_Word_Counts`
  - 较高的正面词汇计数表明评分较高（无需检查情感性）
- `Review_Date`和`days_since_review`
  - 可以对评论应用新鲜度或陈旧度的衡量（较旧的评论可能不如较新的评论准确，因为酒店管理可能发生了变化，或者进行了装修，或者新增了泳池等）
- `Tags`
  - 这些是评论者可能选择的简短描述，用于描述他们的客人类型（例如独行或家庭）、房间类型、入住时长以及评论提交方式。
  - 不幸的是，使用这些标签存在问题，请查看下面讨论其实用性的部分

**评论者相关列**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 这可能是推荐模型中的一个因素，例如，如果您可以确定评论数量较多的评论者（有数百条评论）更倾向于给出负面而非正面评论。然而，任何特定评论的评论者并未通过唯一代码标识，因此无法链接到一组评论。有30位评论者有100条或更多评论，但很难看出这如何帮助推荐模型。
- `Reviewer_Nationality`
  - 有些人可能认为某些国籍更倾向于给出正面或负面评论，因为有某种国家倾向。构建这样的轶事观点到模型中时要小心。这些是国家（有时是种族）刻板印象，每位评论者都是根据自己的经历写评论的个体。评论可能受到许多因素的影响，例如他们之前的酒店住宿经历、旅行距离以及个人性格。认为评论评分是由国籍决定的很难证明。

##### 示例

| 平均评分 | 评论总数 | 评论者评分 | 负面评论                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 正面评论                 | 标签                                                                                      |
| -------- | -------- | ---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------- |
| 7.8      | 1945     | 2.5        | 这家酒店目前不是酒店而是一个施工现场，我在长途旅行后休息时被早晨和全天的建筑噪音折磨。人们整天在相邻房间工作，例如使用凿岩机。我要求换房，但没有安静的房间可用。更糟糕的是，我被多收了费用。我在晚上退房，因为我需要赶早班飞机，并收到了一张适当的账单。一天后，酒店未经我同意又收取了超出预订价格的费用。这是一个可怕的地方，不要惩罚自己来这里预订。 | 没有任何好处，糟糕的地方，远离这里 | 商务旅行，情侣，标准双人房，入住2晚 |

如您所见，这位客人在这家酒店的入住体验非常糟糕。酒店的平均评分为7.8，有1945条评论，但这位评论者给出了2.5分，并写了115个词描述他们的负面体验。如果他们在正面评论列中未写任何内容，您可能会推测没有任何正面内容，但他们写了7个词警告其他人。如果我们仅仅统计词汇数量而不是词汇的意义或情感，我们可能会对评论者的意图有一个偏差的看法。奇怪的是，他们的评分为2.5令人困惑，因为如果酒店体验如此糟糕，为什么还给了任何分数？仔细调查数据集，您会发现最低可能评分是2.5，而不是0。最高可能评分是10。

##### 标签

如上所述，乍一看，使用`Tags`来分类数据似乎是合理的。不幸的是，这些标签并未标准化，这意味着在某个酒店中，选项可能是*单人房*、*双床房*和*双人房*，但在另一个酒店中，它们可能是*豪华单人房*、*经典大床房*和*行政特大床房*。这些可能是相同的房型，但有如此多的变体，选择变成了：

1. 尝试将所有术语更改为单一标准，这非常困难，因为不清楚每种情况的转换路径（例如，*经典单人房*映射到*单人房*，但*带庭院花园或城市景观的高级大床房*则更难映射）

1. 我们可以采取NLP方法，测量某些术语的频率，例如*独行*、*商务旅客*或*带小孩的家庭*，并将其应用到每家酒店中，从而将其纳入推荐模型  

标签通常（但并非总是）是一个包含5到6个逗号分隔值的单一字段，对应于*旅行类型*、*客人类型*、*房间类型*、*入住天数*以及*评论提交设备类型*。然而，由于某些评论者未填写每个字段（可能留空一个字段），值并不总是按相同顺序排列。

例如，考虑*群体类型*。在`Tags`列中，此字段有1025种独特可能性，不幸的是，其中只有部分提到群体（有些是房间类型等）。如果您仅过滤提到家庭的标签，结果包含许多*家庭房*类型的结果。如果您包括术语*with*，即统计*家庭带*的值，结果会更好，在515,000条结果中有超过80,000条包含短语“带小孩的家庭”或“带大孩的家庭”。

这意味着标签列对我们来说并非完全无用，但需要一些工作才能使其变得有用。

##### 酒店平均评分

数据集中有一些奇怪或不一致的地方我无法解释，但在此列出以便您在构建模型时注意。如果您能解决，请在讨论区告诉我们！

数据集有以下与平均评分和评论数量相关的列：

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

数据集中评论最多的单一酒店是*Britannia International Hotel Canary Wharf*，有4789条评论（总计515,000条）。但如果我们查看此酒店的`Total_Number_of_Reviews`值，它是9086。您可能会推测有更多评分没有评论，因此我们可能需要加上`Additional_Number_of_Scoring`列的值。该值是2682，加上4789得到7471，仍然比`Total_Number_of_Reviews`少1615。

如果您查看`Average_Score`列，您可能会推测它是数据集中评论的平均值，但Kaggle的描述是“*酒店的平均评分，基于过去一年内的最新评论计算*”。这似乎不太有用，但我们可以根据数据集中的评论评分计算自己的平均值。以同一家酒店为例，给出的平均酒店评分是7.1，但计算得出的评分（数据集中评论者评分的平均值）是6.8。这很接近，但不是相同的值，我们只能猜测`Additional_Number_of_Scoring`评论中的评分将平均值提高到7.1。不幸的是，由于无法测试或证明这一假设，使用或信任`Average_Score`、`Additional_Number_of_Scoring`和`Total_Number_of_Reviews`变得困难，因为它们基于或引用了我们没有的数据。

更复杂的是，评论数量第二多的酒店的计算平均评分是8.12，而数据集中的`Average_Score`是8.1。这是否正确评分是巧合还是第一家酒店存在不一致？

考虑到这些酒店可能是异常值，并且可能大多数值是匹配的（但由于某些原因有些不匹配），我们将在下一步编写一个简短的程序来探索数据集中的值并确定这些值的正确使用（或不使用）。
> 🚨 注意事项
>
> 在处理这个数据集时，你将编写代码从文本中计算某些内容，而无需自己阅读或分析文本。这正是自然语言处理（NLP）的核心：无需人工参与即可解读意义或情感。然而，有可能你会读到一些负面评论。我建议你不要这样做，因为没有必要。有些评论很荒谬，或者是与酒店无关的负面评论，比如“天气不好”，这是酒店甚至任何人都无法控制的事情。但有些评论也有阴暗的一面。有时负面评论可能带有种族歧视、性别歧视或年龄歧视。这种情况令人遗憾，但在从公共网站抓取的数据集中是可以预料的。一些评论者会留下让人觉得反感、不适或不安的评论。最好让代码来衡量情感，而不是自己阅读这些评论后感到不快。话虽如此，这类评论只占少数，但它们确实存在。
## 练习 - 数据探索
### 加载数据

通过视觉检查数据已经足够了，现在你需要编写一些代码来获取答案！本节将使用 pandas 库。你的第一个任务是确保能够加载并读取 CSV 数据。pandas 库提供了一个快速的 CSV 加载器，加载结果会存储在一个 dataframe 中，就像之前的课程一样。我们加载的 CSV 文件有超过 50 万行，但只有 17 列。pandas 提供了许多强大的方法来与 dataframe 交互，包括对每一行执行操作的能力。

从现在开始，这节课将包含代码片段、代码解释以及对结果的讨论。请使用提供的 _notebook.ipynb_ 文件来编写代码。

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

现在数据已经加载，我们可以对其进行一些操作。在接下来的部分中，请将这段代码保留在程序的顶部。

## 数据探索

在这个例子中，数据已经是*干净的*，这意味着它已经可以直接使用，并且没有其他语言的字符，这些字符可能会干扰只期望英文字符的算法。

✅ 你可能需要处理一些需要初步格式化的数据，然后再应用 NLP 技术，但这次不需要。如果需要处理非英文字符，你会怎么做？

花点时间确保数据加载后，你可以通过代码来探索它。很容易想要直接关注 `Negative_Review` 和 `Positive_Review` 列。它们包含了自然文本，供你的 NLP 算法处理。但等等！在跳入 NLP 和情感分析之前，你应该按照下面的代码检查数据集中给出的值是否与通过 pandas 计算的值一致。

## Dataframe 操作

本节的第一个任务是通过编写代码检查以下断言是否正确（无需更改 dataframe）。

> 就像许多编程任务一样，完成这些任务的方法有很多，但一个好的建议是尽可能简单、易懂，尤其是当你以后需要回顾这段代码时。对于 dataframe，pandas 提供了一个全面的 API，通常可以高效地完成你想要的操作。

将以下问题视为编码任务，尝试在不查看答案的情况下完成它们。

1. 打印出刚刚加载的 dataframe 的*形状*（即行数和列数）。
2. 计算评论者国籍的频率统计：
   1. `Reviewer_Nationality` 列中有多少个不同的值？它们分别是什么？
   2. 数据集中最常见的评论者国籍是什么？（打印国家和评论数量）
   3. 接下来最常见的 10 个国籍及其频率统计是什么？
3. 对于评论最多的前 10 个国籍，每个国籍评论最多的酒店是什么？
4. 数据集中每个酒店的评论数量是多少？（按酒店统计频率）
5. 数据集中每个酒店都有一个 `Average_Score` 列，但你也可以计算一个平均分（即根据数据集中每个酒店的所有评论分数计算平均值）。为 dataframe 添加一个新列，列名为 `Calc_Average_Score`，存储计算的平均分。
6. 是否有酒店的 `Average_Score` 和 `Calc_Average_Score`（四舍五入到小数点后一位）相同？
   1. 尝试编写一个 Python 函数，该函数接受一个 Series（行）作为参数，比较这两个值，并在值不相等时打印消息。然后使用 `.apply()` 方法对每一行应用该函数。
7. 计算并打印 `Negative_Review` 列中值为 "No Negative" 的行数。
8. 计算并打印 `Positive_Review` 列中值为 "No Positive" 的行数。
9. 计算并打印 `Positive_Review` 列中值为 "No Positive" 且 `Negative_Review` 列中值为 "No Negative" 的行数。

### 代码答案

1. 打印出刚刚加载的 dataframe 的*形状*（即行数和列数）

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. 计算评论者国籍的频率统计：

   1. `Reviewer_Nationality` 列中有多少个不同的值？它们分别是什么？
   2. 数据集中最常见的评论者国籍是什么？（打印国家和评论数量）

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

   3. 接下来最常见的 10 个国籍及其频率统计是什么？

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

3. 对于评论最多的前 10 个国籍，每个国籍评论最多的酒店是什么？

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

4. 数据集中每个酒店的评论数量是多少？（按酒店统计频率）

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

   你可能会注意到，*数据集中统计的*结果与 `Total_Number_of_Reviews` 的值不匹配。目前尚不清楚数据集中该值是否表示酒店的总评论数，但并未全部被抓取，或者是其他计算方式。由于这种不确定性，`Total_Number_of_Reviews` 并未用于模型中。

5. 数据集中每个酒店都有一个 `Average_Score` 列，但你也可以计算一个平均分（即根据数据集中每个酒店的所有评论分数计算平均值）。为 dataframe 添加一个新列，列名为 `Calc_Average_Score`，存储计算的平均分。打印出 `Hotel_Name`、`Average_Score` 和 `Calc_Average_Score` 列。

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

   你可能还会疑惑 `Average_Score` 的值为何有时与计算的平均分不同。由于我们无法知道为什么有些值匹配，而其他值存在差异，在这种情况下，最安全的做法是使用评论分数自行计算平均分。不过，差异通常非常小，以下是数据集中平均分与计算平均分差异最大的酒店：

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

   只有 1 家酒店的分数差异大于 1，这意味着我们可以忽略这些差异，使用计算的平均分。

6. 计算并打印 `Negative_Review` 列中值为 "No Negative" 的行数。

7. 计算并打印 `Positive_Review` 列中值为 "No Positive" 的行数。

8. 计算并打印 `Positive_Review` 列中值为 "No Positive" 且 `Negative_Review` 列中值为 "No Negative" 的行数。

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

   你可能注意到，有 127 行的 `Negative_Review` 和 `Positive_Review` 列分别为 "No Negative" 和 "No Positive"。这意味着评论者给酒店打了一个数字分数，但没有写任何正面或负面的评论。幸运的是，这只是很少的一部分数据（127 行占 515738 行的 0.02%），所以它可能不会对我们的模型或结果产生显著影响。不过，你可能没有预料到一个评论数据集中会有没有评论内容的行，因此值得探索数据以发现类似的情况。

现在你已经探索了数据集，在下一节课中，你将过滤数据并添加一些情感分析。

---
## 🚀挑战

正如我们在之前的课程中看到的，这节课展示了理解数据及其特性在执行操作之前是多么重要。特别是基于文本的数据需要仔细检查。深入挖掘各种以文本为主的数据集，看看是否能发现可能引入偏差或导致情感倾斜的地方。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

参加 [NLP 学习路径](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott)，了解构建语音和文本模型时可以尝试的工具。

## 作业

[NLTK](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。对于因使用本翻译而引起的任何误解或误读，我们概不负责。