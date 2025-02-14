# Анализ настроений по отзывам об отелях

Теперь, когда вы подробно изучили набор данных, пришло время отфильтровать столбцы, а затем использовать методы обработки естественного языка (NLP) для получения новых сведений о отелях.
## [Викторина перед лекцией](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Операции фильтрации и анализа настроений

Как вы, вероятно, заметили, в наборе данных есть несколько проблем. Некоторые столбцы заполнены бесполезной информацией, другие кажутся некорректными. Если они правильные, неясно, как они были рассчитаны, и ответы не могут быть независимо проверены с помощью ваших собственных расчетов.

## Упражнение: немного больше обработки данных

Очистите данные еще немного. Добавьте столбцы, которые будут полезны позже, измените значения в других столбцах и полностью удалите некоторые столбцы.

1. Первичная обработка столбцов

   1. Удалите `lat` и `lng`

   2. Замените значения `Hotel_Address` следующими значениями (если адрес содержит название города и страны, измените его на просто город и страну).

      Вот единственные города и страны в наборе данных:

      Амстердам, Нидерланды

      Барселона, Испания

      Лондон, Великобритания

      Милан, Италия

      Париж, Франция

      Вена, Австрия 

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

      Теперь вы можете запрашивать данные на уровне страны:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Амстердам, Нидерланды |    105     |
      | Барселона, Испания     |    211     |
      | Лондон, Великобритания |    400     |
      | Милан, Италия         |    162     |
      | Париж, Франция        |    458     |
      | Вена, Австрия        |    158     |

2. Обработка столбцов мета-отзывов об отелях

   1. Удалите `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` с нашим собственным рассчитанным баллом

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Обработка столбцов отзывов

   1. Удалите `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, вы можете начать задаваться вопросом, возможно ли значительно сократить объем обработки, которую вам нужно выполнить. К счастью, это возможно - но сначала вам нужно пройти несколько шагов, чтобы определить интересующие теги.

### Фильтрация тегов

Помните, что цель набора данных - добавить настроения и столбцы, которые помогут вам выбрать лучший отель (для себя или, возможно, для клиента, который поручил вам создать бота для рекомендаций по отелям). Вам нужно задать себе вопрос, полезны ли теги в конечном наборе данных или нет. Вот одно из толкований (если вам нужен был набор данных по другим причинам, разные теги могут остаться в/вне выбора):

1. Тип поездки имеет значение, и он должен остаться
2. Тип группы гостей важен, и он должен остаться
3. Тип номера, люкса или студии, в котором остановился гость, не имеет значения (все отели, по сути, имеют одни и те же номера)
4. Устройство, на котором был представлен отзыв, не имеет значения
5. Количество ночей, которые гость провел в отеле, *может* быть актуальным, если вы считаете, что более длительное пребывание связано с тем, что им понравился отель больше, но это натяжка и, вероятно, неуместно

В итоге, **оставьте 2 типа тегов и удалите остальные**.

Сначала вы не хотите подсчитывать теги, пока они не будут в лучшем формате, поэтому это означает удаление квадратных скобок и кавычек. Вы можете сделать это несколькими способами, но вам нужен самый быстрый, так как это может занять много времени для обработки большого объема данных. К счастью, в pandas есть простой способ выполнить каждый из этих шагов.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Каждый тег становится чем-то вроде: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

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

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` столбец совпадает с одним из новых столбцов, добавьте 1, если нет, добавьте 0. Конечный результат будет подсчетом того, сколько рецензентов выбрали этот отель (в совокупности), например, для деловой поездки против отдыха или для того, чтобы взять с собой питомца, и это полезная информация при рекомендации отеля.

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

### Сохраните ваш файл

Наконец, сохраните набор данных в том виде, в каком он есть сейчас, с новым именем.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Операции анализа настроений

В этом последнем разделе вы примените анализ настроений к столбцам отзывов и сохраните результаты в наборе данных.

## Упражнение: загрузка и сохранение отфильтрованных данных

Обратите внимание, что теперь вы загружаете отфильтрованный набор данных, который был сохранен в предыдущем разделе, **а не** оригинальный набор данных.

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

### Удаление стоп-слов

Если вы собираетесь провести анализ настроений по столбцам негативных и позитивных отзывов, это может занять много времени. На мощном тестовом ноутбуке с быстрым процессором это заняло 12 - 14 минут в зависимости от используемой библиотеки для анализа настроений. Это (относительно) долго, поэтому стоит выяснить, можно ли ускорить этот процесс.

Удаление стоп-слов, или общих английских слов, которые не изменяют смысл предложения, - это первый шаг. Удалив их, анализ настроений должен проходить быстрее, но не менее точно (поскольку стоп-слова не влияют на настроение, но замедляют анализ).

Самый длинный негативный отзыв содержал 395 слов, но после удаления стоп-слов он стал 195 словами.

Удаление стоп-слов также является быстрой операцией: удаление стоп-слов из 2 столбцов отзывов на 515,000 строк заняло 3.3 секунды на тестовом устройстве. Для вас это может занять немного больше или меньше времени в зависимости от скорости процессора вашего устройства, объема ОЗУ, есть ли у вас SSD и некоторых других факторов. Относительная краткость операции означает, что если она улучшает время анализа настроений, то это стоит сделать.

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

### Проведение анализа настроений

Теперь вы должны рассчитать анализ настроений для обоих столбцов негативных и позитивных отзывов и сохранить результат в 2 новых столбцах. Проверкой настроения будет сравнение его с оценкой рецензента за тот же отзыв. Например, если анализ настроений считает, что негативный отзыв имеет настроение 1 (крайне положительное настроение) и положительный отзыв также имеет настроение 1, но рецензент поставил отелю наименьшую возможную оценку, то либо текст отзыва не соответствует оценке, либо анализатор настроений не смог правильно распознать настроение. Вы должны ожидать, что некоторые оценки настроения будут совершенно неверными, и это часто можно объяснить, например, отзыв может быть крайне саркастичным: "Конечно, мне ОЧЕНЬ понравилось спать в комнате без отопления", и анализатор настроений считает, что это положительное настроение, хотя человек, читающий это, поймет, что это сарказм.

NLTK предоставляет различные анализаторы настроений для обучения, и вы можете заменить их и посмотреть, будет ли настроение более или менее точным. Здесь используется анализ настроений VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Анн Арбор, Мичиган, июнь 2014.

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

Позже в вашей программе, когда вы будете готовы рассчитать настроение, вы можете применить его к каждому отзыву следующим образом:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Это занимает примерно 120 секунд на моем компьютере, но на каждом компьютере это будет варьироваться. Если вы хотите распечатать результаты и посмотреть, соответствует ли настроение отзыву:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Последнее, что нужно сделать с файлом перед его использованием в задании, - это сохранить его! Вам также стоит рассмотреть возможность переупорядочивания всех ваших новых столбцов, чтобы с ними было легко работать (для человека это косметическое изменение).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Вы должны запустить весь код для [ноутбука анализа](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (после того как вы запустили [ноутбук фильтрации](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), чтобы сгенерировать файл Hotel_Reviews_Filtered.csv).

Чтобы подвести итоги, шаги следующие:

1. Оригинальный файл набора данных **Hotel_Reviews.csv** был изучен на предыдущем уроке с помощью [ноутбука исследователя](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv был отфильтрован с помощью [ноутбука фильтрации](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), в результате чего получился **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv был обработан с помощью [ноутбука анализа настроений](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), в результате чего получился **Hotel_Reviews_NLP.csv**
4. Используйте Hotel_Reviews_NLP.csv в NLP Challenge ниже

### Заключение

Когда вы начали, у вас был набор данных со столбцами и данными, но не все из них могли быть проверены или использованы. Вы изучили данные, отфильтровали то, что вам не нужно, преобразовали теги во что-то полезное, рассчитали свои собственные средние значения, добавили несколько столбцов настроений и, надеюсь, узнали интересные вещи о обработке естественного текста.

## [Викторина после лекции](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Задание

Теперь, когда у вас есть набор данных, проанализированный на предмет настроений, посмотрите, сможете ли вы использовать стратегии, которые вы изучили в этой программе (например, кластеризацию?), чтобы определить закономерности вокруг настроений.

## Обзор и самообучение

Пройдите [этот учебный модуль](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), чтобы узнать больше и использовать различные инструменты для изучения настроений в тексте.
## Задание 

[Попробуйте другой набор данных](assignment.md)

**Отказ от ответственности**:  
Этот документ был переведен с использованием услуг машинного перевода на основе ИИ. Хотя мы стремимся к точности, пожалуйста, имейте в виду, что автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на родном языке должен считаться авторитетным источником. Для критически важной информации рекомендуется профессиональный человеческий перевод. Мы не несем ответственности за любые недоразумения или неправильные толкования, возникающие в результате использования этого перевода.