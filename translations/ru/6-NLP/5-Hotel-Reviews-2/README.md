<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-06T08:41:29+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ru"
}
-->
# Анализ настроений с отзывами о гостиницах

Теперь, когда вы подробно изучили набор данных, пришло время отфильтровать столбцы и использовать методы обработки естественного языка (NLP), чтобы получить новые инсайты о гостиницах.

## [Тест перед лекцией](https://ff-quizzes.netlify.app/en/ml/)

### Операции фильтрации и анализа настроений

Как вы, вероятно, заметили, в наборе данных есть несколько проблем. Некоторые столбцы заполнены бесполезной информацией, другие кажутся некорректными. Даже если они корректны, непонятно, как они были рассчитаны, и невозможно независимо проверить их с помощью собственных вычислений.

## Упражнение: немного больше обработки данных

Очистите данные еще немного. Добавьте столбцы, которые будут полезны позже, измените значения в других столбцах и полностью удалите определенные столбцы.

1. Первичная обработка столбцов

   1. Удалите `lat` и `lng`.

   2. Замените значения `Hotel_Address` на следующие (если адрес содержит название города и страны, измените его на просто город и страну).

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
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. Обработка столбцов мета-обзоров гостиниц

   1. Удалите `Additional_Number_of_Scoring`.

   2. Замените `Total_Number_of_Reviews` на общее количество отзывов для этой гостиницы, которые фактически есть в наборе данных.

   3. Замените `Average_Score` на собственно рассчитанный средний балл.

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Обработка столбцов отзывов

   1. Удалите `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` и `days_since_review`.

   2. Оставьте `Reviewer_Score`, `Negative_Review` и `Positive_Review` как есть.

   3. Оставьте `Tags` пока.

      - В следующем разделе мы проведем дополнительную фильтрацию тегов, а затем удалим их.

4. Обработка столбцов рецензентов

   1. Удалите `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Оставьте `Reviewer_Nationality`.

### Столбцы тегов

Столбец `Tag` проблематичен, так как он представляет собой список (в текстовом формате), хранящийся в столбце. К сожалению, порядок и количество подразделов в этом столбце не всегда одинаковы. Человеку сложно определить правильные фразы, которые могут быть интересны, потому что в наборе данных 515,000 строк и 1427 гостиниц, и каждая имеет немного разные варианты, которые рецензент мог выбрать. Здесь на помощь приходит NLP. Вы можете сканировать текст, находить наиболее распространенные фразы и подсчитывать их.

К сожалению, нас не интересуют отдельные слова, а многословные фразы (например, *Деловая поездка*). Запуск алгоритма распределения частоты многословных фраз на таком объеме данных (6762646 слов) может занять чрезвычайно много времени, но без анализа данных кажется, что это необходимая мера. Здесь полезен исследовательский анализ данных, так как вы видели пример тегов, таких как `[' Деловая поездка  ', ' Путешественник в одиночку ', ' Одноместный номер ', ' Проживание 5 ночей ', ' Отправлено с мобильного устройства ']`, вы можете начать задаваться вопросом, возможно ли значительно сократить объем обработки. К счастью, это возможно — но сначала нужно выполнить несколько шагов, чтобы определить интересующие теги.

### Фильтрация тегов

Помните, что цель набора данных — добавить настроения и столбцы, которые помогут вам выбрать лучшую гостиницу (для себя или, возможно, для клиента, поручившего вам создать бота для рекомендаций гостиниц). Вам нужно задать себе вопрос, полезны ли теги в итоговом наборе данных. Вот одно из возможных толкований (если вам нужен набор данных для других целей, выбор тегов может быть другим):

1. Тип поездки важен, его следует оставить.
2. Тип группы гостей важен, его следует оставить.
3. Тип номера, люкса или студии, в котором остановился гость, не имеет значения (все гостиницы имеют примерно одинаковые номера).
4. Устройство, с которого был отправлен отзыв, не имеет значения.
5. Количество ночей, проведенных рецензентом, *может* быть важным, если вы связываете более длительное пребывание с тем, что им понравилась гостиница больше, но это маловероятно и, скорее всего, не имеет значения.

В итоге, **оставьте 2 типа тегов и удалите остальные**.

Сначала вы не хотите подсчитывать теги, пока они не будут в лучшем формате, а это значит, что нужно удалить квадратные скобки и кавычки. Сделать это можно несколькими способами, но вы хотите выбрать самый быстрый, так как обработка большого объема данных может занять много времени. К счастью, в pandas есть простой способ выполнить каждый из этих шагов.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Каждый тег становится чем-то вроде: `Деловая поездка, Путешественник в одиночку, Одноместный номер, Проживание 5 ночей, Отправлено с мобильного устройства`.

Далее возникает проблема. Некоторые отзывы или строки имеют 5 столбцов, некоторые 3, некоторые 6. Это результат того, как был создан набор данных, и его сложно исправить. Вы хотите получить частотный подсчет каждой фразы, но они находятся в разном порядке в каждом отзыве, поэтому подсчет может быть неверным, и гостиница может не получить тег, который ей заслуженно принадлежит.

Вместо этого вы используете разный порядок в свою пользу, так как каждый тег является многословным, но также разделен запятой! Самый простой способ сделать это — создать 6 временных столбцов, в которые каждый тег вставляется в столбец, соответствующий его порядку в теге. Затем вы можете объединить 6 столбцов в один большой столбец и запустить метод `value_counts()` на получившемся столбце. Выводя это, вы увидите, что было 2428 уникальных тегов. Вот небольшой пример:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Отдых                         | 417778 |
| Отправлено с мобильного устройства | 307640 |
| Пара                          | 252294 |
| Проживание 1 ночь             | 193645 |
| Проживание 2 ночи             | 133937 |
| Путешественник в одиночку     | 108545 |
| Проживание 3 ночи             | 95821  |
| Деловая поездка               | 82939  |
| Группа                        | 65392  |
| Семья с маленькими детьми     | 61015  |
| Проживание 4 ночи             | 47817  |
| Двухместный номер             | 35207  |
| Стандартный двухместный номер | 32248  |
| Улучшенный двухместный номер  | 31393  |
| Семья с детьми постарше       | 26349  |
| Делюкс двухместный номер      | 24823  |
| Двухместный номер или номер с двумя кроватями | 22393  |
| Проживание 5 ночей            | 20845  |
| Стандартный двухместный номер или номер с двумя кроватями | 17483  |
| Классический двухместный номер | 16989  |
| Улучшенный двухместный номер или номер с двумя кроватями | 13570 |
| 2 номера                      | 12393  |

Некоторые из распространенных тегов, таких как `Отправлено с мобильного устройства`, нам не нужны, поэтому разумно удалить их перед подсчетом частоты фраз, но это такая быстрая операция, что можно оставить их и просто игнорировать.

### Удаление тегов о длительности пребывания

Удаление этих тегов — первый шаг, он немного сокращает общее количество тегов, которые нужно учитывать. Обратите внимание, что вы не удаляете их из набора данных, а просто выбираете их для исключения из рассмотрения как значений для подсчета/сохранения в наборе данных отзывов.

| Длительность пребывания | Count  |
| ----------------------- | ------ |
| Проживание 1 ночь       | 193645 |
| Проживание 2 ночи       | 133937 |
| Проживание 3 ночи       | 95821  |
| Проживание 4 ночи       | 47817  |
| Проживание 5 ночей      | 20845  |
| Проживание 6 ночей      | 9776   |
| Проживание 7 ночей      | 7399   |
| Проживание 8 ночей      | 2502   |
| Проживание 9 ночей      | 1293   |
| ...                     | ...    |

Существует огромное разнообразие номеров, люксов, студий, апартаментов и так далее. Все они примерно одинаковы и не имеют значения для вас, поэтому удалите их из рассмотрения.

| Тип номера                  | Count |
| --------------------------- | ----- |
| Двухместный номер           | 35207 |
| Стандартный двухместный номер | 32248 |
| Улучшенный двухместный номер | 31393 |
| Делюкс двухместный номер    | 24823 |
| Двухместный номер или номер с двумя кроватями | 22393 |
| Стандартный двухместный номер или номер с двумя кроватями | 17483 |
| Классический двухместный номер | 16989 |
| Улучшенный двухместный номер или номер с двумя кроватями | 13570 |

В итоге, и это приятно (потому что это не потребовало много обработки), у вас останутся следующие *полезные* теги:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Отдых                                         | 417778 |
| Пара                                          | 252294 |
| Путешественник в одиночку                     | 108545 |
| Деловая поездка                               | 82939  |
| Группа (объединено с Путешественники с друзьями) | 67535  |
| Семья с маленькими детьми                     | 61015  |
| Семья с детьми постарше                       | 26349  |
| С животным                                    | 1405   |

Можно утверждать, что `Путешественники с друзьями` примерно то же самое, что и `Группа`, и было бы справедливо объединить их, как показано выше. Код для идентификации правильных тегов находится в [блокноте Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Последний шаг — создать новые столбцы для каждого из этих тегов. Затем для каждой строки отзыва, если столбец `Tag` совпадает с одним из новых столбцов, добавьте 1, если нет — добавьте 0. Итоговый результат будет представлять собой подсчет того, сколько рецензентов выбрали эту гостиницу (в совокупности) для, например, бизнеса или отдыха, или чтобы привезти с собой питомца, и это полезная информация при рекомендации гостиницы.

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

### Сохранение файла

Наконец, сохраните набор данных в текущем виде под новым именем.

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

Если вы запустите анализ настроений на столбцах отрицательных и положительных отзывов, это может занять много времени. Тестирование на мощном ноутбуке с быстрым процессором заняло 12–14 минут в зависимости от используемой библиотеки анализа настроений. Это (относительно) долго, поэтому стоит изучить возможность ускорения процесса.

Удаление стоп-слов, или распространенных английских слов, которые не изменяют настроение предложения, — первый шаг. Удалив их, анализ настроений должен работать быстрее, но не менее точно (так как стоп-слова не влияют на настроение, но замедляют анализ).

Самый длинный отрицательный отзыв содержал 395 слов, но после удаления стоп-слов он стал содержать 195 слов.

Удаление стоп-слов также является быстрой операцией: удаление стоп-слов из 2 столбцов отзывов в 515,000 строк заняло 3.3 секунды на тестовом устройстве. Это может занять немного больше или меньше времени в зависимости от скорости процессора вашего устройства, объема оперативной памяти, наличия SSD и некоторых других факторов. Относительная краткость операции означает, что если она улучшает время анализа настроений, то ее стоит выполнить.

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

### Выполнение анализа настроений

Теперь вам нужно рассчитать анализ настроений для столбцов отрицательных и положительных отзывов и сохранить результат в 2 новых столбцах. Тест анализа настроений будет заключаться в сравнении его с оценкой рецензента для того же отзыва. Например, если анализ настроений показывает, что отрицательный отзыв имеет настроение 1 (крайне положительное настроение), а положительный отзыв — тоже 1, но рецензент дал гостинице самый низкий возможный балл, то либо текст отзыва не соответствует оценке, либо анализатор настроений не смог правильно распознать настроение. Вы должны ожидать, что некоторые оценки настроений будут полностью неверными, и часто это будет объяснимо, например, отзыв может быть крайне саркастичным: "Конечно, мне ПОНРАВИЛОСЬ спать в комнате без отопления", и анализатор настроений считает это положительным настроением, хотя человек, читающий это, поймет, что это сарказм.
NLTK предоставляет различные анализаторы настроений, с которыми можно работать, и вы можете заменять их, чтобы проверить, насколько точным будет анализ настроений. Здесь используется анализ настроений VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Простая модель на основе правил для анализа настроений текстов из социальных сетей. Восьмая международная конференция по блогам и социальным медиа (ICWSM-14). Анн-Арбор, Мичиган, июнь 2014.

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

Позже, в вашей программе, когда вы будете готовы вычислить настроение, вы можете применить его к каждому отзыву следующим образом:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

На моем компьютере это занимает примерно 120 секунд, но на каждом компьютере время может отличаться. Если вы хотите вывести результаты и проверить, соответствует ли настроение отзыву:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Последнее, что нужно сделать с файлом перед использованием его в задании, — это сохранить его! Также стоит подумать о переупорядочивании всех новых столбцов, чтобы с ними было проще работать (для человека это косметическое изменение).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Вы должны запустить весь код из [ноутбука анализа](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (после того, как вы запустите [ноутбук фильтрации](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), чтобы сгенерировать файл Hotel_Reviews_Filtered.csv).

Для обзора, шаги следующие:

1. Исходный файл данных **Hotel_Reviews.csv** был изучен в предыдущем уроке с помощью [ноутбука исследования](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv был отфильтрован с помощью [ноутбука фильтрации](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), в результате чего получился файл **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv был обработан с помощью [ноутбука анализа настроений](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), в результате чего получился файл **Hotel_Reviews_NLP.csv**
4. Используйте Hotel_Reviews_NLP.csv в задании NLP ниже

### Заключение

Когда вы начали, у вас был набор данных с колонками и данными, но не все из них можно было проверить или использовать. Вы изучили данные, отфильтровали ненужное, преобразовали теги в полезные данные, рассчитали свои собственные средние значения, добавили несколько колонок с настроениями и, надеюсь, узнали что-то интересное о обработке естественного текста.

## [Тест после лекции](https://ff-quizzes.netlify.app/en/ml/)

## Задание

Теперь, когда ваш набор данных проанализирован на предмет настроений, попробуйте использовать стратегии, которые вы изучили в этом курсе (например, кластеризацию), чтобы определить закономерности, связанные с настроением.

## Обзор и самостоятельное изучение

Пройдите [этот модуль Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), чтобы узнать больше и использовать различные инструменты для изучения настроений в тексте.

## Домашнее задание

[Попробуйте другой набор данных](assignment.md)

---

**Отказ от ответственности**:  
Этот документ был переведен с помощью сервиса автоматического перевода [Co-op Translator](https://github.com/Azure/co-op-translator). Несмотря на наши усилия обеспечить точность, автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на его родном языке следует считать авторитетным источником. Для получения критически важной информации рекомендуется профессиональный перевод человеком. Мы не несем ответственности за любые недоразумения или неправильные интерпретации, возникшие в результате использования данного перевода.