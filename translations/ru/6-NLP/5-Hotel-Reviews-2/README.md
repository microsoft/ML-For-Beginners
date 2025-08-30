<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T22:40:01+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ru"
}
-->
# Анализ настроений с отзывами о гостиницах

Теперь, когда вы подробно изучили набор данных, пришло время отфильтровать столбцы и применить методы обработки естественного языка (NLP) к набору данных, чтобы получить новые инсайты о гостиницах.

## [Тест перед лекцией](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Операции фильтрации и анализа настроений

Как вы, вероятно, заметили, в наборе данных есть несколько проблем. Некоторые столбцы заполнены бесполезной информацией, другие кажутся некорректными. Если они корректны, то непонятно, как они были рассчитаны, и их нельзя проверить независимо с помощью собственных расчетов.

## Упражнение: немного больше обработки данных

Очистите данные еще немного. Добавьте столбцы, которые будут полезны позже, измените значения в других столбцах и полностью удалите определенные столбцы.

1. Первичная обработка столбцов

   1. Удалите `lat` и `lng`.

   2. Замените значения в `Hotel_Address` на следующие (если адрес содержит название города и страны, измените его на только город и страну).

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

2. Обработка столбцов мета-отзывов о гостиницах

   1. Удалите `Additional_Number_of_Scoring`.

   2. Замените `Total_Number_of_Reviews` на общее количество отзывов для этой гостиницы, которые фактически есть в наборе данных.

   3. Замените `Average_Score` на собственный рассчитанный балл.

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

   3. Оставьте `Tags` на данный момент.

      - Мы будем выполнять дополнительные операции фильтрации на тегах в следующем разделе, а затем теги будут удалены.

4. Обработка столбцов рецензентов

   1. Удалите `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Оставьте `Reviewer_Nationality`.

### Столбец тегов

Столбец `Tag` проблематичен, так как он представляет собой список (в текстовом формате), хранящийся в столбце. К сожалению, порядок и количество подкатегорий в этом столбце не всегда одинаковы. Человеку сложно определить, какие фразы важны, потому что в наборе данных 515 000 строк и 1427 гостиниц, и у каждой есть немного разные варианты, которые мог выбрать рецензент. Здесь на помощь приходит NLP. Вы можете сканировать текст, находить наиболее часто встречающиеся фразы и подсчитывать их.

К сожалению, нас не интересуют отдельные слова, а только многословные фразы (например, *Деловая поездка*). Запуск алгоритма частотного распределения многословных фраз на таком объеме данных (6762646 слов) может занять чрезвычайно много времени, но, не глядя на данные, кажется, что это необходимая мера. Здесь полезен исследовательский анализ данных, потому что, увидев пример тегов, таких как `[' Деловая поездка  ', ' Путешественник в одиночку ', ' Одноместный номер ', ' Проживание 5 ночей ', ' Отправлено с мобильного устройства ']`, вы можете начать задаваться вопросом, возможно ли значительно сократить объем обработки. К счастью, это возможно — но сначала нужно выполнить несколько шагов, чтобы определить интересующие теги.

### Фильтрация тегов

Помните, что цель набора данных — добавить настроения и столбцы, которые помогут вам выбрать лучшую гостиницу (для себя или, возможно, для клиента, поручившего вам создать бота для рекомендаций гостиниц). Вам нужно спросить себя, полезны ли теги в итоговом наборе данных. Вот одна из интерпретаций (если бы вам нужен был набор данных для других целей, выбор тегов мог бы быть другим):

1. Тип поездки важен, и он должен остаться.
2. Тип группы гостей важен, и он должен остаться.
3. Тип номера, люкса или студии, в котором останавливался гость, не имеет значения (все гостиницы имеют примерно одинаковые номера).
4. Устройство, с которого был отправлен отзыв, не имеет значения.
5. Количество ночей, проведенных рецензентом, *может* быть важным, если вы связываете более длительное пребывание с тем, что им понравилась гостиница больше, но это маловероятно и, скорее всего, не имеет значения.

В итоге, **оставьте 2 типа тегов и удалите остальные**.

Сначала вы не хотите подсчитывать теги, пока они не будут в лучшем формате, а это значит, что нужно удалить квадратные скобки и кавычки. Это можно сделать несколькими способами, но вы хотите выбрать самый быстрый, так как обработка большого объема данных может занять много времени. К счастью, в pandas есть простой способ выполнить каждый из этих шагов.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Каждый тег становится чем-то вроде: `Деловая поездка, Путешественник в одиночку, Одноместный номер, Проживание 5 ночей, Отправлено с мобильного устройства`.

Далее возникает проблема. Некоторые отзывы или строки имеют 5 столбцов, некоторые 3, некоторые 6. Это результат того, как был создан набор данных, и это сложно исправить. Вы хотите получить частотный подсчет каждой фразы, но они находятся в разном порядке в каждом отзыве, поэтому подсчет может быть неточным, и гостиница может не получить тег, который она заслуживает.

Вместо этого вы используете разный порядок в свою пользу, так как каждый тег является многословным, но также разделен запятой! Самый простой способ сделать это — создать 6 временных столбцов, в которые каждый тег вставляется в столбец, соответствующий его порядку в теге. Затем вы можете объединить 6 столбцов в один большой столбец и запустить метод `value_counts()` на получившемся столбце. Выводя это, вы увидите, что было 2428 уникальных тегов. Вот небольшой пример:

| Тег                             | Количество |
| ------------------------------- | ---------- |
| Отдых                           | 417778     |
| Отправлено с мобильного устройства | 307640     |
| Пара                            | 252294     |
| Проживание 1 ночь               | 193645     |
| Проживание 2 ночи               | 133937     |
| Путешественник в одиночку       | 108545     |
| Проживание 3 ночи               | 95821      |
| Деловая поездка                 | 82939      |
| Группа                          | 65392      |
| Семья с маленькими детьми       | 61015      |
| Проживание 4 ночи               | 47817      |
| Двухместный номер               | 35207      |
| Стандартный двухместный номер   | 32248      |
| Улучшенный двухместный номер    | 31393      |
| Семья с детьми постарше         | 26349      |
| Люкс                            | 24823      |
| Двухместный или с двумя кроватями | 22393      |
| Проживание 5 ночей              | 20845      |
| Стандартный двухместный или с двумя кроватями | 17483 |
| Классический двухместный номер  | 16989      |
| Улучшенный двухместный или с двумя кроватями | 13570 |

Некоторые из распространенных тегов, таких как `Отправлено с мобильного устройства`, нам не нужны, поэтому, возможно, разумно удалить их перед подсчетом частоты фраз, но это такая быстрая операция, что вы можете оставить их и просто игнорировать.

### Удаление тегов о длительности пребывания

Удаление этих тегов — это первый шаг, он немного сокращает общее количество тегов, которые нужно учитывать. Обратите внимание, что вы не удаляете их из набора данных, а просто выбираете не учитывать их как значения для подсчета/сохранения в наборе данных отзывов.

| Длительность пребывания | Количество |
| ------------------------ | ---------- |
| Проживание 1 ночь       | 193645     |
| Проживание 2 ночи       | 133937     |
| Проживание 3 ночи       | 95821      |
| Проживание 4 ночи       | 47817      |
| Проживание 5 ночей      | 20845      |
| Проживание 6 ночей      | 9776       |
| Проживание 7 ночей      | 7399       |
| Проживание 8 ночей      | 2502       |
| Проживание 9 ночей      | 1293       |
| ...                     | ...        |

Существует огромное разнообразие номеров, люксов, студий, апартаментов и так далее. Все они означают примерно одно и то же и не имеют значения для вас, поэтому удалите их из рассмотрения.

| Тип номера                     | Количество |
| ------------------------------ | ---------- |
| Двухместный номер              | 35207      |
| Стандартный двухместный номер  | 32248      |
| Улучшенный двухместный номер   | 31393      |
| Люкс                           | 24823      |
| Двухместный или с двумя кроватями | 22393      |
| Стандартный двухместный или с двумя кроватями | 17483 |
| Классический двухместный номер | 16989      |
| Улучшенный двухместный или с двумя кроватями | 13570 |

В итоге, и это приятно (потому что это не потребовало много обработки), вы останетесь с следующими *полезными* тегами:

| Тег                                           | Количество |
| --------------------------------------------- | ---------- |
| Отдых                                         | 417778     |
| Пара                                          | 252294     |
| Путешественник в одиночку                    | 108545     |
| Деловая поездка                               | 82939      |
| Группа (объединено с Путешественники с друзьями) | 67535      |
| Семья с маленькими детьми                    | 61015      |
| Семья с детьми постарше                      | 26349      |
| С питомцем                                   | 1405       |

Можно утверждать, что `Путешественники с друзьями` — это то же самое, что и `Группа`, и было бы справедливо объединить их, как показано выше. Код для определения правильных тегов находится в [блокноте Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Последний шаг — создать новые столбцы для каждого из этих тегов. Затем для каждой строки отзыва, если столбец `Tag` совпадает с одним из новых столбцов, добавьте 1, если нет — добавьте 0. Итоговый результат будет представлять собой подсчет того, сколько рецензентов выбрали эту гостиницу (в совокупности) для, например, деловой поездки, отдыха или поездки с питомцем, и это полезная информация при рекомендации гостиницы.

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

Наконец, сохраните набор данных в его текущем виде с новым именем.

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

Если вы запустите анализ настроений на столбцах с отрицательными и положительными отзывами, это может занять много времени. На тестовом ноутбуке с мощным процессором это заняло 12–14 минут в зависимости от используемой библиотеки анализа настроений. Это (относительно) долго, поэтому стоит изучить, можно ли ускорить этот процесс.

Удаление стоп-слов, или распространенных английских слов, которые не влияют на настроение предложения, — это первый шаг. Удалив их, анализ настроений должен выполняться быстрее, но не менее точно (так как стоп-слова не влияют на настроение, но замедляют анализ).

Самый длинный отрицательный отзыв содержал 395 слов, но после удаления стоп-слов он сократился до 195 слов.

Удаление стоп-слов также является быстрой операцией: удаление стоп-слов из 2 столбцов отзывов в 515 000 строк заняло 3,3 секунды на тестовом устройстве. Это может занять немного больше или меньше времени в зависимости от скорости вашего процессора, объема оперативной памяти, наличия SSD и некоторых других факторов. Относительная быстрота операции означает, что если она улучшает время анализа настроений, то ее стоит выполнить.

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
Теперь вам нужно провести анализ тональности для колонок с отрицательными и положительными отзывами и сохранить результаты в двух новых колонках. Проверка тональности будет заключаться в сравнении её с оценкой, которую поставил рецензент за тот же отзыв. Например, если анализ тональности определил, что отрицательный отзыв имеет тональность 1 (крайне положительная тональность), а положительный отзыв также имеет тональность 1, но рецензент поставил отелю минимальную оценку, то это может означать, что текст отзыва не соответствует оценке, или анализатор тональности не смог корректно распознать тональность. Вы должны быть готовы к тому, что некоторые результаты анализа тональности будут совершенно неверными, и часто это будет объяснимо, например, если отзыв был крайне саркастичным: «Конечно, я ОБОЖАЛ спать в комнате без отопления», а анализатор тональности определил это как положительный отзыв, хотя человек, читающий это, поймёт сарказм.

NLTK предоставляет различные анализаторы тональности для изучения, и вы можете заменить их, чтобы проверить, будет ли тональность определяться более или менее точно. В данном случае используется анализатор тональности VADER.

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

Позже в вашей программе, когда вы будете готовы к вычислению тональности, вы можете применить её к каждому отзыву следующим образом:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

На моём компьютере это занимает примерно 120 секунд, но на других компьютерах время может варьироваться. Если вы хотите вывести результаты и проверить, соответствует ли тональность отзыву:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Самое последнее, что нужно сделать с файлом перед использованием его в задании, — это сохранить его! Также стоит рассмотреть возможность упорядочивания всех новых колонок так, чтобы с ними было удобно работать (для человека это косметическое изменение).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Вы должны запустить весь код из [ноутбука для анализа](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (после того как вы запустите [ноутбук для фильтрации](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), чтобы сгенерировать файл Hotel_Reviews_Filtered.csv).

Чтобы повторить шаги:

1. Исходный файл данных **Hotel_Reviews.csv** был исследован в предыдущем уроке с помощью [ноутбука для исследования](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv был отфильтрован с помощью [ноутбука для фильтрации](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), в результате чего получился файл **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv был обработан с помощью [ноутбука для анализа тональности](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), в результате чего получился файл **Hotel_Reviews_NLP.csv**
4. Используйте Hotel_Reviews_NLP.csv в задании по NLP ниже

### Заключение

Когда вы начали, у вас был набор данных с колонками и данными, но не все из них можно было проверить или использовать. Вы исследовали данные, отфильтровали ненужное, преобразовали теги в полезную информацию, рассчитали свои собственные средние значения, добавили колонки с тональностью и, надеюсь, узнали что-то интересное о работе с естественным текстом.

## [Викторина после лекции](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Задание

Теперь, когда вы проанализировали набор данных на тональность, попробуйте использовать стратегии, которые вы изучили в этом курсе (например, кластеризацию), чтобы определить закономерности, связанные с тональностью.

## Обзор и самостоятельное изучение

Пройдите [этот модуль Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), чтобы узнать больше и использовать различные инструменты для анализа тональности текста.

## Домашнее задание

[Попробуйте другой набор данных](assignment.md)

---

**Отказ от ответственности**:  
Этот документ был переведен с помощью сервиса автоматического перевода [Co-op Translator](https://github.com/Azure/co-op-translator). Хотя мы стремимся к точности, пожалуйста, учитывайте, что автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на его родном языке следует считать авторитетным источником. Для получения критически важной информации рекомендуется профессиональный перевод человеком. Мы не несем ответственности за любые недоразумения или неправильные интерпретации, возникшие в результате использования данного перевода.