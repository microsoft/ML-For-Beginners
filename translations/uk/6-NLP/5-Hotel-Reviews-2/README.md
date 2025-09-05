<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T14:21:48+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "uk"
}
-->
# Аналіз настроїв за відгуками про готелі

Тепер, коли ви детально дослідили набір даних, настав час відфільтрувати стовпці та застосувати техніки обробки природної мови (NLP) до набору даних, щоб отримати нові інсайти про готелі.

## [Тест перед лекцією](https://ff-quizzes.netlify.app/en/ml/)

### Операції фільтрації та аналізу настроїв

Як ви, мабуть, помітили, набір даних має кілька проблем. Деякі стовпці заповнені непотрібною інформацією, інші здаються некоректними. Якщо вони правильні, незрозуміло, як вони були розраховані, і відповіді неможливо перевірити самостійними обчисленнями.

## Вправа: трохи більше обробки даних

Очистіть дані ще трохи. Додайте стовпці, які будуть корисними пізніше, змініть значення в інших стовпцях і повністю видаліть певні стовпці.

1. Початкова обробка стовпців

   1. Видаліть `lat` і `lng`

   2. Замініть значення `Hotel_Address` на наступні (якщо адреса містить назву міста та країни, змініть її на просто місто та країну).

      Ось єдині міста та країни в наборі даних:

      Амстердам, Нідерланди

      Барселона, Іспанія

      Лондон, Сполучене Королівство

      Мілан, Італія

      Париж, Франція

      Відень, Австрія 

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

      Тепер ви можете запитувати дані на рівні країни:

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

2. Обробка стовпців мета-відгуків про готелі

  1. Видаліть `Additional_Number_of_Scoring`

  1. Замініть `Total_Number_of_Reviews` на загальну кількість відгуків для цього готелю, які фактично є в наборі даних 

  1. Замініть `Average_Score` на власний розрахований бал

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Обробка стовпців відгуків

   1. Видаліть `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` і `days_since_review`

   2. Залиште `Reviewer_Score`, `Negative_Review` і `Positive_Review` без змін,
     
   3. Залиште `Tags` поки що

     - Ми будемо виконувати додаткові операції фільтрації на тегах у наступному розділі, а потім теги будуть видалені

4. Обробка стовпців рецензентів

  1. Видаліть `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Залиште `Reviewer_Nationality`

### Стовпці тегів

Стовпець `Tag` є проблематичним, оскільки він містить список (у текстовій формі), збережений у стовпці. На жаль, порядок і кількість підрозділів у цьому стовпці не завжди однакові. Людині важко визначити правильні фрази, які можуть бути цікавими, тому що є 515,000 рядків і 1427 готелів, і кожен має трохи різні варіанти, які рецензент міг вибрати. Тут стає корисним NLP. Ви можете сканувати текст і знаходити найпоширеніші фрази, а також рахувати їх.

На жаль, нас не цікавлять окремі слова, а багатослівні фрази (наприклад, *Ділова поїздка*). Запуск алгоритму частотного розподілу багатослівних фраз на такій кількості даних (6762646 слів) може зайняти надзвичайно багато часу, але без перегляду даних здається, що це необхідна витрата. Тут стає корисним дослідницький аналіз даних, оскільки ви бачили зразок тегів, таких як `[' Ділова поїздка  ', ' Самостійний мандрівник ', ' Одномісний номер ', ' Перебування 5 ночей ', ' Надіслано з мобільного пристрою ']`, ви можете почати запитувати, чи можливо значно скоротити обробку, яку вам потрібно виконати. На щастя, це можливо - але спочатку потрібно виконати кілька кроків, щоб визначити цікаві теги.

### Фільтрація тегів

Пам’ятайте, що мета набору даних — додати настрої та стовпці, які допоможуть вам вибрати найкращий готель (для себе або, можливо, для клієнта, який доручив вам створити бота для рекомендацій готелів). Вам потрібно запитати себе, чи є теги корисними чи ні в остаточному наборі даних. Ось одне з тлумачень (якщо вам потрібен набір даних для інших цілей, різні теги можуть залишитися/вийти з вибору):

1. Тип поїздки є важливим, і він має залишитися
2. Тип групи гостей є важливим, і він має залишитися
3. Тип кімнати, люксу чи студії, в якій зупинявся гість, не має значення (усі готелі мають приблизно однакові кімнати)
4. Пристрій, на якому був надісланий відгук, не має значення
5. Кількість ночей, протягом яких рецензент залишався, *може* бути важливою, якщо ви пов’язуєте довші перебування з тим, що їм більше подобається готель, але це сумнівно і, ймовірно, не має значення

Підсумовуючи, **залиште 2 типи тегів і видаліть інші**.

Спочатку ви не хочете рахувати теги, поки вони не будуть у кращому форматі, а це означає видалення квадратних дужок і лапок. Ви можете зробити це кількома способами, але вам потрібен найшвидший, оскільки це може зайняти багато часу для обробки великої кількості даних. На щастя, pandas має простий спосіб виконати кожен із цих кроків.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Кожен тег стає таким: `Ділова поїздка, Самостійний мандрівник, Одномісний номер, Перебування 5 ночей, Надіслано з мобільного пристрою`. 

Далі ми стикаємося з проблемою. Деякі відгуки або рядки мають 5 стовпців, деякі 3, деякі 6. Це результат того, як був створений набір даних, і важко виправити. Ви хочете отримати частотний підрахунок кожної фрази, але вони знаходяться в різному порядку в кожному відгуку, тому підрахунок може бути неправильним, і готель може не отримати тег, який він заслуговував.

Натомість ви використаєте різний порядок на свою користь, оскільки кожен тег є багатослівним, але також розділений комою! Найпростіший спосіб зробити це — створити 6 тимчасових стовпців, у кожен з яких вставити тег відповідно до його порядку в тегах. Потім ви можете об’єднати 6 стовпців в один великий стовпець і запустити метод `value_counts()` для отриманого стовпця. Надрукувавши це, ви побачите, що було 2428 унікальних тегів. Ось невеликий зразок:

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

Деякі з поширених тегів, таких як `Надіслано з мобільного пристрою`, нам не потрібні, тому може бути розумним видалити їх перед підрахунком частоти фраз, але це така швидка операція, що ви можете залишити їх і просто ігнорувати.

### Видалення тегів тривалості перебування

Видалення цих тегів — це перший крок, він трохи зменшує загальну кількість тегів, які потрібно враховувати. Зверніть увагу, що ви не видаляєте їх із набору даних, а просто вирішуєте видалити їх із розгляду як значення для підрахунку/збереження в наборі даних відгуків.

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

Існує величезна різноманітність кімнат, люксів, студій, апартаментів тощо. Всі вони означають приблизно одне й те саме і не мають значення для вас, тому видаліть їх із розгляду.

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

Нарешті, і це чудово (оскільки це не потребувало багато обробки), ви залишитеся з наступними *корисними* тегами:

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

Можна стверджувати, що `Мандрівники з друзями` — це те саме, що й `Група`, і було б справедливо об’єднати ці два, як показано вище. Код для визначення правильних тегів знаходиться в [ноутбуці Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Останній крок — створити нові стовпці для кожного з цих тегів. Потім для кожного рядка відгуків, якщо стовпець `Tag` відповідає одному з нових стовпців, додайте 1, якщо ні, додайте 0. Кінцевим результатом буде підрахунок того, скільки рецензентів вибрали цей готель (у сукупності) для, наприклад, бізнесу чи відпочинку, або щоб привезти домашнього улюбленця, і це корисна інформація при рекомендації готелю.

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

### Збережіть ваш файл

Нарешті, збережіть набір даних у його поточному вигляді з новою назвою.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Операції аналізу настроїв

У цьому останньому розділі ви застосуєте аналіз настроїв до стовпців відгуків і збережете результати в наборі даних.

## Вправа: завантаження та збереження відфільтрованих даних

Зверніть увагу, що тепер ви завантажуєте відфільтрований набір даних, який був збережений у попередньому розділі, **а не** оригінальний набір даних.

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

### Видалення стоп-слів

Якщо ви запустите аналіз настроїв для стовпців негативних і позитивних відгуків, це може зайняти багато часу. Тест на потужному ноутбуці з швидким процесором зайняв 12–14 хвилин залежно від того, яка бібліотека настроїв використовувалася. Це (відносно) довгий час, тому варто дослідити, чи можна його прискорити. 

Видалення стоп-слів, або поширених англійських слів, які не змінюють настрій речення, є першим кроком. Видаляючи їх, аналіз настроїв має працювати швидше, але не менш точно (оскільки стоп-слова не впливають на настрій, але вони уповільнюють аналіз). 

Найдовший негативний відгук складав 395 слів, але після видалення стоп-слів він становить 195 слів.

Видалення стоп-слів також є швидкою операцією, видалення стоп-слів із 2 стовпців відгуків у 515,000 рядках зайняло 3.3 секунди на тестовому пристрої. Це може зайняти трохи більше або менше часу залежно від швидкості процесора вашого пристрою, оперативної пам’яті, наявності SSD тощо. Відносна короткість операції означає, що якщо вона покращує час аналізу настроїв, то її варто виконати.

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

### Виконання аналізу настроїв

Тепер вам слід розрахувати аналіз настроїв для стовпців негативних і позитивних відгуків і зберегти результат у 2 нових стовпцях. Тест настрою буде порівнювати його з оцінкою рецензента для того ж відгуку. Наприклад, якщо аналіз настроїв вважає, що негативний відгук має настрій 1 (надзвичайно позитивний настрій) і позитивний настрій 1, але рецензент дав готелю найнижчу оцінку, то або текст відгуку не відповідає оцінці, або аналізатор настроїв не зміг правильно розпізнати настрій. Ви повинні очікувати, що деякі оцінки настроїв будуть абсолютно неправильними, і часто це буде пояснювано, наприклад, відгук може бути надзвичайно саркастичним: "Звісно, я ОБОЖНЮВАВ спати в кімнаті без опалення", і аналізатор настроїв вважає, що це позитивний настрій, хоча людина, яка читає це, зрозуміє, що це сарказм.
NLTK пропонує різні аналізатори настрою для навчання, і ви можете замінювати їх, щоб перевірити, чи аналіз настрою є більш точним або менш точним. Тут використовується аналіз настрою VADER.

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

Пізніше у вашій програмі, коли ви будете готові до розрахунку настрою, ви можете застосувати його до кожного відгуку наступним чином:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Це займає приблизно 120 секунд на моєму комп'ютері, але час може варіюватися залежно від комп'ютера. Якщо ви хочете вивести результати і перевірити, чи відповідає настрій відгуку:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Останнє, що потрібно зробити з файлом перед використанням його у завданні, — це зберегти його! Також варто розглянути можливість упорядкування всіх нових колонок, щоб ними було зручно користуватися (для людини це косметична зміна).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Ви повинні запустити весь код для [ноутбука аналізу](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (після того, як ви запустили [ноутбук фільтрації](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), щоб створити файл Hotel_Reviews_Filtered.csv).

Для перегляду, кроки такі:

1. Оригінальний файл набору даних **Hotel_Reviews.csv** досліджується у попередньому уроці за допомогою [ноутбука дослідження](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv фільтрується за допомогою [ноутбука фільтрації](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), що призводить до створення **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv обробляється за допомогою [ноутбука аналізу настрою](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), що призводить до створення **Hotel_Reviews_NLP.csv**
4. Використовуйте Hotel_Reviews_NLP.csv у виклику NLP нижче

### Висновок

Коли ви починали, у вас був набір даних із колонками та даними, але не всі з них могли бути перевірені або використані. Ви дослідили дані, відфільтрували те, що вам не потрібно, перетворили теги у щось корисне, розрахували власні середні значення, додали кілька колонок настрою і, сподіваємося, дізналися щось цікаве про обробку природного тексту.

## [Тест після лекції](https://ff-quizzes.netlify.app/en/ml/)

## Виклик

Тепер, коли ваш набір даних проаналізовано на настрій, спробуйте використати стратегії, які ви вивчили у цьому курсі (можливо, кластеризацію?), щоб визначити шаблони навколо настрою.

## Огляд і самостійне навчання

Пройдіть [цей модуль Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), щоб дізнатися більше і використовувати різні інструменти для дослідження настрою в тексті.

## Завдання

[Спробуйте інший набір даних](assignment.md)

---

**Відмова від відповідальності**:  
Цей документ було перекладено за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, зверніть увагу, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ мовою оригіналу слід вважати авторитетним джерелом. Для критично важливої інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.