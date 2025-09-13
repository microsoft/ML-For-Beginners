<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T01:42:04+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "bg"
}
-->
# Анализ на настроения с хотелски ревюта

Сега, след като сте разгледали набора от данни в детайли, е време да филтрирате колоните и да използвате техники за обработка на естествен език (NLP), за да получите нови прозрения за хотелите.

## [Тест преди лекцията](https://ff-quizzes.netlify.app/en/ml/)

### Операции за филтриране и анализ на настроения

Както вероятно сте забелязали, наборът от данни има някои проблеми. Някои колони са запълнени с ненужна информация, други изглеждат некоректни. Дори ако са коректни, не е ясно как са били изчислени, а отговорите не могат да бъдат независимо проверени чрез ваши собствени изчисления.

## Упражнение: малко повече обработка на данни

Почистете данните още малко. Добавете колони, които ще бъдат полезни по-късно, променете стойностите в други колони и напълно премахнете определени колони.

1. Първоначална обработка на колоните

   1. Премахнете `lat` и `lng`

   2. Заменете стойностите в `Hotel_Address` със следните стойности (ако адресът съдържа името на града и страната, променете го само на града и страната).

      Това са единствените градове и страни в набора от данни:

      Амстердам, Нидерландия

      Барселона, Испания

      Лондон, Обединено кралство

      Милано, Италия

      Париж, Франция

      Виена, Австрия 

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

      Сега можете да правите заявки на ниво страна:

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

2. Обработка на мета-ревю колоните на хотелите

  1. Премахнете `Additional_Number_of_Scoring`

  1. Заменете `Total_Number_of_Reviews` с общия брой ревюта за този хотел, които всъщност са в набора от данни 

  1. Заменете `Average_Score` с нашия собствен изчислен резултат

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Обработка на колоните с ревюта

   1. Премахнете `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` и `days_since_review`

   2. Запазете `Reviewer_Score`, `Negative_Review` и `Positive_Review` такива, каквито са,
     
   3. Запазете `Tags` засега

     - Ще направим допълнителни операции за филтриране на таговете в следващия раздел и след това таговете ще бъдат премахнати

4. Обработка на колоните с информация за ревюиращите

  1. Премахнете `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Запазете `Reviewer_Nationality`

### Колоната с тагове

Колоната `Tag` е проблематична, тъй като представлява списък (в текстов формат), съхраняван в колоната. За съжаление, редът и броят на подсекциите в тази колона не винаги са еднакви. Трудно е за човек да идентифицира правилните фрази, които са от интерес, защото има 515,000 реда и 1427 хотела, и всеки има леко различни опции, които ревюиращият може да избере. Тук NLP е полезен. Можете да сканирате текста и да намерите най-често срещаните фрази и да ги преброите.

За съжаление, не се интересуваме от единични думи, а от многословни фрази (например *Бизнес пътуване*). Изпълнението на алгоритъм за честотно разпределение на многословни фрази върху толкова много данни (6762646 думи) може да отнеме изключително много време, но без да разгледате данните, изглежда, че това е необходим разход. Тук идва полезността на изследователския анализ на данни, защото сте видели пример на таговете като `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, можете да започнете да се питате дали е възможно значително да намалите обработката, която трябва да направите. За щастие, това е възможно - но първо трябва да следвате няколко стъпки, за да установите кои тагове са от интерес.

### Филтриране на таговете

Запомнете, че целта на набора от данни е да добавите настроения и колони, които ще ви помогнат да изберете най-добрия хотел (за себе си или може би за клиент, който ви е възложил да създадете бот за препоръка на хотели). Трябва да се запитате дали таговете са полезни или не в крайния набор от данни. Ето едно тълкуване (ако ви е нужен наборът от данни за други цели, различни тагове може да останат/да бъдат премахнати):

1. Типът на пътуването е релевантен и трябва да остане
2. Типът на групата гости е важен и трябва да остане
3. Типът на стаята, апартамента или студиото, в които гостът е отседнал, е нерелевантен (всички хотели имат основно едни и същи стаи)
4. Устройството, от което е изпратено ревюто, е нерелевантно
5. Броят на нощувките, за които ревюиращият е отседнал, *може* да бъде релевантен, ако свържете по-дългите престои с харесването на хотела, но това е малко вероятно и вероятно нерелевантно

В обобщение, **запазете 2 вида тагове и премахнете останалите**.

Първо, не искате да броите таговете, докато не са в по-добър формат, което означава премахване на квадратните скоби и кавичките. Можете да направите това по няколко начина, но искате най-бързия, тъй като обработката на много данни може да отнеме много време. За щастие, pandas има лесен начин за изпълнение на всяка от тези стъпки.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Всеки таг става нещо като: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

След това откриваме проблем. Някои ревюта или редове имат 5 колони, други 3, трети 6. Това е резултат от начина, по който е създаден наборът от данни, и е трудно за поправяне. Искате да получите честотен брой на всяка фраза, но те са в различен ред във всяко ревю, така че броят може да е неточен, а хотел може да не получи таг, който заслужава.

Вместо това ще използвате различния ред в наша полза, защото всеки таг е многословен, но също така е разделен със запетая! Най-простият начин да направите това е да създадете 6 временни колони, като всеки таг се поставя в колоната, съответстваща на неговия ред в тага. След това можете да обедините 6-те колони в една голяма колона и да изпълните метода `value_counts()` върху получената колона. При отпечатване ще видите, че има 2428 уникални тага. Ето малък пример:

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

Някои от често срещаните тагове като `Submitted from a mobile device` не са полезни за нас, така че може да е умно да ги премахнете преди преброяване на честотата на фразите, но това е толкова бърза операция, че можете да ги оставите и просто да ги игнорирате.

### Премахване на таговете за продължителност на престоя

Премахването на тези тагове е стъпка 1, което леко намалява общия брой тагове, които трябва да бъдат разгледани. Обърнете внимание, че не ги премахвате от набора от данни, а просто избирате да ги премахнете от разглеждане като стойности за броене/запазване в набора от данни с ревюта.

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

Има огромно разнообразие от стаи, апартаменти, студиа, апартаменти и т.н. Всички те означават приблизително едно и също и не са релевантни за вас, така че ги премахнете от разглеждане.

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

Накрая, и това е приятно (защото не отне много обработка), ще останете със следните *полезни* тагове:

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

Може да се твърди, че `Travellers with friends` е същото като `Group` повече или по-малко, и би било справедливо да комбинирате двете, както е показано по-горе. Кодът за идентифициране на правилните тагове е [в notebook-а за тагове](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Последната стъпка е да създадете нови колони за всеки от тези тагове. След това, за всеки ред с ревю, ако колоната `Tag` съвпада с една от новите колони, добавете 1, ако не, добавете 0. Крайният резултат ще бъде броят на ревюиращите, които са избрали този хотел (в агрегат) за, например, бизнес срещу свободно време, или за да доведат домашен любимец, и това е полезна информация при препоръчване на хотел.

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

### Запазете файла си

Накрая, запазете набора от данни в сегашния му вид с ново име.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Операции за анализ на настроения

В този последен раздел ще приложите анализ на настроения към колоните с ревюта и ще запазите резултатите в набора от данни.

## Упражнение: заредете и запазете филтрираните данни

Обърнете внимание, че сега зареждате филтрирания набор от данни, който беше запазен в предишния раздел, **а не** оригиналния набор от данни.

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

### Премахване на стоп думи

Ако изпълните анализ на настроения върху колоните с негативни и позитивни ревюта, това може да отнеме много време. Тествано на мощен лаптоп с бърз процесор, отнема 12 - 14 минути в зависимост от използваната библиотека за анализ на настроения. Това е (относително) дълго време, така че си струва да се изследва дали това може да бъде ускорено. 

Премахването на стоп думи, или често срещани английски думи, които не променят настроението на изречението, е първата стъпка. Чрез премахването им, анализът на настроения трябва да се изпълнява по-бързо, но не и по-малко точно (тъй като стоп думите не влияят на настроението, но забавят анализа). 

Най-дългото негативно ревю беше 395 думи, но след премахването на стоп думите, то е 195 думи.

Премахването на стоп думите също е бърза операция, премахването на стоп думите от 2 колони с ревюта върху 515,000 реда отне 3.3 секунди на тестовото устройство. Това може да отнеме малко повече или по-малко време за вас в зависимост от скоростта на процесора, RAM, дали имате SSD или не, и някои други фактори. Относителната краткост на операцията означава, че ако подобрява времето за анализ на настроения, тогава си струва да се направи.

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

### Изпълнение на анализ на настроения

Сега трябва да изчислите анализа на настроения както за колоните с негативни, така и за позитивни ревюта, и да запазите резултата в 2 нови колони. Тестът на настроението ще бъде да го сравните с оценката на ревюиращия за същото ревю. Например, ако анализът на настроения смята, че негативното ревю има настроение 1 (изключително позитивно настроение) и позитивното ревю има настроение 1, но ревюиращият е дал на хотела най-ниската възможна оценка, тогава или текстът на ревюто не съответства на оценката, или анализаторът на настроения не е успял да разпознае настроението правилно. Трябва да очаквате някои оценки на настроенията да бъдат напълно грешни, и често това ще бъде обяснимо, например ревюто може да бъде изключително саркастично "Разбира се, ОБОЖАВАХ да спя в стая без отопление" и анализаторът на настроения смята, че това е позитивно настроение, въпреки че човек, който го чете, би разбрал, че това е сарказъм.
NLTK предоставя различни анализатори на настроения, с които можете да експериментирате, като ги замените и проверите дали анализът на настроенията е по-точен или не. Тук се използва анализът на настроенията VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, юни 2014.

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

По-късно във вашата програма, когато сте готови да изчислите настроението, можете да го приложите към всяко ревю, както следва:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Това отнема приблизително 120 секунди на моя компютър, но времето може да варира в зависимост от компютъра. Ако искате да отпечатате резултатите и да проверите дали настроението съответства на ревюто:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Последното нещо, което трябва да направите с файла, преди да го използвате в предизвикателството, е да го запазите! Също така трябва да обмислите пренареждането на всички нови колони, за да бъдат по-удобни за работа (за човек това е козметична промяна).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Трябва да изпълните целия код от [анализиращия ноутбук](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (след като сте изпълнили [филтриращия ноутбук](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), за да генерирате файла Hotel_Reviews_Filtered.csv).

За преглед, стъпките са:

1. Оригиналният файл с данни **Hotel_Reviews.csv** е разгледан в предишния урок с [ноутбука за изследване](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv е филтриран чрез [филтриращия ноутбук](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), което води до **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv е обработен чрез [ноутбука за анализ на настроения](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), което води до **Hotel_Reviews_NLP.csv**
4. Използвайте Hotel_Reviews_NLP.csv в предизвикателството за NLP по-долу

### Заключение

Когато започнахте, имахте набор от данни с колони и информация, но не всичко можеше да бъде проверено или използвано. Разгледахте данните, филтрирахте ненужното, преобразувахте таговете в нещо полезно, изчислихте свои собствени средни стойности, добавихте колони за настроения и, надявам се, научихте интересни неща за обработката на естествен текст.

## [Тест след лекцията](https://ff-quizzes.netlify.app/en/ml/)

## Предизвикателство

Сега, когато вашият набор от данни е анализиран за настроения, опитайте да използвате стратегии, които сте научили в този курс (например клъстериране), за да определите модели, свързани с настроенията.

## Преглед и самостоятелно обучение

Вземете [този модул в Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), за да научите повече и да използвате различни инструменти за изследване на настроения в текст.

## Задача

[Опитайте с различен набор от данни](assignment.md)

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.