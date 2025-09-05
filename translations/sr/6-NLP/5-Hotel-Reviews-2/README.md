<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T14:18:41+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "sr"
}
-->
# Анализа сентимента са рецензијама хотела

Сада када сте детаљно истражили скуп података, време је да филтрирате колоне и примените NLP технике на скуп података како бисте добили нове увиде о хотелима.

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/)

### Операције филтрирања и анализе сентимента

Као што сте вероватно приметили, скуп података има неколико проблема. Неке колоне су испуњене бескорисним информацијама, друге изгледају нетачно. Чак и ако су тачне, није јасно како су израчунате, а одговори се не могу независно проверити вашим сопственим прорачунима.

## Вежба: мало више обраде података

Очистите податке још мало. Додајте колоне које ће бити корисне касније, промените вредности у другим колонама и потпуно уклоните одређене колоне.

1. Почетна обрада колона

   1. Уклоните `lat` и `lng`

   2. Замените вредности у `Hotel_Address` следећим вредностима (ако адреса садржи име града и државе, промените је тако да садржи само град и државу).

      Ово су једини градови и државе у скупу података:

      Амстердам, Холандија

      Барселона, Шпанија

      Лондон, Уједињено Краљевство

      Милано, Италија

      Париз, Француска

      Беч, Аустрија 

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

      Сада можете упитати податке на нивоу државе:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Амстердам, Холандија   |    105     |
      | Барселона, Шпанија     |    211     |
      | Лондон, Уједињено Краљевство | 400 |
      | Милано, Италија        |    162     |
      | Париз, Француска       |    458     |
      | Беч, Аустрија          |    158     |

2. Обрада колона мета-рецензија хотела

   1. Уклоните `Additional_Number_of_Scoring`

   2. Замените `Total_Number_of_Reviews` са укупним бројем рецензија за тај хотел које су заправо у скупу података 

   3. Замените `Average_Score` са сопствено израчунатом оценом

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Обрада колона рецензија

   1. Уклоните `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` и `days_since_review`

   2. Задржите `Reviewer_Score`, `Negative_Review` и `Positive_Review` како јесу
     
   3. Задржите `Tags` за сада

     - У следећем одељку ћемо извршити додатне операције филтрирања на таговима, а затим ће тагови бити уклоњени

4. Обрада колона рецензената

   1. Уклоните `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Задржите `Reviewer_Nationality`

### Колона са таговима

Колона `Tag` је проблематична јер је листа (у текстуалном облику) која се чува у колони. Нажалост, редослед и број пододсека у овој колони нису увек исти. Тешко је човеку да идентификује тачне фразе које су од интереса, јер постоји 515,000 редова и 1427 хотела, а сваки има мало другачије опције које рецензент може изабрати. Овде NLP долази до изражаја. Можете скенирати текст и пронаћи најчешће фразе и пребројати их.

Нажалост, нас не интересују појединачне речи, већ фразе са више речи (нпр. *Пословно путовање*). Покретање алгоритма за дистрибуцију учесталости фраза на толико података (6762646 речи) могло би да траје изузетно дуго, али без прегледа података, чини се да је то неопходан трошак. Овде је корисна истраживачка анализа података, јер сте видели узорак тагова као што су `[' Пословно путовање  ', ' Самостални путник ', ' Једнокреветна соба ', ' Боравак од 5 ноћи ', ' Послато са мобилног уређаја ']`, можете почети да се питате да ли је могуће значајно смањити обраду коју морате да урадите. Срећом, јесте - али прво морате да следите неколико корака да бисте утврдили тагове од интереса.

### Филтрирање тагова

Запамтите да је циљ скупа података да додате сентимент и колоне које ће вам помоћи да изаберете најбољи хотел (за себе или можда за клијента који вам задаје задатак да направите бота за препоруке хотела). Морате се запитати да ли су тагови корисни или не у коначном скупу података. Ево једне интерпретације (ако вам је скуп података потребан из других разлога, различити тагови би могли остати/бити уклоњени):

1. Тип путовања је релевантан и треба да остане
2. Тип групе гостију је важан и треба да остане
3. Тип собе, апартмана или студија у којем је гост боравио је ирелевантан (сви хотели имају отприлике исте собе)
4. Уређај са којег је рецензија послата је ирелевантан
5. Број ноћи које је рецензент боравио *може* бити релевантан ако приписујете дужи боравак као знак да им се хотел више свидео, али је то натегнуто и вероватно ирелевантно

Укратко, **задржите 2 врсте тагова и уклоните остале**.

Прво, не желите да бројите тагове док нису у бољем формату, што значи уклањање угластих заграда и наводника. Ово можете урадити на неколико начина, али желите најбржи јер би обрада великог броја података могла да траје дуго. Срећом, pandas има једноставан начин за сваки од ових корака.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Сваки таг постаје нешто попут: `Пословно путовање, Самостални путник, Једнокреветна соба, Боравак од 5 ноћи, Послато са мобилног уређаја`. 

Следећи проблем је различит број колона у рецензијама. Неки редови имају 5 колона, неки 3, неки 6. Ово је резултат начина на који је скуп података креиран и тешко је поправити. Желите да добијете учесталост сваке фразе, али оне су у различитом редоследу у свакој рецензији, па број може бити нетачан, а хотел можда не добије таг који заслужује.

Уместо тога, искористићете различит редослед у своју корист, јер је сваки таг фраза са више речи, али је такође одвојен зарезом! Најједноставнији начин да то урадите је да креирате 6 привремених колона са сваким тагом уметнутим у колону која одговара његовом редоследу у тагу. Затим можете спојити 6 колона у једну велику колону и покренути `value_counts()` метод на резултујућој колони. Када то испишете, видећете да је било 2428 јединствених тагова. Ево малог узорка:

| Таг                            | Број   |
| ------------------------------ | ------ |
| Одмор                          | 417778 |
| Послато са мобилног уређаја    | 307640 |
| Пар                            | 252294 |
| Боравак од 1 ноћи              | 193645 |
| Боравак од 2 ноћи              | 133937 |
| Самостални путник              | 108545 |
| Боравак од 3 ноћи              | 95821  |
| Пословно путовање              | 82939  |
| Група                          | 65392  |
| Породица са малом децом        | 61015  |
| Боравак од 4 ноћи              | 47817  |
| Двокреветна соба               | 35207  |
| Стандардна двокреветна соба    | 32248  |
| Супериор двокреветна соба      | 31393  |
| Породица са старијом децом     | 26349  |
| Делукс двокреветна соба        | 24823  |
| Двокреветна или соба са два кревета | 22393 |
| Боравак од 5 ноћи              | 20845  |
| Стандардна двокреветна или соба са два кревета | 17483 |
| Класична двокреветна соба      | 16989  |
| Супериор двокреветна или соба са два кревета | 13570 |
| 2 собе                         | 12393  |

Неки од уобичајених тагова као што је `Послато са мобилног уређаја` нису нам од користи, па би било паметно уклонити их пре бројања учесталости фраза, али је то тако брза операција да их можете оставити и игнорисати.

### Уклањање тагова о дужини боравка

Уклањање ових тагова је први корак, што благо смањује укупан број тагова који се разматрају. Напомињемо да их не уклањате из скупа података, већ само одлучујете да их уклоните из разматрања као вредности које треба бројати/задржати у скупу рецензија.

| Дужина боравка | Број   |
| -------------- | ------ |
| Боравак од 1 ноћи | 193645 |
| Боравак од 2 ноћи | 133937 |
| Боравак од 3 ноћи | 95821  |
| Боравак од 4 ноћи | 47817  |
| Боравак од 5 ноћи | 20845  |
| Боравак од 6 ноћи | 9776   |
| Боравак од 7 ноћи | 7399   |
| Боравак од 8 ноћи | 2502   |
| Боравак од 9 ноћи | 1293   |
| ...              | ...    |

Постоји огромна разноврсност соба, апартмана, студија, апартмана и тако даље. Сви они значе отприлике исто и нису релевантни за вас, па их уклоните из разматрања.

| Тип собе                     | Број   |
| ---------------------------- | ------ |
| Двокреветна соба             | 35207  |
| Стандардна двокреветна соба  | 32248  |
| Супериор двокреветна соба    | 31393  |
| Делукс двокреветна соба      | 24823  |
| Двокреветна или соба са два кревета | 22393 |
| Стандардна двокреветна или соба са два кревета | 17483 |
| Класична двокреветна соба    | 16989  |
| Супериор двокреветна или соба са два кревета | 13570 |

На крају, и ово је одлично (јер није захтевало много обраде), остаћете са следећим *корисним* таговима:

| Таг                                           | Број   |
| --------------------------------------------- | ------ |
| Одмор                                         | 417778 |
| Пар                                           | 252294 |
| Самостални путник                             | 108545 |
| Пословно путовање                             | 82939  |
| Група (комбиновано са Путници са пријатељима) | 67535  |
| Породица са малом децом                       | 61015  |
| Породица са старијом децом                    | 26349  |
| Са кућним љубимцем                            | 1405   |

Могли бисте тврдити да је `Путници са пријатељима` исто што и `Група` мање-више, и било би фер комбиновати их као горе. Код за идентификовање исправних тагова је [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Последњи корак је креирање нових колона за сваки од ових тагова. Затим, за сваки ред рецензије, ако колона `Tag` одговара једној од нових колона, додајте 1, ако не, додајте 0. Коначни резултат ће бити број рецензената који су изабрали овај хотел (у агрегату) за, рецимо, пословно путовање у односу на одмор, или да доведу кућног љубимца, и то је корисна информација приликом препоруке хотела.

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

### Сачувајте ваш фајл

На крају, сачувајте скуп података у тренутном стању под новим именом.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Операције анализе сентимента

У овом последњем одељку, применићете анализу сентимента на колоне рецензија и сачувати резултате у скупу података.

## Вежба: учитајте и сачувајте филтриране податке

Напомињемо да сада учитавате филтрирани скуп података који је сачуван у претходном одељку, **не** оригинални скуп података.

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

### Уклањање стоп речи

Ако бисте покренули анализу сентимента на колонама негативних и позитивних рецензија, то би могло трајати дуго. Тестирано на моћном лаптопу са брзим CPU-ом, трајало је 12 - 14 минута у зависности од тога која библиотека за анализу сентимента је коришћена. То је (релативно) дуго, па је вредно истражити да ли се то може убрзати. 

Уклањање стоп речи, или уобичајених енглеских речи које не мењају сентимент реченице, је први корак. Уклањањем њих, анализа сентимента би требало да се брже изврши, али да не буде мање тачна (јер стоп речи не утичу на сентимент, али успоравају анализу). 

Најдужа негативна рецензија имала је 395 речи, али након уклањања стоп речи, има 195 речи.

Уклањање стоп речи је такође брза операција, уклањање стоп речи из 2 колоне рецензија преко 515,000 редова трајало је 3.3 секунде на тест уређају. Могло би трајати мало више или мање времена за вас у зависности од брзине вашег CPU-а, RAM-а, да ли имате SSD или не, и неких других фактора. Релативна краткоћа операције значи да ако побољшава време анализе сентимента, онда је вредно урадити.

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

### Извршавање анализе сентимента

Сада треба да израчунате анализу сентимента за обе колоне рецензија, негативне и позитивне, и сачувате резултат у 2 нове колоне. Тест сентимента ће бити упоређивање са оценом рецензента за исту рецензију. На пример, ако сентимент мисли да је негативна рецензија имала сентимент од
NLTK нуди различите анализаторе сентимента за учење, и можете их заменити и проверити да ли је сентимент тачнији или мање тачан. Овде се користи VADER анализа сентимента.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, јун 2014.

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

Касније у вашем програму, када будете спремни да израчунате сентимент, можете га применити на сваку рецензију на следећи начин:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Ово траје приближно 120 секунди на мом рачунару, али ће се време разликовати у зависности од рачунара. Ако желите да одштампате резултате и проверите да ли сентимент одговара рецензији:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Последња ствар коју треба урадити са фајлом пре него што га употребите у изазову је да га сачувате! Такође би требало да размислите о реорганизацији свих нових колона како би биле лакше за рад (за људе, то је козметичка промена).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Требало би да покренете цео код за [аналитички нотебук](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (након што сте покренули [ваш нотебук за филтрирање](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) како бисте генерисали фајл Hotel_Reviews_Filtered.csv).

Да резимирамо, кораци су:

1. Оригинални фајл података **Hotel_Reviews.csv** је истражен у претходној лекцији помоћу [експлорер нотебука](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv је филтриран помоћу [ноутбука за филтрирање](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), што резултира фајлом **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv је обрађен помоћу [ноутбука за анализу сентимента](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), што резултира фајлом **Hotel_Reviews_NLP.csv**
4. Користите Hotel_Reviews_NLP.csv у NLP изазову испод

### Закључак

Када сте почели, имали сте скуп података са колонама и подацима, али не све је могло бити проверено или коришћено. Истражили сте податке, филтрирали оно што вам није потребно, конвертовали тагове у нешто корисно, израчунали сопствене просеке, додали неке колоне за сентимент и, надамо се, научили нешто занимљиво о обради природног текста.

## [Квиз након предавања](https://ff-quizzes.netlify.app/en/ml/)

## Изазов

Сада када сте анализирали сентимент вашег скупа података, видите да ли можете користити стратегије које сте научили у овом курикулуму (кластерисање, можда?) како бисте утврдили обрасце у сентименту.

## Преглед и самостално учење

Погледајте [овај Learn модул](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) како бисте сазнали више и користили различите алате за истраживање сентимента у тексту.

## Задатак

[Испробајте други скуп података](assignment.md)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако се трудимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на његовом изворном језику треба сматрати ауторитативним извором. За критичне информације препоручује се професионални превод од стране људи. Не преузимамо одговорност за било каква погрешна тумачења или неспоразуме који могу настати услед коришћења овог превода.