<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T14:35:51+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ar"
}
-->
# تحليل المشاعر باستخدام تقييمات الفنادق

الآن بعد أن قمت باستكشاف مجموعة البيانات بالتفصيل، حان الوقت لتصفية الأعمدة واستخدام تقنيات معالجة اللغة الطبيعية (NLP) على مجموعة البيانات للحصول على رؤى جديدة حول الفنادق.

## [اختبار ما قبل المحاضرة](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### عمليات التصفية وتحليل المشاعر

كما لاحظت على الأرجح، تحتوي مجموعة البيانات على بعض المشكلات. بعض الأعمدة مليئة بمعلومات غير مفيدة، وأخرى تبدو غير صحيحة. وإذا كانت صحيحة، فمن غير الواضح كيف تم حسابها، ولا يمكن التحقق من الإجابات بشكل مستقل من خلال حساباتك الخاصة.

## تمرين: المزيد من معالجة البيانات

قم بتنظيف البيانات قليلاً. أضف أعمدة ستكون مفيدة لاحقًا، غيّر القيم في أعمدة أخرى، واحذف أعمدة معينة تمامًا.

1. معالجة الأعمدة الأولية

   1. احذف العمودين `lat` و`lng`.

   2. استبدل قيم `Hotel_Address` بالقيم التالية (إذا كان العنوان يحتوي على اسم المدينة والبلد، قم بتغييره ليشمل فقط المدينة والبلد).

      المدن والبلدان الوحيدة الموجودة في مجموعة البيانات هي:

      أمستردام، هولندا  
      برشلونة، إسبانيا  
      لندن، المملكة المتحدة  
      ميلانو، إيطاليا  
      باريس، فرنسا  
      فيينا، النمسا  

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

      الآن يمكنك استعلام بيانات على مستوى البلد:

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

2. معالجة أعمدة المراجعات العامة للفنادق

   1. احذف العمود `Additional_Number_of_Scoring`.

   2. استبدل `Total_Number_of_Reviews` بعدد المراجعات الفعلي لهذا الفندق الموجود في مجموعة البيانات.

   3. استبدل `Average_Score` بالنتيجة التي قمت بحسابها بنفسك.

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. معالجة أعمدة المراجعات

   1. احذف الأعمدة `Review_Total_Negative_Word_Counts`، `Review_Total_Positive_Word_Counts`، `Review_Date` و`days_since_review`.

   2. احتفظ بالأعمدة `Reviewer_Score`، `Negative_Review`، و`Positive_Review` كما هي.

   3. احتفظ بـ `Tags` في الوقت الحالي.

      - سنقوم بإجراء عمليات تصفية إضافية على العلامات في القسم التالي، ثم سيتم حذف العلامات.

4. معالجة أعمدة المراجعين

   1. احذف العمود `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. احتفظ بـ `Reviewer_Nationality`.

### أعمدة العلامات

عمود `Tag` يمثل مشكلة لأنه عبارة عن قائمة (في شكل نصي) مخزنة في العمود. لسوء الحظ، فإن ترتيب وعدد الأقسام الفرعية في هذا العمود ليس دائمًا نفسه. من الصعب على الإنسان تحديد العبارات الصحيحة التي يجب التركيز عليها، لأن هناك 515,000 صف و1427 فندقًا، ولكل منها خيارات مختلفة قليلاً يمكن أن يختارها المراجع. هنا يأتي دور معالجة اللغة الطبيعية (NLP). يمكنك مسح النص والعثور على العبارات الأكثر شيوعًا وعدّها.

لسوء الحظ، نحن لسنا مهتمين بالكلمات الفردية، بل بالعبارات متعددة الكلمات (مثل *رحلة عمل*). تشغيل خوارزمية توزيع التكرار للعبارات متعددة الكلمات على هذا الكم الهائل من البيانات (6762646 كلمة) قد يستغرق وقتًا طويلاً للغاية، ولكن دون النظر إلى البيانات، يبدو أن ذلك ضروري. هنا تكون فائدة تحليل البيانات الاستكشافي، لأنه من خلال عينة من العلامات مثل `['رحلة عمل', 'مسافر منفرد', 'غرفة فردية', 'أقام 5 ليالٍ', 'تم الإرسال من جهاز محمول']`، يمكنك البدء في التساؤل عما إذا كان من الممكن تقليل المعالجة بشكل كبير. لحسن الحظ، يمكن ذلك - ولكن أولاً تحتاج إلى اتباع بعض الخطوات لتحديد العلامات ذات الأهمية.

### تصفية العلامات

تذكر أن الهدف من مجموعة البيانات هو إضافة مشاعر وأعمدة تساعدك في اختيار أفضل فندق (لنفسك أو ربما لمهمة عميل يطلب منك إنشاء روبوت توصيات للفنادق). تحتاج إلى أن تسأل نفسك ما إذا كانت العلامات مفيدة أم لا في مجموعة البيانات النهائية. هنا تفسير واحد (إذا كنت بحاجة إلى مجموعة البيانات لأسباب أخرى، قد تختلف العلامات التي تبقى/تُزال):

1. نوع الرحلة ذو صلة، ويجب أن يبقى.
2. نوع مجموعة الضيوف مهم، ويجب أن يبقى.
3. نوع الغرفة أو الجناح أو الاستوديو الذي أقام فيه الضيف غير ذي صلة (جميع الفنادق تحتوي على غرف متشابهة تقريبًا).
4. الجهاز الذي تم إرسال المراجعة منه غير ذي صلة.
5. عدد الليالي التي أقامها المراجع *قد* يكون ذا صلة إذا كنت تعتقد أن الإقامات الأطول تعني إعجابهم بالفندق أكثر، ولكنه احتمال ضعيف، وربما غير ذي صلة.

باختصار، **احتفظ بنوعين من العلامات وأزل الباقي**.

أولاً، لا تريد عد العلامات حتى تكون في تنسيق أفضل، مما يعني إزالة الأقواس المربعة وعلامات الاقتباس. يمكنك القيام بذلك بعدة طرق، ولكنك تريد الأسرع لأن معالجة الكثير من البيانات قد تستغرق وقتًا طويلاً. لحسن الحظ، يوفر pandas طريقة سهلة لكل خطوة من هذه الخطوات.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

تصبح كل علامة شيئًا مثل: `رحلة عمل، مسافر منفرد، غرفة فردية، أقام 5 ليالٍ، تم الإرسال من جهاز محمول`.

بعد ذلك نجد مشكلة. بعض المراجعات، أو الصفوف، تحتوي على 5 أعمدة، وبعضها 3، وبعضها 6. هذا نتيجة لكيفية إنشاء مجموعة البيانات، ومن الصعب إصلاحها. تريد الحصول على عدد تكرار لكل عبارة، ولكنها في ترتيب مختلف في كل مراجعة، لذا قد يكون العدد غير دقيق، وقد لا يحصل الفندق على علامة مستحقة.

بدلاً من ذلك، ستستخدم الترتيب المختلف لصالحك، لأن كل علامة هي عبارة متعددة الكلمات ولكنها مفصولة بفاصلة! أبسط طريقة للقيام بذلك هي إنشاء 6 أعمدة مؤقتة مع إدخال كل علامة في العمود المقابل لترتيبها في العلامة. يمكنك بعد ذلك دمج الأعمدة الستة في عمود واحد كبير وتشغيل طريقة `value_counts()` على العمود الناتج. عند طباعته، سترى أن هناك 2428 علامة فريدة. هنا عينة صغيرة:

| Tag                            | Count  |
| ------------------------------ | ------ |
| رحلة ترفيهية                  | 417778 |
| تم الإرسال من جهاز محمول      | 307640 |
| زوجان                          | 252294 |
| أقام ليلة واحدة               | 193645 |
| أقام ليلتين                   | 133937 |
| مسافر منفرد                   | 108545 |
| أقام 3 ليالٍ                  | 95821  |
| رحلة عمل                      | 82939  |
| مجموعة                        | 65392  |
| عائلة مع أطفال صغار           | 61015  |
| أقام 4 ليالٍ                  | 47817  |
| غرفة مزدوجة                   | 35207  |
| غرفة مزدوجة قياسية            | 32248  |
| غرفة مزدوجة فاخرة             | 31393  |
| عائلة مع أطفال كبار           | 26349  |
| غرفة مزدوجة ديلوكس            | 24823  |
| غرفة مزدوجة أو توأم           | 22393  |
| أقام 5 ليالٍ                  | 20845  |
| غرفة مزدوجة قياسية أو توأم    | 17483  |
| غرفة مزدوجة كلاسيكية          | 16989  |
| غرفة مزدوجة فاخرة أو توأم     | 13570  |
| غرفتان                         | 12393  |

بعض العلامات الشائعة مثل `تم الإرسال من جهاز محمول` ليست ذات فائدة لنا، لذا قد يكون من الذكاء إزالتها قبل عد تكرار العبارات، ولكنها عملية سريعة جدًا بحيث يمكنك تركها وتجاهلها.

### إزالة علامات مدة الإقامة

إزالة هذه العلامات هي الخطوة الأولى، حيث تقلل العدد الإجمالي للعلامات التي يجب أخذها في الاعتبار قليلاً. لاحظ أنك لا تزيلها من مجموعة البيانات، بل تختار فقط إزالتها من الاعتبار كقيم للعد/الحفظ في مجموعة بيانات المراجعات.

| مدة الإقامة   | العدد  |
| ------------- | ------ |
| أقام ليلة واحدة | 193645 |
| أقام ليلتين   | 133937 |
| أقام 3 ليالٍ  | 95821  |
| أقام 4 ليالٍ  | 47817  |
| أقام 5 ليالٍ  | 20845  |
| أقام 6 ليالٍ  | 9776   |
| أقام 7 ليالٍ  | 7399   |
| أقام 8 ليالٍ  | 2502   |
| أقام 9 ليالٍ  | 1293   |
| ...           | ...    |

هناك مجموعة كبيرة من الغرف، الأجنحة، الاستوديوهات، الشقق وما إلى ذلك. جميعها تعني تقريبًا نفس الشيء وليست ذات صلة بك، لذا قم بإزالتها من الاعتبار.

| نوع الغرفة                  | العدد |
| --------------------------- | ----- |
| غرفة مزدوجة                | 35207 |
| غرفة مزدوجة قياسية         | 32248 |
| غرفة مزدوجة فاخرة          | 31393 |
| غرفة مزدوجة ديلوكس         | 24823 |
| غرفة مزدوجة أو توأم        | 22393 |
| غرفة مزدوجة قياسية أو توأم | 17483 |
| غرفة مزدوجة كلاسيكية       | 16989 |
| غرفة مزدوجة فاخرة أو توأم  | 13570 |

أخيرًا، وهذا أمر رائع (لأنه لم يتطلب الكثير من المعالجة على الإطلاق)، ستبقى مع العلامات التالية *المفيدة*:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| رحلة ترفيهية                                  | 417778 |
| زوجان                                         | 252294 |
| مسافر منفرد                                   | 108545 |
| رحلة عمل                                      | 82939  |
| مجموعة (تم دمجها مع مسافرين مع أصدقاء)       | 67535  |
| عائلة مع أطفال صغار                           | 61015  |
| عائلة مع أطفال كبار                           | 26349  |
| مع حيوان أليف                                 | 1405   |

يمكنك القول إن `مسافرين مع أصدقاء` تعني نفس الشيء تقريبًا مثل `مجموعة`، وسيكون من العدل دمج الاثنين كما هو موضح أعلاه. الكود لتحديد العلامات الصحيحة موجود في [دفتر العلامات](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

الخطوة الأخيرة هي إنشاء أعمدة جديدة لكل من هذه العلامات. ثم، لكل صف مراجعة، إذا كان عمود `Tag` يطابق أحد الأعمدة الجديدة، أضف 1، وإذا لم يكن كذلك، أضف 0. النتيجة النهائية ستكون عدداً يوضح عدد المراجعين الذين اختاروا هذا الفندق (بالمجمل) لأغراض مثل العمل مقابل الترفيه، أو لإحضار حيوان أليف، وهذه معلومات مفيدة عند التوصية بفندق.

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

### حفظ الملف

أخيرًا، احفظ مجموعة البيانات كما هي الآن باسم جديد.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## عمليات تحليل المشاعر

في هذا القسم الأخير، ستقوم بتطبيق تحليل المشاعر على أعمدة المراجعات وحفظ النتائج في مجموعة بيانات.

## تمرين: تحميل وحفظ البيانات المصفاة

لاحظ أنك الآن تقوم بتحميل مجموعة البيانات المصفاة التي تم حفظها في القسم السابق، **وليس** مجموعة البيانات الأصلية.

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

### إزالة الكلمات الشائعة

إذا قمت بتشغيل تحليل المشاعر على أعمدة المراجعات السلبية والإيجابية، فقد يستغرق ذلك وقتًا طويلاً. في اختبار على جهاز لابتوب قوي بمعالج سريع، استغرق الأمر 12-14 دقيقة اعتمادًا على مكتبة تحليل المشاعر المستخدمة. هذا وقت طويل نسبيًا، لذا يستحق التحقيق إذا كان بالإمكان تسريعه.

إزالة الكلمات الشائعة، أو الكلمات الإنجليزية الشائعة التي لا تؤثر على مشاعر الجملة، هي الخطوة الأولى. من خلال إزالتها، يجب أن يعمل تحليل المشاعر بشكل أسرع، دون أن يكون أقل دقة (لأن الكلمات الشائعة لا تؤثر على المشاعر، لكنها تبطئ التحليل).

أطول مراجعة سلبية كانت تحتوي على 395 كلمة، ولكن بعد إزالة الكلمات الشائعة، أصبحت تحتوي على 195 كلمة.

إزالة الكلمات الشائعة هي أيضًا عملية سريعة، حيث استغرقت إزالة الكلمات الشائعة من عمودين للمراجعات عبر 515,000 صف 3.3 ثانية على جهاز الاختبار. قد تستغرق وقتًا أطول أو أقل قليلاً حسب سرعة وحدة المعالجة المركزية لجهازك، وذاكرة الوصول العشوائي، وما إذا كان لديك SSD أم لا، وبعض العوامل الأخرى. قصر مدة العملية يعني أنه إذا حسّنت وقت تحليل المشاعر، فإنها تستحق التنفيذ.

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

### تنفيذ تحليل المشاعر
الآن يجب عليك حساب تحليل المشاعر لكل من عمودي المراجعات السلبية والإيجابية، وتخزين النتيجة في عمودين جديدين. سيكون اختبار المشاعر هو مقارنتها بتقييم المراجع لنفس المراجعة. على سبيل المثال، إذا أظهرت المشاعر أن المراجعة السلبية كانت ذات مشاعر إيجابية للغاية (1) والمراجعة الإيجابية كانت أيضًا (1)، ولكن المراجع أعطى الفندق أدنى تقييم ممكن، فهذا يعني أن نص المراجعة لا يتطابق مع التقييم، أو أن محلل المشاعر لم يتمكن من التعرف على المشاعر بشكل صحيح. يجب أن تتوقع أن تكون بعض نتائج تحليل المشاعر خاطئة تمامًا، وغالبًا ما يكون ذلك قابلاً للتفسير، على سبيل المثال قد تكون المراجعة ساخرة للغاية: "بالطبع أحببت النوم في غرفة بدون تدفئة"، وقد يعتقد محلل المشاعر أن هذا شعور إيجابي، على الرغم من أن الإنسان الذي يقرأها سيعرف أنها سخرية.

يوفر NLTK محللات مشاعر مختلفة للتعلم منها، ويمكنك استبدالها ورؤية ما إذا كانت المشاعر أكثر أو أقل دقة. يتم استخدام تحليل المشاعر VADER هنا.

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

لاحقًا في برنامجك عندما تكون مستعدًا لحساب المشاعر، يمكنك تطبيقها على كل مراجعة كما يلي:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

يستغرق هذا حوالي 120 ثانية على جهازي، ولكن قد يختلف على كل جهاز. إذا كنت تريد طباعة النتائج ورؤية ما إذا كانت المشاعر تتطابق مع المراجعة:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

آخر شيء يجب القيام به مع الملف قبل استخدامه في التحدي هو حفظه! يجب عليك أيضًا التفكير في إعادة ترتيب جميع الأعمدة الجديدة بحيث تكون سهلة الاستخدام (بالنسبة للإنسان، هذا تغيير تجميلي).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

يجب عليك تشغيل الكود بالكامل لـ [دفتر التحليل](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (بعد تشغيل [دفتر التصفية](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) لإنشاء ملف Hotel_Reviews_Filtered.csv).

لمراجعة الخطوات:

1. تم استكشاف ملف البيانات الأصلي **Hotel_Reviews.csv** في الدرس السابق باستخدام [دفتر الاستكشاف](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. تم تصفية Hotel_Reviews.csv باستخدام [دفتر التصفية](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) مما نتج عنه **Hotel_Reviews_Filtered.csv**
3. تم معالجة Hotel_Reviews_Filtered.csv باستخدام [دفتر تحليل المشاعر](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) مما نتج عنه **Hotel_Reviews_NLP.csv**
4. استخدم Hotel_Reviews_NLP.csv في تحدي معالجة اللغة الطبيعية أدناه

### الخلاصة

عندما بدأت، كان لديك مجموعة بيانات تحتوي على أعمدة وبيانات ولكن لم يكن بالإمكان التحقق من جميعها أو استخدامها. لقد استكشفت البيانات، وقمت بتصفية ما لا تحتاجه، وحولت العلامات إلى شيء مفيد، وحسبت متوسطاتك الخاصة، وأضفت بعض الأعمدة الخاصة بالمشاعر، ومن المحتمل أنك تعلمت بعض الأشياء المثيرة للاهتمام حول معالجة النصوص الطبيعية.

## [اختبار ما بعد المحاضرة](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## التحدي

الآن بعد أن قمت بتحليل مجموعة البيانات الخاصة بك للمشاعر، حاول استخدام الاستراتيجيات التي تعلمتها في هذا المنهج (مثل التجميع) لتحديد الأنماط المتعلقة بالمشاعر.

## المراجعة والدراسة الذاتية

خذ [هذا المقرر التعليمي](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) لتتعلم المزيد واستخدام أدوات مختلفة لاستكشاف المشاعر في النصوص.

## الواجب

[جرب مجموعة بيانات مختلفة](assignment.md)

---

**إخلاء المسؤولية**:  
تم ترجمة هذا المستند باستخدام خدمة الترجمة بالذكاء الاصطناعي [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية المصدر الرسمي. للحصول على معلومات حاسمة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.