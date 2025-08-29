<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T14:36:53+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ur"
}
-->
# ہوٹل کے جائزوں کے ساتھ جذباتی تجزیہ

اب جب کہ آپ نے ڈیٹا سیٹ کو تفصیل سے دریافت کر لیا ہے، اب وقت آ گیا ہے کہ کالموں کو فلٹر کریں اور پھر ڈیٹا سیٹ پر NLP تکنیکوں کا استعمال کریں تاکہ ہوٹلوں کے بارے میں نئی بصیرت حاصل کی جا سکے۔
## [لیکچر سے پہلے کا کوئز](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### فلٹرنگ اور جذباتی تجزیہ کے آپریشنز

جیسا کہ آپ نے شاید محسوس کیا ہوگا، ڈیٹا سیٹ میں کچھ مسائل ہیں۔ کچھ کالم غیر ضروری معلومات سے بھرے ہوئے ہیں، جبکہ دوسرے غلط معلوم ہوتے ہیں۔ اگر وہ درست بھی ہیں، تو یہ واضح نہیں ہے کہ انہیں کیسے شمار کیا گیا، اور آپ کے اپنے حسابات کے ذریعے جوابات کی آزادانہ تصدیق نہیں کی جا سکتی۔

## مشق: ڈیٹا پروسیسنگ میں مزید بہتری

ڈیٹا کو تھوڑا اور صاف کریں۔ ایسے کالم شامل کریں جو بعد میں کارآمد ہوں گے، دوسرے کالموں میں موجود اقدار کو تبدیل کریں، اور کچھ کالموں کو مکمل طور پر ہٹا دیں۔

1. ابتدائی کالم پروسیسنگ

   1. `lat` اور `lng` کو ہٹا دیں

   2. `Hotel_Address` کی قدروں کو درج ذیل قدروں سے تبدیل کریں (اگر پتہ شہر اور ملک کا نام رکھتا ہے، تو اسے صرف شہر اور ملک میں تبدیل کریں)۔

      ڈیٹا سیٹ میں صرف یہی شہر اور ممالک ہیں:

      ایمسٹرڈیم، نیدرلینڈز

      بارسلونا، اسپین

      لندن، برطانیہ

      میلان، اٹلی

      پیرس، فرانس

      ویانا، آسٹریا 

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

      اب آپ ملک کی سطح پر ڈیٹا کو تلاش کر سکتے ہیں:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | ایمسٹرڈیم، نیدرلینڈز  |    105     |
      | بارسلونا، اسپین        |    211     |
      | لندن، برطانیہ          |    400     |
      | میلان، اٹلی            |    162     |
      | پیرس، فرانس            |    458     |
      | ویانا، آسٹریا          |    158     |

2. ہوٹل میٹا-ریویو کالمز کو پروسیس کریں

  1. `Additional_Number_of_Scoring` کو ہٹا دیں

  1. `Total_Number_of_Reviews` کو اس ہوٹل کے اصل ڈیٹا سیٹ میں موجود جائزوں کی کل تعداد سے تبدیل کریں 

  1. `Average_Score` کو ہمارے اپنے حساب کردہ اسکور سے تبدیل کریں

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. جائزہ کالمز کو پروسیس کریں

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` اور `days_since_review` کو ہٹا دیں

   2. `Reviewer_Score`, `Negative_Review`, اور `Positive_Review` کو جوں کا توں رکھیں،
     
   3. `Tags` کو فی الحال رکھیں

     - ہم اگلے سیکشن میں ٹیگز پر مزید فلٹرنگ آپریشنز کریں گے اور پھر ٹیگز کو ہٹا دیں گے

4. جائزہ لینے والے کالمز کو پروسیس کریں

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` کو ہٹا دیں
  
  2. `Reviewer_Nationality` کو رکھیں

### ٹیگ کالمز

`Tag` کالم ایک مسئلہ ہے کیونکہ یہ ایک فہرست (متن کی شکل میں) ہے جو کالم میں محفوظ ہے۔ بدقسمتی سے، اس کالم میں ذیلی حصوں کی ترتیب اور تعداد ہمیشہ ایک جیسی نہیں ہوتی۔ 515,000 قطاروں، 1427 ہوٹلوں، اور ہر ایک کے مختلف اختیارات کے ساتھ، یہ انسان کے لیے صحیح جملے کی شناخت کرنا مشکل ہے۔ یہ وہ جگہ ہے جہاں NLP کارآمد ثابت ہوتا ہے۔ آپ متن کو اسکین کر سکتے ہیں، سب سے عام جملے تلاش کر سکتے ہیں، اور ان کی گنتی کر سکتے ہیں۔

بدقسمتی سے، ہم واحد الفاظ میں دلچسپی نہیں رکھتے، بلکہ کئی الفاظ پر مشتمل جملوں میں دلچسپی رکھتے ہیں (مثلاً *کاروباری سفر*)۔ اتنے زیادہ ڈیٹا (6762646 الفاظ) پر ایک کثیر الفاظ کی فریکوئنسی ڈسٹری بیوشن الگورتھم چلانا غیر معمولی وقت لے سکتا ہے، لیکن ڈیٹا کو دیکھے بغیر، یہ ضروری معلوم ہوتا ہے۔ یہ وہ جگہ ہے جہاں ایکسپلورٹری ڈیٹا اینالیسس کارآمد ہوتا ہے، کیونکہ آپ نے ٹیگز کا ایک نمونہ دیکھا ہے جیسے `['کاروباری سفر', 'اکیلا مسافر', 'سنگل روم', '5 راتیں قیام', 'موبائل ڈیوائس سے جمع کرایا گیا']`، آپ یہ پوچھنا شروع کر سکتے ہیں کہ کیا آپ کو پروسیسنگ کو بہت کم کرنے کا کوئی طریقہ ہے۔ خوش قسمتی سے، ایسا ممکن ہے - لیکن پہلے آپ کو دلچسپی کے ٹیگز کا تعین کرنے کے لیے چند اقدامات کرنے ہوں گے۔

### ٹیگز کو فلٹر کرنا

یاد رکھیں کہ ڈیٹا سیٹ کا مقصد جذبات اور ایسے کالمز شامل کرنا ہے جو آپ کو بہترین ہوٹل منتخب کرنے میں مدد دے سکیں (اپنے لیے یا شاید کسی کلائنٹ کے لیے جو آپ کو ہوٹل کی سفارش کرنے والے بوٹ بنانے کا کام دے رہا ہو)۔ آپ کو خود سے پوچھنا ہوگا کہ آیا ٹیگز حتمی ڈیٹا سیٹ میں کارآمد ہیں یا نہیں۔ یہاں ایک تشریح ہے (اگر آپ کو دوسرے مقاصد کے لیے ڈیٹا سیٹ کی ضرورت ہو تو مختلف ٹیگز کو شامل/خارج کیا جا سکتا ہے):

1. سفر کی قسم متعلقہ ہے، اور اسے رکھنا چاہیے
2. مہمانوں کے گروپ کی قسم اہم ہے، اور اسے رکھنا چاہیے
3. کمرے، سوئٹ، یا اسٹوڈیو کی قسم جس میں مہمان ٹھہرا تھا غیر متعلقہ ہے (تمام ہوٹلوں میں بنیادی طور پر ایک جیسے کمرے ہوتے ہیں)
4. وہ ڈیوائس جس پر جائزہ جمع کرایا گیا غیر متعلقہ ہے
5. جائزہ لینے والے نے کتنی راتیں قیام کیا *شاید* متعلقہ ہو اگر آپ طویل قیام کو ہوٹل کو زیادہ پسند کرنے سے منسوب کریں، لیکن یہ ایک قیاس ہے، اور شاید غیر متعلقہ ہو

خلاصہ یہ کہ، **2 قسم کے ٹیگز رکھیں اور باقی کو ہٹا دیں**۔

پہلے، آپ ٹیگز کی گنتی نہیں کرنا چاہتے جب تک کہ وہ بہتر فارمیٹ میں نہ ہوں، اس کا مطلب ہے کہ مربع بریکٹس اور کوٹس کو ہٹانا۔ آپ یہ کئی طریقوں سے کر سکتے ہیں، لیکن آپ تیز ترین طریقہ چاہتے ہیں کیونکہ زیادہ ڈیٹا کو پروسیس کرنے میں وقت لگ سکتا ہے۔ خوش قسمتی سے، pandas میں ان میں سے ہر ایک قدم کو انجام دینے کا ایک آسان طریقہ ہے۔

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ہر ٹیگ کچھ اس طرح بن جاتا ہے: `کاروباری سفر، اکیلا مسافر، سنگل روم، 5 راتیں قیام، موبائل ڈیوائس سے جمع کرایا گیا`۔

اگلا مسئلہ یہ ہے کہ کچھ جائزے، یا قطاریں، 5 کالمز رکھتی ہیں، کچھ 3، کچھ 6۔ یہ اس ڈیٹا سیٹ کے تخلیق کے طریقے کا نتیجہ ہے، اور اسے ٹھیک کرنا مشکل ہے۔ آپ ہر جملے کی فریکوئنسی گنتی حاصل کرنا چاہتے ہیں، لیکن وہ ہر جائزے میں مختلف ترتیب میں ہیں، اس لیے گنتی غلط ہو سکتی ہے، اور ہوٹل کو وہ ٹیگ نہیں مل سکتا جس کا وہ مستحق تھا۔

اس کے بجائے آپ مختلف ترتیب کو اپنے فائدے کے لیے استعمال کریں گے، کیونکہ ہر ٹیگ کئی الفاظ پر مشتمل ہے لیکن ایک کاما سے الگ بھی ہے! اس کا سب سے آسان طریقہ یہ ہے کہ 6 عارضی کالمز بنائیں اور ہر ٹیگ کو اس کے ترتیب کے مطابق کالم میں ڈالیں۔ پھر آپ ان 6 کالمز کو ایک بڑے کالم میں ضم کر سکتے ہیں اور نتیجے میں آنے والے کالم پر `value_counts()` طریقہ چلا سکتے ہیں۔ جب آپ اسے پرنٹ کریں گے، تو آپ دیکھیں گے کہ 2428 منفرد ٹیگز تھے۔ یہاں ایک چھوٹا نمونہ ہے:

| ٹیگ                            | گنتی  |
| ------------------------------ | ------ |
| تفریحی سفر                    | 417778 |
| موبائل ڈیوائس سے جمع کرایا گیا | 307640 |
| جوڑا                          | 252294 |
| 1 رات قیام                    | 193645 |
| 2 راتیں قیام                  | 133937 |
| اکیلا مسافر                   | 108545 |
| 3 راتیں قیام                  | 95821  |
| کاروباری سفر                 | 82939  |
| گروپ                          | 65392  |
| چھوٹے بچوں کے ساتھ خاندان    | 61015  |
| 4 راتیں قیام                  | 47817  |
| ڈبل روم                       | 35207  |
| اسٹینڈرڈ ڈبل روم              | 32248  |
| سپیریئر ڈبل روم               | 31393  |
| بڑے بچوں کے ساتھ خاندان      | 26349  |
| ڈیلکس ڈبل روم                 | 24823  |
| ڈبل یا ٹوئن روم               | 22393  |
| 5 راتیں قیام                  | 20845  |
| اسٹینڈرڈ ڈبل یا ٹوئن روم      | 17483  |
| کلاسک ڈبل روم                 | 16989  |
| سپیریئر ڈبل یا ٹوئن روم       | 13570  |
| 2 کمرے                        | 12393  |

کچھ عام ٹیگز جیسے `موبائل ڈیوائس سے جمع کرایا گیا` ہمارے لیے بے کار ہیں، اس لیے یہ ایک ہوشیار بات ہو سکتی ہے کہ انہیں جملے کی وقوعہ گنتی سے پہلے ہٹا دیا جائے، لیکن یہ ایک تیز آپریشن ہے، آپ انہیں چھوڑ سکتے ہیں اور نظر انداز کر سکتے ہیں۔

### قیام کی مدت کے ٹیگز کو ہٹانا

ان ٹیگز کو ہٹانا پہلا قدم ہے، یہ غور کیے جانے والے ٹیگز کی کل تعداد کو تھوڑا کم کر دیتا ہے۔ نوٹ کریں کہ آپ انہیں ڈیٹا سیٹ سے نہیں ہٹاتے، صرف انہیں جائزوں کے ڈیٹا سیٹ میں گنتی/رکھنے کے لیے غور سے ہٹاتے ہیں۔

| قیام کی مدت   | گنتی  |
| ------------ | ------ |
| 1 رات قیام   | 193645 |
| 2 راتیں قیام | 133937 |
| 3 راتیں قیام | 95821  |
| 4 راتیں قیام | 47817  |
| 5 راتیں قیام | 20845  |
| 6 راتیں قیام | 9776   |
| 7 راتیں قیام | 7399   |
| 8 راتیں قیام | 2502   |
| 9 راتیں قیام | 1293   |
| ...          | ...    |

کمرے، سوئٹ، اسٹوڈیوز، اپارٹمنٹس وغیرہ کی ایک بڑی قسم ہے۔ یہ سب تقریباً ایک ہی چیز کا مطلب رکھتے ہیں اور آپ کے لیے غیر متعلقہ ہیں، اس لیے انہیں غور سے ہٹا دیں۔

| کمرے کی قسم                  | گنتی |
| ----------------------------- | ----- |
| ڈبل روم                      | 35207 |
| اسٹینڈرڈ ڈبل روم             | 32248 |
| سپیریئر ڈبل روم              | 31393 |
| ڈیلکس ڈبل روم                | 24823 |
| ڈبل یا ٹوئن روم              | 22393 |
| اسٹینڈرڈ ڈبل یا ٹوئن روم     | 17483 |
| کلاسک ڈبل روم                | 16989 |
| سپیریئر ڈبل یا ٹوئن روم      | 13570 |

آخر میں، اور یہ خوشگوار ہے (کیونکہ اس میں زیادہ پروسیسنگ نہیں ہوئی)، آپ کے پاس درج ذیل *کارآمد* ٹیگز باقی رہ جائیں گے:

| ٹیگ                                           | گنتی  |
| --------------------------------------------- | ------ |
| تفریحی سفر                                   | 417778 |
| جوڑا                                         | 252294 |
| اکیلا مسافر                                  | 108545 |
| کاروباری سفر                                | 82939  |
| گروپ (دوستوں کے ساتھ مسافر)                 | 67535  |
| چھوٹے بچوں کے ساتھ خاندان                   | 61015  |
| بڑے بچوں کے ساتھ خاندان                     | 26349  |
| پالتو جانور کے ساتھ                         | 1405   |

آپ یہ دلیل دے سکتے ہیں کہ `دوستوں کے ساتھ مسافر` اور `گروپ` تقریباً ایک ہی چیز ہیں، اور انہیں اوپر کی طرح یکجا کرنا مناسب ہوگا۔ صحیح ٹیگز کی شناخت کے لیے کوڈ [ٹیگز نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) میں موجود ہے۔

آخری قدم یہ ہے کہ ان ٹیگز میں سے ہر ایک کے لیے نئے کالمز بنائیں۔ پھر، ہر جائزہ قطار کے لیے، اگر `Tag` کالم نئے کالمز میں سے کسی سے میل کھاتا ہے، تو 1 شامل کریں، اگر نہیں، تو 0 شامل کریں۔ نتیجہ یہ ہوگا کہ کتنے جائزہ لینے والوں نے اس ہوٹل کو (مجموعی طور پر) کاروبار یا تفریح کے لیے منتخب کیا، یا پالتو جانور کے ساتھ آنے کے لیے، اور یہ ہوٹل کی سفارش کرتے وقت کارآمد معلومات ہوگی۔

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

### اپنی فائل محفوظ کریں

آخر میں، ڈیٹا سیٹ کو موجودہ حالت میں ایک نئے نام کے ساتھ محفوظ کریں۔

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## جذباتی تجزیہ کے آپریشنز

اس آخری حصے میں، آپ جائزہ کالمز پر جذباتی تجزیہ کریں گے اور نتائج کو ایک ڈیٹا سیٹ میں محفوظ کریں گے۔

## مشق: فلٹر شدہ ڈیٹا لوڈ کریں اور محفوظ کریں

نوٹ کریں کہ اب آپ فلٹر شدہ ڈیٹا سیٹ لوڈ کر رہے ہیں جو پچھلے حصے میں محفوظ کیا گیا تھا، **اصل ڈیٹا سیٹ نہیں**۔

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

### اسٹاپ ورڈز کو ہٹانا

اگر آپ منفی اور مثبت جائزہ کالمز پر جذباتی تجزیہ چلائیں، تو اس میں کافی وقت لگ سکتا ہے۔ ایک طاقتور ٹیسٹ لیپ ٹاپ پر، جس میں تیز CPU ہو، یہ 12 - 14 منٹ لگے، اس پر منحصر ہے کہ کون سی جذباتی لائبریری استعمال کی گئی۔ یہ ایک (نسبتاً) طویل وقت ہے، اس لیے یہ جانچنے کے قابل ہے کہ کیا اسے تیز کیا جا سکتا ہے۔

اسٹاپ ورڈز، یا عام انگریزی الفاظ جو جملے کے جذبات کو تبدیل نہیں کرتے، کو ہٹانا پہلا قدم ہے۔ انہیں ہٹانے سے جذباتی تجزیہ تیز ہونا چاہیے، لیکن کم درست نہیں (کیونکہ اسٹاپ ورڈز جذبات کو متاثر نہیں کرتے، لیکن وہ تجزیہ کو سست کر دیتے ہیں)۔

سب سے طویل منفی جائزہ 395 الفاظ کا تھا، لیکن اسٹاپ ورڈز کو ہٹانے کے بعد، یہ 195 الفاظ کا رہ گیا۔

اسٹاپ ورڈز کو ہٹانا بھی ایک تیز آپریشن ہے، 2 جائزہ کالمز سے 515,000 قطاروں پر اسٹاپ ورڈز کو ہٹانے میں 3.3 سیکنڈ لگے۔ آپ کے لیے یہ وقت آپ کے ڈیوائس کی CPU اسپیڈ، RAM، SSD کی موجودگی، اور دیگر عوامل پر منحصر ہو کر تھوڑا زیادہ یا کم ہو سکتا ہے۔ آپریشن کی نسبتاً مختصر مدت کا مطلب ہے کہ اگر یہ جذباتی تجزیہ کے وقت کو بہتر بناتا ہے، تو یہ کرنے کے قابل ہے۔

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

### جذباتی تجزیہ کرنا
اب آپ کو دونوں منفی اور مثبت جائزہ کالمز کے لیے جذباتی تجزیہ کا حساب لگانا چاہیے، اور نتیجہ کو 2 نئے کالمز میں محفوظ کرنا چاہیے۔ جذبات کی جانچ کا معیار یہ ہوگا کہ اسے جائزہ لینے والے کے اس جائزے کے اسکور سے موازنہ کیا جائے۔ مثال کے طور پر، اگر جذباتی تجزیہ یہ سمجھتا ہے کہ منفی جائزے کا جذبات 1 (انتہائی مثبت جذبات) ہے اور مثبت جائزے کا جذبات بھی 1 ہے، لیکن جائزہ لینے والے نے ہوٹل کو سب سے کم ممکنہ اسکور دیا، تو یا تو جائزے کا متن اس اسکور سے میل نہیں کھاتا، یا جذباتی تجزیہ کرنے والا جذبات کو صحیح طور پر پہچاننے میں ناکام رہا۔ آپ کو توقع کرنی چاہیے کہ کچھ جذباتی اسکور بالکل غلط ہوں گے، اور اکثر یہ قابل وضاحت ہوگا، جیسے کہ جائزہ انتہائی طنزیہ ہو سکتا ہے "یقیناً مجھے بغیر ہیٹنگ والے کمرے میں سونا بہت پسند آیا" اور جذباتی تجزیہ کرنے والا اسے مثبت جذبات سمجھتا ہے، حالانکہ ایک انسان اسے پڑھ کر جان سکتا ہے کہ یہ طنز ہے۔

NLTK مختلف جذباتی تجزیہ کرنے والے فراہم کرتا ہے جن سے سیکھا جا سکتا ہے، اور آپ انہیں تبدیل کر کے دیکھ سکتے ہیں کہ جذباتی تجزیہ زیادہ یا کم درست ہے۔ یہاں VADER جذباتی تجزیہ استعمال کیا گیا ہے۔

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

بعد میں اپنے پروگرام میں جب آپ جذبات کا حساب لگانے کے لیے تیار ہوں، تو آپ اسے ہر جائزے پر اس طرح لاگو کر سکتے ہیں:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

یہ میرے کمپیوٹر پر تقریباً 120 سیکنڈ لیتا ہے، لیکن یہ ہر کمپیوٹر پر مختلف ہوگا۔ اگر آپ نتائج کو پرنٹ کرنا چاہتے ہیں اور دیکھنا چاہتے ہیں کہ جذبات جائزے سے میل کھاتے ہیں:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

فائل کے ساتھ چیلنج میں استعمال کرنے سے پہلے آخری کام یہ ہے کہ اسے محفوظ کریں! آپ کو اپنے نئے کالمز کو دوبارہ ترتیب دینے پر بھی غور کرنا چاہیے تاکہ وہ کام کرنے میں آسان ہوں (یہ ایک ظاہری تبدیلی ہے)۔

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

آپ کو [تجزیہ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) کے پورے کوڈ کو چلانا چاہیے (جب آپ نے [فلٹرنگ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) کو چلایا ہو تاکہ Hotel_Reviews_Filtered.csv فائل تیار ہو جائے)۔

جائزہ لینے کے لیے، مراحل یہ ہیں:

1. اصل ڈیٹاسیٹ فائل **Hotel_Reviews.csv** کو پچھلے سبق میں [ایکسپلورر نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) کے ساتھ دریافت کیا گیا ہے۔
2. Hotel_Reviews.csv کو [فلٹرنگ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) کے ذریعے فلٹر کیا گیا، جس کے نتیجے میں **Hotel_Reviews_Filtered.csv** حاصل ہوا۔
3. Hotel_Reviews_Filtered.csv کو [جذباتی تجزیہ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) کے ذریعے پروسیس کیا گیا، جس کے نتیجے میں **Hotel_Reviews_NLP.csv** حاصل ہوا۔
4. Hotel_Reviews_NLP.csv کو نیچے دیے گئے NLP چیلنج میں استعمال کریں۔

### نتیجہ

جب آپ نے شروع کیا، تو آپ کے پاس کالمز اور ڈیٹا کے ساتھ ایک ڈیٹاسیٹ تھا لیکن اس میں سے سب کچھ تصدیق یا استعمال نہیں کیا جا سکتا تھا۔ آپ نے ڈیٹا کو دریافت کیا، غیر ضروری حصے کو فلٹر کیا، ٹیگز کو کچھ مفید چیزوں میں تبدیل کیا، اپنے اوسط کا حساب لگایا، کچھ جذباتی کالمز شامل کیے اور امید ہے کہ قدرتی متن کو پروسیس کرنے کے بارے میں دلچسپ چیزیں سیکھی ہوں گی۔

## [لیکچر کے بعد کا کوئز](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## چیلنج

اب جب کہ آپ نے اپنے ڈیٹاسیٹ کا جذباتی تجزیہ کر لیا ہے، دیکھیں کہ آپ اس نصاب میں سیکھی گئی حکمت عملیوں (شاید کلسٹرنگ؟) کو استعمال کر کے جذبات کے ارد گرد پیٹرنز کا تعین کر سکتے ہیں۔

## جائزہ اور خود مطالعہ

[یہ لرن ماڈیول](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) لیں تاکہ مزید سیکھ سکیں اور مختلف ٹولز کا استعمال کر کے متن میں جذبات کو دریافت کریں۔

## اسائنمنٹ 

[ایک مختلف ڈیٹاسیٹ آزمائیں](assignment.md)

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا عدم درستگی ہو سکتی ہیں۔ اصل دستاویز، جو اس کی اصل زبان میں ہے، کو مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے لیے ہم ذمہ دار نہیں ہیں۔