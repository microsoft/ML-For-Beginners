<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-06T09:03:36+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ur"
}
-->
# ہوٹل کے جائزوں کے ساتھ جذباتی تجزیہ

اب جب کہ آپ نے ڈیٹا سیٹ کو تفصیل سے دریافت کر لیا ہے، اب وقت آ گیا ہے کہ کالموں کو فلٹر کریں اور پھر ڈیٹا سیٹ پر NLP تکنیکوں کا استعمال کریں تاکہ ہوٹلوں کے بارے میں نئی بصیرت حاصل کی جا سکے۔

## [لیکچر سے پہلے کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

### فلٹرنگ اور جذباتی تجزیہ کی کارروائیاں

جیسا کہ آپ نے شاید محسوس کیا ہوگا، ڈیٹا سیٹ میں کچھ مسائل ہیں۔ کچھ کالم غیر ضروری معلومات سے بھرے ہوئے ہیں، جبکہ دوسرے غلط معلوم ہوتے ہیں۔ اگر وہ درست بھی ہیں، تو یہ واضح نہیں ہے کہ انہیں کیسے شمار کیا گیا، اور آپ کے اپنے حسابات سے جوابات کی آزادانہ تصدیق نہیں کی جا سکتی۔

## مشق: ڈیٹا پروسیسنگ میں مزید بہتری

ڈیٹا کو تھوڑا اور صاف کریں۔ ایسے کالم شامل کریں جو بعد میں کارآمد ہوں گے، دوسرے کالموں میں موجود اقدار کو تبدیل کریں، اور کچھ کالموں کو مکمل طور پر ہٹا دیں۔

1. ابتدائی کالم پروسیسنگ

   1. `lat` اور `lng` کو ہٹا دیں۔

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

2. ہوٹل میٹا-ریویو کالموں کی پروسیسنگ

   1. `Additional_Number_of_Scoring` کو ہٹا دیں۔

   2. `Total_Number_of_Reviews` کو اس ہوٹل کے ڈیٹا سیٹ میں موجود جائزوں کی کل تعداد سے تبدیل کریں۔

   3. `Average_Score` کو اپنے حساب سے شمار کردہ اسکور سے تبدیل کریں۔

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. جائزہ کالموں کی پروسیسنگ

   1. `Review_Total_Negative_Word_Counts`، `Review_Total_Positive_Word_Counts`، `Review_Date` اور `days_since_review` کو ہٹا دیں۔

   2. `Reviewer_Score`، `Negative_Review`، اور `Positive_Review` کو جوں کا توں رکھیں۔

   3. `Tags` کو فی الحال رکھیں۔

      - ہم اگلے حصے میں ٹیگز پر مزید فلٹرنگ کریں گے اور پھر انہیں ہٹا دیں گے۔

4. جائزہ لینے والے کالموں کی پروسیسنگ

   1. `Total_Number_of_Reviews_Reviewer_Has_Given` کو ہٹا دیں۔

   2. `Reviewer_Nationality` کو رکھیں۔

### ٹیگ کالم

`Tag` کالم مسئلہ پیدا کرتا ہے کیونکہ یہ ایک فہرست (متن کی شکل میں) کے طور پر محفوظ ہے۔ بدقسمتی سے، اس کالم میں ذیلی حصوں کی ترتیب اور تعداد ہمیشہ ایک جیسی نہیں ہوتی۔ چونکہ 515,000 قطاریں اور 1427 ہوٹل ہیں، اور ہر ایک کے پاس جائزہ لینے والے کے منتخب کردہ مختلف اختیارات ہیں، اس لیے انسان کے لیے صحیح جملے کی شناخت کرنا مشکل ہو جاتا ہے۔ یہ وہ جگہ ہے جہاں NLP کارآمد ثابت ہوتا ہے۔ آپ متن کو اسکین کر سکتے ہیں، سب سے عام جملے تلاش کر سکتے ہیں، اور ان کی گنتی کر سکتے ہیں۔

بدقسمتی سے، ہم واحد الفاظ میں دلچسپی نہیں رکھتے، بلکہ کئی الفاظ پر مشتمل جملے (مثلاً *Business trip*) میں دلچسپی رکھتے ہیں۔ اتنے بڑے ڈیٹا پر ایک کثیر الفاظ کی فریکوئنسی ڈسٹری بیوشن الگورتھم چلانا (6762646 الفاظ) غیر معمولی وقت لے سکتا ہے، لیکن ڈیٹا کو دیکھے بغیر، یہ ضروری معلوم ہوتا ہے۔ یہ وہ جگہ ہے جہاں ایکسپلورٹری ڈیٹا اینالیسس کارآمد ہوتا ہے، کیونکہ آپ نے ٹیگز کا نمونہ دیکھا ہے جیسے `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`، آپ یہ پوچھنا شروع کر سکتے ہیں کہ کیا آپ کو پروسیسنگ کو کم کرنے کا کوئی طریقہ ہے۔ خوش قسمتی سے، ایسا ممکن ہے - لیکن پہلے آپ کو دلچسپی کے ٹیگز کی تصدیق کے لیے کچھ اقدامات کرنے ہوں گے۔

### ٹیگز کو فلٹر کرنا

یاد رکھیں کہ ڈیٹا سیٹ کا مقصد جذبات اور ایسے کالم شامل کرنا ہے جو آپ کو بہترین ہوٹل منتخب کرنے میں مدد دے سکیں (اپنے لیے یا شاید کسی کلائنٹ کے لیے جو آپ سے ہوٹل کی سفارش کرنے والے بوٹ بنانے کا کہے)۔ آپ کو خود سے پوچھنا ہوگا کہ آیا ٹیگز حتمی ڈیٹا سیٹ میں کارآمد ہیں یا نہیں۔ یہاں ایک تشریح ہے (اگر آپ کو دوسرے مقاصد کے لیے ڈیٹا سیٹ کی ضرورت ہو تو مختلف ٹیگز کو شامل/خارج کیا جا سکتا ہے):

1. سفر کی قسم متعلقہ ہے، اور اسے رکھنا چاہیے۔
2. مہمانوں کے گروپ کی قسم اہم ہے، اور اسے رکھنا چاہیے۔
3. کمرے، سوئٹ، یا اسٹوڈیو کی قسم جس میں مہمان ٹھہرا تھا غیر متعلقہ ہے (تمام ہوٹلوں میں بنیادی طور پر ایک جیسے کمرے ہوتے ہیں)۔
4. وہ ڈیوائس جس پر جائزہ جمع کرایا گیا غیر متعلقہ ہے۔
5. جائزہ لینے والے نے کتنی راتیں قیام کیا *شاید* متعلقہ ہو اگر آپ طویل قیام کو ہوٹل کی پسندیدگی سے منسلک کریں، لیکن یہ ایک قیاس ہے، اور شاید غیر متعلقہ ہو۔

خلاصہ یہ کہ **2 قسم کے ٹیگز رکھیں اور باقی کو ہٹا دیں**۔

پہلے، آپ ٹیگز کو بہتر فارمیٹ میں لانے تک ان کی گنتی نہیں کرنا چاہتے، اس کا مطلب ہے کہ مربع بریکٹ اور کوٹس کو ہٹانا۔ آپ یہ کئی طریقوں سے کر سکتے ہیں، لیکن آپ تیز ترین طریقہ چاہتے ہیں کیونکہ یہ بہت زیادہ ڈیٹا پروسیس کرنے میں وقت لے سکتا ہے۔ خوش قسمتی سے، pandas میں ان اقدامات کو آسانی سے کرنے کا طریقہ موجود ہے۔

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ہر ٹیگ کچھ اس طرح بن جاتا ہے: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`۔

اگلا مسئلہ یہ ہے کہ کچھ جائزوں یا قطاروں میں 5 کالم ہیں، کچھ میں 3، اور کچھ میں 6۔ یہ ڈیٹا سیٹ کے تخلیق کے طریقے کا نتیجہ ہے، اور اسے ٹھیک کرنا مشکل ہے۔ آپ ہر جملے کی فریکوئنسی گننا چاہتے ہیں، لیکن وہ ہر جائزے میں مختلف ترتیب میں ہیں، اس لیے گنتی غلط ہو سکتی ہے، اور ہوٹل کو وہ ٹیگ نہیں مل سکتا جس کا وہ مستحق تھا۔

اس کے بجائے آپ مختلف ترتیب کو اپنے فائدے کے لیے استعمال کریں گے، کیونکہ ہر ٹیگ کئی الفاظ پر مشتمل ہے لیکن ایک کاما سے الگ ہے! اس کا سب سے آسان طریقہ یہ ہے کہ 6 عارضی کالم بنائیں اور ہر ٹیگ کو اس کے ترتیب کے مطابق متعلقہ کالم میں ڈالیں۔ پھر آپ ان 6 کالموں کو ایک بڑے کالم میں ضم کر سکتے ہیں اور `value_counts()` طریقہ کار کو نتیجے میں آنے والے کالم پر چلا سکتے ہیں۔ جب آپ اسے پرنٹ کریں گے، تو آپ دیکھیں گے کہ 2428 منفرد ٹیگز تھے۔ یہاں ایک چھوٹا نمونہ ہے:

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

کچھ عام ٹیگز جیسے `Submitted from a mobile device` ہمارے لیے بے کار ہیں، اس لیے انہیں گنتی سے پہلے ہٹانا ایک ہوشیار کام ہو سکتا ہے، لیکن یہ اتنی تیز کارروائی ہے کہ آپ انہیں چھوڑ سکتے ہیں اور نظر انداز کر سکتے ہیں۔

### قیام کی مدت کے ٹیگز کو ہٹانا

ان ٹیگز کو ہٹانا پہلا قدم ہے، یہ غور کیے جانے والے ٹیگز کی کل تعداد کو تھوڑا کم کر دیتا ہے۔ نوٹ کریں کہ آپ انہیں ڈیٹا سیٹ سے نہیں ہٹاتے، صرف انہیں جائزوں کے ڈیٹا سیٹ میں گنتی/رکھنے کے لیے منتخب نہیں کرتے۔

| قیام کی مدت   | Count  |
| ------------- | ------ |
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

کمرے، سوئٹ، اسٹوڈیوز، اپارٹمنٹس وغیرہ کی ایک بڑی قسم موجود ہے۔ ان سب کا مطلب تقریباً ایک جیسا ہے اور یہ آپ کے لیے غیر متعلقہ ہیں، اس لیے انہیں غور سے ہٹا دیں۔

| کمرے کی قسم                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

آخر میں، اور یہ خوش آئند ہے (کیونکہ اس میں زیادہ پروسیسنگ نہیں ہوئی)، آپ کے پاس درج ذیل *کارآمد* ٹیگز ہوں گے:

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

آپ یہ دلیل دے سکتے ہیں کہ `Travellers with friends` کم و بیش `Group` کے برابر ہے، اور انہیں اوپر کی طرح یکجا کرنا مناسب ہوگا۔ صحیح ٹیگز کی شناخت کے لیے کوڈ [Tags نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) میں موجود ہے۔

آخری قدم یہ ہے کہ ان ٹیگز میں سے ہر ایک کے لیے نئے کالم بنائیں۔ پھر، ہر جائزہ قطار کے لیے، اگر `Tag` کالم نئے کالموں میں سے کسی سے میل کھاتا ہے، تو 1 شامل کریں، اگر نہیں، تو 0 شامل کریں۔ نتیجہ یہ ہوگا کہ کتنے جائزہ لینے والوں نے اس ہوٹل کو (مجموعی طور پر) کاروبار یا تفریح کے لیے منتخب کیا، یا پالتو جانور کے ساتھ جانے کے لیے، اور یہ ہوٹل کی سفارش کرتے وقت مفید معلومات ہوگی۔

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

## جذباتی تجزیہ کی کارروائیاں

اس آخری حصے میں، آپ جائزہ کالموں پر جذباتی تجزیہ کریں گے اور نتائج کو ڈیٹا سیٹ میں محفوظ کریں گے۔

## مشق: فلٹر شدہ ڈیٹا لوڈ اور محفوظ کریں

نوٹ کریں کہ اب آپ پچھلے حصے میں محفوظ کردہ فلٹر شدہ ڈیٹا سیٹ کو لوڈ کر رہے ہیں، **اصل ڈیٹا سیٹ کو نہیں**۔

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

اگر آپ منفی اور مثبت جائزہ کالموں پر جذباتی تجزیہ چلائیں، تو اس میں کافی وقت لگ سکتا ہے۔ ایک طاقتور ٹیسٹ لیپ ٹاپ پر تیز CPU کے ساتھ، اس میں 12 - 14 منٹ لگے، اس پر منحصر ہے کہ کون سی جذباتی لائبریری استعمال کی گئی۔ یہ ایک (نسبتاً) طویل وقت ہے، اس لیے یہ دیکھنا ضروری ہے کہ کیا اسے تیز کیا جا سکتا ہے۔

اسٹاپ ورڈز، یا عام انگریزی الفاظ جو جملے کے جذبات کو تبدیل نہیں کرتے، کو ہٹانا پہلا قدم ہے۔ انہیں ہٹانے سے جذباتی تجزیہ تیز ہونا چاہیے، لیکن کم درست نہیں (کیونکہ اسٹاپ ورڈز جذبات کو متاثر نہیں کرتے، لیکن وہ تجزیہ کو سست کر دیتے ہیں)۔

سب سے طویل منفی جائزہ 395 الفاظ پر مشتمل تھا، لیکن اسٹاپ ورڈز کو ہٹانے کے بعد، یہ 195 الفاظ پر آ گیا۔

اسٹاپ ورڈز کو ہٹانا بھی ایک تیز عمل ہے، 515,000 قطاروں پر مشتمل 2 جائزہ کالموں سے اسٹاپ ورڈز کو ہٹانے میں ٹیسٹ ڈیوائس پر 3.3 سیکنڈ لگے۔ آپ کے لیے یہ وقت آپ کے ڈیوائس کی CPU اسپیڈ، RAM، SSD کی موجودگی، اور دیگر عوامل پر منحصر ہو کر تھوڑا زیادہ یا کم ہو سکتا ہے۔ اس عمل کی نسبتاً مختصر مدت کا مطلب ہے کہ اگر یہ جذباتی تجزیہ کے وقت کو بہتر بناتا ہے، تو یہ کرنے کے قابل ہے۔

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

اب آپ کو منفی اور مثبت جائزہ کالموں کے لیے جذباتی تجزیہ کا حساب لگانا چاہیے، اور نتیجہ کو 2 نئے کالموں میں محفوظ کرنا چاہیے۔ جذبات کا امتحان یہ ہوگا کہ اسے اسی جائزے کے لیے جائزہ لینے والے کے اسکور سے موازنہ کیا جائے۔ مثال کے طور پر، اگر جذباتی تجزیہ یہ سمجھتا ہے کہ منفی جائزے کا جذبات 1 (انتہائی مثبت جذبات) ہے اور مثبت جائزے کا جذبات بھی 1 ہے، لیکن جائزہ لینے والے نے ہوٹل کو ممکنہ سب سے کم اسکور دیا، تو یا تو جائزہ متن اسکور سے میل نہیں کھاتا، یا جذباتی تجزیہ کرنے والا جذبات کو صحیح طریقے سے پہچان نہیں سکا۔ آپ کو توقع کرنی چاہیے کہ کچھ جذباتی اسکور مکمل طور پر غلط ہوں گے، اور اکثر یہ وضاحت کے قابل ہوگا، مثلاً جائزہ انتہائی طنزیہ ہو سکتا ہے "ظاہر ہے کہ مجھے بغیر ہیٹنگ کے کمرے میں سونا بہت پسند آیا" اور جذباتی تجزیہ کرنے والا سوچتا ہے کہ یہ مثبت جذبات ہے، حالانکہ ایک انسان اسے طنز کے طور پر سمجھے گا۔
این ایل ٹی کے مختلف جذباتی تجزیہ کار فراہم کرتا ہے جن کے ساتھ آپ تجربہ کر سکتے ہیں، اور آپ انہیں تبدیل کر کے دیکھ سکتے ہیں کہ جذباتی تجزیہ زیادہ یا کم درست ہے۔ یہاں ویڈر جذباتی تجزیہ استعمال کیا گیا ہے۔

> ہٹو، سی جے۔ اور گلبرٹ، ای ای۔ (2014). ویڈر: سوشل میڈیا ٹیکسٹ کے جذباتی تجزیہ کے لیے ایک سادہ اصول پر مبنی ماڈل۔ آٹھویں بین الاقوامی کانفرنس برائے ویبلاگز اور سوشل میڈیا (ICWSM-14)، این آربر، ایم آئی، جون 2014۔

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

بعد میں جب آپ اپنے پروگرام میں جذبات کا حساب لگانے کے لیے تیار ہوں، تو آپ اسے ہر جائزے پر اس طرح لاگو کر سکتے ہیں:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

یہ میرے کمپیوٹر پر تقریباً 120 سیکنڈ لیتا ہے، لیکن ہر کمپیوٹر پر وقت مختلف ہو سکتا ہے۔ اگر آپ نتائج کو پرنٹ کرنا چاہتے ہیں اور دیکھنا چاہتے ہیں کہ آیا جذبات جائزے سے مطابقت رکھتے ہیں:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

چیلنج میں فائل استعمال کرنے سے پہلے آخری کام یہ ہے کہ اسے محفوظ کریں! آپ کو اپنے تمام نئے کالمز کو دوبارہ ترتیب دینے پر بھی غور کرنا چاہیے تاکہ ان کے ساتھ کام کرنا آسان ہو (انسان کے لیے، یہ ایک ظاہری تبدیلی ہے)۔

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

آپ کو [تجزیہ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) کے پورے کوڈ کو چلانا چاہیے (جب آپ نے [فلٹرنگ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) کو چلایا ہو تاکہ Hotel_Reviews_Filtered.csv فائل تیار ہو جائے)۔

جائزہ لینے کے لیے، مراحل یہ ہیں:

1. اصل ڈیٹاسیٹ فائل **Hotel_Reviews.csv** کو پچھلے سبق میں [ایکسپلورر نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) کے ساتھ دریافت کیا گیا۔
2. Hotel_Reviews.csv کو [فلٹرنگ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) کے ذریعے فلٹر کیا گیا جس کے نتیجے میں **Hotel_Reviews_Filtered.csv** حاصل ہوا۔
3. Hotel_Reviews_Filtered.csv کو [جذباتی تجزیہ نوٹ بک](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) کے ذریعے پروسیس کیا گیا جس کے نتیجے میں **Hotel_Reviews_NLP.csv** حاصل ہوا۔
4. Hotel_Reviews_NLP.csv کو نیچے دیے گئے NLP چیلنج میں استعمال کریں۔

### نتیجہ

جب آپ نے شروع کیا، تو آپ کے پاس کالمز اور ڈیٹا کے ساتھ ایک ڈیٹاسیٹ تھا لیکن اس کا تمام حصہ تصدیق شدہ یا قابل استعمال نہیں تھا۔ آپ نے ڈیٹا کو دریافت کیا، غیر ضروری حصے کو فلٹر کیا، ٹیگز کو مفید چیزوں میں تبدیل کیا، اپنے اوسط کا حساب لگایا، کچھ جذباتی کالمز شامل کیے اور امید ہے کہ قدرتی متن کو پروسیس کرنے کے بارے میں دلچسپ چیزیں سیکھی ہوں گی۔

## [لیکچر کے بعد کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

## چیلنج

اب جب کہ آپ نے اپنے ڈیٹاسیٹ کا جذباتی تجزیہ کر لیا ہے، دیکھیں کہ آیا آپ اس نصاب میں سیکھی گئی حکمت عملیوں (شاید کلسٹرنگ؟) کا استعمال کر کے جذبات کے ارد گرد پیٹرنز کا تعین کر سکتے ہیں۔

## جائزہ اور خود مطالعہ

[یہ لرن ماڈیول](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) لیں تاکہ مزید سیکھ سکیں اور مختلف ٹولز کا استعمال کر کے متن میں جذبات کو دریافت کریں۔

## اسائنمنٹ

[ایک مختلف ڈیٹاسیٹ آزمائیں](assignment.md)

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے پوری کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا عدم درستگی ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے لیے ہم ذمہ دار نہیں ہیں۔