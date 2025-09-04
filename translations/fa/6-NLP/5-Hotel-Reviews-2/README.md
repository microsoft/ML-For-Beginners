<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-04T00:57:51+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "fa"
}
-->
# تحلیل احساسات با بررسی‌های هتل

حالا که داده‌ها را به‌طور کامل بررسی کرده‌اید، وقت آن است که ستون‌ها را فیلتر کنید و سپس از تکنیک‌های پردازش زبان طبیعی (NLP) روی داده‌ها استفاده کنید تا اطلاعات جدیدی درباره هتل‌ها به دست آورید.
## [آزمون پیش از درس](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### عملیات فیلتر کردن و تحلیل احساسات

همان‌طور که احتمالاً متوجه شده‌اید، این مجموعه داده چند مشکل دارد. برخی ستون‌ها پر از اطلاعات بی‌فایده هستند، برخی دیگر به نظر نادرست می‌آیند. حتی اگر درست باشند، مشخص نیست چگونه محاسبه شده‌اند و نمی‌توان پاسخ‌ها را به‌طور مستقل با محاسبات خودتان تأیید کرد.

## تمرین: کمی پردازش بیشتر داده‌ها

داده‌ها را کمی بیشتر پاک‌سازی کنید. ستون‌هایی اضافه کنید که بعداً مفید خواهند بود، مقادیر برخی ستون‌ها را تغییر دهید و برخی ستون‌ها را کاملاً حذف کنید.

1. پردازش اولیه ستون‌ها

   1. ستون‌های `lat` و `lng` را حذف کنید.

   2. مقادیر `Hotel_Address` را با مقادیر زیر جایگزین کنید (اگر آدرس شامل نام شهر و کشور باشد، آن را فقط به شهر و کشور تغییر دهید).

      این‌ها تنها شهرها و کشورهایی هستند که در مجموعه داده وجود دارند:

      آمستردام، هلند

      بارسلونا، اسپانیا

      لندن، بریتانیا

      میلان، ایتالیا

      پاریس، فرانسه

      وین، اتریش 

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

      حالا می‌توانید داده‌های سطح کشور را جستجو کنید:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | آمستردام، هلند         |    105     |
      | بارسلونا، اسپانیا       |    211     |
      | لندن، بریتانیا          |    400     |
      | میلان، ایتالیا          |    162     |
      | پاریس، فرانسه          |    458     |
      | وین، اتریش             |    158     |

2. پردازش ستون‌های متا-بررسی هتل

  1. ستون `Additional_Number_of_Scoring` را حذف کنید.

  1. ستون `Total_Number_of_Reviews` را با تعداد کل بررسی‌های واقعی برای آن هتل که در مجموعه داده وجود دارد جایگزین کنید.

  1. ستون `Average_Score` را با امتیاز محاسبه‌شده خودمان جایگزین کنید.

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. پردازش ستون‌های بررسی

   1. ستون‌های `Review_Total_Negative_Word_Counts`، `Review_Total_Positive_Word_Counts`، `Review_Date` و `days_since_review` را حذف کنید.

   2. ستون‌های `Reviewer_Score`، `Negative_Review` و `Positive_Review` را همان‌طور که هستند نگه دارید.
     
   3. ستون `Tags` را فعلاً نگه دارید.

     - در بخش بعدی برخی عملیات فیلتر کردن اضافی روی تگ‌ها انجام خواهد شد و سپس تگ‌ها حذف خواهند شد.

4. پردازش ستون‌های بررسی‌کننده

  1. ستون `Total_Number_of_Reviews_Reviewer_Has_Given` را حذف کنید.
  
  2. ستون `Reviewer_Nationality` را نگه دارید.

### ستون‌های تگ

ستون `Tag` مشکل‌ساز است زیرا به‌صورت یک لیست (در قالب متن) در ستون ذخیره شده است. متأسفانه ترتیب و تعداد بخش‌های فرعی در این ستون همیشه یکسان نیست. برای انسان سخت است که عبارات درست را شناسایی کند، زیرا 515,000 ردیف و 1427 هتل وجود دارد و هر کدام گزینه‌های کمی متفاوتی دارند که یک بررسی‌کننده می‌تواند انتخاب کند. اینجاست که NLP مفید واقع می‌شود. شما می‌توانید متن را اسکن کنید، رایج‌ترین عبارات را پیدا کنید و آن‌ها را شمارش کنید.

متأسفانه ما به کلمات منفرد علاقه‌مند نیستیم، بلکه به عبارات چندکلمه‌ای (مثلاً *سفر کاری*) نیاز داریم. اجرای یک الگوریتم توزیع فراوانی عبارات چندکلمه‌ای روی این حجم از داده (6762646 کلمه) ممکن است زمان بسیار زیادی ببرد، اما بدون بررسی داده‌ها، به نظر می‌رسد که این هزینه ضروری است. اینجاست که تحلیل داده‌های اکتشافی مفید واقع می‌شود، زیرا شما نمونه‌ای از تگ‌ها مانند `[' سفر کاری  ', ' مسافر تنها ', ' اتاق یک‌نفره ', ' اقامت 5 شب ', ' ارسال‌شده از دستگاه موبایل ']` را دیده‌اید، می‌توانید شروع به پرسیدن کنید که آیا امکان کاهش قابل‌توجه پردازش وجود دارد یا خیر. خوشبختانه، این امکان وجود دارد - اما ابتدا باید چند مرحله را دنبال کنید تا تگ‌های مورد علاقه را مشخص کنید.

### فیلتر کردن تگ‌ها

به یاد داشته باشید که هدف این مجموعه داده اضافه کردن احساسات و ستون‌هایی است که به شما کمک می‌کند بهترین هتل را انتخاب کنید (برای خودتان یا شاید یک مشتری که از شما خواسته است یک ربات توصیه هتل بسازید). باید از خودتان بپرسید که آیا تگ‌ها در مجموعه داده نهایی مفید هستند یا خیر. اینجا یک تفسیر ارائه شده است (اگر به دلایل دیگر به مجموعه داده نیاز داشتید، ممکن است تگ‌های مختلفی در انتخاب باقی بمانند/حذف شوند):

1. نوع سفر مرتبط است و باید باقی بماند.
2. نوع گروه مهمان مهم است و باید باقی بماند.
3. نوع اتاق، سوئیت یا استودیویی که مهمان در آن اقامت داشته است بی‌اهمیت است (همه هتل‌ها اساساً اتاق‌های مشابه دارند).
4. دستگاهی که بررسی از آن ارسال شده است بی‌اهمیت است.
5. تعداد شب‌هایی که بررسی‌کننده اقامت داشته است *ممکن است* مرتبط باشد اگر اقامت طولانی‌تر را به دوست داشتن بیشتر هتل نسبت دهید، اما این ارتباط ضعیف است و احتمالاً بی‌اهمیت.

به‌طور خلاصه، **دو نوع تگ را نگه دارید و بقیه را حذف کنید**.

ابتدا، نمی‌خواهید تگ‌ها را شمارش کنید تا زمانی که در قالب بهتری باشند، بنابراین این به معنای حذف براکت‌ها و نقل‌قول‌ها است. می‌توانید این کار را به چند روش انجام دهید، اما سریع‌ترین روش را می‌خواهید زیرا پردازش حجم زیادی از داده‌ها ممکن است زمان زیادی ببرد. خوشبختانه، pandas راه آسانی برای انجام هر یک از این مراحل دارد.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

هر تگ به چیزی شبیه این تبدیل می‌شود: `سفر کاری، مسافر تنها، اتاق یک‌نفره، اقامت 5 شب، ارسال‌شده از دستگاه موبایل`.

سپس با یک مشکل مواجه می‌شویم. برخی بررسی‌ها یا ردیف‌ها 5 ستون دارند، برخی 3، برخی 6. این نتیجه نحوه ایجاد مجموعه داده است و سخت است که آن را اصلاح کنیم. شما می‌خواهید شمارش فراوانی هر عبارت را انجام دهید، اما آن‌ها در هر بررسی ترتیب متفاوتی دارند، بنابراین شمارش ممکن است اشتباه باشد و یک هتل ممکن است تگی را که شایسته آن است دریافت نکند.

در عوض، شما از ترتیب متفاوت به نفع خود استفاده خواهید کرد، زیرا هر تگ چندکلمه‌ای است اما همچنین با کاما جدا شده است! ساده‌ترین راه برای انجام این کار ایجاد 6 ستون موقت است که هر تگ در ستون مربوط به ترتیب خود وارد شود. سپس می‌توانید 6 ستون را به یک ستون بزرگ ادغام کنید و روش `value_counts()` را روی ستون حاصل اجرا کنید. با چاپ آن، خواهید دید که 2428 تگ منحصربه‌فرد وجود داشت. اینجا یک نمونه کوچک آورده شده است:

| Tag                            | Count  |
| ------------------------------ | ------ |
| سفر تفریحی                     | 417778 |
| ارسال‌شده از دستگاه موبایل    | 307640 |
| زوج                            | 252294 |
| اقامت 1 شب                     | 193645 |
| اقامت 2 شب                     | 133937 |
| مسافر تنها                     | 108545 |
| اقامت 3 شب                     | 95821  |
| سفر کاری                       | 82939  |
| گروه                           | 65392  |
| خانواده با کودکان خردسال      | 61015  |
| اقامت 4 شب                     | 47817  |
| اتاق دو نفره                   | 35207  |
| اتاق استاندارد دو نفره         | 32248  |
| اتاق سوپریور دو نفره           | 31393  |
| خانواده با کودکان بزرگ‌تر      | 26349  |
| اتاق دلوکس دو نفره             | 24823  |
| اتاق دو نفره یا دوقلو          | 22393  |
| اقامت 5 شب                     | 20845  |
| اتاق استاندارد دو نفره یا دوقلو| 17483  |
| اتاق کلاسیک دو نفره            | 16989  |
| اتاق سوپریور دو نفره یا دوقلو  | 13570  |
| 2 اتاق                         | 12393  |

برخی از تگ‌های رایج مانند `ارسال‌شده از دستگاه موبایل` برای ما بی‌فایده هستند، بنابراین ممکن است حذف آن‌ها قبل از شمارش فراوانی عبارات کار هوشمندانه‌ای باشد، اما این عملیات بسیار سریع است و می‌توانید آن‌ها را نگه دارید و نادیده بگیرید.

### حذف تگ‌های مربوط به طول اقامت

حذف این تگ‌ها مرحله اول است، این کار تعداد کل تگ‌هایی که باید در نظر گرفته شوند را کمی کاهش می‌دهد. توجه داشته باشید که آن‌ها را از مجموعه داده حذف نمی‌کنید، فقط تصمیم می‌گیرید که آن‌ها را به‌عنوان مقادیر برای شمارش/نگه‌داری در مجموعه داده بررسی‌ها حذف کنید.

| طول اقامت       | Count  |
| ---------------- | ------ |
| اقامت 1 شب       | 193645 |
| اقامت 2 شب       | 133937 |
| اقامت 3 شب       | 95821  |
| اقامت 4 شب       | 47817  |
| اقامت 5 شب       | 20845  |
| اقامت 6 شب       | 9776   |
| اقامت 7 شب       | 7399   |
| اقامت 8 شب       | 2502   |
| اقامت 9 شب       | 1293   |
| ...              | ...    |

انواع مختلفی از اتاق‌ها، سوئیت‌ها، استودیوها، آپارتمان‌ها و غیره وجود دارد. همه آن‌ها تقریباً یک معنی دارند و برای شما مرتبط نیستند، بنابراین آن‌ها را از نظر حذف کنید.

| نوع اتاق                     | Count |
| ----------------------------- | ----- |
| اتاق دو نفره                 | 35207 |
| اتاق استاندارد دو نفره       | 32248 |
| اتاق سوپریور دو نفره         | 31393 |
| اتاق دلوکس دو نفره           | 24823 |
| اتاق دو نفره یا دوقلو        | 22393 |
| اتاق استاندارد دو نفره یا دوقلو | 17483 |
| اتاق کلاسیک دو نفره          | 16989 |
| اتاق سوپریور دو نفره یا دوقلو | 13570 |

در نهایت، و این خوشحال‌کننده است (زیرا پردازش زیادی لازم نبود)، شما با تگ‌های *مفید* زیر باقی خواهید ماند:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| سفر تفریحی                                    | 417778 |
| زوج                                           | 252294 |
| مسافر تنها                                   | 108545 |
| سفر کاری                                      | 82939  |
| گروه (ترکیب‌شده با مسافران با دوستان)        | 67535  |
| خانواده با کودکان خردسال                     | 61015  |
| خانواده با کودکان بزرگ‌تر                    | 26349  |
| با حیوان خانگی                               | 1405   |

می‌توانید استدلال کنید که `مسافران با دوستان` تقریباً همان `گروه` است و ترکیب این دو منطقی است، همان‌طور که در بالا انجام شده است. کد شناسایی تگ‌های درست در [دفترچه تگ‌ها](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) موجود است.

مرحله نهایی ایجاد ستون‌های جدید برای هر یک از این تگ‌ها است. سپس، برای هر ردیف بررسی، اگر ستون `Tag` با یکی از ستون‌های جدید مطابقت داشت، مقدار 1 اضافه کنید، اگر نه، مقدار 0 اضافه کنید. نتیجه نهایی شمارش تعداد بررسی‌کنندگان است که این هتل را (به‌صورت کلی) برای مثال برای کار یا تفریح انتخاب کرده‌اند، یا برای آوردن حیوان خانگی، و این اطلاعات مفیدی هنگام توصیه یک هتل خواهد بود.

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

### ذخیره فایل

در نهایت، مجموعه داده را همان‌طور که اکنون است با یک نام جدید ذخیره کنید.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## عملیات تحلیل احساسات

در این بخش نهایی، شما تحلیل احساسات را روی ستون‌های بررسی اعمال خواهید کرد و نتایج را در یک مجموعه داده ذخیره خواهید کرد.

## تمرین: بارگذاری و ذخیره داده‌های فیلترشده

توجه داشته باشید که اکنون مجموعه داده فیلترشده‌ای را که در بخش قبلی ذخیره شده است بارگذاری می‌کنید، **نه** مجموعه داده اصلی.

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

### حذف کلمات توقف

اگر بخواهید تحلیل احساسات را روی ستون‌های بررسی منفی و مثبت اجرا کنید، ممکن است زمان زیادی طول بکشد. آزمایش‌شده روی یک لپ‌تاپ قدرتمند با CPU سریع، این عملیات 12 تا 14 دقیقه طول کشید، بسته به اینکه از کدام کتابخانه تحلیل احساسات استفاده شده است. این زمان نسبتاً طولانی است، بنابراین ارزش بررسی دارد که آیا می‌توان آن را سریع‌تر کرد.

حذف کلمات توقف، یا کلمات رایج انگلیسی که احساسات یک جمله را تغییر نمی‌دهند، اولین مرحله است. با حذف آن‌ها، تحلیل احساسات باید سریع‌تر اجرا شود، اما دقت کمتری نخواهد داشت (زیرا کلمات توقف بر احساسات تأثیر نمی‌گذارند، اما سرعت تحلیل را کاهش می‌دهند).

طولانی‌ترین بررسی منفی 395 کلمه بود، اما پس از حذف کلمات توقف، به 195 کلمه کاهش یافت.

حذف کلمات توقف نیز یک عملیات سریع است، حذف کلمات توقف از 2 ستون بررسی در بیش از 515,000 ردیف روی دستگاه آزمایشی 3.3 ثانیه طول کشید. ممکن است این زمان برای شما کمی بیشتر یا کمتر باشد، بسته به سرعت CPU دستگاه، RAM، داشتن SSD یا نه، و برخی عوامل دیگر. کوتاهی نسبی این عملیات به این معناست که اگر زمان تحلیل احساسات را بهبود دهد، ارزش انجام دادن دارد.

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

### اجرای تحلیل احساسات
اکنون باید تحلیل احساسات را برای ستون‌های نظرات منفی و مثبت محاسبه کنید و نتیجه را در دو ستون جدید ذخیره کنید. آزمون احساسات این خواهد بود که آن را با امتیاز داده‌شده توسط نویسنده نظر برای همان نظر مقایسه کنید. به عنوان مثال، اگر تحلیل احساسات نشان دهد که نظر منفی دارای احساسات ۱ (احساسات بسیار مثبت) و نظر مثبت نیز دارای احساسات ۱ است، اما نویسنده کمترین امتیاز ممکن را به هتل داده باشد، یا متن نظر با امتیاز مطابقت ندارد یا تحلیلگر احساسات نتوانسته احساسات را به درستی تشخیص دهد. انتظار داشته باشید که برخی از امتیازات احساسات کاملاً اشتباه باشند، و اغلب این قابل توضیح خواهد بود، به عنوان مثال، نظر ممکن است بسیار طعنه‌آمیز باشد: «البته من عاشق خوابیدن در اتاقی بدون گرمایش بودم» و تحلیلگر احساسات فکر کند که این احساسات مثبت است، در حالی که یک انسان با خواندن آن متوجه طعنه خواهد شد.

NLTK ابزارهای مختلفی برای تحلیل احساسات ارائه می‌دهد که می‌توانید با آن‌ها کار کنید و ببینید آیا احساسات دقیق‌تر هستند یا خیر. در اینجا از تحلیل احساسات VADER استفاده شده است.

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

بعداً در برنامه خود، زمانی که آماده محاسبه احساسات هستید، می‌توانید آن را به هر نظر اعمال کنید به این صورت:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

این فرآیند تقریباً ۱۲۰ ثانیه روی کامپیوتر من طول می‌کشد، اما زمان آن روی هر کامپیوتر متفاوت خواهد بود. اگر می‌خواهید نتایج را چاپ کنید و ببینید آیا احساسات با نظر مطابقت دارد:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

آخرین کاری که باید با فایل انجام دهید قبل از استفاده در چالش، ذخیره کردن آن است! همچنین باید به مرتب‌سازی مجدد تمام ستون‌های جدید خود فکر کنید تا کار با آن‌ها آسان‌تر شود (برای انسان، این یک تغییر ظاهری است).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

شما باید کل کد را برای [دفترچه تحلیل](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) اجرا کنید (بعد از اینکه [دفترچه فیلتر کردن](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) را اجرا کردید تا فایل Hotel_Reviews_Filtered.csv تولید شود).

برای مرور، مراحل به این صورت هستند:

1. فایل مجموعه داده اصلی **Hotel_Reviews.csv** در درس قبلی با [دفترچه کاوشگر](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) بررسی شده است.
2. Hotel_Reviews.csv توسط [دفترچه فیلتر کردن](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) فیلتر شده و نتیجه آن **Hotel_Reviews_Filtered.csv** است.
3. Hotel_Reviews_Filtered.csv توسط [دفترچه تحلیل احساسات](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) پردازش شده و نتیجه آن **Hotel_Reviews_NLP.csv** است.
4. از Hotel_Reviews_NLP.csv در چالش NLP زیر استفاده کنید.

### نتیجه‌گیری

وقتی شروع کردید، یک مجموعه داده با ستون‌ها و داده‌ها داشتید اما همه آن قابل تأیید یا استفاده نبود. شما داده‌ها را بررسی کردید، موارد غیرضروری را فیلتر کردید، برچسب‌ها را به چیزی مفید تبدیل کردید، میانگین‌های خود را محاسبه کردید، ستون‌های احساسات اضافه کردید و امیدواریم چیزهای جالبی درباره پردازش متن طبیعی یاد گرفته باشید.

## [آزمون پس از درس](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## چالش

اکنون که مجموعه داده خود را برای احساسات تحلیل کرده‌اید، ببینید آیا می‌توانید با استفاده از استراتژی‌هایی که در این دوره یاد گرفته‌اید (مثلاً خوشه‌بندی)، الگوهایی پیرامون احساسات پیدا کنید.

## مرور و مطالعه شخصی

[این ماژول آموزشی](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) را بگیرید تا بیشتر یاد بگیرید و از ابزارهای مختلف برای بررسی احساسات در متن استفاده کنید.

## تکلیف

[یک مجموعه داده متفاوت را امتحان کنید](assignment.md)

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما برای دقت تلاش می‌کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادقتی‌هایی باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما هیچ مسئولیتی در قبال سوءتفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.