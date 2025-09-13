<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T14:25:19+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "my"
}
-->
# ဟိုတယ်အကြောင်းအမြင်သုံးသပ်မှု (Sentiment Analysis)

အခု dataset ကိုအသေးစိတ်လေ့လာပြီးပြီဆိုတော့၊ column တွေကို filter လုပ်ပြီးနောက်မှာ NLP နည်းလမ်းတွေကိုအသုံးပြုကာ ဟိုတယ်များအကြောင်းအသစ်အမြင်တွေ ရယူနိုင်ဖို့အချိန်ရောက်ပါပြီ။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Filtering & Sentiment Analysis လုပ်ဆောင်မှုများ

သင်မိမိ dataset ကိုကြည့်ပြီးသားဆိုရင်၊ dataset မှာအချို့ပြဿနာတွေရှိနေတယ်ဆိုတာသတိထားမိမှာပါ။ အချို့ column တွေမှာ အသုံးမဝင်တဲ့အချက်အလက်တွေဖြည့်ထားပြီး၊ အချို့ကတော့မှားယွင်းနေတယ်လို့ပုံရပါတယ်။ မှန်ကန်နေရင်တောင်၊ အဲဒီအချက်အလက်တွေကိုဘယ်လိုတွက်ချက်ထားတယ်ဆိုတာရှင်းလင်းမှုမရှိဘဲ၊ မိမိ calculation တွေကိုအသုံးပြုပြီး independent verification လုပ်လို့မရနိုင်ပါ။

## လေ့ကျင့်ခန်း - အချက်အလက်တွေကိုနောက်ထပ် process လုပ်ခြင်း

အချက်အလက်တွေကိုနည်းနည်းပိုပြီးသန့်စင်ပါ။ နောက်ပိုင်းမှာအသုံးဝင်မယ့် column တွေထည့်ပါ၊ အချို့ column တွေမှာ value တွေကိုပြောင်းပါ၊ အချို့ column တွေကိုလုံးဝဖျက်ပစ်ပါ။

1. Initial column processing

   1. `lat` နဲ့ `lng` ကိုဖျက်ပါ။

   2. `Hotel_Address` value တွေကိုအောက်ပါ value တွေဖြင့်အစားထိုးပါ (address မှာမြို့နဲ့နိုင်ငံအမည်ပါဝင်ရင်၊ မြို့နဲ့နိုင်ငံအမည်ကိုသာထားပါ)။

      Dataset မှာပါဝင်တဲ့မြို့နဲ့နိုင်ငံတွေကတော့:

      Amsterdam, Netherlands

      Barcelona, Spain

      London, United Kingdom

      Milan, Italy

      Paris, France

      Vienna, Austria 

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

      အခုတော့ country အဆင့်အချက်အလက်တွေကို query လုပ်နိုင်ပါပြီ:

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

2. Hotel Meta-review column တွေကို process လုပ်ပါ။

  1. `Additional_Number_of_Scoring` ကိုဖျက်ပါ။

  1. `Total_Number_of_Reviews` ကို dataset မှာတကယ်ပါဝင်တဲ့ review အရေအတွက်နဲ့အစားထိုးပါ။

  1. `Average_Score` ကိုမိမိတို့တွက်ချက်ထားတဲ့ score နဲ့အစားထိုးပါ။

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Review column တွေကို process လုပ်ပါ။

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` နဲ့ `days_since_review` ကိုဖျက်ပါ။

   2. `Reviewer_Score`, `Negative_Review`, နဲ့ `Positive_Review` ကိုအတိုင်းထားပါ။

   3. `Tags` ကိုယာယီထားပါ။

     - နောက်ပိုင်းမှာ tag တွေကိုနောက်ထပ် filter လုပ်ပြီးနောက်မှာတော့ tag တွေကိုဖျက်ပစ်ပါမယ်။

4. Reviewer column တွေကို process လုပ်ပါ။

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` ကိုဖျက်ပါ။
  
  2. `Reviewer_Nationality` ကိုအတိုင်းထားပါ။

### Tag column တွေ

`Tag` column ကပြဿနာရှိနေပါတယ်၊ အဲဒါက list (text အနေနဲ့) ကို column ထဲမှာသိမ်းထားတာဖြစ်ပါတယ်။ အလွယ်တကူလူ့အမြင်နဲ့စစ်ဆေးဖို့ခက်ခဲပါတယ်၊ အကြောင်းကတော့ dataset မှာ 515,000 rows ရှိပြီး၊ 1427 ဟိုတယ်တွေရှိပြီး၊ review တစ်ခုစီမှာ reviewer ရွေးချယ်နိုင်တဲ့ option တွေကနည်းနည်းကွဲပြားနေပါတယ်။ ဒီနေရာမှာတော့ NLP ကအထောက်အကူဖြစ်ပါတယ်။ Text ကို scan လုပ်ပြီးအများဆုံးတွေ့ရတဲ့ phrase တွေကို count လုပ်နိုင်ပါတယ်။

သို့သော်၊ single word တွေမဟုတ်ဘဲ multi-word phrase တွေ (ဥပမာ - *Business trip*) ကိုစိတ်ဝင်စားပါတယ်။ အဲဒီလို data အများကြီး (6762646 words) ကို multi-word frequency distribution algorithm နဲ့ run လုပ်ရင် အချိန်အတော်ကြာနိုင်ပါတယ်။ ဒါပေမယ့် data ကိုမကြည့်ဘဲ အဲဒီလိုလုပ်ဖို့လိုအပ်တယ်လို့ပုံရပါတယ်။ ဒီနေရာမှာ exploratory data analysis ကအထောက်အကူဖြစ်ပါတယ်၊ အကြောင်းကတော့ tag တွေကို sample အနေနဲ့ကြည့်ပြီး `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` စသဖြင့် tag တွေကိုစစ်ဆေးနိုင်ပါတယ်။ အဲဒီနောက်မှာတော့ process လုပ်ရမယ့်အချက်အလက်တွေကိုလျှော့ချနိုင်မလားဆိုတာစဉ်းစားနိုင်ပါတယ်။ ကံကောင်းစွာ၊ လျှော့ချနိုင်ပါတယ် - ဒါပေမယ့်အရင်ဆုံးအချို့အဆင့်တွေကိုလိုက်နာဖို့လိုပါတယ်။

### Tag တွေကို filter လုပ်ခြင်း

Dataset ရဲ့ရည်ရွယ်ချက်က sentiment တွေကိုထည့်ပြီး dataset ကိုအသုံးပြုသူအတွက်အကောင်းဆုံးဟိုတယ်ကိုရွေးချယ်နိုင်ဖို့ column တွေထည့်ဖို့ဖြစ်ပါတယ်။ Tag တွေကအသုံးဝင်မလားမသုံးဝင်ဘူးလားကိုမေးမြန်းဖို့လိုပါတယ်။ အောက်မှာ interpretation တစ်ခုကိုပေးထားပါတယ် (သင် dataset ကိုအခြားရည်ရွယ်ချက်အတွက်အသုံးပြုမယ်ဆိုရင် tag တွေကိုထည့်/မထည့်တာကွဲပြားနိုင်ပါတယ်)။

1. ခရီးအမျိုးအစားကအရေးကြီးပါတယ်၊ ထားပါ။
2. ဧည့်သည်အုပ်စုအမျိုးအစားကအရေးကြီးပါတယ်၊ ထားပါ။
3. ဧည့်သည်နေထိုင်ခဲ့တဲ့အခန်း၊ suite, studio အမျိုးအစားကမအရေးကြီးပါဘူး (ဟိုတယ်တိုင်းမှာအခန်းတွေကအခြေခံအားဖြင့်တူတူပါ)။
4. Review ကိုဘယ် device မှတင်ထားတယ်ဆိုတာကမအရေးကြီးပါဘူး။
5. Reviewer နေထိုင်ခဲ့တဲ့ညအရေအတွက်က hotel ကိုပိုကြိုက်တယ်လို့ထင်ရင်အရေးကြီးနိုင်ပါတယ်၊ ဒါပေမယ့်အရေးကြီးမှုနည်းပါတယ်၊ အများအားဖြင့်မအရေးကြီးပါဘူး။

အကျဉ်းချုပ်အားဖြင့် **tag ၂ မျိုးကိုထားပြီးအခြား tag တွေကိုဖယ်ရှားပါ**။

အရင်ဆုံး tag တွေကို count လုပ်မယ့်အချိန်မှာ format ပိုမကောင်းတဲ့အခြေအနေကိုဖယ်ရှားဖို့လိုပါတယ်၊ square bracket နဲ့ quotes တွေကိုဖယ်ရှားပါ။ အဲဒီအဆင့်တွေကိုအလွယ်တကူလုပ်နိုင်တဲ့ pandas library ရှိပါတယ်။

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Tag တစ်ခုစီက `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device` လိုမျိုးဖြစ်လာပါမယ်။

နောက်ထပ်ပြဿနာတစ်ခုကိုတွေ့ရပါတယ်။ Review တစ်ခုစီမှာ column 5 ခုရှိတယ်၊ အချို့မှာ 3 ခု၊ အချို့မှာ 6 ခုရှိတယ်။ Dataset ကိုဘယ်လိုဖန်တီးထားတယ်ဆိုတာကြောင့်ဖြစ်ပါတယ်၊ ပြင်ဖို့ခက်ပါတယ်။ Phrase တစ်ခုစီရဲ့ frequency count ကိုရယူချင်ပေမယ့် review တစ်ခုစီမှာ order ကွဲပြားနေတဲ့အတွက် count မှားနိုင်ပါတယ်၊ hotel တစ်ခုမှာ deserve လုပ်ထားတဲ့ tag ကိုမရနိုင်ပါဘူး။

အဲဒီ order ကွဲပြားမှုကိုအကျိုးရှိအောင်အသုံးပြုပါမယ်၊ tag တစ်ခုစီမှာ multi-word phrase ဖြစ်ပြီး comma နဲ့ခွဲထားပါတယ်။ အလွယ်ဆုံးနည်းကတော့ ယာယီ column 6 ခုဖန်တီးပြီး tag တစ်ခုစီကို tag ရဲ့ order နဲ့ကိုက်ညီတဲ့ column ထဲထည့်ပါ။ အဲဒီနောက် column 6 ခုကို merge လုပ်ပြီး `value_counts()` method ကို run လုပ်ပါ။ Print လုပ်ပြီးရင် unique tag 2428 ခုရှိတယ်ဆိုတာတွေ့ရပါမယ်။ အောက်မှာ sample တစ်ခုကိုပေးထားပါတယ်:

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

`Submitted from a mobile device` လိုမျိုးအသုံးမဝင်တဲ့ tag တွေကိုဖယ်ရှားဖို့ smart ဖြစ်နိုင်ပါတယ်၊ ဒါပေမယ့် operation လုပ်ရတာမြန်လွန်းတဲ့အတွက် ထည့်ထားပြီး ignore လုပ်နိုင်ပါတယ်။

### နေထိုင်ခဲ့တဲ့ညအရေအတွက် tag တွေကိုဖယ်ရှားခြင်း

Tag တွေကိုဖယ်ရှားခြင်းကအဆင့် ၁ ဖြစ်ပါတယ်၊ အစီအစဉ်မှာပါဝင်တဲ့ tag အရေအတွက်ကိုနည်းနည်းလျှော့ချနိုင်ပါတယ်။ သတိပြုပါ၊ dataset မှ tag တွေကိုဖျက်မပစ်ပါဘူး၊ review dataset မှာ count/keep လုပ်ဖို့အတွက်သာဖယ်ရှားပါ။

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

Room, suite, studio, apartment စသဖြင့်အခန်းအမျိုးအစားတွေကအများအားဖြင့်တူတူဖြစ်ပြီး dataset ရဲ့ရည်ရွယ်ချက်အတွက်မအရေးကြီးပါဘူး၊ အဲဒီအခန်းအမျိုးအစား tag တွေကိုဖယ်ရှားပါ။

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

နောက်ဆုံးမှာတော့ delight ဖြစ်စရာကောင်းပါတယ် (process လုပ်ရတာအတော်လွယ်ကူပါတယ်)၊ အသုံးဝင်တဲ့ tag တွေကအောက်ပါအတိုင်းဖြစ်လာပါမယ်:

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

`Travellers with friends` ကို `Group` နဲ့တူတူဖြစ်တယ်လို့ဆိုနိုင်ပါတယ်၊ အဲဒီအတွက်အထက်မှာပေါင်းစပ်ထားပါတယ်။ Correct tag တွေကိုစစ်ဆေးဖို့ code က [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) မှာပါဝင်ပါတယ်။

နောက်ဆုံးအဆင့်ကတော့ tag တစ်ခုစီအတွက် column အသစ်တွေဖန်တီးပါ။ Review row တစ်ခုစီမှာ `Tag` column ကအသစ်ဖန်တီးထားတဲ့ column တစ်ခုနဲ့ကိုက်ညီရင် 1 ထည့်ပါ၊ မကိုက်ညီရင် 0 ထည့်ပါ။ အဆုံးမှာတော့ hotel recommendation အတွက် အသုံးဝင်တဲ့အချက်အလက်တွေကိုရရှိပါမယ်၊ ဥပမာ - business vs leisure, pet နဲ့အတူလာတဲ့ hotel စသဖြင့်။

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

### File ကိုသိမ်းဆည်းပါ

နောက်ဆုံးမှာ dataset ကိုအသစ်နာမည်နဲ့သိမ်းဆည်းပါ။

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentiment Analysis လုပ်ဆောင်မှုများ

နောက်ဆုံးအပိုင်းမှာ review column တွေကို sentiment analysis လုပ်ပြီးရလဒ်တွေကို dataset မှာသိမ်းဆည်းပါမယ်။

## လေ့ကျင့်ခန်း - filtered data ကို load နဲ့ save လုပ်ပါ

အခုတော့နောက်ဆုံးအပိုင်းမှာသိမ်းဆည်းထားတဲ့ filtered dataset ကို load လုပ်ပါ၊ **original dataset ကိုမသုံးပါနဲ့**။

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

### Stop words ဖယ်ရှားခြင်း

Negative နဲ့ Positive review column တွေမှာ Sentiment Analysis run လုပ်မယ်ဆိုရင် အချိန်အတော်ကြာနိုင်ပါတယ်။ စမ်းသပ်ထားတဲ့ laptop (CPU မြန်ဆန်တဲ့ device) မှာ sentiment library အမျိုးအစားပေါ်မူတည်ပြီး 12 - 14 မိနစ်ကြာပါတယ်။ အချိန်အတော်ကြာတဲ့အတွက် အချိန်လျှော့ချနိုင်မလားစဉ်းစားဖို့လိုပါတယ်။

Stop words (sentiment ကိုမပြောင်းလဲတဲ့ common English words) ကိုဖယ်ရှားခြင်းကပထမအဆင့်ဖြစ်ပါတယ်။ Stop words ကိုဖယ်ရှားခြင်းက sentiment analysis run လုပ်တဲ့အချိန်ကိုမြန်ဆန်စေပြီး၊ accuracy ကိုမလျော့ချပါဘူး (stop words တွေက sentiment ကိုမထိခိုက်ပေမယ့် analysis ကိုနှေးစေပါတယ်)။

အရှည်ဆုံး negative review က 395 words ရှိပါတယ်၊ Stop words ကိုဖယ်ရှားပြီးနောက်မှာတော့ 195 words ဖြစ်ပါတယ်။

Stop words ကိုဖယ်ရှားခြင်းကလည်းမြန်ဆန်တဲ့ operation ဖြစ်ပါတယ်၊ 515,000 rows ရှိတဲ့ review column 2 ခုမှာ stop words ကိုဖယ်ရှားဖို့ 3.3 seconds ကြာပါတယ်။ Device ရဲ့ CPU မြန်နှုန်း၊ RAM, SSD ရှိ/မရှိ၊ အခြား factor တွေကြောင့် အချိန်နည်းနည်းကွဲနိုင်ပါတယ်။ Operation ရဲ့အချိန်တိုမှုကြောင့် sentiment analysis အချိန်ကိုလျှော့ချနိုင်ရင်လုပ်ဖို့တန်ပါတယ်။

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

### Sentiment analysis လုပ်ဆောင်ခြင်း

အခုတော့ negative နဲ့ positive review column 2 ခုအတွက် sentiment analysis ကိုတွက်ချက်ပြီး၊ ရလဒ်ကို column အသစ် 2 ခုမှာသိမ်းဆည်းပါ။ Sentiment ကို reviewer ရဲ့ score နဲ့နှိုင်းယှဉ်ပါမယ်။ ဥပမာ - Negative review sentiment က 1 (အလွန် positive sentiment) ဖြစ်ပြီး Positive review sentiment က 1 ဖြစ်တယ်၊ Reviewer က hotel ကိုအနိမ့်ဆုံး score ပေးထားတယ်ဆိုရင် review text က score နဲ့မကိုက်ညီတာဖြစ်နိုင်ပါတယ်၊ ဒါမှမဟုတ် sentiment analyser က sentiment ကိုမှန်ကန်စွာမသိနိုင်တာဖြစ်နိုင်ပါတယ်။ Sentiment score တွေမှားနေတဲ့အချို့ကိုတွေ့ရနိုင်ပါတယ်၊ Sarcasm လိုမျိုးအခြေအနေတွေကြောင့်ဖြစ်နိုင်ပါတယ်၊ ဥပမာ - "Of course I LOVED sleeping in a room with no heating" ဆိုပြီး sarcasm ပြောထားတဲ့ review ကို sentiment analyser က positive sentiment လို့ထင်နိုင်ပါတယ်၊ ဒါပေမယ့်လူ့အမြ
NLTK သည် အမျိုးမျိုးသော စိတ်ခံစားမှု ခွဲခြားစနစ်များကို သင်ယူရန် ပံ့ပိုးပေးပြီး၊ သင်သည် အဲဒီစနစ်များကို အစားထိုးအသုံးပြု၍ စိတ်ခံစားမှု ခွဲခြားမှု၏ တိကျမှုကို ပိုမိုကောင်းမွန်သို့မဟုတ် ပိုမိုနည်းပါးသည်ကို ကြည့်ရှုနိုင်သည်။ ဒီနေရာမှာ VADER စိတ်ခံစားမှု ခွဲခြားမှုကို အသုံးပြုထားသည်။

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

သင်၏ အစီအစဉ်တွင် စိတ်ခံစားမှုကို တွက်ချက်ရန် ပြင်ဆင်ပြီးနောက်၊ သင်သည် အောက်ပါအတိုင်း တစ်ခုချင်းစီကို အသုံးပြုနိုင်သည်-

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

ဤလုပ်ငန်းစဉ်သည် ကျွန်ုပ်၏ ကွန်ပျူတာတွင် ၁၂၀ စက္ကန့်ခန့် ကြာမြင့်သည်၊ သို့သော် ကွန်ပျူတာတစ်ခုချင်းစီတွင် ကွဲပြားနိုင်သည်။ ရလဒ်များကို ပုံနှိပ်ပြီး စိတ်ခံစားမှုသည် သုံးသပ်ချက်နှင့် ကိုက်ညီမညီကြည့်လိုပါက-

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

အခန်းစဉ်တွင် အသုံးပြုရန် မတိုင်မီ ဖိုင်ကို သိမ်းဆည်းရန် အရေးကြီးသည်။ သင်၏ ကော်လံအသစ်များကို လူသားများအတွက် အဆင်ပြေစေရန် ပြန်လည်စီစဉ်ရန်လည်း စဉ်းစားသင့်သည် (ဓာတ်ပုံဆိုင်ရာ ပြောင်းလဲမှုဖြစ်သည်)။

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

သင်သည် [အာနိသင်မှတ်စု](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) အားလုံးကို အပြည့်အဝ လုပ်ဆောင်သင့်သည် (Hotel_Reviews_Filtered.csv ဖိုင်ကို ဖန်တီးရန် [သင်၏ စစ်ထုတ်မှတ်စု](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ကို လုပ်ဆောင်ပြီးနောက်)။

ပြန်လည်သုံးသပ်ရန် အဆင့်များမှာ-

1. မူရင်းဒေတာဖိုင် **Hotel_Reviews.csv** ကို [စူးစမ်းမှတ်စု](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) ဖြင့် ယခင်သင်ခန်းစာတွင် စူးစမ်းခဲ့သည်။
2. Hotel_Reviews.csv ကို [စစ်ထုတ်မှတ်စု](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ဖြင့် စစ်ထုတ်ပြီး **Hotel_Reviews_Filtered.csv** ဖြစ်လာသည်။
3. Hotel_Reviews_Filtered.csv ကို [စိတ်ခံစားမှု ခွဲခြားမှတ်စု](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ဖြင့် လုပ်ဆောင်ပြီး **Hotel_Reviews_NLP.csv** ဖြစ်လာသည်။
4. Hotel_Reviews_NLP.csv ကို အောက်ပါ NLP စိန်ခေါ်မှုတွင် အသုံးပြုပါ။

### နိဂုံးချုပ်

သင်စတင်ခဲ့သောအခါ၊ သင်သည် ကော်လံများနှင့် ဒေတာများပါရှိသော ဒေတာဖိုင်တစ်ခု ရှိခဲ့သည်၊ သို့သော် အားလုံးကို အတည်ပြုနိုင်ခြင်း သို့မဟုတ် အသုံးပြုနိုင်ခြင်း မရှိခဲ့ပါ။ သင်သည် ဒေတာကို စူးစမ်းခဲ့ပြီး၊ မလိုအပ်သောအရာများကို ဖယ်ရှားခဲ့ပြီး၊ tag များကို အသုံးဝင်သောအရာများအဖြစ် ပြောင်းလဲခဲ့ပြီး၊ သင်၏ကိုယ်ပိုင် ပျမ်းမျှချက်များကို တွက်ချက်ခဲ့ပြီး၊ စိတ်ခံစားမှု ကော်လံများကို ထည့်သွင်းခဲ့ပြီး၊ သဘာဝစာသားကို အလုပ်လုပ်စေခြင်းနှင့် ပတ်သက်သော စိတ်ဝင်စားဖွယ် အရာများကို သင်ယူခဲ့မည်ဟု မျှော်လင့်ပါသည်။

## [သင်ခန်းစာပြီးနောက် မေးခွန်း](https://ff-quizzes.netlify.app/en/ml/)

## စိန်ခေါ်မှု

ယခု သင်၏ ဒေတာဖိုင်ကို စိတ်ခံစားမှုအတွက် ခွဲခြားပြီးဖြစ်သောကြောင့်၊ သင်သည် သင်ယူခဲ့သော မဟာဗျူဟာများ (တစ်ခုပြီးတစ်ခု ခွဲခြားခြင်း၊ ဥပမာအားဖြင့်) ကို အသုံးပြု၍ စိတ်ခံစားမှုနှင့် ပတ်သက်သော ပုံစံများကို သတ်မှတ်နိုင်မည်လား ကြိုးစားပါ။

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း

[ဒီ Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) ကို လေ့လာပြီး စိတ်ခံစားမှုကို စူးစမ်းရန် အခြားကိရိယာများကို အသုံးပြုပါ။

## လုပ်ငန်း

[အခြားဒေတာဖိုင်ကို ကြိုးစားပါ](assignment.md)

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်ခြင်းတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာရှိသော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ ပရော်ဖက်ရှင်နယ် ဘာသာပြန်ခြင်းကို အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအလွတ်များ သို့မဟုတ် အနားလွဲမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။