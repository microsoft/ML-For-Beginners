<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T22:39:12+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "bn"
}
-->
# হোটেল রিভিউ দিয়ে সেন্টিমেন্ট অ্যানালাইসিস

আপনি ইতিমধ্যে ডেটাসেটটি বিস্তারিতভাবে অন্বেষণ করেছেন, এখন সময় এসেছে কলামগুলো ফিল্টার করার এবং NLP প্রযুক্তি ব্যবহার করে হোটেল সম্পর্কে নতুন অন্তর্দৃষ্টি লাভ করার।

## [পূর্ব-লেকচার কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### ফিল্টারিং এবং সেন্টিমেন্ট অ্যানালাইসিস অপারেশন

আপনি হয়তো লক্ষ্য করেছেন, ডেটাসেটে কিছু সমস্যা রয়েছে। কিছু কলামে অপ্রয়োজনীয় তথ্য রয়েছে, অন্যগুলো ভুল মনে হচ্ছে। যদি সেগুলো সঠিকও হয়, তাহলে কীভাবে সেগুলো গণনা করা হয়েছে তা অস্পষ্ট, এবং আপনার নিজস্ব গণনার মাধ্যমে উত্তরগুলো স্বাধীনভাবে যাচাই করা সম্ভব নয়।

## অনুশীলন: আরও কিছু ডেটা প্রসেসিং

ডেটা আরও একটু পরিষ্কার করুন। এমন কলাম যোগ করুন যা পরে কাজে লাগবে, অন্য কলামের মান পরিবর্তন করুন, এবং কিছু কলাম সম্পূর্ণভাবে বাদ দিন।

1. প্রাথমিক কলাম প্রসেসিং

   1. `lat` এবং `lng` বাদ দিন

   2. `Hotel_Address` এর মানগুলো নিম্নলিখিত মানগুলো দিয়ে প্রতিস্থাপন করুন (যদি ঠিকানায় শহর এবং দেশের নাম থাকে, তাহলে শুধুমাত্র শহর এবং দেশ রাখুন)।

      ডেটাসেটে শুধুমাত্র এই শহর এবং দেশগুলো রয়েছে:

      আমস্টারডাম, নেদারল্যান্ডস

      বার্সেলোনা, স্পেন

      লন্ডন, যুক্তরাজ্য

      মিলান, ইতালি

      প্যারিস, ফ্রান্স

      ভিয়েনা, অস্ট্রিয়া 

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

      এখন আপনি দেশ-স্তরের ডেটা কুয়েরি করতে পারেন:

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

2. হোটেল মেটা-রিভিউ কলাম প্রসেসিং

  1. `Additional_Number_of_Scoring` বাদ দিন

  1. `Total_Number_of_Reviews` প্রতিস্থাপন করুন সেই হোটেলের জন্য ডেটাসেটে থাকা মোট রিভিউ সংখ্যা দিয়ে 

  1. `Average_Score` আমাদের নিজস্ব গণনা করা স্কোর দিয়ে প্রতিস্থাপন করুন

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. রিভিউ কলাম প্রসেসিং

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` এবং `days_since_review` বাদ দিন

   2. `Reviewer_Score`, `Negative_Review`, এবং `Positive_Review` যেমন আছে তেমনই রাখুন,
     
   3. `Tags` আপাতত রাখুন

     - আমরা পরবর্তী অংশে ট্যাগগুলোর উপর আরও কিছু ফিল্টারিং অপারেশন করব এবং তারপর ট্যাগগুলো বাদ দেওয়া হবে

4. রিভিউয়ার কলাম প্রসেসিং

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` বাদ দিন
  
  2. `Reviewer_Nationality` রাখুন

### ট্যাগ কলাম

`Tag` কলামটি সমস্যাযুক্ত কারণ এটি একটি তালিকা (টেক্সট আকারে) যা কলামে সংরক্ষিত। দুর্ভাগ্যবশত এই কলামের উপ-অংশগুলোর ক্রম এবং সংখ্যা সবসময় একই থাকে না। ৫১৫,০০০টি সারি এবং ১৪২৭টি হোটেল রয়েছে, এবং প্রতিটি রিভিউয়ারের জন্য কিছুটা ভিন্ন অপশন থাকে, যা মানুষের জন্য সঠিক বাক্যাংশ চিহ্নিত করা কঠিন করে তোলে। এখানেই NLP কার্যকর। আপনি টেক্সট স্ক্যান করতে পারেন এবং সবচেয়ে সাধারণ বাক্যাংশগুলো খুঁজে বের করতে পারেন এবং সেগুলো গণনা করতে পারেন।

দুর্ভাগ্যবশত, আমরা একক শব্দে আগ্রহী নই, বরং বহু-শব্দের বাক্যাংশে (যেমন *Business trip*)। এত বড় ডেটাতে (৬৭৬২৬৪৬টি শব্দ) একটি বহু-শব্দের ফ্রিকোয়েন্সি ডিস্ট্রিবিউশন অ্যালগরিদম চালানো অত্যন্ত সময়সাপেক্ষ হতে পারে, তবে ডেটা না দেখে মনে হবে এটি একটি প্রয়োজনীয় ব্যয়। এখানেই এক্সপ্লোরেটরি ডেটা অ্যানালাইসিস কাজে আসে, কারণ আপনি ট্যাগগুলোর একটি নমুনা যেমন `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` দেখেছেন, আপনি প্রশ্ন করতে পারেন যে প্রক্রিয়াটি উল্লেখযোগ্যভাবে কমানো সম্ভব কিনা। সৌভাগ্যক্রমে, এটি সম্ভব - তবে প্রথমে আপনাকে কিছু ধাপ অনুসরণ করতে হবে যাতে আগ্রহের ট্যাগগুলো নিশ্চিত করা যায়।

### ট্যাগ ফিল্টারিং

মনে রাখবেন যে ডেটাসেটের লক্ষ্য হলো সেন্টিমেন্ট এবং এমন কলাম যোগ করা যা আপনাকে সেরা হোটেল বেছে নিতে সাহায্য করবে (আপনার জন্য বা হয়তো কোনো ক্লায়েন্টের জন্য যে আপনাকে একটি হোটেল রিকমেন্ডেশন বট তৈরি করতে বলেছে)। আপনাকে নিজেকে জিজ্ঞাসা করতে হবে যে ট্যাগগুলো চূড়ান্ত ডেটাসেটে উপযোগী কিনা। এখানে একটি ব্যাখ্যা দেওয়া হলো (যদি আপনার ডেটাসেটের প্রয়োজন অন্য কারণে হয়, তাহলে ভিন্ন ট্যাগগুলো থাকতে পারে/বাদ দেওয়া হতে পারে):

1. ট্রিপের ধরন প্রাসঙ্গিক, এবং এটি রাখা উচিত
2. অতিথি দলের ধরন গুরুত্বপূর্ণ, এবং এটি রাখা উচিত
3. অতিথি যে রুম, স্যুট, বা স্টুডিওতে থেকেছে তা অপ্রাসঙ্গিক (সব হোটেলে মূলত একই রকম রুম থাকে)
4. রিভিউ যে ডিভাইসে জমা দেওয়া হয়েছে তা অপ্রাসঙ্গিক
5. রিভিউয়ার কত রাত থেকেছেন তা *প্রাসঙ্গিক* হতে পারে যদি আপনি দীর্ঘ সময় থাকার সঙ্গে হোটেল পছন্দ করার সম্পর্ক স্থাপন করেন, তবে এটি একটি অনুমান, এবং সম্ভবত অপ্রাসঙ্গিক

সংক্ষেপে, **২ ধরনের ট্যাগ রাখুন এবং অন্যগুলো সরিয়ে দিন**।

প্রথমে, আপনি ট্যাগগুলো গণনা করতে চান না যতক্ষণ না সেগুলো আরও ভালো ফরম্যাটে থাকে, তাই স্কয়ার ব্র্যাকেট এবং কোটেশন চিহ্ন সরিয়ে ফেলুন। এটি করার জন্য বিভিন্ন উপায় রয়েছে, তবে আপনি দ্রুততম উপায়টি চান কারণ এটি অনেক ডেটা প্রক্রিয়া করতে দীর্ঘ সময় নিতে পারে। সৌভাগ্যক্রমে, pandas-এর একটি সহজ উপায় রয়েছে প্রতিটি ধাপ সম্পন্ন করার।

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

প্রতিটি ট্যাগ এমন কিছু হয়ে যায়: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`। 

এরপর একটি সমস্যা দেখা দেয়। কিছু রিভিউ বা সারিতে ৫টি কলাম থাকে, কিছুতে ৩টি, কিছুতে ৬টি। এটি ডেটাসেট তৈরির ফলাফল এবং এটি ঠিক করা কঠিন। আপনি প্রতিটি বাক্যাংশের ফ্রিকোয়েন্সি গণনা করতে চান, তবে সেগুলো প্রতিটি রিভিউতে ভিন্ন ক্রমে থাকে, তাই গণনা ভুল হতে পারে, এবং একটি হোটেল একটি ট্যাগ পেতে পারে না যা এটি প্রাপ্য ছিল।

এর পরিবর্তে আপনি ভিন্ন ক্রমকে আমাদের সুবিধায় ব্যবহার করবেন, কারণ প্রতিটি ট্যাগ বহু-শব্দের হলেও কমা দিয়ে পৃথক করা হয়েছে! এর সহজতম উপায় হলো ৬টি অস্থায়ী কলাম তৈরি করা যেখানে প্রতিটি ট্যাগ তার ক্রম অনুযায়ী কলামে প্রবেশ করানো হবে। এরপর আপনি ৬টি কলাম একত্রিত করে একটি বড় কলামে পরিণত করবেন এবং `value_counts()` পদ্ধতি চালাবেন। এটি প্রিন্ট করলে আপনি দেখতে পাবেন ২৪২৮টি ইউনিক ট্যাগ ছিল। এখানে একটি ছোট নমুনা:

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

`Submitted from a mobile device` এর মতো কিছু সাধারণ ট্যাগ আমাদের কোনো কাজে আসে না, তাই এটি গণনা করার আগে সরিয়ে দেওয়া বুদ্ধিমানের কাজ হতে পারে, তবে এটি এত দ্রুত অপারেশন যে আপনি সেগুলো রেখে দিতে পারেন এবং উপেক্ষা করতে পারেন।

### থাকার সময়ের ট্যাগ সরানো

এই ট্যাগগুলো সরানো প্রথম ধাপ, এটি বিবেচনার জন্য ট্যাগের মোট সংখ্যা সামান্য কমিয়ে দেয়। মনে রাখবেন আপনি সেগুলো ডেটাসেট থেকে সরাচ্ছেন না, শুধু রিভিউ ডেটাসেটে গণনা/রাখার জন্য বিবেচনা থেকে সরাচ্ছেন।

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

রুম, স্যুট, স্টুডিও, অ্যাপার্টমেন্ট ইত্যাদির একটি বিশাল বৈচিত্র্য রয়েছে। এগুলো সবই মূলত একই জিনিস বোঝায় এবং আপনার জন্য প্রাসঙ্গিক নয়, তাই সেগুলো বিবেচনা থেকে সরিয়ে দিন।

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

শেষে, এবং এটি আনন্দদায়ক (কারণ এটি খুব বেশি প্রসেসিং নেয়নি), আপনি নিম্নলিখিত *উপযোগী* ট্যাগগুলো পাবেন:

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

আপনি যুক্তি দিতে পারেন যে `Travellers with friends` মূলত `Group` এর সমান এবং এটি একত্রিত করা উচিত, এবং এটি উপরের মতো একত্রিত করা ন্যায্য হবে। সঠিক ট্যাগ চিহ্নিত করার কোডটি [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) এ রয়েছে।

শেষ ধাপ হলো প্রতিটি ট্যাগের জন্য নতুন কলাম তৈরি করা। এরপর, প্রতিটি রিভিউ সারির জন্য, যদি `Tag` কলামটি নতুন কলামের একটির সঙ্গে মিলে যায়, তাহলে ১ যোগ করুন, যদি না মিলে, তাহলে ০ যোগ করুন। চূড়ান্ত ফলাফল হবে কতজন রিভিউয়ার এই হোটেলটি (সমষ্টিগতভাবে) ব্যবসা বনাম অবসর, বা একটি পোষা প্রাণী নিয়ে আসার জন্য বেছে নিয়েছেন তার একটি গণনা, এবং এটি একটি হোটেল সুপারিশ করার সময় উপযোগী তথ্য।

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

### আপনার ফাইল সংরক্ষণ করুন

শেষে, ডেটাসেটটি এখন যেমন আছে তেমনই একটি নতুন নামে সংরক্ষণ করুন।

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## সেন্টিমেন্ট অ্যানালাইসিস অপারেশন

এই চূড়ান্ত অংশে, আপনি রিভিউ কলামগুলোর উপর সেন্টিমেন্ট অ্যানালাইসিস প্রয়োগ করবেন এবং ফলাফলগুলো একটি ডেটাসেটে সংরক্ষণ করবেন।

## অনুশীলন: ফিল্টার করা ডেটা লোড এবং সংরক্ষণ করুন

মনে রাখবেন এখন আপনি পূর্ববর্তী অংশে সংরক্ষিত ফিল্টার করা ডেটাসেটটি লোড করছেন, **মূল ডেটাসেটটি নয়**।

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

### স্টপ ওয়ার্ড সরানো

যদি আপনি `Negative` এবং `Positive` রিভিউ কলামগুলোর উপর সেন্টিমেন্ট অ্যানালাইসিস চালান, এটি দীর্ঘ সময় নিতে পারে। একটি শক্তিশালী টেস্ট ল্যাপটপে দ্রুত CPU সহ পরীক্ষা করা হলে, এটি ১২ - ১৪ মিনিট সময় নিয়েছে, নির্ভর করে কোন সেন্টিমেন্ট লাইব্রেরি ব্যবহার করা হয়েছে। এটি একটি (আপেক্ষিকভাবে) দীর্ঘ সময়, তাই এটি দ্রুত করা সম্ভব কিনা তা তদন্ত করা মূল্যবান।

স্টপ ওয়ার্ড সরানো, বা সাধারণ ইংরেজি শব্দগুলো যা একটি বাক্যের সেন্টিমেন্ট পরিবর্তন করে না, প্রথম ধাপ। এগুলো সরিয়ে দিলে সেন্টিমেন্ট অ্যানালাইসিস দ্রুত চলা উচিত, তবে কম সঠিক হবে না (কারণ স্টপ ওয়ার্ডগুলো সেন্টিমেন্টকে প্রভাবিত করে না, তবে সেগুলো অ্যানালাইসিসকে ধীর করে দেয়)। 

সবচেয়ে দীর্ঘ `Negative` রিভিউ ছিল ৩৯৫টি শব্দ, তবে স্টপ ওয়ার্ড সরানোর পর এটি ১৯৫টি শব্দ।

স্টপ ওয়ার্ড সরানো একটি দ্রুত অপারেশন, ২টি রিভিউ কলাম থেকে ৫১৫,০০০টি সারির স্টপ ওয়ার্ড সরাতে টেস্ট ডিভাইসে ৩.৩ সেকেন্ড সময় লেগেছে। আপনার ডিভাইসের CPU স্পিড, RAM, SSD আছে কিনা, এবং অন্যান্য কিছু ফ্যাক্টরের উপর নির্ভর করে এটি সামান্য বেশি বা কম সময় নিতে পারে। অপারেশনটি আপেক্ষিকভাবে সংক্ষিপ্ত হওয়ায়, যদি এটি সেন্টিমেন্ট অ্যানালাইসিসের সময় উন্নত করে, তাহলে এটি করা মূল্যবান।

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

### সেন্টিমেন্ট অ্যানালাইসিস সম্পাদন করা
এখন আপনাকে নেতিবাচক এবং ইতিবাচক রিভিউ কলামগুলোর জন্য সেন্টিমেন্ট বিশ্লেষণ গণনা করতে হবে এবং ফলাফল দুটি নতুন কলামে সংরক্ষণ করতে হবে। সেন্টিমেন্টের পরীক্ষা হবে এটি রিভিউয়ারের স্কোরের সাথে তুলনা করা। উদাহরণস্বরূপ, যদি সেন্টিমেন্ট বিশ্লেষণ নেতিবাচক রিভিউতে ১ (অত্যন্ত ইতিবাচক সেন্টিমেন্ট) এবং ইতিবাচক রিভিউতে ১ দেখায়, কিন্তু রিভিউয়ার হোটেলকে সর্বনিম্ন স্কোর দেয়, তাহলে হয় রিভিউ টেক্সট স্কোরের সাথে মেলে না, অথবা সেন্টিমেন্ট বিশ্লেষক সঠিকভাবে সেন্টিমেন্ট চিনতে পারেনি। কিছু সেন্টিমেন্ট স্কোর সম্পূর্ণ ভুল হতে পারে, এবং তা ব্যাখ্যা করা সম্ভব হবে, যেমন রিভিউটি অত্যন্ত ব্যঙ্গাত্মক হতে পারে "অবশ্যই আমি গরম ছাড়া একটি রুমে ঘুমাতে ভালোবাসি" এবং সেন্টিমেন্ট বিশ্লেষক এটিকে ইতিবাচক সেন্টিমেন্ট মনে করে, যদিও একজন মানুষ এটি পড়লে বুঝতে পারবে এটি ব্যঙ্গ।

NLTK বিভিন্ন সেন্টিমেন্ট বিশ্লেষক সরবরাহ করে যা দিয়ে শিখতে পারেন, এবং আপনি সেগুলো পরিবর্তন করে দেখতে পারেন সেন্টিমেন্ট আরও সঠিক কিনা। এখানে VADER সেন্টিমেন্ট বিশ্লেষণ ব্যবহার করা হয়েছে।

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

পরবর্তীতে আপনার প্রোগ্রামে যখন সেন্টিমেন্ট গণনা করতে প্রস্তুত হবেন, তখন এটি প্রতিটি রিভিউতে প্রয়োগ করতে পারেন নিম্নলিখিতভাবে:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

এটি আমার কম্পিউটারে প্রায় ১২০ সেকেন্ড সময় নেয়, তবে এটি প্রতিটি কম্পিউটারে ভিন্ন হতে পারে। যদি আপনি ফলাফল প্রিন্ট করতে চান এবং দেখতে চান সেন্টিমেন্ট রিভিউয়ের সাথে মেলে কিনা:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

চ্যালেঞ্জে ব্যবহারের আগে ফাইলটির সাথে করার শেষ কাজটি হলো এটি সংরক্ষণ করা! আপনার নতুন কলামগুলো পুনরায় সাজানোর কথাও বিবেচনা করা উচিত যাতে সেগুলো কাজ করার জন্য সহজ হয় (মানুষের জন্য, এটি একটি কসমেটিক পরিবর্তন)।

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

আপনার পুরো কোডটি চালানো উচিত [বিশ্লেষণ নোটবুকের](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) জন্য (যখন আপনি [ফিল্টারিং নোটবুকটি](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) চালিয়েছেন Hotel_Reviews_Filtered.csv ফাইলটি তৈরি করতে)।

পুনরায় দেখুন, ধাপগুলো হলো:

1. মূল ডেটাসেট ফাইল **Hotel_Reviews.csv** আগের পাঠে [এক্সপ্লোরার নোটবুকের](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) মাধ্যমে অন্বেষণ করা হয়েছে।
2. Hotel_Reviews.csv ফাইলটি [ফিল্টারিং নোটবুকের](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) মাধ্যমে ফিল্টার করা হয়েছে, যার ফলে **Hotel_Reviews_Filtered.csv** তৈরি হয়েছে।
3. Hotel_Reviews_Filtered.csv ফাইলটি [সেন্টিমেন্ট বিশ্লেষণ নোটবুকের](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) মাধ্যমে প্রক্রিয়াজাত করা হয়েছে, যার ফলে **Hotel_Reviews_NLP.csv** তৈরি হয়েছে।
4. NLP চ্যালেঞ্জে নিচে Hotel_Reviews_NLP.csv ব্যবহার করুন।

### উপসংহার

যখন আপনি শুরু করেছিলেন, তখন আপনার কাছে কলাম এবং ডেটাসহ একটি ডেটাসেট ছিল, কিন্তু এর সবকিছু যাচাই বা ব্যবহার করা সম্ভব ছিল না। আপনি ডেটা অন্বেষণ করেছেন, যা প্রয়োজন নেই তা ফিল্টার করেছেন, ট্যাগগুলোকে কিছু উপযোগী জিনিসে রূপান্তর করেছেন, নিজের গড় গণনা করেছেন, কিছু সেন্টিমেন্ট কলাম যোগ করেছেন এবং আশা করি, প্রাকৃতিক টেক্সট প্রক্রিয়াকরণ সম্পর্কে কিছু আকর্ষণীয় জিনিস শিখেছেন।

## [পোস্ট-লেকচার কুইজ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## চ্যালেঞ্জ

এখন যেহেতু আপনার ডেটাসেট সেন্টিমেন্টের জন্য বিশ্লেষণ করা হয়েছে, দেখুন আপনি এই পাঠ্যক্রমে শেখা কৌশলগুলো (সম্ভবত ক্লাস্টারিং?) ব্যবহার করে সেন্টিমেন্টের চারপাশে প্যাটার্ন নির্ধারণ করতে পারেন কিনা।

## পর্যালোচনা ও স্ব-অধ্যয়ন

[এই লার্ন মডিউলটি](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) নিন আরও জানতে এবং টেক্সটে সেন্টিমেন্ট অন্বেষণ করতে বিভিন্ন টুল ব্যবহার করতে।

## অ্যাসাইনমেন্ট

[একটি ভিন্ন ডেটাসেট চেষ্টা করুন](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিক অনুবাদ প্রদানের চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।