<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-04T21:15:30+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "bn"
}
-->
# হোটেল রিভিউ দিয়ে সেন্টিমেন্ট অ্যানালাইসিস - ডেটা প্রক্রিয়াকরণ

এই অংশে আপনি আগের পাঠে শেখা কৌশল ব্যবহার করে একটি বড় ডেটাসেটের উপর কিছু অনুসন্ধানমূলক ডেটা বিশ্লেষণ করবেন। বিভিন্ন কলামের কার্যকারিতা সম্পর্কে ভালো ধারণা পাওয়ার পর, আপনি শিখবেন:

- কীভাবে অপ্রয়োজনীয় কলামগুলো সরানো যায়
- কীভাবে বিদ্যমান কলামগুলোর উপর ভিত্তি করে নতুন ডেটা গণনা করা যায়
- কীভাবে চূড়ান্ত চ্যালেঞ্জের জন্য ডেটাসেট সংরক্ষণ করা যায়

## [পূর্ব-পাঠ কুইজ](https://ff-quizzes.netlify.app/en/ml/)

### ভূমিকা

এখন পর্যন্ত আপনি শিখেছেন যে টেক্সট ডেটা সংখ্যাসূচক ডেটার মতো নয়। যদি এটি মানুষের লেখা বা বলা টেক্সট হয়, তবে এটি বিশ্লেষণ করে প্যাটার্ন, ফ্রিকোয়েন্সি, সেন্টিমেন্ট এবং অর্থ খুঁজে বের করা যায়। এই পাঠ আপনাকে একটি বাস্তব ডেটাসেট এবং একটি বাস্তব চ্যালেঞ্জে নিয়ে যাবে: **[ইউরোপের ৫১৫ হাজার হোটেল রিভিউ ডেটা](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, যা [CC0: পাবলিক ডোমেইন লাইসেন্স](https://creativecommons.org/publicdomain/zero/1.0/) অন্তর্ভুক্ত। এটি Booking.com থেকে পাবলিক সোর্স থেকে সংগ্রহ করা হয়েছে। ডেটাসেটের নির্মাতা হলেন Jiashen Liu।

### প্রস্তুতি

আপনার প্রয়োজন হবে:

* Python 3 ব্যবহার করে .ipynb নোটবুক চালানোর সক্ষমতা
* pandas
* NLTK, [যা আপনাকে লোকালভাবে ইন্সটল করতে হবে](https://www.nltk.org/install.html)
* Kaggle-এ উপলব্ধ ডেটাসেট [ইউরোপের ৫১৫ হাজার হোটেল রিভিউ ডেটা](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)। আনজিপ করার পর এটি প্রায় ২৩০ MB। এটি এই NLP পাঠের সাথে সম্পর্কিত `/data` ফোল্ডারে ডাউনলোড করুন।

## অনুসন্ধানমূলক ডেটা বিশ্লেষণ

এই চ্যালেঞ্জটি ধরে নেয় যে আপনি সেন্টিমেন্ট অ্যানালাইসিস এবং অতিথির রিভিউ স্কোর ব্যবহার করে একটি হোটেল রিকমেন্ডেশন বট তৈরি করছেন। আপনি যে ডেটাসেট ব্যবহার করবেন তাতে ৬টি শহরের ১৪৯৩টি হোটেলের রিভিউ অন্তর্ভুক্ত রয়েছে।

Python, হোটেল রিভিউ ডেটাসেট এবং NLTK-এর সেন্টিমেন্ট অ্যানালাইসিস ব্যবহার করে আপনি জানতে পারবেন:

* রিভিউতে সবচেয়ে বেশি ব্যবহৃত শব্দ এবং বাক্যাংশ কী?
* হোটেল সম্পর্কিত অফিসিয়াল *ট্যাগ* কি রিভিউ স্কোরের সাথে সম্পর্কিত (যেমন, *Family with young children* ট্যাগের জন্য কি *Solo traveller* ট্যাগের তুলনায় বেশি নেতিবাচক রিভিউ রয়েছে, যা হয়তো ইঙ্গিত করে যে এটি *Solo travellers*-এর জন্য বেশি উপযুক্ত)?
* NLTK সেন্টিমেন্ট স্কোর কি হোটেল রিভিউয়ের সংখ্যাসূচক স্কোরের সাথে 'মিল' করে?

#### ডেটাসেট

আপনার ডাউনলোড করা এবং লোকালভাবে সংরক্ষিত ডেটাসেটটি এক্সপ্লোর করুন। ফাইলটি VS Code বা Excel-এর মতো কোনো এডিটরে খুলুন।

ডেটাসেটের হেডারগুলো নিম্নরূপ:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

এগুলোকে সহজে বিশ্লেষণের জন্য গ্রুপ করা হয়েছে:
##### হোটেল কলাম

* `Hotel_Name`, `Hotel_Address`, `lat` (অক্ষাংশ), `lng` (দ্রাঘিমাংশ)
  * *lat* এবং *lng* ব্যবহার করে আপনি Python দিয়ে একটি ম্যাপ তৈরি করতে পারেন যেখানে হোটেলের অবস্থান দেখানো হবে (সম্ভবত নেতিবাচক এবং ইতিবাচক রিভিউয়ের জন্য আলাদা রঙ ব্যবহার করে)
  * Hotel_Address আমাদের জন্য সরাসরি খুব একটা কার্যকর নয়, এবং আমরা সম্ভবত এটি একটি দেশের সাথে প্রতিস্থাপন করব যাতে সহজে সাজানো এবং অনুসন্ধান করা যায়

**হোটেল মেটা-রিভিউ কলাম**

* `Average_Score`
  * ডেটাসেট নির্মাতার মতে, এই কলামটি হলো *হোটেলের গড় স্কোর, যা গত বছরের সর্বশেষ মন্তব্যের ভিত্তিতে গণনা করা হয়েছে*। এটি একটি অস্বাভাবিক পদ্ধতি মনে হতে পারে, তবে এটি সংগ্রহ করা ডেটা, তাই আপাতত আমরা এটিকে গ্রহণ করতে পারি।
  
  ✅ এই ডেটার অন্যান্য কলামের ভিত্তিতে গড় স্কোর গণনার আরেকটি পদ্ধতি কি আপনি ভাবতে পারেন?

* `Total_Number_of_Reviews`
  * এই হোটেলটি কতগুলো রিভিউ পেয়েছে তার মোট সংখ্যা - এটি স্পষ্ট নয় (কোড না লিখে) যে এটি ডেটাসেটে থাকা রিভিউগুলোর কথা উল্লেখ করছে কিনা।
* `Additional_Number_of_Scoring`
  * এর অর্থ হলো একটি স্কোর দেওয়া হয়েছে কিন্তু রিভিউয়ার কোনো ইতিবাচক বা নেতিবাচক রিভিউ লেখেননি

**রিভিউ কলাম**

- `Reviewer_Score`
  - এটি একটি সংখ্যাসূচক মান যা সর্বাধিক ১ দশমিক স্থান পর্যন্ত থাকে এবং এর মান ২.৫ থেকে ১০ এর মধ্যে থাকে
  - কেন ২.৫ সর্বনিম্ন স্কোর সম্ভব তা ব্যাখ্যা করা হয়নি
- `Negative_Review`
  - যদি কোনো রিভিউয়ার কিছু না লেখেন, এই ক্ষেত্রটি "**No Negative**" দেখাবে
  - লক্ষ্য করুন যে একটি রিভিউয়ার নেতিবাচক রিভিউ কলামে ইতিবাচক মন্তব্য লিখতে পারেন (যেমন, "এই হোটেল সম্পর্কে খারাপ কিছু নেই")
- `Review_Total_Negative_Word_Counts`
  - বেশি নেতিবাচক শব্দের সংখ্যা কম স্কোর নির্দেশ করে (সেন্টিমেন্টালিটি যাচাই না করেই)
- `Positive_Review`
  - যদি কোনো রিভিউয়ার কিছু না লেখেন, এই ক্ষেত্রটি "**No Positive**" দেখাবে
  - লক্ষ্য করুন যে একটি রিভিউয়ার ইতিবাচক রিভিউ কলামে নেতিবাচক মন্তব্য লিখতে পারেন (যেমন, "এই হোটেল সম্পর্কে ভালো কিছুই নেই")
- `Review_Total_Positive_Word_Counts`
  - বেশি ইতিবাচক শব্দের সংখ্যা বেশি স্কোর নির্দেশ করে (সেন্টিমেন্টালিটি যাচাই না করেই)
- `Review_Date` এবং `days_since_review`
  - একটি রিভিউয়ের তাজা বা পুরনো অবস্থা নির্ধারণ করা যেতে পারে (পুরনো রিভিউ হয়তো নতুন রিভিউয়ের মতো সঠিক নাও হতে পারে কারণ হোটেল ব্যবস্থাপনা পরিবর্তিত হয়েছে, সংস্কার করা হয়েছে, বা একটি সুইমিং পুল যোগ করা হয়েছে ইত্যাদি)
- `Tags`
  - এগুলো ছোট বর্ণনা যা রিভিউয়ার তাদের অতিথির ধরণ (যেমন, একক বা পরিবার), তাদের রুমের ধরণ, থাকার সময়কাল এবং কীভাবে রিভিউ জমা দেওয়া হয়েছে তা বর্ণনা করতে নির্বাচন করতে পারেন।
  - দুর্ভাগ্যবশত, এই ট্যাগগুলো ব্যবহার করা সমস্যাজনক, নিচের অংশে তাদের কার্যকারিতা নিয়ে আলোচনা করা হয়েছে

**রিভিউয়ার কলাম**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - এটি একটি রিকমেন্ডেশন মডেলে একটি ফ্যাক্টর হতে পারে, উদাহরণস্বরূপ, যদি আপনি নির্ধারণ করতে পারেন যে শত শত রিভিউ দেওয়া বেশি সক্রিয় রিভিউয়াররা নেতিবাচক হওয়ার চেয়ে ইতিবাচক হওয়ার সম্ভাবনা বেশি। তবে, কোনো নির্দিষ্ট রিভিউয়ের রিভিউয়ারকে একটি ইউনিক কোড দিয়ে চিহ্নিত করা হয়নি, এবং তাই এটি একটি সেট রিভিউয়ের সাথে সংযুক্ত করা সম্ভব নয়। ১০০ বা তার বেশি রিভিউ দেওয়া ৩০ জন রিভিউয়ার রয়েছে, তবে এটি রিকমেন্ডেশন মডেলে কীভাবে সহায়তা করতে পারে তা বোঝা কঠিন।
- `Reviewer_Nationality`
  - কিছু মানুষ হয়তো মনে করতে পারেন যে কিছু জাতি ইতিবাচক বা নেতিবাচক রিভিউ দেওয়ার ক্ষেত্রে বেশি প্রবণ। তবে, এমন অ্যানেকডোটাল ধারণা আপনার মডেলে অন্তর্ভুক্ত করার ক্ষেত্রে সতর্ক থাকুন। এগুলো জাতিগত (এবং কখনও কখনও বর্ণগত) স্টেরিওটাইপ, এবং প্রতিটি রিভিউয়ার তাদের অভিজ্ঞতার ভিত্তিতে একটি রিভিউ লিখেছেন। এটি হয়তো অনেক দৃষ্টিভঙ্গি দিয়ে ফিল্টার করা হয়েছে যেমন তাদের পূর্ববর্তী হোটেল থাকার অভিজ্ঞতা, ভ্রমণের দূরত্ব, এবং তাদের ব্যক্তিগত মনোভাব। তাদের জাতি রিভিউ স্কোরের কারণ ছিল এমনটি ভাবা কঠিন।

##### উদাহরণ

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | এটি বর্তমানে একটি হোটেল নয় বরং একটি নির্মাণ সাইট। আমি দীর্ঘ ভ্রমণের পর বিশ্রাম নেওয়ার সময় এবং রুমে কাজ করার সময় সকাল থেকে সন্ধ্যা পর্যন্ত অগ্রহণযোগ্য নির্মাণ শব্দ দ্বারা আতঙ্কিত হয়েছি। পাশের রুমে সারাদিন কাজ চলছিল, যেমন জ্যাকহ্যামার ব্যবহার করে। আমি রুম পরিবর্তনের অনুরোধ করেছিলাম কিন্তু কোনো শান্ত রুম পাওয়া যায়নি। পরিস্থিতি আরও খারাপ করার জন্য আমাকে অতিরিক্ত চার্জ করা হয়েছে। আমি সন্ধ্যায় চেক আউট করেছি কারণ আমাকে খুব সকালে ফ্লাইট নিতে হয়েছিল এবং একটি উপযুক্ত বিল পেয়েছি। একদিন পরে হোটেল আমার সম্মতি ছাড়া অতিরিক্ত চার্জ করেছে যা বুকিং মূল্যের বেশি। এটি একটি ভয়ানক জায়গা। নিজেকে শাস্তি দেবেন না এখানে বুকিং করে। | কিছুই নেই। ভয়ানক জায়গা। দূরে থাকুন। | ব্যবসায়িক ভ্রমণ। দম্পতি। স্ট্যান্ডার্ড ডাবল রুম। ২ রাত অবস্থান। |

যেমনটি আপনি দেখতে পাচ্ছেন, এই অতিথি হোটেলে খুবই খারাপ অভিজ্ঞতা পেয়েছেন। হোটেলের গড় স্কোর ৭.৮ এবং ১৯৪৫টি রিভিউ রয়েছে, কিন্তু এই রিভিউয়ার এটি ২.৫ দিয়েছেন এবং তাদের অবস্থান কতটা নেতিবাচক ছিল তা নিয়ে ১১৫টি শব্দ লিখেছেন। যদি তারা Positive_Review কলামে কিছুই না লিখতেন, আপনি অনুমান করতে পারতেন যে কিছুই ইতিবাচক ছিল না, কিন্তু তারা সতর্কবার্তা হিসেবে ৭টি শব্দ লিখেছেন। যদি আমরা শব্দের সংখ্যা গণনা করি অর্থ বা শব্দের সেন্টিমেন্টের পরিবর্তে, তাহলে রিভিউয়ারের উদ্দেশ্য সম্পর্কে একটি বিকৃত ধারণা পেতে পারি। আশ্চর্যজনকভাবে, তাদের ২.৫ স্কোর বিভ্রান্তিকর, কারণ যদি হোটেল অবস্থান এতটাই খারাপ ছিল, তাহলে কেন কোনো পয়েন্ট দেওয়া হলো? ডেটাসেটটি ঘনিষ্ঠভাবে পরীক্ষা করলে, আপনি দেখতে পাবেন যে সর্বনিম্ন সম্ভব স্কোর ২.৫, ০ নয়। সর্বোচ্চ সম্ভব স্কোর ১০।

##### ট্যাগ

উপরের মতো, প্রথম নজরে `Tags` ব্যবহার করে ডেটা শ্রেণীবদ্ধ করার ধারণাটি অর্থবহ মনে হয়। দুর্ভাগ্যবশত, এই ট্যাগগুলো স্ট্যান্ডার্ডাইজড নয়, যার অর্থ একটি নির্দিষ্ট হোটেলে অপশনগুলো হতে পারে *Single room*, *Twin room*, এবং *Double room*, কিন্তু পরবর্তী হোটেলে এগুলো হতে পারে *Deluxe Single Room*, *Classic Queen Room*, এবং *Executive King Room*। এগুলো হয়তো একই জিনিস, কিন্তু এত বেশি বৈচিত্র্য রয়েছে যে পছন্দগুলো হয়ে যায়:

1. সমস্ত শর্তকে একটি একক স্ট্যান্ডার্ডে পরিবর্তন করার চেষ্টা করা, যা খুবই কঠিন, কারণ প্রতিটি ক্ষেত্রে রূপান্তরের পথ কী হবে তা স্পষ্ট নয় (যেমন, *Classic single room* কে *Single room* এ ম্যাপ করা যায়, কিন্তু *Superior Queen Room with Courtyard Garden or City View* ম্যাপ করা অনেক কঠিন)

1. আমরা একটি NLP পদ্ধতি নিতে পারি এবং *Solo*, *Business Traveller*, বা *Family with young kids* এর মতো নির্দিষ্ট শর্তগুলোর ফ্রিকোয়েন্সি পরিমাপ করতে পারি এবং সেগুলোকে প্রতিটি হোটেলের ক্ষেত্রে বিবেচনায় নিতে পারি

ট্যাগগুলো সাধারণত (কিন্তু সবসময় নয়) একটি একক ক্ষেত্র যেখানে ৫ থেকে ৬টি কমা দ্বারা পৃথক মান থাকে যা *Type of trip*, *Type of guests*, *Type of room*, *Number of nights*, এবং *Type of device review was submitted on* এর সাথে সামঞ্জস্যপূর্ণ। তবে, কিছু রিভিউয়ার প্রতিটি ক্ষেত্র পূরণ করেন না (তারা একটি ফাঁকা রাখতে পারেন), তাই মানগুলো সবসময় একই ক্রমে থাকে না।

উদাহরণস্বরূপ, *Type of group* নিন। এই ক্ষেত্রটিতে `Tags` কলামে ১০২৫টি ইউনিক সম্ভাবনা রয়েছে, এবং দুর্ভাগ্যবশত এর মধ্যে কিছুই গ্রুপের কথা উল্লেখ করে না (কিছু রুমের ধরণ ইত্যাদি)। যদি আপনি শুধুমাত্র সেই মানগুলো ফিল্টার করেন যা পরিবার উল্লেখ করে, ফলাফলগুলোতে অনেক *Family room* ধরণের ফলাফল থাকে। যদি আপনি *with* শব্দটি অন্তর্ভুক্ত করেন, অর্থাৎ *Family with* মানগুলো গণনা করেন, ফলাফলগুলো আরও ভালো হয়, যেখানে ৫১৫,০০০ ফলাফলের মধ্যে ৮০,০০০টির বেশি "Family with young children" বা "Family with older children" বাক্যাংশ অন্তর্ভুক্ত করে।

এর অর্থ হলো ট্যাগ কলামটি আমাদের জন্য সম্পূর্ণ অকার্যকর নয়, তবে এটি কার্যকর করতে কিছু কাজ করতে হবে।

##### গড় হোটেল স্কোর

ডেটাসেটের সাথে কিছু অদ্ভুততা বা অসঙ্গতি রয়েছে যা আমি বুঝতে পারি না, তবে এখানে চিত্রিত করা হয়েছে যাতে আপনি আপনার মডেল তৈরি করার সময় সেগুলো সম্পর্কে সচেতন থাকেন। যদি আপনি এটি বুঝতে পারেন, দয়া করে আলোচনা বিভাগে আমাদের জানান!

ডেটাসেটে গড় স্কোর এবং রিভিউ সংখ্যার সাথে সম্পর্কিত নিম্নলিখিত কলামগুলো রয়েছে:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

এই ডেটাসেটে সবচেয়ে বেশি রিভিউ থাকা একক হোটেল হলো *Britannia International Hotel Canary Wharf*, যার ৫১৫,০০০ রিভিউয়ের মধ্যে ৪৭৮৯টি রয়েছে। কিন্তু যদি আমরা এই হোটেলের `Total_Number_of_Reviews` মান দেখি, এটি ৯০৮৬। আপনি অনুমান করতে পারেন যে অনেক বেশি স্কোর রয়েছে যেগুলো রিভিউ ছাড়াই, তাই সম্ভবত আমরা `Additional_Number_of_Scoring` কলামের মান যোগ করব। সেই মানটি ২৬৮২, এবং ৪৭৮৯ যোগ করলে আমরা পাই ৭,৪৭১, যা এখনও `Total_Number_of_Reviews` থেকে ১৬১৫ কম।

যদি আপনি `Average_Score` কলামটি নেন, আপনি অনুমান করতে পারেন এটি ডেটাসেটে থাকা রিভিউগুলোর গড়, কিন্তু Kaggle-এর বর্ণনা অনুযায়ী এটি হলো "*হোটেলের গড় স্কোর, যা গত বছরের সর্বশেষ মন্তব্যের ভিত্তিতে গণনা করা হয়েছে*।" এটি খুব একটা কার্যকর মনে হয় না, তবে আমরা ডেটাসেটে থাকা রিভিউ স্কোরের ভিত্তিতে আমাদের নিজস্ব গড় স্কোর গণনা করতে পারি। একই হোটেলকে উদাহরণ হিসেবে ব্যবহার করে, গড় হোটেল স্কোর দেওয়া হয়েছে ৭.১, কিন্তু ডেটাসেটে থাকা রিভিউ স্কোরের ভিত্তিতে গণনা করা স্কোর হলো ৬.৮। এটি কাছাকাছি, কিন্তু একই মান নয়, এবং আমরা শুধুমাত্র অনুমান করতে পারি যে `Additional_Number_of_Scoring` রিভিউগুলো গড়কে ৭.১-এ উন্নীত করেছে। দুর্ভাগ্যবশত, সেই অনুমান পরীক্ষা বা প্রমাণ করার কোনো উপায় না থাকায়, `Average_Score`, `Additional_Number_of_Scoring` এবং `Total_Number_of_Reviews` ব্যবহার করা বা বিশ্বাস করা কঠিন।

বিষয়টি আরও জটিল করতে, সবচেয়ে বেশি রিভিউ থাকা দ্বিতীয় হোটেলের গড় স্কোর ৮.১২ এবং ডেটাসেটের `Average_Score` হলো ৮.১। এটি কি সঠিক স্কোর একটি কাকতালীয় ঘটনা, নাকি প্রথম হোটেলটি একটি অসঙ্গতি?

এই সম্ভাবনার উপর ভিত্তি করে যে এই হোটেলটি একটি ব্যতিক্রম হতে পারে, এবং হয়তো বেশিরভাগ মানগুলো সঠিকভাবে মিলে যায় (কিন্তু কিছু কারণে কিছু মান মিলে না), আমরা পরবর্তী অংশে একটি ছোট প্রোগ্রাম লিখব যাতে ডেটাসেটে থাকা মানগুলো এক্সপ্লোর করা যায় এবং মানগুলোর সঠিক ব্যবহার (বা অ-ব্যবহার) নির্ধারণ করা যায়।
> 🚨 সতর্কতার একটি নোট
>
> এই ডেটাসেট নিয়ে কাজ করার সময় আপনি এমন কোড লিখবেন যা টেক্সট থেকে কিছু হিসাব করবে, কিন্তু আপনাকে নিজে টেক্সট পড়তে বা বিশ্লেষণ করতে হবে না। এটি NLP-এর মূল ধারণা, অর্থ বা অনুভূতি ব্যাখ্যা করা, যেখানে মানুষের হস্তক্ষেপের প্রয়োজন হয় না। তবে, এটি সম্ভব যে আপনি কিছু নেতিবাচক রিভিউ পড়বেন। আমি আপনাকে এটি না করার পরামর্শ দেব, কারণ আপনাকে তা করতে হবে না। কিছু রিভিউ হাস্যকর বা অপ্রাসঙ্গিক নেতিবাচক হোটেল রিভিউ হতে পারে, যেমন "আবহাওয়া ভালো ছিল না", যা হোটেলের বা কারও নিয়ন্ত্রণের বাইরে। কিন্তু কিছু রিভিউয়ের একটি অন্ধকার দিকও রয়েছে। কখনও কখনও নেতিবাচক রিভিউগুলো বর্ণবাদী, লিঙ্গবাদী, বা বয়সবাদী হতে পারে। এটি দুর্ভাগ্যজনক, কিন্তু একটি পাবলিক ওয়েবসাইট থেকে স্ক্র্যাপ করা ডেটাসেটে এটি প্রত্যাশিত। কিছু রিভিউয়ার এমন মন্তব্য রেখে যান যা আপনি অস্বস্তিকর, অপ্রিয়, বা বিরক্তিকর বলে মনে করতে পারেন। অনুভূতি পরিমাপের জন্য কোডকে কাজ করতে দেওয়া ভালো, নিজে পড়ে বিরক্ত হওয়ার চেয়ে। তবে, এটি উল্লেখযোগ্য যে এমন মন্তব্যকারী সংখ্যা খুবই কম, কিন্তু তারা রয়েছেই।
## অনুশীলন - ডেটা অনুসন্ধান
### ডেটা লোড করুন

ডেটাকে ভিজ্যুয়ালি পর্যবেক্ষণ করা যথেষ্ট হয়েছে, এখন আপনি কিছু কোড লিখে উত্তর বের করবেন! এই অংশে pandas লাইব্রেরি ব্যবহার করা হয়েছে। আপনার প্রথম কাজ হলো নিশ্চিত করা যে আপনি CSV ডেটা লোড এবং পড়তে পারেন। pandas লাইব্রেরি একটি দ্রুত CSV লোডার সরবরাহ করে, এবং ফলাফলটি একটি ডেটাফ্রেমে রাখা হয়, যেমন আগের পাঠে দেখেছেন। আমরা যে CSV লোড করছি তাতে অর্ধ মিলিয়নেরও বেশি সারি রয়েছে, কিন্তু মাত্র ১৭টি কলাম। pandas আপনাকে ডেটাফ্রেমের সাথে কাজ করার জন্য অনেক শক্তিশালী উপায় সরবরাহ করে, যার মধ্যে প্রতিটি সারিতে অপারেশন করার ক্ষমতা রয়েছে।

এই পাঠের বাকি অংশে কোড স্নিপেট এবং কোডের ব্যাখ্যা এবং ফলাফল কী বোঝায় তা নিয়ে আলোচনা থাকবে। আপনার কোডের জন্য _notebook.ipynb_ ফাইলটি ব্যবহার করুন।

চলুন শুরু করি আপনি যে ডেটা ফাইলটি ব্যবহার করবেন তা লোড করার মাধ্যমে:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

ডেটা লোড হয়ে গেলে, আমরা এর উপর কিছু অপারেশন করতে পারি। পরবর্তী অংশের জন্য এই কোডটি আপনার প্রোগ্রামের শীর্ষে রাখুন।

## ডেটা অনুসন্ধান

এই ক্ষেত্রে, ডেটা ইতিমধ্যেই *পরিষ্কার*, অর্থাৎ এটি কাজ করার জন্য প্রস্তুত এবং এতে অন্য ভাষার অক্ষর নেই যা শুধুমাত্র ইংরেজি অক্ষর প্রত্যাশা করা অ্যালগরিদমকে বিভ্রান্ত করতে পারে।

✅ আপনাকে এমন ডেটার সাথে কাজ করতে হতে পারে যা NLP কৌশল প্রয়োগ করার আগে কিছু প্রাথমিক প্রক্রিয়াকরণ প্রয়োজন, কিন্তু এই সময় তা নয়। যদি করতে হতো, আপনি কীভাবে অ-ইংরেজি অক্ষর পরিচালনা করতেন?

একবার ডেটা লোড হয়ে গেলে নিশ্চিত করুন যে আপনি কোড দিয়ে এটি অনুসন্ধান করতে পারেন। `Negative_Review` এবং `Positive_Review` কলামগুলোর দিকে মনোযোগ দেওয়া সহজ। এগুলো আপনার NLP অ্যালগরিদমের প্রক্রিয়াকরণের জন্য প্রাকৃতিক টেক্সট দিয়ে পূর্ণ। কিন্তু থামুন! NLP এবং সেন্টিমেন্টে যাওয়ার আগে, নিচের কোডটি অনুসরণ করুন এবং নিশ্চিত করুন যে ডেটাসেটে দেওয়া মানগুলো pandas দিয়ে গণনা করা মানের সাথে মিলে যায়।

## ডেটাফ্রেম অপারেশন

এই পাঠের প্রথম কাজ হলো কিছু কোড লিখে ডেটাফ্রেম পরীক্ষা করা (পরিবর্তন না করে) এবং নিচের দাবিগুলো সঠিক কিনা তা যাচাই করা।

> অনেক প্রোগ্রামিং কাজের মতো, এটি সম্পন্ন করার জন্য বিভিন্ন উপায় রয়েছে, তবে ভালো পরামর্শ হলো এটি যতটা সম্ভব সহজ এবং সহজ উপায়ে করা, বিশেষত যদি এটি ভবিষ্যতে কোডটি বুঝতে সহজ হয়। ডেটাফ্রেমের সাথে, একটি বিস্তৃত API রয়েছে যা প্রায়শই আপনার যা করতে চান তা দক্ষতার সাথে করার উপায় সরবরাহ করবে।

নিচের প্রশ্নগুলোকে কোডিং টাস্ক হিসেবে বিবেচনা করুন এবং সমাধান না দেখে উত্তর দেওয়ার চেষ্টা করুন।

1. আপনি যে ডেটাফ্রেমটি লোড করেছেন তার *shape* প্রিন্ট করুন (shape হলো সারি এবং কলামের সংখ্যা)
2. রিভিউয়ারদের জাতীয়তার ফ্রিকোয়েন্সি গণনা করুন:
   1. `Reviewer_Nationality` কলামের জন্য কতগুলো স্বতন্ত্র মান রয়েছে এবং সেগুলো কী কী?
   2. ডেটাসেটে সবচেয়ে সাধারণ রিভিউয়ার জাতীয়তা কোনটি (দেশ এবং রিভিউ সংখ্যা প্রিন্ট করুন)?
   3. পরবর্তী শীর্ষ ১০টি সবচেয়ে বেশি পাওয়া জাতীয়তা এবং তাদের ফ্রিকোয়েন্সি গণনা কী কী?
3. শীর্ষ ১০টি রিভিউয়ার জাতীয়তার জন্য সবচেয়ে বেশি রিভিউ করা হোটেল কোনটি?
4. ডেটাসেটে প্রতি হোটেলের রিভিউ সংখ্যা (হোটেলের ফ্রিকোয়েন্সি গণনা) কত?
5. যদিও ডেটাসেটে প্রতিটি হোটেলের জন্য একটি `Average_Score` কলাম রয়েছে, আপনি প্রতিটি হোটেলের জন্য সমস্ত রিভিউয়ার স্কোরের গড় পেয়ে একটি গড় স্কোরও গণনা করতে পারেন। আপনার ডেটাফ্রেমে `Calc_Average_Score` শিরোনামের একটি নতুন কলাম যোগ করুন যা সেই গণিত গড় ধারণ করে।
6. কোনো হোটেলের `Average_Score` এবং `Calc_Average_Score` কি একই (১ দশমিক স্থানে রাউন্ড করা)?
   1. একটি Python ফাংশন লিখুন যা একটি Series (সারি) আর্গুমেন্ট হিসেবে নেয় এবং মানগুলো তুলনা করে, যখন মানগুলো সমান নয় তখন একটি বার্তা প্রিন্ট করে। তারপর `.apply()` পদ্ধতি ব্যবহার করে প্রতিটি সারি প্রক্রিয়া করুন।
7. `Negative_Review` কলামের "No Negative" মানের কতগুলো সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।
8. `Positive_Review` কলামের "No Positive" মানের কতগুলো সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।
9. `Positive_Review` কলামের "No Positive" **এবং** `Negative_Review` কলামের "No Negative" মানের কতগুলো সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

### কোড উত্তর

1. আপনি যে ডেটাফ্রেমটি লোড করেছেন তার *shape* প্রিন্ট করুন (shape হলো সারি এবং কলামের সংখ্যা)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. রিভিউয়ারদের জাতীয়তার ফ্রিকোয়েন্সি গণনা করুন:

   1. `Reviewer_Nationality` কলামের জন্য কতগুলো স্বতন্ত্র মান রয়েছে এবং সেগুলো কী কী?
   2. ডেটাসেটে সবচেয়ে সাধারণ রিভিউয়ার জাতীয়তা কোনটি (দেশ এবং রিভিউ সংখ্যা প্রিন্ট করুন)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. পরবর্তী শীর্ষ ১০টি সবচেয়ে বেশি পাওয়া জাতীয়তা এবং তাদের ফ্রিকোয়েন্সি গণনা কী কী?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. শীর্ষ ১০টি রিভিউয়ার জাতীয়তার জন্য সবচেয়ে বেশি রিভিউ করা হোটেল কোনটি?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. ডেটাসেটে প্রতি হোটেলের রিভিউ সংখ্যা (হোটেলের ফ্রিকোয়েন্সি গণনা) কত?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   আপনি লক্ষ্য করতে পারেন যে *ডেটাসেটে গণনা করা* ফলাফল `Total_Number_of_Reviews` মানের সাথে মেলে না। এটি স্পষ্ট নয় যে এই মানটি ডেটাসেটে হোটেলের মোট রিভিউ সংখ্যা উপস্থাপন করেছিল, কিন্তু সবগুলো স্ক্র্যাপ করা হয়নি, অথবা অন্য কোনো গণনা। এই অস্পষ্টতার কারণে `Total_Number_of_Reviews` মডেলে ব্যবহার করা হয় না।

5. যদিও ডেটাসেটে প্রতিটি হোটেলের জন্য একটি `Average_Score` কলাম রয়েছে, আপনি প্রতিটি হোটেলের জন্য সমস্ত রিভিউয়ার স্কোরের গড় পেয়ে একটি গড় স্কোরও গণনা করতে পারেন। আপনার ডেটাফ্রেমে `Calc_Average_Score` শিরোনামের একটি নতুন কলাম যোগ করুন যা সেই গণিত গড় ধারণ করে। `Hotel_Name`, `Average_Score`, এবং `Calc_Average_Score` কলামগুলো প্রিন্ট করুন।

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   আপনি `Average_Score` মান এবং গণিত গড় স্কোরের মধ্যে পার্থক্য নিয়ে ভাবতে পারেন। আমরা জানি না কেন কিছু মান মিলে যায়, কিন্তু অন্যগুলোতে পার্থক্য রয়েছে। এই ক্ষেত্রে, আমাদের কাছে থাকা রিভিউ স্কোর ব্যবহার করে গড় নিজেই গণনা করা নিরাপদ। তবে পার্থক্যগুলো সাধারণত খুব ছোট। এখানে ডেটাসেট গড় এবং গণিত গড়ের মধ্যে সবচেয়ে বেশি বিচ্যুতি থাকা হোটেলগুলো:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   মাত্র ১টি হোটেলের স্কোরের পার্থক্য ১-এর বেশি, এটি বোঝায় যে আমরা সম্ভবত পার্থক্য উপেক্ষা করতে পারি এবং গণিত গড় স্কোর ব্যবহার করতে পারি।

6. `Negative_Review` কলামের "No Negative" মানের কতগুলো সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

7. `Positive_Review` কলামের "No Positive" মানের কতগুলো সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

8. `Positive_Review` কলামের "No Positive" **এবং** `Negative_Review` কলামের "No Negative" মানের কতগুলো সারি রয়েছে তা গণনা এবং প্রিন্ট করুন।

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## আরেকটি উপায়

ল্যাম্বডা ছাড়া আইটেম গণনা করার আরেকটি উপায়, এবং সারি গণনার জন্য sum ব্যবহার করুন:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   আপনি লক্ষ্য করতে পারেন যে `Negative_Review` এবং `Positive_Review` কলামের "No Negative" এবং "No Positive" মানের ১২৭টি সারি রয়েছে। এর মানে হলো রিভিউয়ার হোটেলকে একটি সংখ্যাসূচক স্কোর দিয়েছেন, কিন্তু ইতিবাচক বা নেতিবাচক কোনো রিভিউ লিখতে অস্বীকার করেছেন। সৌভাগ্যক্রমে এটি একটি ছোট পরিমাণ সারি (৫১৫৭৩৮-এর মধ্যে ১২৭, বা ০.০২%), তাই এটি সম্ভবত আমাদের মডেল বা ফলাফলকে কোনো নির্দিষ্ট দিকে প্রভাবিত করবে না। তবে আপনি হয়তো আশা করেননি যে রিভিউয়ের ডেটাসেটে কোনো রিভিউ ছাড়া সারি থাকবে, তাই এটি আবিষ্কার করার জন্য ডেটা অনুসন্ধান করা মূল্যবান।

এখন আপনি ডেটাসেটটি অনুসন্ধান করেছেন, পরবর্তী পাঠে আপনি ডেটা ফিল্টার করবেন এবং কিছু সেন্টিমেন্ট অ্যানালাইসিস যোগ করবেন।

---
## 🚀চ্যালেঞ্জ

এই পাঠটি দেখায়, যেমন আমরা আগের পাঠে দেখেছি, আপনার ডেটা এবং এর ত্রুটিগুলো বুঝতে কতটা গুরুত্বপূর্ণ। বিশেষত টেক্সট-ভিত্তিক ডেটা খুব সতর্কতার সাথে পরীক্ষা করা উচিত। বিভিন্ন টেক্সট-ভারী ডেটাসেট খনন করুন এবং দেখুন আপনি এমন ক্ষেত্র আবিষ্কার করতে পারেন কিনা যা একটি মডেলে পক্ষপাত বা বিকৃত সেন্টিমেন্ট প্রবর্তন করতে পারে।

## [পোস্ট-লেকচার কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পর্যালোচনা এবং স্ব-অধ্যয়ন

[NLP নিয়ে এই লার্নিং পাথ](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) গ্রহণ করুন এবং টুলগুলো আবিষ্কার করুন যা স্পিচ এবং টেক্সট-ভারী মডেল তৈরি করার সময় চেষ্টা করতে পারেন।

## অ্যাসাইনমেন্ট 

[NLTK](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিকতার জন্য চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।