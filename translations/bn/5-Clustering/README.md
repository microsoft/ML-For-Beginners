<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T20:51:58+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "bn"
}
-->
# মেশিন লার্নিংয়ের জন্য ক্লাস্টারিং মডেল

ক্লাস্টারিং একটি মেশিন লার্নিং কাজ যেখানে একে এমন বস্তু খুঁজে বের করতে হয় যেগুলো একে অপরের সাথে সাদৃশ্যপূর্ণ এবং সেগুলোকে ক্লাস্টার নামে পরিচিত গ্রুপে ভাগ করা হয়। ক্লাস্টারিংয়ের বিশেষত্ব হলো এটি স্বয়ংক্রিয়ভাবে ঘটে, যা মেশিন লার্নিংয়ের অন্যান্য পদ্ধতির থেকে আলাদা। আসলে, এটি সুপারভাইজড লার্নিংয়ের বিপরীত বলা যেতে পারে। 

## আঞ্চলিক বিষয়: নাইজেরিয়ান শ্রোতাদের সঙ্গীত রুচির জন্য ক্লাস্টারিং মডেল 🎧

নাইজেরিয়ার বৈচিত্র্যময় শ্রোতাদের সঙ্গীতের রুচিও বৈচিত্র্যময়। Spotify থেকে সংগৃহীত ডেটা ব্যবহার করে (এই [প্রবন্ধটি](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) দ্বারা অনুপ্রাণিত), আসুন নাইজেরিয়ায় জনপ্রিয় কিছু সঙ্গীত দেখি। এই ডেটাসেটে বিভিন্ন গানের 'danceability' স্কোর, 'acousticness', শব্দের উচ্চতা, 'speechiness', জনপ্রিয়তা এবং এনার্জি সম্পর্কিত তথ্য অন্তর্ভুক্ত রয়েছে। এই ডেটায় প্যাটার্ন খুঁজে বের করাটা বেশ মজার হবে!

![একটি টার্নটেবিল](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.bn.jpg)

> ছবি তুলেছেন <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>-এ
  
এই পাঠগুলোর মাধ্যমে, আপনি ক্লাস্টারিং টেকনিক ব্যবহার করে ডেটা বিশ্লেষণের নতুন উপায় শিখবেন। ক্লাস্টারিং বিশেষভাবে কার্যকর যখন আপনার ডেটাসেটে লেবেল থাকে না। যদি লেবেল থাকে, তাহলে পূর্ববর্তী পাঠে শেখা ক্লাসিফিকেশন টেকনিকগুলো বেশি কার্যকর হতে পারে। কিন্তু যখন আপনি লেবেলবিহীন ডেটাকে গ্রুপ করতে চান, তখন ক্লাস্টারিং প্যাটার্ন আবিষ্কারের জন্য একটি চমৎকার পদ্ধতি।

> ক্লাস্টারিং মডেলের সাথে কাজ করার জন্য কিছু কার্যকর লো-কোড টুল রয়েছে। এই কাজের জন্য [Azure ML চেষ্টা করুন](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## পাঠসমূহ

1. [ক্লাস্টারিংয়ের পরিচিতি](1-Visualize/README.md)
2. [কে-মিন্স ক্লাস্টারিং](2-K-Means/README.md)

## কৃতজ্ঞতা

এই পাঠগুলো 🎶 দিয়ে লিখেছেন [Jen Looper](https://www.twitter.com/jenlooper), এবং সহায়ক পর্যালোচনা করেছেন [Rishit Dagli](https://rishit_dagli) এবং [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)।

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ডেটাসেটটি Kaggle থেকে সংগৃহীত, যা Spotify থেকে স্ক্র্যাপ করা হয়েছে।

এই পাঠ তৈরিতে সহায়ক কিছু কার্যকর কে-মিন্স উদাহরণ অন্তর্ভুক্ত ছিল, যেমন এই [আইরিস বিশ্লেষণ](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), এই [পরিচিতিমূলক নোটবুক](https://www.kaggle.com/prashant111/k-means-clustering-with-python), এবং এই [কাল্পনিক এনজিও উদাহরণ](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)।

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিক অনুবাদ প্রদানের চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা তার জন্য দায়বদ্ধ থাকব না।