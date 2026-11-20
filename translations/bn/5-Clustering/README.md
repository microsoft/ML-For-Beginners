# যন্ত্র শিক্ষা জন্য ক্লাস্টারিং মডেলসমূহ

ক্লাস্টারিং হল একটি যন্ত্র শিক্ষার কাজ যেখানে এটি দেখতে চেষ্টা করে এমন বস্তুগুলি খুঁজে বের করতে যা একে অপরের অনুরূপ এবং এগুলিকে ক্লাস্টার নামে পরিচিত গোষ্ঠীগুলিতে ভাগ করে। যন্ত্র শিক্ষার অন্যান্য পদ্ধতি থেকে ক্লাস্টারিংকে আলাদা করে তা হল যে কাজগুলি স্বয়ংক্রিয়ভাবে ঘটে, আসলে বলা যায় এটি তত্ত্বাবধানকৃত শিক্ষার বিপরীত। 

## আঞ্চলিক বিষয়: নাইজেরিয়ার শ্রোতাদের সঙ্গীত রুচি অনুযায়ী ক্লাস্টারিং মডেল 🎧

নাইজেরিয়ার বৈচিত্র্যময় শ্রোতাদের বৈচিত্র্যময় সঙ্গীত রুচি রয়েছে। Spotify থেকে সংগ্রহ করা ডেটা ব্যবহার করে (এই [লেখা](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) দ্বারা অনুপ্রাণিত), চলুন নাইজেরিয়ায় জনপ্রিয় কিছু সঙ্গীত দেখি। এই ডেটাসেটে বিভিন্ন গানের 'ডান্সাবিলিটি' স্কোর, 'একাউস্টিকনেস', লাউডনেস, 'স্পিচিনেস', জনপ্রিয়তা এবং শক্তি সম্পর্কিত তথ্য রয়েছে। এই ডেটাতে প্যাটার্ন আবিষ্কার করা আকর্ষণীয় হবে!

![একটি টার্নটেবল](../../../translated_images/bn/turntable.f2b86b13c53302dc.webp)

> ফটো <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">মার্সেলা লাসকোস্কি</a> দ্বারা <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">আনস্প্ল্যাশ</a> এ
  
এই পাঠের সিরিজে, আপনি ক্লাস্টারিং কৌশল ব্যবহার করে ডেটা বিশ্লেষণের নতুন উপায় আবিষ্কার করবেন। ক্লাস্টারিং বিশেষত উপকারী যখন আপনার ডেটাসেটে লেবেল নেই। যদি লেবেল থাকে, তবে পূর্ববর্তী পাঠে শেখা শ্রেণীবিন্যাস কৌশলগুলি আরও উপকারী হতে পারে। তবে যেখানে আপনি লেবেলবিহীন ডেটাকে গোষ্ঠীতে বিভক্ত করতে চান, সেখানে ক্লাস্টারিং প্যাটার্ন আবিষ্কারের একটি চমৎকার উপায়।

> ক্লাস্টারিং মডেল নিয়ে কাজ শেখার জন্য কিছু দরকারী লো-কোড টুল রয়েছে। এই কাজের জন্য [Azure ML ব্যবহার করে দেখুন](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## পাঠসমূহ

1. [ক্লাস্টারিং পরিচিতি](1-Visualize/README.md)
2. [কে-মিনস ক্লাস্টারিং](2-K-Means/README.md)

## স্বীকৃতি

এই পাঠগুলি 🎶 সহ [জেন লুপার](https://www.twitter.com/jenlooper) দ্বারা রচিত হয়েছে এবং সহায়ক পর্যালোচনা করেছেন [রিশিত দাগলি](https://rishit_dagli/) এবং [মুহাম্মদ সাকিব খান ইনান](https://twitter.com/Sakibinan)।

[নাইজেরিয়ান গানগুলো](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ডেটাসেটটি Kaggle থেকে Spotify থেকে সংগ্রহ করা হয়েছে।

এই পাঠ তৈরি করতে সাহায্যকারী দরকারী কে-মিনস উদাহরণগুলির মধ্যে রয়েছে এই [আইরিস এক্সপ্লোরেশন](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), এই [পরিচিতিমূলক নোটবুক](https://www.kaggle.com/prashant111/k-means-clustering-with-python), এবং এই [সাংকল্পিক এনজিও উদাহরণ](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)।

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**অস্বীকৃতি**:
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনূদিত হয়েছে। যদিও আমরা শুদ্ধতার জন্য চেষ্টা করি, অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল নথিটি তার স্বভাষায় কর্তৃত্বপূর্ণ উৎস হিসেবে বিবেচিত হওয়া উচিত। গুরুত্বপূর্ণ তথ্যের জন্য পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদের ব্যবহারে প্রয়োজনীয় ভুল বোঝাবুঝি বা ভুল ব্যাখ্যার জন্য আমরা দায়বদ্ধ নই।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->