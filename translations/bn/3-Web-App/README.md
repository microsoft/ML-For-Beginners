<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-08-29T21:34:15+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "bn"
}
-->
# আপনার মডেল ব্যবহার করার জন্য একটি ওয়েব অ্যাপ তৈরি করুন

এই পাঠ্যক্রমের এই অংশে, আপনি একটি প্রয়োগকৃত মেশিন লার্নিং বিষয়ের সাথে পরিচিত হবেন: কীভাবে আপনার Scikit-learn মডেলকে একটি ফাইলে সংরক্ষণ করবেন যা একটি ওয়েব অ্যাপ্লিকেশনের মধ্যে পূর্বাভাস দেওয়ার জন্য ব্যবহার করা যেতে পারে। মডেলটি সংরক্ষণ করার পরে, আপনি শিখবেন কীভাবে এটি Flask-এ তৈরি একটি ওয়েব অ্যাপে ব্যবহার করবেন। প্রথমে, আপনি কিছু ডেটা ব্যবহার করে একটি মডেল তৈরি করবেন যা UFO দেখার ঘটনার উপর ভিত্তি করে! এরপর, আপনি একটি ওয়েব অ্যাপ তৈরি করবেন যা আপনাকে সেকেন্ডের একটি সংখ্যা, একটি অক্ষাংশ এবং দ্রাঘিমাংশের মান ইনপুট দিয়ে পূর্বাভাস করতে দেবে কোন দেশ UFO দেখার রিপোর্ট করেছে।

![UFO পার্কিং](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.bn.jpg)

ছবি তুলেছেন <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">মাইকেল হেরেন</a> <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">আনস্প্ল্যাশ</a>-এ

## পাঠসমূহ

1. [একটি ওয়েব অ্যাপ তৈরি করুন](1-Web-App/README.md)

## কৃতজ্ঞতা

"একটি ওয়েব অ্যাপ তৈরি করুন" ♥️ দিয়ে লিখেছেন [জেন লুপার](https://twitter.com/jenlooper)।

♥️ কুইজগুলো লিখেছেন রোহান রাজ।

ডেটাসেটটি নেওয়া হয়েছে [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) থেকে।

ওয়েব অ্যাপ আর্কিটেকচার আংশিকভাবে প্রস্তাবিত হয়েছে [এই প্রবন্ধে](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) এবং [এই রিপোতে](https://github.com/abhinavsagar/machine-learning-deployment) অভিনব সাগরের দ্বারা।

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিকতার জন্য চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।