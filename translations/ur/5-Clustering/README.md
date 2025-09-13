<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T13:18:41+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "ur"
}
-->
# مشین لرننگ کے لیے کلسٹرنگ ماڈلز

کلسٹرنگ مشین لرننگ کا ایک ایسا کام ہے جس میں اشیاء کو تلاش کیا جاتا ہے جو ایک دوسرے سے مشابہت رکھتی ہیں اور انہیں کلسٹرز کہلانے والے گروپس میں تقسیم کیا جاتا ہے۔ کلسٹرنگ کو مشین لرننگ کے دیگر طریقوں سے جو چیز مختلف بناتی ہے وہ یہ ہے کہ یہ عمل خودکار طور پر ہوتا ہے۔ حقیقت میں، یہ کہنا مناسب ہوگا کہ یہ سپروائزڈ لرننگ کے بالکل برعکس ہے۔

## علاقائی موضوع: نائجیریا کے سامعین کے موسیقی کے ذوق کے لیے کلسٹرنگ ماڈلز 🎧

نائجیریا کے متنوع سامعین کے موسیقی کے ذوق بھی متنوع ہیں۔ اس سلسلے میں، Spotify سے حاصل کردہ ڈیٹا کا استعمال کرتے ہوئے (جیسا کہ [اس مضمون](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) سے متاثر ہو کر)، آئیے نائجیریا میں مقبول موسیقی پر نظر ڈالیں۔ اس ڈیٹا سیٹ میں مختلف گانوں کے 'ڈانس ایبلٹی' اسکور، 'اکوسٹکنیس'، آواز کی بلندی، 'اسپیچنیس'، مقبولیت اور توانائی کے بارے میں معلومات شامل ہیں۔ اس ڈیٹا میں پیٹرنز دریافت کرنا دلچسپ ہوگا!

![ایک ٹرن ٹیبل](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.ur.jpg)

> تصویر از <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">مارسیلا لاسکوسکی</a>، <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> پر
  
ان اسباق کی سیریز میں، آپ کلسٹرنگ تکنیکوں کا استعمال کرتے ہوئے ڈیٹا کا تجزیہ کرنے کے نئے طریقے دریافت کریں گے۔ کلسٹرنگ خاص طور پر اس وقت مفید ہوتی ہے جب آپ کے ڈیٹا سیٹ میں لیبلز نہ ہوں۔ اگر لیبلز موجود ہوں، تو پچھلے اسباق میں سیکھے گئے کلاسیفکیشن تکنیک زیادہ مفید ہو سکتی ہیں۔ لیکن ان صورتوں میں جہاں آپ بغیر لیبل والے ڈیٹا کو گروپ کرنا چاہتے ہیں، کلسٹرنگ پیٹرنز دریافت کرنے کا ایک بہترین طریقہ ہے۔

> ایسے مفید لو-کوڈ ٹولز موجود ہیں جو آپ کو کلسٹرنگ ماڈلز کے ساتھ کام کرنے کے بارے میں سیکھنے میں مدد دے سکتے ہیں۔ اس کام کے لیے [Azure ML آزمائیں](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## اسباق

1. [کلسٹرنگ کا تعارف](1-Visualize/README.md)
2. [کے-میینز کلسٹرنگ](2-K-Means/README.md)

## کریڈٹس

یہ اسباق 🎶 کے ساتھ [جن لوپر](https://www.twitter.com/jenlooper) نے لکھے ہیں، اور ان پر [رشیت داگلی](https://rishit_dagli) اور [محمد ثاقب خان عنان](https://twitter.com/Sakibinan) کی مددگار نظرثانی شامل ہے۔

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ڈیٹا سیٹ Kaggle سے حاصل کیا گیا ہے، جو Spotify سے حاصل کردہ ہے۔

کے-میینز کے مفید مثالیں جنہوں نے اس سبق کو تخلیق کرنے میں مدد دی، ان میں یہ [آئرس ایکسپلوریشن](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)، یہ [تعارفی نوٹ بک](https://www.kaggle.com/prashant111/k-means-clustering-with-python)، اور یہ [فرضی این جی او مثال](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) شامل ہیں۔

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا غیر درستیاں ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ ہم اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے ذمہ دار نہیں ہیں۔