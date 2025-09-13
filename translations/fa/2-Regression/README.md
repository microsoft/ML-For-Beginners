<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-03T22:15:39+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "fa"
}
-->
# مدل‌های رگرسیون برای یادگیری ماشین
## موضوع منطقه‌ای: مدل‌های رگرسیون برای قیمت کدو تنبل در آمریکای شمالی 🎃

در آمریکای شمالی، کدو تنبل‌ها اغلب برای هالووین به شکل چهره‌های ترسناک تراشیده می‌شوند. بیایید درباره این سبزیجات جذاب بیشتر بدانیم!

![jack-o-lanterns](../../../translated_images/jack-o-lanterns.181c661a9212457d7756f37219f660f1358af27554d856e5a991f16b4e15337c.fa.jpg)
> عکس از <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> در <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## آنچه خواهید آموخت

[![Introduction to Regression](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Regression Introduction video - Click to Watch!")
> 🎥 برای مشاهده ویدئوی معرفی سریع این درس، روی تصویر بالا کلیک کنید

درس‌های این بخش انواع رگرسیون را در زمینه یادگیری ماشین پوشش می‌دهند. مدل‌های رگرسیون می‌توانند _رابطه_ بین متغیرها را تعیین کنند. این نوع مدل می‌تواند مقادیر مانند طول، دما یا سن را پیش‌بینی کند و در نتیجه روابط بین متغیرها را با تحلیل نقاط داده کشف کند.

در این مجموعه درس‌ها، تفاوت‌های بین رگرسیون خطی و لجستیک را کشف خواهید کرد و خواهید آموخت که چه زمانی باید یکی را بر دیگری ترجیح دهید.

[![ML for beginners - Introduction to Regression models for Machine Learning](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "ML for beginners - Introduction to Regression models for Machine Learning")

> 🎥 برای مشاهده ویدئوی کوتاه معرفی مدل‌های رگرسیون، روی تصویر بالا کلیک کنید.

در این گروه از درس‌ها، شما آماده انجام وظایف یادگیری ماشین خواهید شد، از جمله تنظیم Visual Studio Code برای مدیریت نوت‌بوک‌ها، محیط رایج برای دانشمندان داده. شما با Scikit-learn، یک کتابخانه برای یادگیری ماشین، آشنا خواهید شد و اولین مدل‌های خود را خواهید ساخت، با تمرکز بر مدل‌های رگرسیون در این فصل.

> ابزارهای کم‌کد مفیدی وجود دارند که می‌توانند به شما در یادگیری کار با مدل‌های رگرسیون کمک کنند. [Azure ML را برای این کار امتحان کنید](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### درس‌ها

1. [ابزارهای کار](1-Tools/README.md)
2. [مدیریت داده‌ها](2-Data/README.md)
3. [رگرسیون خطی و چندجمله‌ای](3-Linear/README.md)
4. [رگرسیون لجستیک](4-Logistic/README.md)

---
### اعتبارها

"یادگیری ماشین با رگرسیون" با ♥️ توسط [Jen Looper](https://twitter.com/jenlooper) نوشته شده است.

♥️ مشارکت‌کنندگان آزمون شامل: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) و [Ornella Altunyan](https://twitter.com/ornelladotcom)

مجموعه داده کدو تنبل توسط [این پروژه در Kaggle](https://www.kaggle.com/usda/a-year-of-pumpkin-prices) پیشنهاد شده است و داده‌های آن از [گزارش‌های استاندارد بازارهای ترمینال محصولات خاص](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) که توسط وزارت کشاورزی ایالات متحده توزیع شده‌اند، گرفته شده است. ما برخی نقاط مربوط به رنگ بر اساس نوع را اضافه کرده‌ایم تا توزیع را نرمال کنیم. این داده‌ها در حوزه عمومی قرار دارند.

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما برای دقت تلاش می‌کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادقتی‌ها باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما هیچ مسئولیتی در قبال سوءتفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.