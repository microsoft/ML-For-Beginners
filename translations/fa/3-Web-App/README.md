<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-03T23:43:29+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "fa"
}
-->
# ساخت یک اپلیکیشن وب برای استفاده از مدل یادگیری ماشین

در این بخش از دوره آموزشی، با یک موضوع کاربردی در یادگیری ماشین آشنا خواهید شد: چگونگی ذخیره مدل Scikit-learn به‌صورت یک فایل که بتوان از آن برای پیش‌بینی‌ها در یک اپلیکیشن وب استفاده کرد. پس از ذخیره مدل، یاد می‌گیرید که چگونه از آن در یک اپلیکیشن وب ساخته‌شده با Flask استفاده کنید. ابتدا مدلی را با استفاده از داده‌هایی که درباره مشاهده بشقاب‌پرنده‌ها هستند ایجاد می‌کنید! سپس، یک اپلیکیشن وب می‌سازید که به شما امکان می‌دهد با وارد کردن تعداد ثانیه‌ها به همراه مقادیر عرض و طول جغرافیایی، پیش‌بینی کنید که کدام کشور مشاهده بشقاب‌پرنده را گزارش داده است.

![پارک بشقاب‌پرنده](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.fa.jpg)

عکس از <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">مایکل هرن</a> در <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## درس‌ها

1. [ساخت یک اپلیکیشن وب](1-Web-App/README.md)

## منابع

"ساخت یک اپلیکیشن وب" با ♥️ توسط [جن لوپر](https://twitter.com/jenlooper) نوشته شده است.

♥️ آزمون‌ها توسط روهان راج نوشته شده‌اند.

داده‌ها از [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) گرفته شده‌اند.

معماری اپلیکیشن وب تا حدی با الهام از [این مقاله](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) و [این مخزن](https://github.com/abhinavsagar/machine-learning-deployment) توسط آبیناو ساگار پیشنهاد شده است.

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما برای دقت تلاش می‌کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادرستی‌هایی باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما هیچ مسئولیتی در قبال سوءتفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.