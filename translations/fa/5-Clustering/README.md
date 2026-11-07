# مدل‌های خوشه‌بندی برای یادگیری ماشین

خوشه‌بندی یک وظیفه یادگیری ماشین است که در آن به دنبال پیدا کردن اشیایی هستیم که به یکدیگر شباهت داشته باشند و آن‌ها را در گروه‌هایی به نام خوشه‌ها دسته‌بندی کنیم. چیزی که خوشه‌بندی را از دیگر روش‌ها در یادگیری ماشین متمایز می‌کند این است که همه چیز به‌صورت خودکار اتفاق می‌افتد، در واقع می‌توان گفت که دقیقاً عکس یادگیری نظارت‌شده است.

## موضوع منطقه‌ای: مدل‌های خوشه‌بندی برای سلیقه موسیقیایی مخاطبان نیجریه‌ای 🎧

مخاطبان متنوع نیجریه سلیقه‌های موسیقی متنوعی دارند. با استفاده از داده‌های استخراج‌شده از اسپاتیفای (الهام گرفته از [این مقاله](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)، بیایید به برخی از موسیقی‌های محبوب در نیجریه نگاهی بیندازیم. این مجموعه داده شامل اطلاعاتی درباره امتیاز «رقص‌پذیری»، «آکوستیک بودن»، بلندی صدا، «گفتاری بودن»، محبوبیت و انرژی آهنگ‌های مختلف است. کشف الگوهایی در این داده‌ها جالب خواهد بود!

![یک صفحه گردان](../../../translated_images/fa/turntable.f2b86b13c53302dc.webp)

> عکس از <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">مارسلا لاسکوسکی</a> در <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">آن‌اسپلش</a>
  
در این سری درس‌ها، روش‌های جدیدی برای تحلیل داده‌ها با استفاده از تکنیک‌های خوشه‌بندی کشف خواهید کرد. خوشه‌بندی به‌ویژه زمانی مفید است که مجموعه داده شما برچسب نداشته باشد. اگر برچسب داشته باشد، تکنیک‌های دسته‌بندی مانند آنچه در درس‌های قبلی آموخته‌اید ممکن است مفیدتر باشد. اما در مواردی که به دنبال گروه‌بندی داده‌های بدون برچسب هستید، خوشه‌بندی راه بسیار خوبی برای کشف الگوها است.

> ابزارهای کم‌کد مفیدی وجود دارند که می‌توانند به شما کمک کنند تا درباره کار با مدل‌های خوشه‌بندی بیاموزید. برای این کار [Azure ML را امتحان کنید](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## درس‌ها

۱. [مقدمه‌ای بر خوشه‌بندی](1-Visualize/README.md)
۲. [خوشه‌بندی K-میانگین](2-K-Means/README.md)

## اعتبارها

این دروس با 🎶 توسط [Jen Looper](https://www.twitter.com/jenlooper) نوشته شده‌اند و بازبینی‌های مفیدی توسط [Rishit Dagli](https://rishit_dagli/) و [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) ارائه شده است.

مجموعه داده‌های [آهنگ‌های نیجریه‌ای](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) از سایت کگل به‌دست آمده که از اسپاتیفای استخراج شده‌اند.

مثال‌های مفید K-Means که به ایجاد این درس کمک کردند شامل این [کاوش گل سوسن](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)، این [دفترچه مقدماتی](https://www.kaggle.com/prashant111/k-means-clustering-with-python)، و این [مثال فرضی سازمان غیردولتی](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) هستند.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**سلب مسئولیت**:
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما در تلاش برای دقت هستیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادرستی‌هایی باشند. سند اصلی به زبان مادری خود باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حیاتی، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما در قبال هرگونه سوء تفاهم یا برداشت نادرست ناشی از استفاده از این ترجمه مسئولیتی نداریم.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->