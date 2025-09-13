<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T22:56:19+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "fa"
}
-->
# مدل‌های خوشه‌بندی برای یادگیری ماشین

خوشه‌بندی یکی از وظایف یادگیری ماشین است که در آن تلاش می‌شود اشیایی که به یکدیگر شباهت دارند پیدا شده و در گروه‌هایی به نام خوشه‌ها قرار گیرند. تفاوت اصلی خوشه‌بندی با سایر روش‌های یادگیری ماشین در این است که این فرآیند به صورت خودکار انجام می‌شود؛ در واقع می‌توان گفت که خوشه‌بندی نقطه مقابل یادگیری نظارت‌شده است.

## موضوع منطقه‌ای: مدل‌های خوشه‌بندی برای سلیقه موسیقی مخاطبان نیجریه 🎧

مخاطبان متنوع نیجریه دارای سلیقه‌های موسیقی متنوعی هستند. با استفاده از داده‌هایی که از اسپاتیفای جمع‌آوری شده‌اند (با الهام از [این مقاله](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421))، بیایید نگاهی به برخی از موسیقی‌های محبوب در نیجریه بیندازیم. این مجموعه داده شامل اطلاعاتی درباره امتیاز 'رقص‌پذیری'، 'آکوستیک بودن'، بلندی صدا، 'گفتاری بودن'، محبوبیت و انرژی آهنگ‌های مختلف است. کشف الگوها در این داده‌ها می‌تواند بسیار جالب باشد!

![یک صفحه‌گردان](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.fa.jpg)

> عکس از <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">مارسلا لاسکوسکی</a> در <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
در این مجموعه درس‌ها، شما روش‌های جدیدی برای تحلیل داده‌ها با استفاده از تکنیک‌های خوشه‌بندی کشف خواهید کرد. خوشه‌بندی به‌ویژه زمانی مفید است که مجموعه داده شما فاقد برچسب باشد. اگر داده‌ها دارای برچسب باشند، تکنیک‌های طبقه‌بندی که در درس‌های قبلی یاد گرفتید ممکن است مفیدتر باشند. اما در مواردی که به دنبال گروه‌بندی داده‌های بدون برچسب هستید، خوشه‌بندی راهی عالی برای کشف الگوها است.

> ابزارهای کم‌کد مفیدی وجود دارند که می‌توانند به شما در یادگیری کار با مدل‌های خوشه‌بندی کمک کنند. [Azure ML را برای این کار امتحان کنید](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## درس‌ها

1. [مقدمه‌ای بر خوشه‌بندی](1-Visualize/README.md)
2. [خوشه‌بندی K-Means](2-K-Means/README.md)

## اعتبارها

این درس‌ها با 🎶 توسط [جن لوپر](https://www.twitter.com/jenlooper) نوشته شده‌اند و با بازبینی‌های مفید [ریشیت داگلی](https://rishit_dagli) و [محمد ساکب خان اینان](https://twitter.com/Sakibinan) تکمیل شده‌اند.

مجموعه داده [آهنگ‌های نیجریه‌ای](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) از سایت Kaggle و از اسپاتیفای جمع‌آوری شده است.

مثال‌های مفید K-Means که در ایجاد این درس کمک کردند شامل این [بررسی گل زنبق](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)، این [دفترچه مقدماتی](https://www.kaggle.com/prashant111/k-means-clustering-with-python)، و این [مثال فرضی سازمان غیردولتی](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) هستند.

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما برای دقت تلاش می‌کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادقتی‌هایی باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما هیچ مسئولیتی در قبال سوءتفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.