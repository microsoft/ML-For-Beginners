<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-04T22:46:59+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "fa"
}
-->
# تحلیل احساسات با بررسی‌های هتل - پردازش داده‌ها

در این بخش، شما از تکنیک‌های درس‌های قبلی برای انجام تحلیل داده‌های اکتشافی روی یک مجموعه داده بزرگ استفاده خواهید کرد. پس از اینکه درک خوبی از مفید بودن ستون‌های مختلف پیدا کردید، یاد خواهید گرفت:

- چگونه ستون‌های غیرضروری را حذف کنید
- چگونه داده‌های جدیدی بر اساس ستون‌های موجود محاسبه کنید
- چگونه مجموعه داده حاصل را برای استفاده در چالش نهایی ذخیره کنید

## [آزمون پیش از درس](https://ff-quizzes.netlify.app/en/ml/)

### مقدمه

تا اینجا یاد گرفته‌اید که داده‌های متنی کاملاً متفاوت از داده‌های عددی هستند. اگر متن توسط انسان نوشته یا گفته شده باشد، می‌توان آن را برای یافتن الگوها، فراوانی‌ها، احساسات و معنا تحلیل کرد. این درس شما را وارد یک مجموعه داده واقعی با یک چالش واقعی می‌کند: **[515K بررسی‌های هتل در اروپا](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** که شامل یک [مجوز عمومی CC0](https://creativecommons.org/publicdomain/zero/1.0/) است. این داده‌ها از منابع عمومی Booking.com استخراج شده‌اند. سازنده این مجموعه داده Jiashen Liu است.

### آماده‌سازی

شما نیاز دارید به:

* توانایی اجرای نوت‌بوک‌های .ipynb با استفاده از Python 3
* pandas
* NLTK، [که باید آن را به صورت محلی نصب کنید](https://www.nltk.org/install.html)
* مجموعه داده که در Kaggle موجود است [515K بررسی‌های هتل در اروپا](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). حجم آن حدود 230 مگابایت پس از استخراج است. آن را در پوشه اصلی `/data` مرتبط با این درس‌های NLP دانلود کنید.

## تحلیل داده‌های اکتشافی

این چالش فرض می‌کند که شما در حال ساخت یک ربات توصیه‌گر هتل با استفاده از تحلیل احساسات و امتیازات بررسی مهمانان هستید. مجموعه داده‌ای که استفاده خواهید کرد شامل بررسی‌های 1493 هتل مختلف در 6 شهر است.

با استفاده از Python، مجموعه داده بررسی‌های هتل، و تحلیل احساسات NLTK می‌توانید موارد زیر را کشف کنید:

* کلمات و عبارات پرکاربرد در بررسی‌ها کدامند؟
* آیا *برچسب‌های* رسمی که یک هتل را توصیف می‌کنند با امتیازات بررسی‌ها همبستگی دارند (مثلاً آیا بررسی‌های منفی بیشتری برای یک هتل خاص توسط *خانواده با کودکان خردسال* نسبت به *مسافر تنها* وجود دارد، که شاید نشان دهد این هتل برای *مسافران تنها* بهتر است؟)
* آیا امتیازات احساسات NLTK با امتیاز عددی بررسی‌کننده هتل مطابقت دارند؟

#### مجموعه داده

بیایید مجموعه داده‌ای که دانلود کرده‌اید و به صورت محلی ذخیره کرده‌اید را بررسی کنیم. فایل را در یک ویرایشگر مانند VS Code یا حتی Excel باز کنید.

سرصفحه‌های مجموعه داده به شرح زیر هستند:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

در اینجا آن‌ها به صورت گروه‌بندی شده برای بررسی آسان‌تر آمده‌اند:
##### ستون‌های هتل

* `Hotel_Name`, `Hotel_Address`, `lat` (عرض جغرافیایی)، `lng` (طول جغرافیایی)
  * با استفاده از *lat* و *lng* می‌توانید نقشه‌ای با Python رسم کنید که مکان‌های هتل را نشان دهد (شاید با کد رنگی برای بررسی‌های منفی و مثبت)
  * Hotel_Address به نظر نمی‌رسد برای ما مفید باشد، و احتمالاً آن را با یک کشور برای مرتب‌سازی و جستجوی آسان‌تر جایگزین خواهیم کرد.

**ستون‌های متا-بررسی هتل**

* `Average_Score`
  * طبق گفته سازنده مجموعه داده، این ستون *امتیاز میانگین هتل، محاسبه شده بر اساس آخرین نظر در سال گذشته* است. این روش محاسبه کمی غیرمعمول به نظر می‌رسد، اما این داده‌ها استخراج شده‌اند، بنابراین فعلاً آن را به همین شکل می‌پذیریم.

  ✅ با توجه به سایر ستون‌های این داده، آیا می‌توانید روش دیگری برای محاسبه امتیاز میانگین پیشنهاد دهید؟

* `Total_Number_of_Reviews`
  * تعداد کل بررسی‌هایی که این هتل دریافت کرده است - مشخص نیست (بدون نوشتن کد) که آیا این به بررسی‌های موجود در مجموعه داده اشاره دارد.
* `Additional_Number_of_Scoring`
  * این به این معناست که امتیاز بررسی داده شده است اما هیچ بررسی مثبت یا منفی توسط بررسی‌کننده نوشته نشده است.

**ستون‌های بررسی**

- `Reviewer_Score`
  - این یک مقدار عددی با حداکثر 1 رقم اعشار بین مقادیر حداقل و حداکثر 2.5 و 10 است.
  - توضیح داده نشده است که چرا 2.5 کمترین امتیاز ممکن است.
- `Negative_Review`
  - اگر بررسی‌کننده چیزی ننوشته باشد، این فیلد دارای "**No Negative**" خواهد بود.
  - توجه داشته باشید که بررسی‌کننده ممکن است یک بررسی مثبت در ستون بررسی منفی بنویسد (مثلاً "هیچ چیز بدی در مورد این هتل وجود ندارد").
- `Review_Total_Negative_Word_Counts`
  - تعداد کلمات منفی بیشتر نشان‌دهنده امتیاز پایین‌تر است (بدون بررسی احساسات).
- `Positive_Review`
  - اگر بررسی‌کننده چیزی ننوشته باشد، این فیلد دارای "**No Positive**" خواهد بود.
  - توجه داشته باشید که بررسی‌کننده ممکن است یک بررسی منفی در ستون بررسی مثبت بنویسد (مثلاً "هیچ چیز خوبی در مورد این هتل وجود ندارد").
- `Review_Total_Positive_Word_Counts`
  - تعداد کلمات مثبت بیشتر نشان‌دهنده امتیاز بالاتر است (بدون بررسی احساسات).
- `Review_Date` و `days_since_review`
  - ممکن است یک معیار تازگی یا کهنگی برای یک بررسی اعمال شود (بررسی‌های قدیمی ممکن است به اندازه بررسی‌های جدید دقیق نباشند زیرا مدیریت هتل تغییر کرده، یا بازسازی انجام شده، یا یک استخر اضافه شده است و غیره).
- `Tags`
  - این‌ها توصیف‌های کوتاهی هستند که بررسی‌کننده ممکن است برای توصیف نوع مهمان بودن خود انتخاب کند (مثلاً تنها یا خانواده)، نوع اتاقی که داشته‌اند، مدت اقامت و نحوه ارسال بررسی.
  - متأسفانه، استفاده از این برچسب‌ها مشکل‌ساز است، بخش زیر که به مفید بودن آن‌ها می‌پردازد را بررسی کنید.

**ستون‌های بررسی‌کننده**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - این ممکن است عاملی در مدل توصیه باشد، برای مثال، اگر بتوانید تعیین کنید که بررسی‌کنندگان پرکارتر با صدها بررسی بیشتر احتمال دارد منفی باشند تا مثبت. با این حال، بررسی‌کننده هر بررسی خاص با یک کد منحصر به فرد شناسایی نمی‌شود و بنابراین نمی‌توان آن را به مجموعه‌ای از بررسی‌ها مرتبط کرد. 30 بررسی‌کننده با 100 یا بیشتر بررسی وجود دارند، اما سخت است که ببینیم این چگونه می‌تواند به مدل توصیه کمک کند.
- `Reviewer_Nationality`
  - برخی ممکن است فکر کنند که ملیت‌های خاصی بیشتر احتمال دارد بررسی مثبت یا منفی بدهند به دلیل تمایل ملی. مراقب باشید که چنین دیدگاه‌های حکایتی را در مدل‌های خود بگنجانید. این‌ها کلیشه‌های ملی (و گاهی نژادی) هستند، و هر بررسی‌کننده فردی بود که بر اساس تجربه خود بررسی نوشت. ممکن است این بررسی از طریق لنزهای مختلفی مانند اقامت‌های قبلی در هتل، فاصله سفر، و خلق‌وخوی شخصی آن‌ها فیلتر شده باشد. فکر کردن به اینکه ملیت آن‌ها دلیل امتیاز بررسی بوده است سخت است که توجیه شود.

##### مثال‌ها

| امتیاز میانگین | تعداد کل بررسی‌ها | امتیاز بررسی‌کننده | بررسی منفی                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | بررسی مثبت                 | برچسب‌ها                                                                                      |
| -------------- | ------------------ | ------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945               | 2.5                 | این در حال حاضر یک هتل نیست بلکه یک سایت ساخت‌وساز است. من از صبح زود و تمام روز با صدای غیرقابل قبول ساخت‌وساز در حالی که بعد از یک سفر طولانی استراحت می‌کردم و در اتاق کار می‌کردم، مورد آزار قرار گرفتم. مردم تمام روز کار می‌کردند، یعنی با چکش‌های برقی در اتاق‌های مجاور. درخواست تغییر اتاق دادم اما هیچ اتاق ساکتی موجود نبود. بدتر از همه، بیش از حد هزینه از من گرفته شد. عصر چک‌اوت کردم چون باید پرواز خیلی زود داشتم و صورتحساب مناسب دریافت کردم. یک روز بعد، هتل بدون رضایت من هزینه اضافی بیش از قیمت رزرو شده را دریافت کرد. این یک مکان وحشتناک است. خودتان را با رزرو اینجا تنبیه نکنید. | هیچ چیز. مکان وحشتناک. دوری کنید. | سفر کاری                                زوج اتاق استاندارد دوتخته اقامت 2 شب |

همان‌طور که می‌بینید، این مهمان اقامت خوشایندی در این هتل نداشته است. هتل امتیاز میانگین خوبی برابر با 7.8 و 1945 بررسی دارد، اما این بررسی‌کننده به آن امتیاز 2.5 داده و 115 کلمه درباره منفی بودن اقامت خود نوشته است. اگر آن‌ها در ستون بررسی مثبت چیزی ننوشته باشند، ممکن است نتیجه بگیرید که هیچ چیز مثبتی وجود نداشته است، اما آن‌ها 7 کلمه هشدار نوشته‌اند. اگر فقط کلمات را بشماریم به جای معنای آن‌ها یا احساسات کلمات، ممکن است دیدی تحریف‌شده از قصد بررسی‌کننده داشته باشیم. عجیب است که امتیاز 2.5 آن‌ها گیج‌کننده است، زیرا اگر اقامت در آن هتل این‌قدر بد بود، چرا اصلاً امتیازی داده‌اند؟ با بررسی دقیق مجموعه داده، خواهید دید که کمترین امتیاز ممکن 2.5 است، نه 0. بیشترین امتیاز ممکن 10 است.

##### برچسب‌ها

همان‌طور که در بالا ذکر شد، در نگاه اول، ایده استفاده از `Tags` برای دسته‌بندی داده‌ها منطقی به نظر می‌رسد. متأسفانه این برچسب‌ها استاندارد نشده‌اند، که به این معناست که در یک هتل، گزینه‌ها ممکن است *اتاق یک‌نفره*، *اتاق دوتخته* و *اتاق دوتخته بزرگ* باشند، اما در هتل بعدی، آن‌ها *اتاق یک‌نفره لوکس*، *اتاق کلاسیک ملکه* و *اتاق اجرایی شاه* باشند. این‌ها ممکن است همان چیزها باشند، اما تنوع زیادی وجود دارد که انتخاب به این صورت می‌شود:

1. تلاش برای تغییر همه اصطلاحات به یک استاندارد واحد، که بسیار دشوار است، زیرا مشخص نیست مسیر تبدیل در هر مورد چه خواهد بود (مثلاً *اتاق یک‌نفره کلاسیک* به *اتاق یک‌نفره* نگاشت می‌شود اما *اتاق ملکه برتر با چشم‌انداز باغ یا شهر* بسیار سخت‌تر است که نگاشت شود).

1. می‌توانیم یک رویکرد NLP اتخاذ کنیم و فراوانی اصطلاحات خاصی مانند *تنها*، *مسافر کاری* یا *خانواده با کودکان خردسال* را همان‌طور که به هر هتل اعمال می‌شوند اندازه‌گیری کنیم و آن را در مدل توصیه لحاظ کنیم.

برچسب‌ها معمولاً (اما نه همیشه) یک فیلد واحد هستند که شامل لیستی از 5 تا 6 مقدار جدا شده با کاما است که با *نوع سفر*، *نوع مهمانان*، *نوع اتاق*، *تعداد شب‌ها* و *نوع دستگاهی که بررسی با آن ارسال شده است* هم‌راستا هستند. با این حال، چون برخی بررسی‌کنندگان هر فیلد را پر نمی‌کنند (ممکن است یکی را خالی بگذارند)، مقادیر همیشه به همان ترتیب نیستند.

به عنوان مثال، *نوع گروه* را در نظر بگیرید. در این فیلد در ستون `Tags`، 1025 امکان منحصر به فرد وجود دارد، و متأسفانه فقط برخی از آن‌ها به یک گروه اشاره دارند (برخی نوع اتاق و غیره هستند). اگر فقط مواردی را که به خانواده اشاره دارند فیلتر کنید، نتایج شامل بسیاری از نتایج نوع *اتاق خانواده* می‌شود. اگر عبارت *با* را شامل کنید، یعنی تعداد مقادیر *خانواده با* را بشمارید، نتایج بهتر می‌شود، با بیش از 80,000 از 515,000 نتیجه که شامل عبارت "خانواده با کودکان خردسال" یا "خانواده با کودکان بزرگ‌تر" هستند.

این بدان معناست که ستون برچسب‌ها کاملاً بی‌فایده نیست، اما برای مفید شدن آن نیاز به کار دارد.

##### امتیاز میانگین هتل

تعدادی عجایب یا تناقضات در مجموعه داده وجود دارد که نمی‌توانم آن‌ها را بفهمم، اما در اینجا نشان داده شده‌اند تا هنگام ساخت مدل‌های خود از آن‌ها آگاه باشید. اگر آن‌ها را فهمیدید، لطفاً در بخش بحث به ما اطلاع دهید!

مجموعه داده دارای ستون‌های زیر مربوط به امتیاز میانگین و تعداد بررسی‌ها است:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

هتلی که بیشترین تعداد بررسی‌ها را در این مجموعه داده دارد *Britannia International Hotel Canary Wharf* است با 4789 بررسی از 515,000. اما اگر به مقدار `Total_Number_of_Reviews` برای این هتل نگاه کنیم، 9086 است. ممکن است حدس بزنید که تعداد بیشتری امتیاز بدون بررسی وجود دارد، بنابراین شاید باید مقدار ستون `Additional_Number_of_Scoring` را اضافه کنیم. آن مقدار 2682 است، و اضافه کردن آن به 4789 ما را به 7471 می‌رساند که هنوز 1615 کمتر از `Total_Number_of_Reviews` است.

اگر ستون‌های `Average_Score` را بگیرید، ممکن است حدس بزنید که میانگین بررسی‌های موجود در مجموعه داده است، اما توضیح Kaggle این است: "*امتیاز میانگین هتل، محاسبه شده بر اساس آخرین نظر در سال گذشته*". این به نظر چندان مفید نمی‌رسد، اما می‌توانیم میانگین خود را بر اساس امتیازات بررسی در مجموعه داده محاسبه کنیم. با استفاده از همان هتل به عنوان مثال، امتیاز میانگین هتل به عنوان 7.1 داده شده است اما امتیاز محاسبه شده (میانگین امتیاز بررسی‌کننده *در* مجموعه داده) 6.8 است. این نزدیک است، اما همان مقدار نیست، و فقط می‌توانیم حدس بزنیم که امتیازات داده شده در بررسی‌های `Additional_Number_of_Scoring` میانگین را به 7.1 افزایش داده‌اند. متأسفانه بدون راهی برای آزمایش یا اثبات این ادعا، استفاده یا اعتماد به `Average_Score`، `Additional_Number_of_Scoring` و `Total_Number_of_Reviews` دشوار است وقتی که آن‌ها بر اساس داده‌هایی هستند که نداریم.

برای پیچیده‌تر کردن موضوع، هتلی که دومین تعداد بررسی‌ها را دارد، امتیاز میانگین محاسبه شده 8.12 دارد و امتیاز میانگین مجموعه داده `Average_Score` برابر با 8.1 است. آیا این امتیاز صحیح یک تصادف است یا هتل اول یک تناقض است؟

با احتمال اینکه این هتل ممکن است یک مورد استثنا باشد، و اینکه شاید بیشتر مقادیر با هم مطابقت دارند (اما برخی به دلایلی مطابقت ندارند)، برنامه کوتاهی خواهیم نوشت تا مقادیر موجود در مجموعه داده را بررسی کنیم و استفاده صحیح (یا عدم استفاده) از مقادیر را تعیین کنیم.
🚨 یک نکته احتیاطی

هنگام کار با این مجموعه داده، شما کدی خواهید نوشت که چیزی را از متن محاسبه می‌کند بدون اینکه خودتان متن را بخوانید یا تحلیل کنید. این جوهره پردازش زبان طبیعی (NLP) است، تفسیر معنا یا احساسات بدون نیاز به دخالت انسان. با این حال، ممکن است برخی از نظرات منفی را بخوانید. توصیه می‌کنم این کار را نکنید، زیرا نیازی به آن ندارید. برخی از این نظرات بی‌معنی یا نامربوط هستند، مانند نظرات منفی درباره هتل که می‌گویند "هوا خوب نبود"، چیزی که خارج از کنترل هتل یا هر کسی است. اما جنبه تاریکی نیز در برخی نظرات وجود دارد. گاهی اوقات نظرات منفی نژادپرستانه، جنسیت‌زده یا تبعیض‌آمیز نسبت به سن هستند. این موضوع ناخوشایند است اما در مجموعه داده‌ای که از یک وب‌سایت عمومی جمع‌آوری شده، قابل انتظار است. برخی از نویسندگان نظراتی می‌گذارند که ممکن است برای شما ناخوشایند، ناراحت‌کننده یا آزاردهنده باشد. بهتر است اجازه دهید کد احساسات را اندازه‌گیری کند تا اینکه خودتان آن‌ها را بخوانید و ناراحت شوید. با این حال، این افراد در اقلیت هستند، اما همچنان وجود دارند.
## تمرین - بررسی داده‌ها
### بارگذاری داده‌ها

کافی است داده‌ها را به صورت بصری بررسی کنید، حالا وقت آن است که کدی بنویسید و پاسخ‌هایی دریافت کنید! این بخش از کتابخانه pandas استفاده می‌کند. اولین وظیفه شما این است که مطمئن شوید می‌توانید داده‌های CSV را بارگذاری و بخوانید. کتابخانه pandas یک بارگذار سریع CSV دارد و نتیجه در یک dataframe قرار می‌گیرد، همانطور که در درس‌های قبلی دیدید. فایل CSV که بارگذاری می‌کنیم بیش از نیم میلیون ردیف دارد، اما فقط 17 ستون. pandas روش‌های قدرتمند زیادی برای تعامل با dataframe ارائه می‌دهد، از جمله امکان انجام عملیات روی هر ردیف.

از اینجا به بعد در این درس، کدهایی به همراه توضیحات و بحث‌هایی درباره معنای نتایج ارائه خواهد شد. از فایل _notebook.ipynb_ که همراه درس است برای نوشتن کد خود استفاده کنید.

بیایید با بارگذاری فایل داده‌ای که استفاده خواهید کرد شروع کنیم:

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

حالا که داده‌ها بارگذاری شده‌اند، می‌توانیم برخی عملیات را روی آن انجام دهیم. این کد را در بالای برنامه خود برای بخش بعدی نگه دارید.

## بررسی داده‌ها

در این مورد، داده‌ها از قبل *تمیز* هستند، به این معنی که آماده کار هستند و کاراکترهایی به زبان‌های دیگر که ممکن است الگوریتم‌هایی که فقط کاراکترهای انگلیسی انتظار دارند را دچار مشکل کنند، ندارند.

✅ ممکن است لازم باشد با داده‌هایی کار کنید که نیاز به پردازش اولیه برای قالب‌بندی قبل از اعمال تکنیک‌های NLP دارند، اما این بار چنین نیست. اگر لازم بود، چگونه با کاراکترهای غیر انگلیسی برخورد می‌کردید؟

لحظه‌ای وقت بگذارید تا مطمئن شوید که پس از بارگذاری داده‌ها، می‌توانید با کد آن را بررسی کنید. بسیار آسان است که بخواهید روی ستون‌های `Negative_Review` و `Positive_Review` تمرکز کنید. این ستون‌ها پر از متن طبیعی هستند که الگوریتم‌های NLP شما می‌توانند پردازش کنند. اما صبر کنید! قبل از اینکه وارد NLP و تحلیل احساسات شوید، باید کد زیر را دنبال کنید تا مطمئن شوید که مقادیر داده شده در مجموعه داده با مقادیری که با pandas محاسبه می‌کنید مطابقت دارند.

## عملیات روی dataframe

اولین وظیفه در این درس این است که بررسی کنید آیا ادعاهای زیر درست هستند یا خیر، با نوشتن کدی که dataframe را بررسی می‌کند (بدون تغییر آن).

> مانند بسیاری از وظایف برنامه‌نویسی، روش‌های مختلفی برای انجام این کار وجود دارد، اما توصیه خوب این است که آن را به ساده‌ترین و آسان‌ترین روش ممکن انجام دهید، به خصوص اگر در آینده بازگشت به این کد آسان‌تر باشد. با dataframe‌ها، یک API جامع وجود دارد که اغلب راهی برای انجام کار مورد نظر شما به صورت کارآمد ارائه می‌دهد.

سوالات زیر را به عنوان وظایف کدنویسی در نظر بگیرید و سعی کنید بدون نگاه کردن به راه‌حل به آنها پاسخ دهید.

1. شکل (shape) dataframe که تازه بارگذاری کرده‌اید را چاپ کنید (شکل تعداد ردیف‌ها و ستون‌ها است).
2. تعداد تکرار ملیت‌های بررسی‌کننده را محاسبه کنید:
   1. چند مقدار متمایز برای ستون `Reviewer_Nationality` وجود دارد و آنها چه هستند؟
   2. کدام ملیت بررسی‌کننده در مجموعه داده بیشترین تعداد بررسی‌ها را دارد (کشور و تعداد بررسی‌ها را چاپ کنید)؟
   3. 10 ملیت بعدی که بیشترین تعداد بررسی‌ها را دارند و تعداد تکرار آنها چیست؟
3. کدام هتل برای هر یک از 10 ملیت بررسی‌کننده برتر بیشترین تعداد بررسی‌ها را داشته است؟
4. تعداد بررسی‌ها برای هر هتل (تعداد تکرار هتل) در مجموعه داده چقدر است؟
5. در حالی که یک ستون `Average_Score` برای هر هتل در مجموعه داده وجود دارد، می‌توانید یک میانگین امتیاز نیز محاسبه کنید (میانگین تمام امتیازهای بررسی‌کننده‌ها در مجموعه داده برای هر هتل). یک ستون جدید به dataframe خود اضافه کنید با عنوان `Calc_Average_Score` که شامل میانگین محاسبه شده است.
6. آیا هتل‌هایی وجود دارند که `Average_Score` و `Calc_Average_Score` آنها (گرد شده به یک رقم اعشار) برابر باشند؟
   1. سعی کنید یک تابع پایتون بنویسید که یک Series (ردیف) را به عنوان آرگومان بگیرد و مقادیر را مقایسه کند، و هنگامی که مقادیر برابر نیستند یک پیام چاپ کند. سپس از متد `.apply()` استفاده کنید تا هر ردیف را با این تابع پردازش کنید.
7. تعداد ردیف‌هایی که مقدار ستون `Negative_Review` آنها "No Negative" است را محاسبه و چاپ کنید.
8. تعداد ردیف‌هایی که مقدار ستون `Positive_Review` آنها "No Positive" است را محاسبه و چاپ کنید.
9. تعداد ردیف‌هایی که مقدار ستون `Positive_Review` آنها "No Positive" **و** مقدار ستون `Negative_Review` آنها "No Negative" است را محاسبه و چاپ کنید.

### پاسخ‌های کد

1. شکل (shape) dataframe که تازه بارگذاری کرده‌اید را چاپ کنید (شکل تعداد ردیف‌ها و ستون‌ها است).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. تعداد تکرار ملیت‌های بررسی‌کننده را محاسبه کنید:

   1. چند مقدار متمایز برای ستون `Reviewer_Nationality` وجود دارد و آنها چه هستند؟
   2. کدام ملیت بررسی‌کننده در مجموعه داده بیشترین تعداد بررسی‌ها را دارد (کشور و تعداد بررسی‌ها را چاپ کنید)؟

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

   3. 10 ملیت بعدی که بیشترین تعداد بررسی‌ها را دارند و تعداد تکرار آنها چیست؟

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

3. کدام هتل برای هر یک از 10 ملیت بررسی‌کننده برتر بیشترین تعداد بررسی‌ها را داشته است؟

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

4. تعداد بررسی‌ها برای هر هتل (تعداد تکرار هتل) در مجموعه داده چقدر است؟

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
   
   ممکن است متوجه شوید که نتایج *محاسبه شده در مجموعه داده* با مقدار `Total_Number_of_Reviews` مطابقت ندارند. مشخص نیست که آیا این مقدار در مجموعه داده تعداد کل بررسی‌هایی که هتل داشته را نشان می‌دهد، اما همه آنها استخراج نشده‌اند، یا محاسبه دیگری انجام شده است. به دلیل این عدم وضوح، `Total_Number_of_Reviews` در مدل استفاده نمی‌شود.

5. در حالی که یک ستون `Average_Score` برای هر هتل در مجموعه داده وجود دارد، می‌توانید یک میانگین امتیاز نیز محاسبه کنید (میانگین تمام امتیازهای بررسی‌کننده‌ها در مجموعه داده برای هر هتل). یک ستون جدید به dataframe خود اضافه کنید با عنوان `Calc_Average_Score` که شامل میانگین محاسبه شده است. ستون‌های `Hotel_Name`، `Average_Score` و `Calc_Average_Score` را چاپ کنید.

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

   ممکن است درباره مقدار `Average_Score` و اینکه چرا گاهی با میانگین محاسبه شده متفاوت است تعجب کنید. از آنجا که نمی‌توانیم بدانیم چرا برخی مقادیر مطابقت دارند، اما برخی دیگر تفاوت دارند، در این مورد بهتر است از امتیازهای بررسی‌کننده‌ها که داریم برای محاسبه میانگین خودمان استفاده کنیم. با این حال، تفاوت‌ها معمولاً بسیار کوچک هستند، در اینجا هتل‌هایی با بیشترین انحراف از میانگین مجموعه داده و میانگین محاسبه شده آورده شده‌اند:

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

   با وجود اینکه فقط 1 هتل دارای تفاوت امتیاز بیشتر از 1 است، احتمالاً می‌توانیم تفاوت را نادیده بگیریم و از میانگین محاسبه شده استفاده کنیم.

6. تعداد ردیف‌هایی که مقدار ستون `Negative_Review` آنها "No Negative" است را محاسبه و چاپ کنید.

7. تعداد ردیف‌هایی که مقدار ستون `Positive_Review` آنها "No Positive" است را محاسبه و چاپ کنید.

8. تعداد ردیف‌هایی که مقدار ستون `Positive_Review` آنها "No Positive" **و** مقدار ستون `Negative_Review` آنها "No Negative" است را محاسبه و چاپ کنید.

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

## روش دیگر

روش دیگر برای شمارش آیتم‌ها بدون استفاده از Lambdas، و استفاده از sum برای شمارش ردیف‌ها:

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

   ممکن است متوجه شده باشید که 127 ردیف وجود دارد که هر دو مقدار "No Negative" و "No Positive" را برای ستون‌های `Negative_Review` و `Positive_Review` دارند. این بدان معناست که بررسی‌کننده به هتل یک امتیاز عددی داده است، اما از نوشتن یک بررسی مثبت یا منفی خودداری کرده است. خوشبختانه این تعداد کمی از ردیف‌ها است (127 از 515738، یا 0.02%)، بنابراین احتمالاً مدل یا نتایج ما را به هیچ جهت خاصی منحرف نمی‌کند، اما ممکن است انتظار نداشته باشید که مجموعه داده‌ای از بررسی‌ها شامل ردیف‌هایی بدون بررسی باشد، بنابراین ارزش دارد که داده‌ها را بررسی کنید تا ردیف‌هایی مانند این را کشف کنید.

حالا که مجموعه داده را بررسی کردید، در درس بعدی داده‌ها را فیلتر کرده و تحلیل احساسات را اضافه خواهید کرد.

---
## 🚀چالش

این درس نشان می‌دهد، همانطور که در درس‌های قبلی دیدیم، چقدر مهم است که داده‌های خود و ویژگی‌های آن را قبل از انجام عملیات روی آن‌ها به دقت درک کنید. داده‌های مبتنی بر متن، به خصوص، نیاز به بررسی دقیق دارند. مجموعه داده‌های مختلفی که متن زیادی دارند را بررسی کنید و ببینید آیا می‌توانید مناطقی را کشف کنید که ممکن است باعث ایجاد سوگیری یا احساسات منحرف در یک مدل شوند.

## [آزمون پس از درس](https://ff-quizzes.netlify.app/en/ml/)

## مرور و مطالعه شخصی

[این مسیر یادگیری در NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) را دنبال کنید تا ابزارهایی را کشف کنید که هنگام ساخت مدل‌های مبتنی بر گفتار و متن می‌توانید امتحان کنید.

## تکلیف 

[NLTK](assignment.md)

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما تلاش می‌کنیم دقت را حفظ کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادرستی‌ها باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، توصیه می‌شود از ترجمه حرفه‌ای انسانی استفاده کنید. ما مسئولیتی در قبال سوء تفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.