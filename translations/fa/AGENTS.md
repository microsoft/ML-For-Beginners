<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:58:53+00:00",
  "source_file": "AGENTS.md",
  "language_code": "fa"
}
-->
# AGENTS.md

## نمای کلی پروژه

این پروژه **یادگیری ماشین برای مبتدیان** است، یک برنامه جامع ۱۲ هفته‌ای و ۲۶ درس که مفاهیم کلاسیک یادگیری ماشین را با استفاده از پایتون (عمدتاً با Scikit-learn) و R پوشش می‌دهد. این مخزن به عنوان یک منبع یادگیری خودمحور طراحی شده است که شامل پروژه‌های عملی، آزمون‌ها و تمرین‌ها می‌باشد. هر درس مفاهیم یادگیری ماشین را با استفاده از داده‌های واقعی از فرهنگ‌ها و مناطق مختلف جهان بررسی می‌کند.

اجزای کلیدی:
- **محتوای آموزشی**: ۲۶ درس شامل مقدمه‌ای بر یادگیری ماشین، رگرسیون، طبقه‌بندی، خوشه‌بندی، پردازش زبان طبیعی (NLP)، سری‌های زمانی و یادگیری تقویتی
- **برنامه آزمون**: اپلیکیشن آزمون مبتنی بر Vue.js با ارزیابی‌های قبل و بعد از درس
- **پشتیبانی چندزبانه**: ترجمه‌های خودکار به بیش از ۴۰ زبان از طریق GitHub Actions
- **پشتیبانی دو زبانه**: درس‌ها در هر دو زبان پایتون (دفترچه‌های Jupyter) و R (فایل‌های R Markdown) موجود هستند
- **یادگیری مبتنی بر پروژه**: هر موضوع شامل پروژه‌ها و تمرین‌های عملی است

## ساختار مخزن

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

هر پوشه درس معمولاً شامل موارد زیر است:
- `README.md` - محتوای اصلی درس
- `notebook.ipynb` - دفترچه Jupyter پایتون
- `solution/` - کد حل (نسخه‌های پایتون و R)
- `assignment.md` - تمرین‌های عملی
- `images/` - منابع تصویری

## دستورات راه‌اندازی

### برای درس‌های پایتون

بیشتر درس‌ها از دفترچه‌های Jupyter استفاده می‌کنند. وابستگی‌های مورد نیاز را نصب کنید:

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### برای درس‌های R

درس‌های R در پوشه‌های `solution/R/` به صورت فایل‌های `.rmd` یا `.ipynb` قرار دارند:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### برای برنامه آزمون

برنامه آزمون یک اپلیکیشن Vue.js است که در پوشه `quiz-app/` قرار دارد:

```bash
cd quiz-app
npm install
```

### برای سایت مستندات

برای اجرای مستندات به صورت محلی:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## جریان کاری توسعه

### کار با دفترچه‌های درس

1. به پوشه درس مورد نظر بروید (مثلاً `2-Regression/1-Tools/`)
2. دفترچه Jupyter را باز کنید:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. محتوای درس و تمرین‌ها را مرور کنید
4. در صورت نیاز، راه‌حل‌ها را در پوشه `solution/` بررسی کنید

### توسعه پایتون

- درس‌ها از کتابخانه‌های استاندارد علوم داده پایتون استفاده می‌کنند
- دفترچه‌های Jupyter برای یادگیری تعاملی
- کد حل در پوشه `solution/` هر درس موجود است

### توسعه R

- درس‌های R به صورت فرمت `.rmd` (R Markdown) هستند
- راه‌حل‌ها در زیرپوشه‌های `solution/R/` قرار دارند
- از RStudio یا Jupyter با هسته R برای اجرای دفترچه‌های R استفاده کنید

### توسعه برنامه آزمون

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## دستورالعمل‌های تست

### تست برنامه آزمون

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**توجه**: این مخزن عمدتاً یک برنامه آموزشی است. هیچ تست خودکاری برای محتوای درس‌ها وجود ندارد. اعتبارسنجی از طریق موارد زیر انجام می‌شود:
- تکمیل تمرین‌های درس
- اجرای موفقیت‌آمیز سلول‌های دفترچه
- بررسی خروجی در مقابل نتایج مورد انتظار در راه‌حل‌ها

## دستورالعمل‌های سبک کدنویسی

### کد پایتون
- از دستورالعمل‌های سبک PEP 8 پیروی کنید
- از نام‌های متغیر واضح و توصیفی استفاده کنید
- برای عملیات پیچیده توضیحات اضافه کنید
- دفترچه‌های Jupyter باید سلول‌های مارک‌داون برای توضیح مفاهیم داشته باشند

### جاوااسکریپت/Vue.js (برنامه آزمون)
- از راهنمای سبک Vue.js پیروی کنید
- پیکربندی ESLint در `quiz-app/package.json`
- با اجرای `npm run lint` مشکلات را بررسی و خودکار رفع کنید

### مستندات
- فایل‌های مارک‌داون باید واضح و ساختارمند باشند
- مثال‌های کد را در بلوک‌های کد محصور قرار دهید
- از لینک‌های نسبی برای ارجاعات داخلی استفاده کنید
- از قالب‌بندی موجود پیروی کنید

## ساخت و استقرار

### استقرار برنامه آزمون

برنامه آزمون را می‌توان در Azure Static Web Apps مستقر کرد:

1. **پیش‌نیازها**:
   - حساب Azure
   - مخزن GitHub (قبلاً فورک شده)

2. **استقرار در Azure**:
   - ایجاد منبع Azure Static Web App
   - اتصال به مخزن GitHub
   - تنظیم مکان برنامه: `/quiz-app`
   - تنظیم مکان خروجی: `dist`
   - Azure به صورت خودکار جریان کاری GitHub Actions ایجاد می‌کند

3. **جریان کاری GitHub Actions**:
   - فایل جریان کاری در `.github/workflows/azure-static-web-apps-*.yml` ایجاد می‌شود
   - به صورت خودکار با هر بار فشار به شاخه اصلی ساخته و مستقر می‌شود

### مستندات PDF

تولید PDF از مستندات:

```bash
npm install
npm run convert
```

## جریان کاری ترجمه

**مهم**: ترجمه‌ها به صورت خودکار از طریق GitHub Actions با استفاده از Co-op Translator انجام می‌شوند.

- ترجمه‌ها به صورت خودکار هنگام اعمال تغییرات در شاخه `main` ایجاد می‌شوند
- **به صورت دستی محتوا را ترجمه نکنید** - سیستم این کار را انجام می‌دهد
- جریان کاری در `.github/workflows/co-op-translator.yml` تعریف شده است
- از خدمات Azure AI/OpenAI برای ترجمه استفاده می‌کند
- از بیش از ۴۰ زبان پشتیبانی می‌کند

## دستورالعمل‌های مشارکت

### برای مشارکت‌کنندگان محتوا

1. **مخزن را فورک کنید** و یک شاخه ویژگی ایجاد کنید
2. **تغییرات در محتوای درس ایجاد کنید** اگر درس جدید اضافه یا به‌روزرسانی می‌کنید
3. **فایل‌های ترجمه شده را تغییر ندهید** - آنها به صورت خودکار تولید می‌شوند
4. **کد خود را تست کنید** - مطمئن شوید که تمام سلول‌های دفترچه بدون خطا اجرا می‌شوند
5. **لینک‌ها و تصاویر را بررسی کنید** که به درستی کار کنند
6. **یک درخواست کشش ارسال کنید** با توضیحات واضح

### دستورالعمل‌های درخواست کشش

- **فرمت عنوان**: `[بخش] توضیح مختصر تغییرات`
  - مثال: `[Regression] اصلاح اشتباه تایپی در درس ۵`
  - مثال: `[Quiz-App] به‌روزرسانی وابستگی‌ها`
- **قبل از ارسال**:
  - مطمئن شوید که تمام سلول‌های دفترچه بدون خطا اجرا می‌شوند
  - اگر برنامه آزمون را تغییر داده‌اید، `npm run lint` را اجرا کنید
  - قالب‌بندی مارک‌داون را بررسی کنید
  - هر مثال کد جدید را تست کنید
- **PR باید شامل موارد زیر باشد**:
  - توضیح تغییرات
  - دلیل تغییرات
  - تصاویر اگر تغییرات UI وجود دارد
- **قانون رفتار**: از [قانون رفتار منبع باز مایکروسافت](CODE_OF_CONDUCT.md) پیروی کنید
- **CLA**: شما باید توافقنامه مجوز مشارکت‌کننده را امضا کنید

## ساختار درس

هر درس از یک الگوی ثابت پیروی می‌کند:

1. **آزمون قبل از درس** - دانش اولیه را آزمایش کنید
2. **محتوای درس** - دستورالعمل‌ها و توضیحات نوشته شده
3. **نمایش کد** - مثال‌های عملی در دفترچه‌ها
4. **بررسی دانش** - در طول درس درک را بررسی کنید
5. **چالش** - مفاهیم را به صورت مستقل اعمال کنید
6. **تمرین** - تمرین گسترده
7. **آزمون بعد از درس** - نتایج یادگیری را ارزیابی کنید

## مرجع دستورات رایج

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## منابع اضافی

- **مجموعه Microsoft Learn**: [ماژول‌های یادگیری ماشین برای مبتدیان](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **برنامه آزمون**: [آزمون‌های آنلاین](https://ff-quizzes.netlify.app/en/ml/)
- **تابلوی بحث**: [بحث‌های GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **راهنمای ویدیویی**: [لیست پخش YouTube](https://aka.ms/ml-beginners-videos)

## فناوری‌های کلیدی

- **پایتون**: زبان اصلی برای درس‌های یادگیری ماشین (Scikit-learn، Pandas، NumPy، Matplotlib)
- **R**: پیاده‌سازی جایگزین با استفاده از tidyverse، tidymodels، caret
- **Jupyter**: دفترچه‌های تعاملی برای درس‌های پایتون
- **R Markdown**: اسناد برای درس‌های R
- **Vue.js 3**: چارچوب برنامه آزمون
- **Flask**: چارچوب برنامه وب برای استقرار مدل‌های یادگیری ماشین
- **Docsify**: تولیدکننده سایت مستندات
- **GitHub Actions**: CI/CD و ترجمه‌های خودکار

## ملاحظات امنیتی

- **بدون اطلاعات محرمانه در کد**: هرگز کلیدهای API یا اطلاعات ورود را در کد قرار ندهید
- **وابستگی‌ها**: بسته‌های npm و pip را به‌روز نگه دارید
- **ورودی کاربر**: مثال‌های برنامه وب Flask شامل اعتبارسنجی اولیه ورودی هستند
- **داده‌های حساس**: مجموعه داده‌های مثال عمومی و غیر حساس هستند

## رفع اشکال

### دفترچه‌های Jupyter

- **مشکلات هسته**: اگر سلول‌ها گیر کردند، هسته را مجدداً راه‌اندازی کنید: Kernel → Restart
- **خطاهای وارد کردن**: مطمئن شوید که تمام بسته‌های مورد نیاز با pip نصب شده‌اند
- **مشکلات مسیر**: دفترچه‌ها را از پوشه حاوی آنها اجرا کنید

### برنامه آزمون

- **npm install شکست خورد**: کش npm را پاک کنید: `npm cache clean --force`
- **تعارض پورت**: پورت را تغییر دهید با: `npm run serve -- --port 8081`
- **خطاهای ساخت**: `node_modules` را حذف کرده و دوباره نصب کنید: `rm -rf node_modules && npm install`

### درس‌های R

- **بسته پیدا نشد**: با دستور `install.packages("package-name")` نصب کنید
- **رندر RMarkdown**: مطمئن شوید که بسته rmarkdown نصب شده است
- **مشکلات هسته**: ممکن است نیاز به نصب IRkernel برای Jupyter داشته باشید

## یادداشت‌های خاص پروژه

- این پروژه عمدتاً یک **برنامه آموزشی** است، نه کد تولیدی
- تمرکز بر **درک مفاهیم یادگیری ماشین** از طریق تمرین عملی است
- مثال‌های کد **وضوح را بر بهینه‌سازی** اولویت می‌دهند
- بیشتر درس‌ها **خودمختار** هستند و می‌توانند به صورت مستقل تکمیل شوند
- **راه‌حل‌ها ارائه شده‌اند** اما یادگیرندگان باید ابتدا تمرین‌ها را انجام دهند
- مخزن از **Docsify** برای مستندات وب بدون مرحله ساخت استفاده می‌کند
- **Sketchnotes** خلاصه‌های تصویری مفاهیم را ارائه می‌دهند
- **پشتیبانی چندزبانه** محتوا را به صورت جهانی در دسترس قرار می‌دهد

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما تلاش می‌کنیم دقت را حفظ کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادرستی‌ها باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، ترجمه حرفه‌ای انسانی توصیه می‌شود. ما مسئولیتی در قبال سوء تفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.