<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:03:29+00:00",
  "source_file": "AGENTS.md",
  "language_code": "bn"
}
-->
# AGENTS.md

## প্রকল্পের সংক্ষিপ্ত বিবরণ

এটি **Machine Learning for Beginners**, একটি ১২-সপ্তাহের, ২৬-লেসনের পাঠক্রম যা Python (মূলত Scikit-learn) এবং R ব্যবহার করে ক্লাসিক মেশিন লার্নিং ধারণাগুলি কভার করে। এই রিপোজিটরি একটি স্ব-গতি সম্পন্ন শিক্ষার উৎস হিসেবে ডিজাইন করা হয়েছে, যেখানে হাতে-কলমে প্রকল্প, কুইজ এবং অ্যাসাইনমেন্ট অন্তর্ভুক্ত রয়েছে। প্রতিটি পাঠ বাস্তব-জীবনের ডেটা ব্যবহার করে বিভিন্ন সংস্কৃতি এবং অঞ্চলের মাধ্যমে মেশিন লার্নিং ধারণাগুলি অন্বেষণ করে।

মূল উপাদানসমূহ:
- **শিক্ষামূলক বিষয়বস্তু**: ২৬টি পাঠ যা মেশিন লার্নিং-এর পরিচিতি, রিগ্রেশন, ক্লাসিফিকেশন, ক্লাস্টারিং, NLP, টাইম সিরিজ এবং রিইনফোর্সমেন্ট লার্নিং কভার করে
- **কুইজ অ্যাপ্লিকেশন**: Vue.js ভিত্তিক কুইজ অ্যাপ যা প্রি- এবং পোস্ট-লেসন মূল্যায়ন প্রদান করে
- **বহুভাষা সমর্থন**: GitHub Actions এর মাধ্যমে ৪০+ ভাষায় স্বয়ংক্রিয় অনুবাদ
- **দ্বৈত ভাষা সমর্থন**: পাঠগুলি Python (Jupyter notebooks) এবং R (R Markdown files) উভয় ভাষায় উপলব্ধ
- **প্রকল্প ভিত্তিক শিক্ষা**: প্রতিটি বিষয়ের সাথে ব্যবহারিক প্রকল্প এবং অ্যাসাইনমেন্ট অন্তর্ভুক্ত

## রিপোজিটরি কাঠামো

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

প্রতিটি পাঠের ফোল্ডারে সাধারণত থাকে:
- `README.md` - প্রধান পাঠের বিষয়বস্তু
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - সমাধানের কোড (Python এবং R সংস্করণ)
- `assignment.md` - অনুশীলনের জন্য কাজ
- `images/` - ভিজ্যুয়াল রিসোর্স

## সেটআপ কমান্ড

### Python পাঠের জন্য

অধিকাংশ পাঠ Jupyter notebooks ব্যবহার করে। প্রয়োজনীয় ডিপেনডেন্সি ইনস্টল করুন:

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

### R পাঠের জন্য

R পাঠগুলি `solution/R/` ফোল্ডারে `.rmd` বা `.ipynb` ফাইল হিসেবে রয়েছে:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### কুইজ অ্যাপ্লিকেশনের জন্য

কুইজ অ্যাপটি `quiz-app/` ডিরেক্টরিতে অবস্থিত একটি Vue.js অ্যাপ্লিকেশন:

```bash
cd quiz-app
npm install
```

### ডকুমেন্টেশন সাইটের জন্য

লোকালভাবে ডকুমেন্টেশন চালানোর জন্য:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## ডেভেলপমেন্ট ওয়ার্কফ্লো

### পাঠের নোটবুক নিয়ে কাজ করা

1. পাঠের ডিরেক্টরিতে যান (যেমন, `2-Regression/1-Tools/`)
2. Jupyter notebook খুলুন:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. পাঠের বিষয়বস্তু এবং অনুশীলন সম্পন্ন করুন
4. প্রয়োজন হলে `solution/` ফোল্ডারে সমাধান পরীক্ষা করুন

### Python ডেভেলপমেন্ট

- পাঠগুলি স্ট্যান্ডার্ড Python ডেটা সায়েন্স লাইব্রেরি ব্যবহার করে
- ইন্টারঅ্যাকটিভ শিক্ষার জন্য Jupyter notebooks
- প্রতিটি পাঠের `solution/` ফোল্ডারে সমাধানের কোড উপলব্ধ

### R ডেভেলপমেন্ট

- R পাঠগুলি `.rmd` ফরম্যাটে (R Markdown)
- সমাধান `solution/R/` সাবডিরেক্টরিতে অবস্থিত
- RStudio বা Jupyter এর R kernel ব্যবহার করে R notebooks চালান

### কুইজ অ্যাপ্লিকেশন ডেভেলপমেন্ট

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

## টেস্টিং নির্দেশনা

### কুইজ অ্যাপ্লিকেশন টেস্টিং

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**নোট**: এটি মূলত একটি শিক্ষামূলক পাঠক্রমের রিপোজিটরি। পাঠের বিষয়বস্তুতে কোনো স্বয়ংক্রিয় টেস্ট নেই। যাচাই করা হয়:
- পাঠের অনুশীলন সম্পন্ন করে
- নোটবুক সেল সফলভাবে চালিয়ে
- সমাধানের সাথে আউটপুট মিলিয়ে

## কোড স্টাইল নির্দেশিকা

### Python কোড
- PEP 8 স্টাইল নির্দেশিকা অনুসরণ করুন
- পরিষ্কার, বর্ণনামূলক ভেরিয়েবল নাম ব্যবহার করুন
- জটিল অপারেশনের জন্য মন্তব্য অন্তর্ভুক্ত করুন
- Jupyter notebooks-এ ধারণাগুলি ব্যাখ্যা করার জন্য markdown সেল থাকা উচিত

### JavaScript/Vue.js (কুইজ অ্যাপ)
- Vue.js স্টাইল গাইড অনুসরণ করে
- `quiz-app/package.json` এ ESLint কনফিগারেশন
- `npm run lint` চালিয়ে সমস্যা পরীক্ষা এবং স্বয়ংক্রিয়ভাবে ঠিক করুন

### ডকুমেন্টেশন
- Markdown ফাইলগুলি পরিষ্কার এবং সুগঠিত হওয়া উচিত
- ফেন্সড কোড ব্লকে কোড উদাহরণ অন্তর্ভুক্ত করুন
- অভ্যন্তরীণ রেফারেন্সের জন্য আপেক্ষিক লিঙ্ক ব্যবহার করুন
- বিদ্যমান ফরম্যাটিং কনভেনশন অনুসরণ করুন

## বিল্ড এবং ডিপ্লয়মেন্ট

### কুইজ অ্যাপ্লিকেশন ডিপ্লয়মেন্ট

কুইজ অ্যাপটি Azure Static Web Apps-এ ডিপ্লয় করা যেতে পারে:

1. **প্রয়োজনীয়তা**:
   - Azure অ্যাকাউন্ট
   - GitHub রিপোজিটরি (আগে থেকেই fork করা)

2. **Azure-এ ডিপ্লয় করুন**:
   - Azure Static Web App রিসোর্স তৈরি করুন
   - GitHub রিপোজিটরির সাথে সংযুক্ত করুন
   - অ্যাপ লোকেশন সেট করুন: `/quiz-app`
   - আউটপুট লোকেশন সেট করুন: `dist`
   - Azure স্বয়ংক্রিয়ভাবে GitHub Actions workflow তৈরি করে

3. **GitHub Actions Workflow**:
   - Workflow ফাইল `.github/workflows/azure-static-web-apps-*.yml` এ তৈরি হয়
   - মূল ব্রাঞ্চে push করলে স্বয়ংক্রিয়ভাবে বিল্ড এবং ডিপ্লয় হয়

### ডকুমেন্টেশন PDF

ডকুমেন্টেশন থেকে PDF তৈরি করুন:

```bash
npm install
npm run convert
```

## অনুবাদ ওয়ার্কফ্লো

**গুরুত্বপূর্ণ**: অনুবাদগুলি GitHub Actions এর মাধ্যমে Co-op Translator ব্যবহার করে স্বয়ংক্রিয়ভাবে সম্পন্ন হয়।

- মূল ব্রাঞ্চে পরিবর্তন push করলে অনুবাদ স্বয়ংক্রিয়ভাবে তৈরি হয়
- **কোনোভাবেই ম্যানুয়ালি অনুবাদ করবেন না** - সিস্টেম এটি পরিচালনা করে
- Workflow `.github/workflows/co-op-translator.yml` এ সংজ্ঞায়িত
- Azure AI/OpenAI পরিষেবাগুলি অনুবাদের জন্য ব্যবহার করে
- ৪০+ ভাষা সমর্থন করে

## কন্ট্রিবিউটিং নির্দেশিকা

### বিষয়বস্তু কন্ট্রিবিউটরদের জন্য

1. **রিপোজিটরি fork করুন** এবং একটি ফিচার ব্রাঞ্চ তৈরি করুন
2. **পাঠের বিষয়বস্তু পরিবর্তন করুন** যদি নতুন পাঠ যোগ বা আপডেট করেন
3. **অনুবাদ করা ফাইল পরিবর্তন করবেন না** - সেগুলি স্বয়ংক্রিয়ভাবে তৈরি হয়
4. **আপনার কোড পরীক্ষা করুন** - নিশ্চিত করুন যে সমস্ত নোটবুক সেল সফলভাবে চালায়
5. **লিঙ্ক এবং ইমেজ যাচাই করুন** সঠিকভাবে কাজ করছে কিনা
6. **একটি pull request জমা দিন** পরিষ্কার বিবরণ সহ

### Pull Request নির্দেশিকা

- **শিরোনামের ফরম্যাট**: `[Section] পরিবর্তনের সংক্ষিপ্ত বিবরণ`
  - উদাহরণ: `[Regression] পাঠ ৫-এ টাইপো সংশোধন`
  - উদাহরণ: `[Quiz-App] ডিপেনডেন্সি আপডেট`
- **জমা দেওয়ার আগে**:
  - নিশ্চিত করুন যে সমস্ত নোটবুক সেল ত্রুটি ছাড়াই চালায়
  - `npm run lint` চালান যদি quiz-app পরিবর্তন করেন
  - Markdown ফরম্যাটিং যাচাই করুন
  - নতুন কোড উদাহরণ পরীক্ষা করুন
- **PR-এ অন্তর্ভুক্ত থাকতে হবে**:
  - পরিবর্তনের বিবরণ
  - পরিবর্তনের কারণ
  - UI পরিবর্তনের ক্ষেত্রে স্ক্রিনশট
- **আচরণবিধি**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) অনুসরণ করুন
- **CLA**: Contributor License Agreement সাইন করতে হবে

## পাঠের কাঠামো

প্রতিটি পাঠ একটি ধারাবাহিক প্যাটার্ন অনুসরণ করে:

1. **পূর্ব-লেকচার কুইজ** - প্রাথমিক জ্ঞান পরীক্ষা করুন
2. **পাঠের বিষয়বস্তু** - লিখিত নির্দেশনা এবং ব্যাখ্যা
3. **কোড প্রদর্শনী** - নোটবুকে হাতে-কলমে উদাহরণ
4. **জ্ঞান যাচাই** - পাঠের সময় বোঝার যাচাই করুন
5. **চ্যালেঞ্জ** - ধারণাগুলি স্বাধীনভাবে প্রয়োগ করুন
6. **অ্যাসাইনমেন্ট** - দীর্ঘমেয়াদী অনুশীলন
7. **পোস্ট-লেকচার কুইজ** - শিক্ষার ফলাফল মূল্যায়ন করুন

## সাধারণ কমান্ড রেফারেন্স

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

## অতিরিক্ত রিসোর্স

- **Microsoft Learn Collection**: [ML for Beginners মডিউল](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **কুইজ অ্যাপ**: [অনলাইন কুইজ](https://ff-quizzes.netlify.app/en/ml/)
- **আলোচনা বোর্ড**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **ভিডিও ওয়াকথ্রু**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## মূল প্রযুক্তি

- **Python**: ML পাঠের প্রধান ভাষা (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret ব্যবহার করে বিকল্প বাস্তবায়ন
- **Jupyter**: Python পাঠের জন্য ইন্টারঅ্যাকটিভ নোটবুক
- **R Markdown**: R পাঠের জন্য ডকুমেন্ট
- **Vue.js 3**: কুইজ অ্যাপ্লিকেশনের ফ্রেমওয়ার্ক
- **Flask**: ML মডেল ডিপ্লয়মেন্টের জন্য ওয়েব অ্যাপ্লিকেশন ফ্রেমওয়ার্ক
- **Docsify**: ডকুমেন্টেশন সাইট জেনারেটর
- **GitHub Actions**: CI/CD এবং স্বয়ংক্রিয় অনুবাদ

## নিরাপত্তা বিবেচনা

- **কোডে কোনো গোপন তথ্য নয়**: API key বা credentials কখনোই কমিট করবেন না
- **ডিপেনডেন্সি**: npm এবং pip প্যাকেজ আপডেট রাখুন
- **ব্যবহারকারীর ইনপুট**: Flask ওয়েব অ্যাপ উদাহরণে মৌলিক ইনপুট যাচাই অন্তর্ভুক্ত
- **সংবেদনশীল ডেটা**: উদাহরণ ডেটাসেটগুলি পাবলিক এবং অ-সংবেদনশীল

## সমস্যা সমাধান

### Jupyter Notebooks

- **কর্নেল সমস্যা**: সেল আটকে গেলে কর্নেল রিস্টার্ট করুন: Kernel → Restart
- **ইমপোর্ট ত্রুটি**: pip দিয়ে প্রয়োজনীয় প্যাকেজ ইনস্টল করুন
- **পাথ সমস্যা**: নোটবুক তাদের কন্টেইনিং ডিরেক্টরি থেকে চালান

### কুইজ অ্যাপ্লিকেশন

- **npm install ব্যর্থ**: npm cache পরিষ্কার করুন: `npm cache clean --force`
- **পোর্ট কনফ্লিক্ট**: পোর্ট পরিবর্তন করুন: `npm run serve -- --port 8081`
- **বিল্ড ত্রুটি**: `node_modules` মুছে পুনরায় ইনস্টল করুন: `rm -rf node_modules && npm install`

### R পাঠ

- **প্যাকেজ পাওয়া যায়নি**: ইনস্টল করুন: `install.packages("package-name")`
- **RMarkdown রেন্ডারিং**: নিশ্চিত করুন rmarkdown প্যাকেজ ইনস্টল করা আছে
- **কর্নেল সমস্যা**: Jupyter এর জন্য IRkernel ইনস্টল করতে হতে পারে

## প্রকল্প-নির্দিষ্ট নোট

- এটি মূলত একটি **শিক্ষামূলক পাঠক্রম**, প্রোডাকশন কোড নয়
- **হাতে-কলমে অনুশীলনের মাধ্যমে মেশিন লার্নিং ধারণা বোঝা**-তে ফোকাস
- কোড উদাহরণগুলি **স্পষ্টতার উপর জোর দেয়, অপ্টিমাইজেশনের উপর নয়**
- অধিকাংশ পাঠ **স্বতন্ত্র** এবং স্বাধীনভাবে সম্পন্ন করা যেতে পারে
- **সমাধান প্রদান করা হয়েছে**, তবে শিক্ষার্থীদের প্রথমে অনুশীলন করা উচিত
- রিপোজিটরি **Docsify** ব্যবহার করে ওয়েব ডকুমেন্টেশন তৈরি করে, কোনো বিল্ড স্টেপ ছাড়াই
- **Sketchnotes** ধারণাগুলির ভিজ্যুয়াল সারাংশ প্রদান করে
- **বহুভাষা সমর্থন** বিষয়বস্তুকে বিশ্বব্যাপী অ্যাক্সেসযোগ্য করে তোলে

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসাধ্য সঠিকতার জন্য চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।