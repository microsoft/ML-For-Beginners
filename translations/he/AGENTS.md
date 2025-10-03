<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:12:21+00:00",
  "source_file": "AGENTS.md",
  "language_code": "he"
}
-->
# AGENTS.md

## סקירת הפרויקט

זהו **למידת מכונה למתחילים**, תוכנית לימודים מקיפה בת 12 שבועות ו-26 שיעורים המכסה מושגים קלאסיים בלמידת מכונה באמצעות Python (בעיקר עם Scikit-learn) ו-R. המאגר נועד להיות משאב לימוד בקצב אישי עם פרויקטים מעשיים, חידונים ומשימות. כל שיעור חוקר מושגים בלמידת מכונה דרך נתונים מעולם האמיתי ממגוון תרבויות ואזורים ברחבי העולם.

רכיבים מרכזיים:
- **תוכן חינוכי**: 26 שיעורים המכסים מבוא ללמידת מכונה, רגרסיה, סיווג, אשכולות, NLP, סדרות זמן ולמידת חיזוק
- **אפליקציית חידונים**: אפליקציה מבוססת Vue.js עם הערכות לפני ואחרי השיעור
- **תמיכה רב-שפתית**: תרגומים אוטומטיים ליותר מ-40 שפות באמצעות GitHub Actions
- **תמיכה בשתי שפות**: שיעורים זמינים גם ב-Python (מחברות Jupyter) וגם ב-R (קבצי R Markdown)
- **למידה מבוססת פרויקטים**: כל נושא כולל פרויקטים ומשימות מעשיות

## מבנה המאגר

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

כל תיקיית שיעור מכילה בדרך כלל:
- `README.md` - תוכן השיעור הראשי
- `notebook.ipynb` - מחברת Jupyter ב-Python
- `solution/` - קוד פתרון (גרסאות Python ו-R)
- `assignment.md` - תרגילים מעשיים
- `images/` - משאבים חזותיים

## פקודות התקנה

### לשיעורי Python

רוב השיעורים משתמשים במחברות Jupyter. התקן את התלויות הנדרשות:

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

### לשיעורי R

שיעורי R נמצאים בתיקיות `solution/R/` כקבצי `.rmd` או `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### לאפליקציית החידונים

אפליקציית החידונים היא אפליקציה מבוססת Vue.js שנמצאת בתיקיית `quiz-app/`:

```bash
cd quiz-app
npm install
```

### לאתר התיעוד

כדי להפעיל את התיעוד באופן מקומי:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## זרימת עבודה לפיתוח

### עבודה עם מחברות שיעור

1. נווט לתיקיית השיעור (לדוגמה, `2-Regression/1-Tools/`)
2. פתח את מחברת Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. עבד דרך תוכן השיעור והתרגילים
4. בדוק פתרונות בתיקיית `solution/` במידת הצורך

### פיתוח ב-Python

- השיעורים משתמשים בספריות סטנדרטיות של מדעי הנתונים ב-Python
- מחברות Jupyter ללמידה אינטראקטיבית
- קוד פתרון זמין בתיקיית `solution/` של כל שיעור

### פיתוח ב-R

- שיעורי R נמצאים בפורמט `.rmd` (R Markdown)
- פתרונות ממוקמים בתיקיות משנה `solution/R/`
- השתמש ב-RStudio או Jupyter עם ליבת R להפעלת מחברות R

### פיתוח אפליקציית החידונים

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

## הוראות בדיקה

### בדיקת אפליקציית החידונים

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**הערה**: זהו בעיקר מאגר תוכנית לימודים חינוכית. אין בדיקות אוטומטיות לתוכן השיעורים. האימות מתבצע באמצעות:
- השלמת תרגילי השיעור
- הפעלת תאים במחברות בהצלחה
- בדיקת הפלט מול התוצאות הצפויות בפתרונות

## הנחיות לסגנון קוד

### קוד Python
- עקוב אחר הנחיות הסגנון של PEP 8
- השתמש בשמות משתנים ברורים ותיאוריים
- הוסף הערות לפעולות מורכבות
- מחברות Jupyter צריכות לכלול תאי Markdown המסבירים מושגים

### JavaScript/Vue.js (אפליקציית חידונים)
- עקוב אחר מדריך הסגנון של Vue.js
- תצורת ESLint ב-`quiz-app/package.json`
- הפעל `npm run lint` לבדיקה ותיקון אוטומטי של בעיות

### תיעוד
- קבצי Markdown צריכים להיות ברורים ומובנים היטב
- כלול דוגמאות קוד בבלוקים מוקפים
- השתמש בקישורים יחסיים להפניות פנימיות
- עקוב אחר מוסכמות העיצוב הקיימות

## בנייה ופריסה

### פריסת אפליקציית החידונים

ניתן לפרוס את אפליקציית החידונים ל-Azure Static Web Apps:

1. **דרישות מוקדמות**:
   - חשבון Azure
   - מאגר GitHub (כבר משוכפל)

2. **פריסה ל-Azure**:
   - צור משאב Azure Static Web App
   - התחבר למאגר GitHub
   - הגדר מיקום אפליקציה: `/quiz-app`
   - הגדר מיקום פלט: `dist`
   - Azure יוצר באופן אוטומטי זרימת עבודה של GitHub Actions

3. **זרימת עבודה של GitHub Actions**:
   - קובץ זרימת עבודה נוצר ב-`.github/workflows/azure-static-web-apps-*.yml`
   - נבנה ונפרס באופן אוטומטי בעת דחיפה לענף הראשי

### PDF של תיעוד

צור PDF מהתיעוד:

```bash
npm install
npm run convert
```

## זרימת עבודה לתרגום

**חשוב**: תרגומים מתבצעים באופן אוטומטי באמצעות GitHub Actions עם Co-op Translator.

- תרגומים נוצרים אוטומטית כאשר שינויים נדחפים לענף `main`
- **אין לתרגם תוכן באופן ידני** - המערכת מטפלת בכך
- זרימת עבודה מוגדרת ב-`.github/workflows/co-op-translator.yml`
- משתמש בשירותי Azure AI/OpenAI לתרגום
- תומך ביותר מ-40 שפות

## הנחיות לתרומה

### עבור תורמי תוכן

1. **שכפל את המאגר** ויצור ענף תכונה
2. **בצע שינויים בתוכן השיעור** אם מוסיפים/מעדכנים שיעורים
3. **אין לשנות קבצים מתורגמים** - הם נוצרים אוטומטית
4. **בדוק את הקוד שלך** - ודא שכל תאי המחברות פועלים בהצלחה
5. **אמת קישורים ותמונות** פועלים כראוי
6. **שלח בקשת משיכה** עם תיאור ברור

### הנחיות לבקשת משיכה

- **פורמט כותרת**: `[Section] תיאור קצר של השינויים`
  - דוגמה: `[Regression] תיקון שגיאת כתיב בשיעור 5`
  - דוגמה: `[Quiz-App] עדכון תלויות`
- **לפני שליחה**:
  - ודא שכל תאי המחברות פועלים ללא שגיאות
  - הפעל `npm run lint` אם משנים את quiz-app
  - אמת עיצוב Markdown
  - בדוק כל דוגמאות קוד חדשות
- **PR חייב לכלול**:
  - תיאור השינויים
  - סיבת השינויים
  - צילומי מסך אם יש שינויים בממשק המשתמש
- **קוד התנהגות**: עקוב אחר [קוד ההתנהגות של Microsoft Open Source](CODE_OF_CONDUCT.md)
- **CLA**: תצטרך לחתום על הסכם רישיון התורם

## מבנה השיעור

כל שיעור עוקב אחר דפוס עקבי:

1. **חידון לפני ההרצאה** - בדיקת ידע בסיסי
2. **תוכן השיעור** - הוראות והסברים כתובים
3. **הדגמות קוד** - דוגמאות מעשיות במחברות
4. **בדיקות ידע** - אימות הבנה לאורך השיעור
5. **אתגר** - יישום מושגים באופן עצמאי
6. **משימה** - תרגול מורחב
7. **חידון לאחר ההרצאה** - הערכת תוצאות הלמידה

## הפניות לפקודות נפוצות

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

## משאבים נוספים

- **אוסף Microsoft Learn**: [מודולים של למידת מכונה למתחילים](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **אפליקציית חידונים**: [חידונים מקוונים](https://ff-quizzes.netlify.app/en/ml/)
- **לוח דיונים**: [דיונים ב-GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **הדרכות וידאו**: [רשימת השמעה ביוטיוב](https://aka.ms/ml-beginners-videos)

## טכנולוגיות מרכזיות

- **Python**: שפת התכנות הראשית לשיעורי למידת מכונה (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: יישום חלופי באמצעות tidyverse, tidymodels, caret
- **Jupyter**: מחברות אינטראקטיביות לשיעורי Python
- **R Markdown**: מסמכים לשיעורי R
- **Vue.js 3**: מסגרת אפליקציית החידונים
- **Flask**: מסגרת אפליקציות אינטרנט לפריסת מודלים של למידת מכונה
- **Docsify**: מחולל אתרי תיעוד
- **GitHub Actions**: CI/CD ותרגומים אוטומטיים

## שיקולי אבטחה

- **אין סודות בקוד**: לעולם אל תתחייב מפתחות API או אישורים
- **תלויות**: שמור על חבילות npm ו-pip מעודכנות
- **קלט משתמש**: דוגמאות אפליקציות אינטרנט Flask כוללות אימות קלט בסיסי
- **נתונים רגישים**: מערכי הנתונים לדוגמה הם ציבוריים ולא רגישים

## פתרון בעיות

### מחברות Jupyter

- **בעיות ליבה**: הפעל מחדש את הליבה אם תאים נתקעים: Kernel → Restart
- **שגיאות ייבוא**: ודא שכל החבילות הנדרשות מותקנות עם pip
- **בעיות נתיב**: הפעל מחברות מתיקייתן המכילה

### אפליקציית חידונים

- **npm install נכשל**: נקה את מטמון npm: `npm cache clean --force`
- **התנגשויות פורט**: שנה פורט עם: `npm run serve -- --port 8081`
- **שגיאות בנייה**: מחק `node_modules` והתקן מחדש: `rm -rf node_modules && npm install`

### שיעורי R

- **חבילה לא נמצאה**: התקן עם: `install.packages("package-name")`
- **המרת RMarkdown**: ודא שחבילת rmarkdown מותקנת
- **בעיות ליבה**: ייתכן שתצטרך להתקין IRkernel עבור Jupyter

## הערות ספציפיות לפרויקט

- זהו בעיקר **תוכנית לימודים ללמידה**, לא קוד ייצור
- המיקוד הוא ב-**הבנת מושגים בלמידת מכונה** דרך תרגול מעשי
- דוגמאות קוד נותנות עדיפות ל-**בהירות על פני אופטימיזציה**
- רוב השיעורים הם **עצמאיים** וניתן להשלים אותם בנפרד
- **פתרונות מסופקים** אך הלומדים צריכים לנסות את התרגילים תחילה
- המאגר משתמש ב-**Docsify** לתיעוד אינטרנטי ללא שלב בנייה
- **סקצ'נוטים** מספקים סיכומים חזותיים של מושגים
- **תמיכה רב-שפתית** הופכת את התוכן לנגיש גלובלית

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי-דיוקים. המסמך המקורי בשפתו המקורית נחשב למקור הסמכותי. למידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי בני אדם. איננו נושאים באחריות לכל אי-הבנה או פרשנות שגויה הנובעת משימוש בתרגום זה.