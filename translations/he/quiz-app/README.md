<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T19:47:28+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "he"
}
-->
# חידונים

החידונים האלה הם חידוני טרום ואחרי הרצאה עבור תוכנית הלימודים של למידת מכונה בכתובת https://aka.ms/ml-beginners

## הגדרת הפרויקט

```
npm install
```

### קומפילציה וטעינה מחדש לפיתוח

```
npm run serve
```

### קומפילציה ומזעור עבור הפקה

```
npm run build
```

### בדיקת קוד ותיקון קבצים

```
npm run lint
```

### התאמת ההגדרות

ראו [הפניה להגדרות](https://cli.vuejs.org/config/).

קרדיטים: תודה לגרסה המקורית של אפליקציית החידונים הזו: https://github.com/arpan45/simple-quiz-vue

## פריסה ל-Azure

הנה מדריך שלב-אחר-שלב שיעזור לכם להתחיל:

1. עשו Fork למאגר GitHub  
ודאו שקוד האפליקציה שלכם נמצא במאגר GitHub. עשו Fork למאגר הזה.

2. צרו אפליקציית אינטרנט סטטית ב-Azure  
- צרו [חשבון Azure](http://azure.microsoft.com)  
- עברו ל-[פורטל Azure](https://portal.azure.com)  
- לחצו על "Create a resource" וחפשו "Static Web App".  
- לחצו על "Create".  

3. הגדרת אפליקציית האינטרנט הסטטית  
- בסיסים:  
  - Subscription: בחרו את המנוי שלכם ב-Azure.  
  - Resource Group: צרו קבוצת משאבים חדשה או השתמשו בקיימת.  
  - Name: ספקו שם לאפליקציית האינטרנט הסטטית שלכם.  
  - Region: בחרו את האזור הקרוב ביותר למשתמשים שלכם.  

- #### פרטי פריסה:  
  - Source: בחרו "GitHub".  
  - GitHub Account: תנו הרשאה ל-Azure לגשת לחשבון GitHub שלכם.  
  - Organization: בחרו את הארגון שלכם ב-GitHub.  
  - Repository: בחרו את המאגר שמכיל את אפליקציית האינטרנט הסטטית שלכם.  
  - Branch: בחרו את הענף שממנו תרצו לפרוס.  

- #### פרטי בנייה:  
  - Build Presets: בחרו את המסגרת שבה האפליקציה שלכם נבנתה (לדוגמה, React, Angular, Vue וכו').  
  - App Location: ציינו את התיקייה שמכילה את קוד האפליקציה שלכם (לדוגמה, / אם היא נמצאת בשורש).  
  - API Location: אם יש לכם API, ציינו את מיקומו (אופציונלי).  
  - Output Location: ציינו את התיקייה שבה נוצר פלט הבנייה (לדוגמה, build או dist).  

4. סקירה ויצירה  
סקור את ההגדרות שלך ולחץ על "Create". Azure יגדיר את המשאבים הנדרשים וייצור קובץ זרימת עבודה של GitHub Actions במאגר שלך.

5. זרימת עבודה של GitHub Actions  
Azure ייצור באופן אוטומטי קובץ זרימת עבודה של GitHub Actions במאגר שלך (.github/workflows/azure-static-web-apps-<name>.yml). קובץ זה יטפל בתהליך הבנייה והפריסה.

6. מעקב אחר הפריסה  
עברו ללשונית "Actions" במאגר GitHub שלכם.  
תוכלו לראות זרימת עבודה פועלת. זרימת עבודה זו תבנה ותפרוס את אפליקציית האינטרנט הסטטית שלכם ל-Azure.  
לאחר סיום זרימת העבודה, האפליקציה שלכם תהיה זמינה בכתובת ה-URL שסופקה על ידי Azure.

### דוגמה לקובץ זרימת עבודה

הנה דוגמה לאיך קובץ זרימת העבודה של GitHub Actions עשוי להיראות:  
name: Azure Static Web Apps CI/CD  
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### משאבים נוספים  
- [תיעוד אפליקציות אינטרנט סטטיות של Azure](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [תיעוד GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.