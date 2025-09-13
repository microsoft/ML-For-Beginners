<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-05T18:37:21+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "he"
}
-->
# מודלים של רגרסיה ללמידת מכונה
## נושא אזורי: מודלים של רגרסיה למחירי דלעת בצפון אמריקה 🎃

בצפון אמריקה, דלעות משמשות לעיתים קרובות ליצירת פרצופים מפחידים לכבוד ליל כל הקדושים. בואו נגלה עוד על הירקות המרתקים האלה!

![jack-o-lanterns](../../../2-Regression/images/jack-o-lanterns.jpg)
> צילום על ידי <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> ב-<a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## מה תלמדו

[![מבוא לרגרסיה](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "סרטון מבוא לרגרסיה - לחצו לצפייה!")
> 🎥 לחצו על התמונה למעלה לצפייה בסרטון מבוא קצר לשיעור זה

השיעורים בסעיף זה עוסקים בסוגי רגרסיה בהקשר של למידת מכונה. מודלים של רגרסיה יכולים לעזור לקבוע את _הקשר_ בין משתנים. סוג זה של מודל יכול לחזות ערכים כמו אורך, טמפרטורה או גיל, ובכך לחשוף קשרים בין משתנים תוך ניתוח נקודות נתונים.

בסדרת השיעורים הזו, תגלו את ההבדלים בין רגרסיה לינארית לרגרסיה לוגיסטית, ומתי כדאי להעדיף אחת על פני השנייה.

[![למידת מכונה למתחילים - מבוא למודלים של רגרסיה בלמידת מכונה](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "למידת מכונה למתחילים - מבוא למודלים של רגרסיה בלמידת מכונה")

> 🎥 לחצו על התמונה למעלה לצפייה בסרטון קצר שמציג את מודלי הרגרסיה.

בקבוצת השיעורים הזו, תתארגנו להתחיל משימות למידת מכונה, כולל הגדרת Visual Studio Code לניהול מחברות, הסביבה הנפוצה למדעני נתונים. תגלו את Scikit-learn, ספרייה ללמידת מכונה, ותבנו את המודלים הראשונים שלכם, עם דגש על מודלים של רגרסיה בפרק זה.

> ישנם כלים שימושיים עם מעט קוד שיכולים לעזור לכם ללמוד על עבודה עם מודלים של רגרסיה. נסו [Azure ML למשימה זו](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### שיעורים

1. [כלים מקצועיים](1-Tools/README.md)
2. [ניהול נתונים](2-Data/README.md)
3. [רגרסיה לינארית ופולינומית](3-Linear/README.md)
4. [רגרסיה לוגיסטית](4-Logistic/README.md)

---
### קרדיטים

"למידת מכונה עם רגרסיה" נכתב באהבה על ידי [Jen Looper](https://twitter.com/jenlooper)

♥️ תורמי חידונים כוללים: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) ו-[Ornella Altunyan](https://twitter.com/ornelladotcom)

מאגר הנתונים של דלעות הוצע על ידי [הפרויקט הזה ב-Kaggle](https://www.kaggle.com/usda/a-year-of-pumpkin-prices) והנתונים שלו נלקחו מ-[דוחות סטנדרטיים של שווקי טרמינל לגידולים מיוחדים](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) שמופצים על ידי משרד החקלאות של ארצות הברית. הוספנו כמה נקודות סביב צבע בהתבסס על מגוון כדי לנרמל את ההתפלגות. נתונים אלה נמצאים בתחום הציבורי.

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.