<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T20:09:36+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "he"
}
-->
# מבוא ללמידת חיזוק

למידת חיזוק, RL, נחשבת לאחת מהפרדיגמות הבסיסיות של למידת מכונה, לצד למידה מונחית ולמידה בלתי מונחית. RL עוסקת בקבלת החלטות: קבלת ההחלטות הנכונות או לפחות ללמוד מהן.

דמיינו שיש לכם סביבה מדומה כמו שוק המניות. מה קורה אם אתם מטילים רגולציה מסוימת? האם יש לכך השפעה חיובית או שלילית? אם קורה משהו שלילי, עליכם לקחת את ה_חיזוק השלילי_, ללמוד ממנו ולשנות כיוון. אם התוצאה חיובית, עליכם לבנות על אותו _חיזוק חיובי_.

![פטר והזאב](../../../8-Reinforcement/images/peter.png)

> פטר וחבריו צריכים לברוח מהזאב הרעב! תמונה מאת [Jen Looper](https://twitter.com/jenlooper)

## נושא אזורי: פטר והזאב (רוסיה)

[פטר והזאב](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) הוא אגדה מוזיקלית שנכתבה על ידי המלחין הרוסי [סרגיי פרוקופייב](https://en.wikipedia.org/wiki/Sergei_Prokofiev). זהו סיפור על החלוץ הצעיר פטר, שיוצא באומץ מביתו אל קרחת היער כדי לרדוף אחרי הזאב. בחלק זה, נלמד אלגוריתמים של למידת מכונה שיעזרו לפטר:

- **לחקור** את האזור הסובב ולבנות מפה ניווט אופטימלית.
- **ללמוד** כיצד להשתמש בסקייטבורד ולשמור על איזון עליו, כדי לנוע מהר יותר.

[![פטר והזאב](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 לחצו על התמונה למעלה כדי להאזין ל"פטר והזאב" מאת פרוקופייב

## למידת חיזוק

בחלקים הקודמים ראיתם שני סוגים של בעיות למידת מכונה:

- **מונחית**, שבה יש לנו מערכי נתונים שמציעים פתרונות לדוגמה לבעיה שאנו רוצים לפתור. [סיווג](../4-Classification/README.md) ו[רגרסיה](../2-Regression/README.md) הם משימות של למידה מונחית.
- **בלתי מונחית**, שבה אין לנו נתוני אימון מתויגים. הדוגמה העיקרית ללמידה בלתי מונחית היא [אשכולות](../5-Clustering/README.md).

בחלק זה, נציג בפניכם סוג חדש של בעיית למידה שאינה דורשת נתוני אימון מתויגים. ישנם כמה סוגים של בעיות כאלה:

- **[למידה חצי-מונחית](https://wikipedia.org/wiki/Semi-supervised_learning)**, שבה יש לנו הרבה נתונים לא מתויגים שניתן להשתמש בהם כדי לאמן את המודל מראש.
- **[למידת חיזוק](https://wikipedia.org/wiki/Reinforcement_learning)**, שבה סוכן לומד כיצד להתנהג על ידי ביצוע ניסויים בסביבה מדומה.

### דוגמה - משחק מחשב

נניח שאתם רוצים ללמד מחשב לשחק במשחק, כמו שחמט או [סופר מריו](https://wikipedia.org/wiki/Super_Mario). כדי שהמחשב ישחק במשחק, אנו צריכים שהוא ינבא איזו פעולה לבצע בכל אחד ממצבי המשחק. למרות שזה עשוי להיראות כמו בעיית סיווג, זה לא - מכיוון שאין לנו מערך נתונים עם מצבים ופעולות תואמות. למרות שאולי יש לנו נתונים כמו משחקי שחמט קיימים או הקלטות של שחקנים משחקים סופר מריו, סביר להניח שהנתונים הללו לא יכסו מספיק מצבים אפשריים.

במקום לחפש נתוני משחק קיימים, **למידת חיזוק** (RL) מבוססת על הרעיון של *לגרום למחשב לשחק* פעמים רבות ולצפות בתוצאה. לכן, כדי ליישם למידת חיזוק, אנו צריכים שני דברים:

- **סביבה** ו**סימולטור** שמאפשרים לנו לשחק במשחק פעמים רבות. הסימולטור יגדיר את כללי המשחק, כמו גם את המצבים והפעולות האפשריים.

- **פונקציית תגמול**, שתספר לנו עד כמה הצלחנו במהלך כל מהלך או משחק.

ההבדל העיקרי בין סוגי למידת מכונה אחרים לבין RL הוא שב-RL בדרך כלל איננו יודעים אם ניצחנו או הפסדנו עד לסיום המשחק. לכן, איננו יכולים לומר אם מהלך מסוים לבדו הוא טוב או לא - אנו מקבלים תגמול רק בסוף המשחק. והמטרה שלנו היא לעצב אלגוריתמים שיאפשרו לנו לאמן מודל בתנאים של אי ודאות. נלמד על אלגוריתם RL אחד שנקרא **Q-learning**.

## שיעורים

1. [מבוא ללמידת חיזוק ו-Q-Learning](1-QLearning/README.md)
2. [שימוש בסביבת סימולציה של Gym](2-Gym/README.md)

## קרדיטים

"מבוא ללמידת חיזוק" נכתב באהבה על ידי [Dmitry Soshnikov](http://soshnikov.com)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.