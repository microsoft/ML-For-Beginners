<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T20:38:56+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "he"
}
-->
# תרגום וניתוח רגשות עם למידת מכונה

בשיעורים הקודמים למדתם כיצד לבנות בוט בסיסי באמצעות `TextBlob`, ספרייה שמשתמשת בלמידת מכונה מאחורי הקלעים כדי לבצע משימות בסיסיות של עיבוד שפה טבעית כמו חילוץ ביטויי שם עצם. אתגר חשוב נוסף בבלשנות חישובית הוא תרגום מדויק של משפט משפה מדוברת או כתובה אחת לשפה אחרת.

## [שאלון לפני השיעור](https://ff-quizzes.netlify.app/en/ml/)

תרגום הוא בעיה קשה מאוד, במיוחד לאור העובדה שיש אלפי שפות שלכל אחת מהן כללי דקדוק שונים מאוד. גישה אחת היא להמיר את כללי הדקדוק הפורמליים של שפה אחת, כמו אנגלית, למבנה שאינו תלוי בשפה, ואז לתרגם אותו על ידי המרה חזרה לשפה אחרת. גישה זו כוללת את השלבים הבאים:

1. **זיהוי**. זיהוי או תיוג של המילים בשפת המקור כעצם, פועל וכו'.
2. **יצירת תרגום**. הפקת תרגום ישיר של כל מילה בפורמט של שפת היעד.

### משפט לדוגמה, מאנגלית לאירית

ב'אנגלית', המשפט _I feel happy_ מורכב משלוש מילים בסדר הבא:

- **נושא** (I)
- **פועל** (feel)
- **תואר** (happy)

עם זאת, בשפה 'אירית', לאותו משפט יש מבנה דקדוקי שונה מאוד - רגשות כמו "*שמח*" או "*עצוב*" מתוארים כמשהו *שעליך*.

הביטוי האנגלי `I feel happy` באירית יהיה `Tá athas orm`. תרגום *מילולי* יהיה `שמח עליי`.

דובר אירית שמתרגם לאנגלית יאמר `I feel happy`, ולא `Happy is upon me`, כי הוא מבין את משמעות המשפט, גם אם המילים ומבנה המשפט שונים.

הסדר הפורמלי של המשפט באירית הוא:

- **פועל** (Tá או is)
- **תואר** (athas, או happy)
- **נושא** (orm, או עליי)

## תרגום

תוכנית תרגום נאיבית עשויה לתרגם מילים בלבד, תוך התעלמות ממבנה המשפט.

✅ אם למדתם שפה שנייה (או שלישית או יותר) כמבוגרים, ייתכן שהתחלתם לחשוב בשפת האם שלכם, לתרגם מושגים מילה במילה בראשכם לשפה השנייה, ואז לומר את התרגום שלכם. זה דומה למה שתוכניות תרגום נאיביות עושות. חשוב להתקדם מעבר לשלב הזה כדי להגיע לשטף!

תרגום נאיבי מוביל לתרגומים גרועים (ולפעמים מצחיקים): `I feel happy` מתורגם באופן מילולי ל-`Mise bhraitheann athas` באירית. זה אומר (מילולית) `אני מרגיש שמח` ואינו משפט אירי תקני. למרות שאנגלית ואירית הן שפות המדוברות בשני איים שכנים, הן שפות שונות מאוד עם מבני דקדוק שונים.

> תוכלו לצפות בכמה סרטונים על מסורות לשוניות איריות כמו [זה](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### גישות למידת מכונה

עד כה, למדתם על הגישה של כללים פורמליים לעיבוד שפה טבעית. גישה נוספת היא להתעלם ממשמעות המילים, ו_במקום זאת להשתמש בלמידת מכונה כדי לזהות דפוסים_. זה יכול לעבוד בתרגום אם יש לכם הרבה טקסטים (*corpus*) או טקסטים (*corpora*) בשפת המקור ובשפת היעד.

לדוגמה, שקלו את המקרה של *גאווה ודעה קדומה*, רומן אנגלי ידוע שנכתב על ידי ג'יין אוסטן בשנת 1813. אם תעיינו בספר באנגלית ובתרגום אנושי של הספר ל*צרפתית*, תוכלו לזהות ביטויים באחד שמתורגמים באופן _אידיומטי_ לשני. תעשו זאת בעוד רגע.

לדוגמה, כאשר ביטוי באנגלית כמו `I have no money` מתורגם באופן מילולי לצרפתית, הוא עשוי להפוך ל-`Je n'ai pas de monnaie`. "Monnaie" הוא 'דמיון שווא' צרפתי מסובך, שכן 'money' ו-'monnaie' אינם מילים נרדפות. תרגום טוב יותר שדובר אנושי עשוי לעשות יהיה `Je n'ai pas d'argent`, כי הוא מעביר טוב יותר את המשמעות שאין לך כסף (ולא 'כסף קטן' שהוא המשמעות של 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> תמונה מאת [Jen Looper](https://twitter.com/jenlooper)

אם למודל למידת מכונה יש מספיק תרגומים אנושיים לבניית מודל, הוא יכול לשפר את דיוק התרגומים על ידי זיהוי דפוסים נפוצים בטקסטים שתורגמו בעבר על ידי דוברים אנושיים מומחים של שתי השפות.

### תרגיל - תרגום

תוכלו להשתמש ב-`TextBlob` כדי לתרגם משפטים. נסו את השורה הראשונה המפורסמת של **גאווה ודעה קדומה**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` עושה עבודה די טובה בתרגום: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

אפשר לטעון שהתרגום של TextBlob מדויק הרבה יותר, למעשה, מהתרגום הצרפתי של הספר משנת 1932 על ידי V. Leconte ו-Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

במקרה זה, התרגום המבוסס על למידת מכונה עושה עבודה טובה יותר מהמתרגם האנושי שהוסיף מילים מיותרות לפיו של המחבר המקורי לצורך 'בהירות'.

> מה קורה כאן? ולמה TextBlob כל כך טוב בתרגום? ובכן, מאחורי הקלעים, הוא משתמש ב-Google Translate, AI מתוחכם שמסוגל לנתח מיליוני ביטויים כדי לחזות את המחרוזות הטובות ביותר למשימה. אין כאן שום דבר ידני ואתם צריכים חיבור לאינטרנט כדי להשתמש ב-`blob.translate`.

✅ נסו עוד משפטים. מה עדיף, תרגום בלמידת מכונה או תרגום אנושי? באילו מקרים?

## ניתוח רגשות

תחום נוסף שבו למידת מכונה יכולה לעבוד היטב הוא ניתוח רגשות. גישה שאינה מבוססת למידת מכונה לניתוח רגשות היא לזהות מילים וביטויים שהם 'חיוביים' ו'שליליים'. לאחר מכן, בהתחשב בטקסט חדש, לחשב את הערך הכולל של המילים החיוביות, השליליות והנייטרליות כדי לזהות את הרגש הכללי.

גישה זו ניתנת להטעיה בקלות כפי שראיתם במשימת מרווין - המשפט `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` הוא משפט סרקסטי עם רגש שלילי, אך האלגוריתם הפשוט מזהה 'great', 'wonderful', 'glad' כחיוביים ו-'waste', 'lost' ו-'dark' כשליליים. הרגש הכללי מושפע מהמילים הסותרות הללו.

✅ עצרו רגע וחשבו כיצד אנו מעבירים סרקזם כדוברים אנושיים. אינטונציה משחקת תפקיד גדול. נסו לומר את המשפט "Well, that film was awesome" בדרכים שונות כדי לגלות כיצד הקול שלכם מעביר משמעות.

### גישות למידת מכונה

הגישה של למידת מכונה תהיה לאסוף באופן ידני גופי טקסט שליליים וחיוביים - ציוצים, או ביקורות סרטים, או כל דבר שבו האדם נתן ציון *וגם* דעה כתובה. לאחר מכן ניתן ליישם טכניקות עיבוד שפה טבעית על דעות וציונים, כך שדפוסים יופיעו (לדוגמה, ביקורות סרטים חיוביות נוטות לכלול את הביטוי 'Oscar worthy' יותר מאשר ביקורות שליליות, או ביקורות מסעדות חיוביות אומרות 'gourmet' הרבה יותר מאשר 'disgusting').

> ⚖️ **דוגמה**: אם עבדתם במשרד של פוליטיקאי ויש חוק חדש שנדון, ייתכן שתושבים יכתבו למשרד עם מיילים שתומכים או מתנגדים לחוק החדש. נניח שאתם מתבקשים לקרוא את המיילים ולמיין אותם לשתי ערימות, *בעד* ו-*נגד*. אם היו הרבה מיילים, ייתכן שתהיו מוצפים בניסיון לקרוא את כולם. לא יהיה נחמד אם בוט יוכל לקרוא את כולם עבורכם, להבין אותם ולומר לכם לאיזו ערימה כל מייל שייך? 
> 
> דרך אחת להשיג זאת היא להשתמש בלמידת מכונה. הייתם מאמנים את המודל עם חלק מהמיילים ה*נגד* וחלק מהמיילים ה*בעד*. המודל היה נוטה לשייך ביטויים ומילים לצד הנגד ולצד הבעד, *אך הוא לא היה מבין שום תוכן*, רק שמילים ודפוסים מסוימים נוטים להופיע יותר במיילים נגד או בעד. הייתם בודקים אותו עם כמה מיילים שלא השתמשתם בהם כדי לאמן את המודל, ורואים אם הוא הגיע לאותה מסקנה כמוכם. לאחר מכן, ברגע שהייתם מרוצים מהדיוק של המודל, הייתם יכולים לעבד מיילים עתידיים מבלי לקרוא כל אחד מהם.

✅ האם התהליך הזה נשמע כמו תהליכים שהשתמשתם בהם בשיעורים קודמים?

## תרגיל - משפטים רגשיים

רגש נמדד עם *קוטביות* של -1 עד 1, כלומר -1 הוא הרגש השלילי ביותר, ו-1 הוא הרגש החיובי ביותר. רגש נמדד גם עם ציון של 0 - 1 עבור אובייקטיביות (0) וסובייקטיביות (1).

הסתכלו שוב על *גאווה ודעה קדומה* של ג'יין אוסטן. הטקסט זמין כאן ב-[Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). הדוגמה למטה מציגה תוכנית קצרה שמנתחת את הרגש של המשפטים הראשונים והאחרונים מהספר ומציגה את קוטביות הרגש ואת ציון הסובייקטיביות/אובייקטיביות.

עליכם להשתמש בספריית `TextBlob` (שתוארה לעיל) כדי לקבוע `sentiment` (אין צורך לכתוב מחשבון רגשות משלכם) במשימה הבאה.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

אתם רואים את הפלט הבא:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## אתגר - בדיקת קוטביות רגשית

המשימה שלכם היא לקבוע, באמצעות קוטביות רגשית, האם ל-*גאווה ודעה קדומה* יש יותר משפטים חיוביים לחלוטין מאשר שליליים לחלוטין. לצורך משימה זו, תוכלו להניח שקוטביות של 1 או -1 היא חיובית לחלוטין או שלילית לחלוטין בהתאמה.

**שלבים:**

1. הורידו [עותק של גאווה ודעה קדומה](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) מ-Project Gutenberg כקובץ .txt. הסירו את המטא-דאטה בתחילת ובסוף הקובץ, והשאירו רק את הטקסט המקורי
2. פתחו את הקובץ ב-Python והוציאו את התוכן כמחרוזת
3. צרו TextBlob באמצעות מחרוזת הספר
4. נתחו כל משפט בספר בלולאה
   1. אם הקוטביות היא 1 או -1, אחסנו את המשפט במערך או רשימה של הודעות חיוביות או שליליות
5. בסוף, הדפיסו את כל המשפטים החיוביים והשליליים (בנפרד) ואת המספר של כל אחד.

הנה [פתרון לדוגמה](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ בדיקת ידע

1. הרגש מבוסס על מילים שנמצאות במשפט, אבל האם הקוד *מבין* את המילים?
2. האם לדעתכם קוטביות הרגש מדויקת, או במילים אחרות, האם אתם *מסכימים* עם הציונים?
   1. במיוחד, האם אתם מסכימים או לא מסכימים עם הקוטביות החיובית **המוחלטת** של המשפטים הבאים?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. שלושת המשפטים הבאים דורגו עם קוטביות חיובית מוחלטת, אבל בקריאה מעמיקה, הם אינם משפטים חיוביים. מדוע ניתוח הרגש חשב שהם משפטים חיוביים?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. האם אתם מסכימים או לא מסכימים עם הקוטביות השלילית **המוחלטת** של המשפטים הבאים?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ כל חובב של ג'יין אוסטן יבין שהיא לעיתים קרובות משתמשת בספריה כדי לבקר את ההיבטים המגוחכים יותר של החברה האנגלית בתקופת הריג'נסי. אליזבת בנט, הדמות הראשית ב-*גאווה ודעה קדומה*, היא צופה חברתית חדה (כמו המחברת) והשפה שלה לעיתים קרובות מאוד מעודנת. אפילו מר דארסי (מושא האהבה בסיפור) מציין את השימוש המשחקי והמתגרה של אליזבת בשפה: "היה לי העונג להכיר אותך מספיק זמן כדי לדעת שאת נהנית מאוד מדי פעם להביע דעות שאינן באמת שלך."

---

## 🚀אתגר

האם תוכלו לשפר את מרווין על ידי חילוץ תכונות נוספות מהקלט של המשתמש?

## [שאלון אחרי השיעור](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי
ישנן דרכים רבות להפיק רגשות מטקסט. חשבו על יישומים עסקיים שיכולים להשתמש בטכניקה זו. חשבו על איך זה יכול להשתבש. קראו עוד על מערכות מתקדמות המוכנות לשימוש ארגוני שמנתחות רגשות, כמו [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). בדקו כמה מהמשפטים מתוך "גאווה ודעה קדומה" למעלה וראו אם ניתן לזהות בהם ניואנסים.

## משימה

[רישיון פואטי](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.