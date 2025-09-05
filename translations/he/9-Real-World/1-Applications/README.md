<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T19:21:54+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "he"
}
-->
# פוסטסקריפט: למידת מכונה בעולם האמיתי

![סיכום של למידת מכונה בעולם האמיתי בסקצ'נוט](../../../../sketchnotes/ml-realworld.png)
> סקצ'נוט מאת [Tomomi Imura](https://www.twitter.com/girlie_mac)

במהלך הקורס הזה, למדתם דרכים רבות להכנת נתונים לאימון וליצירת מודלים של למידת מכונה. בניתם סדרה של מודלים קלאסיים כמו רגרסיה, אשכולות, סיווג, עיבוד שפה טבעית ומודלים של סדרות זמן. כל הכבוד! עכשיו, אתם אולי תוהים למה כל זה נועד... מהן היישומים בעולם האמיתי של המודלים הללו?

למרות שהרבה עניין בתעשייה מתמקד בבינה מלאכותית, שלרוב עושה שימוש בלמידה עמוקה, עדיין יש יישומים חשובים למודלים קלאסיים של למידת מכונה. ייתכן שאפילו אתם משתמשים בחלק מהיישומים הללו היום! בשיעור הזה, תחקור כיצד שמונה תעשיות ותחומים שונים משתמשים במודלים אלו כדי להפוך את היישומים שלהם ליותר יעילים, אמינים, חכמים ובעלי ערך למשתמשים.

## [שאלון לפני השיעור](https://ff-quizzes.netlify.app/en/ml/)

## 💰 פיננסים

תחום הפיננסים מציע הזדמנויות רבות ללמידת מכונה. בעיות רבות בתחום זה מתאימות למידול ופתרון באמצעות למידת מכונה.

### זיהוי הונאות בכרטיסי אשראי

למדנו על [אשכולות k-means](../../5-Clustering/2-K-Means/README.md) מוקדם יותר בקורס, אבל איך ניתן להשתמש בהם כדי לפתור בעיות הקשורות להונאות בכרטיסי אשראי?

אשכולות k-means מועילים בטכניקה לזיהוי הונאות בכרטיסי אשראי הנקראת **זיהוי חריגות**. חריגות, או סטיות בתצפיות על סט נתונים, יכולות להצביע אם כרטיס אשראי נמצא בשימוש רגיל או אם מתרחש משהו חריג. כפי שמוצג במאמר המקושר למטה, ניתן למיין נתוני כרטיסי אשראי באמצעות אלגוריתם אשכולות k-means ולהקצות כל עסקה לאשכול על סמך מידת החריגות שלה. לאחר מכן, ניתן להעריך את האשכולות המסוכנים ביותר כדי להבחין בין עסקאות הונאה לעסקאות לגיטימיות.  
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### ניהול עושר

בניהול עושר, אדם או חברה מנהלים השקעות עבור לקוחותיהם. תפקידם הוא לשמר ולהגדיל את העושר לטווח הארוך, ולכן חשוב לבחור השקעות שמניבות ביצועים טובים.

אחת הדרכים להעריך את ביצועי ההשקעה היא באמצעות רגרסיה סטטיסטית. [רגרסיה ליניארית](../../2-Regression/1-Tools/README.md) היא כלי חשוב להבנת ביצועי קרן ביחס למדד מסוים. ניתן גם להסיק האם תוצאות הרגרסיה הן משמעותיות סטטיסטית, או כמה הן ישפיעו על השקעות הלקוח. ניתן להרחיב את הניתוח באמצעות רגרסיה מרובה, שבה ניתן לקחת בחשבון גורמי סיכון נוספים. לדוגמה כיצד זה יעבוד עבור קרן ספציפית, עיינו במאמר למטה על הערכת ביצועי קרן באמצעות רגרסיה.  
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 חינוך

תחום החינוך הוא גם תחום מעניין שבו ניתן ליישם למידת מכונה. ישנן בעיות מעניינות להתמודד איתן כמו זיהוי רמאות במבחנים או חיבורים, או ניהול הטיות, מכוונות או לא, בתהליך התיקון.

### חיזוי התנהגות תלמידים

[Coursera](https://coursera.com), ספק קורסים פתוחים מקוון, מחזיק בלוג טכנולוגי נהדר שבו הם דנים בהחלטות הנדסיות רבות. במקרה זה, הם שרטטו קו רגרסיה כדי לנסות לחקור כל קשר בין דירוג NPS (Net Promoter Score) נמוך לבין שמירה על קורס או נשירה ממנו.  
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### הפחתת הטיות

[Grammarly](https://grammarly.com), עוזר כתיבה שבודק שגיאות כתיב ודקדוק, משתמש במערכות מתקדמות של [עיבוד שפה טבעית](../../6-NLP/README.md) במוצריו. הם פרסמו מחקר מעניין בבלוג הטכנולוגי שלהם על איך הם התמודדו עם הטיה מגדרית בלמידת מכונה, כפי שלמדתם בשיעור ההוגנות המבואי שלנו.  
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 קמעונאות

תחום הקמעונאות יכול בהחלט להרוויח משימוש בלמידת מכונה, החל מיצירת מסע לקוח טוב יותר ועד לניהול מלאי בצורה אופטימלית.

### התאמת מסע הלקוח

ב-Wayfair, חברה שמוכרת מוצרים לבית כמו רהיטים, עזרה ללקוחות למצוא את המוצרים הנכונים לטעמם ולצרכיהם היא קריטית. במאמר זה, מהנדסים מהחברה מתארים כיצד הם משתמשים בלמידת מכונה ובעיבוד שפה טבעית כדי "להציג את התוצאות הנכונות ללקוחות". במיוחד, מנוע כוונת השאילתה שלהם נבנה כדי להשתמש בחילוץ ישויות, אימון מסווגים, חילוץ נכסים ודעות, ותיוג רגשות על ביקורות לקוחות. זהו מקרה שימוש קלאסי של איך NLP עובד בקמעונאות מקוונת.  
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### ניהול מלאי

חברות חדשניות וזריזות כמו [StitchFix](https://stitchfix.com), שירות קופסאות ששולח בגדים לצרכנים, מסתמכות רבות על למידת מכונה להמלצות וניהול מלאי. צוותי הסטיילינג שלהם עובדים יחד עם צוותי הסחורה שלהם, למעשה: "אחד ממדעני הנתונים שלנו התנסה באלגוריתם גנטי ויישם אותו על בגדים כדי לחזות מה יהיה פריט לבוש מצליח שלא קיים היום. הבאנו את זה לצוות הסחורה ועכשיו הם יכולים להשתמש בזה ככלי."  
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 בריאות

תחום הבריאות יכול לנצל למידת מכונה כדי לייעל משימות מחקר וגם בעיות לוגיסטיות כמו אשפוז חוזר של מטופלים או עצירת התפשטות מחלות.

### ניהול ניסויים קליניים

רעילות בניסויים קליניים היא דאגה מרכזית עבור יצרני תרופות. כמה רעילות היא נסבלת? במחקר זה, ניתוח שיטות ניסוי קליניות שונות הוביל לפיתוח גישה חדשה לחיזוי הסיכויים לתוצאות ניסויים קליניים. במיוחד, הם הצליחו להשתמש ביער אקראי כדי ליצור [מסווג](../../4-Classification/README.md) שמסוגל להבחין בין קבוצות של תרופות.  
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### ניהול אשפוז חוזר בבתי חולים

טיפול בבתי חולים הוא יקר, במיוחד כאשר מטופלים צריכים להתאשפז שוב. מאמר זה דן בחברה שמשתמשת בלמידת מכונה כדי לחזות פוטנציאל אשפוז חוזר באמצעות אלגוריתמי [אשכולות](../../5-Clustering/README.md). אשכולות אלו עוזרים לאנליסטים "לגלות קבוצות של אשפוזים חוזרים שעשויים לחלוק סיבה משותפת".  
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### ניהול מחלות

המגפה האחרונה שמה זרקור על הדרכים שבהן למידת מכונה יכולה לעזור בעצירת התפשטות מחלות. במאמר זה, תזהו שימוש ב-ARIMA, עקומות לוגיסטיות, רגרסיה ליניארית ו-SARIMA. "עבודה זו היא ניסיון לחשב את שיעור התפשטות הנגיף הזה וכך לחזות את מקרי המוות, ההחלמות והמקרים המאושרים, כך שזה עשוי לעזור לנו להתכונן טוב יותר ולהישאר בחיים."  
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 אקולוגיה וטכנולוגיה ירוקה

הטבע והאקולוגיה מורכבים ממערכות רגישות רבות שבהן האינטראקציה בין בעלי חיים לטבע נכנסת למוקד. חשוב להיות מסוגלים למדוד מערכות אלו בצורה מדויקת ולפעול בהתאם אם משהו קורה, כמו שריפת יער או ירידה באוכלוסיית בעלי החיים.

### ניהול יערות

למדתם על [למידת חיזוקים](../../8-Reinforcement/README.md) בשיעורים קודמים. היא יכולה להיות מאוד שימושית כאשר מנסים לחזות דפוסים בטבע. במיוחד, ניתן להשתמש בה כדי לעקוב אחר בעיות אקולוגיות כמו שריפות יער והתפשטות מינים פולשים. בקנדה, קבוצת חוקרים השתמשה בלמידת חיזוקים כדי לבנות מודלים של דינמיקת שריפות יער מתמונות לוויין. באמצעות "תהליך התפשטות מרחבי (SSP)" חדשני, הם דמיינו שריפת יער כ"סוכן בכל תא בנוף". "סט הפעולות שהאש יכולה לבצע ממיקום בכל נקודת זמן כולל התפשטות צפונה, דרומה, מזרחה או מערבה או אי התפשטות.

גישה זו הופכת את ההגדרה הרגילה של למידת חיזוקים מכיוון שהדינמיקה של תהליך ההחלטה של מרקוב (MDP) המתאים היא פונקציה ידועה להתפשטות מיידית של שריפות." קראו עוד על האלגוריתמים הקלאסיים שבהם השתמשה הקבוצה בקישור למטה.  
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### חישת תנועה של בעלי חיים

בעוד שלמידה עמוקה יצרה מהפכה במעקב חזותי אחר תנועות בעלי חיים (ניתן לבנות [עוקב דובי קוטב](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) משלכם כאן), למידת מכונה קלאסית עדיין יש מקום במשימה זו.

חיישנים למעקב אחר תנועות של בעלי חיים בחוות ו-IoT עושים שימוש בסוג זה של עיבוד חזותי, אך טכניקות למידת מכונה בסיסיות יותר מועילות לעיבוד מקדים של נתונים. לדוגמה, במאמר זה, תנוחות כבשים נוטרו ונותחו באמצעות אלגוריתמי מסווגים שונים. ייתכן שתזהו את עקומת ROC בעמוד 335.  
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ ניהול אנרגיה

בשיעורים שלנו על [חיזוי סדרות זמן](../../7-TimeSeries/README.md), העלינו את הרעיון של מדחנים חכמים כדי לייצר הכנסות לעיר על סמך הבנת היצע וביקוש. מאמר זה דן בפירוט כיצד אשכולות, רגרסיה וחיזוי סדרות זמן שולבו כדי לעזור לחזות שימוש עתידי באנרגיה באירלנד, בהתבסס על מדידה חכמה.  
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 ביטוח

תחום הביטוח הוא תחום נוסף שמשתמש בלמידת מכונה כדי לבנות ולייעל מודלים פיננסיים ואקטואריים.

### ניהול תנודתיות

MetLife, ספק ביטוח חיים, פתוח לגבי הדרך שבה הם מנתחים ומפחיתים תנודתיות במודלים הפיננסיים שלהם. במאמר זה תבחינו בהדמיות סיווג בינאריות ואורדינליות. תגלו גם הדמיות חיזוי.  
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 אמנות, תרבות וספרות

בתחום האמנות, למשל בעיתונאות, ישנן בעיות מעניינות רבות. זיהוי חדשות מזויפות הוא בעיה גדולה שכן הוכח שהיא משפיעה על דעת הקהל ואפילו על הפלת דמוקרטיות. מוזיאונים יכולים גם להרוויח משימוש בלמידת מכונה בכל דבר, החל ממציאת קשרים בין פריטים ועד תכנון משאבים.

### זיהוי חדשות מזויפות

זיהוי חדשות מזויפות הפך למשחק של חתול ועכבר במדיה של היום. במאמר זה, חוקרים מציעים שמערכת שמשלבת כמה מטכניקות למידת המכונה שלמדנו יכולה להיבחן והמודל הטוב ביותר ייושם: "מערכת זו מבוססת על עיבוד שפה טבעית כדי לחלץ תכונות מהנתונים ולאחר מכן תכונות אלו משמשות לאימון מסווגי למידת מכונה כמו Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), ו-Logistic Regression (LR)."  
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

מאמר זה מראה כיצד שילוב תחומים שונים של למידת מכונה יכול להפיק תוצאות מעניינות שיכולות לעזור לעצור את התפשטות החדשות המזויפות ולמנוע נזק אמיתי; במקרה זה, המניע היה התפשטות שמועות על טיפולי COVID שגרמו לאלימות המונית.

### למידת מכונה במוזיאונים

מוזיאונים נמצאים על סף מהפכת AI שבה קטלוג ודיגיטציה של אוספים ומציאת קשרים בין פריטים הופכים לקלים יותר ככל שהטכנולוגיה מתקדמת. פרויקטים כמו [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) עוזרים לפתוח את המסתורין של אוספים בלתי נגישים כמו הארכיונים של הוותיקן. אבל, ההיבט העסקי של מוזיאונים מרוויח גם ממודלים של למידת מכונה.

לדוגמה, מכון האמנות של שיקגו בנה מודלים כדי לחזות מה מעניין את הקהל ומתי הוא יגיע לתערוכות. המטרה היא ליצור חוויות מבקר מותאמות ואופטימליות בכל פעם שהמשתמש מבקר במוזיאון. "במהלך שנת הכספים 2017, המודל חזה נוכחות והכנסות בדיוק של 1 אחוז, אומר אנדרו סימניק, סגן נשיא בכיר במכון האמנות."  
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 שיווק

### פילוח לקוחות

אסטרטגיות השיווק היעילות ביותר מכוונות ללקוחות בדרכים שונות בהתבסס על קבוצות שונות. במאמר זה, נדונים השימושים באלגוריתמי אשכולות כדי לתמוך בשיווק מובחן. שיווק מובחן עוזר לחברות לשפר את זיהוי המותג, להגיע ליותר לקוחות ולהרוויח יותר כסף.  
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 אתגר

זהו תחום נוסף שמרוויח מחלק מהטכניקות שלמדתם בקורס זה, וגלה כיצד הוא משתמש בלמידת מכונה.
## [שאלון לאחר ההרצאה](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

לצוות מדעי הנתונים של Wayfair יש כמה סרטונים מעניינים על איך הם משתמשים בלמידת מכונה בחברה שלהם. שווה [להציץ](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## משימה

[חיפוש אוצר בלמידת מכונה](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). בעוד שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.