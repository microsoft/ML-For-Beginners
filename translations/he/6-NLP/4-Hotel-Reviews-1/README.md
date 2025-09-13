<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T20:29:24+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "he"
}
-->
# ניתוח רגשות עם ביקורות על מלונות - עיבוד הנתונים

בפרק זה תשתמשו בטכניקות שלמדתם בשיעורים הקודמים כדי לבצע ניתוח נתונים חקרני על מערך נתונים גדול. לאחר שתבינו היטב את השימושיות של העמודות השונות, תלמדו:

- כיצד להסיר עמודות שאינן נחוצות
- כיצד לחשב נתונים חדשים בהתבסס על עמודות קיימות
- כיצד לשמור את מערך הנתונים המתקבל לשימוש באתגר הסופי

## [מבחן מקדים להרצאה](https://ff-quizzes.netlify.app/en/ml/)

### מבוא

עד כה למדתם כיצד נתוני טקסט שונים מאוד מסוגי נתונים מספריים. אם מדובר בטקסט שנכתב או נאמר על ידי אדם, ניתן לנתח אותו כדי למצוא דפוסים ותדירויות, רגשות ומשמעויות. השיעור הזה לוקח אתכם למערך נתונים אמיתי עם אתגר אמיתי: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** הכולל [רישיון CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). הנתונים נגרדו מ-Booking.com ממקורות ציבוריים. יוצר מערך הנתונים הוא Jiashen Liu.

### הכנה

תצטרכו:

* יכולת להריץ מחברות .ipynb באמצעות Python 3
* pandas
* NLTK, [שאותו יש להתקין באופן מקומי](https://www.nltk.org/install.html)
* מערך הנתונים הזמין ב-Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). גודלו כ-230 MB לאחר חילוץ. הורידו אותו לתיקיית השורש `/data` המשויכת לשיעורי NLP אלו.

## ניתוח נתונים חקרני

האתגר הזה מניח שאתם בונים בוט המלצות למלונות באמצעות ניתוח רגשות ודירוגי אורחים. מערך הנתונים שבו תשתמשו כולל ביקורות על 1493 מלונות שונים ב-6 ערים.

באמצעות Python, מערך נתונים של ביקורות על מלונות, וניתוח רגשות של NLTK תוכלו לגלות:

* מהם המילים והביטויים הנפוצים ביותר בביקורות?
* האם *תגיות* רשמיות המתארות מלון מתואמות עם דירוגי ביקורות (לדוגמה, האם יש יותר ביקורות שליליות עבור מלון מסוים מ-*משפחה עם ילדים קטנים* מאשר מ-*מטייל יחיד*, אולי מצביע על כך שהוא מתאים יותר ל-*מטיילים יחידים*)?
* האם דירוגי הרגשות של NLTK 'מסכימים' עם הדירוג המספרי של המבקר?

#### מערך הנתונים

בואו נחקור את מערך הנתונים שהורדתם ושמרתם באופן מקומי. פתחו את הקובץ בעורך כמו VS Code או אפילו Excel.

הכותרות במערך הנתונים הן כדלקמן:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

הנה הן מקובצות בצורה שעשויה להיות קלה יותר לבדיקה:
##### עמודות מלון

* `Hotel_Name`, `Hotel_Address`, `lat` (קו רוחב), `lng` (קו אורך)
  * באמצעות *lat* ו-*lng* תוכלו למפות מיקום המלונות עם Python (אולי בצבעים שונים עבור ביקורות חיוביות ושליליות)
  * Hotel_Address אינו שימושי במיוחד עבורנו, וכנראה נחליף אותו במדינה לצורך מיון וחיפוש קלים יותר

**עמודות מטא-ביקורת של מלון**

* `Average_Score`
  * לפי יוצר מערך הנתונים, עמודה זו היא *הציון הממוצע של המלון, מחושב על סמך התגובה האחרונה בשנה האחרונה*. זו נראית דרך לא שגרתית לחשב את הציון, אך אלו הנתונים שנגרדו ולכן נוכל לקבלם כפי שהם לעת עתה.

  ✅ בהתבסס על העמודות האחרות במערך הנתונים, האם תוכלו לחשוב על דרך אחרת לחשב את הציון הממוצע?

* `Total_Number_of_Reviews`
  * המספר הכולל של הביקורות שהמלון קיבל - לא ברור (ללא כתיבת קוד) אם זה מתייחס לביקורות במערך הנתונים.
* `Additional_Number_of_Scoring`
  * פירושו שניתן ציון ביקורת אך לא נכתבה ביקורת חיובית או שלילית על ידי המבקר

**עמודות ביקורת**

- `Reviewer_Score`
  - זהו ערך מספרי עם מקסימום ספרה עשרונית אחת בין הערכים המינימליים והמקסימליים 2.5 ו-10
  - לא מוסבר מדוע 2.5 הוא הציון הנמוך ביותר האפשרי
- `Negative_Review`
  - אם מבקר לא כתב דבר, שדה זה יכיל "**No Negative**"
  - שימו לב שמבקר עשוי לכתוב ביקורת חיובית בעמודת הביקורת השלילית (לדוגמה, "אין שום דבר רע במלון הזה")
- `Review_Total_Negative_Word_Counts`
  - ספירת מילים שליליות גבוהה יותר מצביעה על ציון נמוך יותר (ללא בדיקת הרגש)
- `Positive_Review`
  - אם מבקר לא כתב דבר, שדה זה יכיל "**No Positive**"
  - שימו לב שמבקר עשוי לכתוב ביקורת שלילית בעמודת הביקורת החיובית (לדוגמה, "אין שום דבר טוב במלון הזה בכלל")
- `Review_Total_Positive_Word_Counts`
  - ספירת מילים חיוביות גבוהה יותר מצביעה על ציון גבוה יותר (ללא בדיקת הרגש)
- `Review_Date` ו-`days_since_review`
  - ניתן ליישם מדד של טריות או התיישנות על ביקורת (ביקורות ישנות עשויות להיות פחות מדויקות מביקורות חדשות מכיוון שהניהול השתנה, או שיפוצים בוצעו, או נוספה בריכה וכו')
- `Tags`
  - אלו תיאורים קצרים שמבקר עשוי לבחור כדי לתאר את סוג האורח שהוא היה (לדוגמה, יחיד או משפחה), סוג החדר שהיה לו, משך השהות ואופן הגשת הביקורת.
  - למרבה הצער, השימוש בתגיות אלו בעייתי, בדקו את הסעיף למטה שמדבר על השימושיות שלהן

**עמודות מבקר**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - זה עשוי להיות גורם במודל המלצות, למשל, אם תוכלו לקבוע שמבקרים פורים יותר עם מאות ביקורות היו נוטים יותר להיות שליליים מאשר חיוביים. עם זאת, המבקר של כל ביקורת מסוימת אינו מזוהה עם קוד ייחודי, ולכן לא ניתן לקשר אותו למערך ביקורות. ישנם 30 מבקרים עם 100 או יותר ביקורות, אך קשה לראות כיצד זה יכול לסייע במודל ההמלצות.
- `Reviewer_Nationality`
  - יש אנשים שעשויים לחשוב שלאומים מסוימים נוטים יותר לתת ביקורת חיובית או שלילית בגלל נטייה לאומית. היו זהירים בבניית השקפות אנקדוטליות כאלה לתוך המודלים שלכם. אלו סטריאוטיפים לאומיים (ולפעמים גזעיים), וכל מבקר היה אדם שכתב ביקורת על סמך חווייתו. ייתכן שהיא סוננה דרך עדשות רבות כמו שהיותיו הקודמות במלונות, המרחק שנסע, והטמפרמנט האישי שלו. קשה להצדיק את המחשבה שהלאום היה הסיבה לציון הביקורת.

##### דוגמאות

| ציון ממוצע | מספר כולל של ביקורות | ציון מבקר | ביקורת שלילית                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ביקורת חיובית                 | תגיות                                                                                      |
| ----------- | ---------------------- | --------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8         | 1945                   | 2.5       | זה כרגע לא מלון אלא אתר בנייה. הטרידו אותי מהבוקר המוקדם ועד כל היום עם רעש בנייה בלתי נסבל בזמן מנוחה לאחר נסיעה ארוכה ועבודה בחדר. אנשים עבדו כל היום עם פטישי אוויר בחדרים סמוכים. ביקשתי להחליף חדר אך לא היה חדר שקט זמין. כדי להחמיר את המצב, חויבתי יתר על המידה. יצאתי בערב מכיוון שהייתי צריך לעזוב טיסה מוקדמת מאוד וקיבלתי חשבון מתאים. יום לאחר מכן המלון ביצע חיוב נוסף ללא הסכמתי מעבר למחיר שהוזמן. זה מקום נורא. אל תענישו את עצמכם על ידי הזמנה כאן. | שום דבר. מקום נורא. התרחקו. | נסיעת עסקים. זוג. חדר זוגי סטנדרטי. שהייה של 2 לילות. |

כפי שאתם יכולים לראות, האורח הזה לא נהנה מהשהות שלו במלון. למלון יש ציון ממוצע טוב של 7.8 ו-1945 ביקורות, אך המבקר הזה נתן לו 2.5 וכתב 115 מילים על כמה שהשהות שלו הייתה שלילית. אם הוא לא כתב דבר בעמודת הביקורת החיובית, אפשר להסיק שלא היה שום דבר חיובי, אך הוא כתב 7 מילים של אזהרה. אם רק נספור מילים במקום את המשמעות או הרגש של המילים, ייתכן שתהיה לנו תמונה מעוותת של כוונת המבקר. באופן מוזר, הציון שלו של 2.5 מבלבל, כי אם השהות במלון הייתה כל כך גרועה, מדוע לתת לו נקודות בכלל? בחקירת מערך הנתונים מקרוב, תראו שהציון הנמוך ביותר האפשרי הוא 2.5, לא 0. הציון הגבוה ביותר האפשרי הוא 10.

##### תגיות

כפי שצוין לעיל, במבט ראשון, הרעיון להשתמש ב-`Tags` כדי לקטלג את הנתונים נראה הגיוני. למרבה הצער, תגיות אלו אינן סטנדרטיות, מה שאומר שבמלון מסוים, האפשרויות עשויות להיות *Single room*, *Twin room*, ו-*Double room*, אך במלון הבא, הן יהיו *Deluxe Single Room*, *Classic Queen Room*, ו-*Executive King Room*. אלו עשויים להיות אותם דברים, אך יש כל כך הרבה וריאציות שהבחירה הופכת ל:

1. ניסיון לשנות את כל המונחים לסטנדרט יחיד, מה שקשה מאוד, מכיוון שלא ברור מה תהיה דרך ההמרה בכל מקרה (לדוגמה, *Classic single room* ממופה ל-*Single room* אך *Superior Queen Room with Courtyard Garden or City View* קשה יותר למפות)

1. נוכל לקחת גישה של NLP ולמדוד את התדירות של מונחים מסוימים כמו *Solo*, *Business Traveller*, או *Family with young kids* כפי שהם חלים על כל מלון, ולשלב זאת במודל ההמלצות  

תגיות הן בדרך כלל (אך לא תמיד) שדה יחיד המכיל רשימה של 5 עד 6 ערכים מופרדים בפסיקים המתאימים ל-*סוג הטיול*, *סוג האורחים*, *סוג החדר*, *מספר הלילות*, ו-*סוג המכשיר שבו הוגשה הביקורת*. עם זאת, מכיוון שחלק מהמבקרים לא ממלאים כל שדה (ייתכן שהם משאירים אחד ריק), הערכים אינם תמיד באותו סדר.

לדוגמה, קחו את *סוג הקבוצה*. ישנם 1025 אפשרויות ייחודיות בשדה זה בעמודת `Tags`, ולמרבה הצער רק חלקן מתייחסות לקבוצה (חלקן הן סוג החדר וכו'). אם תסננו רק את אלו שמזכירים משפחה, התוצאות מכילות הרבה תוצאות מסוג *Family room*. אם תכללו את המונח *with*, כלומר תספרו את הערכים *Family with*, התוצאות טובות יותר, עם מעל 80,000 מתוך 515,000 התוצאות המכילות את הביטוי "Family with young children" או "Family with older children".

זה אומר שעמודת התגיות אינה חסרת תועלת לחלוטין עבורנו, אך יידרש מעט עבודה כדי להפוך אותה לשימושית.

##### ציון ממוצע של מלון

ישנם מספר מוזרויות או אי התאמות במערך הנתונים שאני לא מצליח להבין, אך הן מוצגות כאן כדי שתהיו מודעים להן בעת בניית המודלים שלכם. אם תצליחו להבין, אנא הודיעו לנו במדור הדיונים!

מערך הנתונים כולל את העמודות הבאות הקשורות לציון הממוצע ולמספר הביקורות:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

המלון היחיד עם מספר הביקורות הגבוה ביותר במערך הנתונים הזה הוא *Britannia International Hotel Canary Wharf* עם 4789 ביקורות מתוך 515,000. אך אם נסתכל על הערך `Total_Number_of_Reviews` עבור מלון זה, הוא 9086. ניתן להסיק שיש הרבה יותר ציונים ללא ביקורות, ולכן אולי כדאי להוסיף את ערך העמודה `Additional_Number_of_Scoring`. הערך הזה הוא 2682, והוספתו ל-4789 מביאה אותנו ל-7471, שעדיין חסרים 1615 כדי להגיע ל-`Total_Number_of_Reviews`.

אם ניקח את עמודת `Average_Score`, ניתן להסיק שזהו הממוצע של הביקורות במערך הנתונים, אך התיאור מ-Kaggle הוא "*Average Score of the hotel, calculated based on the latest comment in the last year*". זה לא נראה שימושי במיוחד, אך נוכל לחשב ממוצע משלנו בהתבסס על ציוני הביקורות במערך הנתונים. באמצעות אותו מלון כדוגמה, הציון הממוצע של המלון ניתן כ-7.1 אך הציון המחושב (ממוצע ציוני המבקרים *ב*מערך הנתונים) הוא 6.8. זה קרוב, אך לא אותו ערך, ואנו יכולים רק לנחש שהציונים שניתנו בביקורות `Additional_Number_of_Scoring` העלו את הממוצע ל-7.1. למרבה הצער, ללא דרך לבדוק או להוכיח את ההנחה הזו, קשה להשתמש או לסמוך על `Average_Score`, `Additional_Number_of_Scoring` ו-`Total_Number_of_Reviews` כאשר הם מבוססים על, או מתייחסים לנתונים שאין לנו.

כדי לסבך את העניינים עוד יותר, המלון עם מספר הביקורות השני הגבוה ביותר יש לו ציון ממוצע מחושב של 8.12 והציון הממוצע במערך הנתונים הוא 8.1. האם הציון הנכון הוא צירוף מקרים או שהמלון הראשון הוא אי התאמה?

בהנחה שהמלון הזה עשוי להיות חריג, ושאולי רוב הערכים מתאימים (אך חלקם לא מסיבה כלשהי) נכתוב תוכנית קצרה בהמשך כדי לחקור את הערכים במערך הנתונים ולקבוע את השימוש הנכון (או אי השימוש) בערכים.
> 🚨 הערה של זהירות  
>  
> כאשר עובדים עם מערך הנתונים הזה, תכתבו קוד שמחשב משהו מתוך הטקסט מבלי שתצטרכו לקרוא או לנתח את הטקסט בעצמכם. זו המהות של עיבוד שפה טבעית (NLP), לפרש משמעות או תחושה מבלי שאדם יצטרך לעשות זאת. עם זאת, ייתכן שתתקלו בכמה ביקורות שליליות. אני ממליץ לכם לא לקרוא אותן, כי אין צורך בכך. חלקן טיפשיות או לא רלוונטיות, כמו ביקורות שליליות על מלון בסגנון "מזג האוויר לא היה טוב", דבר שאינו בשליטת המלון, או למעשה, אף אחד. אבל יש גם צד אפל לחלק מהביקורות. לפעמים הביקורות השליליות הן גזעניות, סקסיסטיות, או מפלות על בסיס גיל. זה מצער אך צפוי במערך נתונים שנאסף מאתר ציבורי. ישנם מבקרים שמשאירים ביקורות שתמצאו אותן דוחות, לא נוחות, או מטרידות. עדיף לתת לקוד למדוד את התחושה מאשר לקרוא אותן בעצמכם ולהתעצב. עם זאת, מדובר במיעוט שכותב דברים כאלה, אבל הם קיימים בכל זאת.
## תרגיל - חקר נתונים
### טעינת הנתונים

זה מספיק לבחון את הנתונים באופן חזותי, עכשיו תכתבו קצת קוד ותקבלו תשובות! החלק הזה משתמש בספריית pandas. המשימה הראשונה שלכם היא לוודא שאתם יכולים לטעון ולקרוא את נתוני ה-CSV. לספריית pandas יש טוען CSV מהיר, והתוצאה נשמרת ב-DataFrame, כפי שראיתם בשיעורים הקודמים. ה-CSV שאנחנו טוענים מכיל יותר מחצי מיליון שורות, אבל רק 17 עמודות. pandas מספקת דרכים רבות וחזקות לעבוד עם DataFrame, כולל היכולת לבצע פעולות על כל שורה.

מכאן והלאה בשיעור הזה, יהיו קטעי קוד והסברים על הקוד וגם דיון על מה המשמעות של התוצאות. השתמשו בקובץ _notebook.ipynb_ המצורף עבור הקוד שלכם.

בואו נתחיל בטעינת קובץ הנתונים שבו תשתמשו:

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

עכשיו כשהנתונים נטענו, אנחנו יכולים לבצע עליהם פעולות. שמרו את הקוד הזה בראש התוכנית שלכם עבור החלק הבא.

## חקר הנתונים

במקרה הזה, הנתונים כבר *נקיים*, כלומר הם מוכנים לעבודה, ואין בהם תווים בשפות אחרות שעלולים להפריע לאלגוריתמים שמצפים לתווים באנגלית בלבד.

✅ ייתכן שתצטרכו לעבוד עם נתונים שדורשים עיבוד ראשוני כדי לעצב אותם לפני יישום טכניקות NLP, אבל לא הפעם. אם הייתם צריכים, איך הייתם מתמודדים עם תווים שאינם באנגלית?

קחו רגע לוודא שברגע שהנתונים נטענו, אתם יכולים לחקור אותם באמצעות קוד. קל מאוד להתמקד בעמודות `Negative_Review` ו-`Positive_Review`. הן מלאות בטקסט טבעי עבור אלגוריתמי ה-NLP שלכם לעיבוד. אבל חכו! לפני שאתם קופצים ל-NLP ולניתוח רגשות, כדאי שתעקבו אחרי הקוד למטה כדי לוודא שהערכים שניתנו בנתונים תואמים לערכים שאתם מחשבים עם pandas.

## פעולות על DataFrame

המשימה הראשונה בשיעור הזה היא לבדוק אם ההנחות הבאות נכונות על ידי כתיבת קוד שבוחן את ה-DataFrame (מבלי לשנות אותו).

> כמו בהרבה משימות תכנות, יש כמה דרכים להשלים את זה, אבל עצה טובה היא לעשות זאת בדרך הפשוטה והקלה ביותר, במיוחד אם זה יהיה קל יותר להבנה כשתחזרו לקוד הזה בעתיד. עם DataFrames, יש API מקיף שלרוב יציע דרך לעשות את מה שאתם רוצים בצורה יעילה.

התייחסו לשאלות הבאות כמשימות קוד ונסו לענות עליהן מבלי להסתכל על הפתרון.

1. הדפיסו את *הצורה* של ה-DataFrame שזה עתה טענתם (הצורה היא מספר השורות והעמודות).
2. חשבו את תדירות הלאומים של הסוקרים:
   1. כמה ערכים ייחודיים יש לעמודה `Reviewer_Nationality` ומה הם?
   2. איזה לאום של סוקרים הוא הנפוץ ביותר בנתונים (הדפיסו את המדינה ומספר הביקורות)?
   3. מהם 10 הלאומים הנפוצים ביותר הבאים ותדירותם?
3. איזה מלון קיבל את מספר הביקורות הגבוה ביותר עבור כל אחד מ-10 הלאומים הנפוצים ביותר?
4. כמה ביקורות יש לכל מלון (תדירות הביקורות לכל מלון) בנתונים?
5. למרות שיש עמודה `Average_Score` לכל מלון בנתונים, אתם יכולים גם לחשב ציון ממוצע (לקחת את הממוצע של כל ציוני הסוקרים בנתונים עבור כל מלון). הוסיפו עמודה חדשה ל-DataFrame שלכם עם הכותרת `Calc_Average_Score` שמכילה את הממוצע המחושב.
6. האם יש מלונות עם אותו ציון (מעוגל למקום העשרוני הראשון) ב-`Average_Score` וב-`Calc_Average_Score`?
   1. נסו לכתוב פונקציה ב-Python שמקבלת Series (שורה) כארגומנט ומשווה את הערכים, ומדפיסה הודעה כשהערכים אינם שווים. לאחר מכן השתמשו בשיטה `.apply()` כדי לעבד כל שורה עם הפונקציה.
7. חשבו והדפיסו כמה שורות יש עם ערכים של "No Negative" בעמודה `Negative_Review`.
8. חשבו והדפיסו כמה שורות יש עם ערכים של "No Positive" בעמודה `Positive_Review`.
9. חשבו והדפיסו כמה שורות יש עם ערכים של "No Positive" בעמודה `Positive_Review` **וגם** ערכים של "No Negative" בעמודה `Negative_Review`.

### תשובות בקוד

1. הדפיסו את *הצורה* של ה-DataFrame שזה עתה טענתם (הצורה היא מספר השורות והעמודות).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. חשבו את תדירות הלאומים של הסוקרים:

   1. כמה ערכים ייחודיים יש לעמודה `Reviewer_Nationality` ומה הם?
   2. איזה לאום של סוקרים הוא הנפוץ ביותר בנתונים (הדפיסו את המדינה ומספר הביקורות)?

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

   3. מהם 10 הלאומים הנפוצים ביותר הבאים ותדירותם?

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

3. איזה מלון קיבל את מספר הביקורות הגבוה ביותר עבור כל אחד מ-10 הלאומים הנפוצים ביותר?

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

4. כמה ביקורות יש לכל מלון (תדירות הביקורות לכל מלון) בנתונים?

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
   
   ייתכן שתשימו לב שהתוצאות *שנספרו בנתונים* אינן תואמות את הערך ב-`Total_Number_of_Reviews`. לא ברור אם הערך הזה בנתונים מייצג את מספר הביקורות הכולל שהמלון קיבל, אבל לא כולן נגרפו, או חישוב אחר. `Total_Number_of_Reviews` אינו משמש במודל בגלל חוסר הבהירות הזה.

5. למרות שיש עמודה `Average_Score` לכל מלון בנתונים, אתם יכולים גם לחשב ציון ממוצע (לקחת את הממוצע של כל ציוני הסוקרים בנתונים עבור כל מלון). הוסיפו עמודה חדשה ל-DataFrame שלכם עם הכותרת `Calc_Average_Score` שמכילה את הממוצע המחושב. הדפיסו את העמודות `Hotel_Name`, `Average_Score`, ו-`Calc_Average_Score`.

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

   ייתכן שתהיתם לגבי הערך `Average_Score` ולמה הוא לפעמים שונה מהממוצע המחושב. מכיוון שאנחנו לא יכולים לדעת למה חלק מהערכים תואמים, אבל אחרים יש להם הבדל, הכי בטוח במקרה הזה להשתמש בציוני הביקורות שיש לנו כדי לחשב את הממוצע בעצמנו. עם זאת, ההבדלים בדרך כלל מאוד קטנים, הנה המלונות עם ההבדל הגדול ביותר בין הממוצע בנתונים לבין הממוצע המחושב:

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

   עם רק מלון אחד שיש לו הבדל בציון גדול מ-1, זה אומר שאנחנו כנראה יכולים להתעלם מההבדל ולהשתמש בממוצע המחושב.

6. חשבו והדפיסו כמה שורות יש עם ערכים של "No Negative" בעמודה `Negative_Review`.

7. חשבו והדפיסו כמה שורות יש עם ערכים של "No Positive" בעמודה `Positive_Review`.

8. חשבו והדפיסו כמה שורות יש עם ערכים של "No Positive" בעמודה `Positive_Review` **וגם** ערכים של "No Negative" בעמודה `Negative_Review`.

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

## דרך נוספת

דרך נוספת לספור פריטים ללא Lambdas, ולהשתמש ב-sum כדי לספור את השורות:

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

   ייתכן ששמתם לב שיש 127 שורות שיש להן גם "No Negative" וגם "No Positive" כערכים בעמודות `Negative_Review` ו-`Positive_Review` בהתאמה. זה אומר שהסוקר נתן למלון ציון מספרי, אבל נמנע מלכתוב ביקורת חיובית או שלילית. למרבה המזל מדובר בכמות קטנה של שורות (127 מתוך 515738, או 0.02%), כך שזה כנראה לא יטה את המודל או התוצאות לכיוון מסוים, אבל ייתכן שלא ציפיתם שמאגר נתונים של ביקורות יכיל שורות ללא ביקורות, ולכן כדאי לחקור את הנתונים כדי לגלות שורות כאלה.

עכשיו כשחקרתם את מאגר הנתונים, בשיעור הבא תסננו את הנתונים ותוסיפו ניתוח רגשות.

---
## 🚀אתגר

השיעור הזה מדגים, כפי שראינו בשיעורים קודמים, כמה חשוב להבין את הנתונים ואת המוזרויות שלהם לפני ביצוע פעולות עליהם. נתונים מבוססי טקסט, במיוחד, דורשים בדיקה מדוקדקת. חפרו במאגרי נתונים שונים שמבוססים על טקסט ונסו לגלות אזורים שיכולים להכניס הטיה או רגשות מוטים למודל.

## [שאלון לאחר השיעור](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

קחו [את מסלול הלמידה הזה על NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) כדי לגלות כלים לנסות כשבונים מודלים מבוססי דיבור וטקסט.

## משימה 

[NLTK](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.