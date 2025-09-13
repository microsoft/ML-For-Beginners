<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T20:43:53+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "he"
}
-->
# ניתוח רגשות עם ביקורות על מלונות

עכשיו, לאחר שחקרת את מערך הנתונים לעומק, הגיע הזמן לסנן את העמודות ולהשתמש בטכניקות עיבוד שפה טבעית (NLP) על מערך הנתונים כדי לקבל תובנות חדשות על המלונות.

## [מבחן מקדים להרצאה](https://ff-quizzes.netlify.app/en/ml/)

### פעולות סינון וניתוח רגשות

כפי שכנראה שמת לב, יש כמה בעיות במערך הנתונים. חלק מהעמודות מלאות במידע חסר תועלת, אחרות נראות לא נכונות. גם אם הן נכונות, לא ברור כיצד חושבו, ואין אפשרות לאמת את התשובות באופן עצמאי באמצעות חישובים משלך.

## תרגיל: עוד קצת עיבוד נתונים

נקה את הנתונים עוד קצת. הוסף עמודות שיהיו שימושיות בהמשך, שנה את הערכים בעמודות אחרות, והסר עמודות מסוימות לחלוטין.

1. עיבוד ראשוני של עמודות

   1. הסר את `lat` ו-`lng`

   2. החלף את הערכים ב-`Hotel_Address` בערכים הבאים (אם הכתובת מכילה את שם העיר והמדינה, שנה אותה רק לעיר ולמדינה).

      אלו הערים והמדינות היחידות במערך הנתונים:

      אמסטרדם, הולנד

      ברצלונה, ספרד

      לונדון, בריטניה

      מילאנו, איטליה

      פריז, צרפת

      וינה, אוסטריה 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      עכשיו תוכל לשאול נתונים ברמת מדינה:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. עיבוד עמודות מטא-ביקורת של מלונות

  1. הסר את `Additional_Number_of_Scoring`

  1. החלף את `Total_Number_of_Reviews` במספר הכולל של הביקורות על המלון שבפועל נמצאות במערך הנתונים 

  1. החלף את `Average_Score` בציון שחושב על ידינו

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. עיבוד עמודות ביקורת

   1. הסר את `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ו-`days_since_review`

   2. שמור את `Reviewer_Score`, `Negative_Review` ו-`Positive_Review` כפי שהם,
     
   3. שמור את `Tags` לעת עתה

     - נבצע פעולות סינון נוספות על התגים בחלק הבא ואז נסיר את התגים

4. עיבוד עמודות מבקר

  1. הסר את `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. שמור את `Reviewer_Nationality`

### עמודות תג

עמודת ה-`Tag` בעייתית מכיוון שהיא רשימה (בצורת טקסט) המאוחסנת בעמודה. למרבה הצער, הסדר ומספר תתי-הקטעים בעמודה זו אינם תמיד זהים. קשה לאדם לזהות את הביטויים הנכונים שיש להתעניין בהם, מכיוון שיש 515,000 שורות ו-1427 מלונות, ולכל אחד יש אפשרויות מעט שונות שהמבקר יכול לבחור. כאן נכנס לתמונה NLP. ניתן לסרוק את הטקסט ולמצוא את הביטויים הנפוצים ביותר ולספור אותם.

למרבה הצער, אנחנו לא מעוניינים במילים בודדות, אלא בביטויים מרובי מילים (לדוגמה, *נסיעת עסקים*). הפעלת אלגוריתם חלוקת תדירות ביטויים מרובי מילים על כמות נתונים כזו (6762646 מילים) עשויה לקחת זמן רב במיוחד, אך מבלי להסתכל על הנתונים, נראה שזהו מחיר הכרחי. כאן ניתוח נתונים חקרני מועיל, מכיוון שראית דוגמה של התגים כמו `[' נסיעת עסקים  ', ' מטייל יחיד ', ' חדר יחיד ', ' שהה 5 לילות ', ' נשלח ממכשיר נייד ']`, תוכל להתחיל לשאול אם ניתן לצמצם משמעותית את העיבוד שעליך לבצע. למרבה המזל, זה אפשרי - אבל קודם עליך לבצע כמה צעדים כדי לוודא מהם התגים הרלוונטיים.

### סינון תגים

זכור שהמטרה של מערך הנתונים היא להוסיף רגשות ועמודות שיעזרו לך לבחור את המלון הטוב ביותר (עבור עצמך או אולי עבור לקוח שמבקש ממך ליצור בוט המלצות על מלונות). עליך לשאול את עצמך אם התגים שימושיים או לא במערך הנתונים הסופי. הנה פרשנות אחת (אם היית זקוק למערך הנתונים מסיבות אחרות, ייתכן שתגים שונים יישארו/יוסרו מהבחירה):

1. סוג הנסיעה רלוונטי, והוא צריך להישאר
2. סוג קבוצת האורחים חשוב, והוא צריך להישאר
3. סוג החדר, הסוויטה או הסטודיו שבו האורח שהה אינו רלוונטי (לכל המלונות יש בעצם אותם חדרים)
4. המכשיר שעליו נשלחה הביקורת אינו רלוונטי
5. מספר הלילות שהמבקר שהה *יכול* להיות רלוונטי אם תייחס שהיות ארוכות יותר לכך שהם אהבו את המלון יותר, אבל זה גבולי, וסביר להניח שאינו רלוונטי

לסיכום, **שמור 2 סוגי תגים והסר את האחרים**.

ראשית, אינך רוצה לספור את התגים עד שהם יהיו בפורמט טוב יותר, כלומר יש להסיר את הסוגריים המרובעים והמרכאות. ניתן לעשות זאת בכמה דרכים, אך כדאי לבחור את המהירה ביותר מכיוון שזה עשוי לקחת זמן רב לעבד הרבה נתונים. למרבה המזל, ל-pandas יש דרך קלה לבצע כל אחד מהשלבים הללו.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

כל תג הופך למשהו כמו: `נסיעת עסקים, מטייל יחיד, חדר יחיד, שהה 5 לילות, נשלח ממכשיר נייד`. 

לאחר מכן אנו מוצאים בעיה. חלק מהביקורות, או השורות, מכילות 5 עמודות, חלק 3, חלק 6. זהו תוצאה של אופן יצירת מערך הנתונים, וקשה לתקן. אתה רוצה לקבל ספירת תדירות של כל ביטוי, אך הם בסדר שונה בכל ביקורת, כך שהספירה עשויה להיות שגויה, ומלון עשוי לא לקבל תג שהגיע לו.

במקום זאת תשתמש בסדר השונה לטובתנו, מכיוון שכל תג הוא מרובה מילים אך גם מופרד באמצעות פסיק! הדרך הפשוטה ביותר לעשות זאת היא ליצור 6 עמודות זמניות עם כל תג מוכנס לעמודה המתאימה לסדר שלו בתג. לאחר מכן תוכל למזג את 6 העמודות לעמודה גדולה אחת ולהפעיל את השיטה `value_counts()` על העמודה המתקבלת. בהדפסה תראה שהיו 2428 תגים ייחודיים. הנה דוגמה קטנה:

| Tag                            | Count  |
| ------------------------------ | ------ |
| נסיעת פנאי                   | 417778 |
| נשלח ממכשיר נייד             | 307640 |
| זוג                          | 252294 |
| שהה לילה אחד                 | 193645 |
| שהה 2 לילות                 | 133937 |
| מטייל יחיד                   | 108545 |
| שהה 3 לילות                 | 95821  |
| נסיעת עסקים                  | 82939  |
| קבוצה                        | 65392  |
| משפחה עם ילדים קטנים         | 61015  |
| שהה 4 לילות                 | 47817  |
| חדר זוגי                     | 35207  |
| חדר זוגי סטנדרטי             | 32248  |
| חדר זוגי משופר               | 31393  |
| משפחה עם ילדים גדולים        | 26349  |
| חדר זוגי דלוקס               | 24823  |
| חדר זוגי או תאומים           | 22393  |
| שהה 5 לילות                 | 20845  |
| חדר זוגי או תאומים סטנדרטי   | 17483  |
| חדר זוגי קלאסי               | 16989  |
| חדר זוגי או תאומים משופר     | 13570  |
| 2 חדרים                      | 12393  |

חלק מהתגים הנפוצים כמו `נשלח ממכשיר נייד` אינם מועילים לנו, ולכן ייתכן שזה רעיון חכם להסיר אותם לפני ספירת הופעת הביטויים, אך זו פעולה כה מהירה שניתן להשאיר אותם ולהתעלם מהם.

### הסרת תגים של אורך שהייה

הסרת תגים אלו היא שלב 1, היא מפחיתה מעט את מספר התגים שיש לשקול. שים לב שאינך מסיר אותם ממערך הנתונים, אלא פשוט בוחר להסיר אותם משיקול כערכים לספירה/שמירה במערך הביקורות.

| אורך שהייה   | Count  |
| ---------------- | ------ |
| שהה לילה אחד   | 193645 |
| שהה  2 לילות | 133937 |
| שהה 3 לילות  | 95821  |
| שהה  4 לילות | 47817  |
| שהה 5 לילות  | 20845  |
| שהה  6 לילות | 9776   |
| שהה 7 לילות  | 7399   |
| שהה  8 לילות | 2502   |
| שהה 9 לילות  | 1293   |
| ...              | ...    |

יש מגוון עצום של חדרים, סוויטות, סטודיו, דירות וכדומה. כולם פחות או יותר אותו דבר ואינם רלוונטיים עבורך, לכן הסר אותם משיקול.

| סוג חדר                  | Count |
| ----------------------------- | ----- |
| חדר זוגי                   | 35207 |
| חדר זוגי סטנדרטי         | 32248 |
| חדר זוגי משופר          | 31393 |
| חדר זוגי דלוקס           | 24823 |
| חדר זוגי או תאומים           | 22393 |
| חדר זוגי או תאומים סטנדרטי | 17483 |
| חדר זוגי קלאסי            | 16989 |
| חדר זוגי או תאומים משופר | 13570 |

לבסוף, וזה משמח (כי זה לא דרש הרבה עיבוד בכלל), תישאר עם התגים הבאים *שימושיים*:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| נסיעת פנאי                                  | 417778 |
| זוג                                        | 252294 |
| מטייל יחיד                                | 108545 |
| נסיעת עסקים                                 | 82939  |
| קבוצה (משולב עם מטיילים עם חברים) | 67535  |
| משפחה עם ילדים קטנים                    | 61015  |
| משפחה עם ילדים גדולים                   | 26349  |
| עם חיית מחמד                                   | 1405   |

אפשר לטעון ש`מטיילים עם חברים` זהה פחות או יותר ל`קבוצה`, וזה יהיה הוגן לשלב את השניים כפי שמוצג לעיל. הקוד לזיהוי התגים הנכונים נמצא ב-[מחברת התגים](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

השלב האחרון הוא ליצור עמודות חדשות עבור כל אחד מהתגים הללו. לאחר מכן, עבור כל שורת ביקורת, אם עמודת ה-`Tag` תואמת לאחת מהעמודות החדשות, הוסף 1, אם לא, הוסף 0. התוצאה הסופית תהיה ספירה של כמה מבקרים בחרו במלון זה (במצטבר) עבור, למשל, עסקים מול פנאי, או להביא חיית מחמד, וזה מידע שימושי בעת המלצה על מלון.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### שמור את הקובץ שלך

לבסוף, שמור את מערך הנתונים כפי שהוא עכשיו עם שם חדש.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## פעולות ניתוח רגשות

בחלק האחרון הזה, תיישם ניתוח רגשות על עמודות הביקורות ותשמור את התוצאות במערך נתונים.

## תרגיל: טען ושמור את הנתונים המסוננים

שים לב שעכשיו אתה טוען את מערך הנתונים המסונן שנשמר בחלק הקודם, **לא** את מערך הנתונים המקורי.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### הסרת מילים נפוצות

אם היית מפעיל ניתוח רגשות על עמודות הביקורות השליליות והחיוביות, זה יכול לקחת זמן רב. נבדק על מחשב נייד חזק עם מעבד מהיר, זה לקח 12 - 14 דקות תלוי באיזו ספריית ניתוח רגשות נעשה שימוש. זה זמן (יחסית) ארוך, ולכן כדאי לבדוק אם ניתן להאיץ את התהליך.

הסרת מילים נפוצות, או מילים באנגלית שאינן משנות את הרגש של משפט, היא הצעד הראשון. על ידי הסרתן, ניתוח הרגשות אמור לרוץ מהר יותר, אך לא להיות פחות מדויק (מכיוון שהמילים הנפוצות אינן משפיעות על הרגש, אך הן מאטות את הניתוח). 

הביקורת השלילית הארוכה ביותר הייתה 395 מילים, אך לאחר הסרת המילים הנפוצות, היא מכילה 195 מילים.

הסרת המילים הנפוצות היא גם פעולה מהירה, הסרת המילים הנפוצות מ-2 עמודות ביקורות על פני 515,000 שורות לקחה 3.3 שניות במכשיר הבדיקה. זה יכול לקחת מעט יותר או פחות זמן עבורך תלוי במהירות המעבד של המכשיר שלך, זיכרון RAM, האם יש לך SSD או לא, וכמה גורמים נוספים. הקיצור היחסי של הפעולה אומר שאם זה משפר את זמן ניתוח הרגשות, אז זה שווה לעשות.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### ביצוע ניתוח רגשות

עכשיו עליך לחשב את ניתוח הרגשות עבור עמודות הביקורות השליליות והחיוביות, ולשמור את התוצאה ב-2 עמודות חדשות. המבחן של הרגש יהיה להשוות אותו לציון של המבקר עבור אותה ביקורת. לדוגמה, אם הרגש חושב שלביקורת השלילית היה רגש של 1 (רגש חיובי מאוד) ולביקורת החיובית רגש של 1, אך המבקר נתן למלון את הציון הנמוך ביותר האפשרי, אז או שהטקסט של הביקורת אינו תואם לציון, או שמנתח הרגשות לא הצליח לזהות את הרגש בצורה נכונה. עליך לצפות שחלק מציוני הרגשות יהיו שגויים לחלוטין, ולעיתים זה יהיה ניתן להסבר, למשל הביקורת יכולה להיות מאוד סרקסטית "כמובן ש-מ-א-ו-ד אהבתי לישון בחדר בלי חימום" ומנתח הרגשות חושב שזה רגש חיובי, למרות שאדם שקורא את זה היה יודע שזה סרקזם.
NLTK מספקת מנתחי רגשות שונים ללמידה, ואתם יכולים להחליף ביניהם ולבדוק אם הניתוח הרגשי מדויק יותר או פחות. ניתוח הרגשות של VADER משמש כאן.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

מאוחר יותר בתוכנית שלכם, כאשר תהיו מוכנים לחשב רגשות, תוכלו ליישם זאת על כל ביקורת באופן הבא:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

זה לוקח בערך 120 שניות במחשב שלי, אבל זה ישתנה ממחשב למחשב. אם אתם רוצים להדפיס את התוצאות ולבדוק אם הרגש תואם את הביקורת:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

הדבר האחרון שיש לעשות עם הקובץ לפני השימוש בו באתגר הוא לשמור אותו! כדאי גם לשקול לסדר מחדש את כל העמודות החדשות שלכם כך שיהיה קל לעבוד איתן (עבור בני אדם, זה שינוי קוסמטי).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

עליכם להריץ את כל הקוד עבור [מחברת הניתוח](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (לאחר שהרצתם [את מחברת הסינון](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) כדי ליצור את קובץ Hotel_Reviews_Filtered.csv).

לסיכום, השלבים הם:

1. קובץ הנתונים המקורי **Hotel_Reviews.csv** נחקר בשיעור הקודם עם [מחברת החקירה](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv מסונן על ידי [מחברת הסינון](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ומתקבל **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv מעובד על ידי [מחברת ניתוח הרגשות](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ומתקבל **Hotel_Reviews_NLP.csv**
4. השתמשו ב-Hotel_Reviews_NLP.csv באתגר ה-NLP למטה

### מסקנה

כשאתם התחלתם, היה לכם קובץ נתונים עם עמודות ונתונים, אבל לא הכל היה ניתן לאימות או לשימוש. חקרתם את הנתונים, סיננתם את מה שלא היה נחוץ, המרתם תגיות למשהו שימושי, חישבתם ממוצעים משלכם, הוספתם עמודות רגשות, וכנראה למדתם דברים מעניינים על עיבוד טקסט טבעי.

## [שאלון לאחר ההרצאה](https://ff-quizzes.netlify.app/en/ml/)

## אתגר

עכשיו, כשניתחתם את הרגשות בקובץ הנתונים שלכם, נסו להשתמש באסטרטגיות שלמדתם בתוכנית הלימודים הזו (אולי clustering?) כדי לזהות דפוסים סביב רגשות.

## סקירה ולימוד עצמי

קחו [את המודול הזה](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) כדי ללמוד עוד ולהשתמש בכלים שונים לחקר רגשות בטקסט.

## משימה

[נסו קובץ נתונים אחר](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.