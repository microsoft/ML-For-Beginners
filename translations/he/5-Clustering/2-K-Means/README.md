<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T19:17:05+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "he"
}
-->
# אשכולות K-Means

## [מבחן מקדים](https://ff-quizzes.netlify.app/en/ml/)

בשיעור הזה תלמדו כיצד ליצור אשכולות באמצעות Scikit-learn וסט הנתונים של מוזיקה ניגרית שייבאתם קודם לכן. נכסה את היסודות של K-Means לצורך אשכולות. זכרו, כפי שלמדתם בשיעור הקודם, ישנן דרכים רבות לעבוד עם אשכולות, והשיטה שתבחרו תלויה בנתונים שלכם. ננסה את K-Means מכיוון שזו טכניקת האשכולות הנפוצה ביותר. בואו נתחיל!

מונחים שתלמדו עליהם:

- ציון סילואט
- שיטת המרפק
- אינרציה
- שונות

## מבוא

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) היא שיטה שמקורה בתחום עיבוד האותות. היא משמשת לחלוקה וקיבוץ קבוצות נתונים ל-'k' אשכולות באמצעות סדרת תצפיות. כל תצפית פועלת לקיבוץ נקודת נתונים נתונה הקרובה ביותר ל-'ממוצע' שלה, או לנקודת המרכז של האשכול.

ניתן להמחיש את האשכולות כ-[דיאגרמות וורונוי](https://wikipedia.org/wiki/Voronoi_diagram), הכוללות נקודה (או 'זרע') והאזור המתאים לה.

![דיאגרמת וורונוי](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> אינפוגרפיקה מאת [Jen Looper](https://twitter.com/jenlooper)

תהליך האשכולות של K-Means [מתבצע בשלושה שלבים](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. האלגוריתם בוחר מספר נקודות מרכזיות (k) על ידי דגימה מתוך סט הנתונים. לאחר מכן הוא מבצע לולאה:
    1. הוא מקצה כל דגימה לנקודת המרכז הקרובה ביותר.
    2. הוא יוצר נקודות מרכזיות חדשות על ידי חישוב הממוצע של כל הדגימות שהוקצו לנקודות המרכזיות הקודמות.
    3. לאחר מכן, הוא מחשב את ההבדל בין הנקודות המרכזיות החדשות והישנות וחוזר על התהליך עד שהנקודות המרכזיות מתייצבות.

חיסרון אחד בשימוש ב-K-Means הוא הצורך לקבוע את 'k', כלומר את מספר הנקודות המרכזיות. למרבה המזל, שיטת ה'מרפק' עוזרת להעריך ערך התחלתי טוב עבור 'k'. תנסו את זה עוד מעט.

## דרישות מקדימות

תעבדו בקובץ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) של השיעור הזה, הכולל את ייבוא הנתונים והניקוי הראשוני שביצעתם בשיעור הקודם.

## תרגיל - הכנה

נתחיל בהסתכלות נוספת על נתוני השירים.

1. צרו תרשים קופסה (boxplot) על ידי קריאה ל-`boxplot()` עבור כל עמודה:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    הנתונים האלה מעט רועשים: על ידי התבוננות בכל עמודה כתרשים קופסה, תוכלו לראות חריגות.

    ![חריגות](../../../../5-Clustering/2-K-Means/images/boxplots.png)

    תוכלו לעבור על סט הנתונים ולהסיר את החריגות הללו, אך זה יהפוך את הנתונים למינימליים למדי.

1. לעת עתה, בחרו אילו עמודות תשתמשו בהן לתרגיל האשכולות שלכם. בחרו עמודות עם טווחים דומים וקודדו את העמודה `artist_top_genre` כנתונים מספריים:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. עכשיו עליכם לבחור כמה אשכולות למקד. אתם יודעים שיש 3 ז'אנרים של שירים שזיהינו מתוך סט הנתונים, אז בואו ננסה 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

    אתם רואים מערך מודפס עם אשכולות חזויים (0, 1 או 2) עבור כל שורה של מסגרת הנתונים.

1. השתמשו במערך הזה כדי לחשב 'ציון סילואט':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## ציון סילואט

חפשו ציון סילואט קרוב ל-1. הציון הזה נע בין -1 ל-1, ואם הציון הוא 1, האשכול צפוף ומופרד היטב מאשכולות אחרים. ערך קרוב ל-0 מייצג אשכולות חופפים עם דגימות קרובות מאוד לגבול ההחלטה של האשכולות השכנים. [(מקור)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

הציון שלנו הוא **0.53**, כלומר באמצע. זה מצביע על כך שהנתונים שלנו לא מתאימים במיוחד לסוג זה של אשכולות, אבל בואו נמשיך.

### תרגיל - בניית מודל

1. ייבאו את `KMeans` והתחילו את תהליך האשכולות.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    יש כאן כמה חלקים שמצדיקים הסבר.

    > 🎓 טווח: אלו הן האיטרציות של תהליך האשכולות.

    > 🎓 random_state: "קובע את יצירת המספרים האקראיים עבור אתחול הנקודות המרכזיות." [מקור](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "סכום הריבועים בתוך האשכולות" מודד את המרחק הממוצע בריבוע של כל הנקודות בתוך אשכול לנקודת המרכז של האשכול. [מקור](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 אינרציה: אלגוריתמי K-Means מנסים לבחור נקודות מרכזיות כדי למזער את 'האינרציה', "מדד לכמה האשכולות קוהרנטיים פנימית." [מקור](https://scikit-learn.org/stable/modules/clustering.html). הערך נוסף למשתנה wcss בכל איטרציה.

    > 🎓 k-means++: ב-[Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) ניתן להשתמש באופטימיזציה 'k-means++', שמאתחלת את הנקודות המרכזיות כך שיהיו (בדרך כלל) רחוקות זו מזו, מה שמוביל לתוצאות טובות יותר מאתחול אקראי.

### שיטת המרפק

קודם לכן הסקתם, מכיוון שמיקדתם 3 ז'אנרים של שירים, שעליכם לבחור 3 אשכולות. אבל האם זה באמת המקרה?

1. השתמשו בשיטת ה'מרפק' כדי לוודא.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    השתמשו במשתנה `wcss` שבניתם בשלב הקודם כדי ליצור תרשים שמראה היכן ה'כיפוף' במרפק, שמצביע על מספר האשכולות האופטימלי. אולי זה **באמת** 3!

    ![שיטת המרפק](../../../../5-Clustering/2-K-Means/images/elbow.png)

## תרגיל - הצגת האשכולות

1. נסו את התהליך שוב, הפעם עם שלושה אשכולות, והציגו את האשכולות כתרשים פיזור:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. בדקו את דיוק המודל:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    דיוק המודל הזה לא טוב במיוחד, וצורת האשכולות נותנת לכם רמז מדוע.

    ![אשכולות](../../../../5-Clustering/2-K-Means/images/clusters.png)

    הנתונים האלה לא מאוזנים מספיק, לא מתואמים מספיק ויש יותר מדי שונות בין ערכי העמודות כדי ליצור אשכולות טובים. למעשה, האשכולות שנוצרים כנראה מושפעים או מוטים מאוד על ידי שלוש קטגוריות הז'אנרים שהגדרנו קודם. זה היה תהליך למידה!

    בתיעוד של Scikit-learn, תוכלו לראות שמודל כמו זה, עם אשכולות שאינם מוגדרים היטב, סובל מבעיה של 'שונות':

    ![מודלים בעייתיים](../../../../5-Clustering/2-K-Means/images/problems.png)
    > אינפוגרפיקה מתוך Scikit-learn

## שונות

שונות מוגדרת כ-"הממוצע של הריבועים של ההבדלים מהממוצע" [(מקור)](https://www.mathsisfun.com/data/standard-deviation.html). בהקשר של בעיית האשכולות הזו, היא מתייחסת לנתונים שבהם המספרים בסט הנתונים נוטים לסטות יותר מדי מהממוצע.

✅ זהו רגע מצוין לחשוב על כל הדרכים שבהן תוכלו לתקן את הבעיה הזו. לשפר את הנתונים עוד קצת? להשתמש בעמודות אחרות? להשתמש באלגוריתם אחר? רמז: נסו [לשנות את קנה המידה של הנתונים שלכם](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) כדי לנרמל אותם ולבדוק עמודות אחרות.

> נסו את '[מחשבון השונות](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' כדי להבין את המושג קצת יותר.

---

## 🚀אתגר

בלו זמן עם המחברת הזו, שנו פרמטרים. האם תוכלו לשפר את דיוק המודל על ידי ניקוי הנתונים יותר (למשל הסרת חריגות)? תוכלו להשתמש במשקלים כדי לתת משקל רב יותר לדגימות נתונים מסוימות. מה עוד תוכלו לעשות כדי ליצור אשכולות טובים יותר?

רמז: נסו לשנות את קנה המידה של הנתונים שלכם. יש קוד עם הערות במחברת שמוסיף שינוי קנה מידה סטנדרטי כדי לגרום לעמודות הנתונים להיראות דומות יותר זו לזו מבחינת טווח. תגלו שבעוד שציון הסילואט יורד, ה'כיפוף' בתרשים המרפק מתמתן. זאת מכיוון שהשארת הנתונים ללא שינוי קנה מידה מאפשרת לנתונים עם פחות שונות לשאת משקל רב יותר. קראו עוד על הבעיה הזו [כאן](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [מבחן מסכם](https://ff-quizzes.netlify.app/en/ml/)

## סקירה ולימוד עצמי

הסתכלו על סימולטור K-Means [כמו זה](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). תוכלו להשתמש בכלי הזה כדי להמחיש נקודות נתונים לדוגמה ולקבוע את הנקודות המרכזיות שלהן. תוכלו לערוך את אקראיות הנתונים, מספרי האשכולות ומספרי הנקודות המרכזיות. האם זה עוזר לכם לקבל מושג כיצד ניתן לקבץ את הנתונים?

בנוסף, הסתכלו על [המסמך הזה על K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) מסטנפורד.

## משימה

[נסו שיטות אשכולות שונות](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.