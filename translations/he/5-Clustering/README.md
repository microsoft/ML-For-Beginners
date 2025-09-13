<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T19:10:12+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "he"
}
-->
# מודלים של אשכולות ללמידת מכונה

אשכולות הם משימה בלמידת מכונה שבה מחפשים למצוא אובייקטים הדומים זה לזה ולחבר אותם לקבוצות הנקראות אשכולות. מה שמבדיל אשכולות מגישות אחרות בלמידת מכונה הוא שהדברים מתרחשים באופן אוטומטי, למעשה, אפשר לומר שזה ההפך מלמידה מונחית.

## נושא אזורי: מודלים של אשכולות לטעמי מוזיקה של קהל ניגרי 🎧

הקהל המגוון בניגריה מתאפיין בטעמי מוזיקה מגוונים. באמצעות נתונים שנאספו מ-Spotify (בהשראת [המאמר הזה](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), נבחן כמה מהמוזיקה הפופולרית בניגריה. מערך הנתונים הזה כולל מידע על ציוני 'ריקודיות', 'אקוסטיות', עוצמת קול, 'דיבוריות', פופולריות ואנרגיה של שירים שונים. יהיה מעניין לגלות דפוסים בנתונים האלה!

![פלטת תקליטים](../../../5-Clustering/images/turntable.jpg)

> צילום מאת <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> ב-<a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
בסדרת השיעורים הזו, תגלו דרכים חדשות לנתח נתונים באמצעות טכניקות אשכולות. אשכולות שימושיים במיוחד כאשר מערך הנתונים שלכם חסר תוויות. אם יש לו תוויות, אז טכניקות סיווג כמו אלו שלמדתם בשיעורים קודמים עשויות להיות מועילות יותר. אבל במקרים שבהם אתם מחפשים לקבץ נתונים ללא תוויות, אשכולות הם דרך מצוינת לגלות דפוסים.

> ישנם כלים שימושיים בעלי קוד נמוך שיכולים לעזור לכם ללמוד לעבוד עם מודלים של אשכולות. נסו [Azure ML למשימה זו](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## שיעורים

1. [מבוא לאשכולות](1-Visualize/README.md)
2. [אשכולות K-Means](2-K-Means/README.md)

## קרדיטים

השיעורים הללו נכתבו עם 🎶 על ידי [Jen Looper](https://www.twitter.com/jenlooper) עם ביקורות מועילות מאת [Rishit Dagli](https://rishit_dagli) ו-[Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

מערך הנתונים [שירים ניגריים](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) נלקח מ-Kaggle ונאסף מ-Spotify.

דוגמאות שימושיות של K-Means שסייעו ביצירת השיעור כוללות את [חקירת האיריס הזו](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [מחברת מבוא זו](https://www.kaggle.com/prashant111/k-means-clustering-with-python), ואת [דוגמת ה-NGO ההיפותטית הזו](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.