# מודלים של אשכולות ללמידת מכונה

אשכולות היא משימת למידת מכונה שבה מנסים למצוא אובייקטים הדומים זה לזה ולקבץ אותם לקבוצות הנקראות אשכולות. מה שמבדיל את האשכולות מגישות אחרות בלמידת מכונה הוא שהדברים קורים אוטומטית, למעשה, אפשר לומר שזה ההפך מלמידה מונחית.

## נושא אזורי: מודלי אשכולות לטעמי מוזיקה של קהל נגרי 🎧

לקהל המגוון של ניגריה יש טעמי מוזיקה מגוונים. באמצעות נתונים שנאספו מ-Spotify (בהשראת [מאמר זה](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), נבחן מוזיקה פופולרית בניגריה. מערך נתונים זה כולל מידע על ציון 'danceability', 'acousticness', רמת עוצמת הקול, 'speechiness', פופולריות ואנרגיה של שירים שונים. יהיה מעניין לגלות דפוסים בנתונים אלו!

![מערכת טורנーブル](../../../translated_images/he/turntable.f2b86b13c53302dc.webp)

> תמונה מאת <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">מרסלה לאסקוסקי</a> ב-<a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
בסדרת שיעורים זו תגלו דרכים חדשות לנתח נתונים באמצעות טכניקות אשכולות. אשכולות היא שימושית במיוחד כשהמערך שלך חסר תוויות. אם יש לו תוויות, טכניקות סיווג כמו שלמדת בשיעורים קודמים עשויות להיות מועילות יותר. אבל במקרים בהם אתה מחפש לקבץ נתונים ללא תוויות, אשכולות היא דרך מצוינת לגלות דפוסים.

> יש כלים מועילים עם מעט קוד שיכולים לעזור לך ללמוד על עבודה עם מודלי אשכולות. נסה את [Azure ML עבור המשימה הזו](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## שיעורים

1. [מבוא לאשכולות](1-Visualize/README.md)
2. [אשכולות K-Means](2-K-Means/README.md)

## קרדיטים

שיעורים אלה נכתבו עם 🎶 על ידי [ג'ן לופר](https://www.twitter.com/jenlooper) עם סקירות מועילות של [רישיט דגלי](https://rishit_dagli/) ו-[מוחמד סאקיב חאן אינאן](https://twitter.com/Sakibinan).

מערך הנתונים של [שירי ניגריה](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) נוצר מ-Kaggle תוך גרידה מ-Spotify.

דוגמאות שימושיות של K-Means שעזרו ביצירת שיעור זה כוללות את [חקירת האירוס](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), את [מחברת מבוא זו](https://www.kaggle.com/prashant111/k-means-clustering-with-python), ואת [דוגמת NGO היפותטית זו](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**כתב ויתור**:
מסמך זה תורגם באמצעות שירות תרגום אוטומטי [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עלולים להכיל שגיאות או אי-דיוקים. יש להחשיב את המסמך המקורי בשפתו הטבעית כמקור הסמכות. למידע קריטי מומלץ להשתמש בתרגום מקצועי על ידי מתרגם אדם. אנו לא אחראים לכל אי-הבנה או פירוש שגוי הנובע מהשימוש בתרגום זה.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->