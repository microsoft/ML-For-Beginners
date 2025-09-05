<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T19:44:12+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "he"
}
-->
# בנה אפליקציית ווב לשימוש במודל ML שלך

בחלק זה של הקורס, תיחשף לנושא יישומי בתחום למידת מכונה: כיצד לשמור את המודל שלך שנבנה ב-Scikit-learn כקובץ שניתן להשתמש בו כדי לבצע תחזיות בתוך אפליקציית ווב. לאחר שהמודל נשמר, תלמד כיצד להשתמש בו באפליקציית ווב שנבנתה ב-Flask. תחילה תיצור מודל באמצעות נתונים העוסקים בתצפיות על עב"מים! לאחר מכן, תבנה אפליקציית ווב שתאפשר לך להזין מספר שניות יחד עם ערכי קו רוחב וקו אורך כדי לחזות באיזו מדינה דווח על עב"ם.

![חניית עב"מים](../../../3-Web-App/images/ufo.jpg)

צילום על ידי <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> ב-<a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## שיעורים

1. [בנה אפליקציית ווב](1-Web-App/README.md)

## קרדיטים

"בנה אפליקציית ווב" נכתב באהבה על ידי [Jen Looper](https://twitter.com/jenlooper).

♥️ החידונים נכתבו על ידי Rohan Raj.

המאגר נלקח מ-[Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

ארכיטקטורת אפליקציית הווב הוצעה בחלקה על ידי [המאמר הזה](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) ו-[הריפו הזה](https://github.com/abhinavsagar/machine-learning-deployment) מאת Abhinav Sagar.

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. אנו לא נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.