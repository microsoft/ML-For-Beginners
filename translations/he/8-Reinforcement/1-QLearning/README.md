<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T20:12:27+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "he"
}
-->
# מבוא ללמידת חיזוק ולמידת Q

![סיכום למידת חיזוק בלמידת מכונה בסקצ'נוט](../../../../sketchnotes/ml-reinforcement.png)
> סקצ'נוט מאת [Tomomi Imura](https://www.twitter.com/girlie_mac)

למידת חיזוק כוללת שלושה מושגים חשובים: הסוכן, מצבים מסוימים, ומערך פעולות לכל מצב. על ידי ביצוע פעולה במצב מסוים, הסוכן מקבל תגמול. דמיינו שוב את משחק המחשב סופר מריו. אתם מריו, נמצאים ברמת משחק, עומדים ליד קצה צוק. מעליכם יש מטבע. אתם, בתור מריו, ברמת משחק, במיקום ספציפי... זהו המצב שלכם. צעד אחד ימינה (פעולה) יוביל אתכם מעבר לקצה, וזה יעניק לכם ניקוד נמוך. לעומת זאת, לחיצה על כפתור הקפיצה תאפשר לכם לצבור נקודה ולהישאר בחיים. זהו תוצאה חיובית שצריכה להעניק לכם ניקוד חיובי.

באמצעות למידת חיזוק וסימולטור (המשחק), ניתן ללמוד כיצד לשחק את המשחק כדי למקסם את התגמול, כלומר להישאר בחיים ולצבור כמה שיותר נקודות.

[![מבוא ללמידת חיזוק](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 לחצו על התמונה למעלה כדי לשמוע את דמיטרי מדבר על למידת חיזוק

## [שאלון לפני השיעור](https://ff-quizzes.netlify.app/en/ml/)

## דרישות מוקדמות והגדרות

בשיעור זה, נתנסה בקוד בפייתון. עליכם להיות מסוגלים להריץ את הקוד של Jupyter Notebook מהשיעור הזה, בין אם במחשב שלכם או בענן.

ניתן לפתוח את [מחברת השיעור](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) ולעבור על השיעור כדי לבנות.

> **הערה:** אם אתם פותחים את הקוד מהענן, תצטרכו גם להוריד את הקובץ [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), שמשמש בקוד המחברת. הוסיפו אותו לאותה תיקייה כמו המחברת.

## מבוא

בשיעור זה, נחקור את עולמו של **[פטר והזאב](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, בהשראת אגדה מוזיקלית של המלחין הרוסי [סרגיי פרוקופייב](https://en.wikipedia.org/wiki/Sergei_Prokofiev). נשתמש ב**למידת חיזוק** כדי לאפשר לפטר לחקור את סביבתו, לאסוף תפוחים טעימים ולהימנע ממפגש עם הזאב.

**למידת חיזוק** (RL) היא טכניקת למידה שמאפשרת לנו ללמוד התנהגות אופטימלית של **סוכן** בסביבה מסוימת על ידי ביצוע ניסויים רבים. סוכן בסביבה זו צריך שיהיה לו **מטרה**, שמוגדרת על ידי **פונקציית תגמול**.

## הסביבה

לצורך הפשטות, נניח שעולמו של פטר הוא לוח מרובע בגודל `width` x `height`, כמו זה:

![הסביבה של פטר](../../../../8-Reinforcement/1-QLearning/images/environment.png)

כל תא בלוח הזה יכול להיות:

* **קרקע**, שעליה פטר ויצורים אחרים יכולים ללכת.
* **מים**, שעליהם כמובן אי אפשר ללכת.
* **עץ** או **דשא**, מקום שבו אפשר לנוח.
* **תפוח**, שמייצג משהו שפטר ישמח למצוא כדי להאכיל את עצמו.
* **זאב**, שהוא מסוכן ויש להימנע ממנו.

ישנו מודול פייתון נפרד, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), שמכיל את הקוד לעבודה עם הסביבה הזו. מכיוון שהקוד הזה אינו חשוב להבנת המושגים שלנו, נייבא את המודול ונשתמש בו כדי ליצור את הלוח לדוגמה (בלוק קוד 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

הקוד הזה אמור להדפיס תמונה של הסביבה הדומה לזו שמוצגת למעלה.

## פעולות ומדיניות

בדוגמה שלנו, המטרה של פטר תהיה למצוא תפוח, תוך הימנעות מהזאב ומכשולים אחרים. לשם כך, הוא יכול למעשה להסתובב עד שימצא תפוח.

לכן, בכל מיקום, הוא יכול לבחור בין אחת מהפעולות הבאות: למעלה, למטה, שמאלה וימינה.

נגדיר את הפעולות הללו כמילון, ונמפה אותן לזוגות של שינויי קואורדינטות מתאימים. לדוגמה, תנועה ימינה (`R`) תתאים לזוג `(1,0)`. (בלוק קוד 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

לסיכום, האסטרטגיה והמטרה של התרחיש הזה הם כדלקמן:

- **האסטרטגיה**, של הסוכן שלנו (פטר) מוגדרת על ידי מה שנקרא **מדיניות**. מדיניות היא פונקציה שמחזירה את הפעולה בכל מצב נתון. במקרה שלנו, מצב הבעיה מיוצג על ידי הלוח, כולל המיקום הנוכחי של השחקן.

- **המטרה**, של למידת החיזוק היא בסופו של דבר ללמוד מדיניות טובה שתאפשר לנו לפתור את הבעיה ביעילות. עם זאת, כבסיס, נשקול את המדיניות הפשוטה ביותר שנקראת **הליכה אקראית**.

## הליכה אקראית

בואו נפתור את הבעיה שלנו תחילה על ידי יישום אסטרטגיית הליכה אקראית. עם הליכה אקראית, נבחר באופן אקראי את הפעולה הבאה מתוך הפעולות המותרות, עד שנגיע לתפוח (בלוק קוד 3).

1. יישמו את ההליכה האקראית עם הקוד הבא:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    הקריאה ל-`walk` אמורה להחזיר את אורך המסלול המתאים, שיכול להשתנות מריצה אחת לאחרת.

1. הריצו את ניסוי ההליכה מספר פעמים (נניח, 100), והדפיסו את הסטטיסטיקות המתקבלות (בלוק קוד 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    שימו לב שאורך המסלול הממוצע הוא סביב 30-40 צעדים, שזה די הרבה, בהתחשב בכך שהמרחק הממוצע לתפוח הקרוב ביותר הוא סביב 5-6 צעדים.

    תוכלו גם לראות כיצד נראית תנועתו של פטר במהלך ההליכה האקראית:

    ![הליכה אקראית של פטר](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## פונקציית תגמול

כדי להפוך את המדיניות שלנו לאינטליגנטית יותר, עלינו להבין אילו מהלכים הם "טובים" יותר מאחרים. לשם כך, עלינו להגדיר את המטרה שלנו.

המטרה יכולה להיות מוגדרת במונחים של **פונקציית תגמול**, שתחזיר ערך ניקוד עבור כל מצב. ככל שהמספר גבוה יותר, כך פונקציית התגמול טובה יותר. (בלוק קוד 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

דבר מעניין לגבי פונקציות תגמול הוא שבמקרים רבים, *אנו מקבלים תגמול משמעותי רק בסוף המשחק*. משמעות הדבר היא שהאלגוריתם שלנו צריך לזכור "צעדים טובים" שהובילו לתגמול חיובי בסוף, ולהגדיל את חשיבותם. באופן דומה, כל המהלכים שמובילים לתוצאות רעות צריכים להיות מדוכאים.

## למידת Q

האלגוריתם שנדון בו כאן נקרא **למידת Q**. באלגוריתם זה, המדיניות מוגדרת על ידי פונקציה (או מבנה נתונים) שנקראת **טבלת Q**. היא מתעדת את "הטוב" של כל אחת מהפעולות במצב נתון.

היא נקראת טבלת Q מכיוון שלעתים קרובות נוח לייצג אותה כטבלה, או מערך רב-ממדי. מכיוון שלוח המשחק שלנו הוא בגודל `width` x `height`, נוכל לייצג את טבלת Q באמצעות מערך numpy עם צורה `width` x `height` x `len(actions)`: (בלוק קוד 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

שימו לב שאנו מאתחלים את כל הערכים בטבלת Q עם ערך שווה, במקרה שלנו - 0.25. זה תואם למדיניות "הליכה אקראית", מכיוון שכל המהלכים בכל מצב הם טובים באותה מידה. נוכל להעביר את טבלת Q לפונקציית `plot` כדי להמחיש את הטבלה על הלוח: `m.plot(Q)`.

![הסביבה של פטר](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

במרכז כל תא יש "חץ" שמצביע על כיוון התנועה המועדף. מכיוון שכל הכיוונים שווים, מוצגת נקודה.

כעת עלינו להריץ את הסימולציה, לחקור את הסביבה שלנו, וללמוד חלוקה טובה יותר של ערכי טבלת Q, שתאפשר לנו למצוא את הדרך לתפוח הרבה יותר מהר.

## מהות למידת Q: משוואת בלמן

ברגע שנתחיל לזוז, לכל פעולה יהיה תגמול מתאים, כלומר נוכל באופן תיאורטי לבחור את הפעולה הבאה על סמך התגמול המיידי הגבוה ביותר. עם זאת, ברוב המצבים, המהלך לא ישיג את מטרתנו להגיע לתפוח, ולכן לא נוכל להחליט מיד איזה כיוון טוב יותר.

> זכרו שזה לא התוצאה המיידית שחשובה, אלא התוצאה הסופית, שנקבל בסוף הסימולציה.

כדי לקחת בחשבון את התגמול המושהה, עלינו להשתמש בעקרונות של **[תכנות דינמי](https://en.wikipedia.org/wiki/Dynamic_programming)**, שמאפשרים לנו לחשוב על הבעיה שלנו באופן רקורסיבי.

נניח שאנחנו נמצאים כעת במצב *s*, ורוצים לעבור למצב הבא *s'*. על ידי כך, נקבל את התגמול המיידי *r(s,a)*, שמוגדר על ידי פונקציית התגמול, בתוספת תגמול עתידי כלשהו. אם נניח שטבלת Q שלנו משקפת נכון את "האטרקטיביות" של כל פעולה, אז במצב *s'* נבחר פעולה *a* שתואמת לערך המקסימלי של *Q(s',a')*. לכן, התגמול העתידי הטוב ביותר שנוכל לקבל במצב *s* יוגדר כ-`max`

## בדיקת המדיניות

מכיוון ש-Q-Table מציג את "האטרקטיביות" של כל פעולה בכל מצב, קל מאוד להשתמש בו כדי להגדיר ניווט יעיל בעולם שלנו. במקרה הפשוט ביותר, ניתן לבחור את הפעולה המתאימה לערך הגבוה ביותר ב-Q-Table: (בלוק קוד 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> אם תנסו את הקוד למעלה מספר פעמים, ייתכן שתשימו לב שלפעמים הוא "נתקע", ותצטרכו ללחוץ על כפתור ה-STOP במחברת כדי להפסיק אותו. זה קורה מכיוון שיכולות להיות מצבים שבהם שני מצבים "מצביעים" זה על זה מבחינת ערך Q אופטימלי, ובמקרה כזה הסוכן ימשיך לנוע בין אותם מצבים ללא סוף.

## 🚀אתגר

> **משימה 1:** שנו את הפונקציה `walk` כך שתוגבל אורך המסלול המרבי למספר מסוים של צעדים (לדוגמה, 100), וצפו בקוד למעלה מחזיר את הערך הזה מדי פעם.

> **משימה 2:** שנו את הפונקציה `walk` כך שלא תחזור למקומות שבהם כבר הייתה בעבר. זה ימנע מ-`walk` להיכנס ללולאה, אך עדיין ייתכן שהסוכן ימצא את עצמו "תקוע" במקום שממנו אינו יכול לברוח.

## ניווט

מדיניות ניווט טובה יותר תהיה זו שהשתמשנו בה במהלך האימון, שמשלבת ניצול וחקר. במדיניות זו, נבחר כל פעולה עם הסתברות מסוימת, פרופורציונלית לערכים ב-Q-Table. אסטרטגיה זו עדיין עשויה לגרום לסוכן לחזור למיקום שכבר חקר, אך כפי שניתן לראות מהקוד למטה, היא מביאה למסלול ממוצע קצר מאוד למיקום הרצוי (זכרו ש-`print_statistics` מריץ את הסימולציה 100 פעמים): (בלוק קוד 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

לאחר הרצת הקוד הזה, אתם אמורים לקבל אורך מסלול ממוצע קטן בהרבה מאשר קודם, בטווח של 3-6.

## חקירת תהליך הלמידה

כפי שציינו, תהליך הלמידה הוא איזון בין חקר לבין ניצול הידע שנצבר על מבנה מרחב הבעיה. ראינו שהתוצאות של הלמידה (היכולת לעזור לסוכן למצוא מסלול קצר למטרה) השתפרו, אך גם מעניין לצפות כיצד אורך המסלול הממוצע מתנהג במהלך תהליך הלמידה:

## סיכום הלמידות:

- **אורך המסלול הממוצע עולה**. מה שאנו רואים כאן הוא שבתחילה, אורך המסלול הממוצע עולה. זה כנראה נובע מכך שכאשר איננו יודעים דבר על הסביבה, אנו נוטים להיתקע במצבים גרועים, כמו מים או זאב. ככל שאנו לומדים יותר ומתחילים להשתמש בידע הזה, אנו יכולים לחקור את הסביבה לזמן ארוך יותר, אך עדיין איננו יודעים היטב היכן נמצאים התפוחים.

- **אורך המסלול יורד ככל שאנו לומדים יותר**. ברגע שאנו לומדים מספיק, קל יותר לסוכן להשיג את המטרה, ואורך המסלול מתחיל לרדת. עם זאת, אנו עדיין פתוחים לחקר, ולכן לעיתים קרובות אנו סוטים מהמסלול הטוב ביותר ובוחנים אפשרויות חדשות, מה שגורם למסלול להיות ארוך יותר מהאופטימלי.

- **אורך המסלול עולה באופן פתאומי**. מה שאנו גם רואים בגרף הוא שבשלב מסוים, האורך עלה באופן פתאומי. זה מצביע על האופי הסטוכסטי של התהליך, ועל כך שבשלב מסוים אנו יכולים "לקלקל" את מקדמי ה-Q-Table על ידי החלפתם בערכים חדשים. באופן אידיאלי, יש למזער זאת על ידי הפחתת קצב הלמידה (לדוגמה, לקראת סוף האימון, אנו משנים את ערכי ה-Q-Table רק במידה קטנה).

בסך הכל, חשוב לזכור שההצלחה והאיכות של תהליך הלמידה תלויים באופן משמעותי בפרמטרים, כמו קצב הלמידה, דעיכת קצב הלמידה, ופקטור ההנחה. אלו נקראים לעיתים קרובות **היפר-פרמטרים**, כדי להבדילם מ-**פרמטרים**, שאותם אנו ממטבים במהלך האימון (לדוגמה, מקדמי Q-Table). תהליך מציאת הערכים הטובים ביותר להיפר-פרמטרים נקרא **אופטימיזציה של היפר-פרמטרים**, והוא ראוי לנושא נפרד.

## [שאלון לאחר ההרצאה](https://ff-quizzes.netlify.app/en/ml/)

## משימה 
[עולם מציאותי יותר](assignment.md)

---

**כתב ויתור**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס בינה מלאכותית [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור סמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. איננו נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.