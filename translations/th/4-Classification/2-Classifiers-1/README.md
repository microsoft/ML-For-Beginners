<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T21:49:20+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "th"
}
-->
# ตัวจำแนกประเภทอาหาร 1

ในบทเรียนนี้ คุณจะใช้ชุดข้อมูลที่คุณบันทึกไว้จากบทเรียนก่อน ซึ่งเต็มไปด้วยข้อมูลที่สมดุลและสะอาดเกี่ยวกับอาหารหลากหลายประเภท

คุณจะใช้ชุดข้อมูลนี้ร่วมกับตัวจำแนกประเภทหลากหลายชนิดเพื่อ _ทำนายประเภทอาหารของชาติใดชาติหนึ่งจากกลุ่มของส่วนผสม_ ในขณะที่ทำเช่นนั้น คุณจะได้เรียนรู้เพิ่มเติมเกี่ยวกับวิธีการที่อัลกอริทึมสามารถนำมาใช้ในงานการจำแนกประเภท

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/)
# การเตรียมตัว

สมมติว่าคุณได้ทำ [บทเรียนที่ 1](../1-Introduction/README.md) เสร็จแล้ว ตรวจสอบให้แน่ใจว่าไฟล์ _cleaned_cuisines.csv_ มีอยู่ในโฟลเดอร์ `/data` หลักสำหรับบทเรียนทั้งสี่นี้

## แบบฝึกหัด - ทำนายประเภทอาหารของชาติ

1. ทำงานในโฟลเดอร์ _notebook.ipynb_ ของบทเรียนนี้ นำเข้าไฟล์นั้นพร้อมกับไลบรารี Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ข้อมูลจะมีลักษณะดังนี้:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ตอนนี้นำเข้าไลบรารีเพิ่มเติม:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. แบ่งพิกัด X และ y ออกเป็นสอง DataFrame สำหรับการฝึก `cuisine` สามารถเป็น DataFrame ของป้ายกำกับ:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    จะมีลักษณะดังนี้:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. ลบคอลัมน์ `Unnamed: 0` และคอลัมน์ `cuisine` โดยใช้ `drop()` และบันทึกข้อมูลที่เหลือเป็นคุณสมบัติที่สามารถฝึกได้:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    คุณสมบัติของคุณจะมีลักษณะดังนี้:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ตอนนี้คุณพร้อมที่จะฝึกโมเดลของคุณแล้ว!

## การเลือกตัวจำแนกประเภท

เมื่อข้อมูลของคุณสะอาดและพร้อมสำหรับการฝึก คุณต้องตัดสินใจว่าจะใช้อัลกอริทึมใดสำหรับงานนี้

Scikit-learn จัดกลุ่มการจำแนกประเภทไว้ในหมวดการเรียนรู้แบบมีผู้สอน และในหมวดหมู่นี้คุณจะพบวิธีการจำแนกประเภทมากมาย [ความหลากหลาย](https://scikit-learn.org/stable/supervised_learning.html) อาจดูน่าตกใจในตอนแรก วิธีการต่อไปนี้ล้วนมีเทคนิคการจำแนกประเภท:

- โมเดลเชิงเส้น
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- อัลกอริทึมแบบหลายคลาสและหลายผลลัพธ์ (multiclass และ multilabel classification, multiclass-multioutput classification)

> คุณสามารถใช้ [neural networks เพื่อจำแนกข้อมูล](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) ได้เช่นกัน แต่สิ่งนี้อยู่นอกขอบเขตของบทเรียนนี้

### จะเลือกตัวจำแนกประเภทใด?

แล้วคุณควรเลือกตัวจำแนกประเภทใด? บ่อยครั้ง การลองใช้หลายตัวและมองหาผลลัพธ์ที่ดีเป็นวิธีการทดสอบ Scikit-learn มี [การเปรียบเทียบแบบเคียงข้างกัน](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) บนชุดข้อมูลที่สร้างขึ้น โดยเปรียบเทียบ KNeighbors, SVC สองวิธี, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB และ QuadraticDiscrinationAnalysis พร้อมแสดงผลลัพธ์ในรูปแบบภาพ:

![การเปรียบเทียบตัวจำแนกประเภท](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> กราฟที่สร้างขึ้นจากเอกสารของ Scikit-learn

> AutoML แก้ปัญหานี้ได้อย่างง่ายดายโดยการรันการเปรียบเทียบเหล่านี้ในคลาวด์ ทำให้คุณสามารถเลือกอัลกอริทึมที่ดีที่สุดสำหรับข้อมูลของคุณ ลองใช้ [ที่นี่](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### วิธีที่ดีกว่า

วิธีที่ดีกว่าการเดาแบบสุ่มคือการปฏิบัติตามแนวคิดใน [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ที่สามารถดาวน์โหลดได้ ที่นี่เราพบว่า สำหรับปัญหาแบบหลายคลาส เรามีตัวเลือกบางอย่าง:

![cheatsheet สำหรับปัญหาแบบหลายคลาส](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> ส่วนหนึ่งของ Algorithm Cheat Sheet ของ Microsoft ที่แสดงตัวเลือกการจำแนกประเภทแบบหลายคลาส

✅ ดาวน์โหลด cheat sheet นี้ พิมพ์ออกมา และติดไว้บนผนังของคุณ!

### การใช้เหตุผล

ลองดูว่าเราสามารถใช้เหตุผลเพื่อเลือกวิธีการต่าง ๆ ได้หรือไม่ โดยพิจารณาจากข้อจำกัดที่เรามี:

- **Neural networks หนักเกินไป** เนื่องจากชุดข้อมูลของเราสะอาดแต่มีขนาดเล็ก และเรากำลังฝึกโมเดลในเครื่องผ่านโน้ตบุ๊ก Neural networks จึงหนักเกินไปสำหรับงานนี้
- **ไม่ใช้ตัวจำแนกประเภทแบบสองคลาส** เราไม่ใช้ตัวจำแนกประเภทแบบสองคลาส ดังนั้นจึงไม่ใช้ one-vs-all
- **Decision tree หรือ logistic regression อาจใช้ได้** Decision tree อาจใช้ได้ หรือ logistic regression สำหรับข้อมูลแบบหลายคลาส
- **Multiclass Boosted Decision Trees แก้ปัญหาอื่น** Multiclass Boosted Decision Tree เหมาะสมที่สุดสำหรับงานที่ไม่ใช่พารามิเตอร์ เช่น งานที่ออกแบบมาเพื่อสร้างการจัดอันดับ ดังนั้นจึงไม่เหมาะสำหรับเรา

### การใช้ Scikit-learn 

เราจะใช้ Scikit-learn เพื่อวิเคราะห์ข้อมูลของเรา อย่างไรก็ตาม มีหลายวิธีในการใช้ logistic regression ใน Scikit-learn ลองดู [พารามิเตอร์ที่ต้องส่ง](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)  

โดยพื้นฐานแล้วมีสองพารามิเตอร์สำคัญ - `multi_class` และ `solver` - ที่เราต้องระบุเมื่อเราขอให้ Scikit-learn ทำ logistic regression ค่า `multi_class` จะกำหนดพฤติกรรมบางอย่าง ส่วนค่าของ solver คืออัลกอริทึมที่จะใช้ ไม่ใช่ทุก solver ที่สามารถจับคู่กับค่าของ `multi_class` ได้

ตามเอกสาร ในกรณีแบบหลายคลาส อัลกอริทึมการฝึก:

- **ใช้ one-vs-rest (OvR)** หากตัวเลือก `multi_class` ถูกตั้งค่าเป็น `ovr`
- **ใช้ cross-entropy loss** หากตัวเลือก `multi_class` ถูกตั้งค่าเป็น `multinomial` (ปัจจุบันตัวเลือก `multinomial` รองรับเฉพาะ solver ‘lbfgs’, ‘sag’, ‘saga’ และ ‘newton-cg’)

> 🎓 'scheme' ในที่นี้สามารถเป็น 'ovr' (one-vs-rest) หรือ 'multinomial' เนื่องจาก logistic regression ถูกออกแบบมาเพื่อรองรับการจำแนกประเภทแบบไบนารี scheme เหล่านี้ช่วยให้มันจัดการงานการจำแนกประเภทแบบหลายคลาสได้ดีขึ้น [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' ถูกกำหนดว่าเป็น "อัลกอริทึมที่จะใช้ในปัญหาการเพิ่มประสิทธิภาพ" [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn มีตารางนี้เพื่ออธิบายว่า solver จัดการกับความท้าทายต่าง ๆ ที่เกิดจากโครงสร้างข้อมูลประเภทต่าง ๆ อย่างไร:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## แบบฝึกหัด - แบ่งข้อมูล

เราสามารถมุ่งเน้นไปที่ logistic regression สำหรับการทดลองฝึกครั้งแรกของเรา เนื่องจากคุณเพิ่งเรียนรู้เกี่ยวกับเรื่องนี้ในบทเรียนก่อนหน้า
แบ่งข้อมูลของคุณออกเป็นกลุ่มการฝึกและการทดสอบโดยเรียกใช้ `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## แบบฝึกหัด - ใช้ logistic regression

เนื่องจากคุณกำลังใช้กรณีแบบหลายคลาส คุณจำเป็นต้องเลือกว่าจะใช้ _scheme_ ใดและตั้งค่า _solver_ อย่างไร ใช้ LogisticRegression พร้อมการตั้งค่าแบบหลายคลาสและ solver **liblinear** เพื่อฝึก

1. สร้าง logistic regression โดยตั้งค่า multi_class เป็น `ovr` และ solver เป็น `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ ลองใช้ solver อื่น เช่น `lbfgs` ซึ่งมักถูกตั้งค่าเป็นค่าเริ่มต้น
> หมายเหตุ ใช้ฟังก์ชัน Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) เพื่อแปลงข้อมูลของคุณให้อยู่ในรูปแบบแบนเมื่อจำเป็น
ความแม่นยำดีเกินกว่า **80%**!

1. คุณสามารถดูการทำงานของโมเดลนี้ได้โดยการทดสอบข้อมูลหนึ่งแถว (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ผลลัพธ์จะถูกพิมพ์ออกมา:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ ลองใช้หมายเลขแถวอื่นและตรวจสอบผลลัพธ์

1. เจาะลึกลงไปอีก คุณสามารถตรวจสอบความแม่นยำของการทำนายนี้ได้:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ผลลัพธ์จะถูกพิมพ์ออกมา - อาหารอินเดียเป็นการคาดเดาที่ดีที่สุด พร้อมความน่าจะเป็นที่ดี:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ คุณสามารถอธิบายได้ไหมว่าทำไมโมเดลถึงมั่นใจว่านี่คืออาหารอินเดีย?

1. ดูรายละเอียดเพิ่มเติมโดยการพิมพ์รายงานการจำแนกประเภท เช่นเดียวกับที่คุณทำในบทเรียนการถดถอย:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀ความท้าทาย

ในบทเรียนนี้ คุณใช้ข้อมูลที่ทำความสะอาดแล้วเพื่อสร้างโมเดลการเรียนรู้ของเครื่องที่สามารถทำนายอาหารประจำชาติได้จากชุดของส่วนผสม ใช้เวลาสักครู่เพื่ออ่านตัวเลือกมากมายที่ Scikit-learn มีให้สำหรับการจำแนกข้อมูล เจาะลึกลงไปในแนวคิดของ 'solver' เพื่อทำความเข้าใจสิ่งที่เกิดขึ้นเบื้องหลัง

## [แบบทดสอบหลังบทเรียน](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

เจาะลึกลงไปอีกในคณิตศาสตร์เบื้องหลังการถดถอยโลจิสติกใน [บทเรียนนี้](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## งานที่ได้รับมอบหมาย 

[ศึกษาตัวแก้ปัญหา](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ แนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้