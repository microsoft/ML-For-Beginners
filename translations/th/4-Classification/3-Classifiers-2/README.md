<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T21:53:19+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "th"
}
-->
# ตัวจำแนกประเภทอาหาร 2

ในบทเรียนการจำแนกประเภทครั้งที่สองนี้ คุณจะได้สำรวจวิธีการเพิ่มเติมในการจำแนกข้อมูลเชิงตัวเลข นอกจากนี้คุณยังจะได้เรียนรู้ผลกระทบจากการเลือกตัวจำแนกประเภทหนึ่งเหนืออีกประเภทหนึ่ง

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/)

### ความรู้พื้นฐานที่ต้องมี

เราสมมติว่าคุณได้เรียนบทเรียนก่อนหน้านี้แล้ว และมีชุดข้อมูลที่ทำความสะอาดแล้วในโฟลเดอร์ `data` ซึ่งมีชื่อว่า _cleaned_cuisines.csv_ อยู่ในโฟลเดอร์รากของบทเรียน 4 บทนี้

### การเตรียมตัว

เราได้โหลดไฟล์ _notebook.ipynb_ ของคุณพร้อมกับชุดข้อมูลที่ทำความสะอาดแล้ว และได้แบ่งข้อมูลออกเป็น dataframe X และ y ซึ่งพร้อมสำหรับกระบวนการสร้างโมเดล

## แผนที่การจำแนกประเภท

ก่อนหน้านี้ คุณได้เรียนรู้เกี่ยวกับตัวเลือกต่าง ๆ ที่คุณมีเมื่อจำแนกข้อมูลโดยใช้แผ่นโกงของ Microsoft Scikit-learn มีแผ่นโกงที่คล้ายกันแต่ละเอียดกว่า ซึ่งสามารถช่วยจำกัดตัวเลือกของคุณให้แคบลงได้ (อีกคำหนึ่งสำหรับตัวจำแนกประเภท):

![แผนที่ ML จาก Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> เคล็ดลับ: [เยี่ยมชมแผนที่นี้ออนไลน์](https://scikit-learn.org/stable/tutorial/machine_learning_map/) และคลิกตามเส้นทางเพื่ออ่านเอกสารประกอบ

### แผนการดำเนินการ

แผนที่นี้มีประโยชน์มากเมื่อคุณเข้าใจข้อมูลของคุณอย่างชัดเจน เพราะคุณสามารถ 'เดิน' ตามเส้นทางเพื่อไปสู่การตัดสินใจ:

- เรามีตัวอย่าง >50 ตัวอย่าง
- เราต้องการทำนายหมวดหมู่
- เรามีข้อมูลที่มีป้ายกำกับ
- เรามีตัวอย่างน้อยกว่า 100K ตัวอย่าง
- ✨ เราสามารถเลือก Linear SVC
- หากไม่ได้ผล เนื่องจากเรามีข้อมูลเชิงตัวเลข
    - เราสามารถลองใช้ ✨ KNeighbors Classifier 
      - หากไม่ได้ผล ลองใช้ ✨ SVC และ ✨ Ensemble Classifiers

นี่เป็นเส้นทางที่มีประโยชน์มากในการติดตาม

## แบบฝึกหัด - แบ่งข้อมูล

ตามเส้นทางนี้ เราควรเริ่มต้นด้วยการนำเข้าบางไลบรารีที่จำเป็นต้องใช้

1. นำเข้าไลบรารีที่จำเป็น:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. แบ่งข้อมูลการฝึกอบรมและการทดสอบของคุณ:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## ตัวจำแนก Linear SVC

Support-Vector clustering (SVC) เป็นส่วนหนึ่งของกลุ่มเทคนิค ML ในตระกูล Support-Vector machines (เรียนรู้เพิ่มเติมเกี่ยวกับสิ่งเหล่านี้ด้านล่าง) ในวิธีนี้ คุณสามารถเลือก 'kernel' เพื่อกำหนดวิธีการจัดกลุ่มป้ายกำกับ พารามิเตอร์ 'C' หมายถึง 'regularization' ซึ่งควบคุมอิทธิพลของพารามิเตอร์ Kernel สามารถเป็นหนึ่งใน [หลายตัวเลือก](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); ที่นี่เราตั้งค่าเป็น 'linear' เพื่อให้แน่ใจว่าเราใช้ Linear SVC ค่า Probability จะตั้งค่าเริ่มต้นเป็น 'false'; ที่นี่เราตั้งค่าเป็น 'true' เพื่อรวบรวมการประมาณความน่าจะเป็น เราตั้งค่า random state เป็น '0' เพื่อสับเปลี่ยนข้อมูลเพื่อรับความน่าจะเป็น

### แบบฝึกหัด - ใช้ Linear SVC

เริ่มต้นด้วยการสร้างอาร์เรย์ของตัวจำแนกประเภท คุณจะเพิ่มไปยังอาร์เรย์นี้ทีละขั้นตอนเมื่อเราทดสอบ

1. เริ่มต้นด้วย Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. ฝึกโมเดลของคุณโดยใช้ Linear SVC และพิมพ์รายงานออกมา:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    ผลลัพธ์ค่อนข้างดี:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## ตัวจำแนก K-Neighbors

K-Neighbors เป็นส่วนหนึ่งของกลุ่มวิธี ML ในตระกูล "neighbors" ซึ่งสามารถใช้ได้ทั้งการเรียนรู้แบบมีผู้สอนและไม่มีผู้สอน ในวิธีนี้ จะมีการสร้างจุดที่กำหนดไว้ล่วงหน้า และรวบรวมข้อมูลรอบ ๆ จุดเหล่านี้เพื่อให้สามารถทำนายป้ายกำกับทั่วไปสำหรับข้อมูลได้

### แบบฝึกหัด - ใช้ตัวจำแนก K-Neighbors

ตัวจำแนกประเภทก่อนหน้านี้ดีและทำงานได้ดีกับข้อมูล แต่บางทีเราอาจได้ความแม่นยำที่ดีกว่า ลองใช้ตัวจำแนก K-Neighbors

1. เพิ่มบรรทัดในอาร์เรย์ตัวจำแนกประเภทของคุณ (เพิ่มเครื่องหมายจุลภาคหลังรายการ Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    ผลลัพธ์แย่ลงเล็กน้อย:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ เรียนรู้เกี่ยวกับ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## ตัวจำแนก Support Vector

ตัวจำแนก Support-Vector เป็นส่วนหนึ่งของกลุ่ม [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) ในวิธี ML ที่ใช้สำหรับงานการจำแนกประเภทและการถดถอย SVMs "แมปตัวอย่างการฝึกอบรมไปยังจุดในพื้นที่" เพื่อเพิ่มระยะห่างระหว่างสองหมวดหมู่ ข้อมูลที่ตามมาจะถูกแมปเข้าสู่พื้นที่นี้เพื่อให้สามารถทำนายหมวดหมู่ของข้อมูลได้

### แบบฝึกหัด - ใช้ตัวจำแนก Support Vector

ลองหาความแม่นยำที่ดีกว่าด้วยตัวจำแนก Support Vector

1. เพิ่มเครื่องหมายจุลภาคหลังรายการ K-Neighbors และเพิ่มบรรทัดนี้:

    ```python
    'SVC': SVC(),
    ```

    ผลลัพธ์ค่อนข้างดี!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ เรียนรู้เกี่ยวกับ [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## ตัวจำแนก Ensemble

ลองติดตามเส้นทางจนถึงจุดสิ้นสุด แม้ว่าการทดสอบก่อนหน้านี้จะค่อนข้างดี ลองใช้ 'Ensemble Classifiers' โดยเฉพาะ Random Forest และ AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

ผลลัพธ์ดีมาก โดยเฉพาะสำหรับ Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ เรียนรู้เกี่ยวกับ [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

วิธีการ Machine Learning นี้ "รวมการทำนายของตัวประมาณฐานหลายตัว" เพื่อปรับปรุงคุณภาพของโมเดล ในตัวอย่างของเรา เราใช้ Random Trees และ AdaBoost 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) ซึ่งเป็นวิธีการเฉลี่ย สร้าง 'ป่า' ของ 'ต้นไม้ตัดสินใจ' ที่มีการสุ่มเพื่อหลีกเลี่ยงการ overfitting พารามิเตอร์ n_estimators ถูกตั้งค่าเป็นจำนวนต้นไม้

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ปรับตัวจำแนกประเภทให้เข้ากับชุดข้อมูล และปรับตัวจำแนกประเภทสำเนาให้เข้ากับชุดข้อมูลเดียวกัน โดยมุ่งเน้นไปที่น้ำหนักของรายการที่จำแนกผิด และปรับการฟิตสำหรับตัวจำแนกประเภทถัดไปเพื่อแก้ไข

---

## 🚀ความท้าทาย

เทคนิคแต่ละอย่างมีพารามิเตอร์จำนวนมากที่คุณสามารถปรับแต่งได้ ศึกษาพารามิเตอร์เริ่มต้นของแต่ละเทคนิค และคิดเกี่ยวกับผลกระทบของการปรับแต่งพารามิเตอร์เหล่านี้ต่อคุณภาพของโมเดล

## [แบบทดสอบหลังเรียน](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

มีคำศัพท์เฉพาะทางมากมายในบทเรียนเหล่านี้ ดังนั้นใช้เวลาสักครู่เพื่อทบทวน [รายการนี้](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) ของคำศัพท์ที่มีประโยชน์!

## งานที่ได้รับมอบหมาย 

[Parameter play](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้องมากที่สุด แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความผิดที่เกิดจากการใช้การแปลนี้