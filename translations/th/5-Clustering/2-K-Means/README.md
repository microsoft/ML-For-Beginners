<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T21:28:49+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "th"
}
-->
# การจัดกลุ่มด้วย K-Means

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/)

ในบทเรียนนี้ คุณจะได้เรียนรู้วิธีการสร้างกลุ่มโดยใช้ Scikit-learn และชุดข้อมูลเพลงไนจีเรียที่คุณนำเข้าไว้ก่อนหน้านี้ เราจะครอบคลุมพื้นฐานของ K-Means สำหรับการจัดกลุ่ม โปรดจำไว้ว่า ตามที่คุณได้เรียนรู้ในบทเรียนก่อนหน้า มีหลายวิธีในการทำงานกับการจัดกลุ่ม และวิธีที่คุณใช้ขึ้นอยู่กับข้อมูลของคุณ เราจะลองใช้ K-Means เนื่องจากเป็นเทคนิคการจัดกลุ่มที่พบได้บ่อยที่สุด มาเริ่มกันเลย!

คำศัพท์ที่คุณจะได้เรียนรู้:

- คะแนน Silhouette
- วิธี Elbow
- Inertia
- Variance

## บทนำ

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) เป็นวิธีที่มาจากสาขาการประมวลผลสัญญาณ ใช้ในการแบ่งและจัดกลุ่มข้อมูลออกเป็น 'k' กลุ่มโดยใช้ชุดของการสังเกตการณ์ การสังเกตการณ์แต่ละครั้งจะทำงานเพื่อจัดกลุ่มจุดข้อมูลที่ใกล้ที่สุดกับ 'ค่าเฉลี่ย' หรือจุดศูนย์กลางของกลุ่มนั้น

กลุ่มเหล่านี้สามารถแสดงผลเป็น [แผนภาพ Voronoi](https://wikipedia.org/wiki/Voronoi_diagram) ซึ่งรวมถึงจุด (หรือ 'seed') และพื้นที่ที่เกี่ยวข้อง

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> อินโฟกราฟิกโดย [Jen Looper](https://twitter.com/jenlooper)

กระบวนการจัดกลุ่มด้วย K-Means [ดำเนินการในสามขั้นตอน](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. อัลกอริทึมเลือกจุดศูนย์กลางจำนวน k โดยสุ่มจากชุดข้อมูล หลังจากนั้นจะวนซ้ำ:
    1. กำหนดตัวอย่างแต่ละตัวให้กับจุดศูนย์กลางที่ใกล้ที่สุด
    2. สร้างจุดศูนย์กลางใหม่โดยการคำนวณค่าเฉลี่ยของตัวอย่างทั้งหมดที่ถูกกำหนดให้กับจุดศูนย์กลางก่อนหน้า
    3. จากนั้นคำนวณความแตกต่างระหว่างจุดศูนย์กลางใหม่และเก่า และทำซ้ำจนกว่าจุดศูนย์กลางจะคงที่

ข้อเสียของการใช้ K-Means คือคุณต้องกำหนด 'k' ซึ่งเป็นจำนวนจุดศูนย์กลาง โชคดีที่ 'วิธี Elbow' ช่วยประมาณค่าที่ดีสำหรับ 'k' คุณจะได้ลองในอีกสักครู่

## สิ่งที่ต้องเตรียม

คุณจะทำงานในไฟล์ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) ของบทเรียนนี้ ซึ่งรวมถึงการนำเข้าข้อมูลและการทำความสะอาดเบื้องต้นที่คุณทำในบทเรียนก่อนหน้า

## แบบฝึกหัด - การเตรียมตัว

เริ่มต้นด้วยการดูข้อมูลเพลงอีกครั้ง

1. สร้าง boxplot โดยเรียกใช้ `boxplot()` สำหรับแต่ละคอลัมน์:

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

    ข้อมูลนี้ค่อนข้างมีเสียงรบกวน: โดยการสังเกตแต่ละคอลัมน์ในรูปแบบ boxplot คุณจะเห็นค่าผิดปกติ

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

คุณสามารถตรวจสอบชุดข้อมูลและลบค่าผิดปกติเหล่านี้ออกได้ แต่จะทำให้ข้อมูลลดลงไปมาก

1. สำหรับตอนนี้ เลือกคอลัมน์ที่คุณจะใช้สำหรับการจัดกลุ่ม เลือกคอลัมน์ที่มีช่วงค่าคล้ายกันและเข้ารหัสคอลัมน์ `artist_top_genre` เป็นข้อมูลตัวเลข:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. ตอนนี้คุณต้องเลือกจำนวนกลุ่มที่จะกำหนดเป้าหมาย คุณทราบว่ามี 3 ประเภทเพลงที่เราสกัดออกมาจากชุดข้อมูล ดังนั้นลองใช้ 3:

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

คุณจะเห็นอาร์เรย์ที่พิมพ์ออกมาพร้อมกลุ่มที่คาดการณ์ไว้ (0, 1 หรือ 2) สำหรับแต่ละแถวของ dataframe

1. ใช้อาร์เรย์นี้เพื่อคำนวณ 'คะแนน Silhouette':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## คะแนน Silhouette

มองหาคะแนน Silhouette ที่ใกล้เคียงกับ 1 คะแนนนี้มีค่าตั้งแต่ -1 ถึง 1 และหากคะแนนเป็น 1 กลุ่มจะมีความหนาแน่นและแยกออกจากกลุ่มอื่นได้ดี ค่าใกล้ 0 แสดงถึงกลุ่มที่ทับซ้อนกันโดยมีตัวอย่างที่อยู่ใกล้กับขอบเขตการตัดสินใจของกลุ่มที่อยู่ใกล้เคียง [(แหล่งข้อมูล)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

คะแนนของเราคือ **.53** ซึ่งอยู่ตรงกลาง แสดงว่าข้อมูลของเราไม่เหมาะสมกับการจัดกลุ่มประเภทนี้มากนัก แต่ลองดำเนินการต่อไป

### แบบฝึกหัด - สร้างโมเดล

1. นำเข้า `KMeans` และเริ่มกระบวนการจัดกลุ่ม

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    มีบางส่วนที่ควรอธิบายเพิ่มเติม

    > 🎓 range: นี่คือจำนวนครั้งของกระบวนการจัดกลุ่ม

    > 🎓 random_state: "กำหนดการสร้างตัวเลขสุ่มสำหรับการเริ่มต้นจุดศูนย์กลาง" [แหล่งข้อมูล](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "ผลรวมของกำลังสองภายในกลุ่ม" วัดระยะทางเฉลี่ยกำลังสองของจุดทั้งหมดภายในกลุ่มไปยังจุดศูนย์กลางของกลุ่ม [แหล่งข้อมูล](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 Inertia: อัลกอริทึม K-Means พยายามเลือกจุดศูนย์กลางเพื่อลด 'Inertia' ซึ่งเป็น "การวัดความสอดคล้องภายในของกลุ่ม" [แหล่งข้อมูล](https://scikit-learn.org/stable/modules/clustering.html) ค่านี้จะถูกเพิ่มเข้าไปในตัวแปร wcss ในแต่ละครั้ง

    > 🎓 k-means++: ใน [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) คุณสามารถใช้การปรับแต่ง 'k-means++' ซึ่ง "เริ่มต้นจุดศูนย์กลางให้ห่างกัน (โดยทั่วไป) ซึ่งนำไปสู่ผลลัพธ์ที่ดีกว่าการเริ่มต้นแบบสุ่ม"

### วิธี Elbow

ก่อนหน้านี้ คุณคาดการณ์ว่า เนื่องจากคุณกำหนดเป้าหมาย 3 ประเภทเพลง คุณควรเลือก 3 กลุ่ม แต่เป็นเช่นนั้นจริงหรือ?

1. ใช้วิธี 'Elbow' เพื่อให้แน่ใจ

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    ใช้ตัวแปร `wcss` ที่คุณสร้างในขั้นตอนก่อนหน้าเพื่อสร้างแผนภูมิที่แสดงจุด 'โค้ง' ในกราฟ Elbow ซึ่งบ่งชี้จำนวนกลุ่มที่เหมาะสมที่สุด อาจจะเป็น **3** จริงๆ!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## แบบฝึกหัด - แสดงกลุ่ม

1. ลองกระบวนการอีกครั้ง คราวนี้ตั้งค่ากลุ่มเป็นสามกลุ่ม และแสดงกลุ่มในรูปแบบ scatterplot:

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

1. ตรวจสอบความแม่นยำของโมเดล:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    ความแม่นยำของโมเดลนี้ไม่ค่อยดีนัก และรูปร่างของกลุ่มให้คำใบ้ว่าเหตุใด 

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    ข้อมูลนี้ไม่สมดุลกันมากเกินไป มีความสัมพันธ์น้อยเกินไป และมีความแปรปรวนระหว่างค่าคอลัมน์มากเกินไปที่จะจัดกลุ่มได้ดี ในความเป็นจริง กลุ่มที่เกิดขึ้นอาจได้รับอิทธิพลหรือเบี่ยงเบนอย่างมากจากสามประเภทเพลงที่เรากำหนดไว้ข้างต้น นี่เป็นกระบวนการเรียนรู้!

    ในเอกสารของ Scikit-learn คุณสามารถเห็นว่าโมเดลแบบนี้ ที่กลุ่มไม่ได้ถูกแบ่งแยกอย่างชัดเจน มีปัญหา 'Variance':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > อินโฟกราฟิกจาก Scikit-learn

## Variance

Variance ถูกนิยามว่าเป็น "ค่าเฉลี่ยของผลต่างกำลังสองจากค่าเฉลี่ย" [(แหล่งข้อมูล)](https://www.mathsisfun.com/data/standard-deviation.html) ในบริบทของปัญหาการจัดกลุ่มนี้ หมายถึงข้อมูลที่ตัวเลขในชุดข้อมูลมีแนวโน้มที่จะเบี่ยงเบนจากค่าเฉลี่ยมากเกินไป 

✅ นี่เป็นช่วงเวลาที่ดีในการคิดถึงวิธีต่างๆ ที่คุณสามารถแก้ไขปัญหานี้ได้ ปรับข้อมูลเพิ่มเติม? ใช้คอลัมน์อื่น? ใช้อัลกอริทึมอื่น? คำใบ้: ลอง [ปรับขนาดข้อมูลของคุณ](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) เพื่อทำให้ข้อมูลเป็นปกติและทดสอบคอลัมน์อื่นๆ

> ลองใช้ '[เครื่องคำนวณ Variance](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' เพื่อทำความเข้าใจแนวคิดนี้เพิ่มเติม

---

## 🚀ความท้าทาย

ใช้เวลาสักครู่กับ notebook นี้ ปรับพารามิเตอร์ คุณสามารถปรับปรุงความแม่นยำของโมเดลได้โดยการทำความสะอาดข้อมูลเพิ่มเติม (เช่น ลบค่าผิดปกติ)? คุณสามารถใช้น้ำหนักเพื่อให้น้ำหนักมากขึ้นกับตัวอย่างข้อมูลบางตัว คุณสามารถทำอะไรอีกเพื่อสร้างกลุ่มที่ดีกว่า?

คำใบ้: ลองปรับขนาดข้อมูลของคุณ มีโค้ดที่ถูกคอมเมนต์ไว้ใน notebook ที่เพิ่มการปรับขนาดมาตรฐานเพื่อทำให้คอลัมน์ข้อมูลมีลักษณะคล้ายกันมากขึ้นในแง่ของช่วงค่า คุณจะพบว่าในขณะที่คะแนน Silhouette ลดลง กราฟ Elbow จะราบเรียบขึ้น นี่เป็นเพราะการปล่อยข้อมูลที่ไม่ได้ปรับขนาดทำให้ข้อมูลที่มีความแปรปรวนน้อยมีน้ำหนักมากขึ้น อ่านเพิ่มเติมเกี่ยวกับปัญหานี้ [ที่นี่](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226)

## [แบบทดสอบหลังเรียน](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

ลองดูตัวจำลอง K-Means [เช่นนี้](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) คุณสามารถใช้เครื่องมือนี้เพื่อแสดงจุดข้อมูลตัวอย่างและกำหนดจุดศูนย์กลาง คุณสามารถแก้ไขความสุ่มของข้อมูล จำนวนกลุ่ม และจำนวนจุดศูนย์กลาง สิ่งนี้ช่วยให้คุณเข้าใจวิธีการจัดกลุ่มข้อมูลได้หรือไม่?

นอกจากนี้ ลองดู [เอกสารประกอบเกี่ยวกับ K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) จาก Stanford

## งานที่ได้รับมอบหมาย

[ลองใช้วิธีการจัดกลุ่มแบบต่างๆ](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้