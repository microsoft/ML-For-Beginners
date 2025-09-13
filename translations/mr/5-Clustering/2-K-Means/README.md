<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-06T06:10:57+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "mr"
}
-->
# K-Means क्लस्टरिंग

## [पूर्व व्याख्यान प्रश्नमंजूषा](https://ff-quizzes.netlify.app/en/ml/)

या धड्यात, तुम्ही Scikit-learn आणि तुम्ही आधी आयात केलेल्या नायजेरियन संगीत डेटासेटचा वापर करून क्लस्टर तयार करणे शिकाल. आपण क्लस्टरिंगसाठी K-Means च्या मूलभूत गोष्टींचा अभ्यास करू. लक्षात ठेवा, जसे तुम्ही मागील धड्यात शिकले, क्लस्टरसह काम करण्याचे अनेक मार्ग आहेत आणि तुम्ही वापरलेली पद्धत तुमच्या डेटावर अवलंबून असते. आपण K-Means वापरून पाहू कारण ही सर्वात सामान्य क्लस्टरिंग तंत्र आहे. चला सुरुवात करूया!

तुम्ही शिकणार असलेल्या संज्ञा:

- सिल्हूट स्कोअरिंग
- एल्बो पद्धत
- इनर्शिया
- व्हेरियन्स

## परिचय

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) ही सिग्नल प्रोसेसिंगच्या क्षेत्रातून आलेली पद्धत आहे. ही पद्धत 'k' क्लस्टरमध्ये डेटा गट विभाजित आणि विभागण्यासाठी वापरली जाते, निरीक्षणांच्या मालिकेचा वापर करून. प्रत्येक निरीक्षण दिलेल्या डेटापॉइंटला त्याच्या जवळच्या 'मीन' किंवा क्लस्टरच्या केंद्र बिंदूपाशी गटबद्ध करण्यासाठी कार्य करते.

क्लस्टर [वोरोनोई डायग्राम्स](https://wikipedia.org/wiki/Voronoi_diagram) म्हणून व्हिज्युअलाइझ केले जाऊ शकतात, ज्यामध्ये एक बिंदू (किंवा 'सीड') आणि त्याचा संबंधित प्रदेश समाविष्ट असतो.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> [Jen Looper](https://twitter.com/jenlooper) यांनी तयार केलेले माहितीपट

K-Means क्लस्टरिंग प्रक्रिया [तीन-स्टेप प्रक्रियेत कार्य करते](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. अल्गोरिदम डेटासेटमधून नमुना घेऊन k-नंबर केंद्र बिंदू निवडतो. त्यानंतर तो लूप करतो:
    1. प्रत्येक नमुन्याला जवळच्या सेंटरॉइडला असाइन करतो.
    2. मागील सेंटरॉइड्सला असाइन केलेल्या सर्व नमुन्यांचे सरासरी मूल्य घेऊन नवीन सेंटरॉइड तयार करतो.
    3. मग नवीन आणि जुन्या सेंटरॉइड्समधील फरकाची गणना करतो आणि सेंटरॉइड्स स्थिर होईपर्यंत प्रक्रिया पुन्हा करतो.

K-Means वापरण्याचा एक तोटा म्हणजे तुम्हाला 'k' म्हणजे सेंटरॉइड्सची संख्या निश्चित करावी लागेल. सुदैवाने, 'एल्बो पद्धत' 'k' साठी चांगली सुरुवातीची किंमत अंदाजे ठरवण्यास मदत करते. तुम्ही ते थोड्या वेळात वापरून पाहाल.

## पूर्वतयारी

तुम्ही या धड्याच्या [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) फाइलमध्ये काम कराल ज्यामध्ये तुम्ही मागील धड्यात केलेला डेटा आयात आणि प्राथमिक स्वच्छता समाविष्ट आहे.

## व्यायाम - तयारी

गाण्यांच्या डेटावर पुन्हा एकदा नजर टाका.

1. प्रत्येक स्तंभासाठी `boxplot()` कॉल करून बॉक्सप्लॉट तयार करा:

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

    हा डेटा थोडा गोंधळलेला आहे: प्रत्येक स्तंभ बॉक्सप्लॉट म्हणून पाहून, तुम्हाला बाह्य घटक दिसतात.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

तुम्ही डेटासेटमधून हे बाह्य घटक काढून टाकू शकता, परंतु त्यामुळे डेटा खूपच कमी होईल.

1. सध्या, तुम्ही क्लस्टरिंग व्यायामासाठी कोणते स्तंभ वापरायचे ते निवडा. समान श्रेणी असलेले स्तंभ निवडा आणि `artist_top_genre` स्तंभाला संख्यात्मक डेटामध्ये एन्कोड करा:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. आता तुम्हाला किती क्लस्टर लक्ष्य करायचे आहेत ते निवडायचे आहे. तुम्हाला माहित आहे की डेटासेटमधून आम्ही 3 गाण्यांच्या शैली काढल्या आहेत, त्यामुळे 3 वापरून पाहूया:

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

तुम्हाला डेटाफ्रेमच्या प्रत्येक ओळीसाठी अंदाजे क्लस्टर (0, 1, किंवा 2) असलेली एक अ‍ॅरे प्रिंट झालेली दिसते.

1. या अ‍ॅरेचा वापर करून 'सिल्हूट स्कोअर'ची गणना करा:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## सिल्हूट स्कोअर

सिल्हूट स्कोअर 1 च्या जवळ शोधा. हा स्कोअर -1 ते 1 पर्यंत बदलतो, आणि जर स्कोअर 1 असेल, तर क्लस्टर घन आणि इतर क्लस्टरपासून चांगले वेगळे असते. 0 च्या जवळ असलेली किंमत शेजारील क्लस्टरच्या निर्णय सीमा जवळ असलेल्या नमुन्यांसह ओव्हरलॅपिंग क्लस्टरचे प्रतिनिधित्व करते. [(स्रोत)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

आपला स्कोअर **.53** आहे, म्हणजे अगदी मध्यम. याचा अर्थ असा की आपला डेटा या प्रकारच्या क्लस्टरिंगसाठी विशेषतः योग्य नाही, परंतु चला पुढे जाऊया.

### व्यायाम - मॉडेल तयार करा

1. `KMeans` आयात करा आणि क्लस्टरिंग प्रक्रिया सुरू करा.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    येथे काही भाग आहेत ज्यांचे स्पष्टीकरण आवश्यक आहे.

    > 🎓 range: क्लस्टरिंग प्रक्रियेच्या पुनरावृत्ती आहेत

    > 🎓 random_state: "सेंटरॉइड प्रारंभासाठी यादृच्छिक क्रमांक निर्मिती निश्चित करते." [स्रोत](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" क्लस्टर सेंटरॉइडच्या जवळ असलेल्या सर्व बिंदूंचे सरासरी अंतर मोजते. [स्रोत](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Inertia: K-Means अल्गोरिदम 'इनर्शिया' कमी करण्यासाठी सेंटरॉइड्स निवडण्याचा प्रयत्न करतो, "क्लस्टर किती अंतर्गत सुसंगत आहेत याचे मोजमाप." [स्रोत](https://scikit-learn.org/stable/modules/clustering.html). प्रत्येक पुनरावृत्तीत wcss व्हेरिएबलमध्ये मूल्य जोडले जाते.

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) मध्ये तुम्ही 'k-means++' ऑप्टिमायझेशन वापरू शकता, जे "सेंटरॉइड्स एकमेकांपासून (सामान्यतः) दूर असतील असे प्रारंभ करते, ज्यामुळे यादृच्छिक प्रारंभापेक्षा चांगले परिणाम मिळण्याची शक्यता असते."

### एल्बो पद्धत

पूर्वी, तुम्ही अंदाज केला होता की, कारण तुम्ही 3 गाण्यांच्या शैली लक्ष्य केल्या आहेत, तुम्ही 3 क्लस्टर निवडले पाहिजेत. पण ते खरे आहे का?

1. 'एल्बो पद्धत' वापरून खात्री करा.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    तुम्ही मागील चरणात तयार केलेल्या `wcss` व्हेरिएबलचा वापर करून एक चार्ट तयार करा ज्यामध्ये 'एल्बो'चा वाकलेला भाग दर्शविला जातो, जो क्लस्टरची आदर्श संख्या दर्शवतो. कदाचित ते **खरोखरच** 3 आहे!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## व्यायाम - क्लस्टर प्रदर्शित करा

1. प्रक्रिया पुन्हा प्रयत्न करा, यावेळी तीन क्लस्टर सेट करा आणि क्लस्टर स्कॅटरप्लॉट म्हणून प्रदर्शित करा:

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

1. मॉडेलची अचूकता तपासा:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    या मॉडेलची अचूकता फारशी चांगली नाही आणि क्लस्टरच्या आकारामुळे तुम्हाला का ते समजते. 

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    हा डेटा खूप असंतुलित आहे, खूप कमी संबंधित आहे आणि स्तंभ मूल्यांमध्ये खूप जास्त फरक आहे ज्यामुळे चांगले क्लस्टर तयार होऊ शकत नाहीत. खरं तर, तयार होणारे क्लस्टर कदाचित आपण वर परिभाषित केलेल्या तीन शैली श्रेणींनी मोठ्या प्रमाणावर प्रभावित किंवा वाकवले जातात. हे एक शिकण्याची प्रक्रिया होती!

    Scikit-learn च्या दस्तऐवजांमध्ये, तुम्ही पाहू शकता की अशा प्रकारच्या मॉडेलमध्ये, क्लस्टर फारसे चांगले परिभाषित नसल्यामुळे, 'व्हेरियन्स' समस्या आहे:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Scikit-learn कडून माहितीपट

## व्हेरियन्स

व्हेरियन्स म्हणजे "मीनपासून चौरस फरकांचे सरासरी" [(स्रोत)](https://www.mathsisfun.com/data/standard-deviation.html). या क्लस्टरिंग समस्येच्या संदर्भात, याचा अर्थ असा आहे की आमच्या डेटासेटमधील संख्या मीनपासून थोड्या जास्त प्रमाणात विचलित होण्याची प्रवृत्ती आहे.

✅ हा एक उत्तम क्षण आहे ज्यामध्ये तुम्ही ही समस्या सुधारण्यासाठी सर्व मार्गांचा विचार करू शकता. डेटा थोडा अधिक बदलणे? वेगळे स्तंभ वापरणे? वेगळा अल्गोरिदम वापरणे? सूचक: तुमचा डेटा [स्केलिंग](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) करून सामान्य करणे आणि इतर स्तंभ तपासणे प्रयत्न करा.

> '[व्हेरियन्स कॅल्क्युलेटर](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' वापरून संकल्पना अधिक चांगल्या प्रकारे समजून घ्या.

---

## 🚀चॅलेंज

या नोटबुकसह काही वेळ घालवा, पॅरामीटर्स बदलून पहा. तुम्ही डेटा अधिक स्वच्छ करून (उदाहरणार्थ बाह्य घटक काढून टाकून) मॉडेलची अचूकता सुधारू शकता का? तुम्ही दिलेल्या डेटा नमुन्यांना अधिक वजन देण्यासाठी वजन वापरू शकता. चांगले क्लस्टर तयार करण्यासाठी तुम्ही आणखी काय करू शकता?

सूचक: तुमचा डेटा स्केल करण्याचा प्रयत्न करा. नोटबुकमध्ये टिप्पणी केलेला कोड आहे जो मानक स्केलिंग जोडतो ज्यामुळे डेटा स्तंभ श्रेणीच्या बाबतीत एकमेकांशी अधिक जवळून जुळतात. तुम्हाला असे आढळेल की सिल्हूट स्कोअर कमी होतो, परंतु एल्बो ग्राफमधील 'किंक' गुळगुळीत होतो. कारण डेटा स्केल न केल्याने कमी व्हेरियन्स असलेल्या डेटाला अधिक वजन मिळते. या समस्येवर अधिक वाचा [येथे](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [व्याख्यानानंतर प्रश्नमंजूषा](https://ff-quizzes.netlify.app/en/ml/)

## पुनरावलोकन आणि स्व-अभ्यास

K-Means सिम्युलेटर [जसे की हा](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) पहा. तुम्ही नमुना डेटा पॉइंट्स व्हिज्युअलाइझ करण्यासाठी आणि त्याचे सेंटरॉइड्स निश्चित करण्यासाठी हे साधन वापरू शकता. तुम्ही डेटाच्या यादृच्छिकतेत, क्लस्टरच्या संख्येत आणि सेंटरॉइड्सच्या संख्येत संपादन करू शकता. यामुळे तुम्हाला डेटा कसा गटबद्ध केला जाऊ शकतो याची कल्पना मिळते का?

तसेच, [Stanford](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) कडून K-Means वर हा हँडआउट पहा.

## असाइनमेंट

[वेगळ्या क्लस्टरिंग पद्धती वापरून पहा](assignment.md)

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) चा वापर करून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात घ्या की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील मूळ दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी, व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर केल्यामुळे उद्भवलेल्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.