<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-06T06:31:09+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "ne"
}
-->
# K-Means क्लस्टरिङ

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

यस पाठमा, तपाईंले Scikit-learn र पहिले आयात गरिएको नाइजेरियन संगीत डेटासेट प्रयोग गरेर क्लस्टरहरू कसरी बनाउने भन्ने कुरा सिक्नुहुनेछ। हामी क्लस्टरिङका लागि K-Means को आधारभूत कुराहरू कभर गर्नेछौं। ध्यान दिनुहोस् कि, जस्तै तपाईंले अघिल्लो पाठमा सिक्नुभयो, क्लस्टरहरूसँग काम गर्ने धेरै तरिकाहरू छन्, र तपाईंले प्रयोग गर्ने विधि तपाईंको डेटामा निर्भर गर्दछ। हामी K-Means प्रयास गर्नेछौं किनकि यो सबैभन्दा सामान्य क्लस्टरिङ प्रविधि हो। सुरु गरौं!

तपाईंले सिक्ने शब्दहरू:

- सिल्हुएट स्कोरिङ
- एल्बो विधि
- इनर्शिया
- भेरियन्स

## परिचय

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) सिग्नल प्रोसेसिङको क्षेत्रबाट व्युत्पन्न विधि हो। यो 'k' क्लस्टरहरूमा डेटा समूहहरू विभाजन र विभाजन गर्न प्रयोग गरिन्छ। प्रत्येक अवलोकनले क्लस्टरको केन्द्र बिन्दु वा 'मीन' नजिकको डेटापोइन्टलाई समूह गर्न काम गर्दछ।

क्लस्टरहरूलाई [Voronoi diagrams](https://wikipedia.org/wiki/Voronoi_diagram) को रूपमा देखाउन सकिन्छ, जसमा एक बिन्दु (वा 'बीज') र यसको सम्बन्धित क्षेत्र समावेश हुन्छ।

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> [Jen Looper](https://twitter.com/jenlooper) द्वारा इन्फोग्राफिक

K-Means क्लस्टरिङ प्रक्रिया [तीन चरणको प्रक्रियामा कार्यान्वयन हुन्छ](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. एल्गोरिदमले डेटासेटबाट नमूना लिएर k-नम्बरको केन्द्र बिन्दु चयन गर्दछ। त्यसपछि यो लूप गर्छ:
    1. यो प्रत्येक नमूनालाई नजिकको सेन्ट्रोइडमा असाइन गर्दछ।
    2. यो नयाँ सेन्ट्रोइडहरू बनाउँछ, अघिल्लो सेन्ट्रोइडहरूमा असाइन गरिएका सबै नमूनाहरूको औसत मान लिँदै।
    3. त्यसपछि, यो नयाँ र पुरानो सेन्ट्रोइडहरू बीचको भिन्नता गणना गर्दछ र सेन्ट्रोइडहरू स्थिर नभएसम्म दोहोर्याउँछ।

K-Means प्रयोग गर्दा एक कमजोरी भनेको तपाईंले 'k' स्थापना गर्न आवश्यक छ, अर्थात सेन्ट्रोइडहरूको संख्या। भाग्यवश, 'एल्बो विधि' ले 'k' को लागि राम्रो सुरुवात मान अनुमान गर्न मद्दत गर्दछ। तपाईंले यसलाई केही समयपछि प्रयास गर्नुहुनेछ।

## पूर्वापेक्षा

तपाईंले यस पाठको [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) फाइलमा काम गर्नुहुनेछ जसमा अघिल्लो पाठमा गरिएको डेटा आयात र प्रारम्भिक सफाइ समावेश छ।

## अभ्यास - तयारी

गीतहरूको डेटालाई फेरि हेर्न सुरु गर्नुहोस्।

1. प्रत्येक स्तम्भको लागि `boxplot()` कल गर्दै बक्सप्लट बनाउनुहोस्:

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

    यो डेटा अलि धेरै शोरयुक्त छ: प्रत्येक स्तम्भलाई बक्सप्लटको रूपमा अवलोकन गर्दा, तपाईंले आउटलायर्स देख्न सक्नुहुन्छ।

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

तपाईं डेटासेटबाट यी आउटलायर्स हटाउन सक्नुहुन्छ, तर यसले डेटालाई धेरै न्यूनतम बनाउनेछ।

1. अहिलेको लागि, तपाईंले क्लस्टरिङ अभ्यासको लागि कुन स्तम्भहरू प्रयोग गर्ने छनोट गर्नुहोस्। समान दायराहरू भएका स्तम्भहरू चयन गर्नुहोस् र `artist_top_genre` स्तम्भलाई संख्यात्मक डेटाको रूपमा एन्कोड गर्नुहोस्:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. अब तपाईंले कति क्लस्टरहरू लक्षित गर्ने छनोट गर्न आवश्यक छ। तपाईंलाई थाहा छ कि डेटासेटबाट हामीले 3 गीत शैलीहरू निकालेका छौं, त्यसैले 3 प्रयास गरौं:

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

तपाईंले डेटा फ्रेमको प्रत्येक पङ्क्तिको लागि भविष्यवाणी गरिएको क्लस्टरहरूको (0, 1, वा 2) एरे प्रिन्ट भएको देख्नुहुन्छ।

1. यो एरे प्रयोग गरेर 'सिल्हुएट स्कोर' गणना गर्नुहोस्:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## सिल्हुएट स्कोर

सिल्हुएट स्कोर 1 नजिक खोज्नुहोस्। यो स्कोर -1 देखि 1 सम्म फरक हुन्छ, र यदि स्कोर 1 छ भने, क्लस्टर घना र अन्य क्लस्टरहरूबाट राम्रोसँग छुट्टिएको हुन्छ। 0 नजिकको मानले छिमेकी क्लस्टरहरूको निर्णय सीमा नजिक नमूनाहरू भएको ओभरल्याप क्लस्टरहरू प्रतिनिधित्व गर्दछ। [(स्रोत)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

हाम्रो स्कोर **.53** छ, त्यसैले ठीक बीचमा। यसले संकेत गर्दछ कि हाम्रो डेटा यो प्रकारको क्लस्टरिङको लागि विशेष रूपमा उपयुक्त छैन, तर अगाडि बढौं।

### अभ्यास - मोडेल निर्माण गर्नुहोस्

1. `KMeans` आयात गर्नुहोस् र क्लस्टरिङ प्रक्रिया सुरु गर्नुहोस्।

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    यहाँ केही भागहरू छन् जसको व्याख्या गर्न आवश्यक छ।

    > 🎓 range: यी क्लस्टरिङ प्रक्रियाका पुनरावृत्तिहरू हुन्।

    > 🎓 random_state: "सेन्ट्रोइड इनिसियलाइजेसनको लागि र्यान्डम नम्बर जेनेरेसन निर्धारण गर्दछ।" [स्रोत](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" ले क्लस्टर सेन्ट्रोइडको लागि क्लस्टर भित्रका सबै बिन्दुहरूको औसत दूरी मापन गर्दछ। [स्रोत](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 Inertia: K-Means एल्गोरिदमले 'इनर्शिया' कम गर्न सेन्ट्रोइडहरू चयन गर्ने प्रयास गर्दछ, "क्लस्टरहरू आन्तरिक रूपमा कति सुसंगत छन् भन्ने मापन।" [स्रोत](https://scikit-learn.org/stable/modules/clustering.html)। मान प्रत्येक पुनरावृत्तिमा wcss भेरिएबलमा थपिन्छ।

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) मा तपाईं 'k-means++' अप्टिमाइजेसन प्रयोग गर्न सक्नुहुन्छ, जसले "सेन्ट्रोइडहरूलाई सामान्यतया एकअर्काबाट टाढा इनिसियलाइज गर्दछ, जसले सम्भवतः र्यान्डम इनिसियलाइजेसन भन्दा राम्रो परिणामहरू दिन्छ।"

### एल्बो विधि

पहिले, तपाईंले अनुमान गर्नुभयो कि, किनभने तपाईंले 3 गीत शैलीहरू लक्षित गर्नुभएको छ, तपाईंले 3 क्लस्टरहरू चयन गर्नुपर्छ। तर के यो सही हो?

1. 'एल्बो विधि' प्रयोग गरेर सुनिश्चित गर्नुहोस्।

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    तपाईंले अघिल्लो चरणमा निर्माण गरेको `wcss` भेरिएबल प्रयोग गरेर चार्ट बनाउनुहोस् जसले एल्बोको 'बेंड' कहाँ छ देखाउँछ, जसले क्लस्टरहरूको इष्टतम संख्या संकेत गर्दछ। सायद यो **3** नै हो!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## अभ्यास - क्लस्टरहरू प्रदर्शन गर्नुहोस्

1. प्रक्रिया फेरि प्रयास गर्नुहोस्, यस पटक तीन क्लस्टर सेट गर्दै, र क्लस्टरहरूलाई स्क्याटरप्लटको रूपमा प्रदर्शन गर्नुहोस्:

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

1. मोडेलको शुद्धता जाँच गर्नुहोस्:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    यो मोडेलको शुद्धता धेरै राम्रो छैन, र क्लस्टरहरूको आकारले तपाईंलाई किनको संकेत दिन्छ।

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    यो डेटा धेरै असन्तुलित छ, धेरै कम सम्बन्धित छ, र स्तम्भ मानहरू बीच धेरै भिन्नता छ जसले राम्रोसँग क्लस्टर गर्न सक्दैन। वास्तवमा, बनाइएका क्लस्टरहरू सम्भवतः माथि परिभाषित तीन शैली वर्गहरूद्वारा धेरै प्रभावित वा झुकाव भएका छन्। यो सिक्ने प्रक्रिया थियो!

    Scikit-learn को दस्तावेजमा, तपाईं देख्न सक्नुहुन्छ कि यस्तो मोडेल, जसको क्लस्टरहरू धेरै राम्रोसँग छुट्टिएका छैनन्, 'भेरियन्स' समस्या छ:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Scikit-learn बाट इन्फोग्राफिक

## भेरियन्स

भेरियन्सलाई "मीनबाट वर्गीय भिन्नताहरूको औसत" भनेर परिभाषित गरिएको छ [(स्रोत)](https://www.mathsisfun.com/data/standard-deviation.html)। यस क्लस्टरिङ समस्याको सन्दर्भमा, यसले हाम्रो डेटासेटका संख्याहरू मीनबाट धेरै टाढा हुने प्रवृत्तिलाई जनाउँछ।

✅ यो समस्या सुधार गर्न सक्ने सबै तरिकाहरूको बारेमा सोच्ने राम्रो समय हो। डेटालाई अलि बढी परिमार्जन गर्ने? फरक स्तम्भहरू प्रयोग गर्ने? फरक एल्गोरिदम प्रयोग गर्ने? संकेत: [तपाईंको डेटालाई स्केल गर्ने प्रयास गर्नुहोस्](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) र अन्य स्तम्भहरू परीक्षण गर्नुहोस्।

> यो '[भेरियन्स क्याल्कुलेटर](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' प्रयास गरेर अवधारणालाई अलि बढी बुझ्ने प्रयास गर्नुहोस्।

---

## 🚀चुनौती

यस नोटबुकसँग केही समय बिताउनुहोस्, प्यारामिटरहरू परिमार्जन गर्दै। के तपाईं आउटलायर्स हटाएर, उदाहरणका लागि, डेटा सफा गरेर मोडेलको शुद्धता सुधार गर्न सक्नुहुन्छ? तपाईंले केही डेटा नमूनाहरूलाई बढी तौल दिन तौलहरू प्रयोग गर्न सक्नुहुन्छ। राम्रो क्लस्टरहरू बनाउन के गर्न सक्नुहुन्छ?

संकेत: तपाईंको डेटालाई स्केल गर्ने प्रयास गर्नुहोस्। नोटबुकमा टिप्पणी गरिएको कोड छ जसले मानक स्केलिङ थप्छ ताकि डेटा स्तम्भहरू दायराको सन्दर्भमा एकअर्कासँग बढी नजिक देखिन्छन्। तपाईंले पाउनुहुनेछ कि सिल्हुएट स्कोर तल जान्छ, तर एल्बो ग्राफको 'किंक' नरम हुन्छ। यो किनभने डेटालाई अनस्केल छोड्दा कम भेरियन्स भएको डेटाले बढी तौल बोक्न अनुमति दिन्छ। यस समस्याको बारेमा [यहाँ](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226) अलि बढी पढ्नुहोस्।

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा र आत्म अध्ययन

K-Means सिमुलेटर [जस्तै यो](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) हेर्नुहोस्। तपाईंले यो उपकरण प्रयोग गरेर नमूना डेटा बिन्दुहरू देखाउन र यसको सेन्ट्रोइडहरू निर्धारण गर्न सक्नुहुन्छ। तपाईंले डेटाको र्यान्डमनेस, क्लस्टरहरूको संख्या र सेन्ट्रोइडहरूको संख्या सम्पादन गर्न सक्नुहुन्छ। के यसले तपाईंलाई डेटालाई कसरी समूह गर्न सकिन्छ भन्ने विचार दिन मद्दत गर्दछ?

साथै, [Stanford को K-Means सम्बन्धी यो ह्यान्डआउट](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) हेर्नुहोस्।

## असाइनमेन्ट

[विभिन्न क्लस्टरिङ विधिहरू प्रयास गर्नुहोस्](assignment.md)

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको हो। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादमा त्रुटिहरू वा अशुद्धताहरू हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।