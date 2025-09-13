<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T10:20:37+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "hi"
}
-->
# K-Means क्लस्टरिंग

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

इस पाठ में, आप Scikit-learn और पहले आयात किए गए नाइजीरियाई संगीत डेटा सेट का उपयोग करके क्लस्टर बनाना सीखेंगे। हम क्लस्टरिंग के लिए K-Means की मूल बातें कवर करेंगे। ध्यान रखें कि, जैसा कि आपने पिछले पाठ में सीखा था, क्लस्टर के साथ काम करने के कई तरीके हैं और आप जो विधि उपयोग करते हैं वह आपके डेटा पर निर्भर करती है। हम K-Means आज़माएंगे क्योंकि यह सबसे सामान्य क्लस्टरिंग तकनीक है। चलिए शुरू करते हैं!

आप जिन शब्दों के बारे में जानेंगे:

- सिल्हूट स्कोरिंग
- एल्बो मेथड
- इनर्शिया
- वेरिएंस

## परिचय

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) सिग्नल प्रोसेसिंग के क्षेत्र से निकली एक विधि है। इसका उपयोग डेटा के समूहों को 'k' क्लस्टर में विभाजित और वर्गीकृत करने के लिए किया जाता है। प्रत्येक ऑब्ज़र्वेशन एक दिए गए डेटा पॉइंट को उसके निकटतम 'मीन' या क्लस्टर के केंद्र बिंदु के सबसे करीब समूहित करने का काम करता है।

क्लस्टर को [Voronoi diagrams](https://wikipedia.org/wiki/Voronoi_diagram) के रूप में देखा जा सकता है, जिसमें एक बिंदु (या 'सीड') और उसका संबंधित क्षेत्र शामिल होता है।

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> [Jen Looper](https://twitter.com/jenlooper) द्वारा इन्फोग्राफिक

K-Means क्लस्टरिंग प्रक्रिया [तीन चरणों में निष्पादित होती है](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. एल्गोरिदम डेटा सेट से k संख्या के केंद्र बिंदु चुनता है। इसके बाद यह लूप करता है:
    1. यह प्रत्येक सैंपल को निकटतम सेंट्रॉइड को असाइन करता है।
    2. यह पिछले सेंट्रॉइड्स को असाइन किए गए सभी सैंपल्स के औसत मान को लेकर नए सेंट्रॉइड बनाता है।
    3. फिर यह नए और पुराने सेंट्रॉइड्स के बीच का अंतर गणना करता है और तब तक दोहराता है जब तक सेंट्रॉइड्स स्थिर न हो जाएं।

K-Means का उपयोग करने की एक कमी यह है कि आपको 'k', यानी सेंट्रॉइड्स की संख्या स्थापित करनी होगी। सौभाग्य से, 'एल्बो मेथड' 'k' के लिए एक अच्छा प्रारंभिक मान अनुमान लगाने में मदद करता है। आप इसे अभी आज़माएंगे।

## पूर्वापेक्षा

आप इस पाठ के [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) फ़ाइल में काम करेंगे, जिसमें पिछले पाठ में किया गया डेटा आयात और प्रारंभिक सफाई शामिल है।

## अभ्यास - तैयारी

गानों के डेटा पर एक बार फिर नज़र डालें।

1. प्रत्येक कॉलम के लिए `boxplot()` कॉल करके एक बॉक्सप्लॉट बनाएं:

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

    यह डेटा थोड़ा शोरयुक्त है: प्रत्येक कॉलम को बॉक्सप्लॉट के रूप में देखकर, आप आउटलायर्स देख सकते हैं।

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

आप डेटा सेट के माध्यम से जा सकते हैं और इन आउटलायर्स को हटा सकते हैं, लेकिन इससे डेटा काफी न्यूनतम हो जाएगा।

1. फिलहाल, तय करें कि आप क्लस्टरिंग अभ्यास के लिए कौन से कॉलम का उपयोग करेंगे। समान रेंज वाले कॉलम चुनें और `artist_top_genre` कॉलम को संख्यात्मक डेटा के रूप में एन्कोड करें:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. अब आपको तय करना होगा कि कितने क्लस्टर को लक्षित करना है। आप जानते हैं कि डेटा सेट से हमने 3 गाने की शैलियों को निकाला है, तो चलिए 3 आज़माते हैं:

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

आप एक प्रिंटेड एरे देखते हैं जिसमें डेटा फ्रेम की प्रत्येक पंक्ति के लिए अनुमानित क्लस्टर (0, 1, या 2) होते हैं।

1. इस एरे का उपयोग करके 'सिल्हूट स्कोर' की गणना करें:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## सिल्हूट स्कोर

सिल्हूट स्कोर को 1 के करीब देखें। यह स्कोर -1 से 1 तक भिन्न होता है, और यदि स्कोर 1 है, तो क्लस्टर घना और अन्य क्लस्टर से अच्छी तरह से अलग होता है। 0 के करीब का मान ओवरलैपिंग क्लस्टर को दर्शाता है, जिसमें सैंपल पड़ोसी क्लस्टर के निर्णय सीमा के बहुत करीब होते हैं। [(स्रोत)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

हमारा स्कोर **.53** है, जो बीच में है। यह इंगित करता है कि हमारा डेटा इस प्रकार की क्लस्टरिंग के लिए विशेष रूप से उपयुक्त नहीं है, लेकिन चलिए आगे बढ़ते हैं।

### अभ्यास - मॉडल बनाएं

1. `KMeans` आयात करें और क्लस्टरिंग प्रक्रिया शुरू करें।

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    यहां कुछ हिस्से हैं जिन्हें समझाने की आवश्यकता है।

    > 🎓 रेंज: ये क्लस्टरिंग प्रक्रिया के पुनरावृत्तियां हैं।

    > 🎓 random_state: "सेंट्रॉइड प्रारंभिककरण के लिए रैंडम नंबर जनरेशन निर्धारित करता है।" [स्रोत](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" मापता है कि क्लस्टर के भीतर सभी बिंदुओं की औसत दूरी क्लस्टर सेंट्रॉइड से कितनी है। [स्रोत](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 इनर्शिया: K-Means एल्गोरिदम सेंट्रॉइड्स को चुनने का प्रयास करता है ताकि 'इनर्शिया' को न्यूनतम किया जा सके, "जो मापता है कि क्लस्टर कितने आंतरिक रूप से सुसंगत हैं।" [स्रोत](https://scikit-learn.org/stable/modules/clustering.html)। इस मान को प्रत्येक पुनरावृत्ति पर wcss वेरिएबल में जोड़ा जाता है।

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) में आप 'k-means++' ऑप्टिमाइज़ेशन का उपयोग कर सकते हैं, जो "सेंट्रॉइड्स को (आमतौर पर) एक-दूसरे से दूर प्रारंभिक करता है, जिससे रैंडम प्रारंभिककरण की तुलना में बेहतर परिणाम प्राप्त होते हैं।"

### एल्बो मेथड

पहले, आपने अनुमान लगाया था कि, क्योंकि आपने 3 गाने की शैलियों को लक्षित किया है, आपको 3 क्लस्टर चुनने चाहिए। लेकिन क्या यह सही है?

1. सुनिश्चित करने के लिए 'एल्बो मेथड' का उपयोग करें।

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    पिछले चरण में बनाए गए `wcss` वेरिएबल का उपयोग करके एक चार्ट बनाएं जो दिखाए कि 'एल्बो' में मोड़ कहां है, जो क्लस्टर की इष्टतम संख्या को इंगित करता है। शायद यह **वास्तव में** 3 है!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## अभ्यास - क्लस्टर प्रदर्शित करें

1. प्रक्रिया को फिर से आज़माएं, इस बार तीन क्लस्टर सेट करें, और क्लस्टर को एक स्कैटरप्लॉट के रूप में प्रदर्शित करें:

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

1. मॉडल की सटीकता जांचें:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    इस मॉडल की सटीकता बहुत अच्छी नहीं है, और क्लस्टर के आकार से आपको इसका कारण पता चलता है।

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    यह डेटा बहुत असंतुलित है, बहुत कम सहसंबद्ध है और कॉलम मानों के बीच बहुत अधिक वेरिएंस है, जिससे क्लस्टर अच्छी तरह से नहीं बन पाते। वास्तव में, जो क्लस्टर बनते हैं वे शायद ऊपर परिभाषित तीन शैली श्रेणियों द्वारा भारी रूप से प्रभावित या विकृत होते हैं। यह एक सीखने की प्रक्रिया थी!

    Scikit-learn के दस्तावेज़ में, आप देख सकते हैं कि इस तरह के मॉडल, जिसमें क्लस्टर बहुत अच्छी तरह से परिभाषित नहीं हैं, में 'वेरिएंस' की समस्या होती है:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Scikit-learn से इन्फोग्राफिक

## वेरिएंस

वेरिएंस को "मीन से वर्गीय अंतर का औसत" के रूप में परिभाषित किया गया है [(स्रोत)](https://www.mathsisfun.com/data/standard-deviation.html)। इस क्लस्टरिंग समस्या के संदर्भ में, इसका मतलब है कि हमारे डेटा सेट की संख्याएं मीन से थोड़ा अधिक भटकने की प्रवृत्ति रखती हैं।

✅ यह एक अच्छा समय है यह सोचने का कि आप इस समस्या को ठीक करने के लिए क्या कर सकते हैं। डेटा को थोड़ा और ट्वीक करें? अलग-अलग कॉलम का उपयोग करें? अलग एल्गोरिदम का उपयोग करें? संकेत: अपने डेटा को [स्केल करें](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) ताकि इसे सामान्यीकृत किया जा सके और अन्य कॉलम का परीक्षण किया जा सके।

> इस '[वेरिएंस कैलकुलेटर](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' को आज़माएं ताकि इस अवधारणा को थोड़ा और समझा जा सके।

---

## 🚀चुनौती

इस नोटबुक के साथ कुछ समय बिताएं और पैरामीटर को ट्वीक करें। क्या आप डेटा को अधिक साफ करके (जैसे आउटलायर्स को हटाकर) मॉडल की सटीकता सुधार सकते हैं? आप वेट्स का उपयोग करके कुछ डेटा सैंपल्स को अधिक वेट दे सकते हैं। बेहतर क्लस्टर बनाने के लिए आप और क्या कर सकते हैं?

संकेत: अपने डेटा को स्केल करने का प्रयास करें। नोटबुक में टिप्पणी की गई कोड है जो मानक स्केलिंग जोड़ता है ताकि डेटा कॉलम रेंज के मामले में एक-दूसरे से अधिक समान दिखें। आप पाएंगे कि जबकि सिल्हूट स्कोर नीचे चला जाता है, एल्बो ग्राफ में 'किंक' स्मूथ हो जाता है। ऐसा इसलिए है क्योंकि डेटा को अनस्केल छोड़ने से कम वेरिएंस वाले डेटा को अधिक वेट मिलता है। इस समस्या पर थोड़ा और पढ़ें [यहां](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226)।

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा और स्व-अध्ययन

एक K-Means सिम्युलेटर [जैसे यह](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) देखें। आप इस टूल का उपयोग करके सैंपल डेटा पॉइंट्स को विज़ुअलाइज़ कर सकते हैं और इसके सेंट्रॉइड्स निर्धारित कर सकते हैं। आप डेटा की रैंडमनेस, क्लस्टर की संख्या और सेंट्रॉइड्स की संख्या को संपादित कर सकते हैं। क्या इससे आपको यह समझने में मदद मिलती है कि डेटा को कैसे समूहित किया जा सकता है?

साथ ही, [Stanford से K-Means पर इस हैंडआउट](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) को देखें।

## असाइनमेंट

[अलग-अलग क्लस्टरिंग विधियों को आज़माएं](assignment.md)

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।