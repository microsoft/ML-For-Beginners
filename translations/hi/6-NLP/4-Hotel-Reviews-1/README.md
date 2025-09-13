<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T10:32:20+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "hi"
}
-->
# होटल समीक्षाओं के साथ भावना विश्लेषण - डेटा को संसाधित करना

इस खंड में, आप पिछले पाठों में सीखी गई तकनीकों का उपयोग करके एक बड़े डेटा सेट का अन्वेषणात्मक डेटा विश्लेषण करेंगे। जब आपको विभिन्न कॉलमों की उपयोगिता का अच्छा समझ आ जाएगा, तो आप सीखेंगे:

- अनावश्यक कॉलमों को कैसे हटाएं
- मौजूदा कॉलमों के आधार पर नया डेटा कैसे गणना करें
- अंतिम चुनौती के लिए उपयोग के लिए परिणामी डेटा सेट को कैसे सहेजें

## [प्री-लेक्चर क्विज़](https://ff-quizzes.netlify.app/en/ml/)

### परिचय

अब तक आपने सीखा है कि टेक्स्ट डेटा संख्यात्मक डेटा प्रकारों से काफी अलग होता है। यदि यह टेक्स्ट किसी मानव द्वारा लिखा या बोला गया है, तो इसे पैटर्न और आवृत्तियों, भावना और अर्थ खोजने के लिए विश्लेषित किया जा सकता है। यह पाठ आपको एक वास्तविक डेटा सेट और एक वास्तविक चुनौती में ले जाता है: **[यूरोप में 515K होटल समीक्षाओं का डेटा](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, जिसमें [CC0: सार्वजनिक डोमेन लाइसेंस](https://creativecommons.org/publicdomain/zero/1.0/) शामिल है। इसे Booking.com से सार्वजनिक स्रोतों से स्क्रैप किया गया था। इस डेटा सेट के निर्माता Jiashen Liu हैं।

### तैयारी

आपको आवश्यकता होगी:

* Python 3 का उपयोग करके .ipynb नोटबुक चलाने की क्षमता
* pandas
* NLTK, [जिसे आपको स्थानीय रूप से इंस्टॉल करना चाहिए](https://www.nltk.org/install.html)
* Kaggle पर उपलब्ध डेटा सेट [यूरोप में 515K होटल समीक्षाओं का डेटा](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)। यह अनज़िप करने के बाद लगभग 230 MB है। इसे इन NLP पाठों से जुड़े रूट `/data` फ़ोल्डर में डाउनलोड करें।

## अन्वेषणात्मक डेटा विश्लेषण

यह चुनौती मानती है कि आप भावना विश्लेषण और अतिथि समीक्षाओं के स्कोर का उपयोग करके एक होटल अनुशंसा बॉट बना रहे हैं। जिस डेटा सेट का आप उपयोग करेंगे, उसमें 6 शहरों के 1493 विभिन्न होटलों की समीक्षाएं शामिल हैं।

Python, होटल समीक्षाओं के डेटा सेट, और NLTK के भावना विश्लेषण का उपयोग करके आप पता लगा सकते हैं:

* समीक्षाओं में सबसे अधिक बार उपयोग किए जाने वाले शब्द और वाक्यांश क्या हैं?
* क्या होटल का आधिकारिक *टैग* समीक्षाओं के स्कोर से मेल खाता है (जैसे, क्या *Family with young children* के लिए किसी विशेष होटल की अधिक नकारात्मक समीक्षाएं हैं, जबकि *Solo traveller* के लिए नहीं, शायद यह संकेत देता है कि यह *Solo travellers* के लिए बेहतर है)?
* क्या NLTK के भावना स्कोर होटल समीक्षक के संख्यात्मक स्कोर से 'सहमत' हैं?

#### डेटा सेट

आइए उस डेटा सेट का अन्वेषण करें जिसे आपने डाउनलोड और स्थानीय रूप से सहेजा है। फ़ाइल को किसी संपादक जैसे VS Code या Excel में खोलें।

डेटा सेट में हेडर निम्नलिखित हैं:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

यहां उन्हें इस तरह से समूहित किया गया है जो जांचने में आसान हो सकता है:
##### होटल कॉलम

* `Hotel_Name`, `Hotel_Address`, `lat` (अक्षांश), `lng` (देशांतर)
  * *lat* और *lng* का उपयोग करके आप Python के साथ होटल स्थानों का एक मानचित्र बना सकते हैं (शायद नकारात्मक और सकारात्मक समीक्षाओं के लिए रंग कोडित)
  * Hotel_Address हमारे लिए स्पष्ट रूप से उपयोगी नहीं है, और हम इसे आसान छंटाई और खोज के लिए देश के साथ बदल सकते हैं

**होटल मेटा-समीक्षा कॉलम**

* `Average_Score`
  * डेटा सेट निर्माता के अनुसार, यह कॉलम *होटल का औसत स्कोर है, जो पिछले वर्ष की नवीनतम टिप्पणी के आधार पर गणना किया गया है*। यह स्कोर की गणना करने का असामान्य तरीका लगता है, लेकिन यह स्क्रैप किया गया डेटा है, इसलिए हम इसे फिलहाल मान सकते हैं।

  ✅ इस डेटा में अन्य कॉलमों के आधार पर, क्या आप औसत स्कोर की गणना करने का कोई अन्य तरीका सोच सकते हैं?

* `Total_Number_of_Reviews`
  * इस होटल को प्राप्त समीक्षाओं की कुल संख्या - यह स्पष्ट नहीं है (कोड लिखे बिना) कि यह डेटा सेट में समीक्षाओं को संदर्भित करता है।
* `Additional_Number_of_Scoring`
  * इसका मतलब है कि एक समीक्षा स्कोर दिया गया था लेकिन समीक्षक द्वारा कोई सकारात्मक या नकारात्मक समीक्षा नहीं लिखी गई थी।

**समीक्षा कॉलम**

- `Reviewer_Score`
  - यह एक संख्यात्मक मान है जिसमें अधिकतम 1 दशमलव स्थान है और न्यूनतम और अधिकतम मान 2.5 और 10 के बीच हैं।
  - यह स्पष्ट नहीं किया गया है कि 2.5 सबसे कम संभव स्कोर क्यों है।
- `Negative_Review`
  - यदि समीक्षक ने कुछ नहीं लिखा, तो यह फ़ील्ड "**No Negative**" होगा।
  - ध्यान दें कि समीक्षक नकारात्मक समीक्षा कॉलम में सकारात्मक समीक्षा लिख ​​सकता है (जैसे, "इस होटल के बारे में कुछ भी बुरा नहीं है")
- `Review_Total_Negative_Word_Counts`
  - उच्च नकारात्मक शब्द गणना कम स्कोर का संकेत देती है (भावना की जांच किए बिना)
- `Positive_Review`
  - यदि समीक्षक ने कुछ नहीं लिखा, तो यह फ़ील्ड "**No Positive**" होगा।
  - ध्यान दें कि समीक्षक सकारात्मक समीक्षा कॉलम में नकारात्मक समीक्षा लिख ​​सकता है (जैसे, "इस होटल में कुछ भी अच्छा नहीं है")
- `Review_Total_Positive_Word_Counts`
  - उच्च सकारात्मक शब्द गणना उच्च स्कोर का संकेत देती है (भावना की जांच किए बिना)
- `Review_Date` और `days_since_review`
  - समीक्षा पर ताजगी या पुरानी होने का माप लागू किया जा सकता है (पुरानी समीक्षाएं नई समीक्षाओं जितनी सटीक नहीं हो सकती हैं क्योंकि होटल प्रबंधन बदल गया है, या नवीनीकरण किया गया है, या एक पूल जोड़ा गया है आदि।)
- `Tags`
  - ये छोटे विवरणकर्ता हैं जिन्हें समीक्षक चुन सकता है ताकि वे जिस प्रकार के अतिथि थे (जैसे, अकेले या परिवार), उनके पास किस प्रकार का कमरा था, ठहरने की अवधि और समीक्षा कैसे प्रस्तुत की गई थी, का वर्णन कर सकें।
  - दुर्भाग्यवश, इन टैग्स का उपयोग करना समस्याग्रस्त है, नीचे दिए गए खंड में उनकी उपयोगिता पर चर्चा की गई है।

**समीक्षक कॉलम**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - यह अनुशंसा मॉडल में एक कारक हो सकता है, उदाहरण के लिए, यदि आप यह निर्धारित कर सकते हैं कि सैकड़ों समीक्षाओं वाले अधिक उत्पादक समीक्षक नकारात्मक होने की तुलना में सकारात्मक होने की अधिक संभावना रखते हैं। हालांकि, किसी विशेष समीक्षा के समीक्षक को एक अद्वितीय कोड के साथ पहचाना नहीं गया है, और इसलिए इसे समीक्षाओं के सेट से जोड़ा नहीं जा सकता। 100 या अधिक समीक्षाओं वाले 30 समीक्षक हैं, लेकिन यह देखना मुश्किल है कि यह अनुशंसा मॉडल में कैसे मदद कर सकता है।
- `Reviewer_Nationality`
  - कुछ लोग सोच सकते हैं कि कुछ राष्ट्रीयताएं सकारात्मक या नकारात्मक समीक्षा देने की अधिक संभावना रखती हैं क्योंकि उनकी राष्ट्रीय प्रवृत्ति होती है। अपने मॉडलों में इस तरह के उपाख्यानात्मक विचारों को शामिल करने से सावधान रहें। ये राष्ट्रीय (और कभी-कभी नस्लीय) रूढ़ियाँ हैं, और प्रत्येक समीक्षक एक व्यक्ति था जिसने अपने अनुभव के आधार पर समीक्षा लिखी। इसे कई दृष्टिकोणों से फ़िल्टर किया जा सकता है जैसे उनके पिछले होटल ठहराव, यात्रा की दूरी, और उनका व्यक्तिगत स्वभाव। यह सोचना कि उनकी राष्ट्रीयता समीक्षा स्कोर का कारण थी, न्यायसंगत ठहराना कठिन है।

##### उदाहरण

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | यह वर्तमान में एक होटल नहीं है बल्कि एक निर्माण स्थल है। मुझे सुबह जल्दी और पूरे दिन अस्वीकार्य निर्माण शोर से परेशान किया गया था, जबकि लंबी यात्रा के बाद आराम कर रहा था और कमरे में काम कर रहा था। लोग पूरे दिन काम कर रहे थे, जैसे कि जैकहैमर के साथ। मैंने कमरे बदलने के लिए कहा लेकिन कोई शांत कमरा उपलब्ध नहीं था। इसे और खराब करने के लिए, मुझसे अधिक शुल्क लिया गया। मैंने शाम को चेक आउट किया क्योंकि मुझे बहुत जल्दी उड़ान लेनी थी और मुझे उचित बिल मिला। एक दिन बाद, होटल ने मेरी सहमति के बिना बुक की गई कीमत से अधिक शुल्क लिया। यह एक भयानक जगह है। खुद को यहां बुक करके सजा न दें। | कुछ भी नहीं। भयानक जगह। दूर रहें। | व्यवसाय यात्रा। जोड़ा। मानक डबल कमरा। 2 रातें रुके। |

जैसा कि आप देख सकते हैं, इस अतिथि का इस होटल में सुखद अनुभव नहीं था। होटल का औसत स्कोर 7.8 और 1945 समीक्षाएं हैं, लेकिन इस समीक्षक ने इसे 2.5 दिया और 115 शब्द लिखे कि उनका अनुभव कितना नकारात्मक था। यदि उन्होंने सकारात्मक समीक्षा कॉलम में कुछ भी नहीं लिखा, तो आप अनुमान लगा सकते हैं कि कुछ भी सकारात्मक नहीं था, लेकिन उन्होंने 7 शब्दों की चेतावनी लिखी। यदि हम केवल शब्दों की गिनती करते हैं बजाय उनके अर्थ या शब्दों की भावना के, तो हमें समीक्षक के इरादे का विकृत दृष्टिकोण मिल सकता है। अजीब बात है, उनका स्कोर 2.5 भ्रमित करने वाला है, क्योंकि अगर वह होटल ठहराव इतना खराब था, तो उन्होंने इसे कोई अंक क्यों दिए? डेटा सेट की बारीकी से जांच करते हुए, आप देखेंगे कि सबसे कम संभव स्कोर 2.5 है, 0 नहीं। सबसे अधिक संभव स्कोर 10 है।

##### टैग्स

जैसा कि ऊपर उल्लेख किया गया है, पहली नज़र में, डेटा को वर्गीकृत करने के लिए `Tags` का उपयोग करने का विचार समझ में आता है। दुर्भाग्यवश, ये टैग्स मानकीकृत नहीं हैं, जिसका मतलब है कि एक दिए गए होटल में विकल्प *Single room*, *Twin room*, और *Double room* हो सकते हैं, लेकिन अगले होटल में वे *Deluxe Single Room*, *Classic Queen Room*, और *Executive King Room* हो सकते हैं। ये वही चीजें हो सकती हैं, लेकिन इतने सारे बदलाव हैं कि विकल्प बन जाता है:

1. सभी शब्दों को एक मानक में बदलने का प्रयास करें, जो बहुत कठिन है, क्योंकि यह स्पष्ट नहीं है कि प्रत्येक मामले में रूपांतरण पथ क्या होगा (जैसे, *Classic single room* को *Single room* में मैप किया जा सकता है लेकिन *Superior Queen Room with Courtyard Garden or City View* को मैप करना बहुत कठिन है)

1. हम एक NLP दृष्टिकोण ले सकते हैं और कुछ शब्दों जैसे *Solo*, *Business Traveller*, या *Family with young kids* की आवृत्ति को माप सकते हैं क्योंकि वे प्रत्येक होटल पर लागू होते हैं, और इसे अनुशंसा में शामिल कर सकते हैं।

टैग्स आमतौर पर (लेकिन हमेशा नहीं) एकल फ़ील्ड होते हैं जिसमें 5 से 6 अल्पविराम से अलग किए गए मान होते हैं जो *यात्रा का प्रकार*, *अतिथि का प्रकार*, *कमरे का प्रकार*, *रातों की संख्या*, और *जिस डिवाइस पर समीक्षा प्रस्तुत की गई थी* के साथ मेल खाते हैं। हालांकि, क्योंकि कुछ समीक्षक प्रत्येक फ़ील्ड को नहीं भरते (वे एक को खाली छोड़ सकते हैं), मान हमेशा एक ही क्रम में नहीं होते।

उदाहरण के लिए, *Type of group* लें। इस फ़ील्ड में `Tags` कॉलम में 1025 अद्वितीय संभावनाएं हैं, और दुर्भाग्यवश उनमें से केवल कुछ ही समूह को संदर्भित करती हैं (कुछ कमरे के प्रकार आदि हैं)। यदि आप केवल उन लोगों को फ़िल्टर करते हैं जो परिवार का उल्लेख करते हैं, तो परिणामों में कई *Family room* प्रकार के परिणाम होते हैं। यदि आप शब्द *with* को शामिल करते हैं, यानी *Family with* मानों की गणना करते हैं, तो परिणाम बेहतर होते हैं, जिसमें "Family with young children" या "Family with older children" वाक्यांश वाले 515,000 परिणामों में से 80,000 से अधिक शामिल होते हैं।

इसका मतलब है कि टैग्स कॉलम हमारे लिए पूरी तरह से बेकार नहीं है, लेकिन इसे उपयोगी बनाने में कुछ काम लगेगा।

##### औसत होटल स्कोर

डेटा सेट में कुछ विचित्रताएं या विसंगतियां हैं जिन्हें मैं समझ नहीं पा रहा हूं, लेकिन यहां उन्हें चित्रित किया गया है ताकि आप अपने मॉडल बनाते समय उनसे अवगत रहें। यदि आप इसे समझते हैं, तो कृपया चर्चा अनुभाग में हमें बताएं!

डेटा सेट में औसत स्कोर और समीक्षाओं की संख्या से संबंधित निम्नलिखित कॉलम हैं:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

इस डेटा सेट में सबसे अधिक समीक्षाओं वाला एकल होटल *Britannia International Hotel Canary Wharf* है, जिसमें 515,000 में से 4789 समीक्षाएं हैं। लेकिन अगर हम इस होटल के लिए `Total_Number_of_Reviews` मान को देखें, तो यह 9086 है। आप अनुमान लगा सकते हैं कि कई और स्कोर बिना समीक्षाओं के हैं, इसलिए शायद हमें `Additional_Number_of_Scoring` कॉलम मान जोड़ना चाहिए। वह मान 2682 है, और इसे 4789 में जोड़ने से हमें 7471 मिलता है, जो अभी भी `Total_Number_of_Reviews` से 1615 कम है।

यदि आप `Average_Score` कॉलम लेते हैं, तो आप अनुमान लगा सकते हैं कि यह डेटा सेट में समीक्षाओं का औसत है, लेकिन Kaggle का विवरण है "*होटल का औसत स्कोर, पिछले वर्ष की नवीनतम टिप्पणी के आधार पर गणना किया गया*।" यह बहुत उपयोगी नहीं लगता है, लेकिन हम डेटा सेट में समीक्षाओं के स्कोर के आधार पर अपना औसत गणना कर सकते हैं। उसी होटल को उदाहरण के रूप में लेते हुए, औसत होटल स्कोर 7.1 दिया गया है लेकिन गणना किया गया स्कोर (समीक्षक का औसत स्कोर *डेटा सेट में*) 6.8 है। यह करीब है, लेकिन समान मान नहीं है, और हम केवल अनुमान लगा सकते हैं कि `Additional_Number_of_Scoring` समीक्षाओं में दिए गए स्कोर ने औसत को 7.1 तक बढ़ा दिया। दुर्भाग्यवश, उस कथन का परीक्षण या प्रमाणित करने का कोई तरीका नहीं होने के कारण, `Average_Score`, `Additional_Number_of_Scoring` और `Total_Number_of_Reviews` का उपयोग करना या उन पर भरोसा करना कठिन है जब वे डेटा पर आधारित हैं या डेटा का संदर्भ देते हैं जो हमारे पास नहीं है।

चीजों को और जटिल करने के लिए, सबसे अधिक समीक्षाओं वाले दूसरे होटल का गणना किया गया औसत स्कोर 8.12 है और डेटा सेट `Average_Score` 8.1 है। क्या यह सही स्कोर एक संयोग है या पहला होटल एक विसंगति है?

इस संभावना पर कि ये होटल एक अपवाद हो सकते हैं, और शायद अधिकांश मान सही हैं (लेकिन कुछ किसी कारण से सही नहीं हैं), हम अगले भाग में डेटा सेट में मानों का अन्वेषण करने और मानों के सही उपयोग (या गैर-उपयोग) को निर्धारित करने के लिए एक छोटा प्रोग्राम लिखेंगे।
> 🚨 सावधानी का एक नोट

> जब आप इस डेटा सेट के साथ काम कर रहे हों, तो आप ऐसा कोड लिखेंगे जो टेक्स्ट से कुछ गणना करता है बिना टेक्स्ट को खुद पढ़े या उसका विश्लेषण किए। यही NLP का सार है, अर्थ या भावना को समझना बिना किसी इंसान के इसे करने की आवश्यकता के। हालांकि, यह संभव है कि आप कुछ नकारात्मक समीक्षाएँ पढ़ लें। मैं आपसे आग्रह करूंगा कि ऐसा न करें, क्योंकि इसकी कोई आवश्यकता नहीं है। इनमें से कुछ समीक्षाएँ मूर्खतापूर्ण या अप्रासंगिक नकारात्मक होटल समीक्षाएँ हो सकती हैं, जैसे "मौसम अच्छा नहीं था", जो होटल या किसी के नियंत्रण से बाहर है। लेकिन कुछ समीक्षाओं का एक अंधकारमय पक्ष भी होता है। कभी-कभी नकारात्मक समीक्षाएँ नस्लवादी, लिंगभेदी, या उम्रभेदी होती हैं। यह दुर्भाग्यपूर्ण है लेकिन एक सार्वजनिक वेबसाइट से स्क्रैप किए गए डेटा सेट में अपेक्षित है। कुछ समीक्षक ऐसी समीक्षाएँ छोड़ते हैं जो आपको अप्रिय, असहज, या परेशान कर सकती हैं। बेहतर होगा कि कोड भावना को मापे बजाय आप खुद उन्हें पढ़कर परेशान हों। यह कहा जा सकता है कि ऐसे लोग अल्पसंख्यक हैं जो ऐसा लिखते हैं, लेकिन वे फिर भी मौजूद हैं।
## अभ्यास - डेटा अन्वेषण
### डेटा लोड करें

अब डेटा को विजुअली जांचने के लिए पर्याप्त है, अब आप कुछ कोड लिखेंगे और उत्तर प्राप्त करेंगे! इस सेक्शन में pandas लाइब्रेरी का उपयोग किया गया है। आपका पहला कार्य यह सुनिश्चित करना है कि आप CSV डेटा को लोड और पढ़ सकते हैं। pandas लाइब्रेरी में एक तेज़ CSV लोडर है, और परिणाम को एक डेटा फ्रेम में रखा जाता है, जैसा कि पिछले पाठों में देखा गया है। जिस CSV को हम लोड कर रहे हैं उसमें आधे मिलियन से अधिक पंक्तियाँ हैं, लेकिन केवल 17 कॉलम हैं। pandas आपको डेटा फ्रेम के साथ इंटरैक्ट करने के कई शक्तिशाली तरीके देता है, जिसमें हर पंक्ति पर ऑपरेशन करने की क्षमता भी शामिल है।

इस पाठ के आगे के हिस्से में कोड स्निपेट्स होंगे, कोड की व्याख्या होगी और परिणामों का अर्थ समझाने पर चर्चा होगी। अपने कोड के लिए शामिल _notebook.ipynb_ का उपयोग करें।

आइए उस डेटा फ़ाइल को लोड करने से शुरू करें जिसे आप उपयोग करेंगे:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

अब जब डेटा लोड हो गया है, तो हम इस पर कुछ ऑपरेशन कर सकते हैं। अगले भाग के लिए इस कोड को अपने प्रोग्राम के शीर्ष पर रखें।

## डेटा का अन्वेषण करें

इस मामले में, डेटा पहले से ही *साफ* है, इसका मतलब है कि यह काम करने के लिए तैयार है और इसमें अन्य भाषाओं के ऐसे अक्षर नहीं हैं जो केवल अंग्रेजी अक्षरों की अपेक्षा करने वाले एल्गोरिदम को भ्रमित कर सकते हैं।

✅ आपको ऐसे डेटा के साथ काम करना पड़ सकता है जिसे NLP तकनीकों को लागू करने से पहले प्रारंभिक प्रोसेसिंग की आवश्यकता हो, लेकिन इस बार ऐसा नहीं है। यदि आपको करना पड़े, तो आप गैर-अंग्रेजी अक्षरों को कैसे संभालेंगे?

एक पल लें और सुनिश्चित करें कि डेटा लोड होने के बाद आप इसे कोड के साथ एक्सप्लोर कर सकते हैं। `Negative_Review` और `Positive_Review` कॉलम पर ध्यान केंद्रित करना बहुत आसान है। ये कॉलम आपके NLP एल्गोरिदम के लिए प्राकृतिक टेक्स्ट से भरे हुए हैं। लेकिन रुको! NLP और सेंटिमेंट में कूदने से पहले, आपको नीचे दिए गए कोड का पालन करना चाहिए ताकि यह सुनिश्चित किया जा सके कि डेटा सेट में दिए गए मान आपके द्वारा pandas के साथ गणना किए गए मानों से मेल खाते हैं।

## डेटा फ्रेम ऑपरेशन

इस पाठ में पहला कार्य यह जांचना है कि निम्नलिखित दावे सही हैं या नहीं, इसके लिए डेटा फ्रेम की जांच करने वाला कोड लिखें (बिना इसे बदले)।

> कई प्रोग्रामिंग कार्यों की तरह, इसे पूरा करने के कई तरीके हैं, लेकिन अच्छा सुझाव यह है कि इसे सबसे सरल और आसान तरीके से करें, खासकर यदि यह भविष्य में इस कोड को समझने में आसान होगा। डेटा फ्रेम के साथ, एक व्यापक API है जो अक्सर आपके काम को कुशलतापूर्वक करने का तरीका प्रदान करता है।

निम्नलिखित प्रश्नों को कोडिंग कार्यों के रूप में मानें और समाधान को देखे बिना उत्तर देने का प्रयास करें।

1. आपने अभी लोड किए गए डेटा फ्रेम का *आकार* प्रिंट करें (आकार पंक्तियों और कॉलम की संख्या है)।
2. समीक्षक राष्ट्रीयताओं के लिए आवृत्ति गणना करें:
   1. `Reviewer_Nationality` कॉलम के लिए कितने अलग-अलग मान हैं और वे क्या हैं?
   2. डेटा सेट में सबसे आम समीक्षक राष्ट्रीयता कौन सी है (देश और समीक्षाओं की संख्या प्रिंट करें)?
   3. अगली शीर्ष 10 सबसे अधिक बार पाई जाने वाली राष्ट्रीयताओं और उनकी आवृत्ति गणना क्या हैं?
3. शीर्ष 10 समीक्षक राष्ट्रीयताओं में से प्रत्येक के लिए सबसे अधिक समीक्षा किया गया होटल कौन सा था?
4. डेटा सेट में प्रति होटल कितनी समीक्षाएँ हैं (होटल की आवृत्ति गणना)?
5. जबकि डेटा सेट में प्रत्येक होटल के लिए `Average_Score` कॉलम है, आप एक औसत स्कोर भी गणना कर सकते हैं (डेटा सेट में प्रत्येक होटल के लिए सभी समीक्षक स्कोर का औसत प्राप्त करना)। अपने डेटा फ्रेम में एक नया कॉलम जोड़ें जिसका कॉलम हेडर `Calc_Average_Score` हो जिसमें वह गणना किया गया औसत हो।
6. क्या किसी होटल का `Average_Score` और `Calc_Average_Score` (1 दशमलव स्थान तक गोल) समान है?
   1. एक Python फ़ंक्शन लिखने का प्रयास करें जो एक Series (पंक्ति) को तर्क के रूप में लेता है और मानों की तुलना करता है, जब मान समान नहीं होते हैं तो एक संदेश प्रिंट करता है। फिर `.apply()` विधि का उपयोग करके हर पंक्ति को फ़ंक्शन के साथ प्रोसेस करें।
7. `Negative_Review` कॉलम में "No Negative" मान वाले कितने पंक्तियाँ हैं, इसकी गणना और प्रिंट करें।
8. `Positive_Review` कॉलम में "No Positive" मान वाले कितने पंक्तियाँ हैं, इसकी गणना और प्रिंट करें।
9. `Positive_Review` कॉलम में "No Positive" **और** `Negative_Review` कॉलम में "No Negative" मान वाले कितने पंक्तियाँ हैं, इसकी गणना और प्रिंट करें।

### कोड उत्तर

1. आपने अभी लोड किए गए डेटा फ्रेम का *आकार* प्रिंट करें (आकार पंक्तियों और कॉलम की संख्या है)।

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. समीक्षक राष्ट्रीयताओं के लिए आवृत्ति गणना करें:

   1. `Reviewer_Nationality` कॉलम के लिए कितने अलग-अलग मान हैं और वे क्या हैं?
   2. डेटा सेट में सबसे आम समीक्षक राष्ट्रीयता कौन सी है (देश और समीक्षाओं की संख्या प्रिंट करें)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. अगली शीर्ष 10 सबसे अधिक बार पाई जाने वाली राष्ट्रीयताओं और उनकी आवृत्ति गणना क्या हैं?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. शीर्ष 10 समीक्षक राष्ट्रीयताओं में से प्रत्येक के लिए सबसे अधिक समीक्षा किया गया होटल कौन सा था?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. डेटा सेट में प्रति होटल कितनी समीक्षाएँ हैं (होटल की आवृत्ति गणना)?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   आप देख सकते हैं कि *डेटा सेट में गिना गया* परिणाम `Total_Number_of_Reviews` मान से मेल नहीं खाता। यह स्पष्ट नहीं है कि डेटा सेट में यह मान होटल की कुल समीक्षाओं का प्रतिनिधित्व करता है, लेकिन सभी स्क्रैप नहीं किए गए थे, या कुछ अन्य गणना। इस अस्पष्टता के कारण मॉडल में `Total_Number_of_Reviews` का उपयोग नहीं किया गया है।

5. जबकि डेटा सेट में प्रत्येक होटल के लिए `Average_Score` कॉलम है, आप एक औसत स्कोर भी गणना कर सकते हैं (डेटा सेट में प्रत्येक होटल के लिए सभी समीक्षक स्कोर का औसत प्राप्त करना)। अपने डेटा फ्रेम में एक नया कॉलम जोड़ें जिसका कॉलम हेडर `Calc_Average_Score` हो जिसमें वह गणना किया गया औसत हो। कॉलम `Hotel_Name`, `Average_Score`, और `Calc_Average_Score` प्रिंट करें।

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   आप यह भी सोच सकते हैं कि `Average_Score` मान और गणना किए गए औसत स्कोर में कभी-कभी अंतर क्यों होता है। जैसा कि हम नहीं जान सकते कि कुछ मान क्यों मेल खाते हैं, लेकिन अन्य में अंतर है, इस मामले में यह सबसे सुरक्षित है कि हम अपने पास मौजूद समीक्षा स्कोर का उपयोग करके औसत स्वयं गणना करें। हालांकि, अंतर आमतौर पर बहुत छोटा होता है, यहाँ डेटा सेट औसत और गणना किए गए औसत के बीच सबसे बड़ा विचलन वाले होटल हैं:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   केवल 1 होटल का स्कोर में 1 से अधिक का अंतर है, इसका मतलब है कि हम शायद अंतर को अनदेखा कर सकते हैं और गणना किए गए औसत स्कोर का उपयोग कर सकते हैं।

6. `Negative_Review` कॉलम में "No Negative" मान वाले कितने पंक्तियाँ हैं, इसकी गणना और प्रिंट करें।

7. `Positive_Review` कॉलम में "No Positive" मान वाले कितने पंक्तियाँ हैं, इसकी गणना और प्रिंट करें।

8. `Positive_Review` कॉलम में "No Positive" **और** `Negative_Review` कॉलम में "No Negative" मान वाले कितने पंक्तियाँ हैं, इसकी गणना और प्रिंट करें।

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## एक और तरीका

लंब्डा का उपयोग किए बिना आइटम गिनने का एक और तरीका, और पंक्तियों को गिनने के लिए sum का उपयोग करें:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   आपने देखा होगा कि `Negative_Review` और `Positive_Review` कॉलम में "No Negative" और "No Positive" मान वाले 127 पंक्तियाँ हैं। इसका मतलब है कि समीक्षक ने होटल को एक संख्यात्मक स्कोर दिया, लेकिन सकारात्मक या नकारात्मक समीक्षा लिखने से इनकार कर दिया। सौभाग्य से यह पंक्तियों की एक छोटी संख्या है (127 में से 515738, या 0.02%), इसलिए यह शायद हमारे मॉडल या परिणामों को किसी विशेष दिशा में प्रभावित नहीं करेगा, लेकिन आप उम्मीद नहीं कर सकते थे कि समीक्षाओं के डेटा सेट में ऐसी पंक्तियाँ हों जिनमें कोई समीक्षा न हो, इसलिए डेटा का अन्वेषण करना और ऐसी पंक्तियों की खोज करना महत्वपूर्ण है।

अब जब आपने डेटा सेट का अन्वेषण कर लिया है, अगले पाठ में आप डेटा को फ़िल्टर करेंगे और कुछ सेंटिमेंट एनालिसिस जोड़ेंगे।

---
## 🚀चुनौती

यह पाठ दिखाता है, जैसा कि हमने पिछले पाठों में देखा, कि आपके डेटा और इसकी विशेषताओं को समझना कितना महत्वपूर्ण है इससे पहले कि आप उस पर ऑपरेशन करें। टेक्स्ट-आधारित डेटा, विशेष रूप से, सावधानीपूर्वक जांच की आवश्यकता होती है। विभिन्न टेक्स्ट-भारी डेटा सेट्स को खंगालें और देखें कि क्या आप ऐसे क्षेत्र खोज सकते हैं जो मॉडल में पूर्वाग्रह या विकृत सेंटिमेंट ला सकते हैं।

## [पाठ के बाद क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा और स्व-अध्ययन

[NLP पर यह लर्निंग पाथ](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) लें ताकि आप भाषण और टेक्स्ट-भारी मॉडल बनाते समय आज़माने के लिए टूल्स की खोज कर सकें।

## असाइनमेंट 

[NLTK](assignment.md)

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  