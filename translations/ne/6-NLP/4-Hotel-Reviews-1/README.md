<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-06T06:41:07+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "ne"
}
-->
# होटल समीक्षाको भावना विश्लेषण - डेटा प्रशोधन

यस खण्डमा, तपाईंले अघिल्लो पाठहरूमा सिकेका प्रविधिहरू प्रयोग गरेर ठूलो डेटासेटको अन्वेषणात्मक डेटा विश्लेषण गर्नुहुनेछ। विभिन्न स्तम्भहरूको उपयोगिताको राम्रो समझ प्राप्त गरेपछि, तपाईं सिक्नुहुनेछ:

- अनावश्यक स्तम्भहरू कसरी हटाउने
- विद्यमान स्तम्भहरूमा आधारित नयाँ डेटा कसरी गणना गर्ने
- अन्तिम चुनौतीको लागि परिणामी डेटासेट कसरी बचत गर्ने

## [पाठ अघि क्विज](https://ff-quizzes.netlify.app/en/ml/)

### परिचय

अहिलेसम्म तपाईंले पाठ डेटा संख्यात्मक प्रकारको डेटासँग कति फरक छ भनेर सिक्नुभएको छ। यदि यो पाठ मानिसद्वारा लेखिएको वा बोलिएको हो भने, यसलाई ढाँचा र आवृत्ति, भावना र अर्थ पत्ता लगाउन विश्लेषण गर्न सकिन्छ। यो पाठले तपाईंलाई वास्तविक चुनौतीसहितको वास्तविक डेटासेटमा लैजान्छ: **[युरोपमा ५१५ हजार होटल समीक्षाको डेटा](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** जसमा [CC0: सार्वजनिक डोमेन लाइसेन्स](https://creativecommons.org/publicdomain/zero/1.0/) समावेश छ। यो सार्वजनिक स्रोतबाट Booking.com बाट स्क्र्याप गरिएको थियो। डेटासेटको सिर्जनाकर्ता Jiashen Liu हुन्।

### तयारी

तपाईंलाई आवश्यक पर्नेछ:

* Python 3 प्रयोग गरेर .ipynb नोटबुकहरू चलाउने क्षमता
* pandas
* NLTK, [स्थानीय रूपमा स्थापना गर्नुपर्ने](https://www.nltk.org/install.html)
* Kaggle मा उपलब्ध डेटासेट [युरोपमा ५१५ हजार होटल समीक्षाको डेटा](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)। यो अनजिप गरेपछि लगभग २३० MB छ। यसलाई यी NLP पाठहरूसँग सम्बन्धित `/data` फोल्डरमा डाउनलोड गर्नुहोस्।

## अन्वेषणात्मक डेटा विश्लेषण

यो चुनौतीले मानिन्छ कि तपाईं भावना विश्लेषण र अतिथि समीक्षाको स्कोर प्रयोग गरेर होटल सिफारिस बोट निर्माण गर्दै हुनुहुन्छ। तपाईंले प्रयोग गर्ने डेटासेटमा ६ शहरका १४९३ विभिन्न होटलहरूको समीक्षा समावेश छ।

Python, होटल समीक्षाको डेटासेट, र NLTK को भावना विश्लेषण प्रयोग गरेर तपाईं पत्ता लगाउन सक्नुहुन्छ:

* समीक्षामा सबैभन्दा धेरै प्रयोग गरिएका शब्द र वाक्यांशहरू के हुन्?
* होटललाई वर्णन गर्ने आधिकारिक *ट्यागहरू* समीक्षाको स्कोरसँग सम्बन्धित छन् कि छैनन् (जस्तै, *Family with young children* को लागि नकारात्मक समीक्षाहरू *Solo traveller* को तुलनामा बढी छन् कि छैनन्, जसले संकेत दिन सक्छ कि यो *Solo travellers* को लागि राम्रो हो)?
* NLTK को भावना स्कोर होटल समीक्षकको संख्यात्मक स्कोरसँग 'सहमत' छ कि छैन?

#### डेटासेट

आउनुहोस्, तपाईंले डाउनलोड गरेर स्थानीय रूपमा बचत गरेको डेटासेट अन्वेषण गरौं। फाइललाई VS Code जस्ता सम्पादकमा वा Excel मा खोल्नुहोस्।

डेटासेटका हेडरहरू निम्न छन्:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

यीलाई जाँच गर्न सजिलो बनाउन समूहबद्ध गरिएको छ:
##### होटल स्तम्भहरू

* `Hotel_Name`, `Hotel_Address`, `lat` (अक्षांश), `lng` (देशान्तर)
  * *lat* र *lng* प्रयोग गरेर तपाईं Python मा होटल स्थानहरू देखाउने नक्सा बनाउन सक्नुहुन्छ (शायद नकारात्मक र सकारात्मक समीक्षाहरूको लागि रंग कोड गरिएको)
  * Hotel_Address हाम्रो लागि स्पष्ट रूपमा उपयोगी छैन, र हामी सम्भवतः देशसँग प्रतिस्थापन गर्नेछौं ताकि सर्टिङ र खोज्न सजिलो होस्।

**होटल मेटा-समीक्षा स्तम्भहरू**

* `Average_Score`
  * डेटासेट सिर्जनाकर्ताका अनुसार, यो स्तम्भ *होटलको औसत स्कोर हो, पछिल्लो वर्षको पछिल्लो टिप्पणीमा आधारित गणना गरिएको*। यो स्कोर गणना गर्ने असामान्य तरिका जस्तो देखिन्छ, तर यो स्क्र्याप गरिएको डेटा हो, त्यसैले हामी यसलाई हाललाई स्वीकार गर्न सक्छौं।

  ✅ यस डेटामा भएका अन्य स्तम्भहरूमा आधारित औसत स्कोर गणना गर्ने अर्को तरिका सोच्न सक्नुहुन्छ?

* `Total_Number_of_Reviews`
  * यो होटलले प्राप्त गरेको समीक्षाहरूको कुल संख्या हो - यो स्पष्ट छैन (केही कोड लेख्न बिना) कि यो डेटासेटमा भएका समीक्षाहरूलाई जनाउँछ कि जनाउँदैन।
* `Additional_Number_of_Scoring`
  * यसको मतलब समीक्षकले स्कोर दिएको छ तर सकारात्मक वा नकारात्मक समीक्षा लेखेको छैन।

**समीक्षा स्तम्भहरू**

- `Reviewer_Score`
  - यो संख्यात्मक मान हो जसमा अधिकतम १ दशमलव स्थान छ, र यसको न्यूनतम र अधिकतम मान २.५ र १० बीचमा छ।
  - किन २.५ न्यूनतम स्कोर हो भनेर स्पष्ट गरिएको छैन।
- `Negative_Review`
  - यदि समीक्षकले केही लेखेन भने, यो क्षेत्रमा "**No Negative**" हुनेछ।
  - ध्यान दिनुहोस् कि समीक्षकले नकारात्मक समीक्षा स्तम्भमा सकारात्मक समीक्षा लेख्न सक्छ (जस्तै, "यो होटलमा केही नराम्रो छैन")।
- `Review_Total_Negative_Word_Counts`
  - उच्च नकारात्मक शब्द गणनाले कम स्कोर संकेत गर्दछ (भावनात्मकता जाँच नगरीकन)।
- `Positive_Review`
  - यदि समीक्षकले केही लेखेन भने, यो क्षेत्रमा "**No Positive**" हुनेछ।
  - ध्यान दिनुहोस् कि समीक्षकले सकारात्मक समीक्षा स्तम्भमा नकारात्मक समीक्षा लेख्न सक्छ (जस्तै, "यो होटलमा केही राम्रो छैन")।
- `Review_Total_Positive_Word_Counts`
  - उच्च सकारात्मक शब्द गणनाले उच्च स्कोर संकेत गर्दछ (भावनात्मकता जाँच नगरीकन)।
- `Review_Date` र `days_since_review`
  - समीक्षामा ताजगी वा पुरानोपनको मापन लागू गर्न सकिन्छ (पुराना समीक्षाहरू नयाँ समीक्षाहरू जत्तिकै सटीक नहुन सक्छन् किनभने होटल व्यवस्थापन परिवर्तन भएको छ, वा नवीकरण गरिएको छ, वा पोखरी थपिएको छ आदि)।
- `Tags`
  - यी छोटो वर्णनात्मक शब्दहरू हुन् जुन समीक्षकले आफूलाई वर्णन गर्न चयन गर्न सक्छ (जस्तै, एक्लो वा परिवार), उनीहरूको कोठाको प्रकार, बसाइको अवधि, र समीक्षा कसरी प्रस्तुत गरिएको थियो।
  - दुर्भाग्यवश, यी ट्यागहरूको उपयोगिता समस्याग्रस्त छ, तलको खण्डमा यसको उपयोगिताको चर्चा गरिएको छ।

**समीक्षक स्तम्भहरू**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - यो सिफारिस मोडेलमा कारक हुन सक्छ, उदाहरणका लागि, यदि तपाईं निर्धारण गर्न सक्नुहुन्छ कि सयौं समीक्षाहरू भएका अधिक उत्पादक समीक्षकहरू नकारात्मक भन्दा सकारात्मक हुने सम्भावना बढी छ। तर, कुनै विशेष समीक्षाको समीक्षकलाई अद्वितीय कोडसँग पहिचान गर्न सकिँदैन, र त्यसैले समीक्षाहरूको सेटसँग लिंक गर्न सकिँदैन। १०० वा बढी समीक्षाहरू भएका ३० समीक्षकहरू छन्, तर यो सिफारिस मोडेलमा कसरी सहयोग गर्न सक्छ भन्ने देख्न गाह्रो छ।
- `Reviewer_Nationality`
  - केही व्यक्तिहरू सोच्न सक्छन् कि निश्चित राष्ट्रियताहरू सकारात्मक वा नकारात्मक समीक्षा दिन अधिक सम्भावना राख्छन्। तर, यस्तो कथनलाई मोडेलमा समावेश गर्दा सावधान रहनुहोस्। यी राष्ट्रिय (र कहिलेकाहीं जातीय) स्टीरियोटाइपहरू हुन्, र प्रत्येक समीक्षकले आफ्नो अनुभवको आधारमा समीक्षा लेखेका थिए। यो धेरै लेंसहरूबाट फिल्टर गरिएको हुन सक्छ, जस्तै उनीहरूको अघिल्लो होटल बसाइ, यात्रा गरिएको दूरी, र उनीहरूको व्यक्तिगत स्वभाव। समीक्षा स्कोरको कारण उनीहरूको राष्ट्रियता हो भन्ने सोच्न गाह्रो छ।

##### उदाहरणहरू

| औसत स्कोर | समीक्षाहरूको कुल संख्या | समीक्षक स्कोर | नकारात्मक समीक्षा                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | सकारात्मक समीक्षा                 | ट्यागहरू                                                                                      |
| ----------- | ------------------------ | -------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| ७.८         | १९४५                    | २.५            | यो हाल होटल होइन तर निर्माण स्थल हो। म लामो यात्रापछि आराम गरिरहेको बेला र कोठामा काम गरिरहेको बेला बिहानैदेखि र दिनभरि असह्य निर्माणको आवाजले आतंकित भएँ। मानिसहरू दिनभरि काम गरिरहेका थिए। मैले कोठा परिवर्तनको अनुरोध गरेँ तर कुनै शान्त कोठा उपलब्ध थिएन। स्थिति झनै खराब बनाउन, मलाई बढी शुल्क लगाइयो। मैले साँझमा चेक आउट गरेँ किनभने मलाई बिहानै उडान लिनु थियो। एक दिन पछि, होटलले मेरो सहमति बिना बुक गरिएको मूल्यभन्दा बढी शुल्क लगायो। यो एक भयानक ठाउँ हो। यहाँ बुक गरेर आफूलाई दण्ड नदिनुहोस्। | केही छैन। भयानक ठाउँ। टाढा रहनुहोस्। | व्यापार यात्रा                                जोडी मानक डबल कोठा। २ रात बसे। |

जस्तो कि तपाईं देख्न सक्नुहुन्छ, यो अतिथिले होटलमा खुसी बसाइको अनुभव गरेन। होटलको औसत स्कोर ७.८ छ र १९४५ समीक्षाहरू छन्, तर यस समीक्षकले यसलाई २.५ दिएको छ र आफ्नो बसाइ कति नकारात्मक थियो भनेर ११५ शब्द लेखेका छन्। यदि उनले सकारात्मक समीक्षा स्तम्भमा केही पनि लेखेनन् भने, तपाईं अनुमान गर्न सक्नुहुन्छ कि त्यहाँ केही सकारात्मक थिएन। तर, उनले चेतावनीका रूपमा ७ शब्द लेखेका छन्। यदि हामी शब्दहरूको गणना मात्र गर्छौं र शब्दहरूको अर्थ वा भावना गणना गर्दैनौं भने, हामी समीक्षकको उद्देश्यको गलत दृष्टिकोण राख्न सक्छौं। अचम्मको कुरा, उनीहरूको स्कोर २.५ भ्रमित छ, किनभने यदि त्यो होटल बसाइ यति खराब थियो भने, किन कुनै पनि अंक दिनु? डेटासेटलाई नजिकबाट जाँच गर्दा, तपाईं देख्नुहुनेछ कि न्यूनतम सम्भावित स्कोर २.५ हो, ० होइन। अधिकतम सम्भावित स्कोर १० हो।

##### ट्यागहरू

जस्तो कि माथि उल्लेख गरिएको छ, पहिलो नजरमा, `Tags` प्रयोग गरेर डेटा वर्गीकरण गर्ने विचार अर्थपूर्ण देखिन्छ। दुर्भाग्यवश, यी ट्यागहरू मानकीकृत छैनन्, जसको अर्थ एक होटलमा विकल्पहरू *Single room*, *Twin room*, र *Double room* हुन सक्छ, तर अर्को होटलमा *Deluxe Single Room*, *Classic Queen Room*, र *Executive King Room* हुन सक्छ। यी उस्तै चीज हुन सक्छन्, तर यति धेरै भिन्नताहरू छन् कि विकल्प यस्तो हुन्छ:

1. सबै सर्तहरूलाई एकल मानकमा परिवर्तन गर्ने प्रयास गर्नुहोस्, जुन धेरै गाह्रो छ, किनभने प्रत्येक केसमा रूपान्तरण मार्ग के हुनेछ भनेर स्पष्ट छैन (जस्तै, *Classic single room* लाई *Single room* मा म्याप गर्न सकिन्छ तर *Superior Queen Room with Courtyard Garden or City View* म्याप गर्न धेरै गाह्रो छ)।

1. हामी NLP दृष्टिकोण लिन सक्छौं र *Solo*, *Business Traveller*, वा *Family with young kids* जस्ता निश्चित सर्तहरूको आवृत्ति मापन गर्न सक्छौं जसले प्रत्येक होटलमा लागू हुन्छ, र सिफारिसमा त्यसलाई समावेश गर्न सक्छौं।

ट्यागहरू सामान्यतया (तर सधैं होइन) एकल क्षेत्र हो जसमा *यात्राको प्रकार*, *अतिथिको प्रकार*, *कोठाको प्रकार*, *रातहरूको संख्या*, र *समीक्षा कसरी प्रस्तुत गरिएको थियो* लाई मिलाउने ५ देखि ६ कमामा छुट्याइएका मानहरू समावेश छन्। तर, किनभने केही समीक्षकहरूले प्रत्येक क्षेत्र भर्दैनन् (उनीहरूले एउटा खाली छोड्न सक्छन्), मानहरू सधैं एउटै क्रममा हुँदैनन्।

उदाहरणका लागि, *समूहको प्रकार* लिनुहोस्। `Tags` स्तम्भमा यस क्षेत्रमा १०२५ अद्वितीय सम्भावनाहरू छन्, र दुर्भाग्यवश, तीमध्ये केही मात्र समूहलाई जनाउँछन् (केही कोठाको प्रकार आदि हुन्)। यदि तपाईं केवल तीलाई फिल्टर गर्नुहुन्छ जसले परिवारलाई उल्लेख गर्छ, परिणामहरूमा धेरै *Family room* प्रकारका परिणामहरू समावेश छन्। यदि तपाईं *with* शब्द समावेश गर्नुहुन्छ, अर्थात् *Family with* मानहरूको गणना गर्नुहोस्, परिणामहरू राम्रो हुन्छन्, ५१५,००० परिणामहरू मध्ये ८०,००० भन्दा बढीमा "Family with young children" वा "Family with older children" वाक्यांश समावेश छ।

यसको मतलब ट्याग स्तम्भ हाम्रो लागि पूर्ण रूपमा बेकार छैन, तर यसलाई उपयोगी बनाउन केही काम गर्नुपर्नेछ।

##### औसत होटल स्कोर

डेटासेटसँग औसत स्कोर र समीक्षाहरूको संख्या सम्बन्धित निम्न स्तम्भहरू छन्:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

यस डेटासेटमा सबैभन्दा धेरै समीक्षाहरू भएको एकल होटल *Britannia International Hotel Canary Wharf* हो जसमा ५१५,००० मध्ये ४७८९ समीक्षाहरू छन्। तर यदि हामी यस होटलको `Total_Number_of_Reviews` मानलाई हेर्छौं भने, यो ९०८६ छ। तपाईं अनुमान गर्न सक्नुहुन्छ कि धेरै स्कोरहरू बिना समीक्षाहरू छन्, त्यसैले शायद हामीले `Additional_Number_of_Scoring` स्तम्भ मानलाई थप्नुपर्छ। त्यो मान २६८२ हो, र यसलाई ४७८९ मा थप्दा ७,४७१ हुन्छ, जुन अझै `Total_Number_of_Reviews` भन्दा १६१५ कम छ।

यदि तपाईं `Average_Score` स्तम्भहरू लिनुहुन्छ, तपाईं अनुमान गर्न सक्नुहुन्छ कि यो डेटासेटमा समीक्षाहरूको औसत हो, तर Kaggle को विवरण "*पछिल्लो वर्षको पछिल्लो टिप्पणीमा आधारित होटलको औसत स्कोर*" हो। यो त्यति उपयोगी देखिँदैन, तर हामी डेटासेटमा समीक्षाको स्कोरमा आधारित हाम्रो आफ्नै औसत गणना गर्न सक्छौं। उही होटललाई उदाहरणको रूपमा प्रयोग गर्दै, औसत होटल स्कोर ७.१ दिइएको छ तर गणना गरिएको स्कोर (समीक्षक स्कोर *डेटासेटमा*) ६.८ छ। यो नजिक छ, तर उस्तै मान होइन, र हामी केवल अनुमान गर्न सक्छौं कि `Additional_Number_of_Scoring` समीक्षाहरूले औसतलाई ७.१ मा वृद्धि गर्यो। दुर्भाग्यवश, परीक्षण गर्ने वा प्रमाणित गर्ने कुनै तरिका नभएकोले, `Average_Score`, `Additional_Number_of_Scoring` र `Total_Number_of_Reviews` लाई प्रयोग गर्न वा विश्वास गर्न गाह्रो छ जब तिनीहरू आधारित छन्, वा हामीसँग नभएको डेटालाई जनाउँछन्।

चीजहरू अझ जटिल बनाउन, डेटासेटमा दोस्रो उच्चतम समीक्षाहरू भएको होटलको गणना गरिएको औसत स्कोर ८.१२ छ र डेटासेट `Average_Score` ८.१ छ। यो सही स्कोर संयोग हो कि पहिलो होटल विसंगति हो?

यो सम्भावनामा कि यी होटल आउटलायर हुन सक्छन्, र शायद अधिकांश मानहरू मिल्छन् (तर केही कारणले मिल्दैनन्) हामी डेटासेटमा मानहरू अन्वेषण गर्न र मानहरूको सही प्रयोग (वा गैर-प्रयोग) निर्धारण गर्न अर्को छोटो कार्यक्रम लेख्नेछौं।
🚨 चेतावनीको नोट

यो डेटासेटसँग काम गर्दा तपाईंले पाठबाट केही गणना गर्ने कोड लेख्नुहुनेछ, जसले पाठलाई आफैं पढ्न वा विश्लेषण गर्न आवश्यक पर्दैन। यही नै NLP को सार हो, अर्थ वा भावना व्याख्या गर्नु जसमा मानिसले आफैंले गर्न नपरोस्। तर, यो सम्भव छ कि तपाईंले केही नकारात्मक समीक्षाहरू पढ्नुहुनेछ। म तपाईंलाई आग्रह गर्दछु कि त्यसो नगर्नुहोस्, किनभने तपाईंलाई त्यसो गर्न आवश्यक छैन। तीमध्ये केही हास्यास्पद वा अप्रासंगिक नकारात्मक होटल समीक्षाहरू हुन सक्छन्, जस्तै "मौसम राम्रो थिएन", जुन होटलको नियन्त्रणभन्दा बाहिरको कुरा हो, वा वास्तवमा, कसैको पनि। तर, केही समीक्षाहरूको अँध्यारो पक्ष पनि छ। कहिलेकाहीं नकारात्मक समीक्षाहरू जातीयतावादी, लैंगिकतावादी, वा उमेरवादी हुन्छन्। यो दुर्भाग्यपूर्ण छ तर सार्वजनिक वेबसाइटबाट स्क्र्याप गरिएको डेटासेटमा अपेक्षित कुरा हो। केही समीक्षकहरूले त्यस्ता समीक्षाहरू छोड्छन् जुन तपाईंलाई अप्रिय, असहज, वा दुःखद लाग्न सक्छ। भावना मापन गर्न कोडलाई नै जिम्मा दिनु राम्रो हुन्छ, आफैंले पढेर दुःखी नहुनुहोस्। यद्यपि, यस्तो कुरा लेख्नेहरू अल्पसंख्यक हुन्, तर तिनीहरू अस्तित्वमा छन्।
## अभ्यास - डेटा अन्वेषण  
### डेटा लोड गर्नुहोस्  

अब डेटा दृश्य रूपमा जाँच्न पर्याप्त भयो, अब तपाईंले केही कोड लेखेर उत्तरहरू प्राप्त गर्नु पर्नेछ! यो खण्डमा pandas लाइब्रेरी प्रयोग गरिएको छ। तपाईंको पहिलो काम भनेको CSV डेटा लोड गर्न र पढ्न सक्षम हुनु हो। pandas लाइब्रेरीसँग छिटो CSV लोडर छ, र नतिजा पहिलेका पाठहरूमा जस्तै एक dataframe मा राखिन्छ। हामीले लोड गर्न लागेको CSV मा आधा मिलियनभन्दा बढी पंक्तिहरू छन्, तर केवल १७ स्तम्भहरू छन्। pandas ले तपाईंलाई dataframe सँग अन्तरक्रिया गर्न धेरै शक्तिशाली तरिकाहरू प्रदान गर्दछ, जसमा प्रत्येक पंक्तिमा अपरेशनहरू गर्न सक्ने क्षमता पनि समावेश छ।  

यस पाठको बाँकी भागमा कोड स्निपेटहरू, कोडको व्याख्या, र नतिजाको अर्थको बारेमा छलफल हुनेछ। तपाईंको कोडको लागि समावेश गरिएको _notebook.ipynb_ प्रयोग गर्नुहोस्।  

आउनुहोस्, तपाईंले प्रयोग गर्ने डेटा फाइल लोड गरेर सुरु गरौं:  

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

अब डेटा लोड भएको छ, हामी यसमा केही अपरेशनहरू गर्न सक्छौं। यो कोडलाई तपाईंको प्रोग्रामको शीर्षमा राख्नुहोस्।  

## डेटा अन्वेषण गर्नुहोस्  

यस अवस्थामा, डेटा पहिले नै *सफा* छ, यसको मतलब यो काम गर्न तयार छ, र अन्य भाषाहरूका अक्षरहरू छैनन् जसले केवल अंग्रेजी अक्षरहरू अपेक्षा गर्ने एल्गोरिदमलाई समस्या दिन सक्छ।  

✅ तपाईंले यस्तो डेटा सँग काम गर्नुपर्ने हुन सक्छ जसलाई NLP प्रविधिहरू लागू गर्नु अघि प्रारम्भिक प्रक्रिया आवश्यक पर्छ, तर यस पटक होइन। यदि तपाईंलाई गैर-अंग्रेजी अक्षरहरू सम्हाल्नुपर्ने भएमा, तपाईंले कसरी सम्हाल्नुहुन्थ्यो?  

एक क्षण लिनुहोस् र सुनिश्चित गर्नुहोस् कि डेटा लोड भएपछि, तपाईं यसलाई कोडको साथ अन्वेषण गर्न सक्नुहुन्छ। `Negative_Review` र `Positive_Review` स्तम्भहरूमा ध्यान केन्द्रित गर्न सजिलो छ। यी स्तम्भहरूमा तपाईंको NLP एल्गोरिदमले प्रक्रिया गर्न प्राकृतिक पाठ भरिएको छ। तर पर्खनुहोस्! NLP र भावना विश्लेषणमा जानु अघि, तपाईंले pandas प्रयोग गरेर dataset मा दिइएका मानहरू तपाईंले गणना गरेको मानसँग मेल खाने सुनिश्चित गर्न तलको कोड अनुसरण गर्नु पर्छ।  

## Dataframe अपरेशनहरू  

यस पाठको पहिलो काम भनेको तलका दाबीहरू सही छन् कि छैनन् भनेर dataframe जाँच गर्ने कोड लेख्नु हो (यसलाई परिवर्तन नगरी)।  

> धेरै प्रोग्रामिङ कार्यहरू जस्तै, यसलाई पूरा गर्न धेरै तरिकाहरू छन्, तर राम्रो सल्लाह भनेको यो सबैभन्दा सरल र सजिलो तरिकामा गर्नु हो, विशेष गरी यदि तपाईं भविष्यमा यो कोडमा फर्कनुहुन्छ भने यो बुझ्न सजिलो हुनेछ। Dataframe सँग, एक व्यापक API छ जसले प्रायः तपाईंले चाहेको कुरा कुशलतापूर्वक गर्न तरिका प्रदान गर्दछ।  

तलका प्रश्नहरूलाई कोडिङ कार्यको रूपमा व्यवहार गर्नुहोस् र समाधान हेर्नु अघि उत्तर दिन प्रयास गर्नुहोस्।  

1. तपाईंले लोड गरेको dataframe को *आकार* प्रिन्ट गर्नुहोस् (आकार भनेको पंक्तिहरू र स्तम्भहरूको संख्या हो)।  
2. समीक्षकहरूको राष्ट्रियता (Reviewer_Nationality) को आवृत्ति गणना गर्नुहोस्:  
   1. `Reviewer_Nationality` स्तम्भको लागि कति भिन्न मानहरू छन् र ती के हुन्?  
   2. dataset मा सबैभन्दा सामान्य समीक्षक राष्ट्रियता कुन हो (देश र समीक्षाहरूको संख्या प्रिन्ट गर्नुहोस्)?  
   3. अर्को शीर्ष १० सबैभन्दा बारम्बार पाइने राष्ट्रियता र तिनीहरूको आवृत्ति गणना के हुन्?  
3. शीर्ष १० समीक्षक राष्ट्रियताहरूको लागि प्रत्येकको सबैभन्दा बारम्बार समीक्षित होटल कुन हो?  
4. dataset मा प्रत्येक होटलको समीक्षाहरूको संख्या (होटलको आवृत्ति गणना) कति छ?  
5. dataset मा प्रत्येक होटलको लागि समीक्षक स्कोरहरूको औसत गणना गरेर `Calc_Average_Score` नामक नयाँ स्तम्भ थप्नुहोस्।  
6. के कुनै होटलहरू छन् जसको `Average_Score` र `Calc_Average_Score` (१ दशमलव स्थानमा गोल गरिएको) समान छ?  
   1. Python function लेख्ने प्रयास गर्नुहोस् जसले Series (पंक्ति) लाई तर्कको रूपमा लिन्छ र मानहरू तुलना गर्छ, जब मानहरू समान छैनन् भने सन्देश प्रिन्ट गर्छ। त्यसपछि `.apply()` विधि प्रयोग गरेर प्रत्येक पंक्तिलाई function सँग प्रक्रिया गर्नुहोस्।  
7. `Negative_Review` स्तम्भको मान "No Negative" भएका कति पंक्तिहरू छन्?  
8. `Positive_Review` स्तम्भको मान "No Positive" भएका कति पंक्तिहरू छन्?  
9. `Positive_Review` स्तम्भको मान "No Positive" **र** `Negative_Review` स्तम्भको मान "No Negative" भएका कति पंक्तिहरू छन्?  

### कोड उत्तरहरू  

1. तपाईंले लोड गरेको dataframe को *आकार* प्रिन्ट गर्नुहोस् (आकार भनेको पंक्तिहरू र स्तम्भहरूको संख्या हो)।  

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```  

2. समीक्षकहरूको राष्ट्रियता (Reviewer_Nationality) को आवृत्ति गणना गर्नुहोस्:  

   1. `Reviewer_Nationality` स्तम्भको लागि कति भिन्न मानहरू छन् र ती के हुन्?  
   2. dataset मा सबैभन्दा सामान्य समीक्षक राष्ट्रियता कुन हो (देश र समीक्षाहरूको संख्या प्रिन्ट गर्नुहोस्)?  

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

   3. अर्को शीर्ष १० सबैभन्दा बारम्बार पाइने राष्ट्रियता र तिनीहरूको आवृत्ति गणना के हुन्?  

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

3. शीर्ष १० समीक्षक राष्ट्रियताहरूको लागि प्रत्येकको सबैभन्दा बारम्बार समीक्षित होटल कुन हो?  

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

4. dataset मा प्रत्येक होटलको समीक्षाहरूको संख्या (होटलको आवृत्ति गणना) कति छ?  

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

   तपाईंले देख्न सक्नुहुन्छ कि *dataset मा गणना गरिएको* नतिजाहरू `Total_Number_of_Reviews` मानसँग मेल खाँदैन। यो स्पष्ट छैन कि dataset मा यो मानले होटलले गरेको कुल समीक्षाहरू प्रतिनिधित्व गरेको हो, तर सबै scraped गरिएको छैन, वा केही अन्य गणना। `Total_Number_of_Reviews` मोडेलमा प्रयोग गरिएको छैन किनभने यो अस्पष्टता।  

5. dataset मा प्रत्येक होटलको लागि समीक्षक स्कोरहरूको औसत गणना गरेर `Calc_Average_Score` नामक नयाँ स्तम्भ थप्नुहोस्। स्तम्भहरू `Hotel_Name`, `Average_Score`, र `Calc_Average_Score` प्रिन्ट गर्नुहोस्।  

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

   तपाईंले `Average_Score` मानको बारेमा पनि सोच्न सक्नुहुन्छ र किन यो कहिलेकाहीं गणना गरिएको औसत स्कोरसँग फरक छ। हामी जान्न सक्दैनौं किन केही मानहरू मेल खाँदछन्, तर अन्यमा फरक छ, यस अवस्थामा सुरक्षित भनेको समीक्षक स्कोरहरू प्रयोग गरेर औसत आफैं गणना गर्नु हो।  

6. `Negative_Review` स्तम्भको मान "No Negative" भएका कति पंक्तिहरू छन्?  

7. `Positive_Review` स्तम्भको मान "No Positive" भएका कति पंक्तिहरू छन्?  

8. `Positive_Review` स्तम्भको मान "No Positive" **र** `Negative_Review` स्तम्भको मान "No Negative" भएका कति पंक्तिहरू छन्?  

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

## अर्को तरिका  

Lambda प्रयोग नगरी वस्तुहरू गणना गर्ने अर्को तरिका, र पंक्तिहरू गणना गर्न sum प्रयोग गर्नुहोस्:  

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

   तपाईंले देख्न सक्नुहुन्छ कि `Negative_Review` र `Positive_Review` स्तम्भहरूको लागि "No Negative" र "No Positive" मान भएका १२७ पंक्तिहरू छन्। यसको मतलब समीक्षकले होटललाई संख्यात्मक स्कोर दिएको छ, तर सकारात्मक वा नकारात्मक समीक्षाहरू लेख्न अस्वीकार गरेको छ।  

अब तपाईंले dataset अन्वेषण गर्नुभयो, अर्को पाठमा तपाईंले डेटा फिल्टर गर्नुहुनेछ र केही भावना विश्लेषण थप्नुहुनेछ।  

---  
## 🚀चुनौती  

यस पाठले देखाउँछ, जस्तै हामीले पहिलेका पाठहरूमा देख्यौं, कि डेटा र यसको कमजोरीहरूलाई बुझ्नु कत्तिको महत्त्वपूर्ण छ। विशेष गरी पाठ-आधारित डेटा ध्यानपूर्वक जाँच गर्नुपर्छ। विभिन्न पाठ-गहन dataset हरू खोतल्नुहोस् र हेर्नुहोस् कि तपाईं मोडेलमा पूर्वाग्रह वा skewed भावना ल्याउन सक्ने क्षेत्रहरू पत्ता लगाउन सक्नुहुन्छ।  

## [पाठ-पछिको क्विज](https://ff-quizzes.netlify.app/en/ml/)  

## समीक्षा र आत्म अध्ययन  

[यो NLP सिक्ने मार्ग](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) लिनुहोस् र भाषण र पाठ-गहन मोडेल निर्माण गर्दा प्रयास गर्न उपकरणहरू पत्ता लगाउनुहोस्।  

## असाइनमेन्ट  

[NLTK](assignment.md)  

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादमा त्रुटिहरू वा अशुद्धताहरू हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।