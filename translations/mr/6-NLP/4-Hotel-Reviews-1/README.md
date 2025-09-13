<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-06T06:20:49+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "mr"
}
-->
# हॉटेल पुनरावलोकनांसह भावना विश्लेषण - डेटाचे प्रक्रिया करणे

या विभागात तुम्ही मागील धड्यांमधील तंत्रांचा वापर करून मोठ्या डेटासेटचे अन्वेषणात्मक डेटा विश्लेषण कराल. विविध स्तंभांच्या उपयुक्ततेची चांगली समज मिळाल्यानंतर तुम्ही शिकाल:

- अनावश्यक स्तंभ कसे काढायचे
- विद्यमान स्तंभांवर आधारित नवीन डेटा कसा मोजायचा
- अंतिम आव्हानासाठी डेटासेट कसे जतन करायचे

## [पूर्व-व्याख्यान क्विझ](https://ff-quizzes.netlify.app/en/ml/)

### परिचय

आतापर्यंत तुम्ही शिकले आहे की मजकूर डेटा संख्यात्मक प्रकारच्या डेटापेक्षा खूप वेगळा असतो. जर तो मानवाने लिहिलेला किंवा बोललेला मजकूर असेल, तर तो नमुने आणि वारंवारता, भावना आणि अर्थ शोधण्यासाठी विश्लेषित केला जाऊ शकतो. हा धडा तुम्हाला एका वास्तविक डेटासेटमध्ये घेऊन जातो ज्यामध्ये एक वास्तविक आव्हान आहे: **[युरोपमधील 515K हॉटेल पुनरावलोकन डेटा](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** ज्यामध्ये [CC0: सार्वजनिक डोमेन परवाना](https://creativecommons.org/publicdomain/zero/1.0/) समाविष्ट आहे. हा डेटा Booking.com वरून सार्वजनिक स्रोतांमधून गोळा केला गेला आहे. या डेटासेटचा निर्माता Jiashen Liu आहे.

### तयारी

तुम्हाला आवश्यक असेल:

* Python 3 वापरून .ipynb नोटबुक चालवण्याची क्षमता
* pandas
* NLTK, [ज्याला तुम्ही स्थानिकपणे स्थापित करावे](https://www.nltk.org/install.html)
* Kaggle वर उपलब्ध असलेला डेटासेट [युरोपमधील 515K हॉटेल पुनरावलोकन डेटा](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). हा डेटा अनझिप केल्यानंतर सुमारे 230 MB आहे. तो या NLP धड्यांशी संबंधित `/data` फोल्डरमध्ये डाउनलोड करा.

## अन्वेषणात्मक डेटा विश्लेषण

हे आव्हान गृहीत धरते की तुम्ही भावना विश्लेषण आणि पाहुण्यांच्या पुनरावलोकन स्कोअर वापरून हॉटेल शिफारस करणारा बॉट तयार करत आहात. तुम्ही वापरणार असलेला डेटासेट 6 शहरांतील 1493 वेगवेगळ्या हॉटेल्सचे पुनरावलोकन समाविष्ट करतो.

Python, हॉटेल पुनरावलोकनांचा डेटासेट आणि NLTK च्या भावना विश्लेषणाचा वापर करून तुम्ही शोधू शकता:

* पुनरावलोकनांमध्ये सर्वाधिक वारंवार वापरले जाणारे शब्द आणि वाक्यांश कोणते आहेत?
* हॉटेलचे अधिकृत *टॅग्स* पुनरावलोकन स्कोअरशी संबंधित आहेत का (उदा. *Family with young children* साठी अधिक नकारात्मक पुनरावलोकने आहेत का *Solo traveller* च्या तुलनेत, कदाचित हे *Solo travellers* साठी चांगले आहे असे दर्शवित आहे)?
* NLTK भावना स्कोअर हॉटेल पुनरावलोकनकर्त्याच्या संख्यात्मक स्कोअरशी 'सहमत' आहेत का?

#### डेटासेट

तुम्ही डाउनलोड केलेला आणि स्थानिकपणे जतन केलेला डेटासेट एक्सप्लोर करूया. फाइल VS Code किंवा Excel सारख्या संपादकात उघडा.

डेटासेटमधील हेडर्स खालीलप्रमाणे आहेत:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

हे एका सोप्या स्वरूपात गटबद्ध केले आहे:
##### हॉटेल स्तंभ

* `Hotel_Name`, `Hotel_Address`, `lat` (अक्षांश), `lng` (रेखांश)
  * *lat* आणि *lng* वापरून तुम्ही Python चा वापर करून हॉटेल स्थानांचे नकाशे तयार करू शकता (कदाचित नकारात्मक आणि सकारात्मक पुनरावलोकनांसाठी रंग कोड केलेले)
  * Hotel_Address आपल्यासाठी स्पष्टपणे उपयुक्त नाही आणि आम्ही सोर्टिंग आणि शोध सुलभ करण्यासाठी देशाने ते बदलू शकतो

**हॉटेल मेटा-पुनरावलोकन स्तंभ**

* `Average_Score`
  * डेटासेट निर्मात्याच्या मते, हा स्तंभ *हॉटेलचा सरासरी स्कोअर आहे, जो मागील वर्षातील नवीनतम टिप्पण्यांवर आधारित गणना केला जातो*. हा स्कोअर मोजण्याचा असामान्य मार्ग वाटतो, परंतु हा डेटा गोळा केला गेला आहे म्हणून आम्ही सध्या तो स्वीकारतो.

  ✅ या डेटामधील इतर स्तंभांवर आधारित सरासरी स्कोअर मोजण्याचा आणखी एक मार्ग तुम्हाला सुचतो का?

* `Total_Number_of_Reviews`
  * या हॉटेलला मिळालेल्या पुनरावलोकनांची एकूण संख्या - काही कोड लिहिल्याशिवाय हे स्पष्ट नाही की हे डेटासेटमधील पुनरावलोकनांना संदर्भित करते का.
* `Additional_Number_of_Scoring`
  * याचा अर्थ पुनरावलोकन स्कोअर दिला गेला आहे परंतु पुनरावलोकनकर्त्याने कोणतेही सकारात्मक किंवा नकारात्मक पुनरावलोकन लिहिलेले नाही

**पुनरावलोकन स्तंभ**

- `Reviewer_Score`
  - हा एक संख्यात्मक मूल्य आहे ज्यामध्ये किमान 1 दशांश स्थान आहे, ज्यामध्ये किमान आणि जास्तीत जास्त मूल्ये 2.5 आणि 10 आहेत
  - 2.5 हे सर्वात कमी स्कोअर का आहे याचे स्पष्टीकरण दिलेले नाही
- `Negative_Review`
  - जर पुनरावलोकनकर्त्याने काहीही लिहिले नाही, तर या फील्डमध्ये "**No Negative**" असेल
  - लक्षात ठेवा की पुनरावलोकनकर्ता नकारात्मक पुनरावलोकन स्तंभात सकारात्मक पुनरावलोकन लिहू शकतो (उदा. "या हॉटेलबद्दल काहीही वाईट नाही")
- `Review_Total_Negative_Word_Counts`
  - जास्त नकारात्मक शब्द संख्या कमी स्कोअर दर्शवते (भावनात्मकता तपासल्याशिवाय)
- `Positive_Review`
  - जर पुनरावलोकनकर्त्याने काहीही लिहिले नाही, तर या फील्डमध्ये "**No Positive**" असेल
  - लक्षात ठेवा की पुनरावलोकनकर्ता सकारात्मक पुनरावलोकन स्तंभात नकारात्मक पुनरावलोकन लिहू शकतो (उदा. "या हॉटेलबद्दल काहीही चांगले नाही")
- `Review_Total_Positive_Word_Counts`
  - जास्त सकारात्मक शब्द संख्या जास्त स्कोअर दर्शवते (भावनात्मकता तपासल्याशिवाय)
- `Review_Date` आणि `days_since_review`
  - पुनरावलोकनावर ताजेपणा किंवा जुनेपणाचे मोजमाप लागू केले जाऊ शकते (जुने पुनरावलोकन नवीन पुनरावलोकनांइतके अचूक नसू शकते कारण हॉटेल व्यवस्थापन बदलले आहे, नूतनीकरण केले गेले आहे, किंवा पूल जोडला गेला आहे इ.)
- `Tags`
  - हे लहान वर्णन आहेत जे पुनरावलोकनकर्ता निवडू शकतो ज्यामध्ये ते पाहुणे कोणत्या प्रकारचे होते (उदा. सोलो किंवा फॅमिली), त्यांना कोणत्या प्रकारची खोली मिळाली, त्यांचा मुक्काम किती होता आणि पुनरावलोकन कसे सादर केले गेले. 
  - दुर्दैवाने, या टॅग्सचा वापर करणे समस्यात्मक आहे, खाली त्यांच्या उपयुक्ततेवर चर्चा करणारा विभाग तपासा

**पुनरावलोकनकर्ता स्तंभ**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - हे शिफारस मॉडेलमध्ये घटक असू शकते, उदाहरणार्थ, जर तुम्ही ठरवू शकत असाल की शंभराहून अधिक पुनरावलोकन असलेले अधिक विपुल पुनरावलोकनकर्ते नकारात्मक होण्याची शक्यता जास्त होती सकारात्मक होण्यापेक्षा. तथापि, कोणत्याही विशिष्ट पुनरावलोकनाचा पुनरावलोकनकर्ता अद्वितीय कोडसह ओळखला जात नाही आणि म्हणूनच पुनरावलोकनांच्या संचाशी जोडला जाऊ शकत नाही. 100 किंवा अधिक पुनरावलोकन असलेले 30 पुनरावलोकनकर्ते आहेत, परंतु हे शिफारस मॉडेलसाठी कसे मदत करू शकते हे पाहणे कठीण आहे.
- `Reviewer_Nationality`
  - काही लोकांना असे वाटू शकते की विशिष्ट राष्ट्रीयतेला राष्ट्रीय कलामुळे सकारात्मक किंवा नकारात्मक पुनरावलोकन देण्याची अधिक शक्यता आहे. अशा उपाख्यानात्मक दृष्टिकोन तुमच्या मॉडेलमध्ये तयार करताना काळजी घ्या. हे राष्ट्रीय (आणि कधीकधी वांशिक) स्टीरिओटाइप्स आहेत आणि प्रत्येक पुनरावलोकनकर्ता एक व्यक्ती होता ज्याने त्यांच्या अनुभवावर आधारित पुनरावलोकन लिहिले. हे अनेक दृष्टिकोनांमधून फिल्टर केले गेले असू शकते जसे की त्यांचे मागील हॉटेल मुक्काम, प्रवास केलेले अंतर आणि त्यांचे वैयक्तिक स्वभाव. त्यांची राष्ट्रीयता पुनरावलोकन स्कोअरचे कारण होती असे विचार करणे योग्य ठरवणे कठीण आहे.

##### उदाहरणे

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | हे सध्या हॉटेल नाही तर बांधकाम साइट आहे. मला सकाळी लवकर आणि दिवसभर असह्य बांधकामाच्या आवाजाने त्रास दिला गेला. मी खोली बदलण्याची विनंती केली पण शांत खोली उपलब्ध नव्हती. परिस्थिती आणखी वाईट करण्यासाठी मला जास्त शुल्क आकारले गेले. मी संध्याकाळी चेक आउट केले कारण मला लवकर फ्लाइट पकडायची होती. दुसऱ्या दिवशी हॉटेलने माझ्या संमतीशिवाय बुक केलेल्या किंमतीपेक्षा जास्त शुल्क आकारले. हे एक भयंकर ठिकाण आहे. येथे बुकिंग करून स्वतःला शिक्षा करू नका. | काहीही नाही. भयंकर ठिकाण. दूर रहा. | व्यवसाय प्रवास, जोडपे, स्टँडर्ड डबल रूम, 2 रात्री मुक्काम |

जसे तुम्ही पाहू शकता, या पाहुण्याचा हॉटेलमध्ये मुक्काम आनंददायक नव्हता. हॉटेलला 7.8 चा चांगला सरासरी स्कोअर आणि 1945 पुनरावलोकने आहेत, परंतु या पुनरावलोकनकर्त्याने त्याला 2.5 दिले आणि त्यांच्या मुक्कामाबद्दल किती नकारात्मक होता याबद्दल 115 शब्द लिहिले. जर त्यांनी सकारात्मक पुनरावलोकन स्तंभात काहीही लिहिले नाही, तर तुम्ही असा अंदाज लावू शकता की काहीही सकारात्मक नव्हते, परंतु त्यांनी 7 शब्दांचा इशारा लिहिला. जर आपण शब्द मोजले तर पुनरावलोकनकर्त्याच्या हेतूचा चुकीचा दृष्टिकोन असू शकतो. आश्चर्यकारकपणे, त्यांचा 2.5 चा स्कोअर गोंधळात टाकणारा आहे, कारण जर हॉटेलचा मुक्काम इतका वाईट असेल, तर त्यांनी काही गुण का दिले? डेटासेट जवळून तपासल्यास, तुम्हाला दिसेल की सर्वात कमी स्कोअर 2.5 आहे, 0 नाही. सर्वात जास्त स्कोअर 10 आहे.

##### टॅग्स

वर नमूद केल्याप्रमाणे, प्रथमदर्शनी, डेटाचे वर्गीकरण करण्यासाठी `Tags` वापरण्याची कल्पना अर्थपूर्ण वाटते. दुर्दैवाने हे टॅग्स प्रमाणित नाहीत, याचा अर्थ असा की एका हॉटेलमध्ये पर्याय *Single room*, *Twin room*, आणि *Double room* असू शकतात, परंतु दुसऱ्या हॉटेलमध्ये ते *Deluxe Single Room*, *Classic Queen Room*, आणि *Executive King Room* असू शकतात. हे कदाचित समान गोष्टी असू शकतात, परंतु इतक्या विविधता आहेत की निवड अशी होते:

1. सर्व अटी एका मानकात बदलण्याचा प्रयत्न करा, जे खूप कठीण आहे, कारण प्रत्येक बाबतीत रूपांतरण मार्ग काय असेल हे स्पष्ट नाही (उदा. *Classic single room* हे *Single room* शी जुळते परंतु *Superior Queen Room with Courtyard Garden or City View* हे जुळवणे खूप कठीण आहे)

1. आम्ही NLP दृष्टिकोन घेऊ शकतो आणि प्रत्येक हॉटेलसाठी *Solo*, *Business Traveller*, किंवा *Family with young kids* सारख्या विशिष्ट अटींच्या वारंवारतेचे मोजमाप करू शकतो आणि शिफारसीमध्ये त्याचा विचार करू शकतो  

टॅग्स सामान्यतः (परंतु नेहमीच नाहीत) एकल फील्ड असतात ज्यामध्ये *प्रवासाचा प्रकार*, *पाहुण्यांचा प्रकार*, *खोलीचा प्रकार*, *रात्रींची संख्या*, आणि *पुनरावलोकन सबमिट करण्यासाठी वापरलेले डिव्हाइस* यांना संरेखित करणाऱ्या 5 ते 6 अल्पविरामाने विभक्त मूल्यांचा समावेश असतो. तथापि, काही पुनरावलोकनकर्ते प्रत्येक फील्ड भरत नाहीत (ते एक रिक्त ठेवू शकतात), त्यामुळे मूल्ये नेहमीच समान क्रमाने नसतात.

उदाहरण म्हणून, *Type of group* घ्या. या `Tags` स्तंभात या फील्डमध्ये 1025 अद्वितीय शक्यता आहेत आणि दुर्दैवाने त्यापैकी काहीच गटाचा संदर्भ देतात (काही खोलीच्या प्रकाराशी संबंधित आहेत). जर तुम्ही फक्त कुटुंबाचा उल्लेख करणारे फिल्टर केले, तर परिणामांमध्ये अनेक *Family room* प्रकारचे परिणाम असतात. जर तुम्ही *with* हा शब्द समाविष्ट केला, म्हणजे *Family with* मूल्ये मोजली, तर परिणाम चांगले आहेत, 515,000 परिणामांपैकी 80,000 हून अधिक परिणामांमध्ये "Family with young children" किंवा "Family with older children" वाक्यांश समाविष्ट आहे.

याचा अर्थ टॅग्स स्तंभ आपल्यासाठी पूर्णपणे निरुपयोगी नाही, परंतु तो उपयुक्त बनवण्यासाठी काही काम करावे लागेल.

##### हॉटेलचा सरासरी स्कोअर

डेटासेटशी संबंधित काही विचित्रता किंवा विसंगती आहेत ज्याचे मी उत्तर शोधू शकत नाही, परंतु येथे स्पष्ट केल्या आहेत जेणेकरून तुम्ही तुमची मॉडेल तयार करताना त्याबद्दल जागरूक असाल. जर तुम्ही उत्तर शोधले, तर कृपया चर्चेच्या विभागात आम्हाला कळवा!

डेटासेटमध्ये सरासरी स्कोअर आणि पुनरावलोकनांच्या संख्येशी संबंधित खालील स्तंभ आहेत:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

या डेटासेटमधील सर्वाधिक पुनरावलोकन असलेले एकमेव हॉटेल *Britannia International Hotel Canary Wharf* आहे ज्यामध्ये 515,000 पैकी 4789 पुनरावलोकने आहेत. परंतु जर आपण या हॉटेलसाठी `Total_Number_of_Reviews` मूल्य पाहिले, तर ते 9086 आहे. तुम्ही असा अंदाज लावू शकता की अनेक स्कोअर पुनरावलोकनांशिवाय आहेत, त्यामुळे कदाचित आपण `Additional_Number_of_Scoring` स्तंभ मूल्य जोडावे. ते मूल्य 2682 आहे आणि 4789 मध्ये जोडल्यास आपल्याला 7471 मिळते जे `Total_Number_of_Reviews` च्या 1615 कमी आहे.

जर तुम्ही `Average_Score` स्तंभ घेतला, तर तुम्ही असा अंदाज लावू शकता की तो डेटासेटमधील पुनरावलोकनांचा सरासरी आहे, परंतु Kaggle मधील वर्णन आहे "*हॉटेलचा सरासरी स्कोअर, जो मागील वर्षातील नवीनतम टिप्पण्यांवर आधारित गणना केला जातो*". ते फारसे उपयुक्त वाटत नाही, परंतु आपण डेटासेटमधील पुनरावलोकन स्कोअरवर आधारित आपली स्वतःची सरासरी मोजू शकतो. त्याच हॉटेलचा उदाहरण म्हणून वापर करून, हॉटेलचा सरासरी स्कोअर 7.1 दिला जातो परंतु डेटासेटमधील (पुनरावलोकनकर्त्याचा सरासरी स्कोअर) गणना केलेला स्कोअर 6.8 आहे. हे जवळ आहे, परंतु समान मूल्य नाही, आणि आपण फक्त असा अंदाज लावू शकतो की `Additional_Number_of_Scoring` पुनरावलोकनांमध्ये दिलेल्या स्कोअरने सरासरी 7.1 पर्यंत वाढवली. दुर्दैवाने, त्या दाव्याची चाचणी घेण्याचा किंवा सिद्ध करण्याचा कोणताही मार्ग नसल्यामुळे,
🚨 एक चेतावणी

जेव्हा तुम्ही या डेटासेटसह काम करता, तेव्हा तुम्ही कोड लिहाल जो मजकुरातून काहीतरी मोजतो, स्वतः मजकूर वाचण्याची किंवा त्याचे विश्लेषण करण्याची गरज नसते. हेच NLP चे सार आहे, मानवी हस्तक्षेपाशिवाय अर्थ किंवा भावना समजून घेणे. मात्र, असे होऊ शकते की तुम्ही काही नकारात्मक पुनरावलोकने वाचाल. मी तुम्हाला असे करण्याचे टाळण्याचा सल्ला देतो, कारण तुम्हाला त्याची गरज नाही. त्यापैकी काही हास्यास्पद किंवा असंबंधित नकारात्मक हॉटेल पुनरावलोकने असू शकतात, जसे की "हवामान चांगले नव्हते", जे हॉटेलच्या किंवा कोणाच्याही नियंत्रणाबाहेर आहे. पण काही पुनरावलोकनांचा एक काळा पैलू देखील आहे. कधी कधी नकारात्मक पुनरावलोकने वांशिक, लैंगिक किंवा वयावर आधारित असतात. हे दुर्दैवी आहे, पण सार्वजनिक वेबसाइटवरून स्क्रॅप केलेल्या डेटासेटमध्ये अपेक्षित आहे. काही पुनरावलोकनकर्ते असे पुनरावलोकने देतात जी तुम्हाला अप्रिय, अस्वस्थ किंवा त्रासदायक वाटू शकतात. भावना मोजण्यासाठी कोडला काम करू द्या, स्वतः वाचून त्रास होऊ नये. असे म्हणता येईल की अशा गोष्टी लिहिणारे अल्पसंख्याक आहेत, पण ते अस्तित्वात आहेत.
## व्यायाम - डेटा अन्वेषण
### डेटा लोड करा

डेटा व्हिज्युअली तपासणे पुरेसे झाले आहे, आता तुम्ही काही कोड लिहून उत्तर शोधणार आहात! या विभागात pandas लायब्ररीचा वापर केला जातो. तुमचे पहिले काम म्हणजे तुम्ही CSV डेटा लोड आणि वाचू शकता याची खात्री करणे. pandas लायब्ररीमध्ये जलद CSV लोडर आहे, आणि परिणाम पूर्वीच्या धड्यांप्रमाणे डेटा फ्रेममध्ये ठेवला जातो. आपण लोड करत असलेला CSV अर्धा मिलियनपेक्षा जास्त रांगा असलेला आहे, परंतु फक्त 17 स्तंभ आहेत. pandas तुम्हाला डेटा फ्रेमशी संवाद साधण्यासाठी अनेक शक्तिशाली पद्धती देते, ज्यामध्ये प्रत्येक रांगेवर ऑपरेशन्स करण्याची क्षमता देखील आहे.

या धड्याच्या पुढील भागात कोड स्निपेट्स असतील आणि कोडचे काही स्पष्टीकरण आणि परिणामांचा अर्थ काय आहे याबद्दल चर्चा असेल. तुमच्या कोडसाठी समाविष्ट केलेल्या _notebook.ipynb_ चा वापर करा.

चला, तुम्ही वापरणार असलेल्या डेटा फाइल लोड करण्यापासून सुरुवात करूया:

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

आता डेटा लोड झाला आहे, आपण त्यावर काही ऑपरेशन्स करू शकतो. पुढील भागासाठी तुमच्या प्रोग्रामच्या शीर्षस्थानी हा कोड ठेवा.

## डेटा अन्वेषण करा

या प्रकरणात, डेटा आधीच *स्वच्छ* आहे, म्हणजे तो काम करण्यासाठी तयार आहे आणि त्यात इतर भाषांतील वर्ण नाहीत जे फक्त इंग्रजी वर्ण अपेक्षित असलेल्या अल्गोरिदममध्ये अडथळा आणू शकतात.

✅ तुम्हाला कदाचित अशा डेटासोबत काम करावे लागेल ज्याला NLP तंत्र लागू करण्यापूर्वी प्रारंभिक प्रक्रिया करून स्वरूपित करणे आवश्यक आहे, परंतु या वेळी नाही. जर तुम्हाला करावे लागले, तर तुम्ही गैर-इंग्रजी वर्ण कसे हाताळाल?

एक क्षण घ्या आणि खात्री करा की डेटा लोड झाल्यानंतर तुम्ही कोडसह त्याचा शोध घेऊ शकता. `Negative_Review` आणि `Positive_Review` स्तंभांवर लक्ष केंद्रित करणे खूप सोपे आहे. ते नैसर्गिक मजकूराने भरलेले आहेत जे तुमच्या NLP अल्गोरिदमसाठी प्रक्रिया करण्यासाठी आहेत. पण थांबा! NLP आणि भावना विश्लेषणात उडी मारण्यापूर्वी, तुम्ही pandas सह दिलेल्या कोडचे अनुसरण करून डेटासेटमधील दिलेल्या मूल्ये तुमच्या गणनेशी जुळतात का ते तपासले पाहिजे.

## डेटा फ्रेम ऑपरेशन्स

या धड्याचे पहिले काम म्हणजे डेटा फ्रेम तपासून (त्यात बदल न करता) खालील दावे योग्य आहेत का ते तपासणे.

> अनेक प्रोग्रामिंग कामांप्रमाणे, हे पूर्ण करण्याचे अनेक मार्ग आहेत, परंतु चांगला सल्ला म्हणजे तुम्ही ते शक्य तितक्या सोप्या आणि सोप्या पद्धतीने करा, विशेषतः जर तुम्हाला भविष्यात या कोडकडे परत येणे सोपे होईल. डेटा फ्रेमसह, एक व्यापक API आहे जो तुम्हाला हवे ते कार्यक्षमतेने करण्याचा मार्ग देईल.

खालील प्रश्नांना कोडिंग कामे म्हणून विचार करा आणि उत्तर शोधण्याचा प्रयत्न करा, समाधानाकडे न पाहता.

1. तुम्ही नुकतेच लोड केलेल्या डेटा फ्रेमचे *आकार* प्रिंट करा (आकार म्हणजे रांगा आणि स्तंभांची संख्या)
2. समीक्षकांच्या राष्ट्रीयत्वासाठी वारंवारता मोजा:
   1. `Reviewer_Nationality` स्तंभासाठी किती वेगवेगळ्या मूल्ये आहेत आणि ती कोणती आहेत?
   2. डेटासेटमध्ये सर्वात सामान्य समीक्षक राष्ट्रीयत्व कोणते आहे (देश आणि पुनरावलोकनांची संख्या प्रिंट करा)?
   3. पुढील 10 सर्वाधिक वारंवार आढळणारी राष्ट्रीयत्वे कोणती आहेत आणि त्यांची वारंवारता मोजा?
3. प्रत्येक टॉप 10 समीक्षक राष्ट्रीयत्वांसाठी सर्वाधिक पुनरावलोकन केलेले हॉटेल कोणते होते?
4. डेटासेटमध्ये प्रति हॉटेल किती पुनरावलोकने आहेत (हॉटेलची वारंवारता मोजा)?
5. डेटासेटमधील प्रत्येक हॉटेलसाठी सर्व समीक्षक स्कोअरचे सरासरी मिळवून सरासरी स्कोअर देखील गणना करू शकता. तुमच्या डेटा फ्रेममध्ये `Calc_Average_Score` या स्तंभ शीर्षकासह एक नवीन स्तंभ जोडा ज्यामध्ये ती गणना केलेली सरासरी असेल.
6. कोणत्याही हॉटेल्समध्ये (1 दशांश स्थानावर गोल केलेले) `Average_Score` आणि `Calc_Average_Score` समान आहेत का?
   1. एक Python फंक्शन लिहिण्याचा प्रयत्न करा जे Series (रांग) ला आर्ग्युमेंट म्हणून घेते आणि मूल्यांची तुलना करते, जेव्हा मूल्ये समान नसतात तेव्हा संदेश प्रिंट करते. नंतर `.apply()` पद्धत वापरून प्रत्येक रांग फंक्शनसह प्रक्रिया करा.
7. `Negative_Review` स्तंभाचे "No Negative" मूल्य असलेल्या किती रांगा आहेत ते गणना करा आणि प्रिंट करा.
8. `Positive_Review` स्तंभाचे "No Positive" मूल्य असलेल्या किती रांगा आहेत ते गणना करा आणि प्रिंट करा.
9. `Positive_Review` स्तंभाचे "No Positive" मूल्य **आणि** `Negative_Review` स्तंभाचे "No Negative" मूल्य असलेल्या किती रांगा आहेत ते गणना करा आणि प्रिंट करा.

### कोड उत्तर

1. तुम्ही नुकतेच लोड केलेल्या डेटा फ्रेमचे *आकार* प्रिंट करा (आकार म्हणजे रांगा आणि स्तंभांची संख्या)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. समीक्षकांच्या राष्ट्रीयत्वासाठी वारंवारता मोजा:

   1. `Reviewer_Nationality` स्तंभासाठी किती वेगवेगळ्या मूल्ये आहेत आणि ती कोणती आहेत?
   2. डेटासेटमध्ये सर्वात सामान्य समीक्षक राष्ट्रीयत्व कोणते आहे (देश आणि पुनरावलोकनांची संख्या प्रिंट करा)?

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

   3. पुढील 10 सर्वाधिक वारंवार आढळणारी राष्ट्रीयत्वे कोणती आहेत आणि त्यांची वारंवारता मोजा?

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

3. प्रत्येक टॉप 10 समीक्षक राष्ट्रीयत्वांसाठी सर्वाधिक पुनरावलोकन केलेले हॉटेल कोणते होते?

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

4. डेटासेटमध्ये प्रति हॉटेल किती पुनरावलोकने आहेत (हॉटेलची वारंवारता मोजा)?

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
   
   तुम्हाला कदाचित *डेटासेटमध्ये मोजलेले* परिणाम `Total_Number_of_Reviews` मूल्याशी जुळत नाहीत असे आढळेल. हे स्पष्ट नाही की डेटासेटमधील हे मूल्य हॉटेलला असलेल्या पुनरावलोकनांची एकूण संख्या दर्शवते, परंतु सर्व स्क्रॅप केले गेले नाहीत, किंवा काही अन्य गणना. `Total_Number_of_Reviews` मॉडेलमध्ये वापरले जात नाही कारण याबद्दल स्पष्टता नाही.

5. डेटासेटमधील प्रत्येक हॉटेलसाठी सर्व समीक्षक स्कोअरचे सरासरी मिळवून सरासरी स्कोअर देखील गणना करू शकता. तुमच्या डेटा फ्रेममध्ये `Calc_Average_Score` या स्तंभ शीर्षकासह एक नवीन स्तंभ जोडा ज्यामध्ये ती गणना केलेली सरासरी असेल. स्तंभ `Hotel_Name`, `Average_Score`, आणि `Calc_Average_Score` प्रिंट करा.

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

   तुम्हाला कदाचित `Average_Score` मूल्याबद्दल आश्चर्य वाटेल आणि ते कधी कधी गणना केलेल्या सरासरी स्कोअरपेक्षा वेगळे का आहे. आम्हाला माहित नाही की काही मूल्ये जुळतात, परंतु इतरांमध्ये फरक आहे, त्यामुळे या प्रकरणात आपल्याकडे असलेल्या पुनरावलोकन स्कोअरचा वापर करून सरासरी स्वतःच गणना करणे सुरक्षित आहे. असे म्हटले तरी, फरक सहसा खूप लहान असतो, येथे डेटासेट सरासरी आणि गणना केलेल्या सरासरीमधून सर्वाधिक विचलन असलेली हॉटेल्स आहेत:

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

   फक्त 1 हॉटेलमध्ये स्कोअरचा फरक 1 पेक्षा जास्त असल्याने, याचा अर्थ असा होतो की आपण फरक दुर्लक्षित करू शकतो आणि गणना केलेला सरासरी स्कोअर वापरू शकतो.

6. `Negative_Review` स्तंभाचे "No Negative" मूल्य असलेल्या किती रांगा आहेत ते गणना करा आणि प्रिंट करा.

7. `Positive_Review` स्तंभाचे "No Positive" मूल्य असलेल्या किती रांगा आहेत ते गणना करा आणि प्रिंट करा.

8. `Positive_Review` स्तंभाचे "No Positive" मूल्य **आणि** `Negative_Review` स्तंभाचे "No Negative" मूल्य असलेल्या किती रांगा आहेत ते गणना करा आणि प्रिंट करा.

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

## दुसरा मार्ग

Lambda वापरल्याशिवाय आयटम मोजण्याचा दुसरा मार्ग, आणि रांगा मोजण्यासाठी sum वापरा:

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

   तुम्ही कदाचित लक्षात घेतले असेल की `Negative_Review` आणि `Positive_Review` स्तंभांसाठी अनुक्रमे "No Negative" आणि "No Positive" मूल्ये असलेल्या 127 रांगा आहेत. याचा अर्थ समीक्षकाने हॉटेलला संख्यात्मक स्कोअर दिला, परंतु सकारात्मक किंवा नकारात्मक पुनरावलोकन लिहिण्यास नकार दिला. सुदैवाने ही रांगा कमी प्रमाणात आहेत (127 पैकी 515738, म्हणजे 0.02%), त्यामुळे कदाचित आमच्या मॉडेल किंवा परिणामांवर कोणत्याही विशिष्ट दिशेने परिणाम होणार नाही, परंतु तुम्हाला कदाचित पुनरावलोकनांच्या डेटासेटमध्ये पुनरावलोकन नसलेल्या रांगा असतील अशी अपेक्षा नसावी, त्यामुळे अशा रांगा शोधण्यासाठी डेटा शोधणे योग्य आहे.

आता तुम्ही डेटासेटचा शोध घेतला आहे, पुढील धड्यात तुम्ही डेटा फिल्टर कराल आणि काही भावना विश्लेषण जोडाल.

---
## 🚀चॅलेंज

या धड्याने, जसे आपण पूर्वीच्या धड्यांमध्ये पाहिले, आपल्या डेटाचा आणि त्याच्या त्रुटींचा अभ्यास करणे किती महत्त्वाचे आहे हे दाखवले. मजकूर-आधारित डेटाला विशेषतः काळजीपूर्वक तपासणी आवश्यक आहे. विविध मजकूर-प्रधान डेटासेट्समध्ये खोदून पहा आणि तुम्ही मॉडेलमध्ये पक्षपात किंवा विकृत भावना आणू शकणारे क्षेत्र शोधू शकता का ते पाहा.

## [पाठ-व्याख्यान क्विझ](https://ff-quizzes.netlify.app/en/ml/)

## पुनरावलोकन आणि स्व-अभ्यास

[NLP वर हा Learning Path](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) घ्या जेव्हा भाषण आणि मजकूर-प्रधान मॉडेल तयार करताना वापरण्यासाठी साधने शोधा.

## असाइनमेंट 

[NLTK](assignment.md)

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) वापरून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी कृपया लक्षात ठेवा की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर करून निर्माण होणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.