<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-03T23:43:37+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "hi"
}
-->
# अपने ML मॉडल का उपयोग करने के लिए एक वेब ऐप बनाएं

इस पाठ्यक्रम के इस भाग में, आपको एक व्यावहारिक ML विषय से परिचित कराया जाएगा: कैसे अपने Scikit-learn मॉडल को एक फाइल के रूप में सेव करें जिसे वेब एप्लिकेशन के भीतर भविष्यवाणी करने के लिए उपयोग किया जा सके। एक बार मॉडल सेव हो जाने के बाद, आप सीखेंगे कि इसे Flask में बनाए गए वेब ऐप में कैसे उपयोग करें। सबसे पहले, आप कुछ डेटा का उपयोग करके एक मॉडल बनाएंगे जो UFO देखे जाने के बारे में है! फिर, आप एक वेब ऐप बनाएंगे जो आपको सेकंड की संख्या, अक्षांश और देशांतर मान दर्ज करने की अनुमति देगा ताकि यह भविष्यवाणी की जा सके कि किस देश ने UFO देखने की रिपोर्ट की है।

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.hi.jpg)

फोटो <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">माइकल हेरेन</a> द्वारा <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> पर

## पाठ

1. [एक वेब ऐप बनाएं](1-Web-App/README.md)

## क्रेडिट्स

"एक वेब ऐप बनाएं" को ♥️ के साथ [जेन लूपर](https://twitter.com/jenlooper) द्वारा लिखा गया था।

♥️ क्विज़ [रोहन राज](https://twitter.com/rohanraj) द्वारा लिखे गए थे।

डेटासेट [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) से लिया गया है।

वेब ऐप आर्किटेक्चर आंशिक रूप से [इस लेख](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) और [इस रिपॉजिटरी](https://github.com/abhinavsagar/machine-learning-deployment) द्वारा सुझाया गया था, जिसे अभिनव सागर ने बनाया है।

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  