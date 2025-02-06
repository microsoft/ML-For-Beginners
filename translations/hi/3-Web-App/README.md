# अपने ML मॉडल का उपयोग करने के लिए एक वेब ऐप बनाएं

इस पाठ्यक्रम के इस भाग में, आपको एक लागू ML विषय से परिचित कराया जाएगा: अपने Scikit-learn मॉडल को एक फ़ाइल के रूप में कैसे सहेजें जिसे वेब एप्लिकेशन के भीतर भविष्यवाणियां करने के लिए उपयोग किया जा सकता है। एक बार जब मॉडल सहेजा जाता है, तो आप सीखेंगे कि इसे Flask में निर्मित एक वेब ऐप में कैसे उपयोग किया जाए। आप सबसे पहले कुछ डेटा का उपयोग करके एक मॉडल बनाएंगे जो यूएफओ देखे जाने के बारे में है! फिर, आप एक वेब ऐप बनाएंगे जो आपको एक संख्या में सेकंड के साथ अक्षांश और देशांतर मान दर्ज करने की अनुमति देगा ताकि यह भविष्यवाणी की जा सके कि किस देश ने यूएफओ देखने की रिपोर्ट की है।

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.hi.jpg)

<a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> द्वारा <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> पर फोटो

## पाठ

1. [एक वेब ऐप बनाएं](1-Web-App/README.md)

## श्रेय

"Build a Web App" को ♥️ से [Jen Looper](https://twitter.com/jenlooper) द्वारा लिखा गया था।

♥️ क्विज़ को Rohan Raj द्वारा लिखा गया था।

डेटासेट [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) से प्राप्त किया गया है।

वेब ऐप आर्किटेक्चर को आंशिक रूप से [इस लेख](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) और [इस रेपो](https://github.com/abhinavsagar/machine-learning-deployment) द्वारा Abhinav Sagar द्वारा सुझाया गया था।

**अस्वीकरण**:
इस दस्तावेज़ का अनुवाद मशीन-आधारित एआई अनुवाद सेवाओं का उपयोग करके किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवादों में त्रुटियाँ या अशुद्धियाँ हो सकती हैं। मूल भाषा में मूल दस्तावेज़ को प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।