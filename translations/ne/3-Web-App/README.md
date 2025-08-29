<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-08-29T17:44:37+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "ne"
}
-->
# आफ्नो ML मोडेल प्रयोग गर्न वेब एप बनाउनुहोस्

यस पाठ्यक्रमको यस भागमा, तपाईंलाई एक प्रयोगात्मक ML विषयमा परिचय गराइनेछ: कसरी आफ्नो Scikit-learn मोडेललाई फाइलको रूपमा सुरक्षित गर्ने जसलाई वेब एप्लिकेसनमा भविष्यवाणी गर्न प्रयोग गर्न सकिन्छ। मोडेल सुरक्षित गरेपछि, तपाईंले यसलाई Flask मा बनाइएको वेब एपमा प्रयोग गर्न सिक्नुहुनेछ। तपाईंले पहिलोमा केही डाटाको प्रयोग गरेर मोडेल बनाउनुहुनेछ, जुन UFO देखिएको घटनाको बारेमा हुनेछ! त्यसपछि, तपाईंले एउटा वेब एप बनाउनुहुनेछ जसले तपाईंलाई सेकन्डको संख्या, अक्षांश र देशान्तरको मान प्रविष्ट गर्न अनुमति दिनेछ, ताकि कुन देशले UFO देखेको रिपोर्ट गरेको हो भनेर भविष्यवाणी गर्न सकियोस्।

![UFO पार्किङ](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.ne.jpg)

फोटो <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">माइकल हेरेन</a> द्वारा <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> मा

## पाठहरू

1. [वेब एप बनाउनुहोस्](1-Web-App/README.md)

## श्रेय

"वेब एप बनाउनुहोस्" ♥️ सहित [Jen Looper](https://twitter.com/jenlooper) द्वारा लेखिएको हो।

♥️ क्विजहरू रोहन राज द्वारा लेखिएका हुन्।

डाटासेट [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) बाट लिइएको हो।

वेब एप आर्किटेक्चर आंशिक रूपमा [यो लेख](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) र [यो रिपो](https://github.com/abhinavsagar/machine-learning-deployment) द्वारा अभिनव सागरको सुझावमा आधारित छ।

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादहरूमा त्रुटि वा अशुद्धता हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।