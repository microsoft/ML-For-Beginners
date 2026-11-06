# मशीन लर्निंग के लिए क्लस्टरिंग मॉडल

क्लस्टरिंग एक मशीन लर्निंग कार्य है जिसमें समान दिखने वाली वस्तुओं को खोजा जाता है और उन्हें क्लस्टर नामक समूहों में वर्गीकृत किया जाता है। जो बात क्लस्टरिंग को मशीन लर्निंग की अन्य विधाओं से अलग करती है, वह यह है कि यह प्रक्रिया स्वचालित रूप से होती है, वास्तव में, यह कहना उचित होगा कि यह सुपरवाइज्ड लर्निंग के विपरीत है।

## क्षेत्रीय विषय: नाइजीरियाई श्रोताओं के संगीत स्वाद के लिए क्लस्टरिंग मॉडल 🎧

नाइजीरिया के विविध श्रोताओं के विविध संगीत स्वाद हैं। Spotify से स्क्रैप किए गए डेटा का उपयोग करते हुए (इस लेख से प्रेरित [यहाँ](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), आइए नाइजीरिया में लोकप्रिय कुछ संगीत पर नजर डालते हैं। इस डेटा सेट में विभिन्न गानों के 'डांसबिलिटी' स्कोर, 'अकॉस्टिकनेस', लाउडनेस, 'स्पीचिनेस', लोकप्रियता और ऊर्जा के बारे में डेटा शामिल है। इस डेटा में पैटर्न खोजना रोचक होगा!

![A turntable](../../../translated_images/hi/turntable.f2b86b13c53302dc.webp)

> फोटो <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">मारसेला लस्कोस्की</a> द्वारा <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">अन्सप्लैश</a> पर
  
इस पाठ श्रृंखला में, आप क्लस्टरिंग तकनीकों का उपयोग करके डेटा विश्लेषण के नए तरीके खोजेंगे। क्लस्टरिंग विशेष रूप से तब उपयोगी होती है जब आपके पास लेबल नहीं होते। यदि लेबल होते हैं, तो उस स्थिति में पूर्व पाठों में सीखी गई वर्गीकरण तकनीकें अधिक उपयोगी हो सकती हैं। लेकिन जब आप बिना लेबल वाले डेटा को समूहित करना चाहते हैं, तो क्लस्टरिंग पैटर्न खोजने का एक बेहतरीन तरीका है।

> ऐसे उपयोगी लो-कोड टूल्स हैं जो आपको क्लस्टरिंग मॉडल के साथ काम करना सिखा सकते हैं। इस कार्य के लिए [Azure ML का प्रयास करें](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## पाठ

1. [क्लस्टरिंग का परिचय](1-Visualize/README.md)
2. [K-मींस क्लस्टरिंग](2-K-Means/README.md)

## क्रेडिट्स

यह पाठ [जेन लूपर](https://www.twitter.com/jenlooper) द्वारा 🎶 के साथ लिखा गया था, और [ऋषित दागली](https://rishit_dagli/) तथा [मुहम्मद साकिब खान इनान](https://twitter.com/Sakibinan) के उपयोगी समीक्षाओं के साथ।

[नाइजीरियाई गाने](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) का डेटा सेट Spotify से स्क्रैप करके Kaggle से प्राप्त किया गया था।

उपयोगी K-मींस उदाहरणों में इस [आइरिस एक्सप्लोरेशन](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), इस [परिचयात्मक नोटबुक](https://www.kaggle.com/prashant111/k-means-clustering-with-python), और इस [काल्पनिक NGO उदाहरण](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) शामिल हैं, जिनसे इस पाठ को बनाने में मदद मिली।

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**अस्वीकरण**:
इस दस्तावेज़ का अनुवाद AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवादों में त्रुटियाँ या अशुद्धियाँ हो सकती हैं। मूल दस्तावेज़ अपनी मूल भाषा में ही प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->