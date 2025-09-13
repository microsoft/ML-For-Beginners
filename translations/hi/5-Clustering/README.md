<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T22:56:31+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "hi"
}
-->
# मशीन लर्निंग के लिए क्लस्टरिंग मॉडल

क्लस्टरिंग एक मशीन लर्निंग कार्य है जिसमें समान वस्तुओं को खोजा जाता है और उन्हें समूहों में बांटा जाता है जिन्हें क्लस्टर्स कहा जाता है। अन्य मशीन लर्निंग दृष्टिकोणों से क्लस्टरिंग को अलग बनाता है कि यह प्रक्रिया स्वचालित रूप से होती है। वास्तव में, इसे सुपरवाइज्ड लर्निंग का उल्टा कहना उचित होगा।

## क्षेत्रीय विषय: नाइजीरियाई दर्शकों के संगीत स्वाद के लिए क्लस्टरिंग मॉडल 🎧

नाइजीरिया के विविध दर्शकों के संगीत स्वाद भी विविध हैं। Spotify से डेटा स्क्रैप करके (प्रेरित [इस लेख](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) से), आइए नाइजीरिया में लोकप्रिय कुछ संगीत पर नज़र डालें। इस डेटा सेट में विभिन्न गानों के 'डांसएबिलिटी' स्कोर, 'एकॉस्टिकनेस', लाउडनेस, 'स्पीचनेस', लोकप्रियता और ऊर्जा के बारे में जानकारी शामिल है। इस डेटा में पैटर्न्स की खोज करना दिलचस्प होगा!

![एक टर्नटेबल](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.hi.jpg)

> फोटो <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> द्वारा <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> पर
  
इस पाठ श्रृंखला में, आप क्लस्टरिंग तकनीकों का उपयोग करके डेटा का विश्लेषण करने के नए तरीके खोजेंगे। क्लस्टरिंग विशेष रूप से उपयोगी है जब आपके डेटा सेट में लेबल नहीं होते। यदि इसमें लेबल होते हैं, तो पिछले पाठों में सीखे गए वर्गीकरण तकनीक अधिक उपयोगी हो सकते हैं। लेकिन जब आप बिना लेबल वाले डेटा को समूहबद्ध करना चाहते हैं, तो क्लस्टरिंग पैटर्न्स खोजने का एक शानदार तरीका है।

> कुछ उपयोगी लो-कोड टूल्स हैं जो आपको क्लस्टरिंग मॉडल के साथ काम करने के बारे में सीखने में मदद कर सकते हैं। [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) को इस कार्य के लिए आज़माएं।

## पाठ

1. [क्लस्टरिंग का परिचय](1-Visualize/README.md)
2. [K-Means क्लस्टरिंग](2-K-Means/README.md)

## क्रेडिट्स

ये पाठ 🎶 के साथ [Jen Looper](https://www.twitter.com/jenlooper) द्वारा लिखे गए हैं, और [Rishit Dagli](https://rishit_dagli) और [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) द्वारा सहायक समीक्षाओं के साथ।

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) डेटा सेट Kaggle से प्राप्त किया गया था, जो Spotify से स्क्रैप किया गया था।

K-Means के उपयोगी उदाहरण जिन्होंने इस पाठ को बनाने में मदद की, उनमें शामिल हैं यह [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), यह [प्रारंभिक नोटबुक](https://www.kaggle.com/prashant111/k-means-clustering-with-python), और यह [काल्पनिक NGO उदाहरण](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)।

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  