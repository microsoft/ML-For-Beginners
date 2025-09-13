<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T17:08:13+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "ne"
}
-->
# मेसिन लर्निङका लागि क्लस्टरिङ मोडेलहरू

क्लस्टरिङ एक मेसिन लर्निङ कार्य हो जहाँ समान विशेषताहरू भएका वस्तुहरूलाई पहिचान गरी समूहहरूमा विभाजन गरिन्छ, जसलाई क्लस्टर भनिन्छ। अन्य मेसिन लर्निङ विधिहरूसँग तुलना गर्दा, क्लस्टरिङ स्वतः हुन्छ। वास्तवमा, यो सुपरभाइज्ड लर्निङको विपरीत हो भन्नु उचित हुन्छ।

## क्षेत्रीय विषय: नाइजेरियन दर्शकहरूको संगीत रुचिका लागि क्लस्टरिङ मोडेलहरू 🎧

नाइजेरियाको विविध दर्शकहरूको संगीत रुचि पनि विविध छ। Spotify बाट सङ्कलित डाटाको प्रयोग गरेर (यस [लेख](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) बाट प्रेरित), नाइजेरियामा लोकप्रिय केही संगीतलाई हेरौं। यो डेटासेटमा विभिन्न गीतहरूको 'डान्सएबिलिटी' स्कोर, 'एकुस्टिकनेस', लाउडनेस, 'स्पिचिनेस', लोकप्रियता र ऊर्जा सम्बन्धी डाटा समावेश छ। यस डाटामा पैटर्नहरू पत्ता लगाउनु रोचक हुनेछ!

![एक टर्नटेबल](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.ne.jpg)

> फोटो <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">मार्सेला लास्कोस्की</a> द्वारा <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">अनस्प्ल्यास</a> मा
  
यस पाठहरूको श्रृंखलामा, तपाईं क्लस्टरिङ प्रविधिहरू प्रयोग गरेर डाटा विश्लेषण गर्ने नयाँ तरिकाहरू पत्ता लगाउनुहुनेछ। क्लस्टरिङ विशेष गरी उपयोगी हुन्छ जब तपाईंको डेटासेटमा लेबलहरू हुँदैनन्। यदि लेबलहरू छन् भने, तपाईंले अघिल्लो पाठहरूमा सिकेका वर्गीकरण प्रविधिहरू अधिक उपयोगी हुन सक्छ। तर, जब तपाईं लेबल नभएको डाटालाई समूहबद्ध गर्न खोज्दै हुनुहुन्छ, क्लस्टरिङ पैटर्नहरू पत्ता लगाउनको लागि उत्कृष्ट तरिका हो।

> क्लस्टरिङ मोडेलहरूसँग काम गर्न सिक्न उपयोगी लो-कोड उपकरणहरू उपलब्ध छन्। यस कार्यका लागि [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) प्रयास गर्नुहोस्।

## पाठहरू

1. [क्लस्टरिङको परिचय](1-Visualize/README.md)
2. [के-मिन्स क्लस्टरिङ](2-K-Means/README.md)

## श्रेय

यी पाठहरू 🎶 सहित [जेन लूपर](https://www.twitter.com/jenlooper) द्वारा लेखिएका हुन्, र [रिशित डागली](https://rishit_dagli) र [मुहम्मद साकिब खान इनान](https://twitter.com/Sakibinan) द्वारा सहायक समीक्षाहरू गरिएको छ।

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) डेटासेट Kaggle बाट Spotify बाट सङ्कलन गरिएको हो।

यी पाठहरू तयार गर्न मद्दत गर्ने उपयोगी के-मिन्स उदाहरणहरूमा यो [आइरिस अन्वेषण](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), यो [परिचयात्मक नोटबुक](https://www.kaggle.com/prashant111/k-means-clustering-with-python), र यो [काल्पनिक एनजीओ उदाहरण](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) समावेश छन्।

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादमा त्रुटिहरू वा अशुद्धताहरू हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।