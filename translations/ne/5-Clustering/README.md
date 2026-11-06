# मेसिन शिक्षणका लागि क्लस्टरिङ मोडेलहरू

क्लस्टरिङ मेसिन शिक्षणको एउटा कार्य हो जहाँ एउटै प्रकारका वस्तुहरू एकअर्का जस्तै देखिने र समूहमा राखिने खोजिन्छन् जसलाई क्लस्टर भनिन्छ। मेसिन शिक्षणका अन्य तरिकाहरूबाट क्लस्टरिङ फरक पर्छ किनकि यसमा कुरा स्वचालित रूपमा हुन्छ, वस्तुतः यसलाई सुपरवाइज्ड शिक्षणको विपरित भन्न सकिन्छ।

## क्षेत्रीय विषय: नाइजेरियाली दर्शकको सङ्गीत रुचिका लागि क्लस्टरिङ मोडेलहरू 🎧

नाइजेरियाका विविध दर्शकहरूसँग विविध संगीत रुचिहरू छन्। Spotify बाट सङ्कलित डेटा प्रयोग गर्दै (यो [यो लेखबाट](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) प्रेरित), नाइजेरियामा लोकप्रिय केही संगीतहरू हेरौं। यस डेटासेटमा विभिन्न गीतहरूको 'डान्सबिलिटी' स्कोर, 'अकोस्टिकनेस', आवाजको तीव्रता, 'स्पीचिनेस', लोकप्रियता र ऊर्जा जस्ता डेटा समावेश छन्। यस डेटामा पैटर्नहरू फेला पार्नु रोचक हुनेछ!

![एउटा टर्नटेबल](../../../translated_images/ne/turntable.f2b86b13c53302dc.webp)

> फोटो <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">मार्सेला लसकोस्की</a> द्वारा <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">अन्सप्ल्याश</a> मा
  
यस पाठ श्रृंखलामा, तपाईं क्लस्टरिङ प्रविधिहरूको प्रयोग गरेर डेटा विश्लेषण गर्ने नयाँ तरिकाहरू पत्ता लगाउनु हुनेछ। क्लस्टरिङ तब विशेष उपयोगी हुन्छ जब तपाईंको डेटासेटमा लेबलहरू हुँदैनन्। यदि लेबलहरू छन् भने, पूर्वका पाठहरूमा सिकेका वर्गीकरण प्रविधिहरू बढी उपयोगी हुन सक्छन्। तर जब तपाईंले बिना लेबलको डेटा समूह बनाउन खोज्नुहुन्छ, क्लस्टरिङ पैटर्नहरू खोज्न उत्कृष्ट उपाय हो।

> क्लस्टरिङ मोडेलहरूसँग काम गर्न सिक्न उपयोगी लो-कोड उपकरणहरू उपलब्ध छन्। यस कार्यका लागि [Azure ML प्रयोग गर्नुहोस्](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## पाठहरू

1. [क्लस्टरिङमा परिचय](1-Visualize/README.md)
2. [के-मीन्स क्लस्टरिङ](2-K-Means/README.md)

## श्रेयहरू

यी पाठहरू 🎶 [जेन लूपर](https://www.twitter.com/jenlooper) द्वारा लेखिएको हो र सहयोगी समीक्षा [ऋषित डागली](https://rishit_dagli/) र [मुहम्मद साकिब खान इनान](https://twitter.com/Sakibinan) द्वारा गरिएको हो।

[नाइजेरियाली गीतहरू](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) को डेटासेट Kaggle बाट Spotify बाट सङ्कलित गरिएको हो।

यस पाठ निर्माणमा सहयोग पुर्‍याएका उपयोगी के-मीन्स उदाहरणहरूमा यो [आइरिस अन्वेषण](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), यो [परिचयात्मक नोटबुक](https://www.kaggle.com/prashant111/k-means-clustering-with-python), र यो [कल्पनात्मक एनजीओ उदाहरण](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) समावेश छन्।

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**अस्वीकरण**:
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको हो। हामी सही हुन प्रयास गर्छौं, तर कृपया जानकार हुनुस् कि स्वचालित अनुवादमा त्रुटिहरू वा अशुद्धताहरू हुन सक्छन्। मूल दस्तावेज़ यसको मूल भाषामा आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीका लागि व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न कुनै पनि गलत बुझाइ वा त्रुटिको लागि हामी जिम्मेवार छैनौं।
<!-- CO-OP TRANSLATOR DISCLAIMER END -->