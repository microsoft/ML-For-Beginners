# मशीन लर्निंगसाठी क्लस्टरिंग मॉडेल्स

क्लस्टरिंग ही एक मशीन लर्निंगची कार्य आहे जिथे सारखे दिसणारे ऑब्जेक्ट्स शोधून त्यांना क्लस्टर्स म्हणणाऱ्या गटांमध्ये विभागले जाते. मशीन लर्निंगमधील इतर पद्धतींपेक्षा क्लस्टरिंग वेगळी आहे कारण येथे गोष्टी आपोआप घडतात, खऱ्या अर्थाने ही supervised learning च्या उलट आहे असे म्हणायला हरकत नाही.

## स्थानिक विषय: नायजेरियन प्रेक्षकांच्या संगीत आवडीनुसार क्लस्टरिंग मॉडेल्स 🎧

नायजेरियाच्या विविध प्रेक्षकांची संगीत आवडही वेगवेगळी आहे. Spotify वरून स्क्रॅप केलेल्या डेटाचा वापर करून (या [आलेखाद्वारे](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) प्रेरित होऊन), नायजेरियामध्ये लोकप्रिय असलेल्या काही संगीताचा आढावा घेऊया. या डेटासेटमध्ये विविध गाण्यांचे 'danceability' स्कोर, 'acousticness', आवाजाचा तीव्रता, 'speechiness', लोकप्रियता आणि ऊर्जा यांची माहिती आहे. या डेटामध्ये नमुने शोधणे मनोरंजक ठरेल!

![एका टर्नटेबलचा फोटो](../../../translated_images/mr/turntable.f2b86b13c53302dc.webp)

> <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">मार्सेला लासकोस्की</a> यांनी <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">अनस्प्लॅशवर</a> घेतलेला फोटो
  
या धड्यांच्या मालिकेत, तुम्हाला क्लस्टरिंग तंत्रांचा वापर करून डेटाचे नवीन प्रकाराने विश्लेषण करायला शिकवले जाईल. जेव्हा तुमच्या डेटासेटमध्ये लेबले नसतात तेव्हा क्लस्टरिंग विशेष प्रभावी ठरते. जर लेबले असतील, तर तुम्हाला आधीच्या धड्यांमध्ये शिकवलेल्या वर्गीकरण तंत्रांचा जास्त फायदा होऊ शकतो. पण जेव्हा तुम्हाला अनलेबल केलेला डेटा गटात विभागायचा असेल, तेव्हा क्लस्टरिंग वापरून नमुने शोधणे खूपच उपयुक्त आहे.

> कार्य करण्यासाठी क्लस्टरिंग मॉडेल्ससह काम शिकण्यास मदत करणाऱ्या कमी कोडिंगच्या उपयुक्त साधनांचा वापर करा. या कार्यासाठी [Azure ML वापरून पाहा](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## धडे

1. [क्लस्टरिंगची ओळख](1-Visualize/README.md)
2. [K-Means क्लस्टरिंग](2-K-Means/README.md)

## श्रेय

हे धडे 🎶 [जेन लूपर](https://www.twitter.com/jenlooper) यांनी लिहिले असून यासाठी [ऋषित डागळी](https://rishit_dagli/) आणि [मुहम्मद साकिब खान इनान](https://twitter.com/Sakibinan) यांनी उपयुक्त अभिप्राय दिले.

[नायजेरियन गाणी](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) हा डेटासेट कॅगल वरून Spotify वरून स्क्रॅप करून मिळविला गेला आहे.

या धड्याच्या निर्मितीस मदत करणारे उपयुक्त K-Means उदाहरणांमध्ये हा [अयर्स (iris) शोध](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), हा [प्रास्ताविक नोटबुक](https://www.kaggle.com/prashant111/k-means-clustering-with-python), आणि हे [काल्पनिक NGO उदाहरण](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) यांचा समावेश आहे.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**अस्वीकरण**:
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) चा वापर करून अनुवादित केला आहे. जरी आम्ही अचूकतेसाठी प्रयत्न करतो, तरी कृपया लक्षात घ्या की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेची कमतरता असू शकते. मूळ दस्तऐवज त्याच्या मूळ भाषेत अधिकृत स्रोत मानला पाहिजे. महत्त्वाची माहिती असल्यास, व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराच्या वापरामुळे उद्भवणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थलावणीसाठी आम्ही जबाबदार नाही.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->