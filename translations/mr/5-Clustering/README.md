<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T17:08:01+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "mr"
}
-->
# मशीन लर्निंगसाठी क्लस्टरिंग मॉडेल्स

क्लस्टरिंग ही मशीन लर्निंगची एक कार्यप्रणाली आहे ज्यामध्ये एकमेकांसारखे दिसणारे ऑब्जेक्ट शोधले जातात आणि त्यांना क्लस्टर्स नावाच्या गटांमध्ये वर्गीकृत केले जाते. मशीन लर्निंगमधील इतर पद्धतींपेक्षा क्लस्टरिंग वेगळे आहे कारण गोष्टी आपोआप घडतात. खरं तर, हे सुपरवाइज्ड लर्निंगच्या अगदी उलट आहे असे म्हणणे योग्य ठरेल.

## प्रादेशिक विषय: नायजेरियन प्रेक्षकांच्या संगीत आवडीसाठी क्लस्टरिंग मॉडेल्स 🎧

नायजेरियाच्या विविध प्रेक्षकांची संगीताची आवडही विविध आहे. Spotify वरून डेटा स्क्रॅप करून (या [लेखातून प्रेरित](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), नायजेरियामध्ये लोकप्रिय असलेल्या काही संगीतावर नजर टाकूया. या डेटासेटमध्ये विविध गाण्यांच्या 'danceability' स्कोअर, 'acousticness', loudness, 'speechiness', लोकप्रियता आणि ऊर्जा याबद्दलचा डेटा समाविष्ट आहे. या डेटामध्ये नमुने शोधणे खूपच मनोरंजक ठरेल!

![एक टर्नटेबल](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.mr.jpg)

> <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> यांनी <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> वर फोटो दिला आहे
  
या धड्यांच्या मालिकेत तुम्ही क्लस्टरिंग तंत्रांचा वापर करून डेटा विश्लेषण करण्याचे नवीन मार्ग शोधाल. क्लस्टरिंग विशेषतः उपयुक्त आहे जेव्हा तुमच्या डेटासेटमध्ये लेबल्स नसतात. जर लेबल्स असतील, तर तुम्ही मागील धड्यांमध्ये शिकलेल्या वर्गीकरण तंत्रे अधिक उपयुक्त ठरू शकतात. परंतु अशा परिस्थितीत जिथे तुम्ही लेबल नसलेल्या डेटाचे गट तयार करू इच्छित असाल, क्लस्टरिंग हे नमुने शोधण्यासाठी एक उत्कृष्ट पद्धत आहे.

> क्लस्टरिंग मॉडेल्ससह काम करण्याबद्दल शिकण्यासाठी उपयुक्त लो-कोड टूल्स उपलब्ध आहेत. [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) वापरून हे कार्य करून पहा.

## धडे

1. [क्लस्टरिंगची ओळख](1-Visualize/README.md)
2. [K-Means क्लस्टरिंग](2-K-Means/README.md)

## श्रेय

हे धडे 🎶 सह [Jen Looper](https://www.twitter.com/jenlooper) यांनी लिहिले असून [Rishit Dagli](https://rishit_dagli) आणि [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) यांनी उपयुक्त पुनरावलोकने केली आहेत.

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) डेटासेट Kaggle वरून Spotify मधून स्क्रॅप केलेले आहे.

K-Means च्या उपयुक्त उदाहरणांमध्ये या [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), या [प्रारंभिक नोटबुक](https://www.kaggle.com/prashant111/k-means-clustering-with-python), आणि या [काल्पनिक NGO उदाहरणाचा](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) समावेश आहे, ज्यांनी हा धडा तयार करण्यात मदत केली.

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) वापरून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात ठेवा की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर करून उद्भवलेल्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.