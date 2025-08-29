<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-08-29T17:44:27+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "mr"
}
-->
# तुमच्या ML मॉडेलसाठी वेब अॅप तयार करा

या अभ्यासक्रमाच्या या विभागात, तुम्हाला एक अनुप्रयुक्त ML विषयाची ओळख करून दिली जाईल: तुमचे Scikit-learn मॉडेल कसे जतन करायचे जेणेकरून ते वेब अॅप्लिकेशनमध्ये अंदाज वर्तवण्यासाठी वापरले जाऊ शकेल. एकदा मॉडेल जतन केल्यानंतर, तुम्ही ते Flask मध्ये तयार केलेल्या वेब अॅपमध्ये कसे वापरायचे ते शिकाल. तुम्ही प्रथम UFO पाहण्याच्या डेटाचा वापर करून एक मॉडेल तयार कराल! त्यानंतर, तुम्ही एक वेब अॅप तयार कराल जो तुम्हाला सेकंदांची संख्या, अक्षांश, आणि रेखांश मूल्य प्रविष्ट करून अंदाज लावण्याची परवानगी देईल की कोणत्या देशाने UFO पाहिल्याचा अहवाल दिला आहे.

![UFO Parking](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.mr.jpg)

फोटो <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">मायकेल हेरन</a> यांनी Unsplash वर <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">प्रकाशित केला</a>.

## धडे

1. [वेब अॅप तयार करा](1-Web-App/README.md)

## श्रेय

"वेब अॅप तयार करा" हे [जेन लूपर](https://twitter.com/jenlooper) यांनी ♥️ सह लिहिले आहे.

♥️ प्रश्नमंजुषा रोहन राज यांनी लिहिल्या आहेत.

डेटासेट [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) वरून घेतले आहे.

वेब अॅप आर्किटेक्चरचा काही भाग [या लेखातून](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) आणि [या रेपो](https://github.com/abhinavsagar/machine-learning-deployment) मधून अभिनव सागर यांनी सुचवला आहे.

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) वापरून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात ठेवा की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर करून निर्माण होणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.