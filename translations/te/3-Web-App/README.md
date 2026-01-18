<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-12-19T12:59:15+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "te"
}
-->
# మీ ML మోడల్‌ను ఉపయోగించడానికి వెబ్ యాప్‌ను నిర్మించండి

ఈ పాఠ్యాంశంలో, మీరు ఒక అన్వయించిన ML అంశాన్ని పరిచయం చేయబడతారు: మీ Scikit-learn మోడల్‌ను ఫైల్‌గా ఎలా సేవ్ చేయాలో, అది వెబ్ అప్లికేషన్‌లో అంచనాలు చేయడానికి ఉపయోగించవచ్చు. మోడల్ సేవ్ అయిన తర్వాత, మీరు దాన్ని Flaskలో నిర్మించిన వెబ్ యాప్‌లో ఎలా ఉపయోగించాలో నేర్చుకుంటారు. మీరు మొదట UFO సాక్ష్యాల గురించి ఉన్న కొన్ని డేటాతో ఒక మోడల్‌ను సృష్టిస్తారు! ఆ తర్వాత, మీరు సెకన్ల సంఖ్య, అక్షాంశం మరియు రేఖాంశం విలువలను ఇన్‌పుట్‌గా ఇచ్చి ఏ దేశం UFO చూసిందని అంచనా వేయగల వెబ్ యాప్‌ను నిర్మిస్తారు.

![UFO Parking](../../../translated_images/te/ufo.9e787f5161da9d4d.webp)

ఫోటో <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> ద్వారా <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## పాఠాలు

1. [వెబ్ యాప్‌ను నిర్మించండి](1-Web-App/README.md)

## క్రెడిట్స్

"వెబ్ యాప్‌ను నిర్మించండి" ను ♥️ తో [Jen Looper](https://twitter.com/jenlooper) రాశారు.

♥️ క్విజ్‌లు రోహన్ రాజ్ రాశారు.

డేటాసెట్ [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) నుండి తీసుకోబడింది.

వెబ్ యాప్ ఆర్కిటెక్చర్ భాగంగా సూచించబడింది [ఈ ఆర్టికల్](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) మరియు [ఈ రిపో](https://github.com/abhinavsagar/machine-learning-deployment) Abhinav Sagar ద్వారా.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**అస్పష్టత**:  
ఈ పత్రాన్ని AI అనువాద సేవ [Co-op Translator](https://github.com/Azure/co-op-translator) ఉపయోగించి అనువదించబడింది. మేము ఖచ్చితత్వానికి ప్రయత్నించినప్పటికీ, ఆటోమేటెడ్ అనువాదాల్లో పొరపాట్లు లేదా తప్పిదాలు ఉండవచ్చు. మూల పత్రం దాని స్వదేశీ భాషలో అధికారిక మూలంగా పరిగణించాలి. ముఖ్యమైన సమాచారానికి, ప్రొఫెషనల్ మానవ అనువాదం సిఫార్సు చేయబడుతుంది. ఈ అనువాదం వాడకంలో ఏర్పడిన ఏవైనా అపార్థాలు లేదా తప్పుదారితీసే అర్థాలు కోసం మేము బాధ్యత వహించము.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->