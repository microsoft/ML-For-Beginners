<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-12-19T12:59:36+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "ml"
}
-->
# നിങ്ങളുടെ ML മോഡൽ ഉപയോഗിക്കാൻ ഒരു വെബ് ആപ്പ് നിർമ്മിക്കുക

പാഠ്യപദ്ധതിയുടെ ഈ ഭാഗത്തിൽ, നിങ്ങൾക്ക് പ്രയോഗാത്മകമായ ഒരു ML വിഷയം പരിചയപ്പെടുത്തും: നിങ്ങളുടെ Scikit-learn മോഡൽ ഫയലായി സേവ് ചെയ്യുന്നത്, അത് വെബ് ആപ്ലിക്കേഷനിൽ പ്രവചനങ്ങൾ നടത്താൻ ഉപയോഗിക്കാവുന്നതാണ്. മോഡൽ സേവ് ചെയ്ത ശേഷം, Flask-ൽ നിർമ്മിച്ച ഒരു വെബ് ആപ്പിൽ അത് എങ്ങനെ ഉപയോഗിക്കാമെന്ന് നിങ്ങൾ പഠിക്കും. ആദ്യം, UFO കാണപ്പെട്ടതുമായി ബന്ധപ്പെട്ട ചില ഡാറ്റ ഉപയോഗിച്ച് ഒരു മോഡൽ നിങ്ങൾ സൃഷ്ടിക്കും! പിന്നീട്, ഒരു വെബ് ആപ്പ് നിർമ്മിക്കും, അതിലൂടെ നിങ്ങൾ സെക്കൻഡുകളുടെ എണ്ണം, അക്ഷാംശവും രേഖാംശവും നൽകുമ്പോൾ ഏത് രാജ്യമാണ് UFO കണ്ടതായി റിപ്പോർട്ട് ചെയ്തതെന്ന് പ്രവചിക്കാനാകും.

![UFO Parking](../../../translated_images/ml/ufo.9e787f5161da9d4d.webp)

ഫോട്ടോ <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> എന്നവരിൽ നിന്നാണ് <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## പാഠങ്ങൾ

1. [Build a Web App](1-Web-App/README.md)

## ക്രെഡിറ്റുകൾ

"Build a Web App" ♥️ ഉപയോഗിച്ച് എഴുതിയത് [Jen Looper](https://twitter.com/jenlooper) ആണ്.

♥️ ക്വിസുകൾ എഴുതിയത് Rohan Raj ആണ്.

ഡാറ്റാസെറ്റ് [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) നിന്നാണ് ലഭിച്ചത്.

വെബ് ആപ്പ് ആർക്കിടെക്ചർ ഭാഗികമായി [ഈ ലേഖനം](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) ഉം [ഈ റിപോ](https://github.com/abhinavsagar/machine-learning-deployment) ഉം Abhinav Sagar നിർദ്ദേശിച്ചവയാണ്.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**അസൂയാപത്രം**:  
ഈ രേഖ AI വിവർത്തന സേവനം [Co-op Translator](https://github.com/Azure/co-op-translator) ഉപയോഗിച്ച് വിവർത്തനം ചെയ്തതാണ്. നാം കൃത്യതയ്ക്ക് ശ്രമിച്ചിട്ടുണ്ടെങ്കിലും, യന്ത്രം ചെയ്ത വിവർത്തനങ്ങളിൽ പിശകുകൾ അല്ലെങ്കിൽ തെറ്റുകൾ ഉണ്ടാകാമെന്ന് ദയവായി ശ്രദ്ധിക്കുക. അതിന്റെ മാതൃഭാഷയിലുള്ള യഥാർത്ഥ രേഖ അധികാരപരമായ ഉറവിടമായി കണക്കാക്കണം. നിർണായക വിവരങ്ങൾക്ക്, പ്രൊഫഷണൽ മനുഷ്യ വിവർത്തനം ശുപാർശ ചെയ്യപ്പെടുന്നു. ഈ വിവർത്തനം ഉപയോഗിക്കുന്നതിൽ നിന്നുണ്ടാകുന്ന ഏതെങ്കിലും തെറ്റിദ്ധാരണകൾക്കോ തെറ്റായ വ്യാഖ്യാനങ്ങൾക്കോ ഞങ്ങൾ ഉത്തരവാദികളല്ല.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->