<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T21:46:14+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "sv"
}
-->
# Bygg en webbapp för att använda din ML-modell

I den här delen av kursen kommer du att introduceras till ett tillämpat ML-ämne: hur du sparar din Scikit-learn-modell som en fil som kan användas för att göra förutsägelser i en webbapplikation. När modellen är sparad kommer du att lära dig hur du använder den i en webbapp byggd med Flask. Du kommer först att skapa en modell med hjälp av data som handlar om UFO-observationer! Därefter bygger du en webbapp som låter dig mata in ett antal sekunder tillsammans med en latitud- och longitudvärde för att förutsäga vilket land som rapporterade att de såg ett UFO.

![UFO Parkering](../../../3-Web-App/images/ufo.jpg)

Foto av <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> på <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Lektioner

1. [Bygg en webbapp](1-Web-App/README.md)

## Krediter

"Bygg en webbapp" skrevs med ♥️ av [Jen Looper](https://twitter.com/jenlooper).

♥️ Quizen skrevs av Rohan Raj.

Datasettet kommer från [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

Webbappens arkitektur föreslogs delvis av [denna artikel](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) och [detta repo](https://github.com/abhinavsagar/machine-learning-deployment) av Abhinav Sagar.

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.