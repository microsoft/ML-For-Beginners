<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T16:12:01+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "hu"
}
-->
# Készíts egy webalkalmazást az ML modelled használatához

A tananyag ezen részében egy alkalmazott gépi tanulási témával ismerkedhetsz meg: hogyan lehet a Scikit-learn modelledet fájlként elmenteni, amelyet egy webalkalmazásban használhatsz előrejelzések készítésére. Miután a modellt elmentetted, megtanulod, hogyan használd egy Flask-ben épített webalkalmazásban. Először létrehozol egy modellt egy olyan adathalmaz alapján, amely UFO-észlelésekről szól! Ezután építesz egy webalkalmazást, amely lehetővé teszi, hogy megadj egy másodpercértéket, valamint egy szélességi és hosszúsági koordinátát, hogy előre jelezd, melyik ország jelentett UFO-észlelést.

![UFO Parkolás](../../../3-Web-App/images/ufo.jpg)

Fotó: <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> az <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> oldalán

## Leckék

1. [Webalkalmazás készítése](1-Web-App/README.md)

## Köszönetnyilvánítás

A "Webalkalmazás készítése" leckét ♥️-vel írta [Jen Looper](https://twitter.com/jenlooper).

♥️ A kvízeket Rohan Raj írta.

Az adathalmaz a [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings) oldalról származik.

A webalkalmazás architektúráját részben [ez a cikk](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) és [ez a repo](https://github.com/abhinavsagar/machine-learning-deployment) javasolta, amelyet Abhinav Sagar készített.

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.