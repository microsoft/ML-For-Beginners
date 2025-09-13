<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-05T15:07:54+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "hu"
}
-->
# Regressziós modellek gépi tanuláshoz
## Regionális téma: Regressziós modellek tökárakhoz Észak-Amerikában 🎃

Észak-Amerikában a tököket gyakran ijesztő arcokká faragják Halloween alkalmából. Fedezzük fel ezeket a lenyűgöző zöldségeket!

![jack-o-lanterns](../../../2-Regression/images/jack-o-lanterns.jpg)
> Fotó: <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> az <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> oldalán
  
## Amit megtanulsz

[![Bevezetés a regresszióba](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Bevezető videó a regresszióhoz - Kattints a megtekintéshez!")
> 🎥 Kattints a fenti képre egy rövid bevezető videóért ehhez a leckéhez

Az ebben a részben található leckék a regresszió típusait tárgyalják a gépi tanulás kontextusában. A regressziós modellek segíthetnek meghatározni a _kapcsolatot_ a változók között. Ez a modell képes előre jelezni olyan értékeket, mint például hosszúság, hőmérséklet vagy életkor, így feltárva a változók közötti összefüggéseket az adatok elemzése során.

Ebben a leckesorozatban megismerheted a lineáris és logisztikus regresszió közötti különbségeket, valamint azt, hogy mikor érdemes az egyiket a másik helyett használni.

[![ML kezdőknek - Bevezetés a regressziós modellekbe gépi tanuláshoz](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "ML kezdőknek - Bevezetés a regressziós modellekbe gépi tanuláshoz")

> 🎥 Kattints a fenti képre egy rövid videóért, amely bemutatja a regressziós modelleket.

Ebben a leckesorozatban felkészülsz a gépi tanulási feladatok megkezdésére, beleértve a Visual Studio Code konfigurálását notebookok kezelésére, amely a data scientist-ek által használt közös környezet. Megismered a Scikit-learn könyvtárat, amely a gépi tanuláshoz készült, és elkészíted az első modelljeidet, különös tekintettel a regressziós modellekre ebben a fejezetben.

> Hasznos, kevés kódolást igénylő eszközök állnak rendelkezésre, amelyek segítenek a regressziós modellekkel való munka elsajátításában. Próbáld ki [Azure ML-t ehhez a feladathoz](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### Leckék

1. [Eszközök használata](1-Tools/README.md)
2. [Adatok kezelése](2-Data/README.md)
3. [Lineáris és polinomiális regresszió](3-Linear/README.md)
4. [Logisztikus regresszió](4-Logistic/README.md)

---
### Köszönetnyilvánítás

"ML regresszióval" szívvel ♥️ írta [Jen Looper](https://twitter.com/jenlooper)

♥️ A kvíz közreműködői: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) és [Ornella Altunyan](https://twitter.com/ornelladotcom)

A tök adatállományt [ez a Kaggle projekt](https://www.kaggle.com/usda/a-year-of-pumpkin-prices) javasolta, és az adatok a [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) oldalról származnak, amelyet az Egyesült Államok Mezőgazdasági Minisztériuma terjeszt. Néhány pontot hozzáadtunk a szín alapján, hogy normalizáljuk az eloszlást. Ezek az adatok közkincsnek számítanak.

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.