<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T16:35:31+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "hu"
}
-->
# Bevezetés a megerősítéses tanulásba

A megerősítéses tanulás, azaz RL, a gépi tanulás egyik alapvető paradigmájának számít, a felügyelt tanulás és a nem felügyelt tanulás mellett. Az RL a döntésekről szól: helyes döntések meghozatala vagy legalább tanulás a hibákból.

Képzeld el, hogy van egy szimulált környezeted, például a tőzsde. Mi történik, ha bevezetsz egy adott szabályozást? Pozitív vagy negatív hatása lesz? Ha valami negatív történik, akkor ezt _negatív megerősítésként_ kell értelmezned, tanulnod kell belőle, és változtatnod kell az irányon. Ha pozitív eredményt érünk el, akkor arra kell építenünk, _pozitív megerősítésként_.

![Péter és a farkas](../../../8-Reinforcement/images/peter.png)

> Péternek és barátainak menekülniük kell az éhes farkas elől! Kép: [Jen Looper](https://twitter.com/jenlooper)

## Regionális téma: Péter és a farkas (Oroszország)

A [Péter és a farkas](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) egy zenés mese, amelyet egy orosz zeneszerző, [Szergej Prokofjev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) írt. A történet egy fiatal pionírról, Péterről szól, aki bátran kimegy az erdő tisztására, hogy üldözze a farkast. Ebben a részben gépi tanulási algoritmusokat fogunk tanítani, amelyek segítenek Péternek:

- **Felfedezni** a környező területet és optimális navigációs térképet készíteni.
- **Megtanulni** gördeszkázni és egyensúlyozni rajta, hogy gyorsabban tudjon közlekedni.

[![Péter és a farkas](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Kattints a fenti képre, hogy meghallgasd Prokofjev Péter és a farkas című művét.

## Megerősítéses tanulás

Az előző részekben két gépi tanulási problémát láttál:

- **Felügyelt tanulás**, ahol vannak adataink, amelyek mintamegoldásokat javasolnak az általunk megoldani kívánt problémára. A [klasszifikáció](../4-Classification/README.md) és a [regresszió](../2-Regression/README.md) felügyelt tanulási feladatok.
- **Nem felügyelt tanulás**, ahol nincsenek címkézett tanulási adatok. A nem felügyelt tanulás fő példája a [klaszterezés](../5-Clustering/README.md).

Ebben a részben egy új tanulási problémát mutatunk be, amely nem igényel címkézett tanulási adatokat. Az ilyen problémák több típusa létezik:

- **[Félig felügyelt tanulás](https://wikipedia.org/wiki/Semi-supervised_learning)**, ahol rengeteg címkézetlen adat áll rendelkezésre, amelyet felhasználhatunk a modell előzetes betanítására.
- **[Megerősítéses tanulás](https://wikipedia.org/wiki/Reinforcement_learning)**, amelyben egy ügynök kísérletek végrehajtásával tanulja meg, hogyan viselkedjen egy szimulált környezetben.

### Példa - számítógépes játék

Tegyük fel, hogy meg akarod tanítani a számítógépet egy játék, például sakk vagy [Super Mario](https://wikipedia.org/wiki/Super_Mario) játszására. Ahhoz, hogy a számítógép játszani tudjon, meg kell tanítanunk neki, hogy minden játékállapotban megjósolja, melyik lépést tegye meg. Bár ez elsőre klasszifikációs problémának tűnhet, valójában nem az – mivel nincs olyan adatbázisunk, amely állapotokat és hozzájuk tartozó lépéseket tartalmazna. Bár lehet, hogy van némi adatunk, például meglévő sakkjátszmák vagy Super Mario játékosok felvételei, valószínű, hogy ezek az adatok nem fedik le eléggé a lehetséges állapotok nagy számát.

Ahelyett, hogy meglévő játékadatokat keresnénk, a **megerősítéses tanulás** (RL) azon az ötleten alapul, hogy *a számítógépet sokszor játszatjuk*, és megfigyeljük az eredményt. Így a megerősítéses tanulás alkalmazásához két dologra van szükségünk:

- **Egy környezetre** és **egy szimulátorra**, amely lehetővé teszi, hogy sokszor játszunk. Ez a szimulátor határozza meg az összes játékszabályt, valamint a lehetséges állapotokat és lépéseket.

- **Egy jutalomfüggvényre**, amely megmondja, mennyire teljesítettünk jól minden lépés vagy játék során.

A fő különbség a többi gépi tanulási típus és az RL között az, hogy az RL-ben általában nem tudjuk, hogy nyerünk vagy veszítünk, amíg be nem fejezzük a játékot. Ezért nem mondhatjuk meg, hogy egy bizonyos lépés önmagában jó-e vagy sem – csak a játék végén kapunk jutalmat. A célunk olyan algoritmusok tervezése, amelyek lehetővé teszik, hogy bizonytalan körülmények között is modelleket tanítsunk. Megismerkedünk egy RL algoritmussal, amelyet **Q-learningnek** hívnak.

## Leckék

1. [Bevezetés a megerősítéses tanulásba és a Q-Learningbe](1-QLearning/README.md)
2. [Egy gym szimulációs környezet használata](2-Gym/README.md)

## Köszönetnyilvánítás

"A megerősítéses tanulás bevezetése" ♥️-vel készült [Dmitry Soshnikov](http://soshnikov.com) által.

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.