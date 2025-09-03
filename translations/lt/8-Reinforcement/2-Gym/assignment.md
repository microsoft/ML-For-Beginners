<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-03T18:43:33+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "lt"
}
-->
# Treniruokite „Mountain Car“

[OpenAI Gym](http://gym.openai.com) sukurta taip, kad visi aplinkos modeliai turėtų vienodą API – t. y. tuos pačius metodus `reset`, `step` ir `render`, bei tas pačias **veiksmų erdvės** ir **stebėjimo erdvės** abstrakcijas. Todėl turėtų būti įmanoma pritaikyti tuos pačius stiprinamojo mokymosi algoritmus skirtingoms aplinkoms su minimaliomis kodo pakeitimais.

## „Mountain Car“ aplinka

[„Mountain Car“ aplinka](https://gym.openai.com/envs/MountainCar-v0/) apima automobilį, įstrigusį slėnyje:

Tikslas yra išvažiuoti iš slėnio ir pasiekti vėliavą, atliekant vieną iš šių veiksmų kiekviename žingsnyje:

| Reikšmė | Reikšmė |
|---|---|
| 0 | Pagreitinti į kairę |
| 1 | Nepagreitinti |
| 2 | Pagreitinti į dešinę |

Pagrindinis šios problemos triukas yra tas, kad automobilio variklis nėra pakankamai galingas, kad užvažiuotų ant kalno vienu bandymu. Todėl vienintelis būdas pasiekti tikslą yra važiuoti pirmyn ir atgal, kad sukauptumėte pagreitį.

Stebėjimo erdvė susideda tik iš dviejų reikšmių:

| Nr. | Stebėjimas  | Min | Maks |
|-----|--------------|-----|-----|
|  0  | Automobilio pozicija | -1.2| 0.6 |
|  1  | Automobilio greitis | -0.07 | 0.07 |

Atlygio sistema „Mountain Car“ aplinkoje yra gana sudėtinga:

 * Atlygis 0 suteikiamas, jei agentas pasiekė vėliavą (pozicija = 0.5) ant kalno viršūnės.
 * Atlygis -1 suteikiamas, jei agento pozicija yra mažesnė nei 0.5.

Epizodas baigiasi, jei automobilio pozicija yra didesnė nei 0.5 arba epizodo ilgis viršija 200.
## Instrukcijos

Pritaikykite mūsų stiprinamojo mokymosi algoritmą, kad išspręstumėte „Mountain Car“ problemą. Pradėkite nuo esamo [notebook.ipynb](notebook.ipynb) kodo, pakeiskite aplinką, pakeiskite būsenos diskretizavimo funkcijas ir pabandykite priversti esamą algoritmą mokytis su minimaliomis kodo modifikacijomis. Optimizuokite rezultatą koreguodami hiperparametrus.

> **Pastaba**: Hiperparametrų koregavimas greičiausiai bus reikalingas, kad algoritmas susikoncentruotų. 
## Vertinimo kriterijai

| Kriterijai | Puikiai | Pakankamai | Reikia patobulinimų |
| -------- | --------- | -------- | ----------------- |
|          | Q-Learning algoritmas sėkmingai pritaikytas iš „CartPole“ pavyzdžio su minimaliomis kodo modifikacijomis, kuris sugeba išspręsti vėliavos pasiekimo problemą per mažiau nei 200 žingsnių. | Naujas Q-Learning algoritmas buvo pritaikytas iš interneto, tačiau gerai dokumentuotas; arba esamas algoritmas pritaikytas, bet nepasiekia norimų rezultatų. | Studentas nesugebėjo sėkmingai pritaikyti jokio algoritmo, tačiau padarė reikšmingus žingsnius sprendimo link (įgyvendino būsenos diskretizavimą, Q-lentelės duomenų struktūrą ir pan.). |

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipiame dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Dėl svarbios informacijos rekomenduojame kreiptis į profesionalius vertėjus. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus aiškinimus, kylančius dėl šio vertimo naudojimo.