<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-05T15:08:11+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "sk"
}
-->
# Regresné modely pre strojové učenie
## Regionálna téma: Regresné modely pre ceny tekvíc v Severnej Amerike 🎃

V Severnej Amerike sa tekvice často vyrezávajú do strašidelných tvárí na Halloween. Poďme objaviť viac o týchto fascinujúcich zeleninách!

![jack-o-lanterns](../../../2-Regression/images/jack-o-lanterns.jpg)
> Foto od <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> na <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## Čo sa naučíte

[![Úvod do regresie](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Úvodné video o regresii - Kliknite na sledovanie!")
> 🎥 Kliknite na obrázok vyššie pre rýchle úvodné video k tejto lekcii

Lekcie v tejto sekcii pokrývajú typy regresie v kontexte strojového učenia. Regresné modely môžu pomôcť určiť _vzťah_ medzi premennými. Tento typ modelu dokáže predpovedať hodnoty, ako sú dĺžka, teplota alebo vek, čím odhaľuje vzťahy medzi premennými pri analýze dátových bodov.

V tejto sérii lekcií objavíte rozdiely medzi lineárnou a logistickou regresiou a zistíte, kedy je vhodné použiť jednu alebo druhú.

[![ML pre začiatočníkov - Úvod do regresných modelov pre strojové učenie](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "ML pre začiatočníkov - Úvod do regresných modelov pre strojové učenie")

> 🎥 Kliknite na obrázok vyššie pre krátke video predstavujúce regresné modely.

V tejto skupine lekcií sa pripravíte na začiatok úloh strojového učenia, vrátane konfigurácie Visual Studio Code na správu notebookov, bežného prostredia pre dátových vedcov. Objavíte knižnicu Scikit-learn pre strojové učenie a vytvoríte svoje prvé modely, pričom sa v tejto kapitole zameriate na regresné modely.

> Existujú užitočné nástroje s nízkym kódom, ktoré vám môžu pomôcť naučiť sa pracovať s regresnými modelmi. Vyskúšajte [Azure ML pre túto úlohu](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### Lekcie

1. [Nástroje remesla](1-Tools/README.md)
2. [Správa dát](2-Data/README.md)
3. [Lineárna a polynomiálna regresia](3-Linear/README.md)
4. [Logistická regresia](4-Logistic/README.md)

---
### Kredity

"ML s regresiou" bolo napísané s ♥️ od [Jen Looper](https://twitter.com/jenlooper)

♥️ Prispievatelia kvízov zahŕňajú: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) a [Ornella Altunyan](https://twitter.com/ornelladotcom)

Dataset tekvíc je navrhnutý [týmto projektom na Kaggle](https://www.kaggle.com/usda/a-year-of-pumpkin-prices) a jeho dáta pochádzajú zo [Štandardných správ terminálových trhov pre špeciálne plodiny](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuovaných Ministerstvom poľnohospodárstva Spojených štátov. Pridali sme niekoľko bodov týkajúcich sa farby na základe odrody, aby sme normalizovali distribúciu. Tieto dáta sú vo verejnej doméne.

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.