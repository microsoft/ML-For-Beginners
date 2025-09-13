<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T16:35:50+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "sk"
}
-->
# Úvod do posilňovacieho učenia

Posilňovacie učenie, RL, je považované za jeden zo základných paradigmatov strojového učenia, vedľa učenia s učiteľom a učenia bez učiteľa. RL je o rozhodnutiach: robiť správne rozhodnutia alebo sa aspoň z nich učiť.

Predstavte si, že máte simulované prostredie, napríklad akciový trh. Čo sa stane, ak zavediete určitú reguláciu? Má to pozitívny alebo negatívny efekt? Ak sa stane niečo negatívne, musíte prijať toto _negatívne posilnenie_, poučiť sa z neho a zmeniť smer. Ak je výsledok pozitívny, musíte na tom _pozitívnom posilnení_ stavať.

![Peter a vlk](../../../8-Reinforcement/images/peter.png)

> Peter a jeho priatelia musia uniknúť hladnému vlkovi! Obrázok od [Jen Looper](https://twitter.com/jenlooper)

## Regionálna téma: Peter a vlk (Rusko)

[Peter a vlk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) je hudobná rozprávka napísaná ruským skladateľom [Sergejom Prokofievom](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Je to príbeh o mladom pionierovi Petrovi, ktorý odvážne vyjde z domu na lesnú čistinu, aby prenasledoval vlka. V tejto sekcii budeme trénovať algoritmy strojového učenia, ktoré pomôžu Petrovi:

- **Preskúmať** okolitú oblasť a vytvoriť optimálnu navigačnú mapu
- **Naučiť sa** používať skateboard a udržiavať rovnováhu, aby sa mohol pohybovať rýchlejšie.

[![Peter a vlk](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Kliknite na obrázok vyššie a vypočujte si Peter a vlk od Prokofieva

## Posilňovacie učenie

V predchádzajúcich sekciách ste videli dva príklady problémov strojového učenia:

- **S učiteľom**, kde máme datasety, ktoré naznačujú vzorové riešenia problému, ktorý chceme vyriešiť. [Klasifikácia](../4-Classification/README.md) a [regresia](../2-Regression/README.md) sú úlohy učenia s učiteľom.
- **Bez učiteľa**, kde nemáme označené tréningové dáta. Hlavným príkladom učenia bez učiteľa je [Zhlukovanie](../5-Clustering/README.md).

V tejto sekcii vás zoznámime s novým typom problému učenia, ktorý nevyžaduje označené tréningové dáta. Existuje niekoľko typov takýchto problémov:

- **[Poloučenie s učiteľom](https://wikipedia.org/wiki/Semi-supervised_learning)**, kde máme veľa neoznačených dát, ktoré môžeme použiť na predtréning modelu.
- **[Posilňovacie učenie](https://wikipedia.org/wiki/Reinforcement_learning)**, v ktorom sa agent učí, ako sa správať, vykonávaním experimentov v nejakom simulovanom prostredí.

### Príklad - počítačová hra

Predstavte si, že chcete naučiť počítač hrať hru, napríklad šach alebo [Super Mario](https://wikipedia.org/wiki/Super_Mario). Aby počítač hral hru, potrebujeme, aby predpovedal, aký ťah urobiť v každom stave hry. Aj keď sa to môže zdať ako problém klasifikácie, nie je to tak - pretože nemáme dataset so stavmi a zodpovedajúcimi akciami. Aj keď môžeme mať nejaké dáta, ako existujúce šachové partie alebo záznamy hráčov hrajúcich Super Mario, je pravdepodobné, že tieto dáta nebudú dostatočne pokrývať veľké množstvo možných stavov.

Namiesto hľadania existujúcich herných dát je **Posilňovacie učenie** (RL) založené na myšlienke *nechať počítač hrať* mnohokrát a pozorovať výsledok. Na aplikáciu posilňovacieho učenia potrebujeme dve veci:

- **Prostredie** a **simulátor**, ktoré nám umožnia hrať hru mnohokrát. Tento simulátor by definoval všetky pravidlá hry, ako aj možné stavy a akcie.

- **Funkciu odmeny**, ktorá nám povie, ako dobre sme si počínali počas každého ťahu alebo hry.

Hlavný rozdiel medzi inými typmi strojového učenia a RL je ten, že v RL zvyčajne nevieme, či vyhráme alebo prehráme, kým nedokončíme hru. Preto nemôžeme povedať, či je určitý ťah sám o sebe dobrý alebo nie - odmenu dostaneme až na konci hry. Naším cieľom je navrhnúť algoritmy, ktoré nám umožnia trénovať model za neistých podmienok. Naučíme sa o jednom RL algoritme nazývanom **Q-learning**.

## Lekcie

1. [Úvod do posilňovacieho učenia a Q-Learningu](1-QLearning/README.md)
2. [Používanie simulačného prostredia Gym](2-Gym/README.md)

## Kredity

"Úvod do posilňovacieho učenia" napísal s ♥️ [Dmitry Soshnikov](http://soshnikov.com)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.