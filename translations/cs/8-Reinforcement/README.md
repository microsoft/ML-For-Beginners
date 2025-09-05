<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T01:03:44+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "cs"
}
-->
# Úvod do posilovaného učení

Posilované učení, RL, je považováno za jeden ze základních paradigmat strojového učení, vedle učení s učitelem a učení bez učitele. RL se zaměřuje na rozhodování: poskytování správných rozhodnutí nebo alespoň učení se z nich.

Představte si, že máte simulované prostředí, například akciový trh. Co se stane, pokud zavedete určitou regulaci? Má to pozitivní nebo negativní dopad? Pokud se stane něco negativního, musíte vzít tento _negativní posilovací podnět_, poučit se z něj a změnit směr. Pokud je výsledek pozitivní, musíte na tomto _pozitivním posilovacím podnětu_ stavět.

![peter a vlk](../../../8-Reinforcement/images/peter.png)

> Petr a jeho přátelé musí utéct hladovému vlkovi! Obrázek od [Jen Looper](https://twitter.com/jenlooper)

## Regionální téma: Petr a vlk (Rusko)

[Petr a vlk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) je hudební pohádka napsaná ruským skladatelem [Sergejem Prokofjevem](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Je to příběh o mladém pionýrovi Petrovi, který odvážně vyjde z domu na lesní mýtinu, aby pronásledoval vlka. V této části budeme trénovat algoritmy strojového učení, které Petrovi pomohou:

- **Prozkoumat** okolní oblast a vytvořit optimální navigační mapu
- **Naučit se** jezdit na skateboardu a udržovat rovnováhu, aby se mohl pohybovat rychleji.

[![Petr a vlk](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klikněte na obrázek výše a poslechněte si Petra a vlka od Prokofjeva

## Posilované učení

V předchozích částech jste viděli dva příklady problémů strojového učení:

- **S učitelem**, kde máme datové sady, které naznačují vzorová řešení problému, který chceme vyřešit. [Klasifikace](../4-Classification/README.md) a [regrese](../2-Regression/README.md) jsou úkoly učení s učitelem.
- **Bez učitele**, kde nemáme označená tréninková data. Hlavním příkladem učení bez učitele je [shlukování](../5-Clustering/README.md).

V této části vás seznámíme s novým typem problému učení, který nevyžaduje označená tréninková data. Existuje několik typů takových problémů:

- **[Poloučení s učitelem](https://wikipedia.org/wiki/Semi-supervised_learning)**, kde máme velké množství neoznačených dat, která mohou být použita k předtrénování modelu.
- **[Posilované učení](https://wikipedia.org/wiki/Reinforcement_learning)**, při kterém se agent učí, jak se chovat, prováděním experimentů v nějakém simulovaném prostředí.

### Příklad - počítačová hra

Představte si, že chcete naučit počítač hrát hru, například šachy nebo [Super Mario](https://wikipedia.org/wiki/Super_Mario). Aby počítač mohl hrát hru, potřebujeme, aby předpověděl, jaký tah udělat v každém herním stavu. I když se to může zdát jako problém klasifikace, není tomu tak - protože nemáme datovou sadu se stavy a odpovídajícími akcemi. I když můžeme mít nějaká data, jako jsou existující šachové partie nebo záznamy hráčů hrajících Super Mario, je pravděpodobné, že tato data nebudou dostatečně pokrývat velké množství možných stavů.

Místo hledání existujících herních dat je **posilované učení** (RL) založeno na myšlence *nechat počítač hrát* mnohokrát a pozorovat výsledek. Abychom mohli aplikovat posilované učení, potřebujeme dvě věci:

- **Prostředí** a **simulátor**, které nám umožní hru hrát mnohokrát. Tento simulátor by definoval všechna pravidla hry, stejně jako možné stavy a akce.

- **Funkci odměny**, která nám řekne, jak dobře jsme si vedli během každého tahu nebo hry.

Hlavní rozdíl mezi ostatními typy strojového učení a RL je ten, že v RL obvykle nevíme, zda vyhrajeme nebo prohrajeme, dokud nedokončíme hru. Nemůžeme tedy říci, zda je určitý tah sám o sobě dobrý nebo ne - odměnu dostáváme až na konci hry. Naším cílem je navrhnout algoritmy, které nám umožní trénovat model za nejistých podmínek. Naučíme se o jednom RL algoritmu nazvaném **Q-learning**.

## Lekce

1. [Úvod do posilovaného učení a Q-Learningu](1-QLearning/README.md)
2. [Použití simulovaného prostředí Gym](2-Gym/README.md)

## Poděkování

"Úvod do posilovaného učení" byl napsán s ♥️ od [Dmitry Soshnikov](http://soshnikov.com)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.