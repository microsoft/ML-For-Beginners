<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T16:04:10+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "sk"
}
-->
# Techniky strojového učenia

Proces vytvárania, používania a udržiavania modelov strojového učenia a dát, ktoré používajú, je veľmi odlišný od mnohých iných vývojových pracovných postupov. V tejto lekcii tento proces objasníme a načrtneme hlavné techniky, ktoré potrebujete poznať. Naučíte sa:

- Pochopiť procesy, ktoré sú základom strojového učenia na vysokej úrovni.
- Preskúmať základné koncepty, ako sú „modely“, „predikcie“ a „tréningové dáta“.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

[![ML pre začiatočníkov - Techniky strojového učenia](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML pre začiatočníkov - Techniky strojového učenia")

> 🎥 Kliknite na obrázok vyššie pre krátke video, ktoré prechádza touto lekciou.

## Úvod

Na vysokej úrovni sa remeslo vytvárania procesov strojového učenia (ML) skladá z niekoľkých krokov:

1. **Určte otázku**. Väčšina procesov ML začína položením otázky, na ktorú nemožno odpovedať jednoduchým podmieneným programom alebo pravidlovo založeným enginom. Tieto otázky sa často týkajú predikcií na základe zbierky dát.
2. **Zbierajte a pripravte dáta**. Aby ste mohli odpovedať na svoju otázku, potrebujete dáta. Kvalita a niekedy aj množstvo vašich dát určí, ako dobre dokážete odpovedať na svoju pôvodnú otázku. Vizualizácia dát je dôležitým aspektom tejto fázy. Táto fáza zahŕňa aj rozdelenie dát na tréningovú a testovaciu skupinu na vytvorenie modelu.
3. **Vyberte metódu tréningu**. V závislosti od vašej otázky a povahy vašich dát musíte zvoliť spôsob, akým chcete model trénovať, aby čo najlepšie odrážal vaše dáta a robil presné predikcie. Táto časť procesu ML vyžaduje špecifické odborné znalosti a často značné množstvo experimentovania.
4. **Trénujte model**. Pomocou vašich tréningových dát použijete rôzne algoritmy na trénovanie modelu, aby rozpoznal vzory v dátach. Model môže využívať interné váhy, ktoré je možné upraviť tak, aby uprednostňoval určité časti dát pred inými, aby vytvoril lepší model.
5. **Vyhodnoťte model**. Použijete dáta, ktoré model nikdy predtým nevidel (vaše testovacie dáta) z vašej zbierky, aby ste zistili, ako model funguje.
6. **Ladenie parametrov**. Na základe výkonu vášho modelu môžete proces zopakovať s použitím rôznych parametrov alebo premenných, ktoré ovládajú správanie algoritmov použitých na trénovanie modelu.
7. **Predikujte**. Použite nové vstupy na testovanie presnosti vášho modelu.

## Akú otázku položiť

Počítače sú obzvlášť zručné v objavovaní skrytých vzorov v dátach. Táto schopnosť je veľmi užitočná pre výskumníkov, ktorí majú otázky o danej oblasti, na ktoré nemožno ľahko odpovedať vytvorením pravidlovo založeného enginu. Pri aktuárskej úlohe, napríklad, by dátový vedec mohol vytvoriť ručne vytvorené pravidlá týkajúce sa úmrtnosti fajčiarov vs. nefajčiarov.

Keď sa však do rovnice pridá mnoho ďalších premenných, model ML môže byť efektívnejší pri predikcii budúcich úmrtnostných mier na základe minulých zdravotných záznamov. Pozitívnejším príkladom môže byť predpovedanie počasia na mesiac apríl na danom mieste na základe dát, ktoré zahŕňajú zemepisnú šírku, dĺžku, klimatické zmeny, blízkosť k oceánu, vzory prúdenia vzduchu a ďalšie.

✅ Táto [prezentácia](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o modeloch počasia ponúka historický pohľad na používanie ML v analýze počasia.  

## Úlohy pred vytvorením modelu

Predtým, než začnete vytvárať svoj model, musíte dokončiť niekoľko úloh. Aby ste mohli otestovať svoju otázku a vytvoriť hypotézu na základe predikcií modelu, musíte identifikovať a nakonfigurovať niekoľko prvkov.

### Dáta

Aby ste mohli odpovedať na svoju otázku s akoukoľvek istotou, potrebujete dostatočné množstvo dát správneho typu. V tomto bode musíte urobiť dve veci:

- **Zbierajte dáta**. Majte na pamäti predchádzajúcu lekciu o spravodlivosti v analýze dát a zbierajte svoje dáta opatrne. Buďte si vedomí zdrojov týchto dát, akýchkoľvek inherentných predsudkov, ktoré môžu obsahovať, a dokumentujte ich pôvod.
- **Pripravte dáta**. Proces prípravy dát zahŕňa niekoľko krokov. Možno budete musieť dáta zoskupiť a normalizovať, ak pochádzajú z rôznych zdrojov. Kvalitu a množstvo dát môžete zlepšiť rôznymi metódami, ako je konverzia reťazcov na čísla (ako to robíme v [Clustering](../../5-Clustering/1-Visualize/README.md)). Môžete tiež generovať nové dáta na základe pôvodných (ako to robíme v [Classification](../../4-Classification/1-Introduction/README.md)). Dáta môžete čistiť a upravovať (ako to robíme pred lekciou [Web App](../../3-Web-App/README.md)). Nakoniec ich možno budete musieť náhodne usporiadať a premiešať, v závislosti od vašich tréningových techník.

✅ Po zozbieraní a spracovaní vašich dát si chvíľu overte, či ich štruktúra umožní odpovedať na vašu zamýšľanú otázku. Môže sa stať, že dáta nebudú dobre fungovať pri vašej úlohe, ako zistíme v našich lekciách [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Vlastnosti a cieľ

[Vlastnosť](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) je merateľná vlastnosť vašich dát. V mnohých datasetoch je vyjadrená ako nadpis stĺpca, napríklad „dátum“, „veľkosť“ alebo „farba“. Vaša premenná vlastnosti, zvyčajne reprezentovaná ako `X` v kóde, predstavuje vstupnú premennú, ktorá sa použije na trénovanie modelu.

Cieľ je vec, ktorú sa snažíte predpovedať. Cieľ, zvyčajne reprezentovaný ako `y` v kóde, predstavuje odpoveď na otázku, ktorú sa snažíte položiť vašim dátam: v decembri, aká **farba** tekvíc bude najlacnejšia? V San Franciscu, ktoré štvrte budú mať najlepšiu cenu **nehnuteľností**? Niekedy sa cieľ označuje aj ako atribút označenia.

### Výber premenných vlastností

🎓 **Výber vlastností a extrakcia vlastností** Ako viete, ktorú premennú si vybrať pri vytváraní modelu? Pravdepodobne prejdete procesom výberu vlastností alebo extrakcie vlastností, aby ste si vybrali správne premenné pre najvýkonnejší model. Nie sú to však rovnaké veci: „Extrakcia vlastností vytvára nové vlastnosti z funkcií pôvodných vlastností, zatiaľ čo výber vlastností vracia podmnožinu vlastností.“ ([zdroj](https://wikipedia.org/wiki/Feature_selection))

### Vizualizácia dát

Dôležitým aspektom nástrojov dátového vedca je schopnosť vizualizovať dáta pomocou niekoľkých vynikajúcich knižníc, ako sú Seaborn alebo MatPlotLib. Vizualizácia dát vám môže umožniť odhaliť skryté korelácie, ktoré môžete využiť. Vaše vizualizácie vám môžu tiež pomôcť odhaliť predsudky alebo nevyvážené dáta (ako zistíme v [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Rozdelenie datasetu

Pred tréningom musíte rozdeliť svoj dataset na dve alebo viac častí nerovnakej veľkosti, ktoré stále dobre reprezentujú dáta.

- **Tréning**. Táto časť datasetu sa prispôsobí vášmu modelu, aby ho trénovala. Táto sada tvorí väčšinu pôvodného datasetu.
- **Testovanie**. Testovací dataset je nezávislá skupina dát, často získaná z pôvodných dát, ktorú použijete na potvrdenie výkonu vytvoreného modelu.
- **Validácia**. Validačná sada je menšia nezávislá skupina príkladov, ktorú použijete na ladenie hyperparametrov modelu alebo jeho architektúry, aby ste model zlepšili. V závislosti od veľkosti vašich dát a otázky, ktorú kladiete, možno nebudete musieť vytvoriť túto tretiu sadu (ako poznamenávame v [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Vytváranie modelu

Pomocou vašich tréningových dát je vaším cieľom vytvoriť model, alebo štatistickú reprezentáciu vašich dát, pomocou rôznych algoritmov na **tréning**. Tréning modelu ho vystavuje dátam a umožňuje mu robiť predpoklady o vnímaných vzoroch, ktoré objaví, overí a prijme alebo odmietne.

### Rozhodnite sa pre metódu tréningu

V závislosti od vašej otázky a povahy vašich dát si vyberiete metódu na ich tréning. Prechádzajúc [dokumentáciou Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - ktorú používame v tomto kurze - môžete preskúmať mnoho spôsobov, ako trénovať model. V závislosti od vašich skúseností možno budete musieť vyskúšať niekoľko rôznych metód, aby ste vytvorili najlepší model. Pravdepodobne prejdete procesom, pri ktorom dátoví vedci hodnotia výkon modelu tým, že mu poskytnú nevidené dáta, kontrolujú presnosť, predsudky a ďalšie problémy znižujúce kvalitu a vyberajú najvhodnejšiu metódu tréningu pre danú úlohu.

### Trénujte model

S vašimi tréningovými dátami ste pripravení „prispôsobiť“ ich na vytvorenie modelu. Všimnete si, že v mnohých knižniciach ML nájdete kód „model.fit“ - práve v tomto momente pošlete svoju premennú vlastnosti ako pole hodnôt (zvyčajne „X“) a cieľovú premennú (zvyčajne „y“).

### Vyhodnoťte model

Keď je proces tréningu dokončený (tréning veľkého modelu môže trvať mnoho iterácií alebo „epoch“), budete môcť vyhodnotiť kvalitu modelu pomocou testovacích dát na posúdenie jeho výkonu. Tieto dáta sú podmnožinou pôvodných dát, ktoré model predtým neanalyzoval. Môžete vytlačiť tabuľku metrík o kvalite vášho modelu.

🎓 **Prispôsobenie modelu**

V kontexte strojového učenia prispôsobenie modelu odkazuje na presnosť základnej funkcie modelu, keď sa pokúša analyzovať dáta, s ktorými nie je oboznámený.

🎓 **Nedostatočné prispôsobenie** a **nadmerné prispôsobenie** sú bežné problémy, ktoré zhoršujú kvalitu modelu, keď model buď nevyhovuje dostatočne dobre, alebo príliš dobre. To spôsobuje, že model robí predikcie buď príliš úzko alebo príliš voľne spojené s jeho tréningovými dátami. Nadmerne prispôsobený model predpovedá tréningové dáta príliš dobre, pretože sa naučil detaily a šum dát príliš dobre. Nedostatočne prispôsobený model nie je presný, pretože nedokáže presne analyzovať ani svoje tréningové dáta, ani dáta, ktoré ešte „nevidel“.

![nadmerné prispôsobenie modelu](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografika od [Jen Looper](https://twitter.com/jenlooper)

## Ladenie parametrov

Keď je váš počiatočný tréning dokončený, pozorujte kvalitu modelu a zvážte jeho zlepšenie úpravou jeho „hyperparametrov“. Prečítajte si viac o tomto procese [v dokumentácii](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predikcia

Toto je moment, keď môžete použiť úplne nové dáta na testovanie presnosti vášho modelu. V „aplikovanom“ nastavení ML, kde vytvárate webové aktíva na použitie modelu v produkcii, tento proces môže zahŕňať zhromažďovanie vstupov od používateľa (napríklad stlačenie tlačidla), aby sa nastavila premenná a poslala modelu na inferenciu alebo vyhodnotenie.

V týchto lekciách objavíte, ako používať tieto kroky na prípravu, vytváranie, testovanie, vyhodnocovanie a predikciu - všetky gestá dátového vedca a viac, ako postupujete na svojej ceste stať sa „full stack“ ML inžinierom.

---

## 🚀Výzva

Nakreslite diagram toku, ktorý odráža kroky praktika ML. Kde sa momentálne vidíte v procese? Kde predpokladáte, že narazíte na ťažkosti? Čo sa vám zdá jednoduché?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Vyhľadajte online rozhovory s dátovými vedcami, ktorí diskutujú o svojej každodennej práci. Tu je [jeden](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Zadanie

[Urobte rozhovor s dátovým vedcom](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.