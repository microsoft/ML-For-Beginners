# Budovanie riešení strojového učenia so zodpovednou umelou inteligenciou
 
![Zhrnutie zodpovednej umelej inteligencie v strojovom učení v sketchnote](../../../../translated_images/sk/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Prednáškový kvíz](https://ff-quizzes.netlify.app/en/ml/)
 
## Úvod

V tomto učebnom pláne začnete objavovať, ako strojové učenie môže a už vplýva na náš každodenný život. Už teraz sú systémy a modely zapojené do každodenných rozhodovacích úloh, ako sú zdravotné diagnostiky, schvaľovanie pôžičiek alebo detekcia podvodov. Preto je dôležité, aby tieto modely fungovali dobre a poskytovali spoľahlivé výsledky. Rovnako ako každý softvérový program, aj AI systémy môžu nesplniť očakávania alebo mať nežiaduci výsledok. Preto je nevyhnutné vedieť pochopiť a vysvetliť správanie AI modelu.

Predstavte si, čo sa môže stať, keď dáta, ktoré používate na tvorbu týchto modelov, neobsahujú určité demografické skupiny, ako je rasa, pohlavie, politický názor, náboženstvo alebo neprimerane zastupujú tieto skupiny. Čo ak výstup modelu je interpretovaný tak, že uprednostňuje niektorú demografickú skupinu? Aký to má dôsledok pre aplikáciu? Ďalej, čo sa stane, keď model dosiahne nepriaznivý výsledok a škodí ľuďom? Kto je zodpovedný za správanie AI systému? Toto sú niektoré otázky, ktoré budeme skúmať v tomto učebnom pláne.

V tejto lekcii sa naučíte:

- Zvýšiť svoju informovanosť o dôležitosti spravodlivosti v strojovom učení a škodách súvisiacich so spravodlivosťou.
- Zoznámiť sa s praxou skúmania odľahlých prípadov a nezvyčajných situácií na zabezpečenie spoľahlivosti a bezpečnosti.
- Pochopiť potrebu posilniť všetkých navrhovaním inkluzívnych systémov.
- Preskúmať, aké je dôležité chrániť súkromie a bezpečnosť údajov a ľudí.
- Zdôrazniť význam prístupu "sklenenej krabice" pre vysvetlenie správania AI modelov.
- Byť si vedomý, ako je zodpovednosť nevyhnutná na vybudovanie dôvery v AI systémy.

## Predpoklady

Ako predpoklad je potrebné absolvovať "Princípy zodpovednej AI" v rámci Learning Path a pozrieť si video nižšie o tejto téme:

Zistite viac o Zodpovednej AI prostredníctvom tohto [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoftov prístup k zodpovednej AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoftov prístup k zodpovednej AI")

> 🎥 Kliknite na obrázok vyššie pre video: Microsoftov prístup k zodpovednej AI

## Spravodlivosť

AI systémy by mali zaobchádzať so všetkými spravodlivo a vyhnúť sa tomu, aby ovplyvňovali podobné skupiny ľudí rôznymi spôsobmi. Napríklad keď AI systémy poskytujú odporúčania v lekárskej liečbe, žiadostiach o pôžičku alebo zamestnaní, mali by dávať rovnaké odporúčania všetkým s podobnými symptómami, finančnými podmienkami alebo odbornou kvalifikáciou. Každý z nás ako človek nesie dedičné predsudky, ktoré ovplyvňujú naše rozhodnutia a konania. Tieto predsudky môžu byť zjavné v dátach, ktoré používame na tréning AI systémov. Takáto manipulácia môže niekedy prebiehať neúmyselne. Často je ťažké vedome vedieť, kedy do dát zavádzame predsudky.

**„Nespravodlivosť“** zahŕňa negatívne dopady, alebo "škody", pre skupinu ľudí, napríklad definovaných podľa rasy, pohlavia, veku alebo zdravotného postihnutia. Hlavné škody súvisiace so spravodlivosťou možno klasifikovať ako:

- **Pridelenie**, ak je napríklad uprednostnené jedno pohlavie alebo etnická skupina pred inou.
- **Kvalita služby**. Ak trénujete dáta pre jednu konkrétnu situáciu, ale realita je oveľa zložitejšia, vedie to k zle fungujúcej službe. Napríklad dávkovač tekutého mydla, ktorý nedokázal rozpoznať ľudí s tmavou pleťou. [Referencie](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Poškodzovanie**. Nespravodlivo kritizovať a označovať niečo alebo niekoho. Napríklad technológia označovania obrázkov neslávne nesprávne označila obrázky ľudí s tmavou pleťou ako gorily.
- **Nad- alebo podreprezentácia**. Myšlienka, že určitá skupina sa neobjavuje v určitom povolaní, a každá služba alebo funkcia, ktorá to pokračuje podporovať, prispieva k škode.
- **Stereotypizácia**. Priraďovanie určitej skupine preddefinovaných vlastností. Napríklad jazykový preklad medzi angličtinou a turečtinou môže mať nepresnosti kvôli slovám s prenesenými stereotypnými súvislosťami s pohlavím.

![preklad do turečtiny](../../../../translated_images/sk/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> preklad do turečtiny

![preklad späť do angličtiny](../../../../translated_images/sk/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> preklad späť do angličtiny

Pri navrhovaní a testovaní AI systémov musíme zabezpečiť, že AI je spravodlivá a nie je naprogramovaná tak, aby robila zaujaté alebo diskriminačné rozhodnutia, ktoré sú tiež zakázané pre ľudí. Zaručovanie spravodlivosti v AI a strojovom učení zostáva zložitou sociotechnickou výzvou.

### Spoľahlivosť a bezpečnosť

Na vybudovanie dôvery musia byť AI systémy spoľahlivé, bezpečné a konzistentné za normálnych i neočakávaných podmienok. Je dôležité vedieť, ako sa AI systémy budú správať v rôznych situáciách, najmä keď ide o odľahlé prípady. Pri tvorbe AI riešení je potrebné venovať veľkú pozornosť tomu, ako riešiť širokú škálu okolností, s ktorými sa AI riešenia môžu stretnúť. Napríklad autonómne vozidlo musí klásť bezpečnosť ľudí na prvé miesto. Preto musí AI poháňajúca auto zvážiť všetky možné scenáre, s ktorými sa môže auto stretnúť, ako nočná jazda, búrky alebo snehové víchrice, deti bežiace cez cestu, domáce zvieratá, cestné práce atď. Ako dobre dokáže AI systém spoľahlivo a bezpečne zvládnuť rôzne podmienky, odzrkadľuje úroveň predvídavosti, ktorú si dátový vedec alebo AI vývojár zvážil pri navrhovaní alebo testovaní systému.

> [🎥 Kliknite tu pre video: Spoľahlivosť a bezpečnosť v AI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzívnosť

AI systémy by mali byť navrhnuté tak, aby zapájali a posilňovali všetkých. Pri navrhovaní a implementácii AI systémov identifikujú dátoví vedci a vývojári AI potenciálne bariéry v systéme, ktoré by mohli neúmyselne vylučovať ľudí. Napríklad na svete je 1 miliarda ľudí so zdravotným postihnutím. Vďaka pokroku AI môžu jednoduchšie pristupovať k širokej škále informácií a príležitostí v každodennom živote. Riešením bariér sa vytvárajú príležitosti na inovácie a vývoj AI produktov s lepšími skúsenosťami, ktoré prospievajú všetkým.

> [🎥 Kliknite tu pre video: Inkluzívnosť v AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Bezpečnosť a súkromie

AI systémy by mali byť bezpečné a rešpektovať súkromie ľudí. Ľudia majú menšiu dôveru v systémy, ktoré ohrozujú ich súkromie, informácie alebo životy. Pri trénovaní modelov strojového učenia sa spoliehame na dáta na dosiahnutie najlepších výsledkov. Pri tomto procese je potrebné zvážiť pôvod údajov a ich integritu. Napríklad, či boli dáta podané používateľmi alebo sú verejne dostupné. Ďalej pri práci s dátami je nevyhnutné vyvíjať AI systémy, ktoré dokážu chrániť dôverné informácie a odolávať útokom. Ako AI získava na význame, ochrana súkromia a zabezpečenie dôležitých osobných a firemných informácií sa stáva čoraz kritickejšou a zložitejšou. Problémy s ochranou súkromia a bezpečnosťou dát si vyžadujú osobitnú pozornosť v AI, pretože prístup k dátam je nevyhnutný, aby AI systémy mohli robiť presné a informované predpovede a rozhodnutia o ľuďoch.

> [🎥 Kliknite tu pre video: Bezpečnosť v AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ako odvetvie sme dosiahli významné pokroky v oblasti ochrany súkromia a bezpečnosti, podstatne naštartované reguláciami ako GDPR (Všeobecné nariadenie o ochrane údajov).
- Napriek tomu pri AI systémoch musíme uznať napätie medzi potrebou viac osobných údajov na zefektívnenie systémov a ochranou súkromia.
- Rovnako ako pri príchode prepojených počítačov s internetom, dnes zaznamenávame výrazný nárast bezpečnostných problémov súvisiacich s AI.
- Zároveň sme videli použitie AI na zlepšenie bezpečnosti. Napríklad väčšina moderných antivírusových skenerov dnes funguje pomocou AI heuristiky.
- Musíme zabezpečiť, aby naše procesy dátovej vedy harmonicky zapadali do najnovších bezpečnostných a súkromných praktík.


### Transparentnosť
AI systémy by mali byť zrozumiteľné. Kľúčovou súčasťou transparentnosti je vysvetlenie správania AI systémov a ich komponentov. Zlepšenie porozumenia AI systémom vyžaduje, aby zainteresované strany pochopili, ako a prečo fungujú, aby mohli identifikovať potenciálne problémy s výkonom, bezpečnostné a súkromné obavy, predsudky, vylučujúce praktiky alebo neúmyselné výsledky. Tiež veríme, že tí, ktorí AI systémy používajú, by mali byť úprimní a otvorení o tom, kedy, prečo a ako ich nasadzujú, ako aj o obmedzeniach systémov, ktoré používajú. Napríklad ak banka používa AI systém na podporu svojich rozhodnutí o spotrebiteľských pôžičkách, je dôležité preskúmať výsledky a pochopiť, ktoré dáta ovplyvňujú odporúčania systému. Vlády začínajú regulovať AI naprieč odvetviami, takže dátoví vedci a organizácie musia vysvetliť, či AI systém spĺňa regulačné požiadavky, najmä keď dôjde k nežiaducemu výsledku.

> [🎥 Kliknite tu pre video: Transparentnosť v AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Pretože AI systémy sú také zložité, je ťažké pochopiť, ako fungujú a interpretovať výsledky.
- Tento nedostatok porozumenia ovplyvňuje spôsob, akým sú tieto systémy spravované, prevádzkované a dokumentované.
- Tento nedostatok porozumenia najmä ovplyvňuje rozhodnutia prijímané na základe výsledkov, ktoré tieto systémy produkujú.

### Zodpovednosť
 
Ľudia, ktorí navrhujú a nasadzujú AI systémy, musia byť zodpovední za to, ako ich systémy fungujú. Potreba zodpovednosti je obzvlášť dôležitá pri citlivých technológiách, ako je rozpoznávanie tváre. Nedávno rástol dopyt po technológii rozpoznávania tváre, najmä zo strany orgánov činných v trestnom konaní, ktoré vidia potenciál tejto technológie pri hľadaní nezvestných detí. Tieto technológie však môžu byť potenciálne zneužité vládou na ohrozenie základných slobôd občanov, napríklad umožnením nepretržitého sledovania určitých jednotlivcov. Preto dátoví vedci a organizácie musia byť zodpovední za to, ako ich AI systém vplýva na jednotlivcov alebo spoločnosť.

[![Popredný výskumník AI varuje pred masovým sledovaním cez rozpoznávanie tváre](../../../../translated_images/sk/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoftov prístup k zodpovednej AI")

> 🎥 Kliknite na obrázok vyššie pre video: Varovania pred masovým sledovaním cez rozpoznávanie tváre

Nakoniec jedna z najväčších otázok pre našu generáciu, ktorá je prvou generáciou prinášajúcou AI do spoločnosti, je, ako zabezpečiť, aby počítače zostali zodpovedné ľuďom a aby ľudia, ktorí počítače navrhujú, zostali zodpovední všetkým ostatným.

## Hodnotenie dopadov

Pred trénovaním modelu strojového učenia je dôležité vykonať hodnotenie dopadov, aby ste pochopili účel AI systému; aké je zamýšľané použitie; kde bude nasadený; a kto bude so systémom interagovať. Toto je užitočné pre recenzentov alebo testerov hodnotiacich systém, aby vedeli, aké faktory brať do úvahy pri identifikácii potenciálnych rizík a očakávaných následkov.

Nasledovné sú oblasti záujmu pri vykonávaní hodnotenia dopadov:

* **Nežiadaný dopad na jednotlivcov**. Byť si vedomý akýchkoľvek obmedzení alebo požiadaviek, nepodporovaného použitia alebo známych limitácií, ktoré by mohli brániť výkonu systému, je kľúčové na zabezpečenie, že systém nebude použitý spôsobom, ktorý by mohol poškodiť jednotlivcov.
* **Požiadavky na dáta**. Pochopenie, ako a kde systém použije dáta, umožňuje recenzentom preskúmať akékoľvek požiadavky na dáta, na ktoré by ste si mali dávať pozor (napr. GDPR alebo HIPAA regulácie). Ďalej preskúmajte, či zdroj alebo množstvo dát je dostatočné na tréning.
* **Zhrnutie dopadov**. Zhromaždite zoznam potenciálnych škôd, ktoré by mohli vzniknúť použitím systému. Počas životného cyklu ML kontrolujte, či sú identifikované problémy zmiernené alebo riešené.
* **Použiteľné ciele** pre každý zo šiestich základných princípov. Zhodnoťte, či sú ciele z každého princípu dosiahnuté a či existujú nejaké medzery.


## Ladenie s zodpovednou AI

Podobne ako pri ladení softvérovej aplikácie, ladanie AI systému je nevyhnutný proces identifikácie a riešenia problémov v systéme. Existuje mnoho faktorov, ktoré môžu ovplyvniť, že model nebude fungovať podľa očakávaní či zodpovedne. Väčšina tradičných metrík výkonu modelu sú kvantitatívne agregáty jeho výkonnosti, ktoré nie sú postačujúce na analýzu, ako model porušuje princípy zodpovednej AI. Navyše model strojového učenia je čiernou skrinkou, ktorá sťažuje pochopenie čo riadi jeho výsledok alebo poskytnutie vysvetlenia, keď urobí chybu. Neskôr v tomto kurze sa naučíme používať dashboard Zodpovednej AI na pomoc pri ladení AI systémov. Dashboard poskytuje komplexný nástroj pre dátových vedcov a vývojárov AI na vykonávanie:

* **Analýzu chýb**. Na identifikáciu rozdelenia chýb modelu, ktoré môžu ovplyvniť spravodlivosť alebo spoľahlivosť systému.
* **Prehľad modelu**. Na objavenie rozdielov vo výkonnosti modelu medzi dátovými skupinami.
* **Analýzu dát**. Na pochopenie rozdelenia dát a identifikovanie prípadných predsudkov v dátach, ktoré by mohli viesť k problémom so spravodlivosťou, inkluzívnosťou a spoľahlivosťou.
* **Interpretovateľnosť modelu**. Na pochopenie čo ovplyvňuje predikcie modelu. Pomáha to pri vysvetľovaní správania modelu, čo je dôležité pre transparentnosť a zodpovednosť.


## 🚀 Výzva
 
Aby sme zabránili vzniku škôd už na prvom mieste, mali by sme:

- mať rozmanité zázemie a pohľady medzi ľuďmi pracujúcimi na systémoch
- investovať do dátových súborov, ktoré odrážajú rozmanitosť našej spoločnosti
- vyvíjať lepšie metódy počas celého životného cyklu strojového učenia na odhaľovanie a nápravu nezodpovednej AI, keď k nej dôjde

Premyslite si reálne situácie, kde je nedôveryhodnosť modelu zjavná pri tvorbe a používaní modelu. Čo ešte by sme mali zvážiť?

## [Po prednáške kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium
 
V tejto lekcii ste sa naučili základy konceptov spravodlivosti a nespravodlivosti v strojovom učení.
 
Pozrite si tento workshop na hlbšie preskúmanie tém:

- V snahe o zodpovednú AI: prinášanie princípov do praxe od Besmira Nushi, Mehrnoosh Sameki a Amit Sharma

[![Responsible AI Toolbox: Open-source rámec na budovanie zodpovednej AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Open-source rámec na budovanie zodpovednej AI")

> 🎥 Kliknite na obrázok vyššie pre video: RAI Toolbox: Open-source rámec na budovanie zodpovednej AI od Besmira Nushi, Mehrnoosh Sameki a Amit Sharma

Tiež si prečítajte: 

- Microsoft centrum zdrojov pre RAI: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoft výskumná skupina FATE: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [GitHub repozitár Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Prečítajte si o nástrojoch Azure Machine Learning na zabezpečenie spravodlivosti:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Zadanie

[Preskúmajte RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vyhlásenie o zodpovednosti**:
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, vezmite prosím na vedomie, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho natívnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->