<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T16:00:41+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "sk"
}
-->
# Budovanie riešení strojového učenia s dôrazom na zodpovednú AI

![Zhrnutie zodpovednej AI v strojovom učení v sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

V tomto kurze začnete objavovať, ako strojové učenie ovplyvňuje naše každodenné životy. Už teraz sú systémy a modely zapojené do rozhodovacích úloh, ako sú diagnostika v zdravotníctve, schvaľovanie úverov alebo odhaľovanie podvodov. Preto je dôležité, aby tieto modely fungovali spoľahlivo a poskytovali dôveryhodné výsledky. Rovnako ako akákoľvek softvérová aplikácia, aj systémy AI môžu zlyhať alebo mať nežiaduci výsledok. Preto je nevyhnutné rozumieť a vedieť vysvetliť správanie modelu AI.

Predstavte si, čo sa môže stať, keď údaje, ktoré používate na vytvorenie týchto modelov, neobsahujú určité demografické skupiny, ako sú rasa, pohlavie, politické názory, náboženstvo, alebo ich neprimerane zastupujú. Čo ak je výstup modelu interpretovaný tak, že uprednostňuje určitú demografickú skupinu? Aké sú dôsledky pre aplikáciu? A čo sa stane, keď model má nepriaznivý výsledok a je škodlivý pre ľudí? Kto je zodpovedný za správanie systému AI? Toto sú niektoré otázky, ktoré budeme skúmať v tomto kurze.

V tejto lekcii sa naučíte:

- Zvýšiť povedomie o dôležitosti spravodlivosti v strojovom učení a o škodách súvisiacich so spravodlivosťou.
- Oboznámiť sa s praxou skúmania odchýlok a neobvyklých scenárov na zabezpečenie spoľahlivosti a bezpečnosti.
- Získať pochopenie potreby posilniť všetkých prostredníctvom navrhovania inkluzívnych systémov.
- Preskúmať, aké dôležité je chrániť súkromie a bezpečnosť údajov a ľudí.
- Vidieť význam prístupu „sklenená krabica“ na vysvetlenie správania modelov AI.
- Byť si vedomý toho, ako je zodpovednosť kľúčová pre budovanie dôvery v systémy AI.

## Predpoklad

Ako predpoklad si prosím preštudujte „Zásady zodpovednej AI“ v rámci učebnej cesty a pozrite si nasledujúce video na túto tému:

Viac o zodpovednej AI sa dozviete na tejto [učebnej ceste](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott).

[![Prístup Microsoftu k zodpovednej AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Prístup Microsoftu k zodpovednej AI")

> 🎥 Kliknite na obrázok vyššie pre video: Prístup Microsoftu k zodpovednej AI

## Spravodlivosť

Systémy AI by mali zaobchádzať so všetkými spravodlivo a vyhnúť sa tomu, aby ovplyvňovali podobné skupiny ľudí rôznymi spôsobmi. Napríklad, keď systémy AI poskytujú odporúčania týkajúce sa lekárskej liečby, žiadostí o úver alebo zamestnania, mali by robiť rovnaké odporúčania všetkým s podobnými symptómami, finančnými okolnosťami alebo odbornými kvalifikáciami. Každý z nás ako človek nesie vrodené predsudky, ktoré ovplyvňujú naše rozhodnutia a činy. Tieto predsudky môžu byť zjavné v údajoch, ktoré používame na trénovanie systémov AI. Takáto manipulácia sa niekedy môže stať neúmyselne. Často je ťažké vedome rozpoznať, kedy do údajov zavádzame predsudky.

**„Nespravodlivosť“** zahŕňa negatívne dopady alebo „škody“ na skupinu ľudí, ako sú tí definovaní podľa rasy, pohlavia, veku alebo zdravotného postihnutia. Hlavné škody súvisiace so spravodlivosťou možno klasifikovať ako:

- **Alokácia**, ak je napríklad uprednostnené jedno pohlavie alebo etnická skupina pred druhou.
- **Kvalita služby**. Ak trénujete údaje pre jeden konkrétny scenár, ale realita je oveľa zložitejšia, vedie to k zle fungujúcej službe. Napríklad dávkovač mydla, ktorý nedokázal rozpoznať ľudí s tmavou pokožkou. [Referencie](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Očierňovanie**. Nespravodlivé kritizovanie a označovanie niečoho alebo niekoho. Napríklad technológia označovania obrázkov neslávne označila obrázky ľudí s tmavou pokožkou ako gorily.
- **Nadmerné alebo nedostatočné zastúpenie**. Myšlienka, že určitá skupina nie je viditeľná v určitom povolaní, a akákoľvek služba alebo funkcia, ktorá to naďalej podporuje, prispieva k škode.
- **Stereotypizácia**. Priraďovanie preddefinovaných atribútov určitej skupine. Napríklad systém prekladu medzi angličtinou a turečtinou môže mať nepresnosti kvôli slovám so stereotypnými asociáciami k pohlaviu.

![preklad do turečtiny](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> preklad do turečtiny

![preklad späť do angličtiny](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> preklad späť do angličtiny

Pri navrhovaní a testovaní systémov AI musíme zabezpečiť, že AI je spravodlivá a nie je naprogramovaná na prijímanie zaujatých alebo diskriminačných rozhodnutí, ktoré sú zakázané aj pre ľudí. Zaručenie spravodlivosti v AI a strojovom učení zostáva komplexnou sociotechnickou výzvou.

### Spoľahlivosť a bezpečnosť

Na budovanie dôvery musia byť systémy AI spoľahlivé, bezpečné a konzistentné za normálnych aj neočakávaných podmienok. Je dôležité vedieť, ako sa systémy AI budú správať v rôznych situáciách, najmä keď ide o odchýlky. Pri budovaní riešení AI je potrebné venovať značnú pozornosť tomu, ako zvládnuť širokú škálu okolností, s ktorými sa riešenia AI môžu stretnúť. Napríklad autonómne auto musí klásť bezpečnosť ľudí na prvé miesto. Výsledkom je, že AI poháňajúca auto musí zohľadniť všetky možné scenáre, s ktorými sa auto môže stretnúť, ako sú noc, búrky alebo snehové búrky, deti bežiace cez ulicu, domáce zvieratá, cestné práce atď. To, ako dobre systém AI dokáže spoľahlivo a bezpečne zvládnuť širokú škálu podmienok, odráža úroveň predvídavosti, ktorú dátový vedec alebo vývojár AI zohľadnil počas návrhu alebo testovania systému.

> [🎥 Kliknite sem pre video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzívnosť

Systémy AI by mali byť navrhnuté tak, aby zapájali a posilňovali každého. Pri navrhovaní a implementácii systémov AI dátoví vedci a vývojári AI identifikujú a riešia potenciálne bariéry v systéme, ktoré by mohli neúmyselne vylúčiť ľudí. Napríklad na svete je 1 miliarda ľudí so zdravotným postihnutím. Vďaka pokroku v AI môžu ľahšie pristupovať k širokému spektru informácií a príležitostí vo svojom každodennom živote. Riešením bariér sa vytvárajú príležitosti na inovácie a vývoj produktov AI s lepšími skúsenosťami, ktoré prospievajú všetkým.

> [🎥 Kliknite sem pre video: inkluzívnosť v AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Bezpečnosť a súkromie

Systémy AI by mali byť bezpečné a rešpektovať súkromie ľudí. Ľudia majú menšiu dôveru v systémy, ktoré ohrozujú ich súkromie, informácie alebo životy. Pri trénovaní modelov strojového učenia sa spoliehame na údaje, aby sme dosiahli čo najlepšie výsledky. Pri tom je potrebné zvážiť pôvod údajov a ich integritu. Napríklad, boli údaje poskytnuté používateľom alebo verejne dostupné? Ďalej, pri práci s údajmi je nevyhnutné vyvíjať systémy AI, ktoré dokážu chrániť dôverné informácie a odolávať útokom. Ako sa AI stáva rozšírenejšou, ochrana súkromia a zabezpečenie dôležitých osobných a obchodných informácií sa stáva čoraz kritickejšou a zložitejšou. Problémy so súkromím a bezpečnosťou údajov si vyžadujú obzvlášť dôkladnú pozornosť pri AI, pretože prístup k údajom je nevyhnutný na to, aby systémy AI mohli robiť presné a informované predpovede a rozhodnutia o ľuďoch.

> [🎥 Kliknite sem pre video: bezpečnosť v AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ako odvetvie sme dosiahli významný pokrok v oblasti súkromia a bezpečnosti, výrazne podporený reguláciami, ako je GDPR (Všeobecné nariadenie o ochrane údajov).
- Napriek tomu musíme pri systémoch AI uznať napätie medzi potrebou viac osobných údajov na zlepšenie systémov a ochranou súkromia.
- Rovnako ako pri vzniku pripojených počítačov s internetom, zaznamenávame aj výrazný nárast počtu bezpečnostných problémov súvisiacich s AI.
- Zároveň sme videli, že AI sa používa na zlepšenie bezpečnosti. Napríklad väčšina moderných antivírusových skenerov je dnes poháňaná heuristikou AI.
- Musíme zabezpečiť, aby naše procesy dátovej vedy harmonicky ladili s najnovšími praktikami v oblasti súkromia a bezpečnosti.

### Transparentnosť

Systémy AI by mali byť zrozumiteľné. Kľúčovou súčasťou transparentnosti je vysvetlenie správania systémov AI a ich komponentov. Zlepšenie porozumenia systémom AI si vyžaduje, aby zainteresované strany pochopili, ako a prečo fungujú, aby mohli identifikovať potenciálne problémy s výkonom, obavy o bezpečnosť a súkromie, predsudky, vylučujúce praktiky alebo neúmyselné výsledky. Veríme tiež, že tí, ktorí používajú systémy AI, by mali byť úprimní a otvorení o tom, kedy, prečo a ako sa rozhodnú ich nasadiť. Rovnako ako o obmedzeniach systémov, ktoré používajú. Napríklad, ak banka používa systém AI na podporu svojich rozhodnutí o poskytovaní úverov, je dôležité preskúmať výsledky a pochopiť, ktoré údaje ovplyvňujú odporúčania systému. Vlády začínajú regulovať AI naprieč odvetviami, takže dátoví vedci a organizácie musia vysvetliť, či systém AI spĺňa regulačné požiadavky, najmä keď dôjde k nežiaducemu výsledku.

> [🎥 Kliknite sem pre video: transparentnosť v AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Keďže systémy AI sú veľmi komplexné, je ťažké pochopiť, ako fungujú a interpretovať výsledky.
- Tento nedostatok porozumenia ovplyvňuje spôsob, akým sú tieto systémy spravované, prevádzkované a dokumentované.
- Tento nedostatok porozumenia ešte dôležitejšie ovplyvňuje rozhodnutia prijaté na základe výsledkov, ktoré tieto systémy produkujú.

### Zodpovednosť

Ľudia, ktorí navrhujú a nasadzujú systémy AI, musia byť zodpovední za to, ako ich systémy fungujú. Potreba zodpovednosti je obzvlášť dôležitá pri technológiách citlivého použitia, ako je rozpoznávanie tváre. V poslednej dobe rastie dopyt po technológii rozpoznávania tváre, najmä zo strany orgánov činných v trestnom konaní, ktoré vidia potenciál tejto technológie v aplikáciách, ako je hľadanie nezvestných detí. Tieto technológie však môžu byť potenciálne použité vládou na ohrozenie základných slobôd občanov, napríklad umožnením nepretržitého sledovania konkrétnych jednotlivcov. Preto musia byť dátoví vedci a organizácie zodpovední za to, ako ich systém AI ovplyvňuje jednotlivcov alebo spoločnosť.

[![Vedúci výskumník AI varuje pred masovým sledovaním prostredníctvom rozpoznávania tváre](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Prístup Microsoftu k zodpovednej AI")

> 🎥 Kliknite na obrázok vyššie pre video: Varovania pred masovým sledovaním prostredníctvom rozpoznávania tváre

Nakoniec jednou z najväčších otázok našej generácie, ako prvej generácie, ktorá prináša AI do spoločnosti, je, ako zabezpečiť, aby počítače zostali zodpovedné voči ľuďom a ako zabezpečiť, aby ľudia, ktorí navrhujú počítače, zostali zodpovední voči všetkým ostatným.

## Hodnotenie dopadu

Pred trénovaním modelu strojového učenia je dôležité vykonať hodnotenie dopadu, aby ste pochopili účel systému AI; aké je jeho zamýšľané použitie; kde bude nasadený; a kto bude so systémom interagovať. Tieto informácie sú užitočné pre recenzentov alebo testerov, ktorí hodnotia systém, aby vedeli, aké faktory treba zohľadniť pri identifikácii potenciálnych rizík a očakávaných dôsledkov.

Nasledujú oblasti zamerania pri vykonávaní hodnotenia dopadu:

* **Nepriaznivý dopad na jednotlivcov**. Byť si vedomý akýchkoľvek obmedzení alebo požiadaviek, nepodporovaného použitia alebo akýchkoľvek známych obmedzení, ktoré bránia výkonu systému, je zásadné na zabezpečenie toho, aby systém nebol používaný spôsobom, ktorý by mohol spôsobiť škodu jednotlivcom.
* **Požiadavky na údaje**. Získanie pochopenia toho, ako a kde systém bude používať údaje, umožňuje recenzentom preskúmať akékoľvek požiadavky na údaje, na ktoré by ste mali byť pozorní (napr. GDPR alebo HIPPA regulácie údajov). Okrem toho preskúmajte, či je zdroj alebo množstvo údajov dostatočné na trénovanie.
* **Zhrnutie dopadu**. Zozbierajte zoznam potenciálnych škôd, ktoré by mohli vzniknúť z používania systému. Počas životného cyklu ML preskúmajte, či sú identifikované problémy zmiernené alebo riešené.
* **Platné ciele** pre každú zo šiestich základných zásad. Posúďte, či sú ciele z každej zásady splnené a či existujú nejaké medzery.

## Ladenie so zodpovednou AI

Podobne ako ladenie softvérovej aplikácie, ladenie systému AI je nevyhnutný proces identifikácie a riešenia problémov v systéme. Existuje mnoho faktorov, ktoré môžu ovplyvniť, že model nefunguje podľa očakávaní alebo zodpovedne. Väčšina tradičných metrík výkonu modelu sú kvantitatívne agregáty výkonu modelu, ktoré nie sú dostatočné
Pozrite si tento workshop, aby ste sa hlbšie ponorili do tém:

- Na ceste k zodpovednej AI: Uplatnenie princípov v praxi od Besmiry Nushi, Mehrnoosh Sameki a Amita Sharmu

[![Responsible AI Toolbox: Open-source framework na budovanie zodpovednej AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Open-source framework na budovanie zodpovednej AI")

> 🎥 Kliknite na obrázok vyššie pre video: RAI Toolbox: Open-source framework na budovanie zodpovednej AI od Besmiry Nushi, Mehrnoosh Sameki a Amita Sharmu

Prečítajte si tiež:

- Microsoftov zdrojový centrum pre zodpovednú AI: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoftova výskumná skupina FATE: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [GitHub repozitár Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Prečítajte si o nástrojoch Azure Machine Learning na zabezpečenie spravodlivosti:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Zadanie

[Preskúmajte RAI Toolbox](assignment.md)

---

**Zrieknutie sa zodpovednosti**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.