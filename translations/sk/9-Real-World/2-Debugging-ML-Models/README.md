<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T15:55:18+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "sk"
}
-->
# Postscript: Ladenie modelov v strojovom učení pomocou komponentov zodpovedného AI dashboardu

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

Strojové učenie ovplyvňuje naše každodenné životy. AI si nachádza cestu do niektorých z najdôležitejších systémov, ktoré ovplyvňujú nás ako jednotlivcov aj našu spoločnosť, od zdravotníctva, financií, vzdelávania až po zamestnanie. Napríklad systémy a modely sa podieľajú na každodenných rozhodovacích úlohách, ako sú diagnostiky v zdravotníctve alebo odhaľovanie podvodov. V dôsledku toho sú pokroky v AI spolu s zrýchleným prijímaním sprevádzané vyvíjajúcimi sa spoločenskými očakávaniami a rastúcou reguláciou. Neustále vidíme oblasti, kde systémy AI nesplňujú očakávania; odhaľujú nové výzvy; a vlády začínajú regulovať AI riešenia. Preto je dôležité, aby boli tieto modely analyzované s cieľom poskytovať spravodlivé, spoľahlivé, inkluzívne, transparentné a zodpovedné výsledky pre všetkých.

V tomto kurze sa pozrieme na praktické nástroje, ktoré môžu byť použité na posúdenie, či má model problémy so zodpovedným AI. Tradičné techniky ladenia strojového učenia majú tendenciu byť založené na kvantitatívnych výpočtoch, ako je agregovaná presnosť alebo priemerná strata chýb. Predstavte si, čo sa môže stať, keď údaje, ktoré používate na vytvorenie týchto modelov, postrádajú určité demografické skupiny, ako sú rasa, pohlavie, politické názory, náboženstvo, alebo neprimerane zastupujú takéto demografické skupiny. Čo ak je výstup modelu interpretovaný tak, že uprednostňuje určitú demografickú skupinu? To môže viesť k nadmernej alebo nedostatočnej reprezentácii týchto citlivých skupín, čo spôsobuje problémy so spravodlivosťou, inkluzívnosťou alebo spoľahlivosťou modelu. Ďalším faktorom je, že modely strojového učenia sú považované za čierne skrinky, čo sťažuje pochopenie a vysvetlenie toho, čo ovplyvňuje predikciu modelu. Toto sú výzvy, ktorým čelia dátoví vedci a vývojári AI, keď nemajú dostatočné nástroje na ladenie a posúdenie spravodlivosti alebo dôveryhodnosti modelu.

V tejto lekcii sa naučíte, ako ladiť svoje modely pomocou:

- **Analýzy chýb**: identifikácia oblastí v distribúcii údajov, kde má model vysoké miery chýb.
- **Prehľadu modelu**: vykonanie porovnávacej analýzy medzi rôznymi kohortami údajov na odhalenie rozdielov vo výkonnostných metrikách modelu.
- **Analýzy údajov**: skúmanie, kde môže byť nadmerná alebo nedostatočná reprezentácia údajov, ktorá môže skresliť model tak, aby uprednostňoval jednu demografickú skupinu pred druhou.
- **Dôležitosti vlastností**: pochopenie, ktoré vlastnosti ovplyvňujú predikcie modelu na globálnej alebo lokálnej úrovni.

## Predpoklady

Ako predpoklad si prosím preštudujte [Nástroje zodpovedného AI pre vývojárov](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif o nástrojoch zodpovedného AI](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Analýza chýb

Tradičné metriky výkonnosti modelu používané na meranie presnosti sú väčšinou výpočty založené na správnych vs nesprávnych predikciách. Napríklad určenie, že model je presný na 89 % s chybovou stratou 0,001, môže byť považované za dobrý výkon. Chyby však nie sú rovnomerne rozložené v podkladovom súbore údajov. Môžete dosiahnuť skóre presnosti modelu 89 %, ale zistiť, že existujú rôzne oblasti vašich údajov, v ktorých model zlyháva na 42 %. Dôsledky týchto vzorcov zlyhania s určitými skupinami údajov môžu viesť k problémom so spravodlivosťou alebo spoľahlivosťou. Je nevyhnutné pochopiť oblasti, kde model funguje dobre alebo nie. Oblasti údajov, kde má váš model vysoký počet nepresností, sa môžu ukázať ako dôležitá demografická skupina údajov.

![Analyzujte a ladte chyby modelu](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Komponent Analýza chýb na RAI dashboarde ilustruje, ako sú zlyhania modelu rozložené medzi rôznymi kohortami pomocou vizualizácie stromu. To je užitočné pri identifikácii vlastností alebo oblastí, kde je vysoká miera chýb vo vašom súbore údajov. Tým, že vidíte, odkiaľ pochádza väčšina nepresností modelu, môžete začať skúmať príčinu. Môžete tiež vytvárať kohorty údajov na vykonanie analýzy. Tieto kohorty údajov pomáhajú v procese ladenia určiť, prečo je výkon modelu dobrý v jednej kohorte, ale chybný v inej.

![Analýza chýb](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Vizualizačné indikátory na mape stromu pomáhajú rýchlejšie lokalizovať problémové oblasti. Napríklad tmavší odtieň červenej farby na uzle stromu znamená vyššiu mieru chýb.

Heatmapa je ďalšou vizualizačnou funkciou, ktorú môžu používatelia použiť na skúmanie miery chýb pomocou jednej alebo dvoch vlastností na nájdenie prispievateľa k chybám modelu v celom súbore údajov alebo kohortách.

![Heatmapa analýzy chýb](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Použite analýzu chýb, keď potrebujete:

* Získať hlboké pochopenie toho, ako sú zlyhania modelu rozložené v súbore údajov a medzi viacerými vstupnými a vlastnostnými dimenziami.
* Rozložiť agregované metriky výkonnosti na automatické objavenie chybných kohort na informovanie o vašich cielených krokoch na zmiernenie problémov.

## Prehľad modelu

Hodnotenie výkonnosti modelu strojového učenia si vyžaduje získanie holistického pochopenia jeho správania. To možno dosiahnuť preskúmaním viacerých metrík, ako sú miera chýb, presnosť, recall, precision alebo MAE (Mean Absolute Error), na odhalenie rozdielov medzi výkonnostnými metrikami. Jedna metrika výkonnosti môže vyzerať skvele, ale nepresnosti môžu byť odhalené v inej metrike. Okrem toho porovnávanie metrík na odhalenie rozdielov v celom súbore údajov alebo kohortách pomáha objasniť, kde model funguje dobre alebo nie. To je obzvlášť dôležité pri sledovaní výkonu modelu medzi citlivými vs necitlivými vlastnosťami (napr. rasa pacienta, pohlavie alebo vek), aby sa odhalila potenciálna nespravodlivosť modelu. Napríklad zistenie, že model je viac chybný v kohorte, ktorá má citlivé vlastnosti, môže odhaliť potenciálnu nespravodlivosť modelu.

Komponent Prehľad modelu na RAI dashboarde pomáha nielen pri analýze výkonnostných metrík reprezentácie údajov v kohorte, ale dáva používateľom možnosť porovnávať správanie modelu medzi rôznymi kohortami.

![Datasetové kohorty - prehľad modelu na RAI dashboarde](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Funkcia analýzy založená na vlastnostiach komponentu umožňuje používateľom zúžiť podskupiny údajov v rámci konkrétnej vlastnosti na identifikáciu anomálií na granulárnej úrovni. Napríklad dashboard má zabudovanú inteligenciu na automatické generovanie kohort pre používateľom vybranú vlastnosť (napr. *"time_in_hospital < 3"* alebo *"time_in_hospital >= 7"*). To umožňuje používateľovi izolovať konkrétnu vlastnosť z väčšej skupiny údajov, aby zistil, či je kľúčovým ovplyvňovateľom chybných výsledkov modelu.

![Kohorty vlastností - prehľad modelu na RAI dashboarde](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Komponent Prehľad modelu podporuje dve triedy metrík rozdielov:

**Rozdiely vo výkonnosti modelu**: Tieto sady metrík vypočítavajú rozdiely (disparity) v hodnotách vybranej metriky výkonnosti medzi podskupinami údajov. Tu je niekoľko príkladov:

* Rozdiely v miere presnosti
* Rozdiely v miere chýb
* Rozdiely v precision
* Rozdiely v recall
* Rozdiely v Mean Absolute Error (MAE)

**Rozdiely v miere výberu**: Táto metrika obsahuje rozdiely v miere výberu (priaznivá predikcia) medzi podskupinami. Príkladom je rozdiel v miere schvaľovania úverov. Miera výberu znamená podiel dátových bodov v každej triede klasifikovaných ako 1 (v binárnej klasifikácii) alebo distribúciu predikčných hodnôt (v regresii).

## Analýza údajov

> "Ak budete údaje mučiť dostatočne dlho, priznajú sa k čomukoľvek" - Ronald Coase

Toto tvrdenie znie extrémne, ale je pravda, že údaje môžu byť manipulované na podporu akéhokoľvek záveru. Takáto manipulácia sa niekedy môže stať neúmyselne. Ako ľudia máme všetci predsudky a často je ťažké vedome vedieť, kedy zavádzame predsudky do údajov. Zaručenie spravodlivosti v AI a strojovom učení zostáva komplexnou výzvou.

Údaje sú veľkým slepým miestom pre tradičné metriky výkonnosti modelu. Môžete mať vysoké skóre presnosti, ale to nemusí vždy odrážať podkladové predsudky v údajoch, ktoré by mohli byť vo vašom súbore údajov. Napríklad, ak má súbor údajov zamestnancov 27 % žien na výkonných pozíciách v spoločnosti a 73 % mužov na rovnakej úrovni, model AI na inzerciu pracovných miest trénovaný na týchto údajoch môže cieliť prevažne na mužské publikum pre seniorné pracovné pozície. Táto nerovnováha v údajoch skreslila predikciu modelu tak, aby uprednostňovala jedno pohlavie. To odhaľuje problém spravodlivosti, kde je v AI modeli predsudok voči pohlaviu.

Komponent Analýza údajov na RAI dashboarde pomáha identifikovať oblasti, kde je nadmerná alebo nedostatočná reprezentácia v súbore údajov. Pomáha používateľom diagnostikovať príčinu chýb a problémov so spravodlivosťou, ktoré sú spôsobené nerovnováhou údajov alebo nedostatkom reprezentácie určitej skupiny údajov. To dáva používateľom možnosť vizualizovať súbory údajov na základe predikovaných a skutočných výsledkov, skupín chýb a konkrétnych vlastností. Niekedy objavenie nedostatočne zastúpenej skupiny údajov môže tiež odhaliť, že model sa neučí dobre, a preto má vysoké nepresnosti. Model s predsudkami v údajoch nie je len problémom spravodlivosti, ale ukazuje, že model nie je inkluzívny ani spoľahlivý.

![Komponent Analýza údajov na RAI dashboarde](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Použite analýzu údajov, keď potrebujete:

* Preskúmať štatistiky vášho súboru údajov výberom rôznych filtrov na rozdelenie údajov do rôznych dimenzií (známych ako kohorty).
* Pochopiť distribúciu vášho súboru údajov medzi rôznymi kohortami a skupinami vlastností.
* Určiť, či vaše zistenia týkajúce sa spravodlivosti, analýzy chýb a kauzality (odvodené z iných komponentov dashboardu) sú výsledkom distribúcie vášho súboru údajov.
* Rozhodnúť, v ktorých oblastiach zbierať viac údajov na zmiernenie chýb spôsobených problémami s reprezentáciou, šumom v označení, šumom vo vlastnostiach, predsudkami v označení a podobnými faktormi.

## Interpretácia modelu

Modely strojového učenia majú tendenciu byť čiernymi skrinkami. Pochopenie, ktoré kľúčové vlastnosti údajov ovplyvňujú predikciu modelu, môže byť náročné. Je dôležité poskytnúť transparentnosť, prečo model robí určitú predikciu. Napríklad, ak AI systém predikuje, že diabetický pacient je ohrozený opätovným prijatím do nemocnice do 30 dní, mal by byť schopný poskytnúť podporné údaje, ktoré viedli k jeho predikcii. Mať podporné indikátory údajov prináša transparentnosť, ktorá pomáha klinikám alebo nemocniciam robiť dobre informované rozhodnutia. Okrem toho schopnosť vysvetliť, prečo model urobil predikciu pre konkrétneho pacienta, umožňuje zodpovednosť voči zdravotným reguláciám. Keď používate modely strojového učenia spôsobmi, ktoré ovplyvňujú životy ľudí, je nevyhnutné pochopiť a vysvetliť, čo ovplyvňuje správanie modelu. Vysvetliteľnosť a interpretácia modelu pomáha odpovedať na otázky v scenároch, ako sú:

* Ladenie modelu: Prečo môj model urobil túto chybu? Ako môžem zlepšiť svoj model?
* Spolupráca človek-AI: Ako môžem pochopiť a dôverovať rozhodnutiam modelu?
* Regulácia: Spĺňa môj model právne požiadavky?

Komponent Dôležitosť vlastností na RAI dashboarde vám pomáha ladiť a získať komplexné pochopenie toho, ako model robí predikcie. Je to tiež užitočný nástroj pre profesionálov v oblasti strojového učenia a rozhodovacích činiteľov na vysvetlenie a ukázanie dôkazov vlastností ovplyvňujúcich správanie modelu pre reguláciu. Používatelia môžu ďalej skúmať globálne aj lokálne vysvetlenia na validáciu, ktoré vlastnosti ovplyvňujú predikciu modelu. Globálne vysvetlenia uvádzajú najdôležitejšie vlastnosti, ktoré ovplyvnili celkovú predikciu modelu. Lokálne vysvetlenia zobrazujú, ktoré vlastnosti viedli k predikcii modelu pre konkrétny prípad. Schopnosť hodnotiť lokálne vysvetlenia je tiež užitočná pri ladení alebo audite konkrétneho prípadu na lepšie pochopenie a interpretáciu, prečo model urobil presnú alebo nepresnú predikciu.

![Komponent Dôležitosť vlastností na RAI dashboarde](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Globálne vysvetlenia: Napríklad, ktoré vlastnosti ovplyvňujú celkové správanie modelu na opätovné prijatie diabetických pacientov do nemocnice?
* Lokálne vysvetlenia: Napríklad, prečo bol diabetický pacient nad 60 rokov s predchádzajúcimi hospitalizáciami predikovaný na opätovné prijatie alebo neprijatie do nemocnice do 30 dní?

V procese ladenia výkonu modelu medzi rôznymi kohortami Dôležitosť vlastností ukazuje, aký vplyv má vlastnosť na kohorty. Pomáha odhaliť anomálie pri porovnávaní úrovne vplyvu vlastnosti na chybnú predikciu modelu. Komponent Dôležitosť vlastností môže ukázať, ktoré
- **Nadmerné alebo nedostatočné zastúpenie**. Ide o to, že určitá skupina nie je viditeľná v určitej profesii, a akákoľvek služba alebo funkcia, ktorá to naďalej podporuje, prispieva k škodám.

### Azure RAI dashboard

[Azure RAI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) je postavený na open-source nástrojoch vyvinutých poprednými akademickými inštitúciami a organizáciami vrátane Microsoftu, ktoré sú nevyhnutné pre dátových vedcov a vývojárov AI na lepšie pochopenie správania modelov, objavovanie a zmierňovanie nežiaducich problémov z AI modelov.

- Naučte sa, ako používať rôzne komponenty, prečítaním dokumentácie k RAI dashboardu [docs.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Pozrite si niektoré [ukážkové notebooky RAI dashboardu](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) na ladenie zodpovednejších AI scenárov v Azure Machine Learning.

---
## 🚀 Výzva

Aby sme zabránili zavádzaniu štatistických alebo dátových predsudkov už na začiatku, mali by sme:

- zabezpečiť rozmanitosť pozadí a perspektív medzi ľuďmi pracujúcimi na systémoch
- investovať do datasetov, ktoré odrážajú rozmanitosť našej spoločnosti
- vyvíjať lepšie metódy na detekciu a opravu predsudkov, keď sa objavia

Premýšľajte o reálnych situáciách, kde je nespravodlivosť evidentná pri budovaní a používaní modelov. Čo ďalšie by sme mali zvážiť?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)
## Prehľad a samostatné štúdium

V tejto lekcii ste sa naučili niektoré praktické nástroje na začlenenie zodpovednej AI do strojového učenia.

Pozrite si tento workshop, aby ste sa hlbšie ponorili do tém:

- Responsible AI Dashboard: Jednotné miesto na operacionalizáciu RAI v praxi od Besmiry Nushi a Mehrnoosh Sameki

[![Responsible AI Dashboard: Jednotné miesto na operacionalizáciu RAI v praxi](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Jednotné miesto na operacionalizáciu RAI v praxi")

> 🎥 Kliknite na obrázok vyššie pre video: Responsible AI Dashboard: Jednotné miesto na operacionalizáciu RAI v praxi od Besmiry Nushi a Mehrnoosh Sameki

Odkážte na nasledujúce materiály, aby ste sa dozvedeli viac o zodpovednej AI a o tom, ako budovať dôveryhodnejšie modely:

- Microsoftove nástroje RAI dashboardu na ladenie ML modelov: [Responsible AI tools resources](https://aka.ms/rai-dashboard)

- Preskúmajte Responsible AI toolkit: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsoftove centrum zdrojov RAI: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoftova výskumná skupina FATE: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Zadanie

[Preskúmajte RAI dashboard](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby na automatický preklad [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, upozorňujeme, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre dôležité informácie sa odporúča profesionálny ľudský preklad. Nezodpovedáme za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.