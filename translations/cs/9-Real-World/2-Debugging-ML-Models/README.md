<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T00:15:11+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "cs"
}
-->
# Postscript: Ladění modelů strojového učení pomocí komponent Responsible AI dashboardu

## [Kvíz před přednáškou](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

Strojové učení ovlivňuje náš každodenní život. AI se dostává do některých z nejdůležitějších systémů, které ovlivňují nás jako jednotlivce i naši společnost, od zdravotnictví, financí, vzdělávání až po zaměstnanost. Například systémy a modely se podílejí na každodenních rozhodovacích úlohách, jako jsou diagnózy ve zdravotnictví nebo odhalování podvodů. V důsledku toho jsou pokroky v AI spolu s rychlou adopcí doprovázeny měnícími se společenskými očekáváními a rostoucí regulací. Neustále vidíme oblasti, kde systémy AI nesplňují očekávání, odhalují nové výzvy a vlády začínají regulovat AI řešení. Je tedy důležité, aby tyto modely byly analyzovány tak, aby poskytovaly spravedlivé, spolehlivé, inkluzivní, transparentní a odpovědné výsledky pro všechny.

V tomto kurzu se podíváme na praktické nástroje, které lze použít k posouzení, zda má model problémy s odpovědnou AI. Tradiční techniky ladění strojového učení bývají založeny na kvantitativních výpočtech, jako je agregovaná přesnost nebo průměrná ztráta chyb. Představte si, co se může stát, když data, která používáte k vytváření těchto modelů, postrádají určité demografické skupiny, jako je rasa, pohlaví, politický názor, náboženství, nebo naopak nepřiměřeně zastupují tyto demografické skupiny. Co když je výstup modelu interpretován tak, že upřednostňuje určitou demografickou skupinu? To může vést k nadměrnému nebo nedostatečnému zastoupení těchto citlivých skupin, což způsobí problémy se spravedlností, inkluzivitou nebo spolehlivostí modelu. Dalším faktorem je, že modely strojového učení jsou považovány za "černé skříňky", což ztěžuje pochopení a vysvětlení toho, co ovlivňuje predikce modelu. Všechny tyto výzvy čelí datoví vědci a vývojáři AI, pokud nemají dostatečné nástroje k ladění a posouzení spravedlnosti nebo důvěryhodnosti modelu.

V této lekci se naučíte ladit své modely pomocí:

- **Analýzy chyb**: identifikace oblastí v distribuci dat, kde má model vysokou míru chyb.
- **Přehledu modelu**: provádění srovnávací analýzy mezi různými datovými kohortami k odhalení rozdílů ve výkonnostních metrikách modelu.
- **Analýzy dat**: zkoumání, kde může docházet k nadměrnému nebo nedostatečnému zastoupení dat, což může zkreslit model ve prospěch jedné demografické skupiny oproti jiné.
- **Důležitosti vlastností**: pochopení, které vlastnosti ovlivňují predikce modelu na globální nebo lokální úrovni.

## Předpoklady

Jako předpoklad si prosím projděte [Nástroje odpovědné AI pro vývojáře](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif o nástrojích odpovědné AI](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Analýza chyb

Tradiční metriky výkonnosti modelu používané k měření přesnosti jsou většinou výpočty založené na správných vs nesprávných predikcích. Například určení, že model je přesný z 89 % s chybovou ztrátou 0,001, může být považováno za dobrý výkon. Chyby však nejsou v základním datovém souboru rozloženy rovnoměrně. Můžete získat skóre přesnosti modelu 89 %, ale zjistit, že existují různé oblasti vašich dat, kde model selhává ve 42 % případů. Důsledky těchto vzorců selhání u určitých datových skupin mohou vést k problémům se spravedlností nebo spolehlivostí. Je zásadní pochopit oblasti, kde model funguje dobře nebo ne. Datové oblasti, kde má váš model vysoký počet nepřesností, se mohou ukázat jako důležitá demografická data.

![Analyzujte a laděte chyby modelu](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Komponenta Analýza chyb na RAI dashboardu ukazuje, jak jsou selhání modelu rozložena napříč různými kohortami pomocí vizualizace stromu. To je užitečné při identifikaci vlastností nebo oblastí, kde je vysoká míra chyb ve vašem datovém souboru. Díky tomu, že vidíte, odkud pochází většina nepřesností modelu, můžete začít zkoumat jejich příčinu. Můžete také vytvořit datové kohorty pro provádění analýzy. Tyto datové kohorty pomáhají v procesu ladění určit, proč je výkon modelu dobrý v jedné kohortě, ale chybný v jiné.

![Analýza chyb](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Vizualizační indikátory na mapě stromu pomáhají rychleji lokalizovat problémové oblasti. Například čím tmavší odstín červené barvy má uzel stromu, tím vyšší je míra chyb.

Heatmapa je další vizualizační funkce, kterou mohou uživatelé použít k vyšetřování míry chyb pomocí jedné nebo dvou vlastností, aby našli přispěvatele k chybám modelu napříč celým datovým souborem nebo kohortami.

![Heatmapa analýzy chyb](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Použijte analýzu chyb, když potřebujete:

* Získat hluboké pochopení toho, jak jsou selhání modelu rozložena napříč datovým souborem a několika vstupními a vlastnostními dimenzemi.
* Rozložit agregované metriky výkonu a automaticky objevit chybné kohorty, které informují o vašich cílených krocích k nápravě.

## Přehled modelu

Hodnocení výkonu modelu strojového učení vyžaduje získání komplexního pochopení jeho chování. Toho lze dosáhnout přezkoumáním více než jedné metriky, jako je míra chyb, přesnost, recall, precision nebo MAE (Mean Absolute Error), aby se odhalily rozdíly mezi výkonnostními metrikami. Jedna metrika výkonu může vypadat skvěle, ale nepřesnosti mohou být odhaleny v jiné metrice. Navíc porovnání metrik pro rozdíly napříč celým datovým souborem nebo kohortami pomáhá osvětlit, kde model funguje dobře nebo ne. To je obzvláště důležité při sledování výkonu modelu mezi citlivými vs necitlivými vlastnostmi (např. rasa pacienta, pohlaví nebo věk), aby se odhalila potenciální nespravedlnost modelu. Například zjištění, že model je více chybný v kohortě, která má citlivé vlastnosti, může odhalit potenciální nespravedlnost modelu.

Komponenta Přehled modelu na RAI dashboardu pomáhá nejen při analýze výkonnostních metrik reprezentace dat v kohortě, ale dává uživatelům možnost porovnávat chování modelu napříč různými kohortami.

![Datové kohorty - přehled modelu na RAI dashboardu](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Funkce analýzy založené na vlastnostech komponenty umožňuje uživatelům zúžit podskupiny dat v rámci konkrétní vlastnosti, aby identifikovali anomálie na granulární úrovni. Například dashboard má vestavěnou inteligenci, která automaticky generuje kohorty pro uživatelem vybranou vlastnost (např. *"time_in_hospital < 3"* nebo *"time_in_hospital >= 7"*). To umožňuje uživateli izolovat konkrétní vlastnost z větší skupiny dat, aby zjistil, zda je klíčovým ovlivňovatelem chybných výsledků modelu.

![Kohorty vlastností - přehled modelu na RAI dashboardu](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Komponenta Přehled modelu podporuje dvě třídy metrik rozdílů:

**Rozdíly ve výkonnosti modelu**: Tyto sady metrik vypočítávají rozdíly (disparity) ve hodnotách vybrané výkonnostní metriky napříč podskupinami dat. Zde je několik příkladů:

* Rozdíly v míře přesnosti
* Rozdíly v míře chyb
* Rozdíly v precision
* Rozdíly v recall
* Rozdíly v průměrné absolutní chybě (MAE)

**Rozdíly v míře výběru**: Tato metrika obsahuje rozdíly v míře výběru (příznivá predikce) mezi podskupinami. Příkladem je rozdíl v míře schvalování půjček. Míra výběru znamená podíl datových bodů v každé třídě klasifikovaných jako 1 (v binární klasifikaci) nebo rozložení hodnot predikce (v regresi).

## Analýza dat

> "Pokud budete data mučit dostatečně dlouho, přiznají cokoliv" - Ronald Coase

Toto tvrzení zní extrémně, ale je pravda, že data mohou být manipulována tak, aby podporovala jakýkoliv závěr. Taková manipulace se někdy může stát neúmyslně. Jako lidé máme všichni předsudky a často je obtížné vědomě vědět, kdy do dat zavádíme předsudky. Zajištění spravedlnosti v AI a strojovém učení zůstává složitou výzvou.

Data jsou velkým slepým místem pro tradiční metriky výkonnosti modelu. Můžete mít vysoké skóre přesnosti, ale to ne vždy odráží základní předsudky v datech, které mohou být ve vašem datovém souboru. Například pokud datový soubor zaměstnanců obsahuje 27 % žen na výkonných pozicích ve firmě a 73 % mužů na stejné úrovni, model AI pro inzerci pracovních míst vyškolený na těchto datech může cílit převážně na mužské publikum pro seniorní pracovní pozice. Tato nerovnováha v datech zkreslila predikci modelu ve prospěch jednoho pohlaví. To odhaluje problém spravedlnosti, kde je v modelu AI přítomna genderová předpojatost.

Komponenta Analýza dat na RAI dashboardu pomáhá identifikovat oblasti, kde je v datovém souboru nadměrné nebo nedostatečné zastoupení. Pomáhá uživatelům diagnostikovat příčinu chyb a problémů se spravedlností, které jsou způsobeny nerovnováhou dat nebo nedostatečným zastoupením určité datové skupiny. To dává uživatelům možnost vizualizovat datové soubory na základě predikovaných a skutečných výsledků, skupin chyb a konkrétních vlastností. Někdy objevení nedostatečně zastoupené datové skupiny může také odhalit, že model se dobře neučí, což vede k vysokým nepřesnostem. Model, který má předsudky v datech, není jen problémem spravedlnosti, ale ukazuje, že model není inkluzivní nebo spolehlivý.

![Komponenta Analýza dat na RAI dashboardu](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Použijte analýzu dat, když potřebujete:

* Prozkoumat statistiky svého datového souboru výběrem různých filtrů pro rozdělení dat do různých dimenzí (známých jako kohorty).
* Pochopit rozložení svého datového souboru napříč různými kohortami a skupinami vlastností.
* Určit, zda vaše zjištění týkající se spravedlnosti, analýzy chyb a kauzality (odvozené z jiných komponent dashboardu) jsou výsledkem rozložení vašeho datového souboru.
* Rozhodnout, v jakých oblastech sbírat více dat, aby se zmírnily chyby způsobené problémy s reprezentací, šumem v označení, šumem ve vlastnostech, předsudky v označení a podobnými faktory.

## Interpretace modelu

Modely strojového učení mají tendenci být "černými skříňkami". Pochopení, které klíčové vlastnosti dat ovlivňují predikci modelu, může být náročné. Je důležité poskytnout transparentnost ohledně toho, proč model dělá určitou predikci. Například pokud systém AI předpovídá, že diabetický pacient je ohrožen opětovným přijetím do nemocnice během méně než 30 dnů, měl by být schopen poskytnout podpůrná data, která vedla k jeho predikci. Mít podpůrné datové indikátory přináší transparentnost, která pomáhá lékařům nebo nemocnicím činit dobře informovaná rozhodnutí. Navíc schopnost vysvětlit, proč model učinil predikci pro konkrétního pacienta, umožňuje odpovědnost vůči zdravotním regulacím. Když používáte modely strojového učení způsoby, které ovlivňují životy lidí, je zásadní pochopit a vysvětlit, co ovlivňuje chování modelu. Vysvětlitelnost a interpretace modelu pomáhá odpovědět na otázky v situacích, jako jsou:

* Ladění modelu: Proč můj model udělal tuto chybu? Jak mohu svůj model zlepšit?
* Spolupráce člověk-AI: Jak mohu pochopit a důvěřovat rozhodnutím modelu?
* Regulativní shoda: Splňuje můj model právní požadavky?

Komponenta Důležitost vlastností na RAI dashboardu vám pomáhá ladit a získat komplexní pochopení toho, jak model dělá predikce. Je to také užitečný nástroj pro profesionály v oblasti strojového učení a rozhodovací činitele, kteří potřebují vysvětlit a ukázat důkazy o vlastnostech ovlivňujících chování modelu pro regulativní shodu. Uživatelé mohou dále zkoumat globální i lokální vysvětlení, aby ověřili, které vlastnosti ovlivňují predikce modelu. Globální vysvětlení uvádí hlavní vlastnosti, které ovlivnily celkovou predikci modelu. Lokální vysvětlení ukazuje, které vlastnosti vedly k predikci modelu pro konkrétní případ. Schopnost hodnotit lokální vysvětlení je také užitečná při ladění nebo auditu konkrétního případu, aby bylo možné lépe pochopit a interpretovat, proč model učinil přesnou nebo nepřesnou predikci.

![Komponenta Důležitost vlastností na RAI dashboardu](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Globální vysvětlení: Například, které vlastnosti ovlivňují celkové chování modelu pro opětovné přijetí diabetických pacientů do nemocnice?
* Lokální vysvětlení: Například, proč byl diabetický pacient starší 60 let s předchozími hospitalizacemi předpovězen jako opětovně přijatý nebo nepřijatý do nemocnice během 30 dnů?

V procesu ladění výkonu modelu napříč různými kohortami Důležitost vlastností ukazuje, jakou úroveň vlivu má vlastnost napříč kohortami. Pomáhá odhalit anomálie při porovnávání úrovně vlivu vlastnosti na chybné predikce modelu. Komponenta Důležitost vlastností může ukázat, které hodnoty ve vlastnosti pozitivně nebo negativně ovlivnily výsledek modelu. Například pokud model učinil nepřesnou predikci, komponenta vám umožňuje podrobněji prozkoumat a určit, které vlastnosti nebo hodnoty vlastností ovlivnily predikci. Tato úroveň detailu pomáhá nejen při ladění, ale poskytuje transparentnost a odpovědnost v auditních situacích. Nakonec komponenta může pomoci identifikovat problémy se spravedlností. Například pokud citlivá vlastnost, jako je etnicita nebo pohlaví, má vysoký vliv na predikci modelu, může to být známka rasové nebo
- **Nad- nebo pod-reprezentace**. Myšlenka spočívá v tom, že určitá skupina není zastoupena v určité profesi, a jakákoli služba nebo funkce, která toto nadále podporuje, přispívá k poškození.

### Azure RAI dashboard

[Azure RAI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) je postaven na open-source nástrojích vyvinutých předními akademickými institucemi a organizacemi, včetně Microsoftu. Tyto nástroje jsou klíčové pro datové vědce a vývojáře AI, aby lépe porozuměli chování modelů, objevovali a zmírňovali nežádoucí problémy v modelech AI.

- Naučte se, jak používat různé komponenty, přečtením dokumentace k RAI dashboardu [docs.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Podívejte se na některé ukázkové [notebooky RAI dashboardu](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) pro ladění odpovědnějších scénářů AI v Azure Machine Learning.

---
## 🚀 Výzva

Abychom zabránili zavádění statistických nebo datových předsudků, měli bychom:

- zajistit rozmanitost zázemí a perspektiv mezi lidmi pracujícími na systémech
- investovat do datových sad, které odrážejí rozmanitost naší společnosti
- vyvíjet lepší metody pro detekci a nápravu předsudků, když k nim dojde

Přemýšlejte o reálných situacích, kde je nespravedlnost zřejmá při vytváření a používání modelů. Co dalšího bychom měli zvážit?

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)
## Přehled a samostudium

V této lekci jste se naučili některé praktické nástroje pro začlenění odpovědné AI do strojového učení.

Podívejte se na tento workshop, abyste se ponořili hlouběji do témat:

- Responsible AI Dashboard: Jednotné místo pro operacionalizaci RAI v praxi od Besmiry Nushi a Mehrnoosh Sameki

[![Responsible AI Dashboard: Jednotné místo pro operacionalizaci RAI v praxi](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Jednotné místo pro operacionalizaci RAI v praxi")

> 🎥 Klikněte na obrázek výše pro video: Responsible AI Dashboard: Jednotné místo pro operacionalizaci RAI v praxi od Besmiry Nushi a Mehrnoosh Sameki

Odkazujte na následující materiály, abyste se dozvěděli více o odpovědné AI a jak vytvářet důvěryhodnější modely:

- Microsoftovy nástroje RAI dashboardu pro ladění modelů ML: [Responsible AI tools resources](https://aka.ms/rai-dashboard)

- Prozkoumejte sadu nástrojů Responsible AI: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsoftovo centrum zdrojů RAI: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoftova výzkumná skupina FATE: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Úkol

[Prozkoumejte RAI dashboard](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.