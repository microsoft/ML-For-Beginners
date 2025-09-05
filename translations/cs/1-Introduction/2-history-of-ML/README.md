<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T00:33:16+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "cs"
}
-->
# Historie strojového učení

![Shrnutí historie strojového učení ve sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pro začátečníky - Historie strojového učení](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML pro začátečníky - Historie strojového učení")

> 🎥 Klikněte na obrázek výše pro krátké video k této lekci.

V této lekci projdeme hlavní milníky v historii strojového učení a umělé inteligence.

Historie umělé inteligence (AI) jako oboru je úzce propojena s historií strojového učení, protože algoritmy a výpočetní pokroky, které tvoří základ ML, přispěly k rozvoji AI. Je užitečné si uvědomit, že i když se tyto obory jako samostatné oblasti zkoumání začaly formovat v 50. letech, důležité [algoritmické, statistické, matematické, výpočetní a technické objevy](https://wikipedia.org/wiki/Timeline_of_machine_learning) předcházely a překrývaly toto období. Ve skutečnosti lidé přemýšlejí o těchto otázkách již [stovky let](https://wikipedia.org/wiki/History_of_artificial_intelligence): tento článek pojednává o historických intelektuálních základech myšlenky „myslícího stroje“.

---
## Významné objevy

- 1763, 1812 [Bayesův teorém](https://wikipedia.org/wiki/Bayes%27_theorem) a jeho předchůdci. Tento teorém a jeho aplikace tvoří základ inferencí, popisujících pravděpodobnost události na základě předchozích znalostí.
- 1805 [Metoda nejmenších čtverců](https://wikipedia.org/wiki/Least_squares) od francouzského matematika Adriena-Marie Legendra. Tato metoda, kterou se naučíte v naší jednotce o regresi, pomáhá při přizpůsobování dat.
- 1913 [Markovovy řetězce](https://wikipedia.org/wiki/Markov_chain), pojmenované po ruském matematikovi Andreji Markovovi, popisují sekvenci možných událostí na základě předchozího stavu.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) je typ lineárního klasifikátoru, který vynalezl americký psycholog Frank Rosenblatt a který tvoří základ pokroků v hlubokém učení.

---

- 1967 [Nejbližší soused](https://wikipedia.org/wiki/Nearest_neighbor) je algoritmus původně navržený pro mapování tras. V kontextu ML se používá k detekci vzorců.
- 1970 [Backpropagace](https://wikipedia.org/wiki/Backpropagation) se používá k trénování [dopředných neuronových sítí](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Rekurentní neuronové sítě](https://wikipedia.org/wiki/Recurrent_neural_network) jsou umělé neuronové sítě odvozené z dopředných neuronových sítí, které vytvářejí časové grafy.

✅ Udělejte si malý průzkum. Které další data v historii ML a AI považujete za klíčové?

---
## 1950: Stroje, které myslí

Alan Turing, skutečně výjimečný člověk, který byl [veřejností v roce 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) zvolen největším vědcem 20. století, je považován za zakladatele konceptu „stroje, který může myslet“. Turing se potýkal s odpůrci i s vlastní potřebou empirických důkazů tohoto konceptu, mimo jiné vytvořením [Turingova testu](https://www.bbc.com/news/technology-18475646), který budete zkoumat v našich lekcích o NLP.

---
## 1956: Letní výzkumný projekt v Dartmouthu

"Letní výzkumný projekt v Dartmouthu o umělé inteligenci byl klíčovou událostí pro AI jako obor," a právě zde byl termín 'umělá inteligence' poprvé použit ([zdroj](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Každý aspekt učení nebo jakákoli jiná vlastnost inteligence může být v principu tak přesně popsán, že stroj může být vytvořen tak, aby jej simuloval.

---

Vedoucí výzkumu, profesor matematiky John McCarthy, doufal, že "pokročí na základě domněnky, že každý aspekt učení nebo jakákoli jiná vlastnost inteligence může být v principu tak přesně popsán, že stroj může být vytvořen tak, aby jej simuloval." Mezi účastníky patřil další významný odborník v oboru, Marvin Minsky.

Workshop je považován za iniciátora a podporovatele několika diskusí, včetně "vzestupu symbolických metod, systémů zaměřených na omezené oblasti (rané expertní systémy) a deduktivních systémů versus induktivních systémů." ([zdroj](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Zlaté roky"

Od 50. let do poloviny 70. let panoval optimismus, že AI může vyřešit mnoho problémů. V roce 1967 Marvin Minsky sebevědomě prohlásil: "Během jedné generace ... problém vytvoření 'umělé inteligence' bude podstatně vyřešen." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Výzkum zpracování přirozeného jazyka vzkvétal, vyhledávání bylo zdokonaleno a učiněno výkonnějším a byl vytvořen koncept 'mikrosvětů', kde byly jednoduché úkoly prováděny pomocí jednoduchých jazykových instrukcí.

---

Výzkum byl dobře financován vládními agenturami, byly učiněny pokroky ve výpočtech a algoritmech a byly vytvořeny prototypy inteligentních strojů. Některé z těchto strojů zahrnují:

* [Shakey robot](https://wikipedia.org/wiki/Shakey_the_robot), který se mohl pohybovat a rozhodovat, jak inteligentně vykonávat úkoly.

    ![Shakey, inteligentní robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey v roce 1972

---

* Eliza, raný 'chatterbot', mohla komunikovat s lidmi a působit jako primitivní 'terapeut'. O Elize se dozvíte více v lekcích o NLP.

    ![Eliza, bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > Verze Elizy, chatbotu

---

* "Blocks world" byl příklad mikrosvěta, kde bylo možné bloky stohovat a třídit, a experimenty s výukou strojů k rozhodování mohly být testovány. Pokroky vytvořené s knihovnami jako [SHRDLU](https://wikipedia.org/wiki/SHRDLU) pomohly posunout zpracování jazyka vpřed.

    [![blocks world s SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world s SHRDLU")

    > 🎥 Klikněte na obrázek výše pro video: Blocks world s SHRDLU

---
## 1974 - 1980: "Zima AI"

Do poloviny 70. let se ukázalo, že složitost vytváření 'inteligentních strojů' byla podceněna a její sliby, vzhledem k dostupné výpočetní síle, byly přehnané. Financování vyschlo a důvěra v obor se zpomalila. Některé problémy, které ovlivnily důvěru, zahrnují:
---
- **Omezení**. Výpočetní síla byla příliš omezená.
- **Kombinatorická exploze**. Počet parametrů potřebných k trénování rostl exponenciálně, jak bylo na počítače kladeno více požadavků, bez paralelního vývoje výpočetní síly a schopností.
- **Nedostatek dat**. Nedostatek dat bránil procesu testování, vývoje a zdokonalování algoritmů.
- **Pokládáme správné otázky?**. Samotné otázky, které byly kladeny, začaly být zpochybňovány. Výzkumníci začali čelit kritice ohledně svých přístupů:
  - Turingovy testy byly zpochybněny mimo jiné teorií 'čínského pokoje', která tvrdila, že "programování digitálního počítače může způsobit, že se zdá, že rozumí jazyku, ale nemůže vytvořit skutečné porozumění." ([zdroj](https://plato.stanford.edu/entries/chinese-room/))
  - Etika zavádění umělých inteligencí, jako je "terapeut" ELIZA, do společnosti byla zpochybněna.

---

Současně se začaly formovat různé školy myšlení AI. Byla vytvořena dichotomie mezi ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) přístupy. _Scruffy_ laboratoře upravovaly programy hodiny, dokud nedosáhly požadovaných výsledků. _Neat_ laboratoře "se zaměřovaly na logiku a formální řešení problémů". ELIZA a SHRDLU byly známé _scruffy_ systémy. V 80. letech, kdy vznikla poptávka po reprodukovatelnosti ML systémů, se postupně dostal do popředí _neat_ přístup, protože jeho výsledky jsou lépe vysvětlitelné.

---
## 1980s Expertní systémy

Jak obor rostl, jeho přínos pro podnikání se stal jasnějším, a v 80. letech se rozšířily 'expertní systémy'. "Expertní systémy byly jednou z prvních skutečně úspěšných forem softwaru umělé inteligence (AI)." ([zdroj](https://wikipedia.org/wiki/Expert_system)).

Tento typ systému je vlastně _hybridní_, skládající se částečně z pravidlového enginu definujícího obchodní požadavky a inferenčního enginu, který využíval pravidlový systém k odvozování nových faktů.

Toto období také přineslo zvýšenou pozornost věnovanou neuronovým sítím.

---
## 1987 - 1993: AI 'Ochlazení'

Rozšíření specializovaného hardwaru expertních systémů mělo nešťastný efekt přílišné specializace. Vzestup osobních počítačů také konkuroval těmto velkým, specializovaným, centralizovaným systémům. Demokratizace výpočetní techniky začala a nakonec připravila cestu pro moderní explozi velkých dat.

---
## 1993 - 2011

Toto období přineslo novou éru pro ML a AI, aby mohly řešit některé problémy způsobené dříve nedostatkem dat a výpočetní síly. Množství dat začalo rychle narůstat a být dostupnější, k lepšímu i k horšímu, zejména s příchodem chytrého telefonu kolem roku 2007. Výpočetní síla se exponenciálně rozšířila a algoritmy se vyvíjely souběžně. Obor začal nabývat na zralosti, protože volné dny minulosti se začaly formovat do skutečné disciplíny.

---
## Současnost

Dnes strojové učení a AI zasahují téměř do každé části našeho života. Toto období vyžaduje pečlivé pochopení rizik a potenciálních dopadů těchto algoritmů na lidské životy. Jak uvedl Brad Smith z Microsoftu: "Informační technologie vyvolávají otázky, které se dotýkají základních lidských práv, jako je ochrana soukromí a svoboda projevu. Tyto otázky zvyšují odpovědnost technologických společností, které tyto produkty vytvářejí. Podle našeho názoru také volají po promyšlené vládní regulaci a po vývoji norem kolem přijatelných použití" ([zdroj](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Zůstává otázkou, co přinese budoucnost, ale je důležité porozumět těmto počítačovým systémům a softwaru a algoritmům, které provozují. Doufáme, že vám tento kurz pomůže získat lepší porozumění, abyste si mohli udělat vlastní názor.

[![Historie hlubokého učení](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Historie hlubokého učení")
> 🎥 Klikněte na obrázek výše pro video: Yann LeCun diskutuje historii hlubokého učení v této přednášce

---
## 🚀Výzva

Ponořte se do jednoho z těchto historických momentů a dozvězte se více o lidech, kteří za nimi stojí. Jsou to fascinující osobnosti a žádný vědecký objev nikdy nevznikl v kulturním vakuu. Co objevíte?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

---
## Přehled & Samostudium

Zde jsou položky ke sledování a poslechu:

[Podcast, kde Amy Boyd diskutuje vývoj AI](http://runasradio.com/Shows/Show/739)

[![Historie AI od Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Historie AI od Amy Boyd")

---

## Úkol

[Vytvořte časovou osu](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.