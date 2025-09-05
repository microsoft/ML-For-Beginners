<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T00:26:41+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "cs"
}
-->
# Techniky strojového učení

Proces vytváření, používání a udržování modelů strojového učení a dat, která využívají, se výrazně liší od mnoha jiných vývojových pracovních postupů. V této lekci tento proces objasníme a nastíníme hlavní techniky, které je třeba znát. Naučíte se:

- Porozumět procesům, které jsou základem strojového učení na vysoké úrovni.
- Prozkoumat základní pojmy, jako jsou „modely“, „predikce“ a „trénovací data“.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

[![ML pro začátečníky - Techniky strojového učení](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML pro začátečníky - Techniky strojového učení")

> 🎥 Klikněte na obrázek výše pro krátké video, které vás provede touto lekcí.

## Úvod

Na vysoké úrovni se tvorba procesů strojového učení (ML) skládá z několika kroků:

1. **Určete otázku**. Většina procesů ML začíná položením otázky, na kterou nelze odpovědět jednoduchým podmíněným programem nebo pravidlovým systémem. Tyto otázky se často týkají predikcí na základě sbírky dat.
2. **Sbírejte a připravte data**. Abyste mohli odpovědět na svou otázku, potřebujete data. Kvalita a někdy i množství vašich dat určí, jak dobře můžete odpovědět na svou původní otázku. Vizualizace dat je důležitým aspektem této fáze. Tato fáze také zahrnuje rozdělení dat na trénovací a testovací skupinu pro vytvoření modelu.
3. **Vyberte metodu trénování**. V závislosti na vaší otázce a povaze vašich dat musíte zvolit způsob, jakým chcete model trénovat, aby co nejlépe odrážel vaše data a poskytoval přesné predikce. Tato část procesu ML vyžaduje specifické odborné znalosti a často značné množství experimentování.
4. **Trénujte model**. Pomocí vašich trénovacích dat použijete různé algoritmy k trénování modelu, aby rozpoznal vzory v datech. Model může využívat interní váhy, které lze upravit tak, aby upřednostňoval určité části dat před jinými, a tím vytvořil lepší model.
5. **Vyhodnoťte model**. Použijete data, která model nikdy předtím neviděl (vaše testovací data), abyste zjistili, jak model funguje.
6. **Ladění parametrů**. Na základě výkonu vašeho modelu můžete proces zopakovat s různými parametry nebo proměnnými, které ovládají chování algoritmů použitých k trénování modelu.
7. **Predikujte**. Použijte nové vstupy k otestování přesnosti vašeho modelu.

## Jakou otázku položit

Počítače jsou obzvláště zdatné v objevování skrytých vzorů v datech. Tato schopnost je velmi užitečná pro výzkumníky, kteří mají otázky o dané oblasti, na které nelze snadno odpovědět vytvořením pravidlového systému založeného na podmínkách. Například při aktuárské úloze by datový vědec mohl vytvořit ručně sestavená pravidla týkající se úmrtnosti kuřáků vs. nekuřáků.

Když se však do rovnice přidá mnoho dalších proměnných, model ML může být efektivnější při predikci budoucích úmrtnostních sazeb na základě minulých zdravotních záznamů. Veselejším příkladem může být předpovídání počasí na měsíc duben v dané lokalitě na základě dat, která zahrnují zeměpisnou šířku, délku, změny klimatu, blízkost oceánu, vzory proudění vzduchu a další.

✅ Tato [prezentace](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o modelech počasí nabízí historický pohled na využití ML v analýze počasí.  

## Úkoly před vytvořením modelu

Než začnete vytvářet svůj model, je třeba splnit několik úkolů. Abyste mohli otestovat svou otázku a vytvořit hypotézu na základě predikcí modelu, musíte identifikovat a nakonfigurovat několik prvků.

### Data

Abyste mohli odpovědět na svou otázku s jakoukoli jistotou, potřebujete dostatečné množství dat správného typu. V této fázi musíte udělat dvě věci:

- **Sbírejte data**. S ohledem na předchozí lekci o spravedlnosti v analýze dat sbírejte svá data pečlivě. Buďte si vědomi zdrojů těchto dat, jakýchkoli inherentních předsudků, které mohou obsahovat, a dokumentujte jejich původ.
- **Připravte data**. Proces přípravy dat zahrnuje několik kroků. Možná budete muset data shromáždit a normalizovat, pokud pocházejí z různých zdrojů. Kvalitu a množství dat můžete zlepšit různými metodami, například převodem textových řetězců na čísla (jak to děláme v [Clusteringu](../../5-Clustering/1-Visualize/README.md)). Můžete také generovat nová data na základě původních (jak to děláme v [Klasifikaci](../../4-Classification/1-Introduction/README.md)). Data můžete čistit a upravovat (jak to uděláme před lekcí o [Webové aplikaci](../../3-Web-App/README.md)). Nakonec je možná budete muset náhodně uspořádat a promíchat, v závislosti na vašich trénovacích technikách.

✅ Po sběru a zpracování dat si udělejte chvíli na to, abyste zjistili, zda jejich struktura umožní odpovědět na vaši zamýšlenou otázku. Může se stát, že data nebudou dobře fungovat pro váš daný úkol, jak zjistíme v našich lekcích o [Clusteringu](../../5-Clustering/1-Visualize/README.md)!

### Vlastnosti a cíl

[Vlastnost](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) je měřitelná vlastnost vašich dat. V mnoha datových sadách je vyjádřena jako záhlaví sloupce, například „datum“, „velikost“ nebo „barva“. Vaše proměnná vlastnosti, obvykle reprezentovaná jako `X` v kódu, představuje vstupní proměnnou, která bude použita k trénování modelu.

Cíl je věc, kterou se snažíte předpovědět. Cíl, obvykle reprezentovaný jako `y` v kódu, představuje odpověď na otázku, kterou se snažíte položit svým datům: v prosinci, jakou **barvu** budou mít nejlevnější dýně? V San Francisku, které čtvrti budou mít nejlepší **cenu** nemovitostí? Někdy se cíl označuje také jako atribut štítku.

### Výběr proměnné vlastnosti

🎓 **Výběr vlastností a extrakce vlastností** Jak poznáte, kterou proměnnou zvolit při vytváření modelu? Pravděpodobně projdete procesem výběru vlastností nebo extrakce vlastností, abyste zvolili správné proměnné pro nejvýkonnější model. Nejsou to však stejné věci: „Extrakce vlastností vytváří nové vlastnosti z funkcí původních vlastností, zatímco výběr vlastností vrací podmnožinu vlastností.“ ([zdroj](https://wikipedia.org/wiki/Feature_selection))

### Vizualizace dat

Důležitým aspektem nástrojů datového vědce je schopnost vizualizovat data pomocí několika vynikajících knihoven, jako jsou Seaborn nebo MatPlotLib. Vizualizace dat vám může umožnit odhalit skryté korelace, které můžete využít. Vaše vizualizace vám také mohou pomoci odhalit předsudky nebo nevyvážená data (jak zjistíme v [Klasifikaci](../../4-Classification/2-Classifiers-1/README.md)).

### Rozdělení datové sady

Před trénováním je třeba rozdělit datovou sadu na dvě nebo více částí nerovnoměrné velikosti, které stále dobře reprezentují data.

- **Trénovací sada**. Tato část datové sady je použita k trénování modelu. Tato sada tvoří většinu původní datové sady.
- **Testovací sada**. Testovací datová sada je nezávislá skupina dat, často získaná z původních dat, kterou používáte k potvrzení výkonu vytvořeného modelu.
- **Validační sada**. Validační sada je menší nezávislá skupina příkladů, kterou používáte k ladění hyperparametrů nebo architektury modelu, aby se zlepšil jeho výkon. V závislosti na velikosti vašich dat a otázce, kterou pokládáte, možná nebudete muset tuto třetí sadu vytvářet (jak poznamenáváme v [Časových řadách](../../7-TimeSeries/1-Introduction/README.md)).

## Vytváření modelu

Pomocí vašich trénovacích dat je vaším cílem vytvořit model, tedy statistické vyjádření vašich dat, pomocí různých algoritmů k jeho **trénování**. Trénování modelu ho vystavuje datům a umožňuje mu dělat předpoklady o vzorech, které objeví, ověří a přijme nebo odmítne.

### Rozhodnutí o metodě trénování

V závislosti na vaší otázce a povaze vašich dat zvolíte metodu trénování. Procházením [dokumentace Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - kterou v tomto kurzu používáme - můžete prozkoumat mnoho způsobů, jak model trénovat. V závislosti na vašich zkušenostech možná budete muset vyzkoušet několik různých metod, abyste vytvořili nejlepší model. Pravděpodobně projdete procesem, kdy datoví vědci hodnotí výkon modelu tím, že mu předkládají neviděná data, kontrolují přesnost, předsudky a další problémy snižující kvalitu a vybírají nejvhodnější metodu trénování pro daný úkol.

### Trénování modelu

S trénovacími daty jste připraveni je „přizpůsobit“ k vytvoření modelu. Všimnete si, že v mnoha knihovnách ML najdete kód „model.fit“ - právě v tomto okamžiku zadáváte svou proměnnou vlastnosti jako pole hodnot (obvykle „X“) a cílovou proměnnou (obvykle „y“).

### Vyhodnocení modelu

Jakmile je proces trénování dokončen (u velkého modelu může trvat mnoho iterací, nebo „epoch“, než se vytrénuje), budete schopni vyhodnotit kvalitu modelu pomocí testovacích dat k posouzení jeho výkonu. Tato data jsou podmnožinou původních dat, která model dosud neanalyzoval. Můžete vytisknout tabulku metrik o kvalitě modelu.

🎓 **Přizpůsobení modelu**

V kontextu strojového učení přizpůsobení modelu odkazuje na přesnost základní funkce modelu, když se snaží analyzovat data, která nezná.

🎓 **Podtrénování** a **přetrénování** jsou běžné problémy, které snižují kvalitu modelu, protože model buď neodpovídá dostatečně dobře, nebo příliš dobře. To způsobuje, že model dělá predikce buď příliš úzce, nebo příliš volně ve vztahu k trénovacím datům. Přetrénovaný model predikuje trénovací data příliš dobře, protože se naučil detaily a šum dat příliš dobře. Podtrénovaný model není přesný, protože nedokáže přesně analyzovat ani trénovací data, ani data, která dosud „neviděl“.

![přetrénovaný model](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografika od [Jen Looper](https://twitter.com/jenlooper)

## Ladění parametrů

Jakmile je vaše počáteční trénování dokončeno, sledujte kvalitu modelu a zvažte jeho zlepšení úpravou jeho „hyperparametrů“. Přečtěte si více o tomto procesu [v dokumentaci](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predikce

Toto je okamžik, kdy můžete použít zcela nová data k otestování přesnosti vašeho modelu. V „aplikovaném“ nastavení ML, kde vytváříte webové nástroje pro použití modelu v produkci, může tento proces zahrnovat shromažďování uživatelských vstupů (například stisknutí tlačítka) k nastavení proměnné a jejímu odeslání modelu k inferenci nebo vyhodnocení.

V těchto lekcích objevíte, jak použít tyto kroky k přípravě, vytvoření, testování, vyhodnocení a predikci - všechny úkony datového vědce a další, jak postupujete na své cestě stát se „full stack“ inženýrem ML.

---

## 🚀Výzva

Nakreslete diagram toku, který odráží kroky praktikanta ML. Kde se právě teď vidíte v procesu? Kde předpokládáte, že narazíte na obtíže? Co se vám zdá snadné?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled a samostudium

Vyhledejte online rozhovory s datovými vědci, kteří diskutují o své každodenní práci. Zde je [jeden](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Úkol

[Rozhovor s datovým vědcem](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.