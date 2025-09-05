<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T00:30:14+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "cs"
}
-->
# Úvod do strojového učení

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pro začátečníky - Úvod do strojového učení pro začátečníky](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pro začátečníky - Úvod do strojového učení pro začátečníky")

> 🎥 Klikněte na obrázek výše pro krátké video k této lekci.

Vítejte v tomto kurzu klasického strojového učení pro začátečníky! Ať už jste v této oblasti úplně noví, nebo zkušený praktik hledající osvěžení znalostí, jsme rádi, že jste se k nám připojili! Chceme vytvořit přátelské místo pro zahájení vašeho studia strojového učení a rádi bychom zhodnotili, reagovali na a začlenili vaši [zpětnou vazbu](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Úvod do ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Úvod do ML")

> 🎥 Klikněte na obrázek výše pro video: John Guttag z MIT představuje strojové učení

---
## Začínáme se strojovým učením

Než začnete s tímto učebním plánem, je třeba mít svůj počítač připravený na lokální spuštění notebooků.

- **Nastavte svůj počítač pomocí těchto videí**. Použijte následující odkazy, abyste se naučili [jak nainstalovat Python](https://youtu.be/CXZYvNRIAKM) do svého systému a [nastavit textový editor](https://youtu.be/EU8eayHWoZg) pro vývoj.
- **Naučte se Python**. Doporučuje se také mít základní znalosti [Pythonu](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programovacího jazyka užitečného pro datové vědce, který v tomto kurzu používáme.
- **Naučte se Node.js a JavaScript**. V tomto kurzu také několikrát používáme JavaScript při vytváření webových aplikací, takže budete potřebovat mít nainstalovaný [node](https://nodejs.org) a [npm](https://www.npmjs.com/), stejně jako [Visual Studio Code](https://code.visualstudio.com/) dostupné pro vývoj v Pythonu i JavaScriptu.
- **Vytvořte si účet na GitHubu**. Protože jste nás našli zde na [GitHubu](https://github.com), možná už máte účet, ale pokud ne, vytvořte si ho a poté si tento učební plán forkněte pro vlastní použití. (Klidně nám dejte hvězdičku 😊)
- **Prozkoumejte Scikit-learn**. Seznamte se s [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), sadou knihoven pro strojové učení, na které se v těchto lekcích odkazujeme.

---
## Co je strojové učení?

Termín 'strojové učení' je jedním z nejpopulárnějších a nejčastěji používaných termínů dneška. Je pravděpodobné, že jste tento termín alespoň jednou slyšeli, pokud máte nějakou znalost technologie, bez ohledu na obor, ve kterém pracujete. Mechanika strojového učení je však pro většinu lidí záhadou. Pro začátečníka ve strojovém učení může být tento obor někdy ohromující. Proto je důležité pochopit, co strojové učení vlastně je, a učit se o něm krok za krokem, prostřednictvím praktických příkladů.

---
## Křivka popularity

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends ukazuje nedávnou 'křivku popularity' termínu 'strojové učení'

---
## Záhadný vesmír

Žijeme ve vesmíru plném fascinujících záhad. Velcí vědci jako Stephen Hawking, Albert Einstein a mnoho dalších zasvětili své životy hledání smysluplných informací, které odhalují tajemství světa kolem nás. To je lidská podstata učení: lidské dítě se učí nové věci a rok za rokem odhaluje strukturu svého světa, jak roste do dospělosti.

---
## Mozek dítěte

Mozek dítěte a jeho smysly vnímají fakta svého okolí a postupně se učí skryté vzory života, které dítěti pomáhají vytvářet logická pravidla pro identifikaci naučených vzorů. Proces učení lidského mozku činí člověka nejsofistikovanějším živým tvorem na tomto světě. Neustálé učení objevováním skrytých vzorů a následné inovace na těchto vzorech nám umožňují se během života stále zlepšovat. Tato schopnost učení a evoluce souvisí s konceptem zvaným [plasticita mozku](https://www.simplypsychology.org/brain-plasticity.html). Povrchně můžeme najít určité motivační podobnosti mezi procesem učení lidského mozku a koncepty strojového učení.

---
## Lidský mozek

[Lidský mozek](https://www.livescience.com/29365-human-brain.html) vnímá věci z reálného světa, zpracovává vnímané informace, činí racionální rozhodnutí a provádí určité akce na základě okolností. To nazýváme inteligentním chováním. Když naprogramujeme napodobeninu procesu inteligentního chování do stroje, nazývá se to umělá inteligence (AI).

---
## Některé pojmy

Ačkoli mohou být termíny zaměňovány, strojové učení (ML) je důležitou podmnožinou umělé inteligence. **ML se zabývá používáním specializovaných algoritmů k odhalování smysluplných informací a hledání skrytých vzorů z vnímaných dat, aby podpořilo proces racionálního rozhodování**.

---
## AI, ML, hluboké učení

![AI, ML, hluboké učení, datová věda](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Diagram ukazující vztahy mezi AI, ML, hlubokým učením a datovou vědou. Infografika od [Jen Looper](https://twitter.com/jenlooper) inspirovaná [tímto grafem](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepty, které pokryjeme

V tomto učebním plánu pokryjeme pouze základní koncepty strojového učení, které by měl začátečník znát. Zaměříme se na to, co nazýváme 'klasické strojové učení', především pomocí Scikit-learn, vynikající knihovny, kterou mnoho studentů používá k naučení základů. Pro pochopení širších konceptů umělé inteligence nebo hlubokého učení je nezbytné mít silné základní znalosti strojového učení, které bychom vám zde rádi nabídli.

---
## V tomto kurzu se naučíte:

- základní koncepty strojového učení
- historii ML
- ML a spravedlnost
- techniky regresního ML
- techniky klasifikačního ML
- techniky shlukovacího ML
- techniky zpracování přirozeného jazyka v ML
- techniky předpovědi časových řad v ML
- posilované učení
- reálné aplikace ML

---
## Co nebudeme pokrývat

- hluboké učení
- neuronové sítě
- AI

Pro lepší zážitek z učení se vyhneme složitostem neuronových sítí, 'hlubokého učení' - modelování s mnoha vrstvami pomocí neuronových sítí - a AI, které probereme v jiném učebním plánu. Také nabídneme připravovaný učební plán datové vědy, který se zaměří na tento aspekt širšího oboru.

---
## Proč studovat strojové učení?

Strojové učení je z pohledu systémů definováno jako tvorba automatizovaných systémů, které dokážou z dat učit skryté vzory, aby pomohly při inteligentním rozhodování.

Tato motivace je volně inspirována tím, jak lidský mozek učí určité věci na základě dat, která vnímá z okolního světa.

✅ Zamyslete se na chvíli, proč by firma chtěla použít strategie strojového učení místo vytvoření pevně zakódovaného systému založeného na pravidlech.

---
## Aplikace strojového učení

Aplikace strojového učení jsou nyní téměř všude a jsou stejně všudypřítomné jako data, která proudí kolem našich společností, generovaná našimi chytrými telefony, připojenými zařízeními a dalšími systémy. Vzhledem k obrovskému potenciálu nejmodernějších algoritmů strojového učení zkoumají vědci jejich schopnost řešit multidimenzionální a multidisciplinární problémy reálného života s velkými pozitivními výsledky.

---
## Příklady aplikovaného ML

**Strojové učení můžete využít mnoha způsoby**:

- K předpovědi pravděpodobnosti onemocnění na základě lékařské historie nebo zpráv pacienta.
- K využití meteorologických dat pro předpověď počasí.
- K pochopení sentimentu textu.
- K detekci falešných zpráv, aby se zabránilo šíření propagandy.

Finance, ekonomie, vědy o Zemi, průzkum vesmíru, biomedicínské inženýrství, kognitivní vědy a dokonce i obory v humanitních vědách adaptovaly strojové učení k řešení náročných problémů těžkých na zpracování dat ve svém oboru.

---
## Závěr

Strojové učení automatizuje proces objevování vzorů tím, že nachází smysluplné poznatky z reálných nebo generovaných dat. Ukázalo se, že je vysoce hodnotné v obchodních, zdravotních a finančních aplikacích, mimo jiné.

V blízké budoucnosti bude pochopení základů strojového učení nezbytné pro lidi z jakéhokoli oboru díky jeho širokému přijetí.

---
# 🚀 Výzva

Nakreslete na papír nebo pomocí online aplikace jako [Excalidraw](https://excalidraw.com/) své pochopení rozdílů mezi AI, ML, hlubokým učením a datovou vědou. Přidejte některé nápady na problémy, které jsou každá z těchto technik dobré při řešení.

# [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

---
# Přehled & Samostudium

Chcete-li se dozvědět více o tom, jak můžete pracovat s algoritmy ML v cloudu, sledujte tento [učební plán](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Absolvujte [učební plán](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o základech ML.

---
# Zadání

[Začněte](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.