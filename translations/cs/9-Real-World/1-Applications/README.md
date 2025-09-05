<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T00:10:07+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "cs"
}
-->
# Dodatek: Strojové učení v reálném světě

![Shrnutí strojového učení v reálném světě ve sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

V tomto kurzu jste se naučili mnoho způsobů, jak připravit data pro trénink a vytvořit modely strojového učení. Postavili jste sérii klasických modelů pro regresi, shlukování, klasifikaci, zpracování přirozeného jazyka a časové řady. Gratulujeme! Možná si teď kladete otázku, k čemu to všechno je... jaké jsou reálné aplikace těchto modelů?

I když v průmyslu vzbuzuje velký zájem AI, která obvykle využívá hluboké učení, stále existují cenné aplikace pro klasické modely strojového učení. Možná některé z těchto aplikací používáte už dnes! V této lekci prozkoumáte, jak osm různých odvětví a oborů využívá tyto typy modelů ke zlepšení výkonu, spolehlivosti, inteligence a hodnoty svých aplikací pro uživatele.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finance

Finanční sektor nabízí mnoho příležitostí pro strojové učení. Mnoho problémů v této oblasti lze modelovat a řešit pomocí ML.

### Detekce podvodů s kreditními kartami

V průběhu kurzu jsme se naučili o [k-means shlukování](../../5-Clustering/2-K-Means/README.md), ale jak může být použito k řešení problémů souvisejících s podvody s kreditními kartami?

K-means shlukování je užitečné při technice detekce podvodů s kreditními kartami nazývané **detekce odlehlých hodnot**. Odlehlé hodnoty, nebo odchylky v pozorováních o sadě dat, nám mohou říci, zda je kreditní karta používána normálně, nebo zda se děje něco neobvyklého. Jak je ukázáno v níže uvedeném článku, můžete data o kreditních kartách třídit pomocí algoritmu k-means shlukování a přiřadit každou transakci ke shluku na základě toho, jak moc se jeví jako odlehlá. Poté můžete vyhodnotit nejrizikovější shluky z hlediska podvodných a legitimních transakcí.
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Správa majetku

Ve správě majetku jednotlivec nebo firma spravuje investice jménem svých klientů. Jejich úkolem je dlouhodobě udržovat a zvyšovat bohatství, takže je zásadní vybírat investice, které dobře fungují.

Jedním ze způsobů, jak hodnotit výkon konkrétní investice, je statistická regrese. [Lineární regrese](../../2-Regression/1-Tools/README.md) je cenný nástroj pro pochopení toho, jak fond funguje ve vztahu k určitému benchmarku. Můžeme také zjistit, zda jsou výsledky regrese statisticky významné, nebo jak moc by ovlivnily investice klienta. Analýzu můžete dále rozšířit pomocí vícenásobné regrese, kde lze zohlednit další rizikové faktory. Příklad toho, jak by to fungovalo pro konkrétní fond, najdete v níže uvedeném článku o hodnocení výkonu fondu pomocí regrese.
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Vzdělávání

Vzdělávací sektor je také velmi zajímavou oblastí, kde lze ML aplikovat. Existují zajímavé problémy, které je třeba řešit, jako je detekce podvádění při testech nebo esejích, nebo řízení zaujatosti, ať už úmyslné nebo neúmyslné, v procesu hodnocení.

### Predikce chování studentů

[Coursera](https://coursera.com), poskytovatel online kurzů, má skvělý technický blog, kde diskutují o mnoha inženýrských rozhodnutích. V této případové studii vykreslili regresní linii, aby prozkoumali jakoukoli korelaci mezi nízkým hodnocením NPS (Net Promoter Score) a udržením nebo odchodem z kurzu.
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Řízení zaujatosti

[Grammarly](https://grammarly.com), asistent pro psaní, který kontroluje pravopis a gramatické chyby, používá sofistikované [systémy zpracování přirozeného jazyka](../../6-NLP/README.md) ve svých produktech. Na svém technickém blogu zveřejnili zajímavou případovou studii o tom, jak se vypořádali s genderovou zaujatostí ve strojovém učení, o které jste se dozvěděli v naší [úvodní lekci o spravedlnosti](../../1-Introduction/3-fairness/README.md).
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Maloobchod

Maloobchodní sektor může rozhodně těžit z využití ML, a to od vytváření lepší zákaznické cesty až po optimální skladování zásob.

### Personalizace zákaznické cesty

Ve společnosti Wayfair, která prodává domácí potřeby jako nábytek, je klíčové pomoci zákazníkům najít správné produkty podle jejich vkusu a potřeb. V tomto článku inženýři společnosti popisují, jak využívají ML a NLP k "zobrazení správných výsledků pro zákazníky". Jejich Query Intent Engine byl navržen tak, aby využíval extrakci entit, trénink klasifikátorů, extrakci aktiv a názorů a označování sentimentu v zákaznických recenzích. Toto je klasický příklad toho, jak NLP funguje v online maloobchodu.
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Řízení zásob

Inovativní, flexibilní společnosti jako [StitchFix](https://stitchfix.com), služba zasílání krabic s oblečením, se silně spoléhají na ML pro doporučení a řízení zásob. Jejich stylingové týmy spolupracují s týmy pro merchandising: "jeden z našich datových vědců experimentoval s genetickým algoritmem a aplikoval jej na oděvy, aby předpověděl, co by mohlo být úspěšným kusem oblečení, který dnes neexistuje. Předložili jsme to týmu pro merchandising a nyní to mohou používat jako nástroj."
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Zdravotnictví

Zdravotnický sektor může využít ML k optimalizaci výzkumných úkolů a také logistických problémů, jako je opětovné přijímání pacientů nebo zastavení šíření nemocí.

### Řízení klinických studií

Toxicita v klinických studiích je hlavním problémem pro výrobce léků. Kolik toxicity je tolerovatelné? V této studii analýza různých metod klinických studií vedla k vývoji nového přístupu pro predikci výsledků klinických studií. Konkrétně byli schopni použít random forest k vytvoření [klasifikátoru](../../4-Classification/README.md), který dokáže rozlišit mezi skupinami léků.
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Řízení opětovného přijímání do nemocnic

Nemocniční péče je nákladná, zejména když musí být pacienti znovu přijati. Tento článek popisuje společnost, která používá ML k predikci potenciálu opětovného přijetí pomocí [shlukovacích algoritmů](../../5-Clustering/README.md). Tyto shluky pomáhají analytikům "objevovat skupiny opětovných přijetí, které mohou sdílet společnou příčinu".
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Řízení nemocí

Nedávná pandemie osvětlila způsoby, jakými může strojové učení pomoci zastavit šíření nemocí. V tomto článku poznáte použití ARIMA, logistických křivek, lineární regrese a SARIMA. "Tato práce je pokusem vypočítat míru šíření tohoto viru a tím předpovědět úmrtí, uzdravení a potvrzené případy, aby nám to mohlo pomoci lépe se připravit a přežít."
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologie a zelené technologie

Příroda a ekologie zahrnují mnoho citlivých systémů, kde se do popředí dostává interakce mezi zvířaty a přírodou. Je důležité být schopen tyto systémy přesně měřit a jednat vhodně, pokud se něco stane, například lesní požár nebo pokles populace zvířat.

### Řízení lesů

V předchozích lekcích jste se naučili o [Reinforcement Learning](../../8-Reinforcement/README.md). Může být velmi užitečné při pokusech o predikci vzorců v přírodě. Zejména může být použito ke sledování ekologických problémů, jako jsou lesní požáry a šíření invazivních druhů. V Kanadě skupina výzkumníků použila Reinforcement Learning k vytvoření modelů dynamiky lesních požárů ze satelitních snímků. Pomocí inovativního "procesu prostorového šíření (SSP)" si představili lesní požár jako "agenta na jakékoli buňce v krajině". "Sada akcí, které může požár podniknout z určitého místa v daném čase, zahrnuje šíření na sever, jih, východ nebo západ nebo nešíření."

Tento přístup obrací obvyklé nastavení RL, protože dynamika odpovídajícího Markovova rozhodovacího procesu (MDP) je známou funkcí pro okamžité šíření požáru. Více o klasických algoritmech používaných touto skupinou si přečtěte na níže uvedeném odkazu.
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Sledování pohybu zvířat

I když hluboké učení způsobilo revoluci ve vizuálním sledování pohybu zvířat (můžete si vytvořit vlastní [sledovač ledních medvědů](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) zde), klasické ML má stále své místo v tomto úkolu.

Senzory pro sledování pohybu hospodářských zvířat a IoT využívají tento typ vizuálního zpracování, ale základnější techniky ML jsou užitečné pro předzpracování dat. Například v tomto článku byly monitorovány a analyzovány postoje ovcí pomocí různých klasifikačních algoritmů. Možná poznáte ROC křivku na straně 335.
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Řízení energie

V našich lekcích o [predikci časových řad](../../7-TimeSeries/README.md) jsme zmínili koncept chytrých parkovacích měřičů pro generování příjmů pro město na základě pochopení nabídky a poptávky. Tento článek podrobně popisuje, jak kombinace shlukování, regrese a predikce časových řad pomohla předpovědět budoucí spotřebu energie v Irsku na základě chytrého měření.
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Pojišťovnictví

Pojišťovnictví je další sektor, který využívá ML k vytváření a optimalizaci životaschopných finančních a pojistně-matematických modelů.

### Řízení volatility

MetLife, poskytovatel životního pojištění, otevřeně popisuje, jak analyzuje a zmírňuje volatilitu ve svých finančních modelech. V tomto článku si všimnete vizualizací binární a ordinální klasifikace. Také objevíte vizualizace predikcí.
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Umění, kultura a literatura

V umění, například v žurnalistice, existuje mnoho zajímavých problémů. Detekce falešných zpráv je obrovský problém, protože bylo prokázáno, že ovlivňuje názory lidí a dokonce může destabilizovat demokracie. Muzea mohou také těžit z využití ML ve všem od hledání spojení mezi artefakty až po plánování zdrojů.

### Detekce falešných zpráv

Detekce falešných zpráv se v dnešních médiích stala hrou na kočku a myš. V tomto článku výzkumníci navrhují, že systém kombinující několik technik ML, které jsme studovali, může být testován a nejlepší model nasazen: "Tento systém je založen na zpracování přirozeného jazyka pro extrakci funkcí z dat a poté jsou tyto funkce použity pro trénink klasifikátorů strojového učení, jako jsou Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) a Logistic Regression (LR)."
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Tento článek ukazuje, jak kombinace různých oblastí ML může přinést zajímavé výsledky, které mohou pomoci zastavit šíření falešných zpráv a zabránit skutečným škodám; v tomto případě byl impulsem šíření fám o léčbě COVID, které podněcovaly násilí davu.

### Muzejní ML

Muzea stojí na prahu revoluce AI, kdy katalogizace a digitalizace sbírek a hledání spojení mezi artefakty je díky technologickému pokroku stále snazší. Projekty jako [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) pomáhají odhalovat tajemství nepřístupných sbírek, jako jsou Vatikánské archivy. Ale obchodní aspekt muzeí také těží z modelů ML.

Například Art Institute of Chicago vytvořil modely pro předpověď, co návštěvníky zajímá a kdy navštíví expozice. Cílem je vytvořit individualizované a optimalizované zážitky pro návštěvníky při každé jejich návštěvě muzea. "Během fiskálního roku 2017 model předpověděl návštěvnost a příjmy s přesností na 1 procento, říká Andrew Simnick, senior viceprezident v Art Institute."
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentace zákazníků

Nejúčinnější marketingové strategie cílí na zákazníky různými způsoby na základě různých skupin. V tomto článku jsou diskutovány využití shlukovacích algoritmů na podporu diferencovaného marketingu. Diferencovaný marketing pomáhá společnostem zlepšit povědomí o značce, oslovit více zákazníků a vydělat více peněz.
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Výzva

Identifikujte další sektor, který těží z některých technik, které jste se naučili v tomto kurzu, a zjistěte, jak využívá ML.
## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Tým datové vědy společnosti Wayfair má několik zajímavých videí o tom, jak využívají strojové učení ve své firmě. Stojí za to [podívat se](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Úkol

[Hledání pokladu s ML](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby AI pro překlady [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatizované překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.