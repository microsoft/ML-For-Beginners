<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T15:51:01+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "sk"
}
-->
# Postscript: Strojové učenie v reálnom svete

![Zhrnutie strojového učenia v reálnom svete v sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

V tomto kurze ste sa naučili mnoho spôsobov, ako pripraviť dáta na tréning a vytvárať modely strojového učenia. Vytvorili ste sériu klasických modelov regresie, zhlukovania, klasifikácie, spracovania prirodzeného jazyka a časových radov. Gratulujeme! Teraz sa možno pýtate, na čo to všetko je... aké sú reálne aplikácie týchto modelov?

Aj keď veľký záujem v priemysle vzbudzuje AI, ktorá zvyčajne využíva hlboké učenie, stále existujú hodnotné aplikácie pre klasické modely strojového učenia. Niektoré z týchto aplikácií možno používate už dnes! V tejto lekcii preskúmate, ako osem rôznych odvetví a odborných oblastí využíva tieto typy modelov na zlepšenie výkonu, spoľahlivosti, inteligencie a hodnoty pre používateľov.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Financie

Finančný sektor ponúka mnoho príležitostí pre strojové učenie. Mnohé problémy v tejto oblasti sa dajú modelovať a riešiť pomocou ML.

### Detekcia podvodov s kreditnými kartami

V priebehu kurzu sme sa naučili o [k-means zhlukovaní](../../5-Clustering/2-K-Means/README.md), ale ako ho možno použiť na riešenie problémov súvisiacich s podvodmi s kreditnými kartami?

K-means zhlukovanie je užitočné pri technike detekcie podvodov s kreditnými kartami nazývanej **detekcia odľahlých hodnôt**. Odľahlé hodnoty, alebo odchýlky v pozorovaniach o súbore dát, nám môžu povedať, či je kreditná karta používaná normálne alebo či sa deje niečo neobvyklé. Ako je uvedené v nižšie uvedenom článku, môžete triediť dáta o kreditných kartách pomocou algoritmu k-means zhlukovania a priradiť každú transakciu do zhluku na základe toho, ako veľmi sa javí ako odľahlá hodnota. Potom môžete vyhodnotiť najrizikovejšie zhluky na podvodné verzus legitímne transakcie.
[Referencie](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Správa majetku

V správe majetku jednotlivec alebo firma spravuje investície v mene svojich klientov. Ich úlohou je dlhodobo udržiavať a zvyšovať bohatstvo, takže je nevyhnutné vybrať investície, ktoré budú dobre fungovať.

Jedným zo spôsobov, ako vyhodnotiť, ako konkrétna investícia funguje, je štatistická regresia. [Lineárna regresia](../../2-Regression/1-Tools/README.md) je cenný nástroj na pochopenie toho, ako fond funguje v porovnaní s nejakým benchmarkom. Môžeme tiež zistiť, či sú výsledky regresie štatisticky významné, alebo ako veľmi by ovplyvnili investície klienta. Analýzu môžete ešte rozšíriť pomocou viacnásobnej regresie, kde sa môžu zohľadniť ďalšie rizikové faktory. Pre príklad, ako by to fungovalo pre konkrétny fond, si pozrite nižšie uvedený článok o hodnotení výkonnosti fondu pomocou regresie.
[Referencie](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Vzdelávanie

Vzdelávací sektor je tiež veľmi zaujímavou oblasťou, kde sa dá aplikovať ML. Existujú zaujímavé problémy, ktoré je možné riešiť, ako napríklad detekcia podvádzania na testoch alebo esejách, alebo zvládanie zaujatosti, či už neúmyselnej alebo nie, v procese hodnotenia.

### Predpovedanie správania študentov

[Coursera](https://coursera.com), poskytovateľ online kurzov, má skvelý technologický blog, kde diskutujú o mnohých inžinierskych rozhodnutiach. V tejto prípadovej štúdii vykreslili regresnú líniu, aby preskúmali akúkoľvek koreláciu medzi nízkym NPS (Net Promoter Score) hodnotením a udržaním alebo odchodom z kurzu.
[Referencie](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Zmierňovanie zaujatosti

[Grammarly](https://grammarly.com), asistent písania, ktorý kontroluje pravopisné a gramatické chyby, používa sofistikované [systémy spracovania prirodzeného jazyka](../../6-NLP/README.md) vo svojich produktoch. Na svojom technologickom blogu publikovali zaujímavú prípadovú štúdiu o tom, ako sa vysporiadali s rodovou zaujatostou v strojovom učení, o ktorej ste sa učili v našej [úvodnej lekcii o spravodlivosti](../../1-Introduction/3-fairness/README.md).
[Referencie](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Maloobchod

Maloobchodný sektor môže určite profitovať z využitia ML, od vytvárania lepšej zákazníckej cesty až po optimálne skladovanie zásob.

### Personalizácia zákazníckej cesty

V spoločnosti Wayfair, ktorá predáva domáce potreby ako nábytok, je pomoc zákazníkom pri hľadaní správnych produktov pre ich vkus a potreby kľúčová. V tomto článku inžinieri zo spoločnosti popisujú, ako používajú ML a NLP na "zobrazenie správnych výsledkov pre zákazníkov". Ich Query Intent Engine bol postavený na využití extrakcie entít, tréningu klasifikátorov, extrakcie aktív a názorov a označovania sentimentu v zákazníckych recenziách. Toto je klasický príklad toho, ako NLP funguje v online maloobchode.
[Referencie](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Správa zásob

Inovatívne, flexibilné spoločnosti ako [StitchFix](https://stitchfix.com), služba boxov, ktorá posiela oblečenie spotrebiteľom, sa silno spoliehajú na ML pri odporúčaniach a správe zásob. Ich stylingové tímy spolupracujú s tímami pre merchandising: "jeden z našich dátových vedcov experimentoval s genetickým algoritmom a aplikoval ho na oblečenie, aby predpovedal, čo by mohlo byť úspešným kúskom oblečenia, ktorý dnes neexistuje. Predložili sme to tímu pre merchandising a teraz to môžu používať ako nástroj."
[Referencie](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Zdravotná starostlivosť

Sektor zdravotnej starostlivosti môže využívať ML na optimalizáciu výskumných úloh a logistických problémov, ako je opätovné prijímanie pacientov alebo zastavenie šírenia chorôb.

### Správa klinických štúdií

Toxicita v klinických štúdiách je veľkým problémom pre výrobcov liekov. Koľko toxicity je tolerovateľné? V tejto štúdii analýza rôznych metód klinických štúdií viedla k vývoju nového prístupu na predpovedanie pravdepodobnosti výsledkov klinických štúdií. Konkrétne boli schopní použiť random forest na vytvorenie [klasifikátora](../../4-Classification/README.md), ktorý dokáže rozlíšiť medzi skupinami liekov.
[Referencie](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Správa opätovného prijímania pacientov

Nemocničná starostlivosť je nákladná, najmä keď musia byť pacienti opätovne prijatí. Tento článok diskutuje o spoločnosti, ktorá používa ML na predpovedanie potenciálu opätovného prijímania pomocou [zhlukovania](../../5-Clustering/README.md) algoritmov. Tieto zhluky pomáhajú analytikom "objaviť skupiny opätovných prijatí, ktoré môžu mať spoločnú príčinu".
[Referencie](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Správa chorôb

Nedávna pandémia osvetlila spôsoby, akými môže strojové učenie pomôcť zastaviť šírenie chorôb. V tomto článku rozpoznáte použitie ARIMA, logistických kriviek, lineárnej regresie a SARIMA. "Táto práca je pokusom vypočítať mieru šírenia tohto vírusu a tým predpovedať úmrtia, zotavenia a potvrdené prípady, aby nám pomohla lepšie sa pripraviť a prežiť."
[Referencie](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekológia a zelené technológie

Príroda a ekológia pozostávajú z mnohých citlivých systémov, kde sa do popredia dostáva interakcia medzi zvieratami a prírodou. Je dôležité byť schopný presne merať tieto systémy a primerane konať, ak sa niečo stane, ako napríklad lesný požiar alebo pokles populácie zvierat.

### Správa lesov

V predchádzajúcich lekciách ste sa naučili o [Reinforcement Learning](../../8-Reinforcement/README.md). Môže byť veľmi užitočné pri predpovedaní vzorcov v prírode. Najmä môže byť použité na sledovanie ekologických problémov, ako sú lesné požiare a šírenie invazívnych druhov. V Kanade skupina výskumníkov použila Reinforcement Learning na vytvorenie modelov dynamiky lesných požiarov zo satelitných snímok. Použitím inovatívneho "procesu šírenia v priestore (SSP)" si predstavili lesný požiar ako "agenta na akejkoľvek bunke v krajine." "Súbor akcií, ktoré môže požiar vykonať z miesta v akomkoľvek čase, zahŕňa šírenie na sever, juh, východ alebo západ alebo nešírenie.

Tento prístup invertuje obvyklé nastavenie RL, pretože dynamika zodpovedajúceho Markov Decision Process (MDP) je známa funkcia pre okamžité šírenie požiaru." Viac o klasických algoritmoch použité touto skupinou si prečítajte na nižšie uvedenom odkaze.
[Referencie](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Sledovanie pohybu zvierat

Aj keď hlboké učenie vytvorilo revolúciu vo vizuálnom sledovaní pohybov zvierat (môžete si vytvoriť vlastný [sledovač polárnych medveďov](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) tu), klasické ML má stále svoje miesto v tejto úlohe.

Senzory na sledovanie pohybov hospodárskych zvierat a IoT využívajú tento typ vizuálneho spracovania, ale základnejšie techniky ML sú užitočné na predspracovanie dát. Napríklad v tomto článku boli monitorované a analyzované postoje oviec pomocou rôznych klasifikačných algoritmov. Na strane 335 môžete rozpoznať ROC krivku.
[Referencie](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Správa energie

V našich lekciách o [predpovedaní časových radov](../../7-TimeSeries/README.md) sme spomenuli koncept inteligentných parkovacích meračov na generovanie príjmov pre mesto na základe pochopenia ponuky a dopytu. Tento článok podrobne diskutuje, ako kombinácia zhlukovania, regresie a predpovedania časových radov pomohla predpovedať budúce využitie energie v Írsku na základe inteligentného merania.
[Referencie](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Poistenie

Sektor poistenia je ďalším sektorom, ktorý využíva ML na konštrukciu a optimalizáciu životaschopných finančných a aktuárskych modelov.

### Správa volatility

MetLife, poskytovateľ životného poistenia, je otvorený vo svojom prístupe k analýze a zmierňovaniu volatility vo svojich finančných modeloch. V tomto článku si všimnete vizualizácie binárnej a ordinálnej klasifikácie. Objavíte tiež vizualizácie predpovedí.
[Referencie](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Umenie, kultúra a literatúra

V umení, napríklad v žurnalistike, existuje mnoho zaujímavých problémov. Detekcia falošných správ je obrovským problémom, pretože sa ukázalo, že ovplyvňuje názory ľudí a dokonca môže ohroziť demokracie. Múzeá môžu tiež profitovať z využitia ML vo všetkom od hľadania spojení medzi artefaktmi až po plánovanie zdrojov.

### Detekcia falošných správ

Detekcia falošných správ sa stala hrou mačky a myši v dnešných médiách. V tomto článku výskumníci navrhujú systém kombinujúci niekoľko techník ML, ktoré sme študovali, a testovanie najlepšieho modelu: "Tento systém je založený na spracovaní prirodzeného jazyka na extrakciu vlastností z dát a potom sú tieto vlastnosti použité na tréning klasifikátorov strojového učenia, ako sú Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) a Logistic Regression (LR)."
[Referencie](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Tento článok ukazuje, ako kombinácia rôznych oblastí ML môže priniesť zaujímavé výsledky, ktoré môžu pomôcť zastaviť šírenie falošných správ a vytvárať skutočné škody; v tomto prípade bol impulzom šírenie fám o liečbe COVID, ktoré podnietilo násilné správanie davu.

### Múzeá a ML

Múzeá sú na prahu AI revolúcie, v ktorej sa katalogizácia a digitalizácia zbierok a hľadanie spojení medzi artefaktmi stáva jednoduchším, ako technológia napreduje. Projekty ako [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) pomáhajú odomknúť tajomstvá neprístupných zbierok, ako sú Vatikánske archívy. Ale obchodný aspekt múzeí tiež profituje z modelov ML.

Napríklad Art Institute of Chicago vytvoril modely na predpovedanie toho, čo publikum zaujíma a kedy navštívi expozície. Cieľom je vytvoriť individuálne a optimalizované zážitky návštevníkov pri každej návšteve múzea. "Počas fiškálneho roku 2017 model predpovedal návštevnosť a príjmy s presnosťou do 1 percenta, hovorí Andrew Simnick, senior viceprezident v Art Institute."
[Referencie](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentácia zákazníkov

Najefektívnejšie marketingové stratégie oslovujú zákazníkov rôznymi spôsobmi na základe rôznych skupín. V tomto článku sa diskutuje o využití algoritmov zhlukovania na podporu diferencovaného marketingu. Diferencovaný marketing pomáha spoločnost
## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Tím dátovej vedy spoločnosti Wayfair má niekoľko zaujímavých videí o tom, ako využívajú strojové učenie vo svojej firme. Stojí za to [pozrieť si ich](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Zadanie

[Lov na strojové učenie](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.