<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T12:25:28+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "sl"
}
-->
# Postscript: Strojno učenje v resničnem svetu

![Povzetek strojnega učenja v resničnem svetu v sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote avtorja [Tomomi Imura](https://www.twitter.com/girlie_mac)

V tem učnem načrtu ste se naučili številnih načinov za pripravo podatkov za učenje in ustvarjanje modelov strojnega učenja. Zgradili ste serijo klasičnih modelov za regresijo, razvrščanje, klasifikacijo, obdelavo naravnega jezika in časovne vrste. Čestitke! Zdaj se morda sprašujete, za kaj vse to služi... kakšne so resnične aplikacije teh modelov?

Čeprav je industrija zelo zainteresirana za umetno inteligenco, ki običajno temelji na globokem učenju, obstajajo še vedno dragocene aplikacije za klasične modele strojnega učenja. Nekatere od teh aplikacij morda že uporabljate danes! V tej lekciji boste raziskali, kako osem različnih industrij in področij uporablja te vrste modelov za izboljšanje zmogljivosti, zanesljivosti, inteligence in vrednosti svojih aplikacij za uporabnike.

## [Predhodni kviz pred predavanjem](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finance

Finančni sektor ponuja številne priložnosti za strojno učenje. Veliko težav na tem področju je mogoče modelirati in rešiti z uporabo strojnega učenja.

### Odkrivanje goljufij s kreditnimi karticami

V tečaju smo se naučili o [k-means razvrščanju](../../5-Clustering/2-K-Means/README.md), vendar kako ga lahko uporabimo za reševanje težav, povezanih z goljufijami s kreditnimi karticami?

K-means razvrščanje je uporabno pri tehniki odkrivanja goljufij s kreditnimi karticami, imenovani **odkrivanje odstopanj**. Odstopanja ali nepravilnosti v opazovanjih podatkovnega nabora nam lahko povedo, ali se kreditna kartica uporablja na običajen način ali pa se dogaja nekaj nenavadnega. Kot je prikazano v spodnjem članku, lahko podatke o kreditnih karticah razvrstimo z algoritmom k-means in vsako transakcijo dodelimo skupini glede na to, kako močno odstopa. Nato lahko ocenimo najbolj tvegane skupine za goljufive ali legitimne transakcije.
[Referenca](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Upravljanje premoženja

Pri upravljanju premoženja posameznik ali podjetje upravlja naložbe v imenu svojih strank. Njihova naloga je dolgoročno ohranjati in povečevati premoženje, zato je ključnega pomena izbrati naložbe, ki dobro delujejo.

Eden od načinov za oceno uspešnosti določene naložbe je statistična regresija. [Linearna regresija](../../2-Regression/1-Tools/README.md) je dragoceno orodje za razumevanje, kako sklad deluje v primerjavi z nekim merilom. Prav tako lahko ugotovimo, ali so rezultati regresije statistično pomembni oziroma kako močno bi vplivali na naložbe stranke. Analizo lahko še razširite z večkratno regresijo, kjer lahko upoštevate dodatne dejavnike tveganja. Za primer, kako bi to delovalo za določen sklad, si oglejte spodnji članek o ocenjevanju uspešnosti sklada z uporabo regresije.
[Referenca](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Izobraževanje

Izobraževalni sektor je prav tako zelo zanimivo področje, kjer se lahko uporabi strojno učenje. Obstajajo zanimive težave, ki jih je mogoče rešiti, kot so odkrivanje goljufij pri testih ali esejih ter obvladovanje pristranskosti, namerne ali nenamerne, v procesu ocenjevanja.

### Napovedovanje vedenja študentov

[Coursera](https://coursera.com), ponudnik spletnih odprtih tečajev, ima odličen tehnični blog, kjer razpravljajo o številnih inženirskih odločitvah. V tej študiji primera so narisali regresijsko črto, da bi raziskali morebitno korelacijo med nizko oceno NPS (Net Promoter Score) in ohranjanjem ali opuščanjem tečaja.
[Referenca](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Zmanjševanje pristranskosti

[Grammarly](https://grammarly.com), pisalni asistent, ki preverja črkovanje in slovnične napake, uporablja sofisticirane [sisteme za obdelavo naravnega jezika](../../6-NLP/README.md) v svojih izdelkih. Na svojem tehničnem blogu so objavili zanimivo študijo primera o tem, kako so se spopadli s spolno pristranskostjo v strojnem učenju, o čemer ste se učili v naši [uvodni lekciji o pravičnosti](../../1-Introduction/3-fairness/README.md).
[Referenca](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Trgovina na drobno

Trgovinski sektor lahko zagotovo koristi uporabo strojnega učenja, od ustvarjanja boljše uporabniške izkušnje do optimalnega upravljanja zalog.

### Personalizacija uporabniške izkušnje

Pri Wayfair, podjetju, ki prodaja izdelke za dom, kot je pohištvo, je ključnega pomena pomagati strankam najti prave izdelke za njihov okus in potrebe. V tem članku inženirji podjetja opisujejo, kako uporabljajo strojno učenje in NLP za "prikazovanje pravih rezultatov za stranke". Njihov Query Intent Engine je bil zasnovan za uporabo ekstrakcije entitet, usposabljanje klasifikatorjev, ekstrakcijo mnenj in označevanje sentimenta v pregledih strank. To je klasičen primer uporabe NLP v spletni trgovini.
[Referenca](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Upravljanje zalog

Inovativna, agilna podjetja, kot je [StitchFix](https://stitchfix.com), storitev, ki pošilja oblačila potrošnikom, se močno zanašajo na strojno učenje za priporočila in upravljanje zalog. Njihove ekipe za stiliranje sodelujejo z ekipami za trgovanje: "eden od naših podatkovnih znanstvenikov je eksperimentiral z genetskim algoritmom in ga uporabil na oblačilih za napovedovanje, kaj bi bil uspešen kos oblačila, ki danes še ne obstaja. To smo predstavili ekipi za trgovanje, ki zdaj lahko to uporablja kot orodje."
[Referenca](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Zdravstvo

Zdravstveni sektor lahko izkoristi strojno učenje za optimizacijo raziskovalnih nalog in logističnih težav, kot so ponovne hospitalizacije ali preprečevanje širjenja bolezni.

### Upravljanje kliničnih študij

Toksičnost v kliničnih študijah je velika skrb za proizvajalce zdravil. Koliko toksičnosti je sprejemljivo? V tej študiji je analiza različnih metod kliničnih študij privedla do razvoja novega pristopa za napovedovanje verjetnosti izidov kliničnih študij. Konkretno, uporabili so random forest za izdelavo [klasifikatorja](../../4-Classification/README.md), ki lahko razlikuje med skupinami zdravil.
[Referenca](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Upravljanje ponovnih hospitalizacij

Bolnišnično zdravljenje je drago, zlasti ko je treba paciente ponovno hospitalizirati. Ta članek obravnava podjetje, ki uporablja strojno učenje za napovedovanje potenciala ponovne hospitalizacije z uporabo [razvrščanja](../../5-Clustering/README.md) algoritmov. Te skupine pomagajo analitikom "odkriti skupine ponovnih hospitalizacij, ki lahko delijo skupni vzrok".
[Referenca](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Upravljanje bolezni

Nedavna pandemija je osvetlila načine, kako lahko strojno učenje pomaga pri zaustavitvi širjenja bolezni. V tem članku boste prepoznali uporabo ARIMA, logističnih krivulj, linearne regresije in SARIMA. "To delo je poskus izračunati stopnjo širjenja tega virusa in tako napovedati smrti, okrevanja in potrjene primere, da bi se lahko bolje pripravili in preživeli."
[Referenca](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologija in zelena tehnologija

Narava in ekologija vključujeta številne občutljive sisteme, kjer je pomembno natančno meriti te sisteme in ustrezno ukrepati, če se zgodi kaj nenavadnega, kot je gozdni požar ali upad populacije živali.

### Upravljanje gozdov

V prejšnjih lekcijah ste se naučili o [okrepitvenem učenju](../../8-Reinforcement/README.md). To je lahko zelo uporabno pri napovedovanju vzorcev v naravi. Zlasti se lahko uporablja za sledenje ekološkim težavam, kot so gozdni požari in širjenje invazivnih vrst. V Kanadi je skupina raziskovalcev uporabila okrepitveno učenje za izdelavo modelov dinamike gozdnih požarov iz satelitskih posnetkov. Z inovativnim "procesom prostorskega širjenja (SSP)" so si zamislili gozdni požar kot "agenta na katerem koli mestu v pokrajini." "Nabor dejanj, ki jih lahko požar izvede z določene lokacije v katerem koli trenutku, vključuje širjenje na sever, jug, vzhod ali zahod ali ne širjenje."

Ta pristop obrne običajno nastavitev okrepitvenega učenja, saj je dinamika ustreznega Markovskega odločitvenega procesa (MDP) znana funkcija za takojšnje širjenje požara. Več o klasičnih algoritmih, ki jih je uporabila ta skupina, si preberite na spodnji povezavi.
[Referenca](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Sledenje gibanju živali

Čeprav je globoko učenje povzročilo revolucijo pri vizualnem sledenju gibanja živali (tukaj lahko zgradite svojega [sledilca polarnih medvedov](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott)), ima klasično strojno učenje še vedno svoje mesto pri tej nalogi.

Senzorji za sledenje gibanju kmetijskih živali in IoT uporabljajo tovrstno vizualno obdelavo, vendar so osnovnejše tehnike strojnega učenja uporabne za predhodno obdelavo podatkov. Na primer, v tem članku so bile drže ovc spremljane in analizirane z različnimi algoritmi klasifikatorjev. Na strani 335 boste morda prepoznali krivuljo ROC.
[Referenca](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Upravljanje energije

V naših lekcijah o [napovedovanju časovnih vrst](../../7-TimeSeries/README.md) smo uvedli koncept pametnih parkirnih števcev za ustvarjanje prihodkov za mesto na podlagi razumevanja ponudbe in povpraševanja. Ta članek podrobno obravnava, kako so razvrščanje, regresija in napovedovanje časovnih vrst združeni za pomoč pri napovedovanju prihodnje porabe energije na Irskem, na podlagi pametnega merjenja.
[Referenca](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Zavarovalništvo

Zavarovalniški sektor je še eno področje, ki uporablja strojno učenje za gradnjo in optimizacijo izvedljivih finančnih in aktuarskih modelov.

### Upravljanje volatilnosti

MetLife, ponudnik življenjskih zavarovanj, je odprt glede načina, kako analizira in zmanjšuje volatilnost v svojih finančnih modelih. V tem članku boste opazili vizualizacije binarne in ordinalne klasifikacije. Prav tako boste odkrili vizualizacije napovedovanja.
[Referenca](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Umetnost, kultura in literatura

Na področju umetnosti, na primer v novinarstvu, obstajajo številne zanimive težave. Odkrivanje lažnih novic je velik problem, saj je dokazano, da vpliva na mnenje ljudi in celo ogroža demokracije. Muzeji lahko prav tako koristijo uporabo strojnega učenja pri vsem, od iskanja povezav med artefakti do načrtovanja virov.

### Odkrivanje lažnih novic

Odkrivanje lažnih novic je postalo igra mačke in miši v današnjih medijih. V tem članku raziskovalci predlagajo sistem, ki združuje več tehnik strojnega učenja, ki smo jih preučevali, in testiranje najboljšega modela: "Ta sistem temelji na obdelavi naravnega jezika za ekstrakcijo značilnosti iz podatkov, nato pa se te značilnosti uporabijo za usposabljanje klasifikatorjev strojnega učenja, kot so Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) in Logistic Regression (LR)."
[Referenca](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Ta članek prikazuje, kako lahko kombiniranje različnih področij strojnega učenja ustvari zanimive rezultate, ki lahko pomagajo preprečiti širjenje lažnih novic in ustvarjanje resnične škode; v tem primeru je bil povod širjenje govoric o zdravljenju COVID, ki je sprožilo nasilje množic.

### Muzejsko strojno učenje

Muzeji so na pragu revolucije umetne inteligence, kjer postaja katalogizacija in digitalizacija zbirk ter iskanje povezav med artefakti lažje z napredkom tehnologije. Projekti, kot je [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.), pomagajo odkriti skrivnosti nedostopnih zbirk, kot so Vatikanski arhivi. Toda poslovni vidik muzejev koristi tudi modelom strojnega učenja.

Na primer, Umetnostni inštitut v Chicagu je zgradil modele za napovedovanje, kaj občinstvo zanima in kdaj bodo obiskali razstave. Cilj je ustvariti individualizirane in optimizirane izkušnje obiskovalcev ob vsakem obisku muzeja. "Med fiskalnim letom 2017 je model napovedal obisk in vstopnine z natančnostjo 1 odstotka, pravi Andrew Simnick, višji podpredsednik v Umetnostnem inštitutu."
[Referenca](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentacija strank

Najbolj učinkovite marketinške strategije ciljajo na stranke na različne načine glede na različne skupine. V tem članku so obravnavane uporabe algoritmov razvrščanja za podporo diferenciranemu marketingu. Diferencirani marketing pomaga podjetjem izboljšati prepoznavnost blagovne znamke, doseči več strank in zaslužiti več denarja.
[Referenca](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Izziv

Prepoznajte še en sektor, ki koristi nekatere tehnike, ki ste se jih naučili v tem učnem načrtu, in odkrijte, kako uporablja strojno učenje.
## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled & Samostojno učenje

Ekipa za podatkovno znanost pri Wayfairu ima več zanimivih videoposnetkov o tem, kako uporabljajo strojno učenje v svojem podjetju. Vredno si je [ogledati](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Naloga

[Lov na zaklad s strojno učenje](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.