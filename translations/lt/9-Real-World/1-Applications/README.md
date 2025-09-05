<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T07:52:15+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "lt"
}
-->
# Postscriptas: Mašininis mokymasis realiame pasaulyje

![Mašininio mokymosi realiame pasaulyje santrauka sketchnote piešinyje](../../../../sketchnotes/ml-realworld.png)  
> Sketchnote piešinys: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Šioje mokymo programoje išmokote daugybę būdų, kaip paruošti duomenis mokymui ir kurti mašininio mokymosi modelius. Jūs sukūrėte klasikinius regresijos, klasterizavimo, klasifikavimo, natūralios kalbos apdorojimo ir laiko eilučių modelius. Sveikiname! Dabar galbūt klausiate savęs, kam visa tai... kokios yra šių modelių realaus pasaulio taikymo sritys?

Nors pramonėje daug dėmesio sulaukia dirbtinis intelektas, kuris dažniausiai remiasi giluminiu mokymusi, klasikiniai mašininio mokymosi modeliai vis dar turi vertingų pritaikymo galimybių. Galbūt kai kurias iš jų jau naudojate šiandien! Šioje pamokoje tyrinėsite, kaip aštuonios skirtingos pramonės šakos ir teminės sritys naudoja šiuos modelius, kad jų taikomosios programos būtų našesnės, patikimesnės, protingesnės ir vertingesnės vartotojams.

## [Prieš paskaitą: testas](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finansai

Finansų sektorius siūlo daugybę galimybių mašininiam mokymuisi. Daugelis šios srities problemų gali būti modeliuojamos ir sprendžiamos naudojant ML.

### Kredito kortelių sukčiavimo aptikimas

Anksčiau kurse mokėmės apie [k-means klasterizavimą](../../5-Clustering/2-K-Means/README.md), bet kaip jis gali būti naudojamas sprendžiant kredito kortelių sukčiavimo problemas?

K-means klasterizavimas yra naudingas kredito kortelių sukčiavimo aptikimo technikoje, vadinamoje **išskirtinių atvejų aptikimu**. Išskirtiniai atvejai arba duomenų rinkinio stebėjimų nukrypimai gali parodyti, ar kredito kortelė naudojama įprastai, ar vyksta kažkas neįprasto. Kaip parodyta žemiau pateiktame straipsnyje, galite rūšiuoti kredito kortelių duomenis naudodami k-means klasterizavimo algoritmą ir priskirti kiekvieną operaciją klasteriui pagal tai, kiek ji atrodo išskirtinė. Tada galite įvertinti rizikingiausius klasterius, kad atskirtumėte sukčiavimą nuo teisėtų operacijų.  
[Nuoroda](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Turto valdymas

Turto valdyme asmuo ar įmonė valdo investicijas savo klientų vardu. Jų darbas yra ilgalaikėje perspektyvoje išlaikyti ir auginti turtą, todėl labai svarbu pasirinkti gerai veikiančias investicijas.

Vienas iš būdų įvertinti, kaip tam tikra investicija veikia, yra statistinė regresija. [Linijinė regresija](../../2-Regression/1-Tools/README.md) yra vertingas įrankis, padedantis suprasti, kaip fondas veikia, palyginti su tam tikru etalonu. Taip pat galime nustatyti, ar regresijos rezultatai yra statistiškai reikšmingi, arba kiek jie paveiktų kliento investicijas. Galite dar labiau išplėsti savo analizę naudodami daugybinę regresiją, kurioje atsižvelgiama į papildomus rizikos veiksnius. Kaip tai veiktų konkrečiam fondui, galite pamatyti žemiau pateiktame straipsnyje apie fondo veiklos vertinimą naudojant regresiją.  
[Nuoroda](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Švietimas

Švietimo sektorius taip pat yra labai įdomi sritis, kurioje galima taikyti ML. Čia galima spręsti įdomias problemas, tokias kaip sukčiavimo testuose ar rašiniuose aptikimas arba šališkumo, tyčinio ar netyčinio, valdymas vertinimo procese.

### Studentų elgesio prognozavimas

[Coursera](https://coursera.com), internetinė atvirų kursų platforma, turi puikų technologijų tinklaraštį, kuriame aptariami įvairūs inžineriniai sprendimai. Šiame atvejo tyrime jie nubrėžė regresijos liniją, siekdami ištirti, ar yra kokia nors koreliacija tarp žemo NPS (Net Promoter Score) įvertinimo ir kurso išlaikymo ar nutraukimo.  
[Nuoroda](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Šališkumo mažinimas

[Grammarly](https://grammarly.com), rašymo asistentas, tikrinantis rašybos ir gramatikos klaidas, savo produktuose naudoja sudėtingas [natūralios kalbos apdorojimo sistemas](../../6-NLP/README.md). Jie paskelbė įdomų atvejo tyrimą savo technologijų tinklaraštyje apie tai, kaip jie sprendė lyčių šališkumo problemą mašininiame mokymesi, apie kurią sužinojote mūsų [įvadinėje pamokoje apie sąžiningumą](../../1-Introduction/3-fairness/README.md).  
[Nuoroda](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Mažmeninė prekyba

Mažmeninės prekybos sektorius tikrai gali pasinaudoti ML, pradedant nuo geresnės klientų kelionės kūrimo iki optimalaus atsargų valdymo.

### Kliento kelionės personalizavimas

Wayfair, įmonė, prekiaujanti namų apyvokos prekėmis, tokiomis kaip baldai, siekia padėti klientams rasti tinkamus produktus pagal jų skonį ir poreikius. Šiame straipsnyje įmonės inžinieriai aprašo, kaip jie naudoja ML ir NLP, kad „pateiktų tinkamus rezultatus klientams“. Ypač jų Užklausų ketinimų variklis buvo sukurtas naudoti subjektų išskyrimą, klasifikatorių mokymą, turto ir nuomonių išskyrimą bei nuotaikų žymėjimą klientų atsiliepimuose. Tai yra klasikinis NLP naudojimo internetinėje mažmeninėje prekyboje pavyzdys.  
[Nuoroda](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Atsargų valdymas

Inovatyvios, lanksčios įmonės, tokios kaip [StitchFix](https://stitchfix.com), dėžutės paslauga, siunčianti drabužius vartotojams, labai remiasi ML rekomendacijoms ir atsargų valdymui. Jų stilistų komandos dirba kartu su prekybos komandomis: „vienas iš mūsų duomenų mokslininkų eksperimentavo su genetiniu algoritmu ir pritaikė jį drabužiams, kad numatytų, kokie drabužiai būtų sėkmingi, nors dar neegzistuoja. Mes tai pristatėme prekybos komandai, ir dabar jie gali naudoti tai kaip įrankį.“  
[Nuoroda](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Sveikatos priežiūra

Sveikatos priežiūros sektorius gali pasinaudoti ML optimizuodamas tyrimų užduotis ir logistikos problemas, tokias kaip pacientų pakartotinis priėmimas ar ligų plitimo stabdymas.

### Klinikiniai tyrimai

Toksikologija klinikiniuose tyrimuose yra didelis rūpestis vaistų kūrėjams. Kiek toksikologijos yra toleruotina? Šiame tyrime, analizuojant įvairius klinikinių tyrimų metodus, buvo sukurtas naujas požiūris į klinikinių tyrimų rezultatų prognozavimą. Konkrečiai, jie sugebėjo naudoti atsitiktinių miškų algoritmą, kad sukurtų [klasifikatorių](../../4-Classification/README.md), kuris gali atskirti vaistų grupes.  
[Nuoroda](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Ligoninių pakartotinio priėmimo valdymas

Ligoninės priežiūra yra brangi, ypač kai pacientai turi būti pakartotinai priimami. Šiame straipsnyje aptariama įmonė, kuri naudoja ML, kad prognozuotų pakartotinio priėmimo tikimybę naudodama [klasterizavimo](../../5-Clustering/README.md) algoritmus. Šie klasteriai padeda analitikams „atrasti pakartotinių priėmimų grupes, kurios gali turėti bendrą priežastį“.  
[Nuoroda](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Ligos valdymas

Neseniai vykusi pandemija atskleidė, kaip mašininis mokymasis gali padėti sustabdyti ligų plitimą. Šiame straipsnyje atpažinsite ARIMA, logistinių kreivių, linijinės regresijos ir SARIMA naudojimą. „Šis darbas yra bandymas apskaičiuoti šio viruso plitimo greitį ir taip prognozuoti mirčių, pasveikimų ir patvirtintų atvejų skaičių, kad galėtume geriau pasiruošti ir išgyventi.“  
[Nuoroda](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologija ir žaliųjų technologijų sektorius

Gamta ir ekologija susideda iš daugybės jautrių sistemų, kuriose dėmesys sutelkiamas į gyvūnų ir gamtos sąveiką. Labai svarbu tiksliai išmatuoti šias sistemas ir tinkamai reaguoti, jei kažkas nutinka, pavyzdžiui, miško gaisras ar gyvūnų populiacijos sumažėjimas.

### Miškų valdymas

Ankstesnėse pamokose mokėtės apie [stiprinamąjį mokymąsi](../../8-Reinforcement/README.md). Jis gali būti labai naudingas prognozuojant gamtos dėsningumus. Pavyzdžiui, jis gali būti naudojamas ekologinėms problemoms, tokioms kaip miško gaisrai ar invazinių rūšių plitimas, stebėti. Kanadoje tyrėjų grupė naudojo stiprinamąjį mokymąsi, kad sukurtų miško gaisrų dinamikos modelius iš palydovinių vaizdų. Naudodami inovatyvų „erdvinio plitimo procesą (SSP)“, jie įsivaizdavo miško gaisrą kaip „agentą bet kurioje kraštovaizdžio vietoje“. „Veiksmų rinkinys, kurį gaisras gali atlikti iš bet kurios vietos bet kuriuo metu, apima plitimą į šiaurę, pietus, rytus, vakarus arba neplitimą.“  

Šis požiūris apverčia įprastą RL nustatymą, nes atitinkamo Markovo sprendimų proceso (MDP) dinamika yra žinoma funkcija tiesioginiam gaisro plitimui. Daugiau apie šios grupės naudojamus klasikinius algoritmus skaitykite žemiau pateiktoje nuorodoje.  
[Nuoroda](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Gyvūnų judėjimo stebėjimas

Nors giluminis mokymasis sukėlė revoliuciją vizualiai stebint gyvūnų judėjimą (galite sukurti savo [poliarinių lokių stebėjimo įrankį](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) čia), klasikinis ML vis dar turi savo vietą šioje užduotyje.

Jutikliai, skirti stebėti ūkinių gyvūnų judėjimą, ir IoT naudoja tokio tipo vizualų apdorojimą, tačiau paprastesni ML metodai yra naudingi duomenims apdoroti. Pavyzdžiui, šiame straipsnyje buvo stebimos ir analizuojamos avių laikysenos naudojant įvairius klasifikatorių algoritmus. Galbūt atpažinsite ROC kreivę 335 puslapyje.  
[Nuoroda](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energijos valdymas

Mūsų pamokose apie [laiko eilučių prognozavimą](../../7-TimeSeries/README.md) buvo aptarta išmaniųjų stovėjimo skaitiklių koncepcija, siekiant generuoti pajamas miestui, remiantis pasiūlos ir paklausos supratimu. Šiame straipsnyje išsamiai aptariama, kaip klasterizavimas, regresija ir laiko eilučių prognozavimas kartu padėjo prognozuoti būsimą energijos naudojimą Airijoje, remiantis išmaniaisiais skaitikliais.  
[Nuoroda](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Draudimas

Draudimo sektorius yra dar viena sritis, kurioje naudojamas ML, siekiant sukurti ir optimizuoti gyvybingus finansinius ir aktuarinius modelius.

### Kintamumo valdymas

MetLife, gyvybės draudimo teikėjas, atvirai dalijasi, kaip jie analizuoja ir mažina kintamumą savo finansiniuose modeliuose. Šiame straipsnyje pastebėsite dvejetainės ir ranginės klasifikacijos vizualizacijas. Taip pat atrasite prognozavimo vizualizacijas.  
[Nuoroda](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Menai, kultūra ir literatūra

Meno srityje, pavyzdžiui, žurnalistikoje, yra daug įdomių problemų. Netikrų naujienų aptikimas yra didelė problema, nes įrodyta, kad jos gali paveikti žmonių nuomonę ir netgi sugriauti demokratijas. Muziejai taip pat gali pasinaudoti ML, pradedant nuo ryšių tarp artefaktų paieškos iki išteklių planavimo.

### Netikrų naujienų aptikimas

Netikrų naujienų aptikimas šiandieninėje žiniasklaidoje tapo katės ir pelės žaidimu. Šiame straipsnyje tyrėjai siūlo sistemą, kuri sujungia kelias ML technikas, kurias mes studijavome, ir gali būti išbandyta, o geriausias modelis pritaikytas: „Ši sistema remiasi natūralios kalbos apdorojimu, kad išgautų duomenų ypatybes, o tada šios ypatybės naudojamos mašininio mokymosi klasifikatorių, tokių kaip Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) ir Logistic Regression (LR), mokymui.“  
[Nuoroda](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Šis straipsnis parodo, kaip skirtingų ML sričių derinimas gali duoti įdomių rezultatų, kurie padeda sustabdyti netikrų naujienų plitimą ir sukeliamą realią žalą; šiuo atveju impulsas buvo gandų apie COVID gydymą plitimas, kuris sukėlė smurtą.

### Muziejų ML

Muziejai yra ant AI revoliucijos slenksčio, kai kolekcijų katalogavimas ir skaitmeninimas bei ryšių tarp artefaktų paieška tampa lengvesni, kai technologijos tobulėja. Tokie projektai kaip [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) padeda atskleisti neprieinam
## [Po paskaitos viktorina](https://ff-quizzes.netlify.app/en/ml/)

## Peržiūra ir savarankiškas mokymasis

Wayfair duomenų mokslo komanda turi keletą įdomių vaizdo įrašų apie tai, kaip jie naudoja mašininį mokymąsi savo įmonėje. Verta [pažiūrėti](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Užduotis

[Mašininio mokymosi lobio paieška](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama profesionali žmogaus vertimo paslauga. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius naudojant šį vertimą.