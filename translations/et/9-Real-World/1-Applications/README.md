<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-10-11T11:49:56+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "et"
}
-->
# Järelsõna: Masinõpe pärismaailmas

![Masinõppe kokkuvõte pärismaailmas sketšina](../../../../translated_images/ml-realworld.26ee2746716155771f8076598b6145e6533fe4a9e2e465ea745f46648cbf1b84.et.png)
> Sketš joonistas [Tomomi Imura](https://www.twitter.com/girlie_mac)

Selles õppekavas õppisite mitmeid viise, kuidas andmeid treenimiseks ette valmistada ja masinõppe mudeleid luua. Te ehitasite klassikalisi regressiooni-, klasterdamis-, klassifitseerimis-, loomuliku keele töötlemise ja ajareamudeleid. Palju õnne! Nüüd võite mõelda, milleks see kõik vajalik on... millised on nende mudelite pärismaailma rakendused?

Kuigi tööstuses on suur huvi AI vastu, mis tavaliselt kasutab süvaõpet, on klassikalistel masinõppe mudelitel endiselt väärtuslikke rakendusi. Võite isegi täna mõnda neist rakendustest kasutada! Selles õppetükis uurite, kuidas kaheksa erinevat tööstusharu ja valdkonda kasutavad neid mudeleid, et muuta oma rakendused tõhusamaks, usaldusväärsemaks, intelligentsemaks ja kasutajatele väärtuslikumaks.

## [Loengu-eelne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finantssektor

Finantssektor pakub palju võimalusi masinõppe rakendamiseks. Paljud probleemid selles valdkonnas sobivad hästi ML-i abil modelleerimiseks ja lahendamiseks.

### Krediitkaardipettuste tuvastamine

Me õppisime [k-means klasterdamist](../../5-Clustering/2-K-Means/README.md) varem kursusel, kuid kuidas saab seda kasutada krediitkaardipettustega seotud probleemide lahendamiseks?

K-means klasterdamine on kasulik krediitkaardipettuste tuvastamise tehnikas, mida nimetatakse **äärmusväärtuste tuvastamiseks**. Äärmusväärtused ehk kõrvalekalded andmekogumi vaatlustes võivad näidata, kas krediitkaarti kasutatakse tavapäraselt või toimub midagi ebatavalist. Nagu allpool lingitud artiklis näidatud, saab krediitkaardi andmeid sorteerida k-means klasterdamise algoritmi abil ja määrata iga tehing klastrisse, lähtudes sellest, kui palju see äärmusväärtusena paistab. Seejärel saab hinnata riskantsemaid klastreid, et tuvastada pettuslikud versus seaduslikud tehingud.
[Viide](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Varahaldus

Varahalduses haldab üksikisik või ettevõte investeeringuid oma klientide nimel. Nende ülesanne on pikaajaliselt säilitada ja kasvatada rikkust, mistõttu on oluline valida hästi toimivaid investeeringuid.

Üks viis, kuidas hinnata konkreetse investeeringu toimivust, on statistiline regressioon. [Lineaarne regressioon](../../2-Regression/1-Tools/README.md) on väärtuslik tööriist, et mõista, kuidas fond toimib võrreldes mõne võrdlusnäitajaga. Samuti saame järeldada, kas regressiooni tulemused on statistiliselt olulised või kui palju need mõjutaksid kliendi investeeringuid. Analüüsi saab veelgi laiendada mitme regressiooni abil, kus arvesse võetakse täiendavaid riskitegureid. Näide selle kohta, kuidas see konkreetse fondi puhul toimiks, on toodud allpool lingitud artiklis.
[Viide](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Haridus

Haridussektor on samuti väga huvitav valdkond, kus ML-i saab rakendada. Seal on huvitavaid probleeme, mida lahendada, näiteks testide või esseede petmise tuvastamine või kallutatuse, tahtliku või tahtmatu, haldamine hindamisprotsessis.

### Õpilaste käitumise ennustamine

[Coursera](https://coursera.com), veebipõhine avatud kursuste pakkuja, omab suurepärast tehnoloogiaalast blogi, kus nad arutavad paljusid insenerialaseid otsuseid. Selles juhtumiuuringus joonistasid nad regressioonijoone, et uurida, kas madal NPS (Net Promoter Score) hinnang korreleerub kursuse säilitamise või katkestamisega.
[Viide](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Kallutatuse vähendamine

[Grammarly](https://grammarly.com), kirjutamisassistent, mis kontrollib õigekirja ja grammatikavigu, kasutab oma toodetes keerukaid [loomuliku keele töötlemise süsteeme](../../6-NLP/README.md). Nad avaldasid huvitava juhtumiuuringu oma tehnoloogiaalases blogis, kus nad käsitlesid soolise kallutatuse probleemi masinõppes, mida te õppisite meie [õigluse sissejuhatavas õppetükis](../../1-Introduction/3-fairness/README.md).
[Viide](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Jaekaubandus

Jaekaubandussektor võib kindlasti kasu saada ML-i kasutamisest, alates parema klienditeekonna loomisest kuni varude optimaalse haldamiseni.

### Klienditeekonna isikupärastamine

Wayfairis, ettevõttes, mis müüb kodutarbeid nagu mööblit, on klientide aitamine õige toote leidmisel nende maitsele ja vajadustele ülimalt tähtis. Selles artiklis kirjeldavad ettevõtte insenerid, kuidas nad kasutavad ML-i ja NLP-d, et "pakkuda klientidele õigeid tulemusi". Eriti nende Päringu Kavatsuse Mootor on loodud kasutama üksuse eraldamist, klassifikaatori treenimist, vara ja arvamuse eraldamist ning sentimentide märgistamist klientide arvustustes. See on klassikaline näide, kuidas NLP töötab veebipõhises jaekaubanduses.
[Viide](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Varude haldamine

Innovatiivsed ja paindlikud ettevõtted nagu [StitchFix](https://stitchfix.com), kastiteenus, mis saadab riideid tarbijatele, tuginevad tugevalt ML-ile soovituste ja varude haldamiseks. Nende stiilimeeskonnad teevad koostööd nende kaubandusmeeskondadega: "üks meie andmeteadlastest katsetas geneetilist algoritmi ja rakendas seda rõivastele, et ennustada, milline rõivaese oleks edukas, kuigi seda täna veel ei eksisteeri. Me tutvustasime seda kaubandusmeeskonnale ja nüüd saavad nad seda kasutada tööriistana."
[Viide](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Tervishoid

Tervishoiusektor saab kasutada ML-i, et optimeerida uurimistöid ja logistilisi probleeme, nagu patsientide uuesti haiglasse vastuvõtmine või haiguste leviku peatamine.

### Kliiniliste uuringute haldamine

Toksilisus kliinilistes uuringutes on ravimite tootjatele suur mure. Kui palju toksilisust on talutav? Selles uuringus viis erinevate kliiniliste uuringumeetodite analüüs uue lähenemisviisi väljatöötamiseni, et ennustada kliiniliste uuringute tulemuste tõenäosust. Eelkõige suutsid nad kasutada random forest meetodit, et luua [klassifikaator](../../4-Classification/README.md), mis eristab ravimite rühmi.
[Viide](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Haigla uuesti vastuvõtmise haldamine

Haiglaravi on kulukas, eriti kui patsiente tuleb uuesti vastu võtta. Selles artiklis arutatakse ettevõtet, mis kasutab ML-i, et ennustada uuesti vastuvõtmise potentsiaali, kasutades [klasterdamise](../../5-Clustering/README.md) algoritme. Need klastrid aitavad analüütikutel "avastada uuesti vastuvõtmise rühmi, millel võib olla ühine põhjus".
[Viide](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Haiguste haldamine

Hiljutine pandeemia on heitnud eredat valgust sellele, kuidas masinõpe võib aidata haiguste levikut peatada. Selles artiklis tunnete ära ARIMA, logistilised kõverad, lineaarse regressiooni ja SARIMA kasutamise. "See töö on katse arvutada selle viiruse leviku kiirust ja seega ennustada surmajuhtumeid, taastumisi ja kinnitatud juhtumeid, et aidata meil paremini valmistuda ja ellu jääda."
[Viide](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ökoloogia ja roheline tehnoloogia

Loodus ja ökoloogia koosnevad paljudest tundlikest süsteemidest, kus loomade ja looduse vastastikmõju on keskmes. Oluline on neid süsteeme täpselt mõõta ja tegutseda asjakohaselt, kui midagi juhtub, näiteks metsatulekahju või loomade populatsiooni langus.

### Metsade haldamine

Te õppisite [Tugevdatud Õppimist](../../8-Reinforcement/README.md) varasemates õppetundides. See võib olla väga kasulik, kui üritatakse ennustada mustreid looduses. Eriti saab seda kasutada ökoloogiliste probleemide, nagu metsatulekahjud ja invasiivsete liikide levik, jälgimiseks. Kanadas kasutas grupp teadlasi Tugevdatud Õppimist, et ehitada metsatulekahjude dünaamika mudeleid satelliidipiltidest. Kasutades uuenduslikku "ruumiliselt levivat protsessi (SSP)", kujutasid nad metsatulekahju "agendina igas maastiku rakus." "Tegevuste komplekt, mida tuli võib igast asukohast igal ajahetkel võtta, hõlmab levimist põhja, lõuna, ida või lääne suunas või mitte levimist.

See lähenemine pöörab tavapärase RL-i seadistuse ümber, kuna vastava Markovi Otsustusprotsessi (MDP) dünaamika on teada funktsioon kohese metsatule leviku jaoks." Lugege rohkem selle grupi kasutatud klassikaliste algoritmide kohta allpool lingitud artiklist.
[Viide](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Loomade liikumise jälgimine

Kuigi süvaõpe on loonud revolutsiooni loomade liikumise visuaalses jälgimises (saate ehitada oma [jääkaru jälgija](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) siin), on klassikalisel ML-il endiselt koht selles ülesandes.

Sensorid farmiloomade liikumise jälgimiseks ja IoT kasutavad seda tüüpi visuaalset töötlemist, kuid lihtsamad ML-tehnikad on kasulikud andmete eeltöötlemiseks. Näiteks selles artiklis jälgiti ja analüüsiti lammaste poose erinevate klassifikaatorite algoritmide abil. Võite ära tunda ROC kõvera leheküljel 335.
[Viide](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energiakasutuse haldamine

Meie õppetundides [ajarealise prognoosimise](../../7-TimeSeries/README.md) kohta tõime esile nutikate parkimismõõturite kontseptsiooni, et genereerida linnale tulu, lähtudes pakkumise ja nõudluse mõistmisest. Selles artiklis arutatakse üksikasjalikult, kuidas klasterdamine, regressioon ja ajarealise prognoosimine kombineeritult aitavad ennustada tulevast energiakasutust Iirimaal, tuginedes nutimõõtmisele.
[Viide](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Kindlustus

Kindlustussektor on veel üks valdkond, mis kasutab ML-i elujõuliste finants- ja aktuaarsusmudelite loomiseks ja optimeerimiseks.

### Volatiilsuse haldamine

MetLife, elukindlustuse pakkuja, on avameelne selle kohta, kuidas nad analüüsivad ja leevendavad volatiilsust oma finantsmudelites. Selles artiklis märkate binaarse ja järjestikuse klassifikatsiooni visualiseeringuid. Samuti avastate prognoosimise visualiseeringuid.
[Viide](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Kunst, kultuur ja kirjandus

Kunstis, näiteks ajakirjanduses, on palju huvitavaid probleeme. Valeuudiste tuvastamine on suur probleem, kuna on tõestatud, et see mõjutab inimeste arvamust ja isegi demokraatiaid. Muuseumid võivad samuti kasu saada ML-i kasutamisest, alates artefaktide vaheliste seoste leidmisest kuni ressursside planeerimiseni.

### Valeuudiste tuvastamine

Valeuudiste tuvastamine on tänapäeva meedias muutunud kassi ja hiire mänguks. Selles artiklis soovitavad teadlased süsteemi, mis kombineerib mitmeid ML-tehnikaid, mida oleme õppinud, ja testivad parimat mudelit: "See süsteem põhineb loomuliku keele töötlemisel, et andmetest funktsioone eraldada, ja seejärel kasutatakse neid funktsioone masinõppe klassifikaatorite, nagu Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) ja Logistic Regression (LR), treenimiseks."
[Viide](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

See artikkel näitab, kuidas erinevate ML-valdkondade kombineerimine võib anda huvitavaid tulemusi, mis aitavad valeuudiste levikut peatada ja tõelist kahju ära hoida; antud juhul oli ajendiks COVID-i ravimeetodite kohta levivate kuulujuttude levik, mis tekitas rahutusi.

### Muuseumide ML

Muuseumid on AI-revolutsiooni künnisel, kus kollektsioonide kataloogimine ja digiteerimine ning artefaktide vaheliste seoste leidmine muutub tehnoloogia edenedes lihtsamaks. Projektid nagu [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) aitavad avada ligipääsmatute kollektsioonide, nagu Vatikani arhiivide, saladusi. Kuid muuseumide äriline aspekt saab samuti kasu ML-mudelitest.

Näiteks Chicago Kunstiinstituut ehitas mudeleid, et ennustada, millised ekspositsioonid publikut huvitavad ja millal nad neid külastavad. Eesmärk on luua iga külastuse ajal individuaalne ja optimeeritud külastajakogemus. "2017. majandusaastal ennustas mudel külastatavust ja sissepääse 1% täpsusega, ütleb Andrew Simnick, Chicago Kunstiinstituudi vanem asepresident."
[Viide](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Turundus

### Kliendisegmentide määratlemine

Kõige tõhusamad turundusstrateegiad sihivad kliente erinevatel viisidel, lähtudes erinevatest rühmitustest. Selles artiklis arutatakse klasterdamise algoritmide kasutamist, et toetada diferentseeritud turundust. Diferentseeritud turundus aitab ettevõtetel parandada brändi tuntust, jõuda rohkemate klientideni ja teenida rohkem raha.
[Viide](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Väljakutse

Tuvastage veel üks sektor, mis saab kasu mõnest tehnikast, mida te selles õppekavas õppisite, ja uurige, kuidas see ML-i kasutab.
## [Loengujärgne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülevaade ja iseseisev õppimine

Wayfairi andmeteaduse meeskonnal on mitmeid huvitavaid videoid, kus nad selgitavad, kuidas nad oma ettevõttes masinõpet kasutavad. Tasub [vaadata](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Ülesanne

[Masinõppe aardejaht](assignment.md)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.