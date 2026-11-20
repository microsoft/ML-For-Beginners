# Masinõppe lahenduste loomine vastutustundliku tehisintellektiga
 
![Vastutustundliku tehisintellekti kokkuvõte masinõppes joonistusena](../../../../translated_images/et/ml-fairness.ef296ebec6afc98a.webp)
> Joonistus autorilt [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Eelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)
 
## Sissejuhatus

Selles õppekavas hakkate avastama, kuidas masinõpe mõjutab ja mõjutab meie igapäevaelu. Isegi praegu on süsteemid ja mudelid igapäevastes otsustusülesannetes, nagu tervishoiudiagnoosid, laenuandmine või pettuste tuvastamine. Seega on oluline, et need mudelid töötaksid hästi, pakkudes usaldusväärseid tulemusi. Nagu iga tarkvararakendus, võivad ka tehisintellektisüsteemid mitte vastata ootustele või anda soovimatuid tulemusi. Seetõttu on oluline mõista ja selgitada tehisintellektimudelite käitumist. 

Kujutage ette, mis võib juhtuda, kui andmed, mida te kasutate nende mudelite loomiseks, puuduvad teatud demograafilised rühmad, nagu rass, sugu, poliitilised vaated, religioon või esindavad ebaproportsionaalselt mõnda demograafilist rühma. Mis juhtub, kui mudeli väljund tõlgendatakse teatud demograafilist rühma soosivaks? Mis on selle rakenduse tagajärg? Lisaks, mis juhtub, kui mudelil on kahjulik väljund ja see kahjustab inimesi? Kes vastutab tehisintellektisüsteemide käitumise eest? Need on mõned küsimused, mida selles õppekavas uurime. 

Selles õppetükis: 

- Teadlikkuse tõstmine õigluse tähtsusest masinõppes ja õiglusega seotud kahjude kohta.
- Harjutada äärmuste ja ebatavaliste stsenaariumide uurimist usaldusväärsuse ja ohutuse tagamiseks.
- Mõista vajadust kõiki võimestada kaasavate süsteemide kujundamise kaudu.
- Uurida, kui tähtis on kaitsta andmete ja inimeste privaatsust ja turvalisust.
- Näha klaaskasti lähenemise olulisust tehisintellektimudelite käitumise selgitamisel.
- Olla teadlik vastutusest kui usalduse ehitamise alusest tehisintellektisüsteemides.

## Eeldused

Eeldusena läbige palun "Vastutustundliku tehisintellekti põhimõtted" õpitee ja vaadake allolevat videot sellel teemal:

Lisateave vastutustundliku tehisintellekti kohta, järgides seda [õppeteed](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsofti lähenemine vastutustundlikule tehisintellektile](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsofti lähenemine vastutustundlikule tehisintellektile")

> 🎥 Klõpsake ülaloleval pildil video vaatamiseks: Microsofti lähenemine vastutustundlikule tehisintellektile

## Õiglus

Tehisintellektisüsteemid peaksid kõiki õiglaselt kohtlema ega tohi mõjutada sarnaseid inimrühmi erinevalt. Näiteks, kui tehisintellektisüsteemid annavad juhiseid meditsiinilise ravi, laenutaotluste või töölevõtmise kohta, peaksid nad tegema samad soovitused kõigile, kellel on sarnased sümptomid, rahalised tingimused või kutseoskused. Igaüks meist kannab endas pärandatud eelarvamusi, mis mõjutavad meie otsuseid ja tegevusi. Need eelarvamused võivad andmetes, mida kasutame tehisintellekti koolitamiseks, ilmne olla. Mõnikord võib selline manipulatsioon toimuda tahtmatult. Teadvustatult on sageli raske teada, millal andmeid kallutatakse. 

**„Õiglust mittevastavus“** hõlmab negatiivseid mõjusid ehk „kahjusid“ teatud inimrühmadele, nagu rass, sugu, vanus või puude staatus. Peamised õiglust puudutavad kahjud võib liigitada järgmiselt: 

- **Jaotamine**, kui näiteks ühte sugu või etnilist rühma eelistatakse teise ees.
- **Teenuse kvaliteet**. Kui andmeid koolitatakse ainult ühe konkreetse stsenaariumi jaoks, kuid tegelikkus on palju keerulisem, võib see põhjustada kehva toimivusega teenuse. Näiteks käsepumbal, mis ei suutnud näida tundvat tumedanahalisi inimesi. [Viide](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Alandamine**. Ebaõiglane kritiseerimine ja sildistamine. Näiteks pildituvastustehnoloogia vales altveerinevusena märgistas tumedanahaliste inimeste pilte gorilladena.
- **Üle- või alerepresentatsioon**. Näide on teatud rühma puudumine mõnes ametis, ning teenus või funktsioon, mis seda soodustab, aitab kahju tekitada.
- **Stereotüübid**. Määratud rühmale eelnevalt määratud omaduste seostamine. Näiteks inglise ja türgi vahelise keeletõlkesüsteemi ebatäpsused seoses sooliste stereotüüpidega seotud sõnadega.

![tõlge türgi keelde](../../../../translated_images/et/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> tõlge türgi keelde

![tõlge tagasi inglise keelde](../../../../translated_images/et/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> tõlge tagasi inglise keelde

Tehisintellektisüsteemide kujundamisel ja testimisel peame tagama, et tehisintellekt on õiglane ega ole programmeeritud tegema kallutatud või diskrimineerivaid otsuseid, mida inimestelgi on keelatud teha. Õigluse tagamine tehisintellektis ja masinõppes on endiselt keeruline sotsiaal-tehniline väljakutse. 

### Usaldusväärsus ja ohutus

Usalduse loomiseks peavad tehisintellektisüsteemid olema usaldusväärsed, ohutud ja järjekindlad nii tavatingimustes kui ootamatutes olukordades. On oluline teada, kuidas tehisintellekt käitub mitmesugustes olukordades, eriti äärmustes. Tehisintellektilahenduste loomisel peab olema suur rõhk sellel, kuidas lahendada erinevaid olukordi, millega need kokku puutuvad. Näiteks isesõitev auto peab seadma inimeste ohutuse esikohale. Seetõttu peab auto juhtimiseks kasutatav tehisintellekt arvestama kõigi võimalike olukordadega, näiteks öö, äikesetormid või lumesajud, laste jooks minek üle tee, lemmikloomad, teetööd jne. Kui hästi suudab tehisintellekt usaldusväärselt ja ohutult erinevate tingimustega toime tulla, peegeldab see seda, kui põhjalikult on andmeteadlane või arendaja disaini või testimise käigus ette mõelnud.  

> [🎥 Klõpsake videole: Usaldusväärsus ja ohutus tehisintellektis](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Kaasatus

Tehisintellekti süsteeme tuleks kujundada nii, et need kaasaksid ja võimestaksid kõiki inimesi. Andmeteadlased ja arendajad otsivad ja lahendavad süsteemi võimalikke takistusi, mis võivad tahtmatult inimesi välistada. Näiteks on maailmas umbes 1 miljard inimest, kellel on puue. Tehisintellekti arenguga pääsevad nad lihtsamalt ligi informatsioonile ja võimalustele oma igapäevaelus. Takistuste kõrvaldamine loob võimalusi uuendusteks ja paremate kogemustega tehisintellektitoodete loomiseks, mis kasu toovad kõigile. 

> [🎥 Klõpsake videole: Kaasatus tehisintellektis](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Turvalisus ja privaatsus 

Tehisintellektisüsteemid peavad olema turvalised ja austama inimeste privaatsust. Inimestel on vähem usaldust selliste süsteemide vastu, mis ohustavad nende privaatsust, andmeid või elu. Masinõppemudelite koolitamisel tugineb tulemuste saavutamine andmetele. Seetõttu tuleb hinnata andmete päritolu ja terviklikkust. Näiteks, kas andmed esitas kasutaja või olid avalikult kättesaadavad? Samuti on väga oluline arendada tehisintellektisüsteeme, mis suudavad kaitsta konfidentsiaalset teavet ja taluda rünnakuid. Tehisintellekti levikuga muutub privaatsuse kaitse ja oluliste isiku- ning ärandmete turvamine üha olulisemaks ja keerukamaks teemaks. Privaatsus ja andmekaitse vajavad eriti tähelepanu, kuna andmete kättesaadavus on vajalik täpsete ja hästi informeeritud otsuste tegemiseks tehisintellektisüsteemide poolt. 

> [🎥 Klõpsake videole: Turvalisus tehisintellektis](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Tööstusena oleme saavutanud märkimisväärseid edusamme privaatsuse ja turvalisuse valdkonnas, mida on oluliselt toetanud sellised regulatsioonid nagu GDPR (Isikuandmete kaitse üldmäärus). 
- Kuid tehisintellektisüsteemide puhul peame tunnistama pinget personaalsema ja tõhusama süsteemi loomise ning privaatsuse vahel. 
- Nagu juhtus internetiühendusega arvutite sünniga, näeme ka tehisintellekti puhul turvaintsidentide hulga kasvu. 
- Samal ajal on tehisintellekti kasutatud turvalisuse parandamiseks. Näiteks on enamik tänapäevaseid viirusetõrjeskännerid juhitud tehisintellekti heuristikatest. 
- Me peame tagama, et andmeteaduse protsessid oleksid kooskõlas uusimate privaatsuse ja turvalisuse tavadega. 


### Läbipaistvus
Tehisintellektisüsteemid peaksid olema arusaadavad. Läbipaistvuse oluline osa on tehisintellektisüsteemide ja nende komponentide käitumise selgitamine. Tehisintellekti parema mõistmise tagamiseks peab huvigruppidele olema arusaadav, kuidas ja miks süsteemid töötavad, et tuvastada võimalikke jõudluse probleeme, ohutus- ja privaatsusmuresid, kallutatusi, välistavaid praktikaid või soovimatut tulemusi. Usume ka, et tehisintellekti kasutajad peaksid ausalt ja avameelselt rääkima, millal, miks ja kuidas nad neid kasutusele võtavad, samuti kasutatavate süsteemide piirangutest. Näiteks kui pank kasutab tehisintellekti süsteemi tarbijalaenude otsuste toetamiseks, on oluline uurida tulemusi ja mõista, millised andmed mõjutavad süsteemi soovitusi. Valitsused hakkavad regulaatorina tehisintellekti kõigis tööstusharudes reguleerima, seega peavad andmeteadlased ja organisatsioonid selgitama, kas tehisintellektisüsteem vastab regulatiivsetele nõuetele, eriti kui tekib soovimatu tulemus. 

> [🎥 Klõpsake videole: Läbipaistvus tehisintellektis](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kuna tehisintellektisüsteemid on nii keerukad, on raske mõista, kuidas need töötavad ja tõlgendada tulemusi. 
- See puudus mõjutab nende süsteemide haldamist, rakendamist ja dokumenteerimist. 
- Olulisem on, et see arusaamatus mõjutab otsuseid, mis tehakse nende süsteemide toodetud tulemuste põhjal. 

### Vastutus
 
Tehisintellektisüsteemide disainerid ja kasutuselevõtjad peavad vastutama süsteemide toimimise eest. Vastutus on eriti oluline tundlike tehnoloogiate puhul, nagu näotuvastus. Viimasel ajal on kasvanud nõudlus näotuvastustehnoloogia järele, eriti õiguskaitseorganisatsioonid näevad selles potentsiaali kadunud laste leidmiseks. Kuid need tehnoloogiad võiksid valitsuse poolt potentsiaalselt kasutada oma kodanike põhiõiguste ohustamiseks, näiteks võimaldades pidevat jälgimist konkreetsetest isikutest. Seega peavad andmeteadlased ja organisatsioonid vastutama selle eest, kuidas nende tehisintellektisüsteem mõjutab üksikisikuid või ühiskonda.

[![Juhtiv tehisintellekti uurija hoiatab näotuvastuse kaudu toimuva massijälgimise eest](../../../../translated_images/et/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsofti lähenemine vastutustundlikule tehisintellektile")

> 🎥 Klõpsake ülaloleval pildil video vaatamiseks: Hoiduge massijälgimisest näotuvastuse kaudu

Lõppkokkuvõttes on üheks suurimaks küsimuseks meie põlvkonnale, kes on esimene, kes toob tehisintellekti ühiskonda, kuidas tagada, et arvutid jäävad inimestele vastutavaks ja kuidas tagada, et arvuteid kujundavad inimesed jäävad vastutavaks kõigi teiste ees.

## Mõjuhindamine 

Enne masinõppemudeli koolitamist on oluline läbi viia mõjuhindamine, et mõista tehisintellektisüsteemi eesmärki; mis on kavandatud kasutus; kus seda rakendatakse; ja kes süsteemiga suhtlevad. Need on abiks ülevaatajatele või testijatele süsteemi hindamisel, et teada saada, mida võtta arvesse võimalike riskide ja oodatavate tagajärgede tuvastamisel.

Järgnevad on keskendatavad valdkonnad mõjuhindamise läbiviimisel:

* **Negatiivne mõju üksikisikutele**. Teadlikkus kõigist piirangutest või nõuetest, lubamatust kasutamisest või teadaolevatest piirangutest, mis takistavad süsteemi toimimist, on ülioluline, et tagada süsteemi kasutamine ohutult.
* **Andmenõuded**. Arusaamine, kuidas ja kus süsteem andmeid kasutab, võimaldab ülevaatajatel uurida, milliseid andmenõudeid tuleb arvesse võtta (nt GDPR või HIPAA andmekaitse), ning hinnata, kas andmete allikas või kogus on mudeli koolitamiseks piisav.
* **Kokkuvõte mõjust**. Koostada nimekiri võimalikest kahjudest, mis võivad süsteemi kasutamisest tekkida. ML elutsükli jooksul hinnata, kas probleemid on lahendatud või leevendatud.
* **Rakendatavad eesmärgid** kõigi kuue põhialuse kohta. Hinnata, kas iga põhimõtte eesmärgid on täidetud ja kas on tühimikke.


## Vastutustundlik AI silumine  

Nagu tarkvararakenduse silumine, on ka tehisintellektisüsteemi silumine vajalik protsess süsteemis esinevate probleemide tuvastamiseks ja lahendamiseks. Mudeli mittetäielik või vastutustundetu toimimine võib olla tingitud mitmetest teguritest. Enamik traditsioonilisi mudeli tulemuslikkuse mõõdikuid on mudeli kvantitatiivsed kokkuvõtted, mis ei ole piisavad vastutustundliku AI põhimõtete rikkumise analüüsimiseks. Lisaks on masinõppemudel must kast, mis teeb raskeks mõista tulemuste põhjusi või selgitada vigu. Kursuse hilisemas osas õpime kasutama Vastutustundliku AI armatuurlaua tööriista, mis aitab AI süsteeme siluda. Armatuurlaud pakub andmeteadlastele ja arendajatele holistilist tööriista:

* **Vigade analüüs**. Mudeli vead, mis võivad mõjutada süsteemi õiglust või usaldusväärsust.
* **Mudeli ülevaade**. Erinevuste tuvastamine mudeli toimimises eri andmekogumite puhul.
* **Andmete analüüs**. Andmete jaotuse mõistmine ning võimaliku kallutatuse tuvastamine, mis võib põhjustada õigluse, kaasatuse ja usaldusväärsuse probleeme.
* **Mudeli tõlgendatavus**. Mõista, mis mõjutab mudeli prognoose. See aitab selgitada mudeli käitumist, mis on oluline läbipaistvuse ja vastutuse jaoks.


## 🚀 Väljakutse 
 
Kahjude tekkimise vältimiseks tuleks:

- meeskonnas olla mitmekesine erinevate taustade ja perspektiividega inimeste osas
- investeerida andmekogumitesse, mis peegeldavad meie ühiskonna mitmekesisust
- arendada masinõppe elutsükli jooksul paremaid meetodeid vastutustundetu AI avastamiseks ja parandamiseks

Mõelge reaalse elu stsenaariumitele, kus mudeli usaldusväärsuse puudumine on ilmne mudelite loomisel ja kasutamisel. Mida veel peaksime arvesse võtma? 

## [Pärast loengut viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Kordamine ja iseseisev õpe 
 
Selles õppetükis olete õppinud mõningaid põhiteadmisi õiglusest ja ebaõiglusest masinõppes.  
 
Vaadake seda töötuba teemasse süvenemiseks: 

- Vastutustundliku tehisintellekti poole püüdlemine: põhimõtete rakendamine praktikas autoritelt Besmira Nushi, Mehrnoosh Sameki ja Amit Sharma

[![Vastutustundliku tehisintellekti tööriistakomplekt: avatud lähtekoodiga raamistik vastutustundliku tehisintellekti loomiseks](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Avatud lähtekoodiga raamistik vastutustundliku tehisintellekti loomiseks")

> 🎥 Klõpsake ülaloleval pildil, et vaadata videot: RAI Toolbox: Avatud lähtekoodiga raamistik vastutustundliku tehisintellekti loomiseks autoritelt Besmira Nushi, Mehrnoosh Sameki ja Amit Sharma

Loe ka: 

- Microsofti RAI ressursikeskus: [Vastutustundliku tehisintellekti ressursid – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsofti FATE uurimisrühm: [FATE: Õiglus, vastutus, läbipaistvus ja eetika tehisintellektis - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Vastutustundliku tehisintellekti tööriistakomplekti GitHub hoidla](https://github.com/microsoft/responsible-ai-toolbox)

Loe Azure Machine Learning tööriistadest õiglust tagamiseks:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Ülesanne

[Avasta RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Lahtiütlus**:
See dokument on tõlgitud kasutades AI tõlketeenust [Co-op Translator](https://github.com/Azure/co-op-translator). Kuigi me püüdleme täpsuse poole, palun pange tähele, et automatiseeritud tõlgetes võib esineda vigu või ebatäpsusi. Originaaldokument selle emakeeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitatakse kasutada professionaalset inimtõlget. Me ei vastuta selle tõlkega seotud eksimustest või valesti mõistmistest.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->