<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T07:55:02+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "lt"
}
-->
# Kuriant mašininio mokymosi sprendimus su atsakingu dirbtiniu intelektu

![Atsakingo dirbtinio intelekto mašininio mokymosi santrauka sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote sukūrė [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Prieš paskaitą atlikite testą](https://ff-quizzes.netlify.app/en/ml/)

## Įvadas

Šioje mokymo programoje pradėsite suprasti, kaip mašininis mokymasis veikia mūsų kasdienį gyvenimą. Jau dabar sistemos ir modeliai dalyvauja kasdieniuose sprendimų priėmimo procesuose, tokiuose kaip sveikatos priežiūros diagnozės, paskolų patvirtinimai ar sukčiavimo aptikimas. Todėl svarbu, kad šie modeliai veiktų patikimai ir teiktų patikimus rezultatus. Kaip ir bet kuri programinė įranga, dirbtinio intelekto sistemos gali neatitikti lūkesčių arba turėti nepageidaujamų rezultatų. Todėl būtina suprasti ir paaiškinti dirbtinio intelekto modelio elgesį.

Įsivaizduokite, kas gali nutikti, kai duomenys, kuriuos naudojate modelių kūrimui, neturi tam tikrų demografinių duomenų, tokių kaip rasė, lytis, politinės pažiūros, religija, arba neproporcingai atspindi šiuos demografinius duomenis. O kas, jei modelio rezultatai yra interpretuojami taip, kad jie palankūs tam tikrai demografinei grupei? Kokios pasekmės tai gali turėti programai? Be to, kas nutinka, kai modelis sukelia neigiamą rezultatą ir yra žalingas žmonėms? Kas atsakingas už dirbtinio intelekto sistemos elgesį? Tai yra klausimai, kuriuos nagrinėsime šioje mokymo programoje.

Šioje pamokoje jūs:

- Suprasite, kodėl svarbu užtikrinti teisingumą mašininiame mokymesi ir išvengti su tuo susijusių žalingų pasekmių.
- Susipažinsite su praktika, kaip analizuoti išskirtinius atvejus ir neįprastas situacijas, siekiant užtikrinti patikimumą ir saugumą.
- Suprasite, kodėl būtina kurti įtraukias sistemas, kurios suteikia galimybes visiems.
- Išnagrinėsite, kodėl svarbu apsaugoti duomenų ir žmonių privatumą bei saugumą.
- Suprasite, kodėl būtina turėti „stiklinės dėžės“ požiūrį, kad būtų galima paaiškinti dirbtinio intelekto modelių elgesį.
- Būsite sąmoningi, kodėl atsakomybė yra būtina, siekiant sukurti pasitikėjimą dirbtinio intelekto sistemomis.

## Privalomos žinios

Prieš pradedant, rekomenduojame peržiūrėti „Atsakingo dirbtinio intelekto principų“ mokymo kelią ir pažiūrėti žemiau pateiktą vaizdo įrašą šia tema:

Sužinokite daugiau apie atsakingą dirbtinį intelektą, sekdami šį [mokymo kelią](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft požiūris į atsakingą dirbtinį intelektą](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft požiūris į atsakingą dirbtinį intelektą")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte vaizdo įrašą: Microsoft požiūris į atsakingą dirbtinį intelektą

## Teisingumas

Dirbtinio intelekto sistemos turėtų elgtis teisingai su visais ir vengti skirtingo poveikio panašioms žmonių grupėms. Pavyzdžiui, kai dirbtinio intelekto sistemos teikia rekomendacijas dėl medicininio gydymo, paskolų paraiškų ar įdarbinimo, jos turėtų pateikti tas pačias rekomendacijas visiems, turintiems panašius simptomus, finansines aplinkybes ar profesinę kvalifikaciją. Kiekvienas iš mūsų, kaip žmonės, turime paveldėtus šališkumus, kurie veikia mūsų sprendimus ir veiksmus. Šie šališkumai gali atsispindėti duomenyse, kuriuos naudojame dirbtinio intelekto sistemų mokymui. Tokia manipuliacija kartais gali įvykti netyčia. Dažnai sunku sąmoningai suvokti, kada į duomenis įvedate šališkumą.

**„Netolygumas“** apima neigiamą poveikį arba „žalą“ žmonių grupei, pavyzdžiui, apibrėžtai pagal rasę, lytį, amžių ar negalios statusą. Pagrindinės su teisingumu susijusios žalos rūšys gali būti klasifikuojamos taip:

- **Paskirstymas**, kai, pavyzdžiui, lytis ar etninė grupė yra palankesnė už kitą.
- **Paslaugos kokybė**. Jei duomenys mokomi tik vienam konkrečiam scenarijui, tačiau realybė yra daug sudėtingesnė, tai lemia prastai veikiančią paslaugą. Pavyzdžiui, rankų muilo dozatorius, kuris negali aptikti žmonių su tamsia oda. [Nuoroda](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Menkinimas**. Nesąžiningai kritikuoti ir žymėti ką nors ar kažką. Pavyzdžiui, vaizdų žymėjimo technologija klaidingai pažymėjo tamsiaodžių žmonių nuotraukas kaip gorilas.
- **Perteklinis arba nepakankamas atstovavimas**. Idėja, kad tam tikra grupė nėra matoma tam tikroje profesijoje, o bet kokia paslauga ar funkcija, kuri toliau tai skatina, prisideda prie žalos.
- **Stereotipai**. Tam tikros grupės susiejimas su iš anksto priskirtais atributais. Pavyzdžiui, kalbos vertimo sistema tarp anglų ir turkų kalbų gali turėti netikslumų dėl žodžių, susijusių su stereotipiniais lyties ryšiais.

![vertimas į turkų kalbą](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> vertimas į turkų kalbą

![vertimas atgal į anglų kalbą](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> vertimas atgal į anglų kalbą

Kuriant ir testuojant dirbtinio intelekto sistemas, turime užtikrinti, kad dirbtinis intelektas būtų teisingas ir neprogramuotas priimti šališkų ar diskriminuojančių sprendimų, kurių žmonėms taip pat draudžiama priimti. Teisingumo užtikrinimas dirbtiniame intelekte ir mašininiame mokymesi išlieka sudėtingu sociotechniniu iššūkiu.

### Patikimumas ir saugumas

Norint sukurti pasitikėjimą, dirbtinio intelekto sistemos turi būti patikimos, saugios ir nuoseklios tiek įprastomis, tiek netikėtomis sąlygomis. Svarbu žinoti, kaip dirbtinio intelekto sistemos elgsis įvairiose situacijose, ypač kai jos yra išskirtinės. Kuriant dirbtinio intelekto sprendimus, reikia skirti daug dėmesio tam, kaip spręsti įvairias aplinkybes, su kuriomis dirbtinio intelekto sprendimai gali susidurti. Pavyzdžiui, savarankiškai vairuojantis automobilis turi prioritetą teikti žmonių saugumui. Todėl dirbtinis intelektas, valdantis automobilį, turi atsižvelgti į visas galimas situacijas, su kuriomis automobilis gali susidurti, tokias kaip naktis, audros, pūgos, vaikai, bėgantys per gatvę, augintiniai, kelio darbai ir pan. Kaip gerai dirbtinio intelekto sistema gali patikimai ir saugiai spręsti įvairias sąlygas, atspindi duomenų mokslininko ar dirbtinio intelekto kūrėjo numatymo lygį projektavimo ar testavimo metu.

> [🎥 Spustelėkite čia, kad peržiūrėtumėte vaizdo įrašą: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Įtrauktis

Dirbtinio intelekto sistemos turėtų būti sukurtos taip, kad įtrauktų ir suteiktų galimybes visiems. Kuriant ir įgyvendinant dirbtinio intelekto sistemas, duomenų mokslininkai ir dirbtinio intelekto kūrėjai identifikuoja ir sprendžia galimas kliūtis sistemoje, kurios netyčia galėtų išskirti žmones. Pavyzdžiui, pasaulyje yra 1 milijardas žmonių su negalia. Su dirbtinio intelekto pažanga jie gali lengviau pasiekti įvairią informaciją ir galimybes savo kasdieniame gyvenime. Sprendžiant kliūtis, atsiranda galimybės inovuoti ir kurti dirbtinio intelekto produktus su geresnėmis patirtimis, kurios naudingos visiems.

> [🎥 Spustelėkite čia, kad peržiūrėtumėte vaizdo įrašą: įtrauktis dirbtiniame intelekte](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Saugumas ir privatumas

Dirbtinio intelekto sistemos turėtų būti saugios ir gerbti žmonių privatumą. Žmonės mažiau pasitiki sistemomis, kurios kelia pavojų jų privatumui, informacijai ar gyvybei. Mokant mašininio mokymosi modelius, mes pasikliaujame duomenimis, kad gautume geriausius rezultatus. Tai darydami turime atsižvelgti į duomenų kilmę ir vientisumą. Pavyzdžiui, ar duomenys buvo pateikti vartotojų, ar viešai prieinami? Be to, dirbant su duomenimis, būtina kurti dirbtinio intelekto sistemas, kurios galėtų apsaugoti konfidencialią informaciją ir atsispirti atakoms. Kadangi dirbtinis intelektas tampa vis labiau paplitęs, privatumo apsauga ir svarbios asmeninės bei verslo informacijos saugumas tampa vis svarbesni ir sudėtingesni. Privatumo ir duomenų saugumo klausimai reikalauja ypatingo dėmesio dirbtiniam intelektui, nes duomenų prieiga yra būtina, kad dirbtinio intelekto sistemos galėtų tiksliai ir informuotai priimti sprendimus apie žmones.

> [🎥 Spustelėkite čia, kad peržiūrėtumėte vaizdo įrašą: saugumas dirbtiniame intelekte](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Pramonėje pasiekėme reikšmingų privatumo ir saugumo pažangų, kurias labai paskatino tokie reglamentai kaip GDPR (Bendrasis duomenų apsaugos reglamentas).
- Tačiau dirbtinio intelekto sistemose turime pripažinti įtampą tarp poreikio turėti daugiau asmeninių duomenų, kad sistemos būtų asmeniškesnės ir efektyvesnės, ir privatumo.
- Kaip ir su interneto atsiradimu, matome didelį saugumo problemų, susijusių su dirbtiniu intelektu, augimą.
- Tuo pačiu metu matome, kaip dirbtinis intelektas naudojamas saugumui gerinti. Pavyzdžiui, dauguma modernių antivirusinių skenerių šiandien yra pagrįsti dirbtinio intelekto heuristika.
- Turime užtikrinti, kad mūsų duomenų mokslų procesai harmoningai derėtų su naujausiomis privatumo ir saugumo praktikomis.

### Skaidrumas

Dirbtinio intelekto sistemos turėtų būti suprantamos. Svarbi skaidrumo dalis yra paaiškinti dirbtinio intelekto sistemų ir jų komponentų elgesį. Gerinant dirbtinio intelekto sistemų supratimą, reikia, kad suinteresuotosios šalys suprastų, kaip ir kodėl jos veikia, kad galėtų identifikuoti galimas veikimo problemas, saugumo ir privatumo rūpesčius, šališkumus, išskirtines praktikas ar netikėtus rezultatus. Taip pat tikime, kad tie, kurie naudoja dirbtinio intelekto sistemas, turėtų būti sąžiningi ir atviri apie tai, kada, kodėl ir kaip jie nusprendžia jas naudoti. Taip pat apie naudojamų sistemų apribojimus. Pavyzdžiui, jei bankas naudoja dirbtinio intelekto sistemą, kad palaikytų vartotojų paskolų sprendimus, svarbu išnagrinėti rezultatus ir suprasti, kurie duomenys daro įtaką sistemos rekomendacijoms. Vyriausybės pradeda reguliuoti dirbtinį intelektą įvairiose pramonės šakose, todėl duomenų mokslininkai ir organizacijos turi paaiškinti, ar dirbtinio intelekto sistema atitinka reglamentavimo reikalavimus, ypač kai yra nepageidaujamas rezultatas.

> [🎥 Spustelėkite čia, kad peržiūrėtumėte vaizdo įrašą: skaidrumas dirbtiniame intelekte](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kadangi dirbtinio intelekto sistemos yra tokios sudėtingos, sunku suprasti, kaip jos veikia ir interpretuoti rezultatus.
- Šis supratimo trūkumas veikia tai, kaip šios sistemos yra valdomos, operatyviai naudojamos ir dokumentuojamos.
- Šis supratimo trūkumas dar svarbiau veikia sprendimus, priimamus remiantis šių sistemų rezultatais.

### Atsakomybė

Žmonės, kurie kuria ir diegia dirbtinio intelekto sistemas, turi būti atsakingi už tai, kaip jų sistemos veikia. Atsakomybės poreikis yra ypač svarbus jautrių technologijų, tokių kaip veido atpažinimas, naudojimo atveju. Pastaruoju metu didėja veido atpažinimo technologijos paklausa, ypač iš teisėsaugos organizacijų, kurios mato technologijos potencialą, pavyzdžiui, ieškant dingusių vaikų. Tačiau šios technologijos galėtų būti naudojamos vyriausybių, kad būtų pažeistos piliečių pagrindinės laisvės, pavyzdžiui, leidžiant nuolatinę tam tikrų asmenų stebėseną. Todėl duomenų mokslininkai ir organizacijos turi būti atsakingi už tai, kaip jų dirbtinio intelekto sistema veikia žmones ar visuomenę.

[![Pirmaujantis dirbtinio intelekto tyrėjas įspėja apie masinę stebėseną per veido atpažinimą](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft požiūris į atsakingą dirbtinį intelektą")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte vaizdo įrašą: Įspėjimai apie masinę stebėseną per veido atpažinimą

Galų gale vienas didžiausių mūsų kartos klausimų, kaip pirmosios kartos, kuri įveda dirbtinį intelektą į visuomenę, yra tai, kaip užtikrinti, kad kompiuteriai išliktų atsakingi žmonėms ir kaip užtikrinti, kad žmonės, kurie kuria kompiuterius, išliktų atsakingi visiems kitiems.

## Poveikio vertinimas

Prieš mokant mašininio mokymosi modelį, svarbu atlikti poveikio vertinimą, kad suprastumėte dirbtinio intelekto sistemos tikslą; kokia yra numatyta paskirtis; kur ji bus naudojama; ir kas sąveikaus su sistema. Tai naudinga vertintojams ar testuotojams, kad jie žinotų, kokius veiksnius reikia apsvarstyti identifikuojant galimas rizikas ir numatomas pasekmes.

Štai sritys, į kurias reikia atkreipti dėmesį atliekant poveikio vertinimą:

* **Neigiamas poveikis asmenims**. Svarbu žinoti apie bet kokius apribojimus
Žiūrėkite šį seminarą, kad giliau suprastumėte temas:

- Atsakingo dirbtinio intelekto siekimas: principų įgyvendinimas praktikoje, pristato Besmira Nushi, Mehrnoosh Sameki ir Amit Sharma

[![Atsakingo DI įrankių rinkinys: atvirojo kodo sistema atsakingam DI kurti](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Atvirojo kodo sistema atsakingam DI kurti")

> 🎥 Spustelėkite paveikslėlį aukščiau, kad peržiūrėtumėte vaizdo įrašą: RAI Toolbox: Atvirojo kodo sistema atsakingam DI kurti, pristato Besmira Nushi, Mehrnoosh Sameki ir Amit Sharma

Taip pat skaitykite:

- Microsoft atsakingo DI išteklių centras: [Atsakingo DI ištekliai – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft FATE tyrimų grupė: [FATE: Sąžiningumas, Atsakomybė, Skaidrumas ir Etika DI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI įrankių rinkinys:

- [Atsakingo DI įrankių rinkinio GitHub saugykla](https://github.com/microsoft/responsible-ai-toolbox)

Skaitykite apie Azure Machine Learning įrankius, skirtus užtikrinti sąžiningumą:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Užduotis

[Susipažinkite su RAI įrankių rinkiniu](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama profesionali žmogaus vertimo paslauga. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.