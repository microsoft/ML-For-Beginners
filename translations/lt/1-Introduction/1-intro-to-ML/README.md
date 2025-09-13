<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T07:56:36+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "lt"
}
-->
# Įvadas į mašininį mokymąsi

## [Prieš paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pradedantiesiems - Įvadas į mašininį mokymąsi pradedantiesiems](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pradedantiesiems - Įvadas į mašininį mokymąsi pradedantiesiems")

> 🎥 Spustelėkite paveikslėlį aukščiau, kad peržiūrėtumėte trumpą vaizdo įrašą apie šią pamoką.

Sveiki atvykę į šį kursą apie klasikinį mašininį mokymąsi pradedantiesiems! Nesvarbu, ar esate visiškai naujas šioje srityje, ar patyręs ML specialistas, norintis atnaujinti žinias, džiaugiamės, kad prisijungėte! Norime sukurti draugišką starto vietą jūsų ML studijoms ir mielai įvertinsime, atsakysime bei įtrauksime jūsų [atsiliepimus](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Įvadas į ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Įvadas į ML")

> 🎥 Spustelėkite paveikslėlį aukščiau, kad peržiūrėtumėte vaizdo įrašą: MIT profesorius John Guttag pristato mašininį mokymąsi.

---
## Pradžia su mašininiu mokymusi

Prieš pradėdami šį mokymo planą, turite paruošti savo kompiuterį darbui su vietiniais užrašais.

- **Konfigūruokite savo kompiuterį naudodamiesi šiais vaizdo įrašais**. Naudokite šias nuorodas, kad sužinotumėte, [kaip įdiegti Python](https://youtu.be/CXZYvNRIAKM) savo sistemoje ir [nustatyti teksto redaktorių](https://youtu.be/EU8eayHWoZg) programavimui.
- **Išmokite Python**. Taip pat rekomenduojama turėti pagrindines [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) žinias – tai programavimo kalba, naudinga duomenų mokslininkams, kurią naudosime šiame kurse.
- **Išmokite Node.js ir JavaScript**. Šiame kurse kelis kartus naudosime JavaScript kuriant interneto programas, todėl jums reikės turėti [node](https://nodejs.org) ir [npm](https://www.npmjs.com/) įdiegtus, taip pat [Visual Studio Code](https://code.visualstudio.com/) Python ir JavaScript programavimui.
- **Sukurkite GitHub paskyrą**. Kadangi mus radote čia, [GitHub](https://github.com), galbūt jau turite paskyrą, bet jei ne, sukurkite ją ir tada nukopijuokite šį mokymo planą, kad galėtumėte naudoti jį savarankiškai. (Taip pat galite mums suteikti žvaigždutę 😊)
- **Susipažinkite su Scikit-learn**. Susipažinkite su [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), ML bibliotekų rinkiniu, kurį naudosime šiose pamokose.

---
## Kas yra mašininis mokymasis?

Terminas „mašininis mokymasis“ yra vienas populiariausių ir dažniausiai vartojamų šiandien. Yra nemaža tikimybė, kad bent kartą girdėjote šį terminą, jei turite kokį nors ryšį su technologijomis, nesvarbu, kokioje srityje dirbate. Tačiau mašininio mokymosi mechanika daugeliui žmonių yra paslaptis. Pradedančiajam mašininio mokymosi tema kartais gali atrodyti bauginanti. Todėl svarbu suprasti, kas iš tikrųjų yra mašininis mokymasis, ir mokytis apie jį žingsnis po žingsnio, naudojant praktinius pavyzdžius.

---
## Populiarumo kreivė

![ml populiarumo kreivė](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends rodo naujausią termino „mašininis mokymasis“ populiarumo kreivę.

---
## Paslaptinga visata

Mes gyvename visatoje, pilnoje įdomių paslapčių. Didieji mokslininkai, tokie kaip Stephenas Hawkingas, Albertas Einšteinas ir daugelis kitų, paskyrė savo gyvenimus ieškodami prasmingos informacijos, kuri atskleistų pasaulio aplink mus paslaptis. Tai yra žmogaus mokymosi būklė: žmogaus vaikas mokosi naujų dalykų ir kasmet, augdamas iki pilnametystės, atskleidžia savo pasaulio struktūrą.

---
## Vaiko smegenys

Vaiko smegenys ir pojūčiai suvokia aplinkos faktus ir palaipsniui išmoksta paslėptus gyvenimo modelius, kurie padeda vaikui sukurti logines taisykles, leidžiančias atpažinti išmoktus modelius. Žmogaus smegenų mokymosi procesas daro žmones pačiais sudėtingiausiais gyvais padarais šiame pasaulyje. Nuolatinis mokymasis, atrandant paslėptus modelius ir vėliau juos tobulinant, leidžia mums visą gyvenimą tobulėti. Šis mokymosi gebėjimas ir evoliucinis pajėgumas yra susijęs su sąvoka, vadinama [smegenų plastiškumu](https://www.simplypsychology.org/brain-plasticity.html). Paviršutiniškai galime rasti motyvacinių panašumų tarp žmogaus smegenų mokymosi proceso ir mašininio mokymosi koncepcijų.

---
## Žmogaus smegenys

[Žmogaus smegenys](https://www.livescience.com/29365-human-brain.html) suvokia dalykus iš realaus pasaulio, apdoroja suvoktą informaciją, priima racionalius sprendimus ir atlieka tam tikrus veiksmus pagal aplinkybes. Tai vadiname protingu elgesiu. Kai programuojame mašiną, kad ji imituotų protingo elgesio procesą, tai vadinama dirbtiniu intelektu (AI).

---
## Kai kurie terminai

Nors terminai gali būti painūs, mašininis mokymasis (ML) yra svarbi dirbtinio intelekto dalis. **ML susijęs su specializuotų algoritmų naudojimu, siekiant atskleisti prasmingą informaciją ir rasti paslėptus modelius iš suvoktų duomenų, kad būtų patvirtintas racionalaus sprendimų priėmimo procesas**.

---
## AI, ML, gilus mokymasis

![AI, ML, gilus mokymasis, duomenų mokslas](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Diagrama, rodanti AI, ML, gilaus mokymosi ir duomenų mokslo ryšius. Infografiką sukūrė [Jen Looper](https://twitter.com/jenlooper), įkvėpta [šios grafikos](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining).

---
## Temos, kurias aptarsime

Šiame mokymo plane aptarsime tik pagrindines mašininio mokymosi koncepcijas, kurias pradedantysis turi žinoti. Daugiausia dėmesio skirsime „klasikiniam mašininiam mokymuisi“, naudodami Scikit-learn – puikią biblioteką, kurią daugelis studentų naudoja mokydamiesi pagrindų. Norint suprasti platesnes dirbtinio intelekto ar gilaus mokymosi koncepcijas, būtina turėti stiprias mašininio mokymosi pagrindines žinias, kurias norime pasiūlyti čia.

---
## Šiame kurse išmoksite:

- pagrindines mašininio mokymosi koncepcijas
- ML istoriją
- ML ir sąžiningumą
- regresijos ML technikas
- klasifikacijos ML technikas
- klasterizacijos ML technikas
- natūralios kalbos apdorojimo ML technikas
- laiko eilučių prognozavimo ML technikas
- pastiprinimo mokymąsi
- realaus pasaulio ML taikymus

---
## Ko neaptarsime

- gilus mokymasis
- neuroniniai tinklai
- AI

Siekiant geresnės mokymosi patirties, vengsime neuroninių tinklų sudėtingumo, „gilaus mokymosi“ – daugiapakopio modelių kūrimo naudojant neuroninius tinklus – ir AI, kurį aptarsime kitame mokymo plane. Taip pat pasiūlysime būsimą duomenų mokslo mokymo planą, skirtą šiam platesnės srities aspektui.

---
## Kodėl verta studijuoti mašininį mokymąsi?

Mašininis mokymasis, iš sistemų perspektyvos, apibrėžiamas kaip automatizuotų sistemų kūrimas, kurios gali išmokti paslėptus modelius iš duomenų, kad padėtų priimti protingus sprendimus.

Ši motyvacija laisvai įkvėpta to, kaip žmogaus smegenys mokosi tam tikrų dalykų, remdamiesi duomenimis, kuriuos suvokia iš išorinio pasaulio.

✅ Pagalvokite minutę, kodėl verslas norėtų naudoti mašininio mokymosi strategijas, o ne kurti griežtai užkoduotą taisyklių sistemą.

---
## Mašininio mokymosi taikymas

Mašininio mokymosi taikymas dabar yra beveik visur ir toks pat paplitęs kaip duomenys, kurie cirkuliuoja mūsų visuomenėse, generuojami mūsų išmaniųjų telefonų, prijungtų įrenginių ir kitų sistemų. Atsižvelgiant į pažangiausių mašininio mokymosi algoritmų potencialą, mokslininkai tyrinėja jų gebėjimą spręsti daugiadimensines ir daugiadisciplinines realaus gyvenimo problemas su puikiais rezultatais.

---
## Taikomo ML pavyzdžiai

**Mašininį mokymąsi galite naudoti įvairiais būdais**:

- Prognozuoti ligos tikimybę pagal paciento medicininę istoriją ar ataskaitas.
- Naudoti orų duomenis, kad prognozuotumėte orų įvykius.
- Suprasti teksto nuotaiką.
- Aptikti netikras naujienas, kad sustabdytumėte propagandos plitimą.

Finansai, ekonomika, žemės mokslai, kosmoso tyrimai, biomedicinos inžinerija, kognityviniai mokslai ir net humanitariniai mokslai pritaikė mašininį mokymąsi, kad išspręstų sudėtingas, duomenų apdorojimo reikalaujančias savo srities problemas.

---
## Išvada

Mašininis mokymasis automatizuoja modelių atradimo procesą, surandant prasmingas įžvalgas iš realaus pasaulio ar generuotų duomenų. Jis pasirodė esąs labai vertingas verslo, sveikatos ir finansų srityse, be kitų.

Artimiausioje ateityje mašininio mokymosi pagrindų supratimas taps būtinybe žmonėms iš bet kurios srities dėl jo plačiai paplitusio pritaikymo.

---
# 🚀 Iššūkis

Nupieškite, popieriuje arba naudodamiesi internetine programa, pvz., [Excalidraw](https://excalidraw.com/), savo supratimą apie AI, ML, gilaus mokymosi ir duomenų mokslo skirtumus. Pridėkite idėjų apie problemas, kurias kiekviena iš šių technikų gerai sprendžia.

# [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

---
# Apžvalga ir savarankiškas mokymasis

Norėdami sužinoti daugiau apie tai, kaip galite dirbti su ML algoritmais debesyje, sekite šį [mokymosi kelią](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Paimkite [mokymosi kelią](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) apie ML pagrindus.

---
# Užduotis

[Pradėkite darbą](assignment.md)

---

**Atsakomybės atsisakymas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama profesionali žmogaus vertimo paslauga. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius naudojant šį vertimą.