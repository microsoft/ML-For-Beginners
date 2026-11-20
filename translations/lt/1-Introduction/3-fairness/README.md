# Atsakingai kuriant mašininio mokymosi sprendimus naudojant dirbtinį intelektą
 
![Atsakingo DI mašininio mokymosi santrauka sketchnote formatu](../../../../translated_images/lt/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote autorius [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pradinis viktorinos testas](https://ff-quizzes.netlify.app/en/ml/)
 
## Įvadas

Šiame mokymo plane pradėsite atrasti, kaip mašininis mokymasis gali ir veikia mūsų kasdienį gyvenimą. Jau dabar sistemos ir modeliai dalyvauja kasdienėse sprendimų priėmimo užduotyse, tokiose kaip sveikatos priežiūros diagnozės, paskolų suteikimas ar sukčiavimo aptikimas. Todėl svarbu, kad šie modeliai veiktų gerai ir teiktų patikimus rezultatus. Kaip ir bet kuri programinė įranga, DI sistemos gali neatitikti lūkesčių arba duoti nepageidaujamą rezultatą. Todėl būtina suprasti ir paaiškinti DI modelio elgesį. 

Įsivaizduokite, kas gali nutikti, jei naudojami duomenys, kuriais kuriami šie modeliai, trūksta tam tikrų demografinių grupių, pavyzdžiui, rasės, lyties, politinių pažiūrų, religijos, arba kurios disproporcingai atstovauja tam tikras demografines grupes. Kas nutiks, kai modelio išvestis interpretuojama palankiai kuriam nors demografiniam ratui? Kokia yra pasekmė taikymui? Be to, kas nutinka, kai modelis duoda neigiamą rezultatą ir kenkia žmonėms? Kas yra atsakingas už DI sistemos elgesį? Šiuos klausimus aptartos šiame mokymo plane. 

Šioje pamokoje jūs:

- Suprasite teisingumo svarbą mašininiame mokyme ir su teisingumu susijusias žalas.
- Susipažinsite su praktika, kaip tirti netipinius atvejus ir neįprastas situacijas, kad užtikrintumėte patikimumą ir saugumą.
- Suprasite poreikį įgalinti visus kurti įtraukią sistemas.
- Išnagrinėsite, kaip svarbu saugoti duomenų ir žmonių privatumą bei saugumą.
- Matysite svarbą taikyti skaidraus (glass box) modelio požiūrį AI modelių elgesiui paaiškinti.
- Būsime atidūs, kaip atsakomybė yra būtina kuriant pasitikėjimą DI sistemomis.

## Prieš sąlygų įvertinimas

Prieš pradedant, prašome išklausyti "Atsakingo DI principus" mokymosi kelią ir peržiūrėti žemiau pateiktą vaizdo įrašą šia tema:

Sužinokite daugiau apie atsakingą DI sekdami šį [Mokymosi kelią](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![„Microsoft“ požiūris į atsakingą DI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "„Microsoft“ požiūris į atsakingą DI")

> 🎥 Paspauskite paveikslėlį aukščiau, kad žiūrėtumėte vaizdo įrašą: „Microsoft“ požiūris į atsakingą DI

## Teisingumas

DI sistemos turi visus elgtis teisingai ir vengti skirtingai paveikti panašias žmonių grupes. Pavyzdžiui, kai DI sistemos teikia patarimus dėl medicininio gydymo, paskolų prašymų ar darbo, jos turi daryti tas pačias rekomendacijas visiems su panašiais simptomais, finansine padėtimi ar profesine kvalifikacija. Kiekvienas iš mūsų kaip žmonės turi paveldėtus šališkumus, kurie veikia mūsų sprendimus ir veiksmus. Šie šališkumai gali būti matomi duomenyse, kuriais treniruojamos DI sistemos. Kartais tai nutinka neintencionaliai. Dažnai sunku sąmoningai žinoti, kada duomenyse pristatote šališkumą. 

**„Neteisingumas“** apima neigiamas pasekmes arba „žalas“ grupėms žmonių, pavyzdžiui, apibrėžtoms pagal rasę, lytį, amžių ar neįgalumo statusą. Pagrindines su teisingumu susijusias žalas galima priskirti: 

- **Skirstymas**, kai, pavyzdžiui, lyčiai arba tautybei teikiama pirmenybė prieš kitą.
- **Paslaugos kokybė**. Jei duomenys yra treniruojami vienai specifinei situacijai, bet realybė yra daug sudėtingesnė, tai lemia prastą veikimą. Pavyzdžiui, skysto muilo dozatorius, kuris negalėjo aptikti žmonių su tamsesne oda. [Nuoroda](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Šmeižimas**. Nesąžiningas kritika ir pažymėjimas kažką ar kažką. Pavyzdžiui, vaizdų paženklinimo technologija garsiai neteisingai pažymėjo tamsiaodžių žmonių nuotraukas kaip gorilas.
- **Per didelė arba per maža atstovybė**. Idėja yra ta, kad tam tikra grupė tam tikroje profesijoje nėra matoma, o bet kokia paslauga ar funkcija, kuri tą skatintų, prisideda prie žalos.
- **Stereotipavimas**. Priskiriant grupei iš anksto nustatytas savybes. Pavyzdžiui, tarp anglų ir turkų kalbų vertimo sistemos gali būti netikslumų dėl žodžių, turinčių stereotipinių asociacijų su lytimi.

![vertimas į turkų kalbą](../../../../translated_images/lt/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> vertimas į turkų kalbą

![vertimas atgal į anglų kalbą](../../../../translated_images/lt/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> vertimas atgal į anglų kalbą

Kuriant ir testuojant DI sistemas, turime užtikrinti, kad DI būtų teisingas ir nebūtų programuojamas priimti šališkus ar diskriminacinius sprendimus, kuriuos taip pat draudžiama priimti žmonėms. Užtikrinti teisingumą DI ir mašininio mokymosi srityje išlieka sudėtinga sociotechninė užduotis. 

### Patikimumas ir saugumas

Kad būtų pasitikima, DI sistemos turi būti patikimos, saugios ir nuoseklios tiek įprastomis, tiek netikėtomis sąlygomis. Svarbu žinoti, kaip DI sistemos elgsis įvairiose situacijose, ypač kai jos pateks į išskirtinius atvejus. Kuriant DI sprendimus, reikia daug dėmesio skirti, kaip spręsti platų dėmesio reikalaujančių aplinkybių spektrą, su kuriomis DI sprendimai susidurs. Pavyzdžiui, savavaldė mašina turi aukščiausiu prioritetu laikyti žmonių saugumą. Todėl mašinos veikimą valdantis DI turi atsižvelgti į visas galimas situacijas, kurias gali patirti automobilis, tokias kaip naktis, perkūnijos ar pūgos, vaikų bėgimas per gatvę, naminių gyvūnų buvimas, kelio darbai ir t.t. Koks geras DI sistema gali patikimai ir saugiai tvarkyti įvairiausias sąlygas, rodo, kiek duomenų mokslininkas ar DI kūrėjas numatė projektuodamas ar testuodamas sistemą.  

> [🎥 Spauskite čia žiūrėti vaizdo įrašą: Patikimumas ir saugumas DI srityje](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Įtrauktis

DI sistemos turi būti kuriamos taip, kad bet kuris žmogus galėtų dalyvauti ir jaustųsi įgalintas. Kuriant ir įgyvendinant DI sistemas, duomenų mokslininkai ir DI kūrėjai identifikuoja ir sprendžia galimas kliūtis, kurios netyčia gali atleisti žmones. Pavyzdžiui, pasaulyje yra 1 milijardas neįgaliųjų. Su DI pažanga jie gali lengviau prieiti prie įvairios informacijos ir galimybių kasdieniame gyvenime. Sprendžiant kliūtis, sukuriamos galimybės kurti inovacijas ir kurti DI produktus su geresne patirtimi, kurie yra naudingi visiems. 

> [🎥 Spauskite čia žiūrėti vaizdo įrašą: Įtrauktis DI srityje](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Saugumas ir privatumas

DI sistemos turi būti saugios ir gerbti žmonių privatumą. Žmonės mažiau pasitiki sistemomis, kurios kelia grėsmę jų privatumui, informacijos saugumui ar gyvybei. Treniruodami mašininio mokymosi modelius, mes remiamės duomenimis, kad gautume geriausius rezultatus. Tai darant, būtina atsižvelgti į duomenų kilmę ir vientisumą. Pavyzdžiui, ar duomenys buvo pateikti naudotojo ar viešai prieinami? Darbe su duomenimis svarbu kurti DI sistemas, galinčias apsaugoti konfidencialią informaciją ir atlaikyti atakas. DI plintant, privatumo apsauga ir svarbios asmeninės bei verslo informacijos saugumas tampa itin svarbūs ir sudėtingi. Privatumo ir duomenų saugumo klausimai reikalauja ypatingo dėmesio DI srityje, nes prieiga prie duomenų yra būtina, kad DI sistemos galėtų tiksliai ir informuotai prognozuoti ir priimti sprendimus apie žmones. 

> [🎥 Spauskite čia žiūrėti vaizdo įrašą: Saugumas DI srityje](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Pramonės sektoriuje padarėme didelių pažangų privatumo ir saugumo srityje, kurias ženkliai paskatino tokie reglamentai kaip GDPR (Bendrasis duomenų apsaugos reglamentas).
- Vis dėlto DI sistemose privalome pripažinti įtampą tarp poreikio turėti daugiau asmeninės informacijos, kad sistemos taptų asmeniškesnės ir veiksmingesnės, ir privatumo.
- Kaip ir prisijungusių kompiuterių su internetu gimimo metu, dabar taip pat matome didelį saugumo problemų skaičiaus augimą, susijusį su DI.
- Tuo pačiu metu mes matome, kaip DI naudojamas saugumui gerinti. Pavyzdžiui, dauguma modernių antivirusų šiandien veikdami naudoja DI euristikas.
- Turime užtikrinti, kad mūsų duomenų mokslo procesai harmoningai derėtųsi su naujausiomis privatumo ir saugumo praktikomis.


### Skaidrumas
DI sistemos turi būti suprantamos. Svarbi skaidrumo dalis yra paaiškinti DI sistemų ir jų sudedamųjų dalių elgesį. Glaudesnis DI sistemų supratimas reikalauja, kad suinteresuotos šalys suprastų, kaip ir kodėl jos veikia, kad galėtų nustatyti galimas našumo problemas, saugumo ir privatumo rūpesčius, šališkumus, atmetimo praktikas ar nepageidaujamus rezultatus. Mes taip pat manome, kad DI sistemų naudotojai turėtų būti sąžiningi ir atviri, kada, kodėl ir kaip jie nusprendžia jas diegti. Taip pat apie naudotų sistemų ribotumus. Pavyzdžiui, jei bankas naudoja DI sistemą remti vartotojų paskolų sprendimus, svarbu patikrinti rezultatus ir suprasti, kokie duomenys veikia sistemos rekomendacijas. Vyriausybės pradėjo reguliuoti DI įvairiose pramonės šakose, todėl duomenų mokslininkai ir organizacijos turi paaiškinti, ar DI sistema atitinka reguliavimo reikalavimus, ypač jei atsiranda nepageidaujamas rezultatas.

> [🎥 Spauskite čia žiūrėti vaizdo įrašą: Skaidrumas DI srityje](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kadangi DI sistemos yra tokios sudėtingos, sunku suprasti, kaip jos veikia, ir interpretuoti rezultatus.
- Šis suvokimo trūkumas veikia tai, kaip šios sistemos yra valdomos, įgyvendinamos ir dokumentuojamos.
- Dar svarbiau, šis supratimo trūkumas veikia sprendimus, priimamų remiantis gautais rezultatais.

### Atsakomybė
 
Žmonės, kurie kuria ir diegia DI sistemas, turi būti atsakingi už savo sistemų veikimą. Atsakomybės būtinybė ypač svarbi su jautriomis technologijomis, tokiomis kaip veidų atpažinimas. Pastaruoju metu didėja paklausa veidų atpažinimo technologijai, ypač iš teisėsaugos institucijų, kurios mato šios technologijos potencialą, pavyzdžiui, ieškant dingusių vaikų. Tačiau šios technologijos potencialiai gali būti naudojamos vyriausybės siekiant kelti pavojų piliečių pagrindinėms laisvėms, pavyzdžiui, įgalinant nuolatinį konkrečių asmenų stebėjimą. Todėl duomenų mokslininkai ir organizacijos turi būti atsakingi už tai, kaip jų DI sistema veikia žmones ar visuomenę.

[![Pirmaujantis DI tyrėjas įspėja apie masinį stebėjimą naudojant veidų atpažinimą](../../../../translated_images/lt/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "„Microsoft“ požiūris į atsakingą DI")

> 🎥 Paspauskite paveikslėlį aukščiau, kad žiūrėtumėte vaizdo įrašą: Įspėjimai apie masinį stebėjimą veidų atpažinimo dėka

Galų gale viena didžiausių mūsų kartos klausimų, kaip pirmosios kartos, diegiančios DI visuomenėje, yra tai, kaip užtikrinti, kad kompiuteriai liktų atsakingi žmonėms ir kaip užtikrinti, kad kompiuterius kuriančios žmonės liktų atsakingi visiems kitiems.

## Poveikio vertinimas

Prieš treniruojant mašininio mokymosi modelį, svarbu atlikti poveikio vertinimą, kad būtų suprasta DI sistemos paskirtis; koks yra numatytas naudojimas; kur ji bus diegiama; ir kas sąveikaus su sistema. Tai padeda peržiūrėtojams ar testuotojams įvertinti sistemą ir žinoti, kokius veiksnius reikia atsižvelgti nustatant galimas rizikas ir numatomas pasekmes.

Poveikio vertinimo metu reikėtų sutelkti dėmesį į šias sritis:

* **Neigiamas poveikis asmenims**. Būtina žinoti apie bet kokius apribojimus ar reikalavimus, palaikomą ar nepalaikomą naudojimą arba žinomus sistemos atlikimo apribojimus, kad sistema nebūtų naudojama būdu, galinčiu pakenkti asmenims.
* **Duomenų reikalavimai**. Supratimas, kaip ir kur sistema naudos duomenis, leidžia peržiūrėtojams įvertinti bet kokius duomenų reikalavimus, kurių reikia laikytis (pvz., GDPR ar HIPAA duomenų reglamentus). Taip pat reikėtų įvertinti, ar duomenų šaltinis ir kiekis yra pakankamas treniruotei.
* **Santrauka apie poveikį**. Surinkite galimų žalos atvejų, kurie gali kilti naudojant sistemą, sąrašą. Per visą ML gyvavimo ciklą peržiūrėkite, ar identifikuotos problemos yra sumažinamos arba sprendžiamos.
* **Taikytini tikslai** kiekvienam iš šešių pagrindinių principų. Įvertinkite, ar principų tikslai yra pasiekti ir ar yra spragų.


## Debug'inimas su atsakingu DI  

Panašiai kaip ir programų klaidų taisymas (debug'inimas), DI sistemos debug'inimas yra būtinas procesas, skirtas identifikuoti ir išspręsti problemas sistemoje. Daugelis veiksnių gali paveikti modelio neveikimą pagal lūkesčius ar atsakingumo principus. Dauguma tradicinių modelių našumo metrikų yra kiekybiniai agregatai, kurie nepakanka analizuoti, kaip modelis pažeidžia atsakingo DI principus. Be to, mašininio mokymosi modelis yra juodoji dėžė, todėl sunku suprasti, kas lemia jo rezultatą ar paaiškinti klaidas. Vėliau šiame kurse sužinosime, kaip naudotis Atsakingo DI informacijos suvestine, kuri padeda debug'inti DI sistemas. Ši suvestinė suteikia visapusišką įrankį duomenų mokslininkams ir DI kūrėjams atlikti:

* **Klaidų analizę**. Norint nustatyti modelio klaidų pasiskirstymą, galintį įtakoti sistemos teisingumą ar patikimumą.
* **Modelio apžvalgą**. Norint aptikti, kur duomenų grupėse yra skirtumų modelio veikime.
* **Duomenų analizę**. Norint suprasti duomenų pasiskirstymą ir identifikuoti galimus šališkumus, galinčius sukelti teisingumo, įtraukties ir patikimumo problemas.
* **Modelio interpretuojamumą**. Norint suprasti, kas veikia ar įtakoja modelio prognozes. Tai padeda paaiškinti modelio elgesį, kas yra svarbu skaidrumui ir atsakomybei.


## 🚀 Iššūkis 
 
Siekdami išvengti žalos atsiradimo iš karto turėtume:

- turėti įvairių kilmių ir perspektyvų žmones, dirbančius su sistemomis.
- investuoti į duomenų rinkinius, atspindinčius mūsų visuomenės įvairovę.
- tobulinti metodus per visą mašininio mokymosi gyvavimo ciklą, skirtus nepilnavertiško DI aptikimui ir taisymui, kai to reikia.

Pagalvokite apie realaus gyvenimo situacijas, kuriose modelio nepatikimumas aiškiai matomas kuriant ir naudojant modelį. Ką dar reikėtų apsvarstyti?

## [Baigiamoji viktorina](https://ff-quizzes.netlify.app/en/ml/)

## Peržiūra ir savarankiškas mokymasis  
 
Šioje pamokoje jūs sužinojote pagrindinius teisingumo ir neteisingumo mašininiame mokyme principus.  
 
Peržiūrėkite šią dirbtuvę, kad giliau įsigilintumėte į temas: 

- Siekdami atsakingo DI: Principų įgyvendinimas, autorių Besmira Nushi, Mehrnoosh Sameki ir Amit Sharma

[![Atsakingo DI rinkinys: Atviro kodo sistema atsakingam DI kūrimui](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "Atsakingo DI rinkinys: Atviro kodo sistema atsakingam DI kūrimui")

> 🎥 Spustelėkite viršutinį paveikslėlį, jei norite peržiūrėti vaizdo įrašą: RAI Toolbox: Atviro kodo sistema atsakingam DI kūrimui, autoriai Besmira Nushi, Mehrnoosh Sameki ir Amit Sharma

Taip pat skaitykite:

- „Microsoft“ atsakingo DI išteklių centras: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- „Microsoft“ FATE tyrimų grupė: [FATE: Teisingumas, Atsakomybė, Skaidrumas ir Etika DI srityje – Microsoft Research](https://www.microsoft.com/research/theme/fate/)

Atsakingo DI rinkinys:

- [Atsakingo DI rinkinio „GitHub“ saugykla](https://github.com/microsoft/responsible-ai-toolbox)

Skaitykite apie „Azure Machine Learning“ įrankius, skirtus užtikrinti teisingumą:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Užduotis

[Išbandykite Atsakingo DI rinkinį](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba laikomas autoritetingu šaltiniu. Svarbiai informacijai rekomenduojama naudoti profesionalų žmogiškąjį vertimą. Mes neatsakome už jokius nesusipratimus ar neteisingą interpretaciją, kilusią naudojantis šiuo vertimu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->