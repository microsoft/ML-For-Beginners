<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T07:55:51+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "lt"
}
-->
# Mašininio mokymosi technikos

Mašininio mokymosi modelių kūrimo, naudojimo ir palaikymo procesas bei duomenys, kuriuos jie naudoja, labai skiriasi nuo daugelio kitų kūrimo darbo eigų. Šioje pamokoje mes išsklaidysime šį procesą ir apžvelgsime pagrindines technikas, kurias turite žinoti. Jūs:

- Suprasite pagrindinius procesus, kuriais grindžiamas mašininis mokymasis.
- Išnagrinėsite pagrindines sąvokas, tokias kaip „modeliai“, „prognozės“ ir „mokymo duomenys“.

## [Prieš paskaitą – testas](https://ff-quizzes.netlify.app/en/ml/)

[![ML pradedantiesiems - Mašininio mokymosi technikos](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML pradedantiesiems - Mašininio mokymosi technikos")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie šią pamoką.

## Įvadas

Aukštu lygiu mašininio mokymosi (ML) procesų kūrimas apima kelis žingsnius:

1. **Nuspręskite, kokį klausimą norite užduoti**. Dauguma ML procesų prasideda nuo klausimo, į kurį negalima atsakyti paprasta sąlygine programa ar taisyklėmis pagrįstu varikliu. Šie klausimai dažnai susiję su prognozėmis, pagrįstomis duomenų rinkiniu.
2. **Surinkite ir paruoškite duomenis**. Norėdami atsakyti į savo klausimą, jums reikia duomenų. Duomenų kokybė ir, kartais, kiekis nulems, kaip gerai galėsite atsakyti į pradinį klausimą. Duomenų vizualizavimas yra svarbi šio etapo dalis. Šis etapas taip pat apima duomenų padalijimą į mokymo ir testavimo grupes, kad būtų galima sukurti modelį.
3. **Pasirinkite mokymo metodą**. Atsižvelgdami į savo klausimą ir duomenų pobūdį, turite pasirinkti, kaip norite mokyti modelį, kad jis geriausiai atspindėtų jūsų duomenis ir tiksliai prognozuotų pagal juos. Ši ML proceso dalis reikalauja specifinių žinių ir dažnai nemažai eksperimentavimo.
4. **Mokykite modelį**. Naudodami mokymo duomenis, taikysite įvairius algoritmus, kad išmokytumėte modelį atpažinti duomenų šablonus. Modelis gali naudoti vidinius svorius, kuriuos galima koreguoti, kad tam tikri duomenų aspektai būtų privilegijuoti, siekiant sukurti geresnį modelį.
5. **Įvertinkite modelį**. Naudojate anksčiau nematytus duomenis (testavimo duomenis) iš surinkto rinkinio, kad pamatytumėte, kaip modelis veikia.
6. **Parametrų derinimas**. Atsižvelgdami į modelio veikimą, galite pakartoti procesą naudodami skirtingus parametrus arba kintamuosius, kurie kontroliuoja algoritmų elgesį mokymo metu.
7. **Prognozuokite**. Naudokite naujus įvesties duomenis, kad patikrintumėte modelio tikslumą.

## Kokį klausimą užduoti

Kompiuteriai ypač gerai aptinka paslėptus duomenų šablonus. Ši savybė labai naudinga tyrėjams, turintiems klausimų apie tam tikrą sritį, į kuriuos negalima lengvai atsakyti sukuriant sąlyginių taisyklių variklį. Pavyzdžiui, aktuaro užduotyje duomenų mokslininkas galėtų sukurti rankiniu būdu sudarytas taisykles apie rūkalių ir nerūkalių mirtingumą.

Tačiau, kai į lygtį įtraukiama daug kitų kintamųjų, ML modelis gali būti efektyvesnis prognozuojant būsimus mirtingumo rodiklius, remiantis ankstesne sveikatos istorija. Džiugesnis pavyzdys galėtų būti orų prognozės balandžio mėnesiui tam tikroje vietovėje, remiantis duomenimis, įskaitant platumą, ilgumą, klimato pokyčius, artumą prie vandenyno, reaktyvinio srauto modelius ir kt.

✅ Ši [skaidrių prezentacija](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) apie orų modelius siūlo istorinę perspektyvą, kaip ML naudojamas orų analizei.

## Prieš modelio kūrimą

Prieš pradėdami kurti modelį, turite atlikti kelias užduotis. Norėdami patikrinti savo klausimą ir suformuoti hipotezę, pagrįstą modelio prognozėmis, turite identifikuoti ir sukonfigūruoti kelis elementus.

### Duomenys

Norėdami atsakyti į savo klausimą su tam tikru tikrumu, jums reikia pakankamo kiekio tinkamo tipo duomenų. Šiuo metu turite atlikti du dalykus:

- **Surinkti duomenis**. Atsižvelgdami į ankstesnę pamoką apie duomenų analizės sąžiningumą, rinkite duomenis atsargiai. Būkite sąmoningi apie šių duomenų šaltinius, galimus jų šališkumus ir dokumentuokite jų kilmę.
- **Paruošti duomenis**. Duomenų paruošimo procesas apima kelis žingsnius. Jums gali tekti sujungti duomenis ir normalizuoti juos, jei jie gaunami iš įvairių šaltinių. Duomenų kokybę ir kiekį galite pagerinti įvairiais būdais, pvz., konvertuodami tekstus į skaičius (kaip darome [Klasterizavime](../../5-Clustering/1-Visualize/README.md)). Taip pat galite generuoti naujus duomenis, remdamiesi originaliais (kaip darome [Klasifikacijoje](../../4-Classification/1-Introduction/README.md)). Galite išvalyti ir redaguoti duomenis (kaip darysime prieš [Tinklalapio programėlės](../../3-Web-App/README.md) pamoką). Galiausiai, priklausomai nuo mokymo technikų, jums gali tekti juos atsitiktinai sumaišyti.

✅ Surinkę ir apdoroję duomenis, skirkite laiko patikrinti, ar jų struktūra leis jums atsakyti į numatytą klausimą. Gali būti, kad duomenys nebus tinkami jūsų užduočiai, kaip sužinome [Klasterizavimo](../../5-Clustering/1-Visualize/README.md) pamokose!

### Savybės ir tikslas

[Savybė](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) yra matuojama jūsų duomenų savybė. Daugelyje duomenų rinkinių ji išreiškiama kaip stulpelio pavadinimas, pvz., „data“, „dydis“ ar „spalva“. Jūsų savybės kintamasis, paprastai žymimas kaip `X` kode, atspindi įvesties kintamąjį, kuris bus naudojamas modelio mokymui.

Tikslas yra tai, ką bandote prognozuoti. Tikslas, paprastai žymimas kaip `y` kode, atspindi atsakymą į klausimą, kurį bandote užduoti savo duomenims: gruodį, kokios **spalvos** moliūgai bus pigiausi? San Franciske, kuriuose rajonuose bus geriausios nekilnojamojo turto **kainos**? Kartais tikslas taip pat vadinamas etiketės atributu.

### Savybių kintamojo pasirinkimas

🎓 **Savybių pasirinkimas ir savybių ištraukimas** Kaip žinoti, kurį kintamąjį pasirinkti kuriant modelį? Tikriausiai pereisite savybių pasirinkimo arba savybių ištraukimo procesą, kad pasirinktumėte tinkamus kintamuosius geriausiam modelio veikimui. Tačiau jie nėra tas pats: „Savybių ištraukimas sukuria naujas savybes iš originalių savybių funkcijų, o savybių pasirinkimas grąžina savybių pogrupį.“ ([šaltinis](https://wikipedia.org/wiki/Feature_selection))

### Vizualizuokite savo duomenis

Svarbi duomenų mokslininko įrankių rinkinio dalis yra galimybė vizualizuoti duomenis naudojant kelias puikias bibliotekas, tokias kaip Seaborn ar MatPlotLib. Duomenų vizualizavimas gali leisti jums atskleisti paslėptas koreliacijas, kurias galite panaudoti. Vizualizacijos taip pat gali padėti atskleisti šališkumą ar nesubalansuotus duomenis (kaip sužinome [Klasifikacijoje](../../4-Classification/2-Classifiers-1/README.md)).

### Padalinkite savo duomenų rinkinį

Prieš mokymą, turite padalinti savo duomenų rinkinį į dvi ar daugiau dalių, kurios yra nevienodo dydžio, bet vis dar gerai atspindi duomenis.

- **Mokymas**. Ši duomenų rinkinio dalis pritaikoma jūsų modeliui, kad jį išmokytumėte. Šis rinkinys sudaro didžiąją dalį pradinio duomenų rinkinio.
- **Testavimas**. Testavimo duomenų rinkinys yra nepriklausoma duomenų grupė, dažnai surinkta iš pradinio duomenų rinkinio, kurią naudojate, kad patvirtintumėte sukurto modelio veikimą.
- **Validacija**. Validacijos rinkinys yra mažesnė nepriklausoma pavyzdžių grupė, kurią naudojate modelio hiperparametrų arba architektūros derinimui, kad pagerintumėte modelį. Priklausomai nuo jūsų duomenų dydžio ir klausimo, kurį užduodate, jums gali nereikėti kurti šio trečiojo rinkinio (kaip pastebime [Laiko eilučių prognozavimo](../../7-TimeSeries/1-Introduction/README.md) pamokoje).

## Modelio kūrimas

Naudodami mokymo duomenis, jūsų tikslas yra sukurti modelį, arba statistinį jūsų duomenų atvaizdavimą, naudojant įvairius algoritmus, kad jį **išmokytumėte**. Modelio mokymas leidžia jam analizuoti duomenis, daryti prielaidas apie aptiktus šablonus, juos patvirtinti ir priimti arba atmesti.

### Pasirinkite mokymo metodą

Atsižvelgdami į savo klausimą ir duomenų pobūdį, pasirinksite mokymo metodą. Naršydami [Scikit-learn dokumentaciją](https://scikit-learn.org/stable/user_guide.html) – kurią naudojame šiame kurse – galite išnagrinėti daugybę būdų, kaip mokyti modelį. Priklausomai nuo jūsų patirties, gali tekti išbandyti kelis skirtingus metodus, kad sukurtumėte geriausią modelį. Tikėtina, kad pereisite procesą, kurio metu duomenų mokslininkai vertina modelio veikimą, pateikdami jam nematytus duomenis, tikrindami tikslumą, šališkumą ir kitus kokybę mažinančius aspektus, ir pasirinkdami tinkamiausią mokymo metodą užduočiai atlikti.

### Mokykite modelį

Turėdami mokymo duomenis, esate pasiruošę „pritaikyti“ juos, kad sukurtumėte modelį. Pastebėsite, kad daugelyje ML bibliotekų rasite kodą „model.fit“ – būtent tuo metu pateikiate savo savybių kintamąjį kaip reikšmių masyvą (paprastai „X“) ir tikslinį kintamąjį (paprastai „y“).

### Įvertinkite modelį

Kai mokymo procesas bus baigtas (dideliems modeliams mokyti gali prireikti daugybės iteracijų arba „epochų“), galėsite įvertinti modelio kokybę, naudodami testavimo duomenis, kad įvertintumėte jo veikimą. Šie duomenys yra pradinio duomenų rinkinio dalis, kurią modelis anksčiau neanalizavo. Galite išspausdinti lentelę su metrikomis apie modelio kokybę.

🎓 **Modelio pritaikymas**

Mašininio mokymosi kontekste modelio pritaikymas reiškia modelio pagrindinės funkcijos tikslumą, kai jis bando analizuoti duomenis, su kuriais nėra susipažinęs.

🎓 **Nepakankamas pritaikymas** ir **perteklinis pritaikymas** yra dažnos problemos, kurios mažina modelio kokybę, nes modelis arba per mažai, arba per daug prisitaiko. Tai sukelia modelio prognozes, kurios yra arba per daug susijusios, arba per mažai susijusios su mokymo duomenimis. Perteklinis modelis per gerai prognozuoja mokymo duomenis, nes jis per daug išmoko duomenų detales ir triukšmą. Nepakankamas modelis nėra tikslus, nes jis negali tiksliai analizuoti nei mokymo duomenų, nei duomenų, kurių dar „nematė“.

![perteklinis modelis](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografikas sukurtas [Jen Looper](https://twitter.com/jenlooper)

## Parametrų derinimas

Kai pradiniai mokymai bus baigti, stebėkite modelio kokybę ir apsvarstykite galimybę ją pagerinti koreguojant jo „hiperparametrus“. Skaitykite daugiau apie procesą [dokumentacijoje](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prognozė

Tai momentas, kai galite naudoti visiškai naujus duomenis, kad patikrintumėte modelio tikslumą. „Taikomojo“ ML aplinkoje, kur kuriate interneto išteklius modelio naudojimui gamyboje, šis procesas gali apimti vartotojo įvesties surinkimą (pvz., mygtuko paspaudimą), kad nustatytumėte kintamąjį ir išsiųstumėte jį modeliui įžvalgoms arba vertinimui.

Šiose pamokose sužinosite, kaip naudoti šiuos žingsnius, kad pasiruoštumėte, sukurtumėte, išbandytumėte, įvertintumėte ir prognozuotumėte – visus duomenų mokslininko veiksmus ir dar daugiau, kai progresuosite savo kelionėje tapti „pilno ciklo“ ML inžinieriumi.

---

## 🚀Iššūkis

Nupieškite srauto diagramą, atspindinčią ML praktiko žingsnius. Kur save matote šiame procese? Kur, jūsų nuomone, susidursite su sunkumais? Kas jums atrodo lengva?

## [Po paskaitos – testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Ieškokite internete interviu su duomenų mokslininkais, kurie aptaria savo kasdienį darbą. Štai [vienas](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Užduotis

[Interviu su duomenų mokslininku](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.