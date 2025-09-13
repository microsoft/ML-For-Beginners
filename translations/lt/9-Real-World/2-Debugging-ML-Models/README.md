<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T07:53:11+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "lt"
}
-->
# Postscriptas: Modelių derinimas mašininio mokymosi srityje naudojant atsakingos dirbtinio intelekto (AI) prietaisų skydelio komponentus

## [Prieš paskaitą pateikiamas testas](https://ff-quizzes.netlify.app/en/ml/)

## Įvadas

Mašininis mokymasis daro įtaką mūsų kasdieniam gyvenimui. Dirbtinis intelektas (AI) vis dažniau naudojamas svarbiose sistemose, kurios veikia tiek mus kaip individus, tiek mūsų visuomenę – nuo sveikatos priežiūros, finansų, švietimo iki įdarbinimo. Pavyzdžiui, sistemos ir modeliai dalyvauja kasdieniuose sprendimų priėmimo procesuose, tokiuose kaip sveikatos priežiūros diagnozės ar sukčiavimo aptikimas. Dėl to AI pažanga ir spartus jos pritaikymas susiduria su besikeičiančiais visuomenės lūkesčiais ir augančiu reguliavimu. Nuolat matome sritis, kuriose AI sistemos neatitinka lūkesčių, atskleidžia naujus iššūkius, o vyriausybės pradeda reguliuoti AI sprendimus. Todėl svarbu analizuoti šiuos modelius, kad jie užtikrintų teisingus, patikimus, įtraukius, skaidrius ir atsakingus rezultatus visiems.

Šioje mokymo programoje nagrinėsime praktinius įrankius, kurie gali būti naudojami siekiant įvertinti, ar modelis turi atsakingo AI problemų. Tradiciniai mašininio mokymosi derinimo metodai dažniausiai grindžiami kiekybiniais skaičiavimais, tokiais kaip apibendrintas tikslumas ar vidutinė klaidų nuostolių vertė. Įsivaizduokite, kas gali nutikti, jei duomenys, kuriuos naudojate šiems modeliams kurti, neturi tam tikrų demografinių duomenų, tokių kaip rasė, lytis, politinės pažiūros, religija, arba jei šie demografiniai duomenys yra neproporcingai atstovaujami. O kas, jei modelio rezultatai interpretuojami taip, kad būtų palankūs tam tikrai demografinei grupei? Tai gali sukelti per didelį arba nepakankamą jautrių savybių grupių atstovavimą, dėl kurio modelis gali tapti neteisingas, neįtraukus ar nepatikimas. Kitas veiksnys yra tas, kad mašininio mokymosi modeliai laikomi „juodosiomis dėžėmis“, todėl sunku suprasti ir paaiškinti, kas lemia modelio prognozes. Visa tai yra iššūkiai, su kuriais susiduria duomenų mokslininkai ir AI kūrėjai, kai jie neturi tinkamų įrankių modelio teisingumui ar patikimumui įvertinti ir derinti.

Šioje pamokoje sužinosite, kaip derinti savo modelius naudojant:

- **Klaidų analizę**: nustatyti, kur jūsų duomenų pasiskirstyme modelis turi didelius klaidų rodiklius.
- **Modelio apžvalgą**: atlikti lyginamąją analizę tarp skirtingų duomenų grupių, kad būtų galima aptikti modelio veikimo rodiklių skirtumus.
- **Duomenų analizę**: ištirti, kur gali būti per didelis arba nepakankamas duomenų atstovavimas, kuris gali iškreipti modelį, kad jis būtų palankus vienai demografinei grupei, o ne kitai.
- **Savybių svarbą**: suprasti, kurios savybės lemia modelio prognozes globaliu ar lokaliu lygmeniu.

## Privalomos žinios

Prieš pradedant, rekomenduojame peržiūrėti [Atsakingo AI įrankius kūrėjams](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard).

> ![Gif apie atsakingo AI įrankius](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Klaidų analizė

Tradiciniai modelio veikimo rodikliai, naudojami tikslumui matuoti, dažniausiai grindžiami teisingų ir neteisingų prognozių skaičiavimais. Pavyzdžiui, nustatyti, kad modelis yra tikslus 89 % atvejų su klaidų nuostoliu 0,001, gali būti laikoma geru veikimu. Tačiau klaidos dažnai nėra tolygiai pasiskirsčiusios jūsų pagrindiniuose duomenyse. Galite gauti 89 % modelio tikslumo įvertinimą, tačiau pastebėti, kad tam tikrose duomenų srityse modelis nesėkmingas 42 % atvejų. Šių nesėkmių pasekmės tam tikrose duomenų grupėse gali sukelti teisingumo ar patikimumo problemų. Svarbu suprasti, kuriose srityse modelis veikia gerai, o kuriose – ne. Duomenų sritys, kuriose modelis turi daug netikslumų, gali būti svarbios demografinės grupės.

![Analizuokite ir derinkite modelio klaidas](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Klaidų analizės komponentas RAI prietaisų skydelyje iliustruoja, kaip modelio nesėkmės pasiskirsto įvairiose grupėse, naudodamas medžio vizualizaciją. Tai naudinga nustatant savybes ar sritis, kuriose jūsų duomenyse yra didelis klaidų rodiklis. Matydami, iš kur kyla dauguma modelio netikslumų, galite pradėti tirti pagrindinę priežastį. Taip pat galite kurti duomenų grupes analizei atlikti. Šios duomenų grupės padeda derinimo procese nustatyti, kodėl modelis veikia gerai vienoje grupėje, bet daro klaidas kitoje.

![Klaidų analizė](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Medžio žemėlapio vizualiniai indikatoriai padeda greičiau nustatyti problemų sritis. Pavyzdžiui, kuo tamsesnė raudona spalva medžio mazge, tuo didesnis klaidų rodiklis.

Šilumos žemėlapis yra dar viena vizualizacijos funkcija, kurią naudotojai gali naudoti klaidų rodikliui tirti, naudodami vieną ar dvi savybes, kad nustatytų modelio klaidų priežastis visame duomenų rinkinyje ar grupėse.

![Klaidų analizės šilumos žemėlapis](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Naudokite klaidų analizę, kai reikia:

* Giliai suprasti, kaip modelio nesėkmės pasiskirsto duomenų rinkinyje ir keliose įvesties bei savybių dimensijose.
* Išskaidyti apibendrintus veikimo rodiklius, kad automatiškai atrastumėte klaidingas grupes ir informuotumėte apie tikslines problemų sprendimo priemones.

## Modelio apžvalga

Mašininio mokymosi modelio veikimo vertinimas reikalauja holistinio jo elgsenos supratimo. Tai galima pasiekti peržiūrint daugiau nei vieną rodiklį, pvz., klaidų rodiklį, tikslumą, atšaukimą, tikslumą ar MAE (vidutinę absoliučią klaidą), kad būtų galima nustatyti veikimo rodiklių skirtumus. Vienas veikimo rodiklis gali atrodyti puikiai, tačiau netikslumai gali išryškėti kitame rodiklyje. Be to, rodiklių palyginimas visame duomenų rinkinyje ar grupėse padeda atskleisti, kur modelis veikia gerai, o kur – ne. Tai ypač svarbu norint pamatyti modelio veikimą tarp jautrių ir nejautrių savybių (pvz., paciento rasės, lyties ar amžiaus), kad būtų galima atskleisti galimą modelio neteisingumą. Pavyzdžiui, atradus, kad modelis yra klaidingesnis grupėje, kurioje yra jautrių savybių, galima atskleisti galimą modelio neteisingumą.

Modelio apžvalgos komponentas RAI prietaisų skydelyje padeda ne tik analizuoti duomenų atstovavimo veikimo rodiklius grupėje, bet ir suteikia naudotojams galimybę palyginti modelio elgseną skirtingose grupėse.

![Duomenų grupės – modelio apžvalga RAI prietaisų skydelyje](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Komponento savybių pagrindu atliekama analizė leidžia naudotojams susiaurinti duomenų pogrupius tam tikroje savybėje, kad būtų galima nustatyti anomalijas detaliu lygmeniu. Pavyzdžiui, prietaisų skydelyje yra įmontuotas intelektas, kuris automatiškai generuoja grupes pagal naudotojo pasirinktą savybę (pvz., *"time_in_hospital < 3"* arba *"time_in_hospital >= 7"*). Tai leidžia naudotojui atskirti tam tikrą savybę nuo didesnės duomenų grupės, kad pamatytų, ar ji yra pagrindinis modelio klaidingų rezultatų veiksnys.

![Savybių grupės – modelio apžvalga RAI prietaisų skydelyje](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Modelio apžvalgos komponentas palaiko dviejų klasių skirtumų rodiklius:

**Skirtumai modelio veikime**: Šie rodikliai apskaičiuoja skirtumus (skirtumą) pasirinktų veikimo rodiklių reikšmėse tarp duomenų pogrupių. Štai keli pavyzdžiai:

* Tikslumo rodiklio skirtumas
* Klaidų rodiklio skirtumas
* Tikslumo skirtumas
* Atšaukimo skirtumas
* Vidutinės absoliučios klaidos (MAE) skirtumas

**Skirtumai pasirinkimo rodiklyje**: Šis rodiklis apima skirtumą pasirinkimo rodiklyje (palankioje prognozėje) tarp pogrupių. Pavyzdžiui, tai gali būti paskolų patvirtinimo rodiklių skirtumas. Pasirinkimo rodiklis reiškia duomenų taškų dalį kiekvienoje klasėje, klasifikuotą kaip 1 (dvejetainėje klasifikacijoje), arba prognozių reikšmių pasiskirstymą (regresijoje).

## Duomenų analizė

> „Jei pakankamai ilgai kankinsite duomenis, jie prisipažins bet ką“ – Ronald Coase

Šis teiginys skamba ekstremaliai, tačiau tiesa, kad duomenis galima manipuliuoti, kad jie palaikytų bet kokią išvadą. Tokia manipuliacija kartais gali įvykti netyčia. Kaip žmonės, mes visi turime šališkumų, ir dažnai sunku sąmoningai suvokti, kada įvedate šališkumą į duomenis. Užtikrinti teisingumą AI ir mašininio mokymosi srityje išlieka sudėtingas iššūkis.

Duomenys yra didelė akloji zona tradiciniams modelio veikimo rodikliams. Galite turėti aukštus tikslumo rodiklius, tačiau tai ne visada atspindi pagrindinį duomenų šališkumą, kuris gali būti jūsų duomenų rinkinyje. Pavyzdžiui, jei darbuotojų duomenų rinkinyje 27 % moterų užima vadovaujančias pareigas įmonėje, o 73 % vyrų yra tame pačiame lygyje, darbo skelbimų AI modelis, apmokytas pagal šiuos duomenis, gali daugiausia taikyti vyrų auditorijai aukšto lygio darbo pozicijoms. Šis duomenų disbalansas iškreipė modelio prognozę, kad ji būtų palanki vienai lyčiai. Tai atskleidžia teisingumo problemą, kai AI modelis turi lyčių šališkumą.

Duomenų analizės komponentas RAI prietaisų skydelyje padeda nustatyti sritis, kuriose duomenų rinkinyje yra per didelis arba nepakankamas atstovavimas. Jis padeda naudotojams diagnozuoti klaidų ir teisingumo problemų, atsirandančių dėl duomenų disbalanso ar tam tikros duomenų grupės atstovavimo trūkumo, pagrindines priežastis. Tai suteikia naudotojams galimybę vizualizuoti duomenų rinkinius pagal prognozuojamus ir faktinius rezultatus, klaidų grupes ir specifines savybes. Kartais atradus nepakankamai atstovaujamą duomenų grupę taip pat galima pastebėti, kad modelis nesimoko tinkamai, todėl yra didelis netikslumas. Modelis, turintis duomenų šališkumą, yra ne tik teisingumo problema, bet ir rodo, kad modelis nėra įtraukus ar patikimas.

![Duomenų analizės komponentas RAI prietaisų skydelyje](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Naudokite duomenų analizę, kai reikia:

* Tyrinėti savo duomenų rinkinio statistiką, pasirenkant skirtingus filtrus, kad suskaidytumėte duomenis į skirtingas dimensijas (dar vadinamas grupėmis).
* Suprasti savo duomenų rinkinio pasiskirstymą skirtingose grupėse ir savybių grupėse.
* Nustatyti, ar jūsų išvados, susijusios su teisingumu, klaidų analize ir priežastingumu (gautos iš kitų prietaisų skydelio komponentų), yra jūsų duomenų rinkinio pasiskirstymo rezultatas.
* Nuspręsti, kuriose srityse rinkti daugiau duomenų, kad būtų sumažintos klaidos, atsirandančios dėl atstovavimo problemų, žymų triukšmo, savybių triukšmo, žymų šališkumo ir panašių veiksnių.

## Modelio interpretacija

Mašininio mokymosi modeliai dažnai laikomi „juodosiomis dėžėmis“. Suprasti, kurios pagrindinės duomenų savybės lemia modelio prognozę, gali būti sudėtinga. Svarbu užtikrinti skaidrumą, kodėl modelis priima tam tikrą prognozę. Pavyzdžiui, jei AI sistema prognozuoja, kad diabetu sergantis pacientas rizikuoja būti pakartotinai hospitalizuotas per mažiau nei 30 dienų, ji turėtų pateikti duomenis, kurie pagrindžia šią prognozę. Turint tokius duomenų rodiklius, atsiranda skaidrumas, kuris padeda gydytojams ar ligoninėms priimti gerai pagrįstus sprendimus. Be to, galimybė paaiškinti, kodėl modelis priėmė tam tikrą prognozę konkrečiam pacientui, leidžia užtikrinti atitiktį sveikatos reguliavimams. Kai naudojate mašininio mokymosi modelius, kurie daro įtaką žmonių gyvenimams, būtina suprasti ir paaiškinti, kas lemia modelio elgseną. Modelio paaiškinamumas ir interpretacija padeda atsakyti į klausimus tokiose situacijose kaip:

* Modelio derinimas: Kodėl mano modelis padarė šią klaidą? Kaip galiu pagerinti savo modelį?
* Žmogaus ir AI bendradarbiavimas: Kaip galiu suprasti ir pasitikėti modelio sprendimais?
* Reguliavimo atitiktis: Ar mano modelis atitinka teisės aktų reikalavimus?

Savybių svarbos komponentas RAI prietaisų skydelyje padeda derinti ir gauti išsamų supratimą apie tai, kaip modelis priima prognozes. Tai taip pat naudingas įrankis mašininio mokymosi specialistams ir sprendimų priėmėjams, norint paaiškinti ir pateikti įrodymus apie savybes, kurios daro įtaką modelio elgsenai, siekiant užtikrinti reguliavimo atitiktį. Naudotojai gali tyrinėti tiek globalius, tiek lokalius paaiškinimus, kad patvirtintų, kurios savybės lemia modelio prognozę. Globalūs paaiškinimai pateikia pagrindines savybes, kurios paveikė bendrą modelio prognozę. Lokalieji paaiškinimai parodo, kurios savybės lėmė modelio prognozę konkrečiu atveju. Galimybė įvertinti lokalius paaiškinimus taip pat naudinga derinant ar audituojant konkretų atvejį, siekiant geriau suprasti ir interpretuoti, kodėl modelis priėmė tikslią ar
- **Per didelė arba per maža reprezentacija**. Idėja yra ta, kad tam tikra grupė nėra matoma tam tikroje profesijoje, o bet kokia paslauga ar funkcija, kuri toliau skatina tokį reiškinį, prisideda prie žalos.

### Azure RAI prietaisų skydelis

[Azure RAI prietaisų skydelis](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) yra sukurtas remiantis atvirojo kodo įrankiais, kuriuos sukūrė pirmaujančios akademinės institucijos ir organizacijos, įskaitant Microsoft. Šie įrankiai yra labai naudingi duomenų mokslininkams ir AI kūrėjams, siekiant geriau suprasti modelio elgseną, aptikti ir sumažinti nepageidaujamus AI modelių aspektus.

- Sužinokite, kaip naudoti skirtingus komponentus, peržiūrėdami RAI prietaisų skydelio [dokumentaciją.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Peržiūrėkite keletą RAI prietaisų skydelio [pavyzdinių užrašų knygelių](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks), skirtų atsakingesnių AI scenarijų derinimui Azure Machine Learning aplinkoje.

---
## 🚀 Iššūkis

Kad statistiniai ar duomenų šališkumai nebūtų įtraukti nuo pat pradžių, turėtume:

- užtikrinti, kad sistemų kūrime dalyvautų žmonės iš įvairių aplinkybių ir perspektyvų
- investuoti į duomenų rinkinius, kurie atspindi mūsų visuomenės įvairovę
- kurti geresnius metodus šališkumui aptikti ir ištaisyti, kai jis pasireiškia

Pagalvokite apie realaus gyvenimo scenarijus, kur modelių kūrime ir naudojime akivaizdus neteisingumas. Ką dar turėtume apsvarstyti?

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)
## Peržiūra ir savarankiškas mokymasis

Šioje pamokoje sužinojote apie praktinius įrankius, kaip įtraukti atsakingą AI į mašininį mokymąsi.

Peržiūrėkite šį seminarą, kad giliau pasinertumėte į temas:

- Atsakingo AI prietaisų skydelis: Vieno langelio principas RAI praktiniam įgyvendinimui, Besmira Nushi ir Mehrnoosh Sameki

[![Atsakingo AI prietaisų skydelis: Vieno langelio principas RAI praktiniam įgyvendinimui](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Atsakingo AI prietaisų skydelis: Vieno langelio principas RAI praktiniam įgyvendinimui")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte vaizdo įrašą: Atsakingo AI prietaisų skydelis: Vieno langelio principas RAI praktiniam įgyvendinimui, Besmira Nushi ir Mehrnoosh Sameki

Naudokitės šiais šaltiniais, kad sužinotumėte daugiau apie atsakingą AI ir kaip kurti patikimesnius modelius:

- Microsoft RAI prietaisų skydelio įrankiai ML modelių derinimui: [Atsakingo AI įrankių šaltiniai](https://aka.ms/rai-dashboard)

- Susipažinkite su Atsakingo AI įrankių rinkiniu: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsoft RAI išteklių centras: [Atsakingo AI ištekliai – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft FATE tyrimų grupė: [FATE: Sąžiningumas, Atsakomybė, Skaidrumas ir Etika AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Užduotis

[Susipažinkite su RAI prietaisų skydeliu](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Dėl svarbios informacijos rekomenduojama profesionali žmogaus vertimo paslauga. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius naudojant šį vertimą.