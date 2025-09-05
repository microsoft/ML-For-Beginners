<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T12:38:13+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "sl"
}
-->
# Gradnja rešitev strojnega učenja z odgovorno umetno inteligenco

![Povzetek odgovorne umetne inteligence v strojnem učenju v sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote avtorja [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

V tem učnem načrtu boste začeli odkrivati, kako strojno učenje vpliva na naše vsakdanje življenje. Že zdaj so sistemi in modeli vključeni v vsakodnevne odločitve, kot so zdravstvene diagnoze, odobritve posojil ali odkrivanje goljufij. Zato je pomembno, da ti modeli delujejo zanesljivo in zagotavljajo rezultate, ki jim lahko zaupamo. Tako kot pri vsaki programski aplikaciji tudi sistemi umetne inteligence včasih ne izpolnijo pričakovanj ali privedejo do neželenih rezultatov. Zato je ključno razumeti in pojasniti vedenje modela umetne inteligence.

Predstavljajte si, kaj se lahko zgodi, če podatki, ki jih uporabljate za gradnjo teh modelov, ne vključujejo določenih demografskih skupin, kot so rasa, spol, politično prepričanje, religija, ali pa so te skupine nesorazmerno zastopane. Kaj pa, če je izhod modela interpretiran tako, da favorizira določeno demografsko skupino? Kakšne so posledice za aplikacijo? Poleg tega, kaj se zgodi, če model povzroči škodljive rezultate? Kdo je odgovoren za vedenje sistema umetne inteligence? To so nekatera vprašanja, ki jih bomo raziskali v tem učnem načrtu.

V tej lekciji boste:

- Povečali zavedanje o pomembnosti pravičnosti v strojnem učenju in škodah, povezanih s pravičnostjo.
- Spoznali prakso raziskovanja odstopanj in nenavadnih scenarijev za zagotavljanje zanesljivosti in varnosti.
- Pridobili razumevanje o potrebi po vključevanju vseh z oblikovanjem inkluzivnih sistemov.
- Raziskali, kako pomembno je varovati zasebnost in varnost podatkov ter ljudi.
- Spoznali pomen pristopa "steklene škatle" za pojasnjevanje vedenja modelov umetne inteligence.
- Postali pozorni na to, kako je odgovornost ključna za gradnjo zaupanja v sisteme umetne inteligence.

## Predpogoj

Kot predpogoj si oglejte učni načrt "Načela odgovorne umetne inteligence" in si oglejte spodnji video na to temo:

Več o odgovorni umetni inteligenci lahko izveste s sledenjem temu [učnemu načrtu](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott).

[![Microsoftov pristop k odgovorni umetni inteligenci](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoftov pristop k odgovorni umetni inteligenci")

> 🎥 Kliknite zgornjo sliko za video: Microsoftov pristop k odgovorni umetni inteligenci

## Pravičnost

Sistemi umetne inteligence bi morali obravnavati vse enako in se izogibati različnemu vplivu na podobne skupine ljudi. Na primer, ko sistemi umetne inteligence svetujejo pri medicinskem zdravljenju, prošnjah za posojila ali zaposlitvi, bi morali podati enaka priporočila vsem z enakimi simptomi, finančnimi okoliščinami ali poklicnimi kvalifikacijami. Vsak od nas kot človek nosi podedovane pristranskosti, ki vplivajo na naše odločitve in dejanja. Te pristranskosti so lahko vidne v podatkih, ki jih uporabljamo za treniranje sistemov umetne inteligence. Takšna manipulacija se včasih zgodi nenamerno. Pogosto je težko zavestno vedeti, kdaj vnašate pristranskost v podatke.

**"Nepravičnost"** zajema negativne vplive ali "škode" za skupino ljudi, kot so tiste, opredeljene glede na raso, spol, starost ali status invalidnosti. Glavne škode, povezane s pravičnostjo, lahko razvrstimo kot:

- **Dodeljevanje**, če je na primer spol ali etnična pripadnost favorizirana pred drugo.
- **Kakovost storitve**. Če trenirate podatke za en specifičen scenarij, vendar je resničnost veliko bolj kompleksna, to vodi do slabo delujoče storitve. Na primer, podajalnik mila, ki ne zazna ljudi s temno kožo. [Referenca](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Omalovaževanje**. Nepravično kritiziranje in označevanje nečesa ali nekoga. Na primer, tehnologija za označevanje slik je zloglasno napačno označila slike temnopolte ljudi kot gorile.
- **Prekomerna ali nezadostna zastopanost**. Ideja, da določena skupina ni vidna v določenem poklicu, in vsaka storitev ali funkcija, ki to še naprej promovira, prispeva k škodi.
- **Stereotipiziranje**. Povezovanje določene skupine s predhodno določenimi lastnostmi. Na primer, sistem za prevajanje med angleščino in turščino lahko vsebuje netočnosti zaradi besed s stereotipnimi povezavami s spolom.

![prevod v turščino](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> prevod v turščino

![prevod nazaj v angleščino](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> prevod nazaj v angleščino

Pri oblikovanju in testiranju sistemov umetne inteligence moramo zagotoviti, da je umetna inteligenca pravična in ni programirana za sprejemanje pristranskih ali diskriminatornih odločitev, ki jih ljudje prav tako ne smejo sprejemati. Zagotavljanje pravičnosti v umetni inteligenci in strojnem učenju ostaja kompleksen sociotehnični izziv.

### Zanesljivost in varnost

Za gradnjo zaupanja morajo biti sistemi umetne inteligence zanesljivi, varni in dosledni v običajnih in nepričakovanih pogojih. Pomembno je vedeti, kako se bodo sistemi umetne inteligence obnašali v različnih situacijah, še posebej, ko gre za odstopanja. Pri gradnji rešitev umetne inteligence je treba nameniti veliko pozornosti temu, kako obravnavati širok spekter okoliščin, s katerimi se lahko srečajo rešitve umetne inteligence. Na primer, avtonomni avtomobil mora postaviti varnost ljudi kot najvišjo prioriteto. Posledično mora umetna inteligenca, ki poganja avtomobil, upoštevati vse možne scenarije, s katerimi se lahko avtomobil sreča, kot so noč, nevihte ali snežni meteži, otroci, ki tečejo čez cesto, hišni ljubljenčki, gradbena dela itd. Kako dobro sistem umetne inteligence obravnava širok spekter pogojev zanesljivo in varno, odraža raven predvidevanja, ki jo je podatkovni znanstvenik ali razvijalec umetne inteligence upošteval med oblikovanjem ali testiranjem sistema.

> [🎥 Kliknite tukaj za video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzivnost

Sistemi umetne inteligence bi morali biti zasnovani tako, da vključujejo in opolnomočijo vse. Pri oblikovanju in implementaciji sistemov umetne inteligence podatkovni znanstveniki in razvijalci umetne inteligence identificirajo in obravnavajo morebitne ovire v sistemu, ki bi lahko nenamerno izključile ljudi. Na primer, po svetu je 1 milijarda ljudi z invalidnostmi. Z napredkom umetne inteligence lahko dostopajo do širokega spektra informacij in priložnosti bolj enostavno v svojem vsakdanjem življenju. Z odpravljanjem ovir se ustvarjajo priložnosti za inovacije in razvoj izdelkov umetne inteligence z boljšimi izkušnjami, ki koristijo vsem.

> [🎥 Kliknite tukaj za video: inkluzivnost v umetni inteligenci](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Varnost in zasebnost

Sistemi umetne inteligence bi morali biti varni in spoštovati zasebnost ljudi. Ljudje manj zaupajo sistemom, ki ogrožajo njihovo zasebnost, informacije ali življenja. Pri treniranju modelov strojnega učenja se zanašamo na podatke za doseganje najboljših rezultatov. Pri tem je treba upoštevati izvor podatkov in njihovo integriteto. Na primer, ali so podatki uporabniško posredovani ali javno dostopni? Poleg tega je med delom s podatki ključno razviti sisteme umetne inteligence, ki lahko zaščitijo zaupne informacije in se uprejo napadom. Ker umetna inteligenca postaja vse bolj razširjena, postaja zaščita zasebnosti in varnost pomembnih osebnih ter poslovnih informacij vse bolj kritična in kompleksna. Zasebnost in varnost podatkov zahtevata posebno pozornost pri umetni inteligenci, saj je dostop do podatkov bistven za to, da sistemi umetne inteligence sprejemajo natančne in informirane napovedi ter odločitve o ljudeh.

> [🎥 Kliknite tukaj za video: varnost v umetni inteligenci](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kot industrija smo dosegli pomemben napredek na področju zasebnosti in varnosti, ki ga močno spodbujajo regulacije, kot je GDPR (Splošna uredba o varstvu podatkov).
- Kljub temu moramo pri sistemih umetne inteligence priznati napetost med potrebo po več osebnih podatkih za izboljšanje učinkovitosti sistemov in zasebnostjo.
- Tako kot ob rojstvu povezanih računalnikov z internetom, opažamo tudi velik porast števila varnostnih težav, povezanih z umetno inteligenco.
- Hkrati pa opažamo, da se umetna inteligenca uporablja za izboljšanje varnosti. Na primer, večina sodobnih protivirusnih skenerjev temelji na AI heuristiki.
- Zagotoviti moramo, da se naši procesi podatkovne znanosti harmonično prepletajo z najnovejšimi praksami zasebnosti in varnosti.

### Transparentnost

Sistemi umetne inteligence bi morali biti razumljivi. Ključni del transparentnosti je pojasnjevanje vedenja sistemov umetne inteligence in njihovih komponent. Izboljšanje razumevanja sistemov umetne inteligence zahteva, da deležniki razumejo, kako in zakaj delujejo, da lahko identificirajo morebitne težave z zmogljivostjo, varnostjo in zasebnostjo, pristranskosti, izključujoče prakse ali nenamerne rezultate. Prav tako verjamemo, da bi morali biti tisti, ki uporabljajo sisteme umetne inteligence, iskreni in odkriti glede tega, kdaj, zakaj in kako se odločijo za njihovo uporabo. Prav tako morajo pojasniti omejitve sistemov, ki jih uporabljajo. Na primer, če banka uporablja sistem umetne inteligence za podporo pri odločanju o potrošniških posojilih, je pomembno preučiti rezultate in razumeti, kateri podatki vplivajo na priporočila sistema. Vlade začenjajo regulirati umetno inteligenco v različnih industrijah, zato morajo podatkovni znanstveniki in organizacije pojasniti, ali sistem umetne inteligence izpolnjuje regulativne zahteve, še posebej, ko pride do neželenega rezultata.

> [🎥 Kliknite tukaj za video: transparentnost v umetni inteligenci](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ker so sistemi umetne inteligence tako kompleksni, je težko razumeti, kako delujejo in interpretirati rezultate.
- To pomanjkanje razumevanja vpliva na način upravljanja, operacionalizacije in dokumentiranja teh sistemov.
- Še pomembneje pa to pomanjkanje razumevanja vpliva na odločitve, sprejete na podlagi rezultatov, ki jih ti sistemi proizvajajo.

### Odgovornost

Ljudje, ki oblikujejo in uvajajo sisteme umetne inteligence, morajo biti odgovorni za delovanje svojih sistemov. Potreba po odgovornosti je še posebej ključna pri občutljivih tehnologijah, kot je prepoznavanje obrazov. V zadnjem času se povečuje povpraševanje po tehnologiji prepoznavanja obrazov, zlasti s strani organov pregona, ki vidijo potencial tehnologije pri uporabi, kot je iskanje pogrešanih otrok. Vendar pa bi te tehnologije lahko potencialno uporabila vlada za ogrožanje temeljnih svoboščin svojih državljanov, na primer z omogočanjem neprekinjenega nadzora določenih posameznikov. Zato morajo podatkovni znanstveniki in organizacije prevzeti odgovornost za to, kako njihov sistem umetne inteligence vpliva na posameznike ali družbo.

[![Vodilni raziskovalec umetne inteligence opozarja na množični nadzor prek prepoznavanja obrazov](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoftov pristop k odgovorni umetni inteligenci")

> 🎥 Kliknite zgornjo sliko za video: Opozorila o množičnem nadzoru prek prepoznavanja obrazov

Na koncu je eno največjih vprašanj naše generacije, kot prve generacije, ki prinaša umetno inteligenco v družbo, kako zagotoviti, da bodo računalniki ostali odgovorni ljudem in kako zagotoviti, da bodo ljudje, ki oblikujejo računalnike, ostali odgovorni vsem drugim.

## Ocena vpliva

Pred treniranjem modela strojnega učenja je pomembno izvesti oceno vpliva, da razumemo namen sistema umetne inteligence; kakšna je predvidena uporaba; kje bo uveden; in kdo bo interagiral s sistemom. To je koristno za pregledovalce ali preizkuševalce, ki ocenjujejo sistem, da vedo, katere dejavnike je treba upoštevati pri prepoznavanju morebitnih tveganj in pričakovanih posledic.

Naslednja področja so v ospredju pri izvajanju ocene vpliva:

* **Negativen vpliv na posameznike**. Zavedanje o kakršnih koli omejitvah ali zahtevah, nepodprti uporabi ali znanih omejitvah, ki ovirajo delovanje sistema, je ključno za zagotovitev, da sistem ni uporabljen na način, ki bi lahko škodoval posameznikom.
* **Zahteve glede podatkov**. Razumevanje, kako in kje bo sistem uporabljal podatke, omogoča pregledovalcem, da raziščejo morebitne zahteve glede podatkov, na katere morate biti pozorni (npr. GDPR ali HIPPA regulacije podatkov). Poleg tega preučite, ali je vir ali količina podatkov zadostna za treniranje.
* **Povzetek vpliva**. Zberite seznam morebitnih škod, ki bi lahko nastale zaradi uporabe sistema. Skozi življenjski cikel strojnega učenja preverite, ali so identificirane težave ublažene ali obravnavane.
* **Ustrezni cilji** za vsako od šestih temeljnih načel. Ocenite, ali so cilji iz vsakega načela doseženi in ali obstajajo kakršne koli vrzeli.

## Odpravljanje napak z odgovorno umetno inteligenco

Podobno kot odpravljanje napak v programski aplikaciji je odpravljanje napak v sistemu umetne inteligence nujen proces prepoznavanja in reševanja težav v sistemu. Obstaja veliko dejavnikov, ki lahko vplivajo na to, da model ne deluje, kot je pričakovano ali odgovorno. Večina tradicionalnih metrik zmogljivosti modela so kvantitativni agregati zmogljivosti modela, ki niso dovolj za analizo, kako model krši načela odgovorne umetne inteligence. Poleg tega je model strojnega učenja črna škatla, kar otežuje razumevanje, kaj vodi do njegovih rezultatov ali zagotavljanje pojasnil, ko naredi napako. Kasneje v tem tečaju se bomo naučili, kako uporabljati nadzorno ploščo odgovorne umetne inteligence za pomoč pri odpravljanju napak v sistemih umetne inteligence. Nadzorna plošča zagotavlja celovit pripomoček za podatkovne znanstvenike in razvijalce umetne inteligence za izvajanje:

* **Analiza napak**. Za prepoznavanje porazdelitve napak modela, ki lahko vplivajo na pravičnost ali zanesljivost sistema.
* **Pregled modela**. Za odkrivanje, kje obstajajo razlike v zmogljivosti modela med podatkovnimi skupinami.
* **Analiza podatkov**. Za razumevanje porazdelitve podatkov in prepoznavanje morebitne pristranskosti v podatkih, ki bi lahko povzročila težave s pravi
Oglejte si ta delavnico za poglobitev v teme:

- Na poti do odgovorne umetne inteligence: Prenos načel v prakso, avtorji Besmira Nushi, Mehrnoosh Sameki in Amit Sharma

[![Responsible AI Toolbox: Odprtokodni okvir za gradnjo odgovorne umetne inteligence](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Odprtokodni okvir za gradnjo odgovorne umetne inteligence")

> 🎥 Kliknite zgornjo sliko za video: RAI Toolbox: Odprtokodni okvir za gradnjo odgovorne umetne inteligence, avtorji Besmira Nushi, Mehrnoosh Sameki in Amit Sharma

Preberite tudi:

- Microsoftov center virov za RAI: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoftova raziskovalna skupina FATE: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [GitHub repozitorij Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Preberite o orodjih Azure Machine Learning za zagotavljanje pravičnosti:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Naloga

[Raziščite RAI Toolbox](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.