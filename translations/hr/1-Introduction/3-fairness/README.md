# Izgradnja rješenja strojnog učenja uz odgovornu umjetnu inteligenciju
 
![Sažetak odgovorne umjetne inteligencije u strojnome učenju u sketchnote](../../../../translated_images/hr/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote autora [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pred-ispitni kviz](https://ff-quizzes.netlify.app/en/ml/)
 
## Uvod

U ovom nastavnom programu počet ćete otkrivati kako strojno učenje može utjecati i već utječe na naše svakodnevne živote. Čak i sada, sustavi i modeli sudjeluju u svakodnevnim zadacima donošenja odluka, poput dijagnoza u zdravstvu, odobrenja kredita ili otkrivanja prijevara. Stoga je važno da ti modeli dobro funkcioniraju pružajući rezultate kojima se može vjerovati. Kao i svaki softverski program, AI sustavi mogu ne udovoljiti očekivanjima ili imati neželjene rezultate. Zato je ključno moći razumjeti i objasniti ponašanje AI modela.

Zamislite što se može dogoditi kad podaci koje koristite za izgradnju tih modela ne sadrže određene demografske skupine, poput rase, spola, političkog stava, religije, ili neproporcionalno predstavljaju takve skupine. Što ako se izlaz modela tumači da favorizira neku demografsku skupinu? Koja je posljedica za aplikaciju? Uz to, što se događa kada model ima štetan ishod i nanosi štetu ljudima? Tko je odgovoran za ponašanje AI sustava? To su neka od pitanja koja ćemo obrađivati u ovom nastavnom programu.

U ovoj lekciji ćete:

- Podići svoju svijest o važnosti pravednosti u strojnome učenju i štetama povezanima s nepravednošću.
- Upoznati se s praksom ispitivanja odstupanja i neobičnih scenarija kako bi se osigurala pouzdanost i sigurnost.
- Razumjeti potrebu osnaživanja svih dizajniranjem inkluzivnih sustava.
- Istražiti koliko je važno zaštititi privatnost i sigurnost podataka i ljudi.
- Vidjeti važnost pristupa kroz staklenu kutiju za objašnjenje ponašanja AI modela.
- Razumjeti koliko je odgovornost ključna za izgradnju povjerenja u AI sustave.

## Preduvjet

Kao preduvjet, molimo vas da prođete "Principi odgovorne umjetne inteligencije" učenje i pogledate video u nastavku na tu temu:

Saznajte više o odgovornoj umjetnoj inteligenciji prateći ovaj [Učenje Put](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoftov pristup odgovornoj umjetnoj inteligenciji](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoftov pristup odgovornoj umjetnoj inteligenciji")

> 🎥 Kliknite sliku gore za video: Microsoftov pristup odgovornoj umjetnoj inteligenciji

## Pravednost

AI sustavi trebaju uvijek jednako tretirati sve i izbjegavati da utječu na slične skupine ljudi različito. Na primjer, kad AI sustavi daju smjernice o medicinskom tretmanu, zahtjevima za kredit ili zapošljavanju, trebali bi donositi iste preporuke svima sa sličnim simptomima, financijskim okolnostima ili profesionalnim kvalifikacijama. Svaki od nas kao ljudi nosi naslijeđene predrasude koje utječu na naše odluke i postupke. Te se predrasude mogu očitovati u podacima koje koristimo za treniranje AI sustava. Takve manipulacije ponekad mogu nastati nenamjerno. Često je teško svjesno znati kada uvodite pristranost u podatke.

**„Nepravednost“** obuhvaća negativne utjecaje ili „štete“ za skupinu ljudi, poput onih definiranih po rasi, spolu, dobi ili statusu invaliditeta. Glavne štete povezane s pravednošću mogu se klasificirati kao:

- **Dodjela**, ako je na primjer favoritiziran spol ili etnička pripadnost nad drugima.
- **Kvaliteta usluge**. Ako trenirate podatke za jedan specifičan scenarij, dok je stvarnost mnogo kompleksnija, to dovodi do lošeg performansa usluge. Na primjer, dozirnik tekućeg sapuna koji nije mogao prepoznati ljude tamnije kože. [Referenca](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Ograničavanje**. Nepravedno kritizirati i nalijepiti etiketu nečemu ili nekome. Na primjer, tehnologija označavanja slika poznata je po pogrešnom označavanju slika ljudi tamnije kože kao gorile.
- **Prevelika ili premala zastupljenost**. Ideja je da određena skupina nije vidljiva u određenom profesijom, a svaki servis ili funkcija koji to dalje promiče doprinosi šteti.
- **Stereotipiziranje**. Povezivanje određene skupine s unaprijed pridruženim atributima. Na primjer, sustav za prijevod jezika između engleskog i turskog može imati netočnosti zbog riječi koje stereotipno povezuje sa spolom.

![prijevod na turski](../../../../translated_images/hr/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> prijevod na turski

![prijevod natrag na engleski](../../../../translated_images/hr/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> prijevod natrag na engleski

Pri dizajniranju i testiranju AI sustava, trebamo osigurati da je AI pravedan i da nije programiran za donošenje pristranih ili diskriminatornih odluka, što je i ljudima zabranjeno. Jamčenje pravednosti u AI i strojnome učenju ostaje složen sociotehnički izazov.

### Pouzdanost i sigurnost

Za izgradnju povjerenja AI sustavi moraju biti pouzdani, sigurni i dosljedni u normalnim i neočekivanim uvjetima. Važno je znati kako će se AI sustavi ponašati u različitim situacijama, posebno kad su odstupanja u pitanju. Pri izgradnji AI rješenja treba posvetiti značajnu pažnju kako se nositi sa širokim rasponom okolnosti koje AI rješenja mogu susresti. Na primjer, samovozeći automobil mora staviti sigurnost ljudi kao najvažniji prioritet. Kao rezultat, AI koji upravlja automobilom mora uzeti u obzir sve moguće scenarije na koje automobil može naići, poput noći, oluja ili snježnih mećava, djece koja trče preko ulice, kućnih ljubimaca, radova na cesti itd. Koliko dobro AI sustav može pouzdano i sigurno podnijeti širok raspon uvjeta odražava razinu anticipacije koju je znanstvenik za podatke ili AI programer uzeo u obzir tijekom dizajna ili ispitivanja sustava.

> [🎥 Kliknite ovdje za video: Pouzdanost i sigurnost u AI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Uključenost

AI sustavi trebaju biti dizajnirani da angažiraju i osnaže sve. Prilikom dizajniranja i implementacije AI sustava, znanstvenici za podatke i AI programeri identificiraju i rješavaju moguće prepreke u sustavu koje bi nenamjerno mogle isključiti ljude. Na primjer, postoji 1 milijarda ljudi s invaliditetom širom svijeta. S napretkom AI, oni mogu lakše pristupiti širokom rasponu informacija i mogućnosti u svakodnevnom životu. Rješavanjem prepreka stvaraju se prilike za inovacije i razvoj AI proizvoda s boljim iskustvima koja koriste svima.

> [🎥 Kliknite ovdje za video: Uključenost u AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Sigurnost i privatnost

AI sustavi trebaju biti sigurni i poštivati privatnost ljudi. Ljudi manje vjeruju sustavima koji ugrožavaju njihovu privatnost, informacije ili živote. Prilikom treniranja modela strojnog učenja oslanjamo se na podatke da bismo postigli najbolje rezultate. Pri tome je važno uzeti u obzir podrijetlo podataka i integritet. Na primjer, jesu li podaci uneseni od strane korisnika ili su javno dostupni? Nadalje, tijekom rada s podacima, ključno je razvijati AI sustave koji mogu zaštititi povjerljive informacije i odoljeti napadima. Kako AI postaje sve rašireniji, zaštita privatnosti i osiguranje važnih osobnih i poslovnih informacija postaju sve kritičniji i složeniji zadaci. Problemi privatnosti i sigurnosti podataka zahtijevaju posebnu pozornost u AI jer je pristup podacima od ključne važnosti za AI sustave da bi mogli donositi točne i informirane prognoze i odluke o ljudima.

> [🎥 Kliknite ovdje za video: Sigurnost u AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kao industrija napravili smo značajan napredak u području privatnosti i sigurnosti, osobito potaknut propisima poput GDPR-a (Opća uredba o zaštiti podataka).
- No, s AI sustavima moramo priznati napetost između potrebe za više osobnih podataka da bi sustavi bili osobniji i učinkovitiji – i privatnosti.
- Baš kao i s pojavom povezanih računala i interneta, i s AI-jem vidimo snažan porast sigurnosnih problema povezanih s AI.
- Istovremeno, vidjeli smo da se AI koristi za poboljšanje sigurnosti. Na primjer, većina modernih antivirusnih skenera danas koristi AI heuristiku.
- Moramo osigurati da naši procesi znanosti o podacima skladno pomiješaju s najnovijim praksama privatnosti i sigurnosti.


### Transparentnost
AI sustavi trebaju biti razumljivi. Ključni dio transparentnosti je objašnjenje ponašanja AI sustava i njihovih komponenti. Poboljšanje razumijevanja AI sustava zahtijeva da dionici shvate kako i zašto sustavi funkcioniraju kako bi mogli identificirati moguće probleme s izvedbom, sigurnosne i privatnosne zabrinutosti, pristranosti, isključujuće prakse ili nepredviđene rezultate. Također vjerujemo da oni koji koriste AI sustave trebaju biti iskreni i otvoreni o tome kada, zašto i kako ih odluče koristiti kao i o ograničenjima sustava koje koriste. Na primjer, ako banka upotrebljava AI sustav za podršku u odlučivanju o potrošačkim kreditima, važno je pregledati rezultate i razumjeti koji podaci utječu na preporuke sustava. Vlade počinju regulirati AI u različitim industrijama, stoga znanstvenici za podatke i organizacije moraju objasniti zadovoljava li AI sustav regulatorne zahtjeve, osobito ako postoji neželjeni ishod.

> [🎥 Kliknite ovdje za video: Transparentnost u AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Budući da su AI sustavi tako složeni, teško je razumjeti kako funkcioniraju i protumačiti rezultate.
- Ovaj nedostatak razumijevanja utječe na način na koji se ti sustavi upravljaju, operacionaliziraju i dokumentiraju.
- Još važnije, ovaj nedostatak razumijevanja utječe na odluke donošene uporabom rezultata koje ti sustavi proizvode.

### Odgovornost
 
Ljudi koji dizajniraju i implementiraju AI sustave moraju biti odgovorni za način na koji njihovi sustavi funkcioniraju. Potreba za odgovornošću posebno je važna kod osjetljivih tehnologija poput prepoznavanja lica. Nedavno je narasla potražnja za tehnologijom prepoznavanja lica, osobito od strane policijskih organizacija koje vide potencijal te tehnologije u primjenama kao što je pronalaženje nestale djece. Međutim, te se tehnologije mogu potencijalno koristiti od strane vlade da ugroze temeljne slobode građana, na primjer omogućavanjem kontinuiranog nadzora određenih osoba. Stoga znanstvenici za podatke i organizacije moraju biti odgovorni za to kako njihov AI sustav utječe na pojedince ili društvo.

[![Vodeći istraživač AI upozorava na masovni nadzor kroz prepoznavanje lica](../../../../translated_images/hr/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoftov pristup odgovornoj umjetnoj inteligenciji")

> 🎥 Kliknite sliku gore za video: Upozorenja o masovnom nadzoru putem prepoznavanja lica

Najvažnije pitanje za naše generacije, kao prve generacije koje uvode AI u društvo, jest kako osigurati da računala ostanu odgovorna ljudima i kako osigurati da ljudi koji ih dizajniraju ostanu odgovorni prema svima ostalima.

## Procjena utjecaja

Prije treniranja modela strojnog učenja važno je provesti procjenu utjecaja kako biste razumjeli svrhu AI sustava; koja je namjeravana uporaba; gdje će se primijeniti; i tko će komunicirati sa sustavom. To je korisno za recenzente ili testere koji ocjenjuju sustav da znaju koje čimbenike uzeti u obzir kod identificiranja potencijalnih rizika i očekivanih posljedica.

Sljedeća su područja fokusa pri provođenju procjene utjecaja:

* **Negativan utjecaj na pojedince**. Svijest o bilo kakvim ograničenjima ili zahtjevima, nedozvoljenoj uporabi ili poznatim ograničenjima koja ograničavaju izvedbu sustava presudna je za osiguranje da se sustav ne koristi na način koji bi mogao nanijeti štetu pojedincima.
* **Zahtjevi za podatke**. Razumijevanje načina i mjesta upotrebe podataka u sustavu omogućuje recenzentima da istraže sve zahtjeve vezane za podatke za koje trebate biti svjesni (npr. propise GDPR-a ili HIPAA-e). Također provjerite je li izvor ili količina podataka dovoljna za treniranje.
* **Sažetak utjecaja**. Prikupite popis potencijalnih šteta koje bi mogle nastati uporabom sustava. Tijekom životnog ciklusa strojnog učenja provjeravajte jesu li identificirani problemi otklonjeni ili riješeni.
* **Primjenjivi ciljevi** za svaki od šest temeljnih principa. Procijenite zadovoljavaju li se ciljevi svakog principa i postoje li praznine.


## Otklanjanje grešaka uz odgovornu umjetnu inteligenciju

Slično kao i otklanjanje pogrešaka u softverskoj aplikaciji, otklanjanje pogrešaka u AI sustavu je potreban proces identificiranja i rješavanja problema u sustavu. Postoji mnogo čimbenika koji mogu utjecati da model ne radi kako se očekuje ili odgovorno. Većina tradicionalnih metrika modela za performanse su kvantitativni agregati izvedbe modela, što nije dovoljno da se analizira kako model krši principe odgovorne umjetne inteligencije. Nadalje, model strojnog učenja je crna kutija što otežava razumijevanje što uzrokuje njegov ishod ili pružanje objašnjenja kad pogriješi. Kasnije u ovom tečaju naučit ćemo kako koristiti nadzornu ploču Responsible AI za pomoć pri otklanjanju pogrešaka u AI sustavima. Nadzorna ploča pruža cjeloviti alat za znanstvenike podataka i AI programere za:

* **Analizu pogrešaka**. Za identifikaciju distribucije pogrešaka modela koje mogu utjecati na pravednost ili pouzdanost sustava.
* **Pregled modela**. Za otkrivanje gdje postoje razlike u izvedbi modela u različitim skupinama podataka.
* **Analizu podataka**. Za razumijevanje raspodjele podataka i identifikaciju moguće pristranosti u podacima koja može uzrokovati probleme pravednosti, uključenosti i pouzdanosti.
* **Tumačenje modela**. Za razumijevanje što utječe ili određuje predviđanja modela. To pomaže u objašnjavanju ponašanja modela, što je važno za transparentnost i odgovornost.


## 🚀 Izazov
 
Kako bismo spriječili da se štete uopće pojave, trebali bismo:

- imati raznolikost u pozadinama i perspektivama među ljudima koji rade na sustavima
- ulagati u skupove podataka koji odražavaju raznolikost našeg društva
- razvijati bolje metode tijekom cijelog životnog ciklusa strojnog učenja za otkrivanje i ispravljanje neodgovorne umjetne inteligencije kad se pojavi

Razmislite o stvarnim scenarijima u kojima je nepouzdana izvedba modela očita u izgradnji i uporabi modela. Što još bismo trebali uzeti u obzir?

## [Post-ispitni kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled & Samostalno učenje
 
U ovoj ste lekciji naučili neke osnove pojmova pravednosti i nepravednosti u strojnome učenju.
 
Pogledajte ovaj radionicu za detaljniji uvid u teme:

- U potrazi za odgovornom umjetnom inteligencijom: Primjena principa u praksi, Besmira Nushi, Mehrnoosh Sameki i Amit Sharma

[![Responsible AI Toolbox: Open-source okvir za izgradnju odgovorne AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Open-source okvir za izgradnju odgovorne AI")

> 🎥 Kliknite na gornju sliku za video: RAI Toolbox: Open-source okvir za izgradnju odgovorne AI autorica Besmira Nushi, Mehrnoosh Sameki i Amit Sharma

Također, pročitajte: 

- Microsoftov RAI centar resursa: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoftova istraživačka skupina FATE: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Responsible AI Toolbox GitHub repozitorij](https://github.com/microsoft/responsible-ai-toolbox)

Pročitajte o alatima Azure Machine Learning za osiguranje pravičnosti:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Zadatak

[Istražite RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Napomena**:
Ovaj dokument je preveden korištenjem AI prevoditeljskog servisa [Co-op Translator](https://github.com/Azure/co-op-translator). Iako težimo točnosti, imajte na umu da automatski prijevodi mogu sadržavati greške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za važne informacije preporuča se profesionalni ljudski prijevod. Nismo odgovorni za bilo kakva nesporazumevanja ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->