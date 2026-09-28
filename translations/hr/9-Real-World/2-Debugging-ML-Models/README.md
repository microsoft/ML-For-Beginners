# Postscript: Otklanjanje pogrešaka u modelima strojnog učenja pomoću komponenti nadzorne ploče za odgovornu umjetnu inteligenciju

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

Strojno učenje utječe na naše svakodnevne živote. Umjetna inteligencija (AI) pronalazi svoj put u neke od najvažnijih sustava koji utječu na nas kao pojedince i na naše društvo, od zdravstva, financija, obrazovanja do zapošljavanja. Na primjer, sustavi i modeli uključeni su u svakodnevne zadatke donošenja odluka, poput dijagnoza u zdravstvu ili otkrivanja prijevara. Posljedično, napredak u AI-u, zajedno s ubrzanim usvajanjem, suočava se s evoluirajućim društvenim očekivanjima i sve većom regulacijom. Stalno svjedočimo područjima u kojima AI sustavi ne ispunjavaju očekivanja; otkrivaju nove izazove; a vlade počinju regulirati AI rješenja. Stoga je važno analizirati ove modele kako bi se osigurali pravedni, pouzdani, uključivi, transparentni i odgovorni ishodi za sve.

U ovom kurikulumu istražit ćemo praktične alate koji se mogu koristiti za procjenu ima li model problema s odgovornom umjetnom inteligencijom. Tradicionalne tehnike otklanjanja pogrešaka u strojnome učenju obično se temelje na kvantitativnim izračunima poput agregirane točnosti ili prosječnog gubitka pogreške. Zamislite što se može dogoditi kada podaci koje koristite za izgradnju ovih modela nedostaju određenih demografskih podataka, poput rase, spola, političkog stava, religije ili su nerazmjerno zastupljeni. Što ako se izlaz modela interpretira tako da favorizira neku demografsku skupinu? To može dovesti do prekomjerne ili nedovoljne zastupljenosti ovih osjetljivih značajki, što rezultira problemima pravednosti, uključivosti ili pouzdanosti modela. Još jedan faktor je taj što se modeli strojnog učenja smatraju "crnim kutijama", što otežava razumijevanje i objašnjenje što pokreće predikciju modela. Sve su to izazovi s kojima se suočavaju znanstvenici za podatke i AI programeri kada nemaju odgovarajuće alate za otklanjanje pogrešaka i procjenu pravednosti ili pouzdanosti modela.

U ovoj lekciji naučit ćete kako otklanjati pogreške u svojim modelima koristeći:

- **Analizu pogrešaka**: identificiranje dijelova distribucije podataka gdje model ima visoke stope pogrešaka.
- **Pregled modela**: provođenje usporedne analize među različitim kohortama podataka kako bi se otkrile razlike u metrikama izvedbe modela.
- **Analizu podataka**: istraživanje područja gdje može postojati prekomjerna ili nedovoljna zastupljenost podataka koja može iskriviti model u korist jedne demografske skupine u odnosu na drugu.
- **Važnost značajki**: razumijevanje koje značajke pokreću predikcije modela na globalnoj ili lokalnoj razini.

## Preduvjet

Kao preduvjet, pregledajte [Alate za odgovornu umjetnu inteligenciju za programere](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif o alatima za odgovornu umjetnu inteligenciju](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Analiza pogrešaka

Tradicionalne metrike izvedbe modela koje se koriste za mjerenje točnosti uglavnom su izračuni temeljeni na ispravnim i pogrešnim predikcijama. Na primjer, određivanje da je model točan 89% vremena s gubitkom pogreške od 0,001 može se smatrati dobrom izvedbom. Međutim, pogreške često nisu ravnomjerno raspoređene u vašem osnovnom skupu podataka. Možete dobiti rezultat točnosti modela od 89%, ali otkriti da postoje različiti dijelovi vaših podataka za koje model griješi 42% vremena. Posljedice ovih obrazaca pogrešaka s određenim skupinama podataka mogu dovesti do problema pravednosti ili pouzdanosti. Ključno je razumjeti područja u kojima model dobro ili loše radi. Područja podataka s velikim brojem netočnosti u vašem modelu mogu se pokazati važnim demografskim podacima.

![Analizirajte i otklonite pogreške modela](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Komponenta za analizu pogrešaka na nadzornoj ploči RAI ilustrira kako su pogreške modela raspoređene među različitim kohortama pomoću vizualizacije stabla. Ovo je korisno za identificiranje značajki ili područja s visokom stopom pogrešaka u vašem skupu podataka. Promatranjem odakle dolazi većina netočnosti modela, možete započeti istraživanje uzroka. Također možete stvoriti kohorte podataka za provođenje analize. Ove kohorte podataka pomažu u procesu otklanjanja pogrešaka kako bi se utvrdilo zašto model dobro radi u jednoj kohorti, a griješi u drugoj.

![Analiza pogrešaka](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Vizualni pokazatelji na karti stabla pomažu brže locirati problematična područja. Na primjer, što je tamnija nijansa crvene boje na čvoru stabla, to je veća stopa pogreške.

Toplinska karta je još jedna funkcionalnost vizualizacije koju korisnici mogu koristiti za istraživanje stope pogreške koristeći jednu ili dvije značajke kako bi pronašli čimbenike koji doprinose pogreškama modela u cijelom skupu podataka ili kohortama.

![Toplinska karta analize pogrešaka](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Koristite analizu pogrešaka kada trebate:

* Dobiti duboko razumijevanje kako su pogreške modela raspoređene u skupu podataka i među različitim ulaznim i značajkama.
* Raspodijeliti agregirane metrike izvedbe kako biste automatski otkrili pogrešne kohorte i informirali svoje ciljne korake za ublažavanje problema.

## Pregled modela

Procjena izvedbe modela strojnog učenja zahtijeva holističko razumijevanje njegovog ponašanja. To se može postići pregledom više od jedne metrike, poput stope pogreške, točnosti, prisjećanja, preciznosti ili MAE (prosječne apsolutne pogreške), kako bi se otkrile razlike među metrikama izvedbe. Jedna metrika izvedbe može izgledati izvrsno, ali netočnosti se mogu otkriti u drugoj metriki. Osim toga, usporedba metrika za razlike u cijelom skupu podataka ili kohortama pomaže rasvijetliti gdje model dobro ili loše radi. Ovo je posebno važno za uočavanje izvedbe modela među osjetljivim i neosjetljivim značajkama (npr. rasa pacijenta, spol ili dob) kako bi se otkrila potencijalna nepravednost modela. Na primjer, otkrivanje da je model pogrešniji u kohorti koja ima osjetljive značajke može otkriti potencijalnu nepravednost modela.

Komponenta Pregled modela na nadzornoj ploči RAI pomaže ne samo u analizi metrika izvedbe reprezentacije podataka u kohorti, već korisnicima daje mogućnost usporedbe ponašanja modela među različitim kohortama.

![Kohorte skupa podataka - pregled modela na nadzornoj ploči RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Funkcionalnost analize temeljene na značajkama komponente omogućuje korisnicima sužavanje podskupina podataka unutar određene značajke kako bi se identificirale anomalije na detaljnoj razini. Na primjer, nadzorna ploča ima ugrađenu inteligenciju za automatsko generiranje kohorti za korisnički odabranu značajku (npr. *"time_in_hospital < 3"* ili *"time_in_hospital >= 7"*). Ovo omogućuje korisniku da izolira određenu značajku iz veće skupine podataka kako bi vidio je li ona ključni čimbenik netočnih ishoda modela.

![Kohorte značajki - pregled modela na nadzornoj ploči RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Komponenta Pregled modela podržava dvije klase metrika razlika:

**Razlike u izvedbi modela**: Ovi skupovi metrika izračunavaju razliku u vrijednostima odabrane metrike izvedbe među podskupinama podataka. Evo nekoliko primjera:

* Razlika u stopi točnosti
* Razlika u stopi pogreške
* Razlika u preciznosti
* Razlika u prisjećanju
* Razlika u prosječnoj apsolutnoj pogrešci (MAE)

**Razlike u stopi odabira**: Ova metrika sadrži razliku u stopi odabira (povoljne predikcije) među podskupinama. Primjer ovoga je razlika u stopama odobravanja kredita. Stopa odabira znači udio točaka podataka u svakoj klasi klasificiranih kao 1 (u binarnoj klasifikaciji) ili distribuciju vrijednosti predikcije (u regresiji).

## Analiza podataka

> "Ako dovoljno dugo mučite podatke, priznat će bilo što" - Ronald Coase

Ova izjava zvuči ekstremno, ali istina je da se podaci mogu manipulirati kako bi podržali bilo koji zaključak. Takva manipulacija ponekad se događa nenamjerno. Kao ljudi, svi imamo pristranosti i često je teško svjesno znati kada unosimo pristranost u podatke. Osiguravanje pravednosti u AI-u i strojnome učenju ostaje složen izazov.

Podaci su velika slijepa točka za tradicionalne metrike izvedbe modela. Možete imati visoke rezultate točnosti, ali to ne odražava uvijek osnovnu pristranost podataka koja bi mogla postojati u vašem skupu podataka. Na primjer, ako skup podataka zaposlenika ima 27% žena na izvršnim pozicijama u tvrtki i 73% muškaraca na istoj razini, AI model za oglašavanje poslova obučen na ovim podacima mogao bi ciljati uglavnom mušku publiku za visoke pozicije. Ova neravnoteža u podacima iskrivila je predikciju modela u korist jednog spola. Ovo otkriva problem pravednosti gdje postoji rodna pristranost u AI modelu.

Komponenta Analiza podataka na nadzornoj ploči RAI pomaže identificirati područja gdje postoji prekomjerna ili nedovoljna zastupljenost u skupu podataka. Pomaže korisnicima dijagnosticirati uzrok pogrešaka i problema pravednosti koji su uvedeni neravnotežom podataka ili nedostatkom zastupljenosti određene skupine podataka. Ovo korisnicima omogućuje vizualizaciju skupova podataka na temelju predviđenih i stvarnih ishoda, skupina pogrešaka i specifičnih značajki. Ponekad otkrivanje nedovoljno zastupljene skupine podataka također može otkriti da model ne uči dobro, što rezultira visokim netočnostima. Model s pristranošću podataka nije samo problem pravednosti, već pokazuje da model nije uključiv ili pouzdan.

![Komponenta za analizu podataka na nadzornoj ploči RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Koristite analizu podataka kada trebate:

* Istražiti statistiku svog skupa podataka odabirom različitih filtera za razdvajanje podataka u različite dimenzije (poznate i kao kohorte).
* Razumjeti distribuciju svog skupa podataka među različitim kohortama i skupinama značajki.
* Utvrditi jesu li vaši nalazi vezani za pravednost, analizu pogrešaka i uzročnost (dobiveni iz drugih komponenti nadzorne ploče) rezultat distribucije vašeg skupa podataka.
* Odlučiti u kojim područjima prikupiti više podataka kako biste ublažili pogreške koje proizlaze iz problema zastupljenosti, buke oznaka, buke značajki, pristranosti oznaka i sličnih čimbenika.

## Interpretacija modela

Modeli strojnog učenja često se smatraju "crnim kutijama". Razumijevanje koje ključne značajke podataka pokreću predikciju modela može biti izazovno. Važno je pružiti transparentnost u vezi s time zašto model donosi određenu predikciju. Na primjer, ako AI sustav predviđa da je dijabetičar u riziku od ponovnog prijema u bolnicu unutar 30 dana, trebao bi moći pružiti potporne podatke koji su doveli do te predikcije. Imati potporne pokazatelje podataka donosi transparentnost kako bi liječnici ili bolnice mogli donositi dobro informirane odluke. Osim toga, mogućnost objašnjenja zašto je model donio predikciju za pojedinog pacijenta omogućuje odgovornost prema zdravstvenim propisima. Kada koristite modele strojnog učenja na načine koji utječu na ljudske živote, ključno je razumjeti i objasniti što utječe na ponašanje modela. Interpretacija i objašnjivost modela pomažu odgovoriti na pitanja u scenarijima kao što su:

* Otklanjanje pogrešaka modela: Zašto je moj model napravio ovu pogrešku? Kako mogu poboljšati svoj model?
* Suradnja čovjeka i AI-a: Kako mogu razumjeti i vjerovati odlukama modela?
* Usklađenost s propisima: Zadovoljava li moj model zakonske zahtjeve?

Komponenta Važnost značajki na nadzornoj ploči RAI pomaže vam otkloniti pogreške i dobiti sveobuhvatno razumijevanje kako model donosi predikcije. Također je koristan alat za stručnjake za strojno učenje i donositelje odluka kako bi objasnili i pokazali dokaze o značajkama koje utječu na ponašanje modela radi usklađenosti s propisima. Korisnici zatim mogu istražiti globalna i lokalna objašnjenja kako bi potvrdili koje značajke pokreću predikciju modela. Globalna objašnjenja navode glavne značajke koje su utjecale na ukupnu predikciju modela. Lokalna objašnjenja prikazuju koje su značajke dovele do predikcije modela za pojedinačni slučaj. Mogućnost procjene lokalnih objašnjenja također je korisna u otklanjanju pogrešaka ili reviziji određenog slučaja kako bi se bolje razumjelo i interpretiralo zašto je model donio točnu ili netočnu predikciju.

![Komponenta Važnost značajki na nadzornoj ploči RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Globalna objašnjenja: Na primjer, koje značajke utječu na ukupno ponašanje modela za ponovno prijem dijabetičara u bolnicu?
* Lokalna objašnjenja: Na primjer, zašto je dijabetičar stariji od 60 godina s prethodnim hospitalizacijama predviđen da će biti ponovno primljen ili neće biti ponovno primljen u bolnicu unutar 30 dana?

U procesu otklanjanja pogrešaka i ispitivanja izvedbe modela među različitim kohortama, Važnost značajki pokazuje koliki utjecaj određena značajka ima među kohortama. Pomaže otkriti anomalije kada se uspoređuje razina utjecaja značajke na pogrešne predikcije modela. Komponenta Važnost značajki može pokazati koje vrijednosti u značajci pozitivno ili negativno utječu na ishod modela. Na primjer, ako je model donio netočnu predikciju, komponenta vam omogućuje da detaljno istražite i identificirate koje značajke ili vrijednosti značajki su dovele do predikcije. Ova razina detalja pomaže ne samo u otklanjanju pogrešaka već pruža transparentnost i odgovornost u situacijama revizije. Konačno, komponenta može pomoći u identificiranju problema pravednosti. Na primjer, ako osjetljiva značajka poput etničke pripadnosti ili spola ima veliki utjecaj na predikciju modela, to bi mogao biti znak rasne ili rodne pristranosti u modelu.

![Važnost značajki](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Koristite interpretaciju kada trebate:

* Odrediti koliko su pouzdane predikcije vašeg AI sustava razumijevanjem koje su značajke najvažnije za predikcije.
* Pristupiti otklanjanju pogrešaka modela tako da ga prvo razumijete i identificirate koristi li model zdrave značajke ili samo lažne korelacije.
* Otkriti potencijalne izvore nepravednosti razumijevanjem
- **Prekomjerna ili nedovoljna zastupljenost**. Ideja je da određena skupina nije zastupljena u određenoj profesiji, a svaka usluga ili funkcija koja to nastavlja promovirati doprinosi šteti.

### Azure RAI nadzorna ploča

[Azure RAI nadzorna ploča](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) temelji se na alatima otvorenog koda koje su razvile vodeće akademske institucije i organizacije, uključujući Microsoft, a koji su ključni za podatkovne znanstvenike i AI programere kako bi bolje razumjeli ponašanje modela, otkrili i ublažili neželjene probleme iz AI modela.

- Naučite kako koristiti različite komponente pregledavajući [dokumentaciju](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) RAI nadzorne ploče.

- Pogledajte neke [primjere bilježnica](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) RAI nadzorne ploče za otklanjanje problema u scenarijima odgovornog AI-a u Azure Machine Learningu.

---
## 🚀 Izazov

Kako bismo spriječili uvođenje statističkih ili podatkovnih pristranosti od samog početka, trebali bismo:

- osigurati raznolikost pozadina i perspektiva među ljudima koji rade na sustavima
- ulagati u skupove podataka koji odražavaju raznolikost našeg društva
- razviti bolje metode za otkrivanje i ispravljanje pristranosti kada se pojavi

Razmislite o stvarnim scenarijima u kojima je nepravda očita u izgradnji i korištenju modela. Što još trebamo uzeti u obzir?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)
## Pregled i samostalno učenje

U ovoj lekciji naučili ste neke praktične alate za uključivanje odgovornog AI-a u strojno učenje.

Pogledajte ovu radionicu za dublje istraživanje tema:

- Odgovorna AI nadzorna ploča: Sve na jednom mjestu za operacionalizaciju RAI-a u praksi, autorice Besmira Nushi i Mehrnoosh Sameki

[![Odgovorna AI nadzorna ploča: Sve na jednom mjestu za operacionalizaciju RAI-a u praksi](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Odgovorna AI nadzorna ploča: Sve na jednom mjestu za operacionalizaciju RAI-a u praksi")

> 🎥 Kliknite na sliku iznad za video: Odgovorna AI nadzorna ploča: Sve na jednom mjestu za operacionalizaciju RAI-a u praksi, autorice Besmira Nushi i Mehrnoosh Sameki

Referencirajte sljedeće materijale kako biste saznali više o odgovornom AI-u i kako izgraditi pouzdanije modele:

- Microsoftovi alati RAI nadzorne ploče za otklanjanje problema u ML modelima: [Resursi alata za odgovorni AI](https://aka.ms/rai-dashboard)

- Istražite alatni okvir za odgovorni AI: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsoftov centar resursa za RAI: [Resursi za odgovorni AI – Microsoft AI](https://www.microsoft.com/en-us/ai/tools-practices)

- Microsoftova istraživačka grupa FATE: [FATE: Pravednost, odgovornost, transparentnost i etika u AI-u - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Zadatak

[Istražite RAI nadzornu ploču](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.