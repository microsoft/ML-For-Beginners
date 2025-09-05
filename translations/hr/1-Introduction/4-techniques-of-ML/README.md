<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T12:43:14+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "hr"
}
-->
# Tehnike strojnog učenja

Proces izrade, korištenja i održavanja modela strojnog učenja te podataka koje koriste vrlo je različit od mnogih drugih razvojnih tijekova rada. U ovoj lekciji razjasnit ćemo taj proces i istaknuti glavne tehnike koje trebate znati. Naučit ćete:

- Razumjeti procese koji podržavaju strojno učenje na visokoj razini.
- Istražiti osnovne pojmove poput 'modela', 'predikcija' i 'podataka za treniranje'.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

[![ML za početnike - Tehnike strojnog učenja](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML za početnike - Tehnike strojnog učenja")

> 🎥 Kliknite na sliku iznad za kratki video koji obrađuje ovu lekciju.

## Uvod

Na visokoj razini, proces stvaranja strojnog učenja (ML) sastoji se od nekoliko koraka:

1. **Odlučite o pitanju**. Većina ML procesa započinje postavljanjem pitanja koje se ne može odgovoriti jednostavnim uvjetnim programom ili sustavom temeljenim na pravilima. Ta pitanja često se odnose na predikcije temeljene na zbirci podataka.
2. **Prikupite i pripremite podatke**. Da biste mogli odgovoriti na svoje pitanje, trebate podatke. Kvaliteta, a ponekad i količina vaših podataka, odredit će koliko dobro možete odgovoriti na početno pitanje. Vizualizacija podataka važan je aspekt ove faze. Ova faza također uključuje podjelu podataka na skup za treniranje i testiranje kako biste izgradili model.
3. **Odaberite metodu treniranja**. Ovisno o vašem pitanju i prirodi vaših podataka, trebate odabrati način na koji želite trenirati model kako bi najbolje odražavao vaše podatke i davao točne predikcije. Ovo je dio ML procesa koji zahtijeva specifičnu stručnost i često značajnu količinu eksperimentiranja.
4. **Trenirajte model**. Koristeći podatke za treniranje, koristit ćete razne algoritme za treniranje modela kako bi prepoznao obrasce u podacima. Model može koristiti unutarnje težine koje se mogu prilagoditi kako bi se privilegirali određeni dijelovi podataka u odnosu na druge za izgradnju boljeg modela.
5. **Procijenite model**. Koristite podatke koje model nikada prije nije vidio (vaše testne podatke) iz prikupljenog skupa kako biste vidjeli kako model funkcionira.
6. **Podešavanje parametara**. Na temelju performansi vašeg modela, možete ponoviti proces koristeći različite parametre ili varijable koje kontroliraju ponašanje algoritama korištenih za treniranje modela.
7. **Predikcija**. Koristite nove ulaze kako biste testirali točnost vašeg modela.

## Koje pitanje postaviti

Računala su posebno vješta u otkrivanju skrivenih obrazaca u podacima. Ova korisnost vrlo je korisna za istraživače koji imaju pitanja o određenom području koja se ne mogu lako odgovoriti stvaranjem sustava temeljenog na uvjetima. Na primjer, u aktuarskom zadatku, podatkovni znanstvenik mogao bi konstruirati ručno izrađena pravila o smrtnosti pušača u odnosu na nepušače.

Međutim, kada se u jednadžbu uključi mnogo drugih varijabli, ML model mogao bi se pokazati učinkovitijim za predviđanje budućih stopa smrtnosti na temelju povijesti zdravlja. Jedan veseliji primjer mogao bi biti izrada vremenskih predikcija za mjesec travanj na određenoj lokaciji na temelju podataka koji uključuju geografsku širinu, dužinu, klimatske promjene, blizinu oceana, obrasce mlaznih struja i više.

✅ Ova [prezentacija](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o vremenskim modelima nudi povijesnu perspektivu korištenja ML-a u analizi vremena.  

## Zadaci prije izgradnje

Prije nego što počnete graditi svoj model, postoji nekoliko zadataka koje trebate dovršiti. Kako biste testirali svoje pitanje i oblikovali hipotezu na temelju predikcija modela, trebate identificirati i konfigurirati nekoliko elemenata.

### Podaci

Da biste mogli odgovoriti na svoje pitanje s bilo kakvom sigurnošću, trebate dobru količinu podataka odgovarajućeg tipa. U ovom trenutku trebate učiniti dvije stvari:

- **Prikupiti podatke**. Imajući na umu prethodnu lekciju o pravednosti u analizi podataka, pažljivo prikupite svoje podatke. Budite svjesni izvora tih podataka, bilo kakvih inherentnih pristranosti koje bi mogli imati, i dokumentirajte njihovo podrijetlo.
- **Pripremiti podatke**. Postoji nekoliko koraka u procesu pripreme podataka. Možda ćete trebati objediniti podatke i normalizirati ih ako dolaze iz različitih izvora. Možete poboljšati kvalitetu i količinu podataka raznim metodama, poput pretvaranja stringova u brojeve (kao što radimo u [Klasterizaciji](../../5-Clustering/1-Visualize/README.md)). Možete također generirati nove podatke na temelju originalnih (kao što radimo u [Klasifikaciji](../../4-Classification/1-Introduction/README.md)). Možete očistiti i urediti podatke (kao što ćemo učiniti prije lekcije o [Web aplikaciji](../../3-Web-App/README.md)). Na kraju, možda ćete trebati nasumično rasporediti i promiješati podatke, ovisno o tehnikama treniranja.

✅ Nakon prikupljanja i obrade podataka, odvojite trenutak da vidite hoće li njihov oblik omogućiti da se pozabavite svojim namjeravanim pitanjem. Može se dogoditi da podaci neće dobro funkcionirati u vašem zadatku, kao što otkrivamo u našim lekcijama o [Klasterizaciji](../../5-Clustering/1-Visualize/README.md)!

### Značajke i cilj

[Značajka](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) je mjerljiva karakteristika vaših podataka. U mnogim skupovima podataka izražena je kao naslov stupca poput 'datum', 'veličina' ili 'boja'. Vaša varijabla značajke, obično predstavljena kao `X` u kodu, predstavlja ulaznu varijablu koja će se koristiti za treniranje modela.

Cilj je ono što pokušavate predvidjeti. Cilj, obično predstavljen kao `y` u kodu, predstavlja odgovor na pitanje koje pokušavate postaviti svojim podacima: u prosincu, koje će **boje** bundeve biti najjeftinije? U San Franciscu, koje će četvrti imati najbolju **cijenu** nekretnina? Ponekad se cilj također naziva atribut oznake.

### Odabir varijable značajke

🎓 **Odabir značajki i ekstrakcija značajki** Kako znati koju varijablu odabrati prilikom izrade modela? Vjerojatno ćete proći kroz proces odabira značajki ili ekstrakcije značajki kako biste odabrali prave varijable za najperformantniji model. Međutim, nisu iste stvari: "Ekstrakcija značajki stvara nove značajke iz funkcija originalnih značajki, dok odabir značajki vraća podskup značajki." ([izvor](https://wikipedia.org/wiki/Feature_selection))

### Vizualizirajte svoje podatke

Važan aspekt alata podatkovnog znanstvenika je moć vizualizacije podataka koristeći nekoliko izvrsnih biblioteka poput Seaborn ili MatPlotLib. Predstavljanje vaših podataka vizualno može vam omogućiti otkrivanje skrivenih korelacija koje možete iskoristiti. Vaše vizualizacije također vam mogu pomoći otkriti pristranost ili neuravnotežene podatke (kao što otkrivamo u [Klasifikaciji](../../4-Classification/2-Classifiers-1/README.md)).

### Podijelite svoj skup podataka

Prije treniranja, trebate podijeliti svoj skup podataka na dva ili više dijelova nejednake veličine koji i dalje dobro predstavljaju podatke.

- **Treniranje**. Ovaj dio skupa podataka koristi se za treniranje modela. Ovaj skup čini većinu originalnog skupa podataka.
- **Testiranje**. Testni skup podataka je neovisna grupa podataka, često prikupljena iz originalnih podataka, koju koristite za potvrdu performansi izgrađenog modela.
- **Validacija**. Skup za validaciju je manja neovisna grupa primjera koju koristite za podešavanje hiperparametara modela ili arhitekture kako biste poboljšali model. Ovisno o veličini vaših podataka i pitanju koje postavljate, možda nećete trebati izgraditi ovaj treći skup (kao što primjećujemo u [Predviđanju vremenskih serija](../../7-TimeSeries/1-Introduction/README.md)).

## Izrada modela

Koristeći podatke za treniranje, vaš cilj je izgraditi model, odnosno statistički prikaz vaših podataka, koristeći razne algoritme za **treniranje**. Treniranje modela izlaže ga podacima i omogućuje mu da donosi pretpostavke o uočenim obrascima koje otkriva, potvrđuje i prihvaća ili odbacuje.

### Odlučite o metodi treniranja

Ovisno o vašem pitanju i prirodi vaših podataka, odabrat ćete metodu za treniranje. Pregledavajući [dokumentaciju Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - koju koristimo u ovom tečaju - možete istražiti mnoge načine treniranja modela. Ovisno o vašem iskustvu, možda ćete morati isprobati nekoliko različitih metoda kako biste izgradili najbolji model. Vjerojatno ćete proći kroz proces u kojem podatkovni znanstvenici procjenjuju performanse modela hranjenjem neviđenih podataka, provjeravajući točnost, pristranost i druge probleme koji degradiraju kvalitetu te odabiru najprikladniju metodu treniranja za zadatak.

### Trenirajte model

Naoružani podacima za treniranje, spremni ste 'prilagoditi' ih kako biste stvorili model. Primijetit ćete da u mnogim ML bibliotekama postoji kod 'model.fit' - u ovom trenutku šaljete svoju varijablu značajke kao niz vrijednosti (obično 'X') i ciljnu varijablu (obično 'y').

### Procijenite model

Nakon što je proces treniranja završen (može potrajati mnogo iteracija ili 'epoha' za treniranje velikog modela), moći ćete procijeniti kvalitetu modela koristeći testne podatke za procjenu njegovih performansi. Ovi podaci su podskup originalnih podataka koje model prethodno nije analizirao. Možete ispisati tablicu metrike o kvaliteti vašeg modela.

🎓 **Prilagodba modela**

U kontekstu strojnog učenja, prilagodba modela odnosi se na točnost osnovne funkcije modela dok pokušava analizirati podatke s kojima nije upoznat.

🎓 **Premalo prilagođavanje** i **preveliko prilagođavanje** su uobičajeni problemi koji degradiraju kvalitetu modela, jer model ili ne odgovara dovoljno dobro ili previše dobro. To uzrokuje da model daje predikcije koje su ili previše usklađene ili premalo usklađene s podacima za treniranje. Previše prilagođen model predviđa podatke za treniranje previše dobro jer je previše naučio detalje i šum podataka. Premalo prilagođen model nije točan jer ne može ni točno analizirati podatke za treniranje niti podatke koje još nije 'vidio'.

![model prevelikog prilagođavanja](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografika od [Jen Looper](https://twitter.com/jenlooper)

## Podešavanje parametara

Nakon što je početno treniranje završeno, promatrajte kvalitetu modela i razmislite o njegovom poboljšanju podešavanjem njegovih 'hiperparametara'. Pročitajte više o procesu [u dokumentaciji](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predikcija

Ovo je trenutak kada možete koristiti potpuno nove podatke za testiranje točnosti vašeg modela. U 'primijenjenom' ML okruženju, gdje gradite web alate za korištenje modela u produkciji, ovaj proces može uključivati prikupljanje korisničkog unosa (na primjer, pritisak na gumb) za postavljanje varijable i slanje modelu za inferenciju ili procjenu.

U ovim lekcijama otkrit ćete kako koristiti ove korake za pripremu, izgradnju, testiranje, procjenu i predikciju - sve geste podatkovnog znanstvenika i više, dok napredujete na svom putu da postanete 'full stack' ML inženjer.

---

## 🚀Izazov

Nacrtajte dijagram toka koji odražava korake ML praktičara. Gdje se trenutno vidite u procesu? Gdje predviđate da ćete naići na poteškoće? Što vam se čini jednostavno?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Pretražite online intervjue s podatkovnim znanstvenicima koji raspravljaju o svom svakodnevnom radu. Evo [jednog](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Zadatak

[Intervjuirajte podatkovnog znanstvenika](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za kritične informacije preporučuje se profesionalni prijevod od strane čovjeka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.