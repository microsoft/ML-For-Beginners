<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T12:43:57+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "sl"
}
-->
# Tehnike strojnega učenja

Proces gradnje, uporabe in vzdrževanja modelov strojnega učenja ter podatkov, ki jih uporabljajo, se močno razlikuje od mnogih drugih razvojnih delovnih tokov. V tej lekciji bomo razjasnili ta proces in predstavili glavne tehnike, ki jih morate poznati. Naučili se boste:

- Razumeti procese, ki so osnova strojnega učenja na visoki ravni.
- Raziskati osnovne koncepte, kot so 'modeli', 'napovedi' in 'podatki za učenje'.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

[![ML za začetnike - Tehnike strojnega učenja](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML za začetnike - Tehnike strojnega učenja")

> 🎥 Kliknite zgornjo sliko za kratek video, ki obravnava to lekcijo.

## Uvod

Na visoki ravni je proces ustvarjanja strojnega učenja (ML) sestavljen iz več korakov:

1. **Odločite se za vprašanje**. Večina procesov ML se začne z vprašanjem, na katerega ni mogoče odgovoriti z enostavnim pogojnim programom ali sistemom, ki temelji na pravilih. Ta vprašanja se pogosto vrtijo okoli napovedi na podlagi zbirke podatkov.
2. **Zberite in pripravite podatke**. Da bi lahko odgovorili na svoje vprašanje, potrebujete podatke. Kakovost in včasih količina vaših podatkov bosta določili, kako dobro lahko odgovorite na začetno vprašanje. Vizualizacija podatkov je pomemben vidik te faze. Ta faza vključuje tudi razdelitev podatkov na skupino za učenje in testiranje za gradnjo modela.
3. **Izberite metodo učenja**. Glede na vaše vprašanje in naravo podatkov morate izbrati način, kako želite model naučiti, da bo najbolje odražal vaše podatke in podajal natančne napovedi. Ta del procesa ML zahteva specifično strokovno znanje in pogosto veliko eksperimentiranja.
4. **Naučite model**. Z uporabo podatkov za učenje boste uporabili različne algoritme za učenje modela, da prepozna vzorce v podatkih. Model lahko uporablja notranje uteži, ki jih je mogoče prilagoditi, da daje prednost določenim delom podatkov pred drugimi, da zgradi boljši model.
5. **Ocenite model**. Uporabite podatke, ki jih model še ni videl (vaše testne podatke), da preverite, kako se model obnese.
6. **Prilagodite parametre**. Na podlagi uspešnosti modela lahko proces ponovite z različnimi parametri ali spremenljivkami, ki nadzorujejo vedenje algoritmov, uporabljenih za učenje modela.
7. **Napovedujte**. Uporabite nove vnose za testiranje natančnosti vašega modela.

## Kakšno vprašanje zastaviti

Računalniki so še posebej spretni pri odkrivanju skritih vzorcev v podatkih. Ta uporabnost je zelo koristna za raziskovalce, ki imajo vprašanja o določenem področju, na katera ni mogoče enostavno odgovoriti z ustvarjanjem sistema, ki temelji na pogojnih pravilih. Pri aktuarskih nalogah, na primer, bi lahko podatkovni znanstvenik oblikoval ročno izdelana pravila o smrtnosti kadilcev v primerjavi z nekadilci.

Ko pa v enačbo vključimo veliko drugih spremenljivk, se lahko model ML izkaže za bolj učinkovitega pri napovedovanju prihodnjih stopenj smrtnosti na podlagi pretekle zdravstvene zgodovine. Bolj vesel primer bi lahko bila napoved vremena za mesec april na določenem območju na podlagi podatkov, ki vključujejo zemljepisno širino, dolžino, podnebne spremembe, bližino oceana, vzorce zračnih tokov in še več.

✅ Ta [predstavitev](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) o vremenskih modelih ponuja zgodovinski pogled na uporabo ML pri analizi vremena.  

## Naloge pred gradnjo

Preden začnete graditi svoj model, morate opraviti več nalog. Da bi preizkusili svoje vprašanje in oblikovali hipotezo na podlagi napovedi modela, morate identificirati in konfigurirati več elementov.

### Podatki

Da bi lahko odgovorili na svoje vprašanje z določeno stopnjo gotovosti, potrebujete zadostno količino podatkov ustrezne vrste. Na tej točki morate storiti dve stvari:

- **Zberite podatke**. Ob upoštevanju prejšnje lekcije o pravičnosti pri analizi podatkov zbirajte podatke skrbno. Bodite pozorni na vire teh podatkov, morebitne inherentne pristranskosti in dokumentirajte njihov izvor.
- **Pripravite podatke**. Obstaja več korakov v procesu priprave podatkov. Morda boste morali združiti podatke in jih normalizirati, če prihajajo iz različnih virov. Kakovost in količino podatkov lahko izboljšate z različnimi metodami, kot je pretvorba nizov v številke (kot to počnemo pri [Gručenju](../../5-Clustering/1-Visualize/README.md)). Morda boste ustvarili nove podatke na podlagi izvirnih (kot to počnemo pri [Klasifikaciji](../../4-Classification/1-Introduction/README.md)). Podatke lahko očistite in uredite (kot bomo storili pred lekcijo o [Spletni aplikaciji](../../3-Web-App/README.md)). Na koncu jih boste morda morali naključno razporediti in premešati, odvisno od vaših tehnik učenja.

✅ Po zbiranju in obdelavi podatkov si vzemite trenutek, da preverite, ali njihova oblika omogoča obravnavo zastavljenega vprašanja. Morda se izkaže, da podatki ne bodo dobro delovali pri vaši nalogi, kot odkrijemo v naših lekcijah o [Gručenju](../../5-Clustering/1-Visualize/README.md)!

### Značilnosti in cilj

[Značilnost](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) je merljiva lastnost vaših podatkov. V mnogih podatkovnih nizih je izražena kot naslov stolpca, kot so 'datum', 'velikost' ali 'barva'. Vaša spremenljivka značilnosti, običajno predstavljena kot `X` v kodi, predstavlja vhodno spremenljivko, ki bo uporabljena za učenje modela.

Cilj je stvar, ki jo poskušate napovedati. Cilj, običajno predstavljen kot `y` v kodi, predstavlja odgovor na vprašanje, ki ga poskušate zastaviti svojim podatkom: v decembru, kakšne **barve** bodo najcenejše buče? v San Franciscu, katera soseska bo imela najboljšo ceno **nepremičnin**? Včasih se cilj imenuje tudi atribut oznake.

### Izbor spremenljivke značilnosti

🎓 **Izbor značilnosti in ekstrakcija značilnosti** Kako veste, katero spremenljivko izbrati pri gradnji modela? Verjetno boste šli skozi proces izbora značilnosti ali ekstrakcije značilnosti, da izberete prave spremenljivke za najbolj zmogljiv model. Vendar pa to ni isto: "Ekstrakcija značilnosti ustvarja nove značilnosti iz funkcij izvirnih značilnosti, medtem ko izbor značilnosti vrne podmnožico značilnosti." ([vir](https://wikipedia.org/wiki/Feature_selection))

### Vizualizirajte svoje podatke

Pomemben vidik orodij podatkovnega znanstvenika je moč vizualizacije podatkov z uporabo več odličnih knjižnic, kot sta Seaborn ali MatPlotLib. Vizualizacija podatkov vam lahko omogoči odkrivanje skritih korelacij, ki jih lahko izkoristite. Vaše vizualizacije vam lahko pomagajo tudi pri odkrivanju pristranskosti ali neuravnoteženih podatkov (kot odkrijemo pri [Klasifikaciji](../../4-Classification/2-Classifiers-1/README.md)).

### Razdelite svoj podatkovni niz

Pred učenjem morate razdeliti svoj podatkovni niz na dva ali več delov neenake velikosti, ki še vedno dobro predstavljajo podatke.

- **Učenje**. Ta del podatkovnega niza se prilega vašemu modelu, da ga nauči. Ta niz predstavlja večino izvirnega podatkovnega niza.
- **Testiranje**. Testni podatkovni niz je neodvisna skupina podatkov, pogosto pridobljena iz izvirnih podatkov, ki jo uporabite za potrditev uspešnosti zgrajenega modela.
- **Validacija**. Validacijski niz je manjša neodvisna skupina primerov, ki jo uporabite za prilagoditev hiperparametrov ali arhitekture modela, da izboljšate model. Glede na velikost vaših podatkov in vprašanje, ki ga zastavljate, morda ne boste potrebovali gradnje tega tretjega niza (kot ugotavljamo pri [Napovedovanju časovnih vrst](../../7-TimeSeries/1-Introduction/README.md)).

## Gradnja modela

Z uporabo podatkov za učenje je vaš cilj zgraditi model ali statistično predstavitev vaših podatkov z uporabo različnih algoritmov za **učenje**. Učenje modela ga izpostavi podatkom in mu omogoči, da naredi predpostavke o zaznanih vzorcih, ki jih odkrije, potrdi in sprejme ali zavrne.

### Odločite se za metodo učenja

Glede na vaše vprašanje in naravo podatkov boste izbrali metodo za učenje. Če preučite [dokumentacijo Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - ki jo uporabljamo v tem tečaju - lahko raziščete številne načine za učenje modela. Glede na vaše izkušnje boste morda morali preizkusiti več različnih metod, da zgradite najboljši model. Verjetno boste šli skozi proces, pri katerem podatkovni znanstveniki ocenjujejo uspešnost modela z uporabo podatkov, ki jih model še ni videl, preverjajo natančnost, pristranskost in druge težave, ki zmanjšujejo kakovost, ter izbirajo najbolj primerno metodo učenja za obravnavano nalogo.

### Naučite model

Oboroženi s podatki za učenje ste pripravljeni, da jih 'prilagodite' za ustvarjanje modela. Opazili boste, da v mnogih knjižnicah ML najdete kodo 'model.fit' - to je trenutek, ko pošljete svojo spremenljivko značilnosti kot niz vrednosti (običajno 'X') in ciljno spremenljivko (običajno 'y').

### Ocenite model

Ko je proces učenja zaključen (za učenje velikega modela lahko traja veliko iteracij ali 'epoh'), boste lahko ocenili kakovost modela z uporabo testnih podatkov za oceno njegove uspešnosti. Ti podatki so podmnožica izvirnih podatkov, ki jih model še ni analiziral. Lahko natisnete tabelo metrik o kakovosti vašega modela.

🎓 **Prilagajanje modela**

V kontekstu strojnega učenja prilagajanje modela pomeni natančnost osnovne funkcije modela, ko poskuša analizirati podatke, s katerimi ni seznanjen.

🎓 **Premalo prilagajanje** in **prekomerno prilagajanje** sta pogosti težavi, ki zmanjšujeta kakovost modela, saj se model prilagodi bodisi premalo bodisi preveč. To povzroči, da model podaja napovedi bodisi preveč usklajene bodisi premalo usklajene s podatki za učenje. Prekomerno prilagojen model preveč dobro napoveduje podatke za učenje, ker se je preveč naučil podrobnosti in šuma podatkov. Premalo prilagojen model ni natančen, saj ne more natančno analizirati niti podatkov za učenje niti podatkov, ki jih še ni 'videl'.

![prekomerno prilagajanje modela](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografika avtorice [Jen Looper](https://twitter.com/jenlooper)

## Prilagoditev parametrov

Ko je vaše začetno učenje zaključeno, opazujte kakovost modela in razmislite o izboljšanju z nastavitvijo njegovih 'hiperparametrov'. Več o procesu preberite [v dokumentaciji](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Napovedovanje

To je trenutek, ko lahko uporabite popolnoma nove podatke za testiranje natančnosti vašega modela. V 'aplikativnem' okolju ML, kjer gradite spletne aplikacije za uporabo modela v produkciji, lahko ta proces vključuje zbiranje uporabniških vnosov (na primer pritisk na gumb), da nastavite spremenljivko in jo pošljete modelu za sklepanje ali oceno.

V teh lekcijah boste odkrili, kako uporabiti te korake za pripravo, gradnjo, testiranje, ocenjevanje in napovedovanje - vse korake podatkovnega znanstvenika in še več, ko napredujete na svoji poti, da postanete 'full stack' inženir strojnega učenja.

---

## 🚀Izziv

Narišite diagram poteka, ki odraža korake strokovnjaka za strojno učenje. Kje se trenutno vidite v procesu? Kje predvidevate, da boste imeli težave? Kaj se vam zdi enostavno?

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Poiščite spletne intervjuje s podatkovnimi znanstveniki, ki razpravljajo o svojem vsakdanjem delu. Tukaj je [eden](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Naloga

[Intervjuirajte podatkovnega znanstvenika](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za prevajanje z umetno inteligenco [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni človeški prevod. Ne prevzemamo odgovornosti za morebitne nesporazume ali napačne razlage, ki bi nastale zaradi uporabe tega prevoda.