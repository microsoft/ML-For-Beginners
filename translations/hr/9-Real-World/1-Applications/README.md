<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T12:24:30+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "hr"
}
-->
# Postscript: Strojno učenje u stvarnom svijetu

![Sažetak strojnog učenja u stvarnom svijetu u obliku sketchnotea](../../../../sketchnotes/ml-realworld.png)
> Sketchnote autorice [Tomomi Imura](https://www.twitter.com/girlie_mac)

U ovom kurikulumu naučili ste mnoge načine pripreme podataka za treniranje i izradu modela strojnog učenja. Izradili ste niz klasičnih modela za regresiju, klasteriranje, klasifikaciju, obradu prirodnog jezika i vremenske serije. Čestitamo! Sada se možda pitate čemu sve to... koje su stvarne primjene ovih modela?

Iako je industrija pokazala veliki interes za AI, koji obično koristi duboko učenje, klasični modeli strojnog učenja i dalje imaju vrijedne primjene. Možda već danas koristite neke od tih primjena! U ovoj lekciji istražit ćete kako osam različitih industrija i područja primjene koriste ove vrste modela kako bi njihove aplikacije bile učinkovitije, pouzdanije, inteligentnije i korisnije za korisnike.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Financije

Financijski sektor nudi mnoge prilike za primjenu strojnog učenja. Mnogi problemi u ovom području mogu se modelirati i riješiti pomoću ML-a.

### Otkrivanje prijevara s kreditnim karticama

Ranije u tečaju naučili smo o [k-means klasteriranju](../../5-Clustering/2-K-Means/README.md), ali kako se ono može koristiti za rješavanje problema povezanih s prijevarama s kreditnim karticama?

K-means klasteriranje korisno je u tehnici otkrivanja prijevara s kreditnim karticama koja se naziva **otkrivanje odstupanja**. Odstupanja, ili devijacije u opažanjima skupa podataka, mogu nam pokazati koristi li se kreditna kartica na uobičajen način ili se događa nešto neobično. Kao što je prikazano u povezanom radu, podatke o kreditnim karticama možete sortirati pomoću k-means algoritma klasteriranja i dodijeliti svaku transakciju klasteru na temelju toga koliko odstupa od norme. Zatim možete procijeniti najrizičnije klastere kako biste razlikovali prijevarne od legitimnih transakcija.
[Referenca](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Upravljanje bogatstvom

U upravljanju bogatstvom, pojedinac ili tvrtka upravlja investicijama u ime svojih klijenata. Njihov je posao dugoročno održavati i povećavati bogatstvo, pa je ključno odabrati investicije koje dobro performiraju.

Jedan od načina procjene performansi određene investicije je statistička regresija. [Linearna regresija](../../2-Regression/1-Tools/README.md) vrijedna je alatka za razumijevanje kako fond performira u odnosu na neki referentni pokazatelj. Također možemo zaključiti jesu li rezultati regresije statistički značajni, odnosno koliko bi mogli utjecati na investicije klijenta. Analizu možete dodatno proširiti koristeći višestruku regresiju, gdje se mogu uzeti u obzir dodatni faktori rizika. Za primjer kako bi to funkcioniralo za određeni fond, pogledajte rad u nastavku o procjeni performansi fonda pomoću regresije.
[Referenca](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Obrazovanje

Obrazovni sektor također je vrlo zanimljivo područje za primjenu ML-a. Postoje zanimljivi problemi koje treba riješiti, poput otkrivanja varanja na testovima ili esejima, ili upravljanja pristranostima, namjernim ili nenamjernim, u procesu ocjenjivanja.

### Predviđanje ponašanja studenata

[Coursera](https://coursera.com), pružatelj online otvorenih tečajeva, ima odličan tehnički blog gdje raspravlja o mnogim inženjerskim odlukama. U ovom studiju slučaja, nacrtali su regresijsku liniju kako bi istražili postoji li korelacija između niskog NPS (Net Promoter Score) ocjenjivanja i zadržavanja ili odustajanja od tečaja.
[Referenca](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Ublažavanje pristranosti

[Grammarly](https://grammarly.com), asistent za pisanje koji provjerava pravopisne i gramatičke pogreške, koristi sofisticirane [sustave za obradu prirodnog jezika](../../6-NLP/README.md) u svojim proizvodima. Objavili su zanimljiv studij slučaja na svom tehničkom blogu o tome kako su se nosili s rodnom pristranošću u strojnim modelima, što ste naučili u našoj [uvodnoj lekciji o pravednosti](../../1-Introduction/3-fairness/README.md).
[Referenca](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Maloprodaja

Maloprodajni sektor definitivno može imati koristi od primjene ML-a, od stvaranja boljeg korisničkog iskustva do optimalnog upravljanja zalihama.

### Personalizacija korisničkog iskustva

U Wayfairu, tvrtki koja prodaje kućne potrepštine poput namještaja, pomaganje kupcima da pronađu proizvode koji odgovaraju njihovom ukusu i potrebama je od ključne važnosti. U ovom članku, inženjeri iz tvrtke opisuju kako koriste ML i NLP za "prikazivanje pravih rezultata za kupce". Njihov Query Intent Engine izgrađen je za korištenje ekstrakcije entiteta, treniranja klasifikatora, ekstrakcije stavova i mišljenja te označavanja sentimenta u recenzijama kupaca. Ovo je klasičan primjer kako NLP funkcionira u online maloprodaji.
[Referenca](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Upravljanje zalihama

Inovativne, agilne tvrtke poput [StitchFix](https://stitchfix.com), usluge kutija koja šalje odjeću potrošačima, uvelike se oslanjaju na ML za preporuke i upravljanje zalihama. Njihovi timovi za stiliziranje surađuju s timovima za nabavu: "jedan od naših znanstvenika za podatke eksperimentirao je s genetskim algoritmom i primijenio ga na odjeću kako bi predvidio što bi bio uspješan komad odjeće koji danas ne postoji. To smo predstavili timu za nabavu i sada to mogu koristiti kao alat."
[Referenca](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Zdravstvo

Sektor zdravstva može koristiti ML za optimizaciju istraživačkih zadataka, kao i logističkih problema poput ponovnog prijema pacijenata ili zaustavljanja širenja bolesti.

### Upravljanje kliničkim ispitivanjima

Toksičnost u kliničkim ispitivanjima veliki je problem za proizvođače lijekova. Koliko je toksičnosti prihvatljivo? U ovom istraživanju, analiza različitih metoda kliničkih ispitivanja dovela je do razvoja novog pristupa za predviđanje ishoda kliničkih ispitivanja. Konkretno, koristili su random forest za izradu [klasifikatora](../../4-Classification/README.md) koji može razlikovati skupine lijekova.
[Referenca](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Upravljanje ponovnim prijemom u bolnicu

Bolničko liječenje je skupo, posebno kada se pacijenti moraju ponovno primiti. Ovaj rad raspravlja o tvrtki koja koristi ML za predviđanje potencijala ponovnog prijema pomoću [klasteriranja](../../5-Clustering/README.md) algoritama. Ovi klasteri pomažu analitičarima da "otkriju skupine ponovnih prijema koji mogu dijeliti zajednički uzrok".
[Referenca](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Upravljanje bolestima

Nedavna pandemija bacila je svjetlo na načine na koje strojno učenje može pomoći u zaustavljanju širenja bolesti. U ovom članku prepoznat ćete korištenje ARIMA, logističkih krivulja, linearne regresije i SARIMA. "Ovaj rad pokušava izračunati stopu širenja ovog virusa i tako predvidjeti smrti, oporavke i potvrđene slučajeve, kako bi nam pomogao da se bolje pripremimo i preživimo."
[Referenca](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologija i zelena tehnologija

Priroda i ekologija sastoje se od mnogih osjetljivih sustava gdje interakcija između životinja i prirode dolazi u fokus. Važno je moći precizno mjeriti ove sustave i djelovati prikladno ako se nešto dogodi, poput šumskog požara ili pada populacije životinja.

### Upravljanje šumama

Naučili ste o [Pojačanom učenju](../../8-Reinforcement/README.md) u prethodnim lekcijama. Može biti vrlo korisno kada pokušavate predvidjeti obrasce u prirodi. Konkretno, može se koristiti za praćenje ekoloških problema poput šumskih požara i širenja invazivnih vrsta. U Kanadi, grupa istraživača koristila je Pojačano učenje za izradu modela dinamike šumskih požara iz satelitskih snimaka. Koristeći inovativni "proces prostornog širenja (SSP)", zamislili su šumski požar kao "agenta na bilo kojoj ćeliji u krajoliku." "Skup akcija koje požar može poduzeti s lokacije u bilo kojem trenutku uključuje širenje na sjever, jug, istok ili zapad ili ne širenje.

Ovaj pristup obrće uobičajeni RL postav budući da su dinamike odgovarajućeg Markovljevog procesa odlučivanja (MDP) poznata funkcija za trenutno širenje požara." Pročitajte više o klasičnim algoritmima koje je koristila ova grupa na poveznici u nastavku.
[Referenca](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Praćenje kretanja životinja

Iako je duboko učenje stvorilo revoluciju u vizualnom praćenju kretanja životinja (možete izraditi vlastiti [tracker za polarne medvjede](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) ovdje), klasični ML i dalje ima svoje mjesto u ovom zadatku.

Senzori za praćenje kretanja domaćih životinja i IoT koriste ovu vrstu vizualne obrade, ali osnovnije ML tehnike korisne su za predobradu podataka. Na primjer, u ovom radu, držanje ovaca praćeno je i analizirano pomoću različitih algoritama klasifikatora. Možda ćete prepoznati ROC krivulju na stranici 335.
[Referenca](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Upravljanje energijom

U našim lekcijama o [prognoziranju vremenskih serija](../../7-TimeSeries/README.md), spomenuli smo koncept pametnih parkirnih metara za generiranje prihoda za grad na temelju razumijevanja ponude i potražnje. Ovaj članak detaljno raspravlja o tome kako su klasteriranje, regresija i prognoziranje vremenskih serija kombinirani kako bi se predvidjela buduća potrošnja energije u Irskoj, na temelju pametnog mjerenja.
[Referenca](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Osiguranje

Sektor osiguranja još je jedno područje koje koristi ML za izradu i optimizaciju održivih financijskih i aktuarskih modela.

### Upravljanje volatilnošću

MetLife, pružatelj životnog osiguranja, otvoreno govori o načinu na koji analizira i ublažava volatilnost u svojim financijskim modelima. U ovom članku primijetit ćete vizualizacije binarne i ordinalne klasifikacije. Također ćete otkriti vizualizacije prognoziranja.
[Referenca](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Umjetnost, kultura i književnost

U umjetnosti, primjerice u novinarstvu, postoje mnogi zanimljivi problemi. Otkrivanje lažnih vijesti veliki je problem jer se pokazalo da utječe na mišljenje ljudi, pa čak i na rušenje demokracija. Muzeji također mogu imati koristi od korištenja ML-a u svemu, od pronalaženja poveznica između artefakata do planiranja resursa.

### Otkrivanje lažnih vijesti

Otkrivanje lažnih vijesti postalo je igra mačke i miša u današnjim medijima. U ovom članku, istraživači predlažu sustav koji kombinira nekoliko ML tehnika koje smo proučavali i testira najbolji model: "Ovaj sustav temelji se na obradi prirodnog jezika za ekstrakciju značajki iz podataka, a zatim se te značajke koriste za treniranje klasifikatora strojnog učenja kao što su Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) i Logistic Regression (LR)."
[Referenca](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Ovaj članak pokazuje kako kombiniranje različitih ML domena može proizvesti zanimljive rezultate koji mogu pomoći u zaustavljanju širenja lažnih vijesti i stvaranju stvarne štete; u ovom slučaju, poticaj je bilo širenje glasina o COVID tretmanima koje su izazvale nasilje.

### Muzejski ML

Muzeji su na pragu AI revolucije u kojoj katalogiziranje i digitalizacija zbirki te pronalaženje poveznica između artefakata postaje lakše kako tehnologija napreduje. Projekti poput [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) pomažu u otkrivanju misterija nedostupnih zbirki poput Vatikanskih arhiva. No, poslovni aspekt muzeja također ima koristi od ML modela.

Na primjer, Umjetnički institut u Chicagu izradio je modele za predviđanje interesa publike i vremena kada će posjetiti izložbe. Cilj je stvoriti individualizirana i optimizirana iskustva posjetitelja svaki put kada korisnik posjeti muzej. "Tijekom fiskalne 2017. godine, model je predvidio posjećenost i prihode s točnošću od 1 posto, kaže Andrew Simnick, viši potpredsjednik u Umjetničkom institutu."
[Referenca](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentacija kupaca

Najučinkovitije marketinške strategije ciljaju kupce na različite načine na temelju različitih grupiranja. U ovom članku raspravlja se o primjeni algoritama klasteriranja za podršku diferenciranom marketingu. Diferencirani marketing pomaže tvrtkama poboljšati prepoznatljivost brenda, dosegnuti više kupaca i ostvariti veći prihod.
[Referenca](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Izazov

Identificirajte još jedan sektor koji koristi neke od tehnika koje ste naučili u ovom kurikulumu i otkrijte kako koristi ML.
## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Tim za podatkovnu znanost u Wayfairu ima nekoliko zanimljivih videa o tome kako koriste ML u svojoj tvrtki. Vrijedi [pogledati](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Zadatak

[Potraga za ML](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za kritične informacije preporučuje se profesionalni prijevod od strane čovjeka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.