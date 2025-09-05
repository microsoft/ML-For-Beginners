<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T12:47:19+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "hr"
}
-->
# Uvod u strojno učenje

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML za početnike - Uvod u strojno učenje za početnike](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML za početnike - Uvod u strojno učenje za početnike")

> 🎥 Kliknite na sliku iznad za kratki video koji prolazi kroz ovu lekciju.

Dobrodošli na ovaj tečaj klasičnog strojnog učenja za početnike! Bez obzira jeste li potpuno novi u ovoj temi ili ste iskusni praktičar strojnog učenja koji želi obnoviti znanje u određenom području, drago nam je što ste s nama! Želimo stvoriti prijateljsko polazište za vaše proučavanje strojnog učenja i rado ćemo procijeniti, odgovoriti na i uključiti vaše [povratne informacije](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Uvod u ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Uvod u ML")

> 🎥 Kliknite na sliku iznad za video: John Guttag s MIT-a uvodi u strojno učenje

---
## Početak sa strojnim učenjem

Prije nego što započnete s ovim kurikulumom, trebate pripremiti svoje računalo za lokalno pokretanje bilježnica.

- **Konfigurirajte svoje računalo pomoću ovih videa**. Koristite sljedeće poveznice kako biste naučili [kako instalirati Python](https://youtu.be/CXZYvNRIAKM) na svoj sustav i [postaviti uređivač teksta](https://youtu.be/EU8eayHWoZg) za razvoj.
- **Naučite Python**. Također se preporučuje osnovno razumijevanje [Pythona](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programskog jezika korisnog za znanstvenike podataka koji koristimo u ovom tečaju.
- **Naučite Node.js i JavaScript**. Također ćemo nekoliko puta koristiti JavaScript u ovom tečaju prilikom izrade web aplikacija, pa ćete trebati imati instalirane [node](https://nodejs.org) i [npm](https://www.npmjs.com/), kao i [Visual Studio Code](https://code.visualstudio.com/) za razvoj u Pythonu i JavaScriptu.
- **Kreirajte GitHub račun**. Budući da ste nas pronašli ovdje na [GitHubu](https://github.com), možda već imate račun, ali ako nemate, kreirajte ga i zatim forkajte ovaj kurikulum kako biste ga koristili sami. (Slobodno nam dajte zvjezdicu 😊)
- **Istražite Scikit-learn**. Upoznajte se s [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), skupom ML biblioteka na koje se pozivamo u ovim lekcijama.

---
## Što je strojno učenje?

Pojam 'strojno učenje' jedan je od najpopularnijih i najčešće korištenih pojmova današnjice. Postoji velika vjerojatnost da ste ovaj pojam čuli barem jednom ako imate bilo kakvu povezanost s tehnologijom, bez obzira na područje u kojem radite. Međutim, mehanika strojnog učenja za većinu ljudi ostaje misterij. Za početnika u strojnome učenju, tema ponekad može djelovati zastrašujuće. Stoga je važno razumjeti što strojno učenje zapravo jest i učiti o njemu korak po korak, kroz praktične primjere.

---
## Krivulja popularnosti

![krivulja popularnosti strojnog učenja](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends prikazuje nedavnu 'krivulju popularnosti' pojma 'strojno učenje'

---
## Tajanstveni svemir

Živimo u svemiru punom fascinantnih misterija. Veliki znanstvenici poput Stephena Hawkinga, Alberta Einsteina i mnogih drugih posvetili su svoje živote traženju značajnih informacija koje otkrivaju misterije svijeta oko nas. Ovo je ljudska potreba za učenjem: ljudsko dijete uči nove stvari i otkriva strukturu svog svijeta iz godine u godinu dok odrasta.

---
## Dječji mozak

Dječji mozak i osjetila percipiraju činjenice iz svoje okoline i postupno uče skrivene obrasce života koji pomažu djetetu da oblikuje logička pravila za prepoznavanje naučenih obrazaca. Proces učenja ljudskog mozga čini ljude najsloženijim živim bićima na svijetu. Kontinuirano učenje otkrivanjem skrivenih obrazaca, a zatim inoviranje na temelju tih obrazaca omogućuje nam da postajemo sve bolji tijekom života. Ova sposobnost učenja i evolucije povezana je s konceptom zvanim [plastičnost mozga](https://www.simplypsychology.org/brain-plasticity.html). Površno gledano, možemo povući neke motivacijske sličnosti između procesa učenja ljudskog mozga i koncepata strojnog učenja.

---
## Ljudski mozak

[Ljudski mozak](https://www.livescience.com/29365-human-brain.html) percipira stvari iz stvarnog svijeta, obrađuje percipirane informacije, donosi racionalne odluke i izvodi određene radnje na temelju okolnosti. To nazivamo inteligentnim ponašanjem. Kada programiramo imitaciju procesa inteligentnog ponašanja u stroj, to nazivamo umjetnom inteligencijom (AI).

---
## Neki pojmovi

Iako se pojmovi mogu zamijeniti, strojno učenje (ML) važan je podskup umjetne inteligencije. **ML se bavi korištenjem specijaliziranih algoritama za otkrivanje značajnih informacija i pronalaženje skrivenih obrazaca iz percipiranih podataka kako bi se podržao proces donošenja racionalnih odluka**.

---
## AI, ML, Duboko učenje

![AI, ML, duboko učenje, znanost o podacima](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Dijagram koji prikazuje odnose između AI, ML, dubokog učenja i znanosti o podacima. Infografika autorice [Jen Looper](https://twitter.com/jenlooper) inspirirana [ovom grafikom](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepti koje ćemo obraditi

U ovom kurikulumu obradit ćemo samo osnovne koncepte strojnog učenja koje početnik mora znati. Obradit ćemo ono što nazivamo 'klasičnim strojnim učenjem', primarno koristeći Scikit-learn, izvrsnu biblioteku koju mnogi studenti koriste za učenje osnova. Za razumijevanje šireg konteksta umjetne inteligencije ili dubokog učenja, snažno temeljno znanje o strojnome učenju je neophodno, i to vam želimo ponuditi ovdje.

---
## Na ovom tečaju naučit ćete:

- osnovne koncepte strojnog učenja
- povijest ML-a
- ML i pravednost
- tehnike regresijskog ML-a
- tehnike klasifikacijskog ML-a
- tehnike klasteriranja ML-a
- tehnike obrade prirodnog jezika ML-a
- tehnike predviđanja vremenskih serija ML-a
- učenje pojačanjem
- primjene ML-a u stvarnom svijetu

---
## Što nećemo obraditi

- duboko učenje
- neuronske mreže
- AI

Kako bismo omogućili bolje iskustvo učenja, izbjeći ćemo složenosti neuronskih mreža, 'dubokog učenja' - modeliranja s više slojeva koristeći neuronske mreže - i AI, o čemu ćemo raspravljati u drugom kurikulumu. Također ćemo ponuditi nadolazeći kurikulum o znanosti o podacima kako bismo se usredotočili na taj aspekt ovog šireg područja.

---
## Zašto učiti strojno učenje?

Strojno učenje, iz perspektive sustava, definira se kao stvaranje automatiziranih sustava koji mogu učiti skrivene obrasce iz podataka kako bi pomogli u donošenju inteligentnih odluka.

Ova motivacija labavo je inspirirana načinom na koji ljudski mozak uči određene stvari na temelju podataka koje percipira iz vanjskog svijeta.

✅ Razmislite na trenutak zašto bi neka tvrtka željela koristiti strategije strojnog učenja umjesto stvaranja strogo kodiranog sustava temeljenog na pravilima.

---
## Primjene strojnog učenja

Primjene strojnog učenja sada su gotovo svugdje i jednako su sveprisutne kao i podaci koji kruže našim društvima, generirani našim pametnim telefonima, povezanim uređajima i drugim sustavima. S obzirom na ogroman potencijal najsuvremenijih algoritama strojnog učenja, istraživači istražuju njihove mogućnosti za rješavanje višedimenzionalnih i multidisciplinarnih problema iz stvarnog života s iznimno pozitivnim rezultatima.

---
## Primjeri primijenjenog ML-a

**Strojno učenje možete koristiti na mnogo načina**:

- Za predviđanje vjerojatnosti bolesti na temelju medicinske povijesti ili izvještaja pacijenta.
- Za korištenje meteoroloških podataka za predviđanje vremenskih događaja.
- Za razumijevanje sentimenta teksta.
- Za otkrivanje lažnih vijesti kako bi se zaustavilo širenje propagande.

Financije, ekonomija, znanost o Zemlji, istraživanje svemira, biomedicinsko inženjerstvo, kognitivna znanost, pa čak i područja humanističkih znanosti prilagodila su strojno učenje za rješavanje teških problema obrade podataka u svojim domenama.

---
## Zaključak

Strojno učenje automatizira proces otkrivanja obrazaca pronalazeći značajne uvide iz stvarnih ili generiranih podataka. Pokazalo se iznimno vrijednim u poslovnim, zdravstvenim i financijskim primjenama, među ostalima.

U bliskoj budućnosti, razumijevanje osnova strojnog učenja postat će nužnost za ljude iz bilo kojeg područja zbog njegove široke primjene.

---
# 🚀 Izazov

Nacrtajte, na papiru ili koristeći online aplikaciju poput [Excalidraw](https://excalidraw.com/), svoje razumijevanje razlika između AI, ML, dubokog učenja i znanosti o podacima. Dodajte neke ideje o problemima koje su ove tehnike dobre u rješavanju.

# [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

---
# Pregled i samostalno učenje

Kako biste saznali više o tome kako raditi s ML algoritmima u oblaku, slijedite ovu [Putanju učenja](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Prođite [Putanju učenja](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o osnovama ML-a.

---
# Zadatak

[Pokrenite se](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za nesporazume ili pogrešne interpretacije koje mogu proizaći iz korištenja ovog prijevoda.