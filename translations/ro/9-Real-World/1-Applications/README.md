<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T15:51:54+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "ro"
}
-->
# Postscript: Învățarea automată în lumea reală

![Rezumat al învățării automate în lumea reală într-o schiță](../../../../sketchnotes/ml-realworld.png)
> Schiță realizată de [Tomomi Imura](https://www.twitter.com/girlie_mac)

În acest curriculum, ai învățat multe moduri de a pregăti datele pentru antrenare și de a crea modele de învățare automată. Ai construit o serie de modele clasice de regresie, clustering, clasificare, procesare a limbajului natural și modele de serii temporale. Felicitări! Acum, probabil te întrebi care este scopul tuturor acestor lucruri... care sunt aplicațiile reale ale acestor modele?

Deși interesul din industrie s-a concentrat mult pe AI, care de obicei utilizează învățarea profundă, există încă aplicații valoroase pentru modelele clasice de învățare automată. Este posibil să folosești unele dintre aceste aplicații chiar astăzi! În această lecție, vei explora cum opt industrii și domenii de specialitate diferite utilizează aceste tipuri de modele pentru a face aplicațiile lor mai performante, fiabile, inteligente și valoroase pentru utilizatori.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanțe

Sectorul financiar oferă multe oportunități pentru învățarea automată. Multe probleme din acest domeniu pot fi modelate și rezolvate folosind ML.

### Detectarea fraudei cu carduri de credit

Am învățat despre [clustering k-means](../../5-Clustering/2-K-Means/README.md) mai devreme în curs, dar cum poate fi utilizat pentru a rezolva probleme legate de frauda cu carduri de credit?

Clustering-ul k-means este util într-o tehnică de detectare a fraudei cu carduri de credit numită **detectarea anomaliilor**. Anomaliile, sau deviațiile în observațiile despre un set de date, ne pot spune dacă un card de credit este utilizat în mod normal sau dacă se întâmplă ceva neobișnuit. Așa cum este prezentat în articolul legat mai jos, poți sorta datele despre carduri de credit folosind un algoritm de clustering k-means și poți atribui fiecare tranzacție unui cluster pe baza gradului de anomalie. Apoi, poți evalua cele mai riscante clustere pentru a determina tranzacțiile frauduloase versus cele legitime.
[Referință](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Managementul averii

În managementul averii, o persoană sau o firmă gestionează investițiile în numele clienților lor. Scopul lor este de a menține și de a crește averea pe termen lung, așa că este esențial să aleagă investiții care performează bine.

Un mod de a evalua performanța unei investiții este prin regresie statistică. [Regresia liniară](../../2-Regression/1-Tools/README.md) este un instrument valoros pentru a înțelege cum performează un fond în raport cu un anumit etalon. De asemenea, putem deduce dacă rezultatele regresiei sunt semnificative din punct de vedere statistic sau cât de mult ar afecta investițiile unui client. Poți extinde analiza folosind regresia multiplă, unde pot fi luate în considerare factori de risc suplimentari. Pentru un exemplu despre cum ar funcționa acest lucru pentru un fond specific, consultă articolul de mai jos despre evaluarea performanței fondurilor folosind regresia.
[Referință](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educație

Sectorul educațional este, de asemenea, un domeniu foarte interesant unde ML poate fi aplicat. Există probleme interesante de abordat, cum ar fi detectarea trișatului la teste sau eseuri sau gestionarea prejudecăților, intenționate sau nu, în procesul de corectare.

### Prezicerea comportamentului studenților

[Coursera](https://coursera.com), un furnizor de cursuri online deschise, are un blog tehnologic excelent unde discută multe decizii de inginerie. În acest studiu de caz, au trasat o linie de regresie pentru a încerca să exploreze orice corelație între un rating NPS (Net Promoter Score) scăzut și retenția sau abandonul cursurilor.
[Referință](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Reducerea prejudecăților

[Grammarly](https://grammarly.com), un asistent de scriere care verifică erorile de ortografie și gramatică, utilizează sisteme sofisticate de [procesare a limbajului natural](../../6-NLP/README.md) în produsele sale. Au publicat un studiu de caz interesant pe blogul lor tehnologic despre cum au abordat problema prejudecăților de gen în învățarea automată, pe care ai învățat-o în [lecția introductivă despre echitate](../../1-Introduction/3-fairness/README.md).
[Referință](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retail

Sectorul retail poate beneficia cu siguranță de utilizarea ML, de la crearea unei experiențe mai bune pentru clienți până la gestionarea optimă a stocurilor.

### Personalizarea experienței clienților

La Wayfair, o companie care vinde produse pentru casă, cum ar fi mobilier, ajutarea clienților să găsească produsele potrivite pentru gusturile și nevoile lor este esențială. În acest articol, inginerii companiei descriu cum folosesc ML și NLP pentru a "oferi rezultate relevante pentru clienți". În mod special, motorul lor de intenție a interogării a fost construit pentru a utiliza extragerea entităților, antrenarea clasificatorilor, extragerea opiniilor și etichetarea sentimentelor din recenziile clienților. Acesta este un caz clasic de utilizare a NLP în retailul online.
[Referință](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestionarea stocurilor

Companii inovatoare și agile precum [StitchFix](https://stitchfix.com), un serviciu de livrare de haine, se bazează foarte mult pe ML pentru recomandări și gestionarea stocurilor. Echipele lor de stilizare colaborează cu echipele de merchandising: "unul dintre oamenii noștri de știința datelor a experimentat cu un algoritm genetic și l-a aplicat la îmbrăcăminte pentru a prezice ce ar fi un articol de succes care nu există încă. Am prezentat acest lucru echipei de merchandising, iar acum pot folosi acest lucru ca instrument."
[Referință](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Sănătate

Sectorul sănătății poate utiliza ML pentru a optimiza sarcinile de cercetare și problemele logistice, cum ar fi readmiterea pacienților sau oprirea răspândirii bolilor.

### Gestionarea studiilor clinice

Toxicitatea în studiile clinice este o preocupare majoră pentru producătorii de medicamente. Cât de multă toxicitate este tolerabilă? În acest studiu, analizarea diferitelor metode de studii clinice a dus la dezvoltarea unei noi abordări pentru prezicerea șanselor de rezultate ale studiilor clinice. În mod specific, au reușit să utilizeze random forest pentru a produce un [clasificator](../../4-Classification/README.md) capabil să distingă între grupuri de medicamente.
[Referință](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestionarea readmiterii în spitale

Îngrijirea spitalicească este costisitoare, mai ales când pacienții trebuie să fie readmiși. Acest articol discută despre o companie care folosește ML pentru a prezice potențialul de readmitere folosind algoritmi de [clustering](../../5-Clustering/README.md). Aceste clustere ajută analiștii să "descopere grupuri de readmiteri care pot avea o cauză comună".
[Referință](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestionarea bolilor

Pandemia recentă a evidențiat modul în care învățarea automată poate ajuta la oprirea răspândirii bolilor. În acest articol, vei recunoaște utilizarea ARIMA, curbelor logistice, regresiei liniare și SARIMA. "Această lucrare este o încercare de a calcula rata de răspândire a acestui virus și, astfel, de a prezice decesele, recuperările și cazurile confirmate, astfel încât să ne ajute să ne pregătim mai bine și să supraviețuim."
[Referință](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecologie și Tehnologie Verde

Natura și ecologia constau în multe sisteme sensibile unde interacțiunea dintre animale și natură devine importantă. Este esențial să putem măsura aceste sisteme cu acuratețe și să acționăm corespunzător dacă se întâmplă ceva, cum ar fi un incendiu de pădure sau o scădere a populației de animale.

### Gestionarea pădurilor

Ai învățat despre [Învățarea prin Recompensă](../../8-Reinforcement/README.md) în lecțiile anterioare. Poate fi foarte utilă atunci când încercăm să prezicem modele în natură. În special, poate fi utilizată pentru a urmări probleme ecologice precum incendiile de pădure și răspândirea speciilor invazive. În Canada, un grup de cercetători a folosit Învățarea prin Recompensă pentru a construi modele de dinamică a incendiilor de pădure din imagini satelitare. Folosind un proces inovator de "răspândire spațială (SSP)", au imaginat un incendiu de pădure ca "agentul de la orice celulă din peisaj." "Setul de acțiuni pe care incendiul le poate lua dintr-o locație la orice moment include răspândirea spre nord, sud, est sau vest sau să nu se răspândească."

Această abordare inversează configurația obișnuită a RL, deoarece dinamica Procesului de Decizie Markovian (MDP) corespunzător este o funcție cunoscută pentru răspândirea imediată a incendiului. Citește mai multe despre algoritmii clasici utilizați de acest grup la linkul de mai jos.
[Referință](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Senzori de mișcare pentru animale

Deși învățarea profundă a creat o revoluție în urmărirea vizuală a mișcărilor animalelor (poți construi propriul [tracker pentru urși polari](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) aici), ML clasic are încă un loc în această sarcină.

Senzorii pentru urmărirea mișcărilor animalelor de fermă și IoT utilizează acest tip de procesare vizuală, dar tehnicile ML mai de bază sunt utile pentru preprocesarea datelor. De exemplu, în acest articol, posturile oilor au fost monitorizate și analizate folosind diferiți algoritmi de clasificare. Poți recunoaște curba ROC la pagina 335.
[Referință](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestionarea energiei

În lecțiile noastre despre [previziunea seriilor temporale](../../7-TimeSeries/README.md), am invocat conceptul de parcometre inteligente pentru a genera venituri pentru un oraș pe baza înțelegerii cererii și ofertei. Acest articol discută în detaliu cum clustering-ul, regresia și previziunea seriilor temporale au fost combinate pentru a ajuta la prezicerea utilizării viitoare a energiei în Irlanda, pe baza contorizării inteligente.
[Referință](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Asigurări

Sectorul asigurărilor este un alt domeniu care utilizează ML pentru a construi și optimiza modele financiare și actuariale viabile.

### Gestionarea volatilității

MetLife, un furnizor de asigurări de viață, este transparent în modul în care analizează și atenuează volatilitatea în modelele lor financiare. În acest articol vei observa vizualizări de clasificare binară și ordinale. De asemenea, vei descoperi vizualizări de previziune.
[Referință](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Arte, Cultură și Literatură

În arte, de exemplu în jurnalism, există multe probleme interesante. Detectarea știrilor false este o problemă majoră, deoarece s-a demonstrat că influențează opinia oamenilor și chiar poate destabiliza democrațiile. Muzeele pot beneficia, de asemenea, de utilizarea ML în tot, de la găsirea legăturilor între artefacte până la planificarea resurselor.

### Detectarea știrilor false

Detectarea știrilor false a devenit un joc de-a șoarecele și pisica în media de astăzi. În acest articol, cercetătorii sugerează că un sistem care combină mai multe dintre tehnicile ML pe care le-am studiat poate fi testat, iar cel mai bun model implementat: "Acest sistem se bazează pe procesarea limbajului natural pentru a extrage caracteristici din date, iar apoi aceste caracteristici sunt utilizate pentru antrenarea clasificatorilor de învățare automată, cum ar fi Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) și Logistic Regression (LR)."
[Referință](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Acest articol arată cum combinarea diferitelor domenii ML poate produce rezultate interesante care pot ajuta la oprirea răspândirii știrilor false și la prevenirea daunelor reale; în acest caz, impulsul a fost răspândirea zvonurilor despre tratamentele COVID care au incitat violență în masă.

### ML în muzee

Muzeele sunt pe punctul de a intra într-o revoluție AI, în care catalogarea și digitizarea colecțiilor și găsirea legăturilor între artefacte devin mai ușoare pe măsură ce tehnologia avansează. Proiecte precum [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) ajută la descifrarea misterelor colecțiilor inaccesibile, cum ar fi Arhivele Vaticanului. Dar aspectul de afaceri al muzeelor beneficiază, de asemenea, de modelele ML.

De exemplu, Institutul de Artă din Chicago a construit modele pentru a prezice ce interesează publicul și când vor participa la expoziții. Scopul este de a crea experiențe individualizate și optimizate pentru vizitatori de fiecare dată când aceștia vizitează muzeul. "În anul fiscal 2017, modelul a prezis participarea și veniturile din bilete cu o precizie de 1%, spune Andrew Simnick, vicepreședinte senior la Institutul de Artă."
[Referință](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentarea clienților

Cele mai eficiente strategii de marketing vizează clienții în moduri diferite pe baza diverselor grupări. În acest articol, utilizările algoritmilor de clustering sunt discutate pentru a sprijini marketingul diferențiat. Marketingul diferențiat ajută companiile să îmbunătățească recunoașterea brandului, să ajungă la mai mulți clienți și să câștige mai mulți bani.
[Referință](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Provocare

Identifică un alt sector care beneficiază de unele dintre tehnicile pe care le-ai învățat în acest curriculum și descoperă cum utilizează ML.
## [Chestionar post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și Studiu Individual

Echipa de știință a datelor de la Wayfair are câteva videoclipuri interesante despre cum folosesc ML în compania lor. Merită [să arunci o privire](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Temă

[O vânătoare de comori ML](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.