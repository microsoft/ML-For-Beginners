# Construirea soluțiilor de învățare automată cu AI responsabil
 
![Sumar al AI responsabil în Învățarea Automată într-o schiță](../../../../translated_images/ro/ml-fairness.ef296ebec6afc98a.webp)
> Schiță de [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Chestionar pre-lectură](https://ff-quizzes.netlify.app/en/ml/)
 
## Introducere

În acest curriculum, veți începe să descoperiți cum învățarea automată poate și impactează viața noastră de zi cu zi. Chiar și acum, sistemele și modelele sunt implicate în sarcini zilnice de luare a deciziilor, cum ar fi diagnosticul medical, aprobarea împrumuturilor sau detectarea fraudelor. Deci, este important ca aceste modele să funcționeze bine pentru a oferi rezultate de încredere. La fel ca orice aplicație software, sistemele AI pot să nu îndeplinească așteptările sau să aibă un rezultat nedorit. De aceea este esențial să putem înțelege și explica comportamentul unui model AI.

Imaginați-vă ce se poate întâmpla când datele pe care le folosiți pentru a construi aceste modele lipsesc anumite demografii, cum ar fi rasa, genul, opiniile politice, religia sau reprezintă disproporționat asemenea demografii. Ce se întâmplă când rezultatul modelului este interpretat să favorizeze un anumit grup demografic? Care este consecința pentru aplicație? În plus, ce se întâmplă când modelul are un rezultat advers și este dăunător oamenilor? Cine este responsabil pentru comportamentul sistemelor AI? Acestea sunt câteva întrebări pe care le vom explora în acest curriculum.

În această lecție, veți:

- Să vă creșteți conștientizarea importanței echității în învățarea automată și a daunelor legate de echitate.
- Să deveniți familiarizați cu practica de explorare a excepțiilor și scenariilor neobișnuite pentru a asigura fiabilitatea și siguranța
- Să înțelegeți necesitatea de a împuternici pe toată lumea prin proiectarea sistemelor incluzive
- Să explorați cât de vitală este protejarea confidențialității și securității datelor și a persoanelor
- Să vedeți importanța unei abordări transparente pentru a explica comportamentul modelelor AI
- Să fiți atenți la modul în care responsabilitatea este esențială pentru construirea încrederii în sistemele AI

## Prerechizite

Ca prerechizit, vă rugăm să parcurgeți traseul de învățare "Principiile AI Responsabil" și să vizionați videoclipul de mai jos pe acest subiect:

Aflați mai multe despre AI responsabil urmând acest [Traseu de Învățare](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordarea Microsoft privind AI responsabil](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordarea Microsoft privind AI responsabil")

> 🎥 Faceți clic pe imaginea de mai sus pentru un videoclip: Abordarea Microsoft privind AI responsabil

## Echitate

Sistemele AI ar trebui să trateze pe toată lumea în mod echitabil și să evite să afecteze diferit grupuri similare de persoane. De exemplu, atunci când sistemele AI oferă ghidare privind tratamentul medical, cererile de împrumut sau angajarea, ele ar trebui să facă aceleași recomandări tuturor persoanelor cu simptome similare, circumstanțe financiare sau calificări profesionale similare. Fiecare dintre noi, ca oameni, purtăm prejudecăți moștenite care ne influențează deciziile și acțiunile. Aceste prejudecăți pot fi evidente în datele pe care le folosim pentru antrenarea sistemelor AI. O astfel de manipulare se poate întâmpla uneori neintenționat. De multe ori este dificil să știi în mod conștient când introduci un prejudiciu în date.

**„Inechitatea”** cuprinde impacturile negative, sau "daunele", pentru un grup de persoane, cum ar fi cele definite pe criterii de rasă, gen, vârstă sau stare de dizabilitate. Principalele daune legate de echitate pot fi clasificate astfel:

- **Alocare**, dacă un gen sau o etnie, de exemplu, este favorizată în detrimentul alteia.
- **Calitatea serviciului**. Dacă antrenezi datele pentru un scenariu specific, dar realitatea este mult mai complexă, rezultă un serviciu cu performanță slabă. De exemplu, un dozator de săpun pentru mâini care nu pare să recunoască persoanele cu pielea închisă la culoare. [Referință](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigrare**. Criticarea și etichetarea nedreaptă a ceva sau cuiva. De exemplu, o tehnologie de etichetare a imaginilor a etichetat infam imagini cu persoane cu pielea închisă la culoare ca fiind gorile.
- **Suprareprezentare sau subreprezentare**. Ideea este că un anumit grup nu este văzut într-o anumită profesie, iar orice serviciu sau funcție care continuă să promoveze această situație contribuie la daune.
- **Stereotipuri**. Asocierea unui grup dat cu atribute predefinite. De exemplu, un sistem de traducere între engleză și turcă poate avea inexactități din cauza cuvintelor cu asocieri stereotipe legate de gen.

![traducere în turcă](../../../../translated_images/ro/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> traducere în turcă

![traducere înapoi în engleză](../../../../translated_images/ro/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> traducere înapoi în engleză

Când proiectăm și testăm sisteme AI, trebuie să ne asigurăm că AI este echitabilă și nu este programată să ia decizii părtinitoare sau discriminatorii, pe care nici oamenilor nu le este permis să le ia. Garantarea echității în AI și învățarea automată rămâne o provocare sociotehnică complexă.

### Fiabilitate și siguranță

Pentru a construi încredere, sistemele AI trebuie să fie fiabile, sigure și consistente atât în condiții normale, cât și în cele neașteptate. Este important să știm cum se vor comporta sistemele AI în diverse situații, mai ales când acestea sunt excepții. Când construim soluții AI, trebuie să acordăm o atenție substanțială modului în care acestea vor gestiona o varietate largă de circumstanțe pe care le-ar putea întâlni. De exemplu, o mașină autonomă trebuie să pună siguranța oamenilor pe primul loc. Drept urmare, AI-ul care controlează mașina trebuie să ia în considerare toate scenariile posibile cu care s-ar putea confrunta, cum ar fi noaptea, furtunile sau viscolele, copiii care aleargă pe stradă, animalele de companie, lucrările de construcție etc. Cât de bine poate gestiona un sistem AI o gamă largă de condiții de mod fiabil și sigur reflectă nivelul de anticipare luat în considerare de data scientist sau dezvoltatorul AI în timpul proiectării sau testării sistemului.

> [🎥 Faceți clic aici pentru un video: Fiabilitate și siguranță în AI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Incluziune

Sistemele AI ar trebui proiectate pentru a implica și împuternici pe toată lumea. La proiectarea și implementarea sistemelor AI, data scientistii și dezvoltatorii AI identifică și abordează potențialele bariere din sistem care ar putea exclude neintenționat persoane. De exemplu, există 1 miliard de persoane cu dizabilități în întreaga lume. Odată cu progresul AI, acestea pot avea acces mai ușor la o gamă largă de informații și oportunități în viața de zi cu zi. Prin abordarea barierelor, se creează oportunități de inovare și dezvoltare a produselor AI cu experiențe mai bune care beneficiază pe toată lumea.

> [🎥 Faceți clic aici pentru un video: Incluziune în AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Securitate și confidențialitate

Sistemele AI trebuie să fie sigure și să respecte confidențialitatea persoanelor. Oamenii au mai puțină încredere în sistemele care pun în pericol intimitatea, informațiile sau viața lor. Când antrenăm modele de învățare automată, ne bazăm pe date pentru a produce cele mai bune rezultate. În acest proces, trebuie să luăm în considerare originea datelor și integritatea lor. De exemplu, datele au fost oferite de utilizatori sau sunt disponibile public? Pe lângă acestea, în lucrul cu datele, este crucial să dezvoltăm sisteme AI care să poată proteja informațiile confidențiale și să reziste atacurilor. Pe măsură ce AI devine mai răspândită, protejarea confidențialității și securizarea informațiilor personale și de afaceri devine din ce în ce mai importantă și complexă. Problemele de confidențialitate și securitate a datelor necesită o atenție deosebită în AI deoarece accesul la date este esențial pentru ca sistemele AI să facă predicții și decizii corecte și informate despre oameni.

> [🎥 Faceți clic aici pentru un video: Securitate în AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ca industrie am făcut progrese semnificative în domeniul confidențialității și securității, alimentate în mod semnificativ de reglementări precum GDPR (Regulamentul General privind Protecția Datelor).
- Totuși, cu sistemele AI trebuie să recunoaștem tensiunea dintre necesitatea unor date personale mai multe pentru ca sistemele să fie mai personale și eficiente – și intimitate.
- Exact ca la apariția computerelor conectate la internet, asistăm și la o creștere masivă a problemelor de securitate legate de AI.
- În același timp, am văzut cum AI este folosită pentru a îmbunătăți securitatea. De exemplu, majoritatea scanerelor antivirus moderne sunt bazate pe euristici AI.
- Trebuie să ne asigurăm că procesele noastre de Data Science se îmbină armonios cu cele mai recente practici de confidențialitate și securitate.


### Transparență
Sistemele AI ar trebui să fie înțelese. O parte crucială a transparenței este explicarea comportamentului sistemelor AI și a componentelor lor. Îmbunătățirea înțelegerii sistemelor AI necesită ca factorii interesați să înțeleagă cum și de ce funcționează acestea pentru a putea identifica posibile probleme de performanță, preocupări legate de siguranță și confidențialitate, prejudecăți, practici excluzive sau rezultate nedorite. De asemenea, credem că cei care utilizează sistemele AI ar trebui să fie sinceri și deschiși cu privire la când, de ce și cum aleg să le implementeze. Și la limitările sistemelor pe care le folosesc. De exemplu, dacă o bancă folosește un sistem AI pentru a susține deciziile de creditare ale consumatorilor, este important să se examineze rezultatele și să se înțeleagă ce date influențează recomandările sistemului. Guvernele încep să reglementeze AI în diverse industrii, așa că data scientistii și organizațiile trebuie să explice dacă un sistem AI îndeplinește cerințele reglementare, mai ales când apare un rezultat nedorit.

> [🎥 Faceți clic aici pentru un video: Transparență în AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Deoarece sistemele AI sunt atât de complexe, este greu de înțeles cum funcționează și de interpretat rezultatele.
- Această lipsă de înțelegere afectează modul în care aceste sisteme sunt gestionate, operaționalizate și documentate.
- Această lipsă de înțelegere afectează mai ales deciziile luate folosind rezultatele produse de aceste sisteme.

### Responsabilitate
 
Persoanele care proiectează și implementează sistemele AI trebuie să fie responsabile pentru modul în care funcționează acestea. Nevoia de responsabilitate este deosebit de crucială în cazul tehnologiilor sensibile, cum ar fi recunoașterea facială. Recent, a crescut cererea pentru tehnologia de recunoaștere facială, în special din partea organizațiilor de aplicare a legii care văd potențialul acestei tehnologii în utilizări precum găsirea copiilor dispăruți. Totuși, aceste tehnologii ar putea fi folosite de un guvern pentru a pune în pericol libertățile fundamentale ale cetățenilor săi, de exemplu, prin supraveghere continuă a unor persoane specifice. Prin urmare, data scientistii și organizațiile trebuie să fie responsabile pentru modul în care sistemul lor AI impactează indivizii sau societatea.

[![Cercetător principal în AI avertizează asupra supravegherii în masă prin recunoaștere facială](../../../../translated_images/ro/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordarea Microsoft privind AI responsabil")

> 🎥 Faceți clic pe imaginea de mai sus pentru un video: Avertismente privind Supravegherea în Masă prin Recunoaștere Facială

În cele din urmă, una dintre cele mai mari întrebări pentru generația noastră, ca prima generație care aduce AI în societate, este cum să ne asigurăm că calculatoarele vor rămâne responsabile față de oameni și cum să ne asigurăm că persoanele care proiectează calculatoarele rămân responsabile față de toți ceilalți.

## Evaluarea impactului

Înainte de a antrena un model de învățare automată, este important să se efectueze o evaluare a impactului pentru a înțelege scopul sistemului AI; la ce este destinat să fie folosit; unde va fi implementat; și cine va interacționa cu sistemul. Acestea sunt utile pentru recenzorii sau testatorii care evaluează sistemul să știe ce factori trebuie luați în considerare pentru a identifica posibile riscuri și consecințe așteptate.

Următoarele sunt domenii de concentrare în timpul evaluării impactului:

* **Impact advers asupra persoanelor**. Conștientizarea oricărei restricții sau cerințe, utilizare neacceptată sau orice limitări cunoscute care împiedică performanța sistemului este vitală pentru a asigura că sistemul nu este folosit într-un mod care ar putea cauza rău persoanelor.
* **Cereri de date**. Înțelegerea modului și locului în care sistemul va folosi datele permite recenzorilor să exploreze orice cerințe referitoare la date cărora trebuie să le acordați atenție (de exemplu, reglementări GDPR sau HIPAA). De asemenea, examinați dacă sursa sau cantitatea de date este substanțială pentru antrenare.
* **Sumar al impactului**. Adunați o listă cu potențiale daune care ar putea apărea din utilizarea sistemului. Pe parcursul ciclului de viață al ML, revizuiți dacă problemele identificate sunt atenuate sau abordate.
* **Obiective aplicabile** pentru fiecare dintre cele șase principii fundamentale. Evaluați dacă obiectivele din fiecare principiu sunt îndeplinite și dacă există vreun decalaj.


## Debugging cu AI responsabil

Similar cu depanarea unei aplicații software, depanarea unui sistem AI este un proces necesar de identificare și rezolvare a problemelor din sistem. Există mulți factori care pot afecta performanța unui model, astfel încât să nu funcționeze conform așteptărilor sau responsabil. Majoritatea metricilor tradiționale de performanță a modelului sunt agregate cantitative ale performanței unui model, care nu sunt suficiente pentru a analiza modul în care un model încalcă principiile AI responsabile. Mai mult, un model de învățare automată este o cutie neagră care face dificilă înțelegerea a ce determină rezultatul său sau oferirea unui răspuns când vine o eroare. Ulterior în acest curs, vom învăța cum să folosim panoul AI responsabil pentru a ajuta la depanarea sistemelor AI. Panoul oferă un instrument holistic pentru data scientisti și dezvoltatorii AI pentru a efectua:

* **Analiză a erorilor**. Pentru a identifica distribuția erorilor modelului care pot afecta echitatea sau fiabilitatea sistemului.
* **Privire de ansamblu a modelului**. Pentru a descoperi unde există disparități în performanța modelului pe diferite cohorte de date.
* **Analiză a datelor**. Pentru a înțelege distribuția datelor și a identifica orice posibilă părtinire în date care ar putea duce la probleme de echitate, incluziune și fiabilitate.
* **Interpretabilitatea modelului**. Pentru a înțelege ce influențează sau afectează predicțiile modelului. Acest lucru ajută la explicarea comportamentului modelului, ceea ce este important pentru transparență și responsabilitate.


## 🚀 Provocare
 
Pentru a preveni introducerea daunelor în primul rând, ar trebui să:

- avem o diversitate de fundaluri și perspective printre persoanele care lucrează la sisteme
- investim în seturi de date care reflectă diversitatea societății noastre
- dezvoltăm metode mai bune pe parcursul ciclului de viață al învățării automate pentru detectarea și corectarea AI-ului iresponsabil atunci când apare

Gândiți-vă la scenarii din viața reală în care lipsa de încredere a unui model este evidentă în construcția și utilizarea modelului. Ce mai ar trebui să luăm în considerare?

## [Chestionar post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Auto-studiu
 
În această lecție, ați învățat câteva elemente de bază despre conceptele de echitate și inechitate în învățarea automată.
 
Vizionați acest atelier pentru a aprofunda subiectele:

- În căutarea unui AI responsabil: aplicarea principiilor în practică de Besmira Nushi, Mehrnoosh Sameki și Amit Sharma

[![Cutia cu instrumente pentru AI responsabil: Un cadru open-source pentru construirea AI responsabil](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "Cutia cu instrumente RAI: Un cadru open-source pentru construirea AI responsabil")

> 🎥 Faceți clic pe imaginea de mai sus pentru un videoclip: Cutia cu instrumente RAI: Un cadru open-source pentru construirea AI responsabil de Besmira Nushi, Mehrnoosh Sameki și Amit Sharma

De asemenea, citiți: 

- Centrul de resurse RAI al Microsoft: [Resurse AI Responsabil – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupul de cercetare FATE al Microsoft: [FATE: Echitate, Responsabilitate, Transparență și Etică în AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

Cutia cu instrumente RAI: 

- [Depozitul GitHub pentru Cutia cu instrumente AI responsabil](https://github.com/microsoft/responsible-ai-toolbox)

Citiți despre instrumentele Azure Machine Learning pentru asigurarea echității:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Temă

[Explorați Cutia cu instrumente RAI](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Declinare a responsabilității**:
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). În timp ce ne străduim pentru acuratețe, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa nativă trebuie considerat sursa autorizată. Pentru informații critice, se recomandă traducerea profesională realizată de un om. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care decurg din utilizarea acestei traduceri.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->