<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T16:01:28+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "ro"
}
-->
# Construirea soluțiilor de învățare automată cu AI responsabil

![Rezumat al AI responsabil în învățarea automată într-o schiță](../../../../sketchnotes/ml-fairness.png)
> Schiță realizată de [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

## Introducere

În acest curriculum, vei începe să descoperi cum învățarea automată poate și influențează viețile noastre de zi cu zi. Chiar și acum, sistemele și modelele sunt implicate în sarcini de luare a deciziilor zilnice, cum ar fi diagnosticarea în sănătate, aprobarea împrumuturilor sau detectarea fraudei. De aceea, este important ca aceste modele să funcționeze bine pentru a oferi rezultate de încredere. La fel ca orice aplicație software, sistemele AI pot să nu îndeplinească așteptările sau să aibă un rezultat nedorit. De aceea, este esențial să putem înțelege și explica comportamentul unui model AI.

Imaginează-ți ce se poate întâmpla atunci când datele pe care le folosești pentru a construi aceste modele nu includ anumite demografii, cum ar fi rasa, genul, opiniile politice, religia sau reprezintă disproporționat astfel de demografii. Ce se întâmplă când rezultatul modelului este interpretat astfel încât să favorizeze o anumită demografie? Care este consecința pentru aplicație? În plus, ce se întâmplă când modelul are un rezultat advers și este dăunător pentru oameni? Cine este responsabil pentru comportamentul sistemelor AI? Acestea sunt câteva întrebări pe care le vom explora în acest curriculum.

În această lecție, vei:

- Conștientiza importanța echității în învățarea automată și daunele legate de echitate.
- Deveni familiar cu practica de explorare a valorilor extreme și scenariilor neobișnuite pentru a asigura fiabilitatea și siguranța.
- Înțelege necesitatea de a împuternici pe toată lumea prin proiectarea de sisteme incluzive.
- Explora cât de vital este să protejezi confidențialitatea și securitatea datelor și a oamenilor.
- Observa importanța unei abordări transparente pentru a explica comportamentul modelelor AI.
- Fi conștient de faptul că responsabilitatea este esențială pentru a construi încrederea în sistemele AI.

## Prerechizite

Ca prerechizite, te rugăm să urmezi "Principiile AI Responsabil" pe Learn Path și să vizionezi videoclipul de mai jos pe acest subiect:

Află mai multe despre AI Responsabil urmând acest [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordarea Microsoft pentru AI Responsabil](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordarea Microsoft pentru AI Responsabil")

> 🎥 Click pe imaginea de mai sus pentru un videoclip: Abordarea Microsoft pentru AI Responsabil

## Echitate

Sistemele AI ar trebui să trateze pe toată lumea în mod echitabil și să evite să afecteze grupuri similare de oameni în moduri diferite. De exemplu, atunci când sistemele AI oferă recomandări pentru tratamente medicale, aplicații de împrumut sau angajare, acestea ar trebui să facă aceleași recomandări pentru toți cei cu simptome, circumstanțe financiare sau calificări profesionale similare. Fiecare dintre noi, ca oameni, purtăm prejudecăți moștenite care ne afectează deciziile și acțiunile. Aceste prejudecăți pot fi evidente în datele pe care le folosim pentru a antrena sistemele AI. Astfel de manipulări pot apărea uneori neintenționat. Este adesea dificil să conștientizezi când introduci prejudecăți în date.

**„Nedreptatea”** cuprinde impacturi negative sau „daune” pentru un grup de oameni, cum ar fi cei definiți în termeni de rasă, gen, vârstă sau statut de dizabilitate. Principalele daune legate de echitate pot fi clasificate astfel:

- **Alocare**, dacă un gen sau o etnie, de exemplu, este favorizată în detrimentul alteia.
- **Calitatea serviciului**. Dacă antrenezi datele pentru un scenariu specific, dar realitatea este mult mai complexă, aceasta duce la un serviciu slab. De exemplu, un dozator de săpun care nu pare să detecteze persoanele cu pielea închisă la culoare. [Referință](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigrare**. Criticarea și etichetarea nedreaptă a ceva sau a cuiva. De exemplu, o tehnologie de etichetare a imaginilor a etichetat în mod infam imagini ale persoanelor cu pielea închisă la culoare ca fiind gorile.
- **Reprezentare excesivă sau insuficientă**. Ideea este că un anumit grup nu este văzut într-o anumită profesie, iar orice serviciu sau funcție care continuă să promoveze acest lucru contribuie la daune.
- **Stereotipizare**. Asocierea unui grup dat cu atribute predefinite. De exemplu, un sistem de traducere între engleză și turcă poate avea inexactități din cauza cuvintelor asociate stereotipic cu genul.

![traducere în turcă](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> traducere în turcă

![traducere înapoi în engleză](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> traducere înapoi în engleză

Când proiectăm și testăm sisteme AI, trebuie să ne asigurăm că AI este echitabil și nu este programat să ia decizii părtinitoare sau discriminatorii, pe care nici oamenii nu au voie să le ia. Garantarea echității în AI și învățarea automată rămâne o provocare complexă sociotehnică.

### Fiabilitate și siguranță

Pentru a construi încredere, sistemele AI trebuie să fie fiabile, sigure și consistente în condiții normale și neașteptate. Este important să știm cum se vor comporta sistemele AI într-o varietate de situații, mai ales când sunt valori extreme. Când construim soluții AI, trebuie să ne concentrăm substanțial pe modul de gestionare a unei varietăți largi de circumstanțe pe care soluțiile AI le-ar putea întâlni. De exemplu, o mașină autonomă trebuie să pună siguranța oamenilor pe primul loc. Ca rezultat, AI-ul care alimentează mașina trebuie să ia în considerare toate scenariile posibile pe care mașina le-ar putea întâlni, cum ar fi noaptea, furtuni, viscol, copii care traversează strada, animale de companie, construcții pe drum etc. Cât de bine poate gestiona un sistem AI o gamă largă de condiții în mod fiabil și sigur reflectă nivelul de anticipare pe care l-a avut specialistul în date sau dezvoltatorul AI în timpul proiectării sau testării sistemului.

> [🎥 Click aici pentru un videoclip: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Incluziune

Sistemele AI ar trebui să fie proiectate pentru a implica și împuternici pe toată lumea. Când proiectează și implementează sisteme AI, specialiștii în date și dezvoltatorii AI identifică și abordează barierele potențiale din sistem care ar putea exclude neintenționat oamenii. De exemplu, există 1 miliard de persoane cu dizabilități în întreaga lume. Cu avansarea AI, acestea pot accesa o gamă largă de informații și oportunități mai ușor în viața lor de zi cu zi. Abordarea barierelor creează oportunități de a inova și dezvolta produse AI cu experiențe mai bune care beneficiază pe toată lumea.

> [🎥 Click aici pentru un videoclip: incluziunea în AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Securitate și confidențialitate

Sistemele AI ar trebui să fie sigure și să respecte confidențialitatea oamenilor. Oamenii au mai puțină încredere în sistemele care le pun în pericol confidențialitatea, informațiile sau viețile. Când antrenăm modele de învățare automată, ne bazăm pe date pentru a produce cele mai bune rezultate. În acest proces, originea datelor și integritatea lor trebuie luate în considerare. De exemplu, datele au fost trimise de utilizatori sau sunt disponibile public? Apoi, în timp ce lucrăm cu datele, este crucial să dezvoltăm sisteme AI care pot proteja informațiile confidențiale și rezista atacurilor. Pe măsură ce AI devine mai răspândit, protejarea confidențialității și securizarea informațiilor personale și de afaceri devin din ce în ce mai critice și complexe. Problemele de confidențialitate și securitate a datelor necesită o atenție deosebită pentru AI, deoarece accesul la date este esențial pentru ca sistemele AI să facă predicții și decizii precise și informate despre oameni.

> [🎥 Click aici pentru un videoclip: securitatea în AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Industria a făcut progrese semnificative în confidențialitate și securitate, alimentate în mod semnificativ de reglementări precum GDPR (Regulamentul General privind Protecția Datelor).
- Totuși, cu sistemele AI trebuie să recunoaștem tensiunea dintre necesitatea mai multor date personale pentru a face sistemele mai personale și eficiente – și confidențialitatea.
- La fel ca la nașterea computerelor conectate la internet, vedem o creștere semnificativă a numărului de probleme de securitate legate de AI.
- În același timp, am văzut AI fiind folosit pentru a îmbunătăți securitatea. De exemplu, majoritatea scanerelor antivirus moderne sunt alimentate de euristici AI.
- Trebuie să ne asigurăm că procesele noastre de știința datelor se îmbină armonios cu cele mai recente practici de confidențialitate și securitate.

### Transparență

Sistemele AI ar trebui să fie ușor de înțeles. O parte crucială a transparenței este explicarea comportamentului sistemelor AI și a componentelor acestora. Îmbunătățirea înțelegerii sistemelor AI necesită ca părțile interesate să înțeleagă cum și de ce funcționează acestea, astfel încât să poată identifica potențiale probleme de performanță, preocupări legate de siguranță și confidențialitate, prejudecăți, practici de excludere sau rezultate neintenționate. De asemenea, credem că cei care folosesc sistemele AI ar trebui să fie sinceri și deschiși cu privire la momentul, motivul și modul în care aleg să le implementeze. La fel ca și limitările sistemelor pe care le folosesc. De exemplu, dacă o bancă folosește un sistem AI pentru a sprijini deciziile de creditare pentru consumatori, este important să examineze rezultatele și să înțeleagă ce date influențează recomandările sistemului. Guvernele încep să reglementeze AI în diverse industrii, astfel încât specialiștii în date și organizațiile trebuie să explice dacă un sistem AI îndeplinește cerințele de reglementare, mai ales atunci când există un rezultat nedorit.

> [🎥 Click aici pentru un videoclip: transparența în AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Deoarece sistemele AI sunt atât de complexe, este dificil să înțelegem cum funcționează și să interpretăm rezultatele.
- Această lipsă de înțelegere afectează modul în care aceste sisteme sunt gestionate, operaționalizate și documentate.
- Mai important, această lipsă de înțelegere afectează deciziile luate pe baza rezultatelor produse de aceste sisteme.

### Responsabilitate

Persoanele care proiectează și implementează sisteme AI trebuie să fie responsabile pentru modul în care funcționează sistemele lor. Necesitatea responsabilității este deosebit de crucială în cazul tehnologiilor sensibile, cum ar fi recunoașterea facială. Recent, a existat o cerere tot mai mare pentru tehnologia de recunoaștere facială, mai ales din partea organizațiilor de aplicare a legii care văd potențialul tehnologiei în utilizări precum găsirea copiilor dispăruți. Cu toate acestea, aceste tehnologii ar putea fi utilizate de un guvern pentru a pune în pericol libertățile fundamentale ale cetățenilor săi, de exemplu, prin supravegherea continuă a unor indivizi specifici. Prin urmare, specialiștii în date și organizațiile trebuie să fie responsabili pentru modul în care sistemul lor AI afectează indivizii sau societatea.

[![Cercetător de top în AI avertizează asupra supravegherii masive prin recunoaștere facială](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordarea Microsoft pentru AI Responsabil")

> 🎥 Click pe imaginea de mai sus pentru un videoclip: Avertismente despre supravegherea masivă prin recunoaștere facială

În cele din urmă, una dintre cele mai mari întrebări pentru generația noastră, ca prima generație care aduce AI în societate, este cum să ne asigurăm că computerele vor rămâne responsabile față de oameni și cum să ne asigurăm că persoanele care proiectează computere rămân responsabile față de toți ceilalți.

## Evaluarea impactului

Înainte de a antrena un model de învățare automată, este important să efectuezi o evaluare a impactului pentru a înțelege scopul sistemului AI; utilizarea intenționată; unde va fi implementat; și cine va interacționa cu sistemul. Acestea sunt utile pentru evaluatorii sau testerii care analizează sistemul pentru a ști ce factori să ia în considerare atunci când identifică riscuri potențiale și consecințe așteptate.

Următoarele sunt domenii de interes atunci când se efectuează o evaluare a impactului:

* **Impact negativ asupra indivizilor**. Conștientizarea oricăror restricții sau cerințe, utilizări neacceptate sau limitări cunoscute care împiedică performanța sistemului este vitală pentru a ne asigura că sistemul nu este utilizat într-un mod care ar putea dăuna indivizilor.
* **Cerințe de date**. Înțelegerea modului și locului în care sistemul va utiliza datele permite evaluatorilor să exploreze orice cerințe de date de care trebuie să fii conștient (de exemplu, reglementările GDPR sau HIPPA). În plus, examinează dacă sursa sau cantitatea de date este suficientă pentru antrenare.
* **Rezumatul impactului**. Adună o listă de potențiale daune care ar putea apărea din utilizarea sistemului. Pe parcursul ciclului de viață al ML, verifică dacă problemele identificate sunt atenuate sau abordate.
* **Obiective aplicabile** pentru fiecare dintre cele șase principii de bază. Evaluează dacă obiectivele fiecărui principiu sunt îndeplinite și dacă există lacune.

## Debugging cu AI responsabil

Similar cu depanarea unei aplicații software, depanarea unui sistem AI este un proces necesar de identificare și rezolvare a problemelor din sistem. Există mulți factori care ar putea afecta un model să nu funcționeze conform așteptărilor sau responsabil. Majoritatea metricilor tradiționale de performanță ale modelului sunt agregate cantitative ale performanței modelului, care nu sunt suficiente pentru a analiza modul în care un model încalcă principiile AI responsabil. Mai mult, un model de învățare automată este o cutie neagră, ceea ce face dificilă înțelegerea a ceea ce determină rezultatul său sau oferirea unei explicații atunci când face o greșeală. Mai târziu în acest curs, vom învăța cum să folosim tabloul de bord AI Responsabil pentru a ajuta la depanarea sistemelor AI. Tabloul de bord oferă un instrument holistic pentru specialiștii în date și dezvoltatorii AI pentru a efectua:

* **Analiza erorilor**. Pentru a identifica distribuția erorilor modelului care poate afecta echitatea sau fiabilitatea sistemului.
* **Prezentarea generală a modelului**. Pentru a descoperi unde există disparități în performanța modelului în diferite cohorte de date.
* **Analiza datelor**. Pentru a înțelege distribuția datelor și a identifica orice potențială prejudecată în date care ar putea duce la probleme de echitate, incluziune și fiabilitate.
* **Interpretabilitatea modelului**. Pentru a înțelege ce afectează sau influențează predicțiile modelului. Acest lucru ajută la explicarea comportamentului modelului, ceea ce este important pentru transparență și
Urmărește acest workshop pentru a aprofunda subiectele:

- În căutarea AI responsabil: Aplicarea principiilor în practică de Besmira Nushi, Mehrnoosh Sameki și Amit Sharma

[![Responsible AI Toolbox: Un cadru open-source pentru construirea AI responsabil](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Un cadru open-source pentru construirea AI responsabil")


> 🎥 Click pe imaginea de mai sus pentru un videoclip: RAI Toolbox: Un cadru open-source pentru construirea AI responsabil de Besmira Nushi, Mehrnoosh Sameki și Amit Sharma

De asemenea, citește:

- Centrul de resurse RAI al Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupul de cercetare FATE al Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox:

- [Repository-ul GitHub Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Citește despre instrumentele Azure Machine Learning pentru asigurarea echității:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Temă

[Explorează RAI Toolbox](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.