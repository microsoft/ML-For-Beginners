<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T16:07:43+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "ro"
}
-->
# Introducere în învățarea automată

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pentru începători - Introducere în Învățarea Automată pentru Începători](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pentru începători - Introducere în Învățarea Automată pentru Începători")

> 🎥 Click pe imaginea de mai sus pentru un scurt videoclip despre această lecție.

Bine ai venit la acest curs despre învățarea automată clasică pentru începători! Indiferent dacă ești complet nou în acest subiect sau un practician experimentat în ML care dorește să își reîmprospăteze cunoștințele, suntem bucuroși să te avem alături! Ne dorim să creăm un punct de plecare prietenos pentru studiul tău în ML și am fi încântați să evaluăm, să răspundem și să integrăm [feedback-ul tău](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introducere în ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introducere în ML")

> 🎥 Click pe imaginea de mai sus pentru un videoclip: John Guttag de la MIT introduce învățarea automată

---
## Începerea cu învățarea automată

Înainte de a începe acest curriculum, trebuie să îți configurezi computerul pentru a putea rula notebook-uri local.

- **Configurează-ți dispozitivul cu aceste videoclipuri**. Folosește următoarele linkuri pentru a învăța [cum să instalezi Python](https://youtu.be/CXZYvNRIAKM) pe sistemul tău și [cum să configurezi un editor de text](https://youtu.be/EU8eayHWoZg) pentru dezvoltare.
- **Învață Python**. Este recomandat să ai o înțelegere de bază a [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un limbaj de programare util pentru oamenii de știință în domeniul datelor pe care îl folosim în acest curs.
- **Învață Node.js și JavaScript**. Vom folosi JavaScript de câteva ori în acest curs pentru a construi aplicații web, așa că va trebui să ai [node](https://nodejs.org) și [npm](https://www.npmjs.com/) instalate, precum și [Visual Studio Code](https://code.visualstudio.com/) disponibil pentru dezvoltarea atât în Python, cât și în JavaScript.
- **Creează un cont GitHub**. Deoarece ne-ai găsit aici pe [GitHub](https://github.com), este posibil să ai deja un cont, dar dacă nu, creează unul și apoi clonează acest curriculum pentru a-l folosi pe cont propriu. (Nu ezita să ne dai și o stea 😊)
- **Explorează Scikit-learn**. Familiarizează-te cu [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un set de biblioteci ML pe care le referim în aceste lecții.

---
## Ce este învățarea automată?

Termenul 'învățare automată' este unul dintre cele mai populare și frecvent utilizate termene din zilele noastre. Există o posibilitate considerabilă ca să fi auzit acest termen cel puțin o dată dacă ai o oarecare familiaritate cu tehnologia, indiferent de domeniul în care lucrezi. Mecanismele învățării automate, însă, sunt un mister pentru majoritatea oamenilor. Pentru un începător în învățarea automată, subiectul poate părea uneori copleșitor. Prin urmare, este important să înțelegem ce este de fapt învățarea automată și să o studiem pas cu pas, prin exemple practice.

---
## Curba hype-ului

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends arată curba recentă de 'hype' a termenului 'învățare automată'

---
## Un univers misterios

Trăim într-un univers plin de mistere fascinante. Mari oameni de știință precum Stephen Hawking, Albert Einstein și mulți alții și-au dedicat viețile căutării de informații semnificative care să dezvăluie misterele lumii din jurul nostru. Aceasta este condiția umană a învățării: un copil învață lucruri noi și descoperă structura lumii sale an de an pe măsură ce crește.

---
## Creierul copilului

Creierul și simțurile unui copil percep faptele din mediul înconjurător și învață treptat tiparele ascunse ale vieții, care ajută copilul să creeze reguli logice pentru a identifica tiparele învățate. Procesul de învățare al creierului uman face ca oamenii să fie cele mai sofisticate ființe vii ale acestei lumi. Învățarea continuă prin descoperirea tiparelor ascunse și apoi inovarea pe baza acestor tipare ne permite să ne îmbunătățim constant pe parcursul vieții. Această capacitate de învățare și evoluție este legată de un concept numit [plasticitatea creierului](https://www.simplypsychology.org/brain-plasticity.html). Superficial, putem trasa unele similitudini motivaționale între procesul de învățare al creierului uman și conceptele de învățare automată.

---
## Creierul uman

[Creierul uman](https://www.livescience.com/29365-human-brain.html) percepe lucruri din lumea reală, procesează informațiile percepute, ia decizii raționale și efectuează anumite acțiuni în funcție de circumstanțe. Acesta este ceea ce numim comportament inteligent. Când programăm o replică a procesului comportamental inteligent într-o mașină, aceasta se numește inteligență artificială (AI).

---
## Un pic de terminologie

Deși termenii pot fi confundați, învățarea automată (ML) este un subset important al inteligenței artificiale. **ML se ocupă de utilizarea algoritmilor specializați pentru a descoperi informații semnificative și a găsi tipare ascunse din datele percepute pentru a susține procesul de luare a deciziilor raționale**.

---
## AI, ML, Învățare profundă

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Un diagramă care arată relațiile dintre AI, ML, învățarea profundă și știința datelor. Infografic de [Jen Looper](https://twitter.com/jenlooper) inspirat de [acest grafic](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepte de acoperit

În acest curriculum, vom acoperi doar conceptele de bază ale învățării automate pe care un începător trebuie să le cunoască. Vom aborda ceea ce numim 'învățare automată clasică', utilizând în principal Scikit-learn, o bibliotecă excelentă pe care mulți studenți o folosesc pentru a învăța elementele de bază. Pentru a înțelege conceptele mai largi ale inteligenței artificiale sau ale învățării profunde, o cunoaștere fundamentală solidă a învățării automate este indispensabilă, și dorim să o oferim aici.

---
## În acest curs vei învăța:

- conceptele de bază ale învățării automate
- istoria ML
- ML și echitatea
- tehnici de regresie ML
- tehnici de clasificare ML
- tehnici de grupare ML
- procesarea limbajului natural ML
- tehnici de prognoză a seriilor temporale ML
- învățare prin întărire
- aplicații reale pentru ML

---
## Ce nu vom acoperi

- învățarea profundă
- rețele neuronale
- AI

Pentru a oferi o experiență de învățare mai bună, vom evita complexitățile rețelelor neuronale, 'învățarea profundă' - construirea de modele cu multe straturi folosind rețele neuronale - și AI, pe care le vom discuta într-un alt curriculum. De asemenea, vom oferi un curriculum viitor despre știința datelor pentru a ne concentra pe acest aspect al acestui domeniu mai larg.

---
## De ce să studiezi învățarea automată?

Învățarea automată, din perspectiva sistemelor, este definită ca crearea de sisteme automate care pot învăța tipare ascunse din date pentru a ajuta la luarea deciziilor inteligente.

Această motivație este vag inspirată de modul în care creierul uman învață anumite lucruri pe baza datelor pe care le percepe din lumea exterioară.

✅ Gândește-te un minut de ce o afacere ar dori să folosească strategii de învățare automată în loc să creeze un motor bazat pe reguli codificate manual.

---
## Aplicații ale învățării automate

Aplicațiile învățării automate sunt acum aproape peste tot și sunt la fel de omniprezente ca datele care circulă în societățile noastre, generate de telefoanele noastre inteligente, dispozitivele conectate și alte sisteme. Având în vedere potențialul imens al algoritmilor de învățare automată de ultimă generație, cercetătorii au explorat capacitatea lor de a rezolva probleme multidimensionale și multidisciplinare din viața reală cu rezultate pozitive remarcabile.

---
## Exemple de ML aplicat

**Poți folosi învățarea automată în multe moduri**:

- Pentru a prezice probabilitatea unei boli pe baza istoricului medical sau a rapoartelor unui pacient.
- Pentru a utiliza datele meteorologice pentru a prezice evenimente meteorologice.
- Pentru a înțelege sentimentul unui text.
- Pentru a detecta știrile false și a opri răspândirea propagandei.

Finanțe, economie, știința pământului, explorarea spațiului, ingineria biomedicală, știința cognitivă și chiar domenii din științele umaniste au adaptat învățarea automată pentru a rezolva problemele grele de procesare a datelor din domeniul lor.

---
## Concluzie

Învățarea automată automatizează procesul de descoperire a tiparelor prin găsirea de informații semnificative din datele reale sau generate. S-a dovedit a fi extrem de valoroasă în aplicații de afaceri, sănătate și financiare, printre altele.

În viitorul apropiat, înțelegerea elementelor de bază ale învățării automate va deveni o necesitate pentru oamenii din orice domeniu datorită adoptării sale pe scară largă.

---
# 🚀 Provocare

Desenează, pe hârtie sau folosind o aplicație online precum [Excalidraw](https://excalidraw.com/), înțelegerea ta despre diferențele dintre AI, ML, învățarea profundă și știința datelor. Adaugă câteva idei despre problemele pe care fiecare dintre aceste tehnici le rezolvă bine.

# [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

---
# Recapitulare & Studiu individual

Pentru a afla mai multe despre cum poți lucra cu algoritmi ML în cloud, urmează acest [Parcurs de învățare](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Urmează un [Parcurs de învățare](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) despre elementele de bază ale ML.

---
# Temă

[Începe să lucrezi](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.