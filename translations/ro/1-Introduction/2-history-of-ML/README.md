<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T16:10:41+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "ro"
}
-->
# Istoria învățării automate

![Rezumat al istoriei învățării automate într-o schiță](../../../../sketchnotes/ml-history.png)
> Schiță realizată de [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pentru începători - Istoria învățării automate](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML pentru începători - Istoria învățării automate")

> 🎥 Click pe imaginea de mai sus pentru un scurt videoclip despre această lecție.

În această lecție, vom parcurge principalele momente din istoria învățării automate și a inteligenței artificiale.

Istoria inteligenței artificiale (IA) ca domeniu este strâns legată de istoria învățării automate, deoarece algoritmii și progresele computaționale care stau la baza ML au contribuit la dezvoltarea IA. Este util să ne amintim că, deși aceste domenii ca arii distincte de cercetare au început să se cristalizeze în anii 1950, descoperiri importante [algoritmice, statistice, matematice, computaționale și tehnice](https://wikipedia.org/wiki/Timeline_of_machine_learning) au precedat și s-au suprapus cu această perioadă. De fapt, oamenii au reflectat asupra acestor întrebări de [sute de ani](https://wikipedia.org/wiki/History_of_artificial_intelligence): acest articol discută fundamentele intelectuale istorice ale ideii de 'mașină care gândește'.

---
## Descoperiri notabile

- 1763, 1812 [Teorema lui Bayes](https://wikipedia.org/wiki/Bayes%27_theorem) și predecesorii săi. Această teoremă și aplicațiile sale stau la baza inferenței, descriind probabilitatea unui eveniment pe baza cunoștințelor anterioare.
- 1805 [Teoria celor mai mici pătrate](https://wikipedia.org/wiki/Least_squares) de matematicianul francez Adrien-Marie Legendre. Această teorie, pe care o veți învăța în unitatea noastră despre regresie, ajută la ajustarea datelor.
- 1913 [Lanțurile Markov](https://wikipedia.org/wiki/Markov_chain), numite după matematicianul rus Andrey Markov, sunt utilizate pentru a descrie o secvență de evenimente posibile bazate pe o stare anterioară.
- 1957 [Perceptronul](https://wikipedia.org/wiki/Perceptron) este un tip de clasificator liniar inventat de psihologul american Frank Rosenblatt, care stă la baza progreselor în învățarea profundă.

---

- 1967 [Cel mai apropiat vecin](https://wikipedia.org/wiki/Nearest_neighbor) este un algoritm inițial conceput pentru a cartografia rute. În contextul ML, este utilizat pentru a detecta modele.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) este utilizat pentru a antrena [rețele neuronale feedforward](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Rețele neuronale recurente](https://wikipedia.org/wiki/Recurrent_neural_network) sunt rețele neuronale artificiale derivate din rețelele neuronale feedforward care creează grafice temporale.

✅ Faceți puțină cercetare. Ce alte date se remarcă ca fiind esențiale în istoria ML și IA?

---
## 1950: Mașini care gândesc

Alan Turing, o persoană cu adevărat remarcabilă, votat [de public în 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) drept cel mai mare om de știință al secolului XX, este creditat cu punerea bazelor conceptului de 'mașină care poate gândi'. El s-a confruntat cu sceptici și cu propria nevoie de dovezi empirice ale acestui concept, în parte prin crearea [Testului Turing](https://www.bbc.com/news/technology-18475646), pe care îl veți explora în lecțiile noastre despre NLP.

---
## 1956: Proiectul de cercetare de vară de la Dartmouth

"Proiectul de cercetare de vară de la Dartmouth despre inteligența artificială a fost un eveniment seminal pentru inteligența artificială ca domeniu," și aici a fost inventat termenul 'inteligență artificială' ([sursa](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Fiecare aspect al învățării sau orice altă caracteristică a inteligenței poate fi, în principiu, descris atât de precis încât o mașină poate fi construită pentru a o simula.

---

Cercetătorul principal, profesorul de matematică John McCarthy, spera "să procedeze pe baza conjecturii că fiecare aspect al învățării sau orice altă caracteristică a inteligenței poate fi, în principiu, descris atât de precis încât o mașină poate fi construită pentru a o simula." Participanții au inclus o altă personalitate marcantă din domeniu, Marvin Minsky.

Atelierul este creditat cu inițierea și încurajarea mai multor discuții, inclusiv "ascensiunea metodelor simbolice, sistemele axate pe domenii limitate (sisteme expert timpurii) și sistemele deductive versus sistemele inductive." ([sursa](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Anii de aur"

Din anii 1950 până la mijlocul anilor '70, optimismul era ridicat în speranța că IA ar putea rezolva multe probleme. În 1967, Marvin Minsky afirma cu încredere că "Într-o generație ... problema creării 'inteligenței artificiale' va fi substanțial rezolvată." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Cercetarea în procesarea limbajului natural a înflorit, căutarea a fost rafinată și făcută mai puternică, iar conceptul de 'micro-lumi' a fost creat, unde sarcini simple erau realizate folosind instrucțiuni în limbaj simplu.

---

Cercetarea era bine finanțată de agențiile guvernamentale, s-au făcut progrese în calcul și algoritmi, iar prototipuri de mașini inteligente au fost construite. Unele dintre aceste mașini includ:

* [Shakey robotul](https://wikipedia.org/wiki/Shakey_the_robot), care putea să se deplaseze și să decidă cum să îndeplinească sarcini 'inteligent'.

    ![Shakey, un robot inteligent](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey în 1972

---

* Eliza, un 'chatterbot' timpuriu, putea conversa cu oamenii și acționa ca un 'terapeut' primitiv. Veți învăța mai multe despre Eliza în lecțiile despre NLP.

    ![Eliza, un bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > O versiune a Eliza, un chatbot

---

* "Blocks world" era un exemplu de micro-lume unde blocurile puteau fi stivuite și sortate, iar experimentele în învățarea mașinilor să ia decizii puteau fi testate. Progresele realizate cu biblioteci precum [SHRDLU](https://wikipedia.org/wiki/SHRDLU) au ajutat la propulsarea procesării limbajului.

    [![blocks world cu SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world cu SHRDLU")

    > 🎥 Click pe imaginea de mai sus pentru un videoclip: Blocks world cu SHRDLU

---
## 1974 - 1980: "Iarna IA"

Până la mijlocul anilor 1970, devenise evident că complexitatea creării 'mașinilor inteligente' fusese subestimată și că promisiunea sa, având în vedere puterea de calcul disponibilă, fusese exagerată. Finanțarea s-a redus și încrederea în domeniu a încetinit. Unele probleme care au afectat încrederea includ:
---
- **Limitări**. Puterea de calcul era prea limitată.
- **Explozia combinatorică**. Numărul de parametri necesari pentru antrenare creștea exponențial pe măsură ce se cerea mai mult de la computere, fără o evoluție paralelă a puterii și capacității de calcul.
- **Lipsa de date**. Lipsa datelor a împiedicat procesul de testare, dezvoltare și rafinare a algoritmilor.
- **Punem întrebările corecte?**. Însăși întrebările care erau puse au început să fie puse sub semnul întrebării. Cercetătorii au început să primească critici cu privire la abordările lor:
  - Testele Turing au fost puse sub semnul întrebării prin mijloace, printre alte idei, ale 'teoriei camerei chineze', care susținea că "programarea unui computer digital poate face să pară că înțelege limbajul, dar nu poate produce o înțelegere reală." ([sursa](https://plato.stanford.edu/entries/chinese-room/))
  - Etica introducerii inteligențelor artificiale, cum ar fi "terapeutul" ELIZA, în societate a fost contestată.

---

În același timp, diverse școli de gândire IA au început să se formeze. S-a stabilit o dihotomie între practicile ["scruffy" vs. "neat IA"](https://wikipedia.org/wiki/Neats_and_scruffies). Laboratoarele _scruffy_ ajustau programele ore întregi până obțineau rezultatele dorite. Laboratoarele _neat_ "se concentrau pe logică și rezolvarea formală a problemelor". ELIZA și SHRDLU erau sisteme _scruffy_ bine cunoscute. În anii 1980, pe măsură ce a apărut cererea de a face sistemele ML reproducibile, abordarea _neat_ a preluat treptat prim-planul, deoarece rezultatele sale sunt mai explicabile.

---
## Sistemele expert din anii 1980

Pe măsură ce domeniul a crescut, beneficiul său pentru afaceri a devenit mai clar, iar în anii 1980 la fel și proliferarea 'sistemelor expert'. "Sistemele expert au fost printre primele forme cu adevărat de succes ale software-ului de inteligență artificială (IA)." ([sursa](https://wikipedia.org/wiki/Expert_system)).

Acest tip de sistem este de fapt _hibrid_, constând parțial dintr-un motor de reguli care definește cerințele de afaceri și un motor de inferență care folosește sistemul de reguli pentru a deduce noi fapte.

Această eră a văzut, de asemenea, o atenție sporită acordată rețelelor neuronale.

---
## 1987 - 1993: Răcirea IA

Proliferarea hardware-ului specializat pentru sistemele expert a avut efectul nefericit de a deveni prea specializat. Ascensiunea computerelor personale a concurat cu aceste sisteme mari, specializate și centralizate. Democratizarea calculului începuse și, în cele din urmă, a deschis calea pentru explozia modernă a datelor mari.

---
## 1993 - 2011

Această epocă a marcat o nouă eră pentru ML și IA, care au reușit să rezolve unele dintre problemele cauzate anterior de lipsa datelor și a puterii de calcul. Cantitatea de date a început să crească rapid și să devină mai accesibilă, în bine și în rău, mai ales odată cu apariția smartphone-ului în jurul anului 2007. Puterea de calcul a crescut exponențial, iar algoritmii au evoluat în paralel. Domeniul a început să câștige maturitate pe măsură ce zilele libere ale trecutului au început să se cristalizeze într-o adevărată disciplină.

---
## Acum

Astăzi, învățarea automată și IA ating aproape fiecare parte a vieților noastre. Această eră necesită o înțelegere atentă a riscurilor și efectelor potențiale ale acestor algoritmi asupra vieților umane. După cum a afirmat Brad Smith de la Microsoft, "Tehnologia informației ridică probleme care ating esența protecțiilor fundamentale ale drepturilor omului, cum ar fi confidențialitatea și libertatea de exprimare. Aceste probleme sporesc responsabilitatea companiilor de tehnologie care creează aceste produse. În opinia noastră, ele cer, de asemenea, reglementări guvernamentale bine gândite și dezvoltarea unor norme privind utilizările acceptabile" ([sursa](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Rămâne de văzut ce ne rezervă viitorul, dar este important să înțelegem aceste sisteme computerizate și software-ul și algoritmii pe care le rulează. Sperăm că acest curriculum vă va ajuta să obțineți o înțelegere mai bună, astfel încât să puteți decide singuri.

[![Istoria învățării profunde](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Istoria învățării profunde")
> 🎥 Click pe imaginea de mai sus pentru un videoclip: Yann LeCun discută despre istoria învățării profunde în această prelegere

---
## 🚀Provocare

Explorați unul dintre aceste momente istorice și aflați mai multe despre oamenii din spatele lor. Există personaje fascinante, iar nicio descoperire științifică nu a fost creată într-un vid cultural. Ce descoperiți?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

---
## Recapitulare și studiu individual

Iată câteva materiale de urmărit și ascultat:

[Acest podcast în care Amy Boyd discută evoluția IA](http://runasradio.com/Shows/Show/739)

[![Istoria IA de Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Istoria IA de Amy Boyd")

---

## Temă

[Crearea unei cronologii](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.