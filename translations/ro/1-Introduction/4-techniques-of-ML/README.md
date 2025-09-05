<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T16:04:52+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "ro"
}
-->
# Tehnici de Învățare Automată

Procesul de construire, utilizare și întreținere a modelelor de învățare automată și a datelor pe care acestea le folosesc este foarte diferit de multe alte fluxuri de lucru din dezvoltare. În această lecție, vom demistifica procesul și vom evidenția principalele tehnici pe care trebuie să le cunoașteți. Veți:

- Înțelege procesele care stau la baza învățării automate la un nivel general.
- Explora concepte de bază precum „modele”, „predicții” și „date de antrenament”.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

[![ML pentru începători - Tehnici de Învățare Automată](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML pentru începători - Tehnici de Învățare Automată")

> 🎥 Faceți clic pe imaginea de mai sus pentru un scurt videoclip despre această lecție.

## Introducere

La un nivel general, arta de a crea procese de învățare automată (ML) constă dintr-o serie de pași:

1. **Decideți întrebarea**. Majoritatea proceselor ML încep prin a pune o întrebare care nu poate fi răspunsă printr-un program condițional simplu sau un motor bazat pe reguli. Aceste întrebări se concentrează adesea pe predicții bazate pe o colecție de date.
2. **Colectați și pregătiți datele**. Pentru a putea răspunde la întrebare, aveți nevoie de date. Calitatea și, uneori, cantitatea datelor vor determina cât de bine puteți răspunde la întrebarea inițială. Vizualizarea datelor este un aspect important al acestei etape. Această etapă include și împărțirea datelor în grupuri de antrenament și testare pentru a construi un model.
3. **Alegeți o metodă de antrenament**. În funcție de întrebare și de natura datelor, trebuie să alegeți cum doriți să antrenați un model pentru a reflecta cel mai bine datele și pentru a face predicții precise.
4. **Antrenați modelul**. Folosind datele de antrenament, veți utiliza diferiți algoritmi pentru a antrena un model să recunoască tipare în date. Modelul poate utiliza ponderi interne care pot fi ajustate pentru a privilegia anumite părți ale datelor în detrimentul altora, pentru a construi un model mai bun.
5. **Evaluați modelul**. Utilizați date pe care modelul nu le-a mai văzut (datele de testare) din setul colectat pentru a vedea cum se comportă modelul.
6. **Ajustarea parametrilor**. Pe baza performanței modelului, puteți relua procesul folosind parametri sau variabile diferite care controlează comportamentul algoritmilor utilizați pentru antrenarea modelului.
7. **Predicție**. Utilizați noi intrări pentru a testa acuratețea modelului.

## Ce întrebare să puneți

Calculatoarele sunt deosebit de abile în descoperirea tiparelor ascunse în date. Această utilitate este foarte utilă pentru cercetătorii care au întrebări despre un anumit domeniu și care nu pot fi ușor răspunse prin crearea unui motor bazat pe reguli condiționale. De exemplu, într-o sarcină actuarială, un specialist în date ar putea construi reguli manuale despre mortalitatea fumătorilor vs. nefumătorilor.

Când sunt introduse multe alte variabile în ecuație, un model ML ar putea fi mai eficient în a prezice ratele viitoare de mortalitate pe baza istoricului de sănătate din trecut. Un exemplu mai vesel ar putea fi realizarea de predicții meteorologice pentru luna aprilie într-o anumită locație, pe baza unor date precum latitudinea, longitudinea, schimbările climatice, proximitatea față de ocean, tiparele curenților de aer și altele.

✅ Acest [set de diapozitive](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) despre modelele meteorologice oferă o perspectivă istorică asupra utilizării ML în analiza vremii.  

## Sarcini înainte de construire

Înainte de a începe să construiți modelul, există mai multe sarcini pe care trebuie să le finalizați. Pentru a testa întrebarea și a forma o ipoteză bazată pe predicțiile unui model, trebuie să identificați și să configurați mai mulți factori.

### Date

Pentru a putea răspunde la întrebare cu un anumit grad de certitudine, aveți nevoie de o cantitate suficientă de date de tipul potrivit. Există două lucruri pe care trebuie să le faceți în acest moment:

- **Colectați date**. Ținând cont de lecția anterioară despre echitatea în analiza datelor, colectați datele cu atenție. Fiți conștienți de sursele acestor date, de eventualele prejudecăți inerente și documentați originea lor.
- **Pregătiți datele**. Există mai mulți pași în procesul de pregătire a datelor. Este posibil să fie nevoie să adunați datele și să le normalizați dacă provin din surse diverse. Puteți îmbunătăți calitatea și cantitatea datelor prin diverse metode, cum ar fi conversia șirurilor de caractere în numere (așa cum facem în [Clustering](../../5-Clustering/1-Visualize/README.md)). De asemenea, puteți genera date noi, bazate pe cele originale (așa cum facem în [Classification](../../4-Classification/1-Introduction/README.md)). Puteți curăța și edita datele (așa cum vom face înainte de lecția despre [Aplicații Web](../../3-Web-App/README.md)). În cele din urmă, este posibil să fie nevoie să le randomizați și să le amestecați, în funcție de tehnicile de antrenament.

✅ După ce ați colectat și procesat datele, luați un moment pentru a verifica dacă forma lor vă va permite să abordați întrebarea propusă. Este posibil ca datele să nu funcționeze bine pentru sarcina dată, așa cum descoperim în lecțiile noastre despre [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Caracteristici și Țintă

O [caracteristică](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) este o proprietate măsurabilă a datelor. În multe seturi de date, aceasta este exprimată ca un antet de coloană, cum ar fi „dată”, „dimensiune” sau „culoare”. Variabila caracteristică, de obicei reprezentată ca `X` în cod, reprezintă variabila de intrare care va fi utilizată pentru a antrena modelul.

Ținta este ceea ce încercați să preziceți. Ținta, de obicei reprezentată ca `y` în cod, reprezintă răspunsul la întrebarea pe care încercați să o puneți datelor: în decembrie, ce **culoare** vor avea dovlecii cei mai ieftini? În San Francisco, ce cartiere vor avea cel mai bun **preț** imobiliar? Uneori, ținta este denumită și atribut etichetă.

### Selectarea variabilei caracteristice

🎓 **Selecția și Extracția Caracteristicilor** Cum știți ce variabilă să alegeți atunci când construiți un model? Probabil veți trece printr-un proces de selecție sau extracție a caracteristicilor pentru a alege variabilele potrivite pentru cel mai performant model. Totuși, acestea nu sunt același lucru: „Extracția caracteristicilor creează noi caracteristici din funcții ale caracteristicilor originale, în timp ce selecția caracteristicilor returnează un subset al caracteristicilor.” ([sursa](https://wikipedia.org/wiki/Feature_selection))

### Vizualizați datele

Un aspect important al trusei de instrumente a unui specialist în date este puterea de a vizualiza datele folosind mai multe biblioteci excelente, cum ar fi Seaborn sau MatPlotLib. Reprezentarea vizuală a datelor vă poate permite să descoperiți corelații ascunse pe care le puteți valorifica. Vizualizările dvs. vă pot ajuta, de asemenea, să descoperiți prejudecăți sau date dezechilibrate (așa cum descoperim în [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Împărțiți setul de date

Înainte de antrenament, trebuie să împărțiți setul de date în două sau mai multe părți de dimensiuni inegale care să reprezinte totuși bine datele.

- **Antrenament**. Această parte a setului de date este utilizată pentru a antrena modelul. Acest set constituie majoritatea setului de date original.
- **Testare**. Un set de testare este un grup independent de date, adesea extras din datele originale, pe care îl utilizați pentru a confirma performanța modelului construit.
- **Validare**. Un set de validare este un grup mai mic de exemple independente pe care îl utilizați pentru a ajusta hiperparametrii sau arhitectura modelului, pentru a-l îmbunătăți. În funcție de dimensiunea datelor și de întrebarea pe care o puneți, este posibil să nu fie nevoie să construiți acest al treilea set (așa cum notăm în [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Construirea unui model

Folosind datele de antrenament, scopul dvs. este să construiți un model, sau o reprezentare statistică a datelor, utilizând diferiți algoritmi pentru a-l **antrena**. Antrenarea unui model îl expune la date și îi permite să facă presupuneri despre tiparele percepute pe care le descoperă, validează și acceptă sau respinge.

### Decideți o metodă de antrenament

În funcție de întrebare și de natura datelor, veți alege o metodă pentru a le antrena. Explorând [documentația Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - pe care o folosim în acest curs - puteți descoperi multe moduri de a antrena un model. În funcție de experiența dvs., este posibil să trebuiască să încercați mai multe metode diferite pentru a construi cel mai bun model. Este probabil să treceți printr-un proces în care specialiștii în date evaluează performanța unui model, oferindu-i date nevăzute, verificând acuratețea, prejudecățile și alte probleme care degradează calitatea, și selectând metoda de antrenament cea mai potrivită pentru sarcina dată.

### Antrenați un model

Cu datele de antrenament pregătite, sunteți gata să le „potriviți” pentru a crea un model. Veți observa că în multe biblioteci ML veți găsi codul „model.fit” - este momentul în care trimiteți variabila caracteristică ca un șir de valori (de obicei „X”) și o variabilă țintă (de obicei „y”).

### Evaluați modelul

Odată ce procesul de antrenament este complet (poate dura multe iterații sau „epoci” pentru a antrena un model mare), veți putea evalua calitatea modelului folosind date de testare pentru a-i măsura performanța. Aceste date sunt un subset al datelor originale pe care modelul nu le-a analizat anterior. Puteți imprima un tabel cu metrici despre calitatea modelului.

🎓 **Potrivirea modelului**

În contextul învățării automate, potrivirea modelului se referă la acuratețea funcției de bază a modelului în timp ce încearcă să analizeze date cu care nu este familiarizat.

🎓 **Subantrenarea** și **supraantrenarea** sunt probleme comune care degradează calitatea modelului, deoarece modelul se potrivește fie prea puțin, fie prea bine. Acest lucru face ca modelul să facă predicții fie prea strâns aliniate, fie prea slab aliniate cu datele de antrenament. Un model supraantrenat prezice datele de antrenament prea bine, deoarece a învățat prea bine detaliile și zgomotul datelor. Un model subantrenat nu este precis, deoarece nu poate analiza corect nici datele de antrenament, nici datele pe care nu le-a „văzut” încă.

![model supraantrenat](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografic de [Jen Looper](https://twitter.com/jenlooper)

## Ajustarea parametrilor

Odată ce antrenamentul inițial este complet, observați calitatea modelului și luați în considerare îmbunătățirea acestuia prin ajustarea „hiperparametrilor”. Citiți mai multe despre proces [în documentație](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predicție

Acesta este momentul în care puteți utiliza date complet noi pentru a testa acuratețea modelului. Într-un context ML „aplicat”, în care construiți active web pentru a utiliza modelul în producție, acest proces ar putea implica colectarea de intrări de la utilizatori (de exemplu, apăsarea unui buton) pentru a seta o variabilă și a o trimite modelului pentru inferență sau evaluare.

În aceste lecții, veți descoperi cum să utilizați acești pași pentru a pregăti, construi, testa, evalua și prezice - toate gesturile unui specialist în date și mai mult, pe măsură ce progresați în călătoria dvs. pentru a deveni un inginer ML „full stack”.

---

## 🚀Provocare

Desenați un diagramă de flux care reflectă pașii unui practician ML. Unde vă vedeți acum în proces? Unde anticipați că veți întâmpina dificultăți? Ce vi se pare ușor?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și Studiu Individual

Căutați online interviuri cu specialiști în date care discută despre munca lor zilnică. Iată [unul](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Temă

[Intervievați un specialist în date](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.