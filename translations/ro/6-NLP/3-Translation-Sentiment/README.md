<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T17:04:46+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "ro"
}
-->
# Traducere și analiză a sentimentelor cu ML

În lecțiile anterioare ai învățat cum să construiești un bot de bază folosind `TextBlob`, o bibliotecă care încorporează ML în culise pentru a efectua sarcini NLP de bază, cum ar fi extragerea frazelor nominale. O altă provocare importantă în lingvistica computațională este traducerea _exactă_ a unei propoziții dintr-o limbă vorbită sau scrisă în alta.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

Traducerea este o problemă foarte dificilă, complicată de faptul că există mii de limbi, fiecare având reguli gramaticale foarte diferite. O abordare este să convertești regulile gramaticale formale ale unei limbi, cum ar fi engleza, într-o structură independentă de limbă și apoi să o traduci prin conversia înapoi într-o altă limbă. Această abordare presupune următorii pași:

1. **Identificare**. Identifică sau etichetează cuvintele din limba de intrare ca substantive, verbe etc.
2. **Crearea traducerii**. Produce o traducere directă a fiecărui cuvânt în formatul limbii țintă.

### Exemplu de propoziție, engleză în irlandeză

În 'engleză', propoziția _I feel happy_ are trei cuvinte în ordinea:

- **subiect** (I)
- **verb** (feel)
- **adjectiv** (happy)

Totuși, în limba 'irlandeză', aceeași propoziție are o structură gramaticală foarte diferită - emoțiile precum "*happy*" sau "*sad*" sunt exprimate ca fiind *asupra* ta.

Fraza engleză `I feel happy` în irlandeză ar fi `Tá athas orm`. O traducere *literală* ar fi `Happy is upon me`.

Un vorbitor de irlandeză care traduce în engleză ar spune `I feel happy`, nu `Happy is upon me`, deoarece înțelege sensul propoziției, chiar dacă cuvintele și structura propoziției sunt diferite.

Ordinea formală pentru propoziția în irlandeză este:

- **verb** (Tá sau is)
- **adjectiv** (athas, sau happy)
- **subiect** (orm, sau upon me)

## Traducere

Un program de traducere naiv ar putea traduce doar cuvintele, ignorând structura propoziției.

✅ Dacă ai învățat o a doua (sau a treia sau mai multe) limbă ca adult, probabil ai început prin a gândi în limba ta maternă, traducând un concept cuvânt cu cuvânt în mintea ta în a doua limbă și apoi exprimând traducerea. Acest lucru este similar cu ceea ce fac programele de traducere naive. Este important să depășești această fază pentru a atinge fluența!

Traducerea naivă duce la traduceri greșite (și uneori amuzante): `I feel happy` se traduce literal în `Mise bhraitheann athas` în irlandeză. Asta înseamnă (literal) `me feel happy` și nu este o propoziție validă în irlandeză. Chiar dacă engleza și irlandeza sunt limbi vorbite pe două insule vecine, ele sunt foarte diferite, având structuri gramaticale diferite.

> Poți viziona câteva videoclipuri despre tradițiile lingvistice irlandeze, cum ar fi [acesta](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Abordări bazate pe învățarea automată

Până acum, ai învățat despre abordarea regulilor formale în procesarea limbajului natural. O altă abordare este să ignori sensul cuvintelor și _în schimb să folosești învățarea automată pentru a detecta modele_. Acest lucru poate funcționa în traducere dacă ai multe texte (un *corpus*) sau texte (*corpora*) în ambele limbi, de origine și țintă.

De exemplu, consideră cazul *Mândrie și Prejudecată*, un roman bine-cunoscut scris de Jane Austen în 1813. Dacă consulți cartea în engleză și o traducere umană a cărții în *franceză*, ai putea detecta fraze în una care sunt traduse _idiomatic_ în cealaltă. Vei face asta în curând.

De exemplu, când o frază engleză precum `I have no money` este tradusă literal în franceză, ar putea deveni `Je n'ai pas de monnaie`. "Monnaie" este un 'fals cognat' francez complicat, deoarece 'money' și 'monnaie' nu sunt sinonime. O traducere mai bună pe care ar putea-o face un om ar fi `Je n'ai pas d'argent`, deoarece transmite mai bine sensul că nu ai bani (mai degrabă decât 'mărunțiș', care este sensul lui 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Imagine de [Jen Looper](https://twitter.com/jenlooper)

Dacă un model ML are suficiente traduceri umane pentru a construi un model, poate îmbunătăți acuratețea traducerilor prin identificarea modelelor comune în texte care au fost traduse anterior de vorbitori experți ai ambelor limbi.

### Exercițiu - traducere

Poți folosi `TextBlob` pentru a traduce propoziții. Încearcă celebra primă frază din **Mândrie și Prejudecată**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` face o treabă destul de bună la traducere: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Se poate argumenta că traducerea lui TextBlob este mult mai exactă, de fapt, decât traducerea franceză din 1932 a cărții de V. Leconte și Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

În acest caz, traducerea informată de ML face o treabă mai bună decât traducătorul uman care pune inutil cuvinte în gura autorului original pentru 'claritate'.

> Ce se întâmplă aici? și de ce este TextBlob atât de bun la traducere? Ei bine, în culise, folosește Google Translate, o inteligență artificială sofisticată capabilă să analizeze milioane de fraze pentru a prezice cele mai bune șiruri pentru sarcina în cauză. Nu se întâmplă nimic manual aici și ai nevoie de o conexiune la internet pentru a folosi `blob.translate`.

✅ Încearcă câteva propoziții. Care este mai bună, traducerea ML sau cea umană? În ce cazuri?

## Analiza sentimentelor

Un alt domeniu în care învățarea automată poate funcționa foarte bine este analiza sentimentelor. O abordare non-ML a sentimentului este identificarea cuvintelor și frazelor care sunt 'pozitive' și 'negative'. Apoi, dat fiind un nou text, se calculează valoarea totală a cuvintelor pozitive, negative și neutre pentru a identifica sentimentul general. 

Această abordare poate fi ușor păcălită, așa cum ai văzut în sarcina Marvin - propoziția `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` este o propoziție sarcastică, cu sentiment negativ, dar algoritmul simplu detectează 'great', 'wonderful', 'glad' ca pozitive și 'waste', 'lost' și 'dark' ca negative. Sentimentul general este influențat de aceste cuvinte contradictorii.

✅ Oprește-te un moment și gândește-te cum transmitem sarcasmul ca vorbitori umani. Inflecția tonului joacă un rol important. Încearcă să spui fraza "Well, that film was awesome" în moduri diferite pentru a descoperi cum vocea ta transmite sensul.

### Abordări ML

Abordarea ML ar fi să aduni manual texte negative și pozitive - tweet-uri, recenzii de filme sau orice altceva unde omul a dat un scor *și* o opinie scrisă. Apoi, tehnicile NLP pot fi aplicate opiniilor și scorurilor, astfel încât să apară modele (de exemplu, recenziile pozitive ale filmelor tind să aibă fraza 'Oscar worthy' mai mult decât recenziile negative ale filmelor, sau recenziile pozitive ale restaurantelor spun 'gourmet' mult mai des decât 'disgusting').

> ⚖️ **Exemplu**: Dacă ai lucra în biroul unui politician și s-ar dezbate o nouă lege, alegătorii ar putea scrie biroului cu e-mailuri de susținere sau e-mailuri împotriva legii respective. Să presupunem că ești însărcinat să citești e-mailurile și să le sortezi în 2 grămezi, *pentru* și *împotrivă*. Dacă ar fi multe e-mailuri, ai putea fi copleșit încercând să le citești pe toate. Nu ar fi minunat dacă un bot ar putea să le citească pe toate pentru tine, să le înțeleagă și să îți spună în ce grămadă aparține fiecare e-mail? 
> 
> O modalitate de a realiza acest lucru este să folosești învățarea automată. Ai antrena modelul cu o parte din e-mailurile *împotrivă* și o parte din e-mailurile *pentru*. Modelul ar tinde să asocieze fraze și cuvinte cu partea împotrivă și partea pentru, *dar nu ar înțelege niciunul dintre conținuturi*, doar că anumite cuvinte și modele erau mai susceptibile să apară într-un e-mail *împotrivă* sau *pentru*. Ai putea să-l testezi cu câteva e-mailuri pe care nu le-ai folosit pentru a antrena modelul și să vezi dacă ajunge la aceeași concluzie ca tine. Apoi, odată ce ești mulțumit de acuratețea modelului, ai putea procesa e-mailurile viitoare fără a fi nevoie să le citești pe fiecare.

✅ Acest proces seamănă cu procesele pe care le-ai folosit în lecțiile anterioare?

## Exercițiu - propoziții sentimentale

Sentimentul este măsurat cu o *polaritate* de -1 la 1, ceea ce înseamnă că -1 este cel mai negativ sentiment, iar 1 este cel mai pozitiv. Sentimentul este de asemenea măsurat cu un scor de 0 - 1 pentru obiectivitate (0) și subiectivitate (1).

Aruncă o altă privire la *Mândrie și Prejudecată* de Jane Austen. Textul este disponibil aici la [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Exemplul de mai jos arată un scurt program care analizează sentimentul primelor și ultimelor propoziții din carte și afișează polaritatea sentimentului și scorul de subiectivitate/obiectivitate.

Ar trebui să folosești biblioteca `TextBlob` (descrisă mai sus) pentru a determina `sentimentul` (nu trebuie să scrii propriul calculator de sentiment) în următoarea sarcină.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vezi următorul output:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Provocare - verifică polaritatea sentimentului

Sarcina ta este să determini, folosind polaritatea sentimentului, dacă *Mândrie și Prejudecată* are mai multe propoziții absolut pozitive decât absolut negative. Pentru această sarcină, poți presupune că un scor de polaritate de 1 sau -1 este absolut pozitiv sau negativ, respectiv.

**Pași:**

1. Descarcă o [copie a Mândrie și Prejudecată](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) de la Project Gutenberg ca fișier .txt. Elimină metadatele de la începutul și sfârșitul fișierului, lăsând doar textul original
2. Deschide fișierul în Python și extrage conținutul ca șir
3. Creează un TextBlob folosind șirul cărții
4. Analizează fiecare propoziție din carte într-un loop
   1. Dacă polaritatea este 1 sau -1, stochează propoziția într-un array sau listă de mesaje pozitive sau negative
5. La final, afișează toate propozițiile pozitive și negative (separat) și numărul fiecăreia.

Iată o [soluție](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Verificare cunoștințe

1. Sentimentul se bazează pe cuvintele folosite în propoziție, dar codul *înțelege* cuvintele?
2. Crezi că polaritatea sentimentului este exactă, sau cu alte cuvinte, *ești de acord* cu scorurile?
   1. În special, ești de acord sau nu cu polaritatea absolut **pozitivă** a următoarelor propoziții?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Următoarele 3 propoziții au fost evaluate cu un sentiment absolut pozitiv, dar la o lectură atentă, ele nu sunt propoziții pozitive. De ce analiza sentimentului a considerat că sunt propoziții pozitive?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Ești de acord sau nu cu polaritatea absolut **negativă** a următoarelor propoziții?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Orice pasionat de Jane Austen va înțelege că ea folosește adesea cărțile sale pentru a critica aspectele mai ridicole ale societății engleze din perioada Regenței. Elizabeth Bennett, personajul principal din *Mândrie și Prejudecată*, este un observator social atent (la fel ca autoarea) și limbajul ei este adesea puternic nuanțat. Chiar și Mr. Darcy (interesul romantic din poveste) observă utilizarea jucăușă și ironică a limbajului de către Elizabeth: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Provocare

Poți face Marvin și mai bun prin extragerea altor caracteristici din inputul utilizatorului?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și auto-studiu
Există multe modalități de a extrage sentimentul din text. Gândește-te la aplicațiile de afaceri care ar putea folosi această tehnică. Reflectează asupra modului în care aceasta poate da greș. Citește mai multe despre sisteme sofisticate, pregătite pentru întreprinderi, care analizează sentimentul, cum ar fi [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Testează câteva dintre propozițiile din Mândrie și Prejudecată de mai sus și vezi dacă poate detecta nuanțele.

## Temă

[Licență poetică](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți de faptul că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.