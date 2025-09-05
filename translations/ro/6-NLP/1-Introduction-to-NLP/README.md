<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T17:01:10+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "ro"
}
-->
# Introducere în procesarea limbajului natural

Această lecție acoperă o scurtă istorie și concepte importante ale *procesării limbajului natural*, un subdomeniu al *lingvisticii computaționale*.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

## Introducere

NLP, cum este cunoscut în mod obișnuit, este unul dintre cele mai cunoscute domenii în care învățarea automată a fost aplicată și utilizată în software-ul de producție.

✅ Poți să te gândești la un software pe care îl folosești zilnic și care probabil are integrat NLP? Ce zici de programele de procesare a textului sau aplicațiile mobile pe care le folosești regulat?

Vei învăța despre:

- **Ideea limbajelor**. Cum s-au dezvoltat limbajele și care au fost principalele domenii de studiu.
- **Definiții și concepte**. Vei învăța definiții și concepte despre cum procesează computerele textul, inclusiv analizarea, gramatica și identificarea substantivelor și verbelor. Există câteva sarcini de codare în această lecție, iar mai multe concepte importante sunt introduse, pe care le vei învăța să le codifici în lecțiile următoare.

## Lingvistică computațională

Lingvistica computațională este un domeniu de cercetare și dezvoltare de-a lungul multor decenii care studiază modul în care computerele pot lucra cu limbajele, le pot înțelege, traduce și comunica. Procesarea limbajului natural (NLP) este un domeniu conex, concentrat pe modul în care computerele pot procesa limbajele 'naturale', adică limbajele umane.

### Exemplu - dictare pe telefon

Dacă ai dictat vreodată unui telefon în loc să tastezi sau ai întrebat un asistent virtual ceva, discursul tău a fost convertit într-o formă text și apoi procesat sau *analizat* din limbajul pe care l-ai vorbit. Cuvintele-cheie detectate au fost apoi procesate într-un format pe care telefonul sau asistentul l-a putut înțelege și pe baza căruia a acționat.

![comprehension](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Înțelegerea lingvistică reală este dificilă! Imagine de [Jen Looper](https://twitter.com/jenlooper)

### Cum este posibilă această tehnologie?

Acest lucru este posibil deoarece cineva a scris un program de computer pentru a face acest lucru. Cu câteva decenii în urmă, unii scriitori de science fiction au prezis că oamenii vor vorbi în principal cu computerele lor, iar computerele vor înțelege întotdeauna exact ce au vrut să spună. Din păcate, s-a dovedit a fi o problemă mai dificilă decât și-au imaginat mulți, și, deși este o problemă mult mai bine înțeleasă astăzi, există provocări semnificative în atingerea unei procesări 'perfecte' a limbajului natural atunci când vine vorba de înțelegerea sensului unei propoziții. Aceasta este o problemă deosebit de dificilă când vine vorba de înțelegerea umorului sau detectarea emoțiilor, cum ar fi sarcasmul, într-o propoziție.

În acest moment, s-ar putea să îți amintești de orele de școală în care profesorul acoperea părțile de gramatică dintr-o propoziție. În unele țări, elevii sunt învățați gramatică și lingvistică ca materie dedicată, dar în multe, aceste subiecte sunt incluse ca parte a învățării unui limbaj: fie limba maternă în școala primară (învățarea cititului și scrisului), fie o limbă secundară în școala gimnazială sau liceu. Nu te îngrijora dacă nu ești expert în diferențierea substantivelor de verbe sau adverbelor de adjective!

Dacă te lupți cu diferența dintre *prezentul simplu* și *prezentul continuu*, nu ești singur. Acesta este un lucru provocator pentru mulți oameni, chiar și pentru vorbitorii nativi ai unei limbi. Vestea bună este că computerele sunt foarte bune la aplicarea regulilor formale, iar tu vei învăța să scrii cod care poate *analiza* o propoziție la fel de bine ca un om. Provocarea mai mare pe care o vei examina mai târziu este înțelegerea *sensului* și *sentimentului* unei propoziții.

## Cerințe preliminare

Pentru această lecție, cerința principală este să poți citi și înțelege limba lecției. Nu există probleme de matematică sau ecuații de rezolvat. Deși autorul original a scris această lecție în engleză, ea este tradusă și în alte limbi, așa că s-ar putea să citești o traducere. Există exemple în care sunt utilizate mai multe limbi diferite (pentru a compara regulile gramaticale diferite ale limbilor). Acestea *nu* sunt traduse, dar textul explicativ este, astfel încât sensul ar trebui să fie clar.

Pentru sarcinile de codare, vei folosi Python, iar exemplele sunt în Python 3.8.

În această secțiune, vei avea nevoie și vei folosi:

- **Înțelegerea Python 3**. Înțelegerea limbajului de programare Python 3, această lecție folosește input, bucle, citirea fișierelor, array-uri.
- **Visual Studio Code + extensie**. Vom folosi Visual Studio Code și extensia sa pentru Python. Poți folosi și un IDE Python la alegerea ta.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) este o bibliotecă simplificată de procesare a textului pentru Python. Urmează instrucțiunile de pe site-ul TextBlob pentru a-l instala pe sistemul tău (instalează și corpora, așa cum este arătat mai jos):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Sfat: Poți rula Python direct în mediile VS Code. Verifică [documentația](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) pentru mai multe informații.

## Vorbind cu mașinile

Istoria încercării de a face computerele să înțeleagă limbajul uman datează de câteva decenii, iar unul dintre primii oameni de știință care a luat în considerare procesarea limbajului natural a fost *Alan Turing*.

### Testul 'Turing'

Când Turing cerceta *inteligența artificială* în anii 1950, el s-a gândit dacă un test conversațional ar putea fi dat unui om și unui computer (prin corespondență scrisă), unde omul din conversație nu era sigur dacă conversa cu un alt om sau cu un computer.

Dacă, după o anumită durată a conversației, omul nu putea determina dacă răspunsurile proveneau de la un computer sau nu, atunci putea fi spus că computerul *gândește*?

### Inspirația - 'jocul de imitație'

Ideea pentru acest test a venit dintr-un joc de petrecere numit *Jocul de Imitație*, unde un interogator este singur într-o cameră și are sarcina de a determina care dintre două persoane (în altă cameră) sunt bărbat și femeie, respectiv. Interogatorul poate trimite note și trebuie să încerce să gândească întrebări ale căror răspunsuri scrise dezvăluie genul persoanei misterioase. Desigur, jucătorii din cealaltă cameră încearcă să inducă în eroare interogatorul prin răspunsuri care să-l deruteze sau să-l confuzeze, în timp ce dau impresia că răspund sincer.

### Dezvoltarea Eliza

În anii 1960, un om de știință de la MIT numit *Joseph Weizenbaum* a dezvoltat [*Eliza*](https://wikipedia.org/wiki/ELIZA), un 'terapeut' computerizat care punea întrebări omului și dădea impresia că înțelege răspunsurile acestuia. Totuși, deși Eliza putea analiza o propoziție și identifica anumite construcții gramaticale și cuvinte-cheie pentru a da un răspuns rezonabil, nu putea fi spus că *înțelege* propoziția. Dacă Eliza era prezentată cu o propoziție în formatul "**Eu sunt** <u>trist</u>", ar putea rearanja și substitui cuvinte în propoziție pentru a forma răspunsul "De cât timp **ești** <u>trist</u>?".

Acest lucru dădea impresia că Eliza înțelegea afirmația și punea o întrebare de continuare, în timp ce, în realitate, schimba timpul verbal și adăuga câteva cuvinte. Dacă Eliza nu putea identifica un cuvânt-cheie pentru care avea un răspuns, ar fi dat un răspuns aleatoriu care ar fi putut fi aplicabil la multe afirmații diferite. Eliza putea fi ușor păcălită, de exemplu, dacă un utilizator scria "**Tu ești** o <u>bicicletă</u>", ar putea răspunde cu "De cât timp **sunt** o <u>bicicletă</u>?", în loc de un răspuns mai rațional.

[![Chatting with Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 Click pe imaginea de mai sus pentru un videoclip despre programul original ELIZA

> Notă: Poți citi descrierea originală a [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicată în 1966 dacă ai un cont ACM. Alternativ, citește despre Eliza pe [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercițiu - codarea unui bot conversațional de bază

Un bot conversațional, precum Eliza, este un program care solicită input de la utilizator și pare să înțeleagă și să răspundă inteligent. Spre deosebire de Eliza, botul nostru nu va avea mai multe reguli care să-i dea impresia unei conversații inteligente. În schimb, botul nostru va avea o singură abilitate, aceea de a menține conversația cu răspunsuri aleatorii care ar putea funcționa în aproape orice conversație trivială.

### Planul

Pașii tăi pentru construirea unui bot conversațional:

1. Printează instrucțiuni care sfătuiesc utilizatorul cum să interacționeze cu botul
2. Pornește o buclă
   1. Acceptă input de la utilizator
   2. Dacă utilizatorul a cerut să iasă, atunci ieși
   3. Procesează inputul utilizatorului și determină răspunsul (în acest caz, răspunsul este o alegere aleatorie dintr-o listă de posibile răspunsuri generice)
   4. Printează răspunsul
3. Revino la pasul 2

### Construirea botului

Să creăm botul acum. Vom începe prin definirea unor fraze.

1. Creează acest bot în Python cu următoarele răspunsuri aleatorii:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Iată un exemplu de output pentru a te ghida (inputul utilizatorului este pe liniile care încep cu `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    O soluție posibilă pentru sarcină este [aici](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Oprește-te și gândește-te

    1. Crezi că răspunsurile aleatorii ar 'păcăli' pe cineva să creadă că botul chiar îi înțelege?
    2. Ce caracteristici ar trebui să aibă botul pentru a fi mai eficient?
    3. Dacă un bot ar putea cu adevărat să 'înțeleagă' sensul unei propoziții, ar trebui să 'își amintească' sensul propozițiilor anterioare dintr-o conversație?

---

## 🚀Provocare

Alege unul dintre elementele "oprește-te și gândește-te" de mai sus și încearcă fie să-l implementezi în cod, fie să scrii o soluție pe hârtie folosind pseudocod.

În lecția următoare, vei învăța despre o serie de alte abordări pentru analizarea limbajului natural și învățarea automată.

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Aruncă o privire la referințele de mai jos ca oportunități de lectură suplimentară.

### Referințe

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Temă 

[Caută un bot](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.