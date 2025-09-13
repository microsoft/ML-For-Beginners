<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T16:51:49+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "ro"
}
-->
# Sarcini și tehnici comune de procesare a limbajului natural

Pentru majoritatea sarcinilor de *procesare a limbajului natural*, textul care trebuie procesat trebuie să fie descompus, examinat, iar rezultatele stocate sau corelate cu reguli și seturi de date. Aceste sarcini permit programatorului să derive _semnificația_ sau _intenția_ sau doar _frecvența_ termenilor și cuvintelor dintr-un text.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

Să descoperim tehnicile comune utilizate în procesarea textului. Combinate cu învățarea automată, aceste tehnici te ajută să analizezi eficient cantități mari de text. Totuși, înainte de a aplica ML acestor sarcini, să înțelegem problemele întâmpinate de un specialist NLP.

## Sarcini comune în NLP

Există diferite moduri de a analiza un text pe care lucrezi. Există sarcini pe care le poți efectua și, prin aceste sarcini, poți înțelege textul și trage concluzii. De obicei, aceste sarcini sunt realizate într-o anumită ordine.

### Tokenizare

Probabil primul lucru pe care majoritatea algoritmilor NLP trebuie să-l facă este să împartă textul în tokeni sau cuvinte. Deși pare simplu, luarea în considerare a punctuației și a delimitatorilor de cuvinte și propoziții din diferite limbi poate fi complicată. Este posibil să fie nevoie să folosești diverse metode pentru a determina delimitările.

![tokenizare](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizarea unei propoziții din **Mândrie și Prejudecată**. Infografic de [Jen Looper](https://twitter.com/jenlooper)

### Embedding-uri

[Embedding-urile de cuvinte](https://wikipedia.org/wiki/Word_embedding) sunt o modalitate de a converti datele text în format numeric. Embedding-urile sunt realizate astfel încât cuvintele cu semnificații similare sau cuvintele utilizate împreună să se grupeze.

![embedding-uri de cuvinte](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Am cel mai mare respect pentru nervii tăi, sunt prietenii mei vechi." - Embedding-uri de cuvinte pentru o propoziție din **Mândrie și Prejudecată**. Infografic de [Jen Looper](https://twitter.com/jenlooper)

✅ Încearcă [acest instrument interesant](https://projector.tensorflow.org/) pentru a experimenta cu embedding-uri de cuvinte. Clic pe un cuvânt arată grupuri de cuvinte similare: 'jucărie' se grupează cu 'disney', 'lego', 'playstation' și 'consolă'.

### Parsing și Etichetarea părților de vorbire

Fiecare cuvânt care a fost tokenizat poate fi etichetat ca parte de vorbire - substantiv, verb sau adjectiv. Propoziția `vulpea roșie rapidă a sărit peste câinele maro leneș` ar putea fi etichetată POS astfel: vulpea = substantiv, sărit = verb.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing-ul unei propoziții din **Mândrie și Prejudecată**. Infografic de [Jen Looper](https://twitter.com/jenlooper)

Parsing-ul înseamnă recunoașterea cuvintelor care sunt legate între ele într-o propoziție - de exemplu, `vulpea roșie rapidă a sărit` este o secvență adjectiv-substantiv-verb care este separată de secvența `câinele maro leneș`.

### Frecvența cuvintelor și expresiilor

O procedură utilă atunci când analizezi un corp mare de text este să construiești un dicționar al fiecărui cuvânt sau expresie de interes și cât de des apare. Expresia `vulpea roșie rapidă a sărit peste câinele maro leneș` are o frecvență de 2 pentru cuvântul "the".

Să analizăm un exemplu de text în care numărăm frecvența cuvintelor. Poemul lui Rudyard Kipling, The Winners, conține următorul vers:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Deoarece frecvențele expresiilor pot fi insensibile sau sensibile la majuscule, expresia `un prieten` are o frecvență de 2, `the` are o frecvență de 6, iar `travels` are o frecvență de 2.

### N-gramuri

Un text poate fi împărțit în secvențe de cuvinte de o lungime setată, un singur cuvânt (unigram), două cuvinte (bigram), trei cuvinte (trigram) sau orice număr de cuvinte (n-gramuri).

De exemplu, `vulpea roșie rapidă a sărit peste câinele maro leneș` cu un scor n-gram de 2 produce următoarele n-gramuri:

1. vulpea roșie  
2. roșie rapidă  
3. rapidă a  
4. a sărit  
5. sărit peste  
6. peste câinele  
7. câinele maro  
8. maro leneș  

Ar putea fi mai ușor să vizualizezi acest lucru ca o fereastră glisantă peste propoziție. Iată cum arată pentru n-gramuri de 3 cuvinte, n-gramul fiind evidențiat în fiecare propoziție:

1.   <u>**vulpea roșie rapidă**</u> a sărit peste câinele maro leneș  
2.   vulpea **<u>roșie rapidă a</u>** sărit peste câinele maro leneș  
3.   vulpea roșie **<u>rapidă a sărit</u>** peste câinele maro leneș  
4.   vulpea roșie rapidă **<u>a sărit peste</u>** câinele maro leneș  
5.   vulpea roșie rapidă a **<u>sărit peste câinele</u>** maro leneș  
6.   vulpea roșie rapidă a sărit **<u>peste câinele maro</u>** leneș  
7.   vulpea roșie rapidă a sărit peste <u>**câinele maro leneș**</u>  

![fereastră glisantă n-gramuri](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Valoare n-gram de 3: Infografic de [Jen Looper](https://twitter.com/jenlooper)

### Extracția expresiilor nominale

În majoritatea propozițiilor, există un substantiv care este subiectul sau obiectul propoziției. În engleză, acesta poate fi identificat adesea prin prezența cuvintelor 'a', 'an' sau 'the' înaintea sa. Identificarea subiectului sau obiectului unei propoziții prin 'extracția expresiei nominale' este o sarcină comună în NLP atunci când se încearcă înțelegerea semnificației unei propoziții.

✅ În propoziția "Nu pot fixa ora, sau locul, sau privirea sau cuvintele, care au pus bazele. Este prea demult. Eram în mijloc înainte să știu că am început.", poți identifica expresiile nominale?

În propoziția `vulpea roșie rapidă a sărit peste câinele maro leneș` există 2 expresii nominale: **vulpea roșie rapidă** și **câinele maro leneș**.

### Analiza sentimentului

O propoziție sau un text poate fi analizat pentru sentiment, sau cât de *pozitiv* sau *negativ* este. Sentimentul este măsurat în *polaritate* și *obiectivitate/subiectivitate*. Polaritatea este măsurată de la -1.0 la 1.0 (negativ la pozitiv) și de la 0.0 la 1.0 (cel mai obiectiv la cel mai subiectiv).

✅ Mai târziu vei învăța că există diferite moduri de a determina sentimentul folosind învățarea automată, dar un mod este să ai o listă de cuvinte și expresii care sunt categorisite ca pozitive sau negative de un expert uman și să aplici acel model textului pentru a calcula un scor de polaritate. Poți vedea cum ar funcționa acest lucru în unele circumstanțe și mai puțin bine în altele?

### Flexiune

Flexiunea îți permite să iei un cuvânt și să obții forma singulară sau plurală a acestuia.

### Lemmatizare

Un *lemma* este rădăcina sau cuvântul principal pentru un set de cuvinte, de exemplu *zburat*, *zboară*, *zburând* au ca lemma verbul *a zbura*.

Există, de asemenea, baze de date utile disponibile pentru cercetătorii NLP, în special:

### WordNet

[WordNet](https://wordnet.princeton.edu/) este o bază de date de cuvinte, sinonime, antonime și multe alte detalii pentru fiecare cuvânt în multe limbi diferite. Este incredibil de utilă atunci când se încearcă construirea de traduceri, verificatoare ortografice sau instrumente lingvistice de orice tip.

## Biblioteci NLP

Din fericire, nu trebuie să construiești toate aceste tehnici singur, deoarece există biblioteci excelente de Python disponibile care le fac mult mai accesibile pentru dezvoltatorii care nu sunt specializați în procesarea limbajului natural sau învățarea automată. Lecțiile următoare includ mai multe exemple despre acestea, dar aici vei învăța câteva exemple utile pentru a te ajuta cu următoarea sarcină.

### Exercițiu - utilizarea bibliotecii `TextBlob`

Să folosim o bibliotecă numită TextBlob, deoarece conține API-uri utile pentru abordarea acestor tipuri de sarcini. TextBlob "se bazează pe umerii giganților [NLTK](https://nltk.org) și [pattern](https://github.com/clips/pattern), și funcționează bine cu ambele." Are o cantitate considerabilă de ML integrată în API-ul său.

> Notă: Un [Ghid de Start Rapid](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) util este disponibil pentru TextBlob și este recomandat pentru dezvoltatorii Python experimentați.

Când încerci să identifici *expresii nominale*, TextBlob oferă mai multe opțiuni de extractoare pentru a găsi expresii nominale.

1. Aruncă o privire la `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Ce se întâmplă aici? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) este "Un extractor de expresii nominale care folosește parsing-ul de fragmente antrenat cu corpusul de antrenament ConLL-2000." ConLL-2000 se referă la Conferința din 2000 despre Învățarea Computațională a Limbajului Natural. Fiecare an, conferința a găzduit un workshop pentru a aborda o problemă dificilă NLP, iar în 2000 aceasta a fost fragmentarea expresiilor nominale. Un model a fost antrenat pe Wall Street Journal, cu "secțiunile 15-18 ca date de antrenament (211727 tokeni) și secțiunea 20 ca date de testare (47377 tokeni)". Poți consulta procedurile utilizate [aici](https://www.clips.uantwerpen.be/conll2000/chunking/) și [rezultatele](https://ifarm.nl/erikt/research/np-chunking.html).

### Provocare - îmbunătățirea botului tău cu NLP

În lecția anterioară ai construit un bot foarte simplu de întrebări și răspunsuri. Acum, vei face ca Marvin să fie puțin mai empatic analizând inputul tău pentru sentiment și afișând un răspuns care să se potrivească sentimentului. De asemenea, va trebui să identifici o `expresie nominală` și să întrebi despre aceasta.

Pașii tăi pentru construirea unui bot conversațional mai bun:

1. Afișează instrucțiuni care sfătuiesc utilizatorul cum să interacționeze cu botul  
2. Pornește bucla  
   1. Acceptă inputul utilizatorului  
   2. Dacă utilizatorul a cerut să iasă, atunci ieși  
   3. Procesează inputul utilizatorului și determină răspunsul adecvat pentru sentiment  
   4. Dacă se detectează o expresie nominală în sentiment, pluralizeaz-o și cere mai multe informații despre acel subiect  
   5. Afișează răspunsul  
3. Revino la pasul 2  

Iată fragmentul de cod pentru determinarea sentimentului folosind TextBlob. Observă că există doar patru *grade* de răspunsuri la sentiment (poți avea mai multe dacă dorești):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Iată un exemplu de output pentru a te ghida (inputul utilizatorului este pe liniile care încep cu >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

O soluție posibilă pentru sarcină este [aici](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Verificare cunoștințe

1. Crezi că răspunsurile empatice ar putea 'păcăli' pe cineva să creadă că botul chiar îi înțelege?  
2. Face identificarea expresiei nominale botul mai 'credibil'?  
3. De ce ar fi utilă extragerea unei 'expresii nominale' dintr-o propoziție?  

---

Implementează botul din verificarea cunoștințelor anterioare și testează-l pe un prieten. Poate să-l păcălească? Poți face botul tău mai 'credibil'?

## 🚀Provocare

Ia o sarcină din verificarea cunoștințelor anterioare și încearcă să o implementezi. Testează botul pe un prieten. Poate să-l păcălească? Poți face botul tău mai 'credibil'?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și Studiu Individual

În următoarele câteva lecții vei învăța mai multe despre analiza sentimentului. Cercetează această tehnică interesantă în articole precum cele de pe [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Temă

[Fă un bot să răspundă](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.