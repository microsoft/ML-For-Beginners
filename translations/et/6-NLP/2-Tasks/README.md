<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-10-11T11:37:14+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "et"
}
-->
# Looduslike keelte töötlemise (NLP) levinud ülesanded ja tehnikad

Enamiku *looduslike keelte töötlemise* ülesannete puhul tuleb töödeldav tekst jagada osadeks, analüüsida ja tulemused salvestada või reeglite ja andmekogumitega võrrelda. Need ülesanded võimaldavad programmeerijal tuletada teksti _tähendust_, _eesmärki_ või lihtsalt _sõnade ja terminite sagedust_.

## [Eelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

Tutvume levinud tehnikatega, mida kasutatakse teksti töötlemisel. Koos masinõppega aitavad need tehnikad analüüsida suuri tekstimahte tõhusalt. Enne ML-i rakendamist nendele ülesannetele on aga oluline mõista probleeme, millega NLP spetsialistid kokku puutuvad.

## NLP-le omased ülesanded

Teksti analüüsimiseks on erinevaid viise. On mitmeid ülesandeid, mida saab täita, ja nende abil on võimalik tekstist aru saada ning järeldusi teha. Tavaliselt viiakse need ülesanded läbi kindlas järjekorras.

### Tokeniseerimine

Esimene asi, mida enamik NLP algoritme teeb, on teksti jagamine tokeniteks ehk sõnadeks. Kuigi see kõlab lihtsana, võib kirjavahemärkide ja erinevate keelte sõna- ja lausepiiride arvestamine olla keeruline. Võib olla vaja kasutada erinevaid meetodeid, et määrata piire.

![tokeniseerimine](../../../../translated_images/et/tokenization.1641a160c66cd2d9.png)
> Lause tokeniseerimine **Uhkus ja eelarvamus** raamatust. Infograafika: [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Sõna embeddings](https://wikipedia.org/wiki/Word_embedding) on viis, kuidas tekstandmeid numbriliselt esitada. Embeddings tehakse nii, et sarnase tähendusega või koos kasutatavad sõnad grupeeritakse.

![sõna embeddings](../../../../translated_images/et/embedding.2cf8953c4b3101d1.png)
> "Mul on teie närvide vastu suurim austus, nad on minu vanad sõbrad." - Sõna embeddings lausele **Uhkus ja eelarvamus** raamatust. Infograafika: [Jen Looper](https://twitter.com/jenlooper)

✅ Proovi [seda huvitavat tööriista](https://projector.tensorflow.org/), et katsetada sõna embeddings. Klõpsates ühel sõnal, näed sarnaste sõnade klastreid: 'mänguasi' grupeerub 'disney', 'lego', 'playstation' ja 'konsooliga'.

### Parssimine ja sõnaliigi määramine

Iga tokeniseeritud sõna saab määrata sõnaliigi järgi - nimisõna, tegusõna või omadussõna. Näiteks lause `kiire punane rebane hüppas üle laisa pruuni koera` võib olla POS märgistatud järgmiselt: rebane = nimisõna, hüppas = tegusõna.

![parssimine](../../../../translated_images/et/parse.d0c5bbe1106eae8f.png)

> Lause parssimine **Uhkus ja eelarvamus** raamatust. Infograafika: [Jen Looper](https://twitter.com/jenlooper)

Parssimine tähendab sõnadevaheliste seoste tuvastamist lauses - näiteks `kiire punane rebane hüppas` on omadussõna-nimisõna-tegusõna järjestus, mis on eraldiseisev `laisa pruuni koera` järjestusest.

### Sõnade ja fraaside sagedused

Kasulik protseduur suure tekstimahu analüüsimisel on koostada sõnade või huvipakkuvate fraaside sõnastik ja määrata, kui sageli need esinevad. Näiteks fraasis `kiire punane rebane hüppas üle laisa pruuni koera` on sõna "the" sagedus 2.

Vaatame näidet, kus loendame sõnade sagedust. Rudyard Kiplingi luuletus "Võitjad" sisaldab järgmist salmi:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Fraaside sagedused võivad olla tõstutundlikud või mitte, vastavalt vajadusele. Näiteks fraasi `a friend` sagedus on 2, `the` sagedus on 6 ja `travels` sagedus on 2.

### N-grammid

Teksti saab jagada kindla pikkusega sõnade järjestusteks: üks sõna (unigramm), kaks sõna (bigramm), kolm sõna (trigramm) või mistahes arv sõnu (n-gramm).

Näiteks fraas `kiire punane rebane hüppas üle laisa pruuni koera` n-grammi pikkusega 2 annab järgmised n-grammid:

1. kiire punane  
2. punane rebane  
3. rebane hüppas  
4. hüppas üle  
5. üle laisa  
6. laisa pruun  
7. pruun koer  

Seda võib olla lihtsam visualiseerida kui libisevat kasti üle lause. Siin on see 3-sõnaliste n-grammide jaoks, n-gramm on igas lauses rasvases kirjas:

1.   <u>**kiire punane rebane**</u> hüppas üle laisa pruuni koera  
2.   kiire **<u>punane rebane hüppas</u>** üle laisa pruuni koera  
3.   kiire punane **<u>rebane hüppas üle</u>** laisa pruuni koera  
4.   kiire punane rebane **<u>hüppas üle laisa</u>** pruuni koera  
5.   kiire punane rebane hüppas **<u>üle laisa pruuni</u>** koera  
6.   kiire punane rebane hüppas üle <u>**laisa pruuni koera**</u>  

![n-grammide libisev aken](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-grammi väärtus 3: Infograafika: [Jen Looper](https://twitter.com/jenlooper)

### Nimisõnafraaside tuvastamine

Enamikus lausetes on nimisõna, mis on lause subjekt või objekt. Inglise keeles on see sageli tuvastatav sõnade 'a', 'an' või 'the' järgi, mis sellele eelnevad. Subjekti või objekti tuvastamine lauses nimisõnafraasi tuvastamise kaudu on NLP-s levinud ülesanne, kui püütakse lause tähendust mõista.

✅ Lauses "Ma ei suuda määrata aega, kohta, pilku ega sõnu, mis panid aluse. See oli liiga ammu. Ma olin keskel, enne kui aru sain, et olin alustanud." Kas suudad tuvastada nimisõnafraasid?

Lauses `kiire punane rebane hüppas üle laisa pruuni koera` on 2 nimisõnafraasi: **kiire punane rebane** ja **laisa pruun koer**.

### Sentimendi analüüs

Lause või tekst võib olla analüüsitud sentimendi ehk selle *positiivsuse* või *negatiivsuse* osas. Sentimenti mõõdetakse *polariteedi* ja *objektiivsuse/subjektiivsuse* järgi. Polariteeti mõõdetakse vahemikus -1.0 kuni 1.0 (negatiivne kuni positiivne) ja 0.0 kuni 1.0 (kõige objektiivsem kuni kõige subjektiivsem).

✅ Hiljem õpid, et sentimendi määramiseks on erinevaid viise masinõppe abil, kuid üks viis on kasutada sõnade ja fraaside loendit, mis on inimese eksperdi poolt kategoriseeritud positiivseks või negatiivseks, ning rakendada seda mudelit tekstile, et arvutada polariteedi skoor. Kas näed, kuidas see mõnes olukorras toimiks ja teistes mitte?

### Käänamine

Käänamine võimaldab võtta sõna ja tuvastada selle ainsuse või mitmuse vormi.

### Lemmatiseerimine

*Lemma* on sõnade kogumi juur- või põhisõna, näiteks *lendas*, *lendavad*, *lendamine* on lemma *lendama*.

Samuti on NLP uurijatele saadaval kasulikud andmebaasid, näiteks:

### WordNet

[WordNet](https://wordnet.princeton.edu/) on andmebaas, mis sisaldab sõnu, sünonüüme, antonüüme ja palju muid detaile iga sõna kohta erinevates keeltes. See on äärmiselt kasulik tõlgete, õigekirjakontrollide või mis tahes keele tööriistade loomisel.

## NLP teegid

Õnneks ei pea kõiki neid tehnikaid ise looma, sest olemas on suurepärased Python teegid, mis muudavad NLP ja masinõppe arendamise palju kättesaadavamaks arendajatele, kes pole spetsialiseerunud. Järgmistes tundides on rohkem näiteid, kuid siin õpid mõningaid kasulikke näiteid, mis aitavad sind järgmise ülesande juures.

### Harjutus - `TextBlob` teegi kasutamine

Kasutame teeki nimega TextBlob, kuna see sisaldab kasulikke API-sid nende ülesannete lahendamiseks. TextBlob "tugineb [NLTK](https://nltk.org) ja [pattern](https://github.com/clips/pattern) teekidele ning töötab hästi mõlemaga." Selle API-s on palju ML-i sisseehitatud.

> Märkus: Kasulik [Kiirstardi juhend](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) on saadaval TextBlob jaoks ja seda soovitatakse kogenud Python arendajatele.

Kui püüad tuvastada *nimisõnafraase*, pakub TextBlob mitmeid valikuid fraaside tuvastamiseks.

1. Vaata `ConllExtractor`-it.

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

    > Mis siin toimub? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) on "Nimisõnafraaside tuvastaja, mis kasutab chunk-parsimist, treenitud ConLL-2000 treeningkorpusega." ConLL-2000 viitab 2000. aasta looduslike keelte õppimise konverentsile. Igal aastal korraldas konverents töötoa, et lahendada keerulist NLP probleemi, ja 2000. aastal oli selleks nimisõnafraaside tuvastamine. Mudel treeniti Wall Street Journali andmetel, kasutades "sektsioone 15-18 treeningandmetena (211727 tokenit) ja sektsiooni 20 testandmetena (47377 tokenit)". Protseduure saab vaadata [siin](https://www.clips.uantwerpen.be/conll2000/chunking/) ja [tulemusi](https://ifarm.nl/erikt/research/np-chunking.html).

### Väljakutse - oma boti täiustamine NLP abil

Eelmises tunnis ehitasid väga lihtsa küsimuste ja vastuste boti. Nüüd teed Marvinist veidi kaastundlikuma, analüüsides sisendit sentimendi osas ja printides vastuse, mis vastab sentimendile. Samuti pead tuvastama `nimisõnafraasi` ja küsima selle kohta rohkem.

Sinu sammud parema vestlusboti loomiseks:

1. Prindi juhised, kuidas kasutaja saab botiga suhelda  
2. Alusta tsüklit  
   1. Võta kasutaja sisend  
   2. Kui kasutaja soovib väljuda, siis lõpeta  
   3. Töötle kasutaja sisend ja määra sobiv sentimendi vastus  
   4. Kui sentimendis tuvastatakse nimisõnafraas, muuda see mitmusesse ja küsi selle teema kohta rohkem  
   5. Prindi vastus  
3. Tagasi sammu 2 juurde  

Siin on koodilõik sentimendi määramiseks TextBlob abil. Märka, et on ainult neli *sentimendi vastuse gradienti* (võid lisada rohkem, kui soovid):

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

Siin on näidisväljund, mis juhendab sind (kasutaja sisend on ridadel, mis algavad >):

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

Üks võimalik lahendus ülesandele on [siin](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Teadmiste kontroll

1. Kas arvad, et kaastundlikud vastused võiksid kedagi 'petta', et bot tegelikult mõistab neid?  
2. Kas nimisõnafraasi tuvastamine muudab boti 'usutavamaks'?  
3. Miks võiks nimisõnafraasi tuvastamine lauses olla kasulik?  

---

Rakenda eelnevas teadmiste kontrollis olev bot ja testi seda sõbraga. Kas see suudab neid petta? Kas suudad muuta oma boti 'usutavamaks'?

## 🚀Väljakutse

Võta ülesanne eelnevast teadmiste kontrollist ja proovi seda rakendada. Testi boti sõbraga. Kas see suudab neid petta? Kas suudad muuta oma boti 'usutavamaks'?

## [Järelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülevaade ja iseseisev õppimine

Järgmistes tundides õpid rohkem sentimendi analüüsist. Uuri seda huvitavat tehnikat artiklitest, näiteks [KDNuggets](https://www.kdnuggets.com/tag/nlp).

## Ülesanne 

[Tee bot, mis vastab](assignment.md)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta arusaamatuste või valesti tõlgenduste eest, mis võivad tekkida selle tõlke kasutamisest.