<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T16:49:56+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "sw"
}
-->
# Kazi za kawaida za usindikaji wa lugha asilia na mbinu zake

Kwa kazi nyingi za *usindikaji wa lugha asilia*, maandishi yanayopaswa kusindikiwa lazima yagawanywe, kuchunguzwa, na matokeo yake kuhifadhiwa au kulinganishwa na sheria na seti za data. Kazi hizi zinamruhusu mpangaji programu kupata _maana_, _nia_, au tu _mara kwa mara_ ya maneno na misemo katika maandishi.

## [Jaribio la awali la somo](https://ff-quizzes.netlify.app/en/ml/)

Hebu tujifunze mbinu za kawaida zinazotumika katika usindikaji wa maandishi. Zikichanganywa na ujifunzaji wa mashine, mbinu hizi zinakusaidia kuchambua kiasi kikubwa cha maandishi kwa ufanisi. Kabla ya kutumia ML kwa kazi hizi, hata hivyo, hebu tuelewe changamoto zinazokumbana na mtaalamu wa NLP.

## Kazi za kawaida za NLP

Kuna njia tofauti za kuchambua maandishi unayofanyia kazi. Kuna kazi unazoweza kutekeleza, na kupitia kazi hizi unaweza kuelewa maandishi na kutoa hitimisho. Kawaida unatekeleza kazi hizi kwa mpangilio.

### Ugawanyaji wa maneno (Tokenization)

Labda jambo la kwanza ambalo algorithimu nyingi za NLP zinapaswa kufanya ni kugawanya maandishi katika tokeni, au maneno. Ingawa hili linaweza kuonekana rahisi, kuzingatia alama za uakifishaji na mipaka ya sentensi na maneno katika lugha tofauti kunaweza kuwa changamoto. Unaweza kulazimika kutumia mbinu mbalimbali kuamua mipaka.

![tokenization](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Kugawanya sentensi kutoka **Pride and Prejudice**. Picha na [Jen Looper](https://twitter.com/jenlooper)

### Uwakilishi wa maneno (Embeddings)

[Uwakilishi wa maneno](https://wikipedia.org/wiki/Word_embedding) ni njia ya kubadilisha data yako ya maandishi kuwa nambari. Uwakilishi hufanywa kwa njia ambayo maneno yenye maana sawa au maneno yanayotumika pamoja yanajikusanya pamoja.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Nawaheshimu sana mishipa yako, ni marafiki wangu wa zamani." - Uwakilishi wa maneno kwa sentensi kutoka **Pride and Prejudice**. Picha na [Jen Looper](https://twitter.com/jenlooper)

✅ Jaribu [chombo hiki cha kuvutia](https://projector.tensorflow.org/) ili kufanya majaribio na uwakilishi wa maneno. Kubonyeza neno moja kunaonyesha makundi ya maneno yanayofanana: 'toy' linajikusanya na 'disney', 'lego', 'playstation', na 'console'.

### Uchambuzi wa muundo na Tagi za Sehemu ya Hotuba (Parsing & Part-of-speech Tagging)

Kila neno lililogawanywa linaweza kutagiwa kama sehemu ya hotuba - nomino, kitenzi, au kivumishi. Sentensi `the quick red fox jumped over the lazy brown dog` inaweza kutagiwa kama fox = nomino, jumped = kitenzi.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Kuchambua sentensi kutoka **Pride and Prejudice**. Picha na [Jen Looper](https://twitter.com/jenlooper)

Uchambuzi wa muundo ni kutambua maneno yanayohusiana katika sentensi - kwa mfano `the quick red fox jumped` ni mfuatano wa kivumishi-nomino-kitenzi ambao ni tofauti na mfuatano wa `lazy brown dog`.

### Mara kwa mara ya Maneno na Misemo

Njia muhimu wakati wa kuchambua maandishi mengi ni kujenga kamusi ya kila neno au msemo wa kuvutia na jinsi unavyojitokeza mara kwa mara. Msemo `the quick red fox jumped over the lazy brown dog` una mara kwa mara ya neno 2 kwa the.

Hebu tuangalie mfano wa maandishi ambapo tunahesabu mara kwa mara ya maneno. Shairi la Rudyard Kipling The Winners lina aya ifuatayo:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Kwa kuwa mara kwa mara ya misemo inaweza kuwa bila kujali herufi kubwa au ndogo kama inavyohitajika, msemo `a friend` una mara kwa mara ya 2 na `the` una mara kwa mara ya 6, na `travels` ni 2.

### N-grams

Maandishi yanaweza kugawanywa katika mfuatano wa maneno ya urefu fulani, neno moja (unigram), maneno mawili (bigrams), maneno matatu (trigrams) au idadi yoyote ya maneno (n-grams).

Kwa mfano `the quick red fox jumped over the lazy brown dog` na alama ya n-gram ya 2 inazalisha n-grams zifuatazo:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Inaweza kuwa rahisi kuiona kama sanduku linalosonga juu ya sentensi. Hapa kuna mfano wa n-grams ya maneno 3, n-gram iko kwa herufi nzito katika kila sentensi:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams sliding window](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Thamani ya N-gram ya 3: Picha na [Jen Looper](https://twitter.com/jenlooper)

### Uchimbaji wa Misemo ya Nomino

Katika sentensi nyingi, kuna nomino ambayo ni somo au kitu cha sentensi. Katika Kiingereza, mara nyingi inaweza kutambulika kwa kuwa na 'a' au 'an' au 'the' kabla yake. Kutambua somo au kitu cha sentensi kwa 'kuchimba msemo wa nomino' ni kazi ya kawaida katika NLP wakati wa kujaribu kuelewa maana ya sentensi.

✅ Katika sentensi "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", unaweza kutambua misemo ya nomino?

Katika sentensi `the quick red fox jumped over the lazy brown dog` kuna misemo 2 ya nomino: **quick red fox** na **lazy brown dog**.

### Uchambuzi wa Hisia

Sentensi au maandishi yanaweza kuchambuliwa kwa hisia, au jinsi *chanya* au *hasi* ilivyo. Hisia hupimwa kwa *polarity* na *objectivity/subjectivity*. Polarity hupimwa kutoka -1.0 hadi 1.0 (hasi hadi chanya) na 0.0 hadi 1.0 (yawezekana zaidi hadi ya kibinafsi zaidi).

✅ Baadaye utajifunza kuwa kuna njia tofauti za kuamua hisia kwa kutumia ujifunzaji wa mashine, lakini njia moja ni kuwa na orodha ya maneno na misemo ambayo imeainishwa kama chanya au hasi na mtaalamu wa binadamu na kutumia mfano huo kwa maandishi ili kuhesabu alama ya polarity. Unaweza kuona jinsi hii ingeweza kufanya kazi katika hali fulani na si vizuri katika hali nyingine?

### Mabadiliko ya Maneno (Inflection)

Mabadiliko ya maneno yanakuwezesha kuchukua neno na kupata umoja au wingi wa neno hilo.

### Lemmatization

*Lemma* ni mzizi au neno la kichwa kwa seti ya maneno, kwa mfano *flew*, *flies*, *flying* yana lemma ya kitenzi *fly*.

Pia kuna hifadhidata muhimu zinazopatikana kwa mtafiti wa NLP, hasa:

### WordNet

[WordNet](https://wordnet.princeton.edu/) ni hifadhidata ya maneno, visawe, maneno kinyume na maelezo mengine mengi kwa kila neno katika lugha nyingi tofauti. Ni muhimu sana wakati wa kujaribu kujenga tafsiri, ukaguzi wa tahajia, au zana za lugha za aina yoyote.

## Maktaba za NLP

Kwa bahati nzuri, huhitaji kujenga mbinu hizi zote mwenyewe, kwani kuna maktaba bora za Python zinazopatikana ambazo zinawafanya waendelezaji wasio maalum katika usindikaji wa lugha asilia au ujifunzaji wa mashine kufikia kwa urahisi. Masomo yanayofuata yanajumuisha mifano zaidi ya hizi, lakini hapa utajifunza mifano muhimu kukusaidia na kazi inayofuata.

### Zoezi - kutumia maktaba ya `TextBlob`

Hebu tutumie maktaba inayoitwa TextBlob kwani ina API za kusaidia kushughulikia aina hizi za kazi. TextBlob "inasimama juu ya mabega makubwa ya [NLTK](https://nltk.org) na [pattern](https://github.com/clips/pattern), na inafanya kazi vizuri na zote mbili." Ina kiasi kikubwa cha ML kilichojumuishwa katika API yake.

> Kumbuka: Mwongozo wa [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) unapatikana kwa TextBlob ambao unapendekezwa kwa waendelezaji wenye uzoefu wa Python.

Wakati wa kujaribu kutambua *misemo ya nomino*, TextBlob inatoa chaguo kadhaa za wachimbaji wa kupata misemo ya nomino.

1. Angalia `ConllExtractor`.

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

    > Nini kinaendelea hapa? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) ni "Kichimbaji cha misemo ya nomino kinachotumia uchambuzi wa vipande vilivyofundishwa na hifadhidata ya mafunzo ya ConLL-2000." ConLL-2000 inahusu Mkutano wa 2000 wa Kujifunza Lugha Asilia kwa Kompyuta. Kila mwaka mkutano huo ulifanya warsha ya kushughulikia tatizo gumu la NLP, na mwaka 2000 ilikuwa uchambuzi wa vipande vya nomino. Mfano ulifundishwa kwenye Wall Street Journal, na "sehemu 15-18 kama data ya mafunzo (tokeni 211727) na sehemu ya 20 kama data ya majaribio (tokeni 47377)". Unaweza kuangalia taratibu zilizotumika [hapa](https://www.clips.uantwerpen.be/conll2000/chunking/) na [matokeo](https://ifarm.nl/erikt/research/np-chunking.html).

### Changamoto - kuboresha bot yako kwa NLP

Katika somo lililopita ulijenga bot rahisi ya Maswali na Majibu. Sasa, utamfanya Marvin awe na huruma zaidi kwa kuchambua maoni yako kwa hisia na kuchapisha jibu linalolingana na hisia hizo. Pia utahitaji kutambua `noun_phrase` na kuuliza kuhusu hilo.

Hatua zako wakati wa kujenga bot bora ya mazungumzo:

1. Chapisha maelekezo yanayoshauri mtumiaji jinsi ya kuingiliana na bot
2. Anzisha mzunguko 
   1. Kubali maoni ya mtumiaji
   2. Ikiwa mtumiaji ameomba kuondoka, basi ondoka
   3. Chakata maoni ya mtumiaji na uamue jibu linalofaa la hisia
   4. Ikiwa msemo wa nomino umetambuliwa katika hisia, fanya wingi wake na uliza maoni zaidi kuhusu mada hiyo
   5. Chapisha jibu
3. rudi hatua ya 2

Hapa kuna kipande cha msimbo wa kuamua hisia kwa kutumia TextBlob. Kumbuka kuna *viwango* vinne tu vya jibu la hisia (unaweza kuwa na zaidi ikiwa unapenda):

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

Hapa kuna baadhi ya matokeo ya sampuli ya kuongoza (maoni ya mtumiaji yako kwenye mistari inayoanza na >):

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

Suluhisho moja linalowezekana kwa kazi hiyo ni [hapa](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Uhakiki wa Maarifa

1. Je, unadhani majibu yenye huruma yanaweza 'kumdanganya' mtu kufikiria kwamba bot inaelewa kweli?
2. Je, kutambua msemo wa nomino kunafanya bot iwe ya 'kuaminika' zaidi?
3. Kwa nini kuchimba 'msemo wa nomino' kutoka sentensi ni jambo muhimu kufanya?

---

Tekeleza bot katika uhakiki wa maarifa wa awali na ujaribu kwa rafiki. Je, inaweza kuwahadaa? Je, unaweza kufanya bot yako iwe ya 'kuaminika' zaidi?

## 🚀Changamoto

Chukua kazi katika uhakiki wa maarifa wa awali na jaribu kuitekeleza. Jaribu bot kwa rafiki. Je, inaweza kuwahadaa? Je, unaweza kufanya bot yako iwe ya 'kuaminika' zaidi?

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Katika masomo yanayofuata utajifunza zaidi kuhusu uchambuzi wa hisia. Tafiti mbinu hii ya kuvutia katika makala kama hizi kwenye [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Kazi 

[Fanya bot izungumze](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.