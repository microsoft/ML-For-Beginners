<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T13:55:03+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "sl"
}
-->
# Pogoste naloge in tehnike obdelave naravnega jezika

Pri večini nalog obdelave *naravnega jezika* je treba besedilo razčleniti, analizirati in rezultate shraniti ali primerjati s pravili in podatkovnimi nabori. Te naloge omogočajo programerju, da iz besedila izpelje _pomen_, _namen_ ali zgolj _pogostost_ izrazov in besed.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

Odkrijmo pogoste tehnike, ki se uporabljajo pri obdelavi besedila. V kombinaciji s strojnim učenjem te tehnike omogočajo učinkovito analizo velikih količin besedila. Preden uporabimo strojno učenje za te naloge, pa moramo razumeti težave, s katerimi se srečuje specialist za obdelavo naravnega jezika.

## Pogoste naloge pri obdelavi naravnega jezika

Obstajajo različni načini za analizo besedila, s katerim delate. Obstajajo naloge, ki jih lahko izvedete, in prek teh nalog lahko pridobite razumevanje besedila ter izpeljete zaključke. Te naloge običajno izvajate v zaporedju.

### Tokenizacija

Prva stvar, ki jo mora večina algoritmov za obdelavo naravnega jezika narediti, je razdelitev besedila na tokene ali besede. Čeprav se to sliši preprosto, lahko upoštevanje ločil in različnih jezikovnih mej med besedami in stavki postane zapleteno. Morda boste morali uporabiti različne metode za določanje teh mej.

![tokenizacija](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizacija stavka iz **Prevzetnosti in pristranosti**. Infografika avtorice [Jen Looper](https://twitter.com/jenlooper)

### Vdelave

[Vdelave besed](https://wikipedia.org/wiki/Word_embedding) so način, kako besedilne podatke pretvoriti v številčno obliko. Vdelave so narejene tako, da se besede s podobnim pomenom ali besede, ki se pogosto uporabljajo skupaj, združijo v skupine.

![vdelave besed](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Imam največje spoštovanje do vaših živcev, so moji stari prijatelji." - Vdelave besed za stavek iz **Prevzetnosti in pristranosti**. Infografika avtorice [Jen Looper](https://twitter.com/jenlooper)

✅ Preizkusite [to zanimivo orodje](https://projector.tensorflow.org/) za eksperimentiranje z vdelavami besed. Klik na eno besedo pokaže skupine podobnih besed: 'igrača' se združi z 'disney', 'lego', 'playstation' in 'konzola'.

### Razčlenjevanje in označevanje delov govora

Vsako besedo, ki je bila tokenizirana, je mogoče označiti kot del govora - samostalnik, glagol ali pridevnik. Stavek `hitra rdeča lisica je skočila čez lenega rjavega psa` bi lahko bil označen kot lisica = samostalnik, skočila = glagol.

![razčlenjevanje](../../../../6-NLP/2-Tasks/images/parse.png)

> Razčlenjevanje stavka iz **Prevzetnosti in pristranosti**. Infografika avtorice [Jen Looper](https://twitter.com/jenlooper)

Razčlenjevanje pomeni prepoznavanje, katere besede so med seboj povezane v stavku - na primer `hitra rdeča lisica je skočila` je zaporedje pridevnik-samostalnik-glagol, ki je ločeno od zaporedja `lenega rjavega psa`.

### Pogostost besed in fraz

Koristen postopek pri analizi velikega besedila je izdelava slovarja vseh besed ali fraz, ki nas zanimajo, in kako pogosto se pojavljajo. Fraza `hitra rdeča lisica je skočila čez lenega rjavega psa` ima pogostost besede "je" 2.

Poglejmo primer besedila, kjer štejemo pogostost besed. Pesem Zmagovalci avtorja Rudyard Kiplinga vsebuje naslednjo kitico:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Ker je pogostost fraz lahko občutljiva na velike in male črke ali ne, ima fraza `prijatelj` pogostost 2, `je` pogostost 6, in `potuje` pogostost 2.

### N-grami

Besedilo je mogoče razdeliti na zaporedja besed določene dolžine, eno besedo (unigram), dve besedi (bigram), tri besede (trigram) ali poljubno število besed (n-grami).

Na primer `hitra rdeča lisica je skočila čez lenega rjavega psa` z n-gramom dolžine 2 ustvari naslednje n-grame:

1. hitra rdeča  
2. rdeča lisica  
3. lisica je  
4. je skočila  
5. skočila čez  
6. čez lenega  
7. lenega rjavega  
8. rjavega psa  

Lahko si to predstavljate kot drsno polje nad stavkom. Tukaj je prikazano za n-grame dolžine 3 besed, n-gram je poudarjen v vsakem stavku:

1.   <u>**hitra rdeča lisica**</u> je skočila čez lenega rjavega psa  
2.   hitra **<u>rdeča lisica je</u>** skočila čez lenega rjavega psa  
3.   hitra rdeča **<u>lisica je skočila</u>** čez lenega rjavega psa  
4.   hitra rdeča lisica **<u>je skočila čez</u>** lenega rjavega psa  
5.   hitra rdeča lisica je **<u>skočila čez lenega</u>** rjavega psa  
6.   hitra rdeča lisica je skočila **<u>čez lenega rjavega</u>** psa  
7.   hitra rdeča lisica je skočila čez <u>**lenega rjavega psa**</u>  

![drsno okno n-gramov](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram dolžine 3: Infografika avtorice [Jen Looper](https://twitter.com/jenlooper)

### Izvleček samostalniških fraz

V večini stavkov je samostalnik, ki je predmet ali objekt stavka. V angleščini ga pogosto prepoznamo po besedah 'a', 'an' ali 'the', ki mu sledijo. Prepoznavanje predmeta ali objekta stavka z 'izvlekom samostalniške fraze' je pogosta naloga pri obdelavi naravnega jezika, ko poskušamo razumeti pomen stavka.

✅ V stavku "Ne morem določiti ure, kraja, pogleda ali besed, ki so postavile temelje. To je bilo predolgo nazaj. Bil sem na sredini, preden sem vedel, da sem začel." Ali lahko prepoznate samostalniške fraze?

V stavku `hitra rdeča lisica je skočila čez lenega rjavega psa` sta 2 samostalniški frazi: **hitra rdeča lisica** in **lenega rjavega psa**.

### Analiza sentimenta

Stavek ali besedilo je mogoče analizirati glede sentimenta, ali je *pozitivno* ali *negativno*. Sentiment se meri v *polariteti* in *objektivnosti/subjektivnosti*. Polariteta se meri od -1.0 do 1.0 (negativno do pozitivno) in od 0.0 do 1.0 (najbolj objektivno do najbolj subjektivno).

✅ Kasneje boste izvedeli, da obstajajo različni načini za določanje sentimenta s strojnim učenjem, vendar je eden od načinov, da imamo seznam besed in fraz, ki jih človeški strokovnjak kategorizira kot pozitivne ali negativne, ter ta model uporabimo na besedilu za izračun polaritetne ocene. Ali vidite, kako bi to delovalo v nekaterih okoliščinah in manj dobro v drugih?

### Pregibanje

Pregibanje omogoča, da vzamete besedo in pridobite njen edninsko ali množinsko obliko.

### Lemmatizacija

*Lema* je koren ali osnovna beseda za niz besed, na primer *letel*, *leti*, *letenje* imajo lemo glagola *leteti*.

Na voljo so tudi uporabne baze podatkov za raziskovalce obdelave naravnega jezika, med njimi:

### WordNet

[WordNet](https://wordnet.princeton.edu/) je baza podatkov besed, sinonimov, antonimov in mnogih drugih podrobnosti za vsako besedo v različnih jezikih. Izjemno je uporabna pri poskusih gradnje prevodov, črkovalnikov ali jezikovnih orodij kakršne koli vrste.

## Knjižnice za obdelavo naravnega jezika

Na srečo vam ni treba sami razvijati vseh teh tehnik, saj so na voljo odlične knjižnice za Python, ki obdelavo naravnega jezika in strojno učenje naredijo veliko bolj dostopno za razvijalce, ki niso specializirani za to področje. V naslednjih lekcijah boste spoznali več primerov teh knjižnic, tukaj pa boste izvedeli nekaj uporabnih primerov, ki vam bodo pomagali pri naslednji nalogi.

### Vaja - uporaba knjižnice `TextBlob`

Uporabimo knjižnico TextBlob, saj vsebuje uporabne API-je za reševanje teh vrst nalog. TextBlob "stoji na ramenih velikanov [NLTK](https://nltk.org) in [pattern](https://github.com/clips/pattern) ter se lepo povezuje z obema." V svojem API-ju ima vgrajeno veliko strojnega učenja.

> Opomba: Priporočamo [hitri začetek](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) za TextBlob, ki je namenjen izkušenim Python razvijalcem.

Pri poskusu prepoznavanja *samostalniških fraz* TextBlob ponuja več možnosti izvlečkov za iskanje samostalniških fraz.

1. Oglejte si `ConllExtractor`.

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

    > Kaj se tukaj dogaja? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) je "Izvleček samostalniških fraz, ki uporablja razčlenjevanje kosov, trenirano na korpusu ConLL-2000." ConLL-2000 se nanaša na konferenco o računalniškem učenju naravnega jezika leta 2000. Vsako leto je konferenca gostila delavnico za reševanje težav obdelave naravnega jezika, leta 2000 pa je bila tema razčlenjevanje samostalniških fraz. Model je bil treniran na Wall Street Journalu, z "oddelki 15-18 kot podatki za treniranje (211727 tokenov) in oddelkom 20 kot testnimi podatki (47377 tokenov)". Postopke, ki so bili uporabljeni, si lahko ogledate [tukaj](https://www.clips.uantwerpen.be/conll2000/chunking/) in [rezultate](https://ifarm.nl/erikt/research/np-chunking.html).

### Izziv - izboljšanje vašega bota z obdelavo naravnega jezika

V prejšnji lekciji ste ustvarili zelo preprost Q&A bot. Zdaj boste Marvina naredili nekoliko bolj sočutnega, tako da boste analizirali vaš vnos glede sentimenta in natisnili odgovor, ki ustreza sentimentu. Prav tako boste morali prepoznati `samostalniško frazo` in o njej povprašati.

Koraki pri gradnji boljšega pogovornega bota:

1. Natisnite navodila, ki uporabnika obveščajo, kako komunicirati z botom  
2. Začnite zanko  
   1. Sprejmite uporabnikov vnos  
   2. Če uporabnik zahteva izhod, izstopite  
   3. Obdelajte uporabnikov vnos in določite ustrezen odgovor glede na sentiment  
   4. Če je v sentimentu zaznana samostalniška fraza, jo postavite v množinsko obliko in povprašajte za več informacij o tej temi  
   5. Natisnite odgovor  
3. Vrnite se na korak 2  

Tukaj je del kode za določanje sentimenta z uporabo TextBlob. Upoštevajte, da obstajajo samo štiri *stopnje* odziva na sentiment (lahko jih dodate več, če želite):

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

Tukaj je nekaj primerov izhoda, ki vas lahko vodi (uporabnikov vnos je na vrsticah, ki se začnejo z >):

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

Ena od možnih rešitev naloge je [tukaj](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Preverjanje znanja

1. Ali menite, da bi sočutni odgovori 'pretentali' nekoga, da bi mislil, da bot dejansko razume?  
2. Ali prepoznavanje samostalniške fraze naredi bota bolj 'prepričljivega'?  
3. Zakaj bi bilo izvlečenje 'samostalniške fraze' iz stavka koristno?  

---

Implementirajte bota iz prejšnjega preverjanja znanja in ga preizkusite na prijatelju. Ali ga lahko pretenta? Ali lahko naredite svojega bota bolj 'prepričljivega'?

## 🚀Izziv

Izvedite nalogo iz prejšnjega preverjanja znanja in jo poskusite implementirati. Preizkusite bota na prijatelju. Ali ga lahko pretenta? Ali lahko naredite svojega bota bolj 'prepričljivega'?

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

V naslednjih lekcijah boste izvedeli več o analizi sentimenta. Raziskujte to zanimivo tehniko v člankih, kot so ti na [KDNuggets](https://www.kdnuggets.com/tag/nlp).

## Naloga 

[Naredite, da bot odgovarja](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas opozarjamo, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.