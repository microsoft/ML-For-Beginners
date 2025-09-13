<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T13:54:13+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "hr"
}
-->
# Uobičajeni zadaci i tehnike obrade prirodnog jezika

Za većinu zadataka obrade prirodnog jezika (*natural language processing*), tekst koji se obrađuje mora se razložiti, analizirati i rezultati pohraniti ili usporediti s pravilima i skupovima podataka. Ovi zadaci omogućuju programeru da izvuče _značenje_, _namjeru_ ili samo _učestalost_ pojmova i riječi u tekstu.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

Otkrijmo uobičajene tehnike koje se koriste u obradi teksta. U kombinaciji sa strojnim učenjem, ove tehnike pomažu vam da učinkovito analizirate velike količine teksta. Međutim, prije nego što primijenimo strojno učenje na ove zadatke, prvo razumijmo probleme s kojima se susreće stručnjak za obradu prirodnog jezika.

## Uobičajeni zadaci u obradi prirodnog jezika

Postoje različiti načini za analizu teksta na kojem radite. Postoje zadaci koje možete izvršiti, a kroz njih možete steći razumijevanje teksta i donijeti zaključke. Obično se ovi zadaci izvode u nizu.

### Tokenizacija

Vjerojatno prva stvar koju većina algoritama za obradu prirodnog jezika mora učiniti jest podijeliti tekst na tokene ili riječi. Iako to zvuči jednostavno, uzimanje u obzir interpunkcije i različitih jezičnih granica riječi i rečenica može biti izazovno. Možda ćete morati koristiti različite metode za određivanje granica.

![tokenizacija](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizacija rečenice iz **Ponosa i predrasuda**. Infografika autorice [Jen Looper](https://twitter.com/jenlooper)

### Ugrađivanja

[Ugrađivanja riječi](https://wikipedia.org/wiki/Word_embedding) su način pretvaranja vaših tekstualnih podataka u numerički oblik. Ugrađivanja se rade na način da riječi sa sličnim značenjem ili riječi koje se često koriste zajedno budu grupirane.

![ugrađivanja riječi](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Imam najveće poštovanje prema vašim živcima, oni su moji stari prijatelji." - Ugrađivanja riječi za rečenicu iz **Ponosa i predrasuda**. Infografika autorice [Jen Looper](https://twitter.com/jenlooper)

✅ Isprobajte [ovaj zanimljiv alat](https://projector.tensorflow.org/) za eksperimentiranje s ugrađivanjima riječi. Klikom na jednu riječ prikazuju se skupine sličnih riječi: 'toy' je grupiran s 'disney', 'lego', 'playstation' i 'console'.

### Parsiranje i označavanje dijelova govora

Svaka riječ koja je tokenizirana može se označiti kao dio govora - imenica, glagol ili pridjev. Rečenica `brza crvena lisica skočila je preko lijenog smeđeg psa` mogla bi biti označena kao lisica = imenica, skočila = glagol.

![parsiranje](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsiranje rečenice iz **Ponosa i predrasuda**. Infografika autorice [Jen Looper](https://twitter.com/jenlooper)

Parsiranje prepoznaje koje su riječi povezane u rečenici - na primjer, `brza crvena lisica skočila` je slijed pridjev-imenica-glagol koji je odvojen od slijeda `lijeni smeđi pas`.

### Učestalost riječi i fraza

Korisna procedura pri analizi velikog teksta je izgradnja rječnika svake riječi ili fraze od interesa i koliko se često pojavljuje. Fraza `brza crvena lisica skočila je preko lijenog smeđeg psa` ima učestalost riječi 2 za riječ "the".

Pogledajmo primjer teksta u kojem brojimo učestalost riječi. Pjesma The Winners autora Rudyarda Kiplinga sadrži sljedeće stihove:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Kako učestalost fraza može biti osjetljiva ili neosjetljiva na velika i mala slova, fraza `a friend` ima učestalost 2, `the` ima učestalost 6, a `travels` ima učestalost 2.

### N-grami

Tekst se može podijeliti na sekvence riječi određene duljine: jedna riječ (unigram), dvije riječi (bigrami), tri riječi (trigrami) ili bilo koji broj riječi (n-grami).

Na primjer, `brza crvena lisica skočila je preko lijenog smeđeg psa` s n-gramom duljine 2 daje sljedeće n-grame:

1. brza crvena  
2. crvena lisica  
3. lisica skočila  
4. skočila preko  
5. preko lijenog  
6. lijenog smeđeg  
7. smeđeg psa  

Možda će biti lakše vizualizirati to kao klizni okvir preko rečenice. Evo kako to izgleda za n-grame od 3 riječi, gdje je n-gram podebljan u svakoj rečenici:

1.   <u>**brza crvena lisica**</u> skočila je preko lijenog smeđeg psa  
2.   brza **<u>crvena lisica skočila</u>** je preko lijenog smeđeg psa  
3.   brza crvena **<u>lisica skočila preko</u>** lijenog smeđeg psa  
4.   brza crvena lisica **<u>skočila preko lijenog</u>** smeđeg psa  
5.   brza crvena lisica skočila **<u>preko lijenog smeđeg</u>** psa  
6.   brza crvena lisica skočila preko <u>**lijenog smeđeg psa**</u>  

![klizni prozor n-grama](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Vrijednost n-grama 3: Infografika autorice [Jen Looper](https://twitter.com/jenlooper)

### Ekstrakcija imenskih fraza

U većini rečenica postoji imenica koja je subjekt ili objekt rečenice. U engleskom jeziku često se može prepoznati po tome što joj prethodi 'a', 'an' ili 'the'. Identificiranje subjekta ili objekta rečenice pomoću 'ekstrakcije imenske fraze' uobičajen je zadatak u obradi prirodnog jezika kada se pokušava razumjeti značenje rečenice.

✅ U rečenici "Ne mogu odrediti sat, ni mjesto, ni pogled ni riječi, koje su postavile temelje. To je bilo davno. Bio sam usred toga prije nego što sam shvatio da sam počeo.", možete li identificirati imenske fraze?

U rečenici `brza crvena lisica skočila je preko lijenog smeđeg psa` postoje 2 imenske fraze: **brza crvena lisica** i **lijeni smeđi pas**.

### Analiza sentimenta

Rečenica ili tekst mogu se analizirati prema sentimentu, odnosno koliko su *pozitivni* ili *negativni*. Sentiment se mjeri prema *polaritetu* i *objektivnosti/subjektivnosti*. Polaritet se mjeri od -1.0 do 1.0 (negativno do pozitivno), a objektivnost od 0.0 do 1.0 (najobjektivnije do najsubjektivnije).

✅ Kasnije ćete naučiti da postoje različiti načini za određivanje sentimenta pomoću strojnog učenja, ali jedan od načina je imati popis riječi i fraza koje su kategorizirane kao pozitivne ili negativne od strane ljudskog stručnjaka i primijeniti taj model na tekst kako biste izračunali polaritet. Možete li vidjeti kako bi ovo funkcioniralo u nekim okolnostima, a u drugima ne?

### Infleksija

Infleksija vam omogućuje da uzmete riječ i dobijete njezin jedninski ili množinski oblik.

### Lemmatizacija

*Lema* je korijen ili osnovna riječ za skup riječi, na primjer *letio*, *leti*, *leteći* imaju lemu glagola *letjeti*.

Također postoje korisne baze podataka dostupne istraživačima obrade prirodnog jezika, a posebno:

### WordNet

[WordNet](https://wordnet.princeton.edu/) je baza podataka riječi, sinonima, antonima i mnogih drugih detalja za svaku riječ na mnogim različitim jezicima. Nevjerojatno je korisna pri pokušaju izrade prijevoda, provjere pravopisa ili alata za jezik bilo koje vrste.

## Biblioteke za obradu prirodnog jezika

Srećom, ne morate sami graditi sve ove tehnike jer postoje izvrsne Python biblioteke koje ih čine mnogo pristupačnijima za programere koji nisu specijalizirani za obradu prirodnog jezika ili strojno učenje. Sljedeće lekcije uključuju više primjera ovih biblioteka, ali ovdje ćete naučiti nekoliko korisnih primjera koji će vam pomoći u sljedećem zadatku.

### Vježba - korištenje biblioteke `TextBlob`

Koristimo biblioteku pod nazivom TextBlob jer sadrži korisne API-je za rješavanje ovakvih zadataka. TextBlob "stoji na ramenima divova [NLTK](https://nltk.org) i [pattern](https://github.com/clips/pattern), i lijepo surađuje s oboje." Ima značajnu količinu strojnog učenja ugrađenu u svoj API.

> Napomena: Koristan [Vodič za brzi početak](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) dostupan je za TextBlob i preporučuje se iskusnim Python programerima.

Kada pokušavate identificirati *imenske fraze*, TextBlob nudi nekoliko opcija ekstraktora za pronalaženje imenskih fraza.

1. Pogledajte `ConllExtractor`.

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

    > Što se ovdje događa? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) je "Ekstraktor imenskih fraza koji koristi parsiranje segmenata trenirano na ConLL-2000 korpusu za treniranje." ConLL-2000 odnosi se na Konferenciju o računalnom učenju jezika iz 2000. godine. Svake godine konferencija je organizirala radionicu za rješavanje složenog problema obrade prirodnog jezika, a 2000. godine to je bilo segmentiranje imenskih fraza. Model je treniran na Wall Street Journalu, s "odjeljcima 15-18 kao podacima za treniranje (211727 tokena) i odjeljkom 20 kao testnim podacima (47377 tokena)". Možete pogledati postupke korištene [ovdje](https://www.clips.uantwerpen.be/conll2000/chunking/) i [rezultate](https://ifarm.nl/erikt/research/np-chunking.html).

### Izazov - poboljšajte svog bota pomoću obrade prirodnog jezika

U prethodnoj lekciji izradili ste vrlo jednostavnog Q&A bota. Sada ćete učiniti Marvina malo suosjećajnijim analizirajući vaš unos za sentiment i ispisujući odgovor koji odgovara sentimentu. Također ćete morati identificirati `imensku frazu` i postaviti pitanje o njoj.

Vaši koraci pri izradi boljeg konverzacijskog bota:

1. Ispišite upute koje savjetuju korisnika kako komunicirati s botom  
2. Pokrenite petlju  
   1. Prihvatite korisnikov unos  
   2. Ako korisnik zatraži izlaz, izađite  
   3. Obradite korisnikov unos i odredite odgovarajući odgovor na temelju sentimenta  
   4. Ako se u sentimentu otkrije imenska fraza, množite je i zatražite dodatni unos o toj temi  
   5. Ispišite odgovor  
3. Vratite se na korak 2  

Evo isječka koda za određivanje sentimenta pomoću TextBlob-a. Imajte na umu da postoje samo četiri *gradijenta* odgovora na sentiment (možete ih imati više ako želite):

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

Evo primjera izlaza koji će vas voditi (korisnikov unos je na linijama koje počinju s >):

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

Jedno moguće rješenje zadatka nalazi se [ovdje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Provjera znanja

1. Mislite li da bi suosjećajni odgovori mogli 'zavarati' nekoga da pomisli da bot zapravo razumije?  
2. Čini li identificiranje imenske fraze bota 'uvjerljivijim'?  
3. Zašto bi ekstrakcija 'imenske fraze' iz rečenice bila korisna stvar?  

---

Implementirajte bota iz prethodne provjere znanja i testirajte ga na prijatelju. Može li ih zavarati? Možete li učiniti svog bota 'uvjerljivijim'?

## 🚀Izazov

Uzmite zadatak iz prethodne provjere znanja i pokušajte ga implementirati. Testirajte bota na prijatelju. Može li ih zavarati? Možete li učiniti svog bota 'uvjerljivijim'?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

U sljedećih nekoliko lekcija naučit ćete više o analizi sentimenta. Istražite ovu zanimljivu tehniku u člancima poput ovih na [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Zadatak 

[Natjerajte bota da odgovara](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden koristeći AI uslugu za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.