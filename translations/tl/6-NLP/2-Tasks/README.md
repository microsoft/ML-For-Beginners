<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T18:24:46+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "tl"
}
-->
# Karaniwang mga gawain at teknika sa natural language processing

Para sa karamihan ng mga gawain sa *natural language processing*, ang teksto na kailangang iproseso ay kailangang hatiin, suriin, at ang mga resulta ay itabi o i-cross reference gamit ang mga patakaran at data set. Ang mga gawaing ito ay nagbibigay-daan sa programmer na matukoy ang _kahulugan_, _layunin_, o kahit ang _dalas_ ng mga termino at salita sa isang teksto.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Tuklasin natin ang mga karaniwang teknika na ginagamit sa pagproseso ng teksto. Kapag pinagsama sa machine learning, ang mga teknika na ito ay tumutulong sa iyo na suriin ang malaking dami ng teksto nang mas epektibo. Bago gamitin ang ML sa mga gawaing ito, gayunpaman, unawain muna natin ang mga problemang kinakaharap ng isang NLP specialist.

## Mga karaniwang gawain sa NLP

May iba't ibang paraan upang suriin ang teksto na iyong pinagtatrabahuhan. May mga gawain na maaari mong isagawa, at sa pamamagitan ng mga gawaing ito, makakakuha ka ng mas malalim na pag-unawa sa teksto at makakagawa ng mga konklusyon. Karaniwan, isinasagawa mo ang mga gawaing ito nang sunud-sunod.

### Tokenization

Marahil ang unang bagay na kailangang gawin ng karamihan sa mga NLP algorithm ay hatiin ang teksto sa mga token, o mga salita. Bagama't mukhang simple ito, ang pagsasaalang-alang sa mga bantas at iba't ibang delimiters ng salita at pangungusap sa iba't ibang wika ay maaaring maging mahirap. Maaaring kailanganin mong gumamit ng iba't ibang mga pamamaraan upang matukoy ang mga hangganan.

![tokenization](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizing ng isang pangungusap mula sa **Pride and Prejudice**. Infographic ni [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) ay isang paraan upang gawing numerikal ang iyong data ng teksto. Ginagawa ang embeddings sa paraang ang mga salitang may magkatulad na kahulugan o mga salitang madalas gamitin nang magkasama ay nagkakagrupo.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Word embeddings para sa isang pangungusap sa **Pride and Prejudice**. Infographic ni [Jen Looper](https://twitter.com/jenlooper)

✅ Subukan ang [kagiliw-giliw na tool na ito](https://projector.tensorflow.org/) upang mag-eksperimento sa word embeddings. Ang pag-click sa isang salita ay nagpapakita ng mga grupo ng magkatulad na salita: 'toy' ay nagkakagrupo sa 'disney', 'lego', 'playstation', at 'console'.

### Parsing & Part-of-speech Tagging

Ang bawat salita na na-tokenize ay maaaring i-tag bilang bahagi ng pananalita - isang pangngalan, pandiwa, o pang-uri. Ang pangungusap na `the quick red fox jumped over the lazy brown dog` ay maaaring POS tagged bilang fox = pangngalan, jumped = pandiwa.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing ng isang pangungusap mula sa **Pride and Prejudice**. Infographic ni [Jen Looper](https://twitter.com/jenlooper)

Ang parsing ay ang pagkilala kung aling mga salita ang magkakaugnay sa isang pangungusap - halimbawa, `the quick red fox jumped` ay isang adjective-noun-verb sequence na hiwalay sa `lazy brown dog` sequence.

### Dalas ng Salita at Parirala

Isang kapaki-pakinabang na pamamaraan kapag sinusuri ang malaking dami ng teksto ay ang paggawa ng isang diksyunaryo ng bawat salita o parirala ng interes at kung gaano kadalas ito lumalabas. Ang parirala na `the quick red fox jumped over the lazy brown dog` ay may word frequency na 2 para sa the.

Tingnan natin ang isang halimbawa ng teksto kung saan binibilang natin ang dalas ng mga salita. Ang tula ni Rudyard Kipling na The Winners ay naglalaman ng sumusunod na taludtod:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Dahil ang dalas ng parirala ay maaaring case insensitive o case sensitive depende sa pangangailangan, ang parirala na `a friend` ay may dalas na 2, `the` ay may dalas na 6, at `travels` ay 2.

### N-grams

Ang isang teksto ay maaaring hatiin sa mga sequence ng mga salita na may nakatakdang haba, isang salita (unigram), dalawang salita (bigrams), tatlong salita (trigrams), o anumang bilang ng mga salita (n-grams).

Halimbawa, ang `the quick red fox jumped over the lazy brown dog` na may n-gram score na 2 ay nagbubunga ng sumusunod na n-grams:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Mas madali itong ma-visualize bilang isang sliding box sa pangungusap. Narito ito para sa n-grams ng 3 salita, ang n-gram ay naka-bold sa bawat pangungusap:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams sliding window](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram value na 3: Infographic ni [Jen Looper](https://twitter.com/jenlooper)

### Pagkuha ng Noun Phrase

Sa karamihan ng mga pangungusap, mayroong pangngalan na siyang paksa o layunin ng pangungusap. Sa Ingles, madalas itong makikilala bilang may 'a', 'an', o 'the' sa unahan nito. Ang pagkilala sa paksa o layunin ng isang pangungusap sa pamamagitan ng 'pagkuha ng noun phrase' ay isang karaniwang gawain sa NLP kapag sinusubukang unawain ang kahulugan ng isang pangungusap.

✅ Sa pangungusap na "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", kaya mo bang tukuyin ang mga noun phrases?

Sa pangungusap na `the quick red fox jumped over the lazy brown dog` mayroong 2 noun phrases: **quick red fox** at **lazy brown dog**.

### Sentiment Analysis

Ang isang pangungusap o teksto ay maaaring suriin para sa sentiment, o kung gaano *positibo* o *negatibo* ito. Ang sentiment ay sinusukat sa *polarity* at *objectivity/subjectivity*. Ang polarity ay sinusukat mula -1.0 hanggang 1.0 (negatibo hanggang positibo) at 0.0 hanggang 1.0 (pinaka-objective hanggang pinaka-subjective).

✅ Sa susunod, matututunan mo na may iba't ibang paraan upang matukoy ang sentiment gamit ang machine learning, ngunit ang isang paraan ay ang pagkakaroon ng listahan ng mga salita at parirala na ikinategorya bilang positibo o negatibo ng isang human expert at ilapat ang modelong iyon sa teksto upang kalkulahin ang polarity score. Nakikita mo ba kung paano ito gumagana sa ilang mga sitwasyon at hindi gaano sa iba?

### Inflection

Ang inflection ay nagbibigay-daan sa iyo na kunin ang singular o plural na anyo ng isang salita.

### Lemmatization

Ang *lemma* ay ang ugat o pangunahing salita para sa isang set ng mga salita, halimbawa *flew*, *flies*, *flying* ay may lemma na pandiwa *fly*.

Mayroon ding mga kapaki-pakinabang na database na magagamit para sa mga mananaliksik ng NLP, partikular:

### WordNet

[WordNet](https://wordnet.princeton.edu/) ay isang database ng mga salita, kasingkahulugan, kasalungat, at maraming iba pang detalye para sa bawat salita sa iba't ibang wika. Ito ay lubos na kapaki-pakinabang kapag sinusubukang bumuo ng mga pagsasalin, spell checkers, o anumang uri ng mga tool sa wika.

## Mga Aklatan ng NLP

Sa kabutihang-palad, hindi mo kailangang bumuo ng lahat ng mga teknika na ito nang mag-isa, dahil mayroong mahusay na mga Python library na magagamit na ginagawang mas naa-access ang NLP sa mga developer na hindi dalubhasa sa natural language processing o machine learning. Ang mga susunod na aralin ay naglalaman ng mas maraming halimbawa ng mga ito, ngunit dito ay matututunan mo ang ilang kapaki-pakinabang na halimbawa upang matulungan ka sa susunod na gawain.

### Ehersisyo - paggamit ng `TextBlob` library

Gamitin natin ang isang library na tinatawag na TextBlob dahil naglalaman ito ng mga kapaki-pakinabang na API para sa pagharap sa ganitong uri ng mga gawain. Ang TextBlob "ay nakatayo sa malalaking balikat ng [NLTK](https://nltk.org) at [pattern](https://github.com/clips/pattern), at mahusay na nakikipaglaro sa pareho." Mayroon itong malaking dami ng ML na naka-embed sa API nito.

> Tandaan: Isang kapaki-pakinabang na [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) guide ay magagamit para sa TextBlob na inirerekomenda para sa mga may karanasan na Python developer.

Kapag sinusubukang tukuyin ang *noun phrases*, ang TextBlob ay nag-aalok ng ilang mga opsyon ng extractors upang makahanap ng noun phrases.

1. Tingnan ang `ConllExtractor`.

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

    > Ano ang nangyayari dito? Ang [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) ay "isang noun phrase extractor na gumagamit ng chunk parsing na sinanay gamit ang ConLL-2000 training corpus." Ang ConLL-2000 ay tumutukoy sa 2000 Conference on Computational Natural Language Learning. Bawat taon ang kumperensya ay nagho-host ng workshop upang harapin ang isang mahirap na problema sa NLP, at noong 2000 ito ay noun chunking. Ang isang modelo ay sinanay sa Wall Street Journal, gamit ang "sections 15-18 bilang training data (211727 tokens) at section 20 bilang test data (47377 tokens)". Maaari mong tingnan ang mga pamamaraan na ginamit [dito](https://www.clips.uantwerpen.be/conll2000/chunking/) at ang [mga resulta](https://ifarm.nl/erikt/research/np-chunking.html).

### Hamon - pagpapabuti ng iyong bot gamit ang NLP

Sa nakaraang aralin, gumawa ka ng isang napakasimpleng Q&A bot. Ngayon, gagawin mong mas simpatetiko si Marvin sa pamamagitan ng pagsusuri sa iyong input para sa sentiment at pag-print ng tugon na tumutugma sa sentiment. Kailangan mo ring tukuyin ang isang `noun_phrase` at magtanong tungkol dito.

Ang iyong mga hakbang sa paggawa ng mas mahusay na conversational bot:

1. Mag-print ng mga tagubilin na nagpapayo sa user kung paano makipag-ugnayan sa bot
2. Simulan ang loop 
   1. Tanggapin ang input ng user
   2. Kung ang user ay humiling na mag-exit, mag-exit
   3. Proseso ang input ng user at tukuyin ang naaangkop na sentiment response
   4. Kung may natukoy na noun phrase sa sentiment, gawing plural ito at magtanong ng karagdagang input tungkol sa paksa
   5. Mag-print ng tugon
3. Bumalik sa hakbang 2

Narito ang code snippet upang tukuyin ang sentiment gamit ang TextBlob. Tandaan na mayroong apat na *gradients* ng sentiment response (maaari kang magkaroon ng higit pa kung nais mo):

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

Narito ang ilang sample output upang gabayan ka (ang input ng user ay nasa mga linya na nagsisimula sa >):

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

Ang isang posibleng solusyon sa gawain ay [dito](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Knowledge Check

1. Sa tingin mo ba ang mga simpatetikong tugon ay maaaring 'lokohin' ang isang tao na isipin na talagang naiintindihan sila ng bot?
2. Ang pagkilala ba sa noun phrase ay nagpapaganda sa bot na maging 'kapani-paniwala'?
3. Bakit kapaki-pakinabang ang pagkuha ng 'noun phrase' mula sa isang pangungusap?

---

Ipatupad ang bot sa nakaraang knowledge check at subukan ito sa isang kaibigan. Kaya ba nitong lokohin sila? Kaya mo bang gawing mas 'kapani-paniwala' ang iyong bot?

## 🚀Hamon

Kunin ang isang gawain sa nakaraang knowledge check at subukang ipatupad ito. Subukan ang bot sa isang kaibigan. Kaya ba nitong lokohin sila? Kaya mo bang gawing mas 'kapani-paniwala' ang iyong bot?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Sa mga susunod na aralin, matututunan mo pa ang tungkol sa sentiment analysis. Mag-research sa kagiliw-giliw na teknik na ito sa mga artikulo tulad ng mga nasa [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Takdang Aralin 

[Pag-usapan ang bot](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.