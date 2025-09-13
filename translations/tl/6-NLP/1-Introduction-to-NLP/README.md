<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T18:26:39+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "tl"
}
-->
# Panimula sa natural language processing

Ang araling ito ay tumatalakay sa maikling kasaysayan at mahahalagang konsepto ng *natural language processing*, isang subfield ng *computational linguistics*.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Panimula

Ang NLP, na karaniwang tawag dito, ay isa sa mga pinakakilalang larangan kung saan ang machine learning ay ginamit at inilapat sa production software.

✅ Maiisip mo ba ang software na ginagamit mo araw-araw na marahil ay may kasamang NLP? Paano ang mga word processing programs o mobile apps na regular mong ginagamit?

Matututuhan mo ang tungkol sa:

- **Ang ideya ng mga wika**. Paano nabuo ang mga wika at ano ang mga pangunahing larangan ng pag-aaral.
- **Mga depinisyon at konsepto**. Malalaman mo rin ang mga depinisyon at konsepto kung paano pinoproseso ng mga computer ang teksto, kabilang ang parsing, grammar, at pagtukoy sa mga pangngalan at pandiwa. May ilang coding tasks sa araling ito, at ilang mahahalagang konsepto ang ipakikilala na matututuhan mong i-code sa mga susunod na aralin.

## Computational linguistics

Ang computational linguistics ay isang larangan ng pananaliksik at pag-unlad sa loob ng maraming dekada na nag-aaral kung paano maaaring magtrabaho ang mga computer sa mga wika, at kahit na maunawaan, maisalin, at makipag-usap gamit ang mga wika. Ang natural language processing (NLP) ay isang kaugnay na larangan na nakatuon sa kung paano maaaring iproseso ng mga computer ang 'natural', o mga wika ng tao.

### Halimbawa - phone dictation

Kung minsan kang nagdikta sa iyong telepono sa halip na mag-type o nagtanong sa isang virtual assistant, ang iyong pagsasalita ay na-convert sa text form at pagkatapos ay pinroseso o *parsed* mula sa wikang iyong ginamit. Ang mga natukoy na keyword ay pagkatapos pinroseso sa isang format na maiintindihan at magagamit ng telepono o assistant.

![comprehension](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Ang tunay na linguistic comprehension ay mahirap! Larawan ni [Jen Looper](https://twitter.com/jenlooper)

### Paano nagiging posible ang teknolohiyang ito?

Nagiging posible ito dahil may isang tao na nagsulat ng computer program para gawin ito. Ilang dekada na ang nakalipas, hinulaan ng ilang science fiction writers na ang mga tao ay kadalasang magsasalita sa kanilang mga computer, at ang mga computer ay palaging maiintindihan nang eksakto ang kanilang ibig sabihin. Sa kasamaang-palad, lumabas na mas mahirap ang problemang ito kaysa sa inaakala ng marami, at bagama't mas nauunawaan na ito ngayon, may mga malalaking hamon pa rin sa pagkamit ng 'perpektong' natural language processing pagdating sa pag-unawa sa kahulugan ng isang pangungusap. Ito ay partikular na mahirap pagdating sa pag-unawa sa humor o pagtukoy sa emosyon tulad ng sarcasm sa isang pangungusap.

Sa puntong ito, maaaring naaalala mo ang mga klase sa paaralan kung saan tinatalakay ng guro ang mga bahagi ng grammar sa isang pangungusap. Sa ilang bansa, ang mga mag-aaral ay tinuturuan ng grammar at linguistics bilang isang dedikadong asignatura, ngunit sa marami, ang mga paksang ito ay kasama bilang bahagi ng pag-aaral ng isang wika: alinman sa iyong unang wika sa primary school (pag-aaral na magbasa at magsulat) at marahil isang pangalawang wika sa post-primary o high school. Huwag mag-alala kung hindi ka eksperto sa pagkakaiba ng mga pangngalan mula sa mga pandiwa o mga pang-abay mula sa mga pang-uri!

Kung nahihirapan ka sa pagkakaiba ng *simple present* at *present progressive*, hindi ka nag-iisa. Ito ay isang hamon para sa maraming tao, kahit na mga katutubong nagsasalita ng isang wika. Ang magandang balita ay ang mga computer ay talagang mahusay sa paglalapat ng mga pormal na tuntunin, at matututuhan mong magsulat ng code na maaaring *mag-parse* ng isang pangungusap na kasinghusay ng isang tao. Ang mas malaking hamon na susuriin mo sa kalaunan ay ang pag-unawa sa *kahulugan* at *sentiment* ng isang pangungusap.

## Mga Kinakailangan

Para sa araling ito, ang pangunahing kinakailangan ay ang kakayahang magbasa at maunawaan ang wika ng araling ito. Walang mga math problems o equations na kailangang lutasin. Bagama't isinulat ng orihinal na may-akda ang araling ito sa Ingles, ito ay isinalin din sa ibang mga wika, kaya maaaring binabasa mo ang isang salin. May mga halimbawa kung saan ginagamit ang iba't ibang wika (upang ihambing ang iba't ibang grammar rules ng iba't ibang wika). Ang mga ito ay *hindi* isinalin, ngunit ang paliwanag na teksto ay isinalin, kaya dapat malinaw ang kahulugan.

Para sa mga coding tasks, gagamit ka ng Python at ang mga halimbawa ay gumagamit ng Python 3.8.

Sa seksyong ito, kakailanganin mo, at gagamitin:

- **Pag-unawa sa Python 3**. Pag-unawa sa programming language sa Python 3, ang araling ito ay gumagamit ng input, loops, file reading, arrays.
- **Visual Studio Code + extension**. Gagamitin natin ang Visual Studio Code at ang Python extension nito. Maaari ka ring gumamit ng Python IDE na iyong pinili.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) ay isang pinasimpleng text processing library para sa Python. Sundin ang mga tagubilin sa TextBlob site upang i-install ito sa iyong sistema (i-install din ang corpora, tulad ng ipinakita sa ibaba):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: Maaari mong patakbuhin ang Python nang direkta sa mga VS Code environment. Tingnan ang [docs](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para sa karagdagang impormasyon.

## Pakikipag-usap sa mga makina

Ang kasaysayan ng pagsubok na gawing maunawaan ng mga computer ang wika ng tao ay bumalik sa mga dekada, at isa sa mga unang siyentipiko na isinasaalang-alang ang natural language processing ay si *Alan Turing*.

### Ang 'Turing test'

Noong nagsasaliksik si Turing tungkol sa *artificial intelligence* noong 1950's, isinasaalang-alang niya kung maaaring ibigay ang isang conversational test sa isang tao at computer (sa pamamagitan ng typed correspondence) kung saan ang tao sa pag-uusap ay hindi sigurado kung nakikipag-usap siya sa isa pang tao o sa isang computer.

Kung, pagkatapos ng isang tiyak na haba ng pag-uusap, hindi matukoy ng tao na ang mga sagot ay mula sa isang computer o hindi, maaari bang sabihin na ang computer ay *nag-iisip*?

### Ang inspirasyon - 'the imitation game'

Ang ideya para dito ay nagmula sa isang party game na tinatawag na *The Imitation Game* kung saan ang isang interrogator ay nag-iisa sa isang silid at may tungkuling tukuyin kung alin sa dalawang tao (sa ibang silid) ang lalaki at babae. Ang interrogator ay maaaring magpadala ng mga tanong, at kailangang mag-isip ng mga tanong kung saan ang mga sagot sa sulat ay magbubunyag ng kasarian ng misteryosong tao. Siyempre, ang mga manlalaro sa kabilang silid ay sinusubukang linlangin ang interrogator sa pamamagitan ng pagsagot sa mga tanong sa paraang nakakalito o nakakalito, habang nagbibigay din ng impresyon ng pagsagot nang tapat.

### Pagbuo ng Eliza

Noong 1960's, isang siyentipiko mula sa MIT na si *Joseph Weizenbaum* ang bumuo ng [*Eliza*](https://wikipedia.org/wiki/ELIZA), isang computer 'therapist' na nagtatanong sa tao at nagbibigay ng impresyon na nauunawaan ang kanilang mga sagot. Gayunpaman, bagama't kayang i-parse ni Eliza ang isang pangungusap at tukuyin ang ilang grammatical constructs at keywords upang makapagbigay ng makatwirang sagot, hindi masasabi na *naiintindihan* nito ang pangungusap. Kung si Eliza ay binigyan ng isang pangungusap na sumusunod sa format na "**Ako ay** <u>malungkot</u>", maaaring ayusin nito at palitan ang mga salita sa pangungusap upang mabuo ang sagot na "Gaano katagal ka nang **malungkot**?"

Ito ay nagbibigay ng impresyon na naiintindihan ni Eliza ang pahayag at nagtatanong ng follow-up na tanong, samantalang sa katotohanan, binabago lamang nito ang tense at nagdaragdag ng ilang salita. Kung si Eliza ay hindi makakilala ng isang keyword na mayroon itong sagot para sa, magbibigay ito ng random na sagot na maaaring magamit sa maraming iba't ibang pahayag. Madaling maloko si Eliza, halimbawa kung ang isang user ay sumulat "**Ikaw ay** isang <u>bisikleta</u>", maaaring tumugon ito ng "Gaano katagal ako naging **bisikleta**?", sa halip na isang mas makatwirang sagot.

[![Chatting with Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 I-click ang larawan sa itaas para sa isang video tungkol sa orihinal na ELIZA program

> Note: Maaari mong basahin ang orihinal na paglalarawan ng [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) na inilathala noong 1966 kung mayroon kang ACM account. Bilang alternatibo, basahin ang tungkol kay Eliza sa [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Ehersisyo - pag-code ng isang simpleng conversational bot

Ang isang conversational bot, tulad ni Eliza, ay isang program na humihikayat ng input mula sa user at tila nauunawaan at tumutugon nang matalino. Hindi tulad ni Eliza, ang ating bot ay walang maraming rules na nagbibigay ng impresyon ng isang intelligent na pag-uusap. Sa halip, ang ating bot ay magkakaroon lamang ng isang kakayahan, ang panatilihin ang pag-uusap gamit ang random na mga sagot na maaaring gumana sa halos anumang simpleng pag-uusap.

### Ang plano

Ang iyong mga hakbang sa paggawa ng conversational bot:

1. Mag-print ng mga tagubilin na nagpapayo sa user kung paano makipag-ugnayan sa bot
2. Simulan ang isang loop
   1. Tanggapin ang input ng user
   2. Kung hiniling ng user na mag-exit, mag-exit
   3. Iproseso ang input ng user at tukuyin ang sagot (sa kasong ito, ang sagot ay isang random na pagpipilian mula sa listahan ng mga posibleng generic na sagot)
   4. I-print ang sagot
3. Bumalik sa hakbang 2

### Pagbuo ng bot

Gawin natin ang bot. Magsisimula tayo sa pagde-define ng ilang phrases.

1. Gumawa ng bot sa Python gamit ang mga sumusunod na random na sagot:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Narito ang ilang sample output bilang gabay (ang input ng user ay nasa mga linya na nagsisimula sa `>`):

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

    Ang isang posibleng solusyon sa task ay [dito](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Huminto at pag-isipan

    1. Sa tingin mo ba ang random na mga sagot ay 'malilinlang' ang isang tao na isipin na ang bot ay talagang naiintindihan sila?
    2. Anong mga tampok ang kakailanganin ng bot upang maging mas epektibo?
    3. Kung ang isang bot ay talagang 'naiintindihan' ang kahulugan ng isang pangungusap, kakailanganin ba nitong 'alalahanin' ang kahulugan ng mga nakaraang pangungusap sa isang pag-uusap?

---

## 🚀Hamunin

Pumili ng isa sa mga "huminto at pag-isipan" na elemento sa itaas at subukang i-implement ito sa code o magsulat ng solusyon sa papel gamit ang pseudocode.

Sa susunod na aralin, matututuhan mo ang tungkol sa iba't ibang mga diskarte sa pag-parse ng natural language at machine learning.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Tingnan ang mga sanggunian sa ibaba bilang karagdagang pagkakataon sa pagbabasa.

### Mga Sanggunian

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Takdang Aralin 

[Maghanap ng bot](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.