<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T16:59:22+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "sw"
}
-->
# Utangulizi wa Usindikaji wa Lugha Asilia

Somo hili linahusu historia fupi na dhana muhimu za *usindikaji wa lugha asilia*, tawi la *isimu ya kompyuta*.

## [Jaribio la kabla ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Utangulizi

NLP, kama inavyojulikana kwa kawaida, ni mojawapo ya maeneo yanayojulikana zaidi ambapo ujifunzaji wa mashine umetumika na kutumika katika programu za uzalishaji.

✅ Je, unaweza kufikiria programu unayotumia kila siku ambayo labda ina NLP ndani yake? Vipi kuhusu programu zako za kuandika maneno au programu za simu unazotumia mara kwa mara?

Utajifunza kuhusu:

- **Wazo la lugha**. Jinsi lugha zilivyoendelea na maeneo makuu ya masomo yalivyokuwa.
- **Ufafanuzi na dhana**. Pia utajifunza ufafanuzi na dhana kuhusu jinsi kompyuta zinavyosindika maandishi, ikiwa ni pamoja na uchambuzi, sarufi, na kutambua nomino na vitenzi. Kuna baadhi ya kazi za kuandika programu katika somo hili, na dhana kadhaa muhimu zinatambulishwa ambazo utajifunza kuandika programu baadaye katika masomo yanayofuata.

## Isimu ya Kompyuta

Isimu ya kompyuta ni eneo la utafiti na maendeleo kwa miongo kadhaa ambalo linachunguza jinsi kompyuta zinavyoweza kufanya kazi na hata kuelewa, kutafsiri, na kuwasiliana na lugha. Usindikaji wa lugha asilia (NLP) ni uwanja unaohusiana unaolenga jinsi kompyuta zinavyoweza kusindika lugha 'asilia', au lugha za binadamu.

### Mfano - Dikteta ya simu

Ikiwa umewahi kuzungumza na simu yako badala ya kuandika au kuuliza msaidizi wa kidijitali swali, hotuba yako ilibadilishwa kuwa maandishi na kisha kusindika au *kuchambuliwa* kutoka lugha uliyosema. Maneno muhimu yaliyotambuliwa yalichakatwa kuwa muundo ambao simu au msaidizi angeweza kuelewa na kuchukua hatua.

![ufahamu](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Ufahamu halisi wa isimu ni mgumu! Picha na [Jen Looper](https://twitter.com/jenlooper)

### Teknolojia hii inawezekanaje?

Hii inawezekana kwa sababu mtu aliandika programu ya kompyuta kufanya hivyo. Miongo kadhaa iliyopita, baadhi ya waandishi wa hadithi za kisayansi walitabiri kwamba watu wangezungumza zaidi na kompyuta zao, na kompyuta zingeelewa kila wanachomaanisha. Kwa bahati mbaya, ilibainika kuwa ni tatizo gumu zaidi kuliko wengi walivyodhani, na ingawa ni tatizo linaloeleweka vyema leo, kuna changamoto kubwa za kufanikisha usindikaji wa lugha asilia 'mkamilifu' linapokuja suala la kuelewa maana ya sentensi. Hili ni tatizo gumu hasa linapokuja suala la kuelewa ucheshi au kugundua hisia kama kejeli katika sentensi.

Kwa wakati huu, unaweza kuwa unakumbuka masomo ya shule ambapo mwalimu alifundisha sehemu za sarufi katika sentensi. Katika baadhi ya nchi, wanafunzi hufundishwa sarufi na isimu kama somo maalum, lakini katika nyingi, mada hizi hujumuishwa kama sehemu ya kujifunza lugha: ama lugha yako ya kwanza shuleni (kujifunza kusoma na kuandika) na labda lugha ya pili baada ya shule ya msingi, au shule ya sekondari. Usijali ikiwa wewe si mtaalamu wa kutofautisha nomino na vitenzi au vielezi na vivumishi!

Ikiwa unapata ugumu wa kutofautisha kati ya *wakati uliopo rahisi* na *wakati uliopo unaoendelea*, hauko peke yako. Hili ni jambo gumu kwa watu wengi, hata wazungumzaji wa lugha ya asili. Habari njema ni kwamba kompyuta ni nzuri sana katika kutumia sheria rasmi, na utajifunza kuandika programu inayoweza *kuchambua* sentensi kama binadamu. Changamoto kubwa zaidi utakayochunguza baadaye ni kuelewa *maana*, na *hisia*, za sentensi.

## Mahitaji ya awali

Kwa somo hili, hitaji kuu ni uwezo wa kusoma na kuelewa lugha ya somo hili. Hakuna matatizo ya hesabu au milinganyo ya kutatua. Ingawa mwandishi wa awali aliandika somo hili kwa Kiingereza, pia limetafsiriwa katika lugha nyingine, kwa hivyo unaweza kuwa unasoma tafsiri. Kuna mifano ambapo lugha kadhaa tofauti zinatumika (kulinganisha sheria tofauti za sarufi za lugha tofauti). Hizi *hazitafsiriwi*, lakini maandishi ya maelezo yanatafsiriwa, kwa hivyo maana inapaswa kuwa wazi.

Kwa kazi za kuandika programu, utatumia Python na mifano inatumia Python 3.8.

Katika sehemu hii, utahitaji, na kutumia:

- **Uelewa wa Python 3**. Uelewa wa lugha ya programu ya Python 3, somo hili linatumia pembejeo, vigezo, usomaji wa faili, safu.
- **Visual Studio Code + kiendelezi**. Tutatumia Visual Studio Code na kiendelezi chake cha Python. Unaweza pia kutumia IDE ya Python unayopendelea.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) ni maktaba rahisi ya usindikaji wa maandishi kwa Python. Fuata maelekezo kwenye tovuti ya TextBlob ili kuisakinisha kwenye mfumo wako (sakinisha corpora pia, kama inavyoonyeshwa hapa chini):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Kidokezo: Unaweza kuendesha Python moja kwa moja katika mazingira ya VS Code. Angalia [docs](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) kwa maelezo zaidi.

## Kuzungumza na mashine

Historia ya kujaribu kufanya kompyuta kuelewa lugha ya binadamu inarudi miongo kadhaa, na mmoja wa wanasayansi wa mapema kuzingatia usindikaji wa lugha asilia alikuwa *Alan Turing*.

### Jaribio la 'Turing'

Wakati Turing alipokuwa akichunguza *akili bandia* katika miaka ya 1950, alifikiria ikiwa jaribio la mazungumzo linaweza kufanywa kwa binadamu na kompyuta (kupitia mawasiliano yaliyoandikwa) ambapo binadamu katika mazungumzo hakuwa na uhakika ikiwa alikuwa akizungumza na binadamu mwingine au kompyuta.

Ikiwa, baada ya muda fulani wa mazungumzo, binadamu hangeweza kubaini kwamba majibu yalikuwa kutoka kwa kompyuta au la, basi je, kompyuta ingeweza kusemekana kuwa *inawaza*?

### Msukumo - 'mchezo wa kuiga'

Wazo hili lilitokana na mchezo wa sherehe uitwao *Mchezo wa Kuiga* ambapo mhojiwa yuko peke yake katika chumba na anapewa jukumu la kubaini ni nani kati ya watu wawili (katika chumba kingine) ni mwanamume na mwanamke mtawalia. Mhojiwa anaweza kutuma maelezo, na lazima ajaribu kufikiria maswali ambapo majibu yaliyoandikwa yanafichua jinsia ya mtu wa fumbo. Bila shaka, wachezaji katika chumba kingine wanajaribu kumdanganya mhojiwa kwa kujibu maswali kwa njia ya kupotosha au kuchanganya mhojiwa, huku wakitoa mwonekano wa kujibu kwa uaminifu.

### Kuunda Eliza

Katika miaka ya 1960 mwanasayansi wa MIT aitwaye *Joseph Weizenbaum* alitengeneza [*Eliza*](https://wikipedia.org/wiki/ELIZA), 'daktari wa kompyuta' ambaye angeuliza maswali ya binadamu na kutoa mwonekano wa kuelewa majibu yao. Hata hivyo, ingawa Eliza angeweza kuchambua sentensi na kutambua miundo fulani ya sarufi na maneno muhimu ili kutoa jibu la kuridhisha, haingeweza kusemekana kuwa *inaelewa* sentensi. Ikiwa Eliza ingepewa sentensi inayofuata muundo "**Mimi ni** <u>huzuni</u>" inaweza kupanga upya na kubadilisha maneno katika sentensi ili kuunda jibu "Umekuwa **wewe** <u>huzuni</u> kwa muda gani".

Hii ilitoa mwonekano kwamba Eliza alielewa taarifa hiyo na alikuwa akiuliza swali la kufuatilia, ilhali kwa kweli, ilikuwa ikibadilisha wakati na kuongeza maneno fulani. Ikiwa Eliza haingeweza kutambua neno muhimu ambalo lilikuwa na jibu lake, badala yake ingetoa jibu la nasibu ambalo linaweza kutumika kwa taarifa nyingi tofauti. Eliza ingeweza kudanganywa kwa urahisi, kwa mfano ikiwa mtumiaji aliandika "**Wewe ni** <u>baiskeli</u>" inaweza kujibu "Umekuwa **mimi** <u>baiskeli</u> kwa muda gani?", badala ya jibu lenye mantiki zaidi.

[![Mazungumzo na Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Mazungumzo na Eliza")

> 🎥 Bofya picha hapo juu kwa video kuhusu programu ya awali ya ELIZA

> Kidokezo: Unaweza kusoma maelezo ya awali ya [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) iliyochapishwa mwaka wa 1966 ikiwa una akaunti ya ACM. Vinginevyo, soma kuhusu Eliza kwenye [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Zoezi - kuandika programu ya bot ya mazungumzo ya msingi

Bot ya mazungumzo, kama Eliza, ni programu inayochochea pembejeo ya mtumiaji na inaonekana kuelewa na kujibu kwa akili. Tofauti na Eliza, bot yetu haitakuwa na sheria kadhaa zinazotoa mwonekano wa kuwa na mazungumzo ya akili. Badala yake, bot yetu itakuwa na uwezo mmoja tu, kuendeleza mazungumzo kwa majibu ya nasibu ambayo yanaweza kufanya kazi katika mazungumzo yoyote ya kijuujuu.

### Mpango

Hatua zako unapoandika bot ya mazungumzo:

1. Chapisha maelekezo yanayomshauri mtumiaji jinsi ya kuingiliana na bot
2. Anzisha kitanzi
   1. Kubali pembejeo ya mtumiaji
   2. Ikiwa mtumiaji ameomba kutoka, basi toka
   3. Sindika pembejeo ya mtumiaji na uamue jibu (katika kesi hii, jibu ni chaguo la nasibu kutoka kwa orodha ya majibu yanayowezekana ya jumla)
   4. Chapisha jibu
3. Rudi kwa hatua ya 2

### Kuunda bot

Hebu tuunde bot sasa. Tutaanza kwa kufafanua baadhi ya misemo.

1. Unda bot hii mwenyewe kwa Python na majibu ya nasibu yafuatayo:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Hapa kuna baadhi ya matokeo ya sampuli ya kukuongoza (pembejeo ya mtumiaji iko kwenye mistari inayoanza na `>`):

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

    Suluhisho moja linalowezekana la kazi ni [hapa](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Simama na fikiria

    1. Je, unadhani majibu ya nasibu yangemdanganya mtu kufikiria kwamba bot inaelewa kweli?
    2. Ni vipengele gani bot ingehitaji kuwa bora zaidi?
    3. Ikiwa bot ingeweza 'kuelewa' maana ya sentensi, je, ingehitaji 'kukumbuka' maana ya sentensi za awali katika mazungumzo pia?

---

## 🚀Changamoto

Chagua mojawapo ya vipengele vya "simama na fikiria" hapo juu na ujaribu kuvitumia katika programu au andika suluhisho kwenye karatasi ukitumia pseudocode.

Katika somo linalofuata, utajifunza kuhusu mbinu kadhaa za kuchambua lugha asilia na ujifunzaji wa mashine.

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Angalia marejeleo hapa chini kama fursa za kusoma zaidi.

### Marejeleo

1. Schubert, Lenhart, "Isimu ya Kompyuta", *The Stanford Encyclopedia of Philosophy* (Toleo la Spring 2020), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Chuo Kikuu cha Princeton "Kuhusu WordNet." [WordNet](https://wordnet.princeton.edu/). Chuo Kikuu cha Princeton. 2010. 

## Kazi 

[Tafuta bot](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati asilia katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.