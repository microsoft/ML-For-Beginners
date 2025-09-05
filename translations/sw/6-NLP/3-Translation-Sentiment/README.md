<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T17:02:18+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "sw"
}
-->
# Tafsiri na uchambuzi wa hisia kwa kutumia ML

Katika masomo ya awali ulijifunza jinsi ya kujenga bot ya msingi kwa kutumia `TextBlob`, maktaba inayotumia ML nyuma ya pazia kutekeleza kazi za msingi za NLP kama uchimbaji wa misemo ya nomino. Changamoto nyingine muhimu katika isimu ya kompyuta ni tafsiri sahihi ya sentensi kutoka lugha moja ya mazungumzo au maandishi kwenda nyingine.

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

Tafsiri ni tatizo gumu sana linalochangiwa na ukweli kwamba kuna maelfu ya lugha na kila moja inaweza kuwa na sheria tofauti za sarufi. Njia moja ni kubadilisha sheria rasmi za sarufi za lugha moja, kama Kiingereza, kuwa muundo usioegemea lugha, kisha kutafsiri kwa kubadilisha tena kuwa lugha nyingine. Njia hii inahusisha hatua zifuatazo:

1. **Utambulisho**. Tambua au weka alama maneno katika lugha ya ingizo kama nomino, vitenzi, n.k.
2. **Unda tafsiri**. Tengeneza tafsiri ya moja kwa moja ya kila neno katika muundo wa lugha lengwa.

### Mfano wa sentensi, Kiingereza hadi Kiarishi

Katika 'Kiingereza', sentensi _I feel happy_ ina maneno matatu kwa mpangilio:

- **somo** (I)
- **kitenzi** (feel)
- **kivumishi** (happy)

Hata hivyo, katika lugha ya 'Kiarishi', sentensi hiyo hiyo ina muundo tofauti kabisa wa kisarufi - hisia kama "*happy*" au "*sad*" zinaonyeshwa kama ziko *juu yako*.

Kifungu cha Kiingereza `I feel happy` katika Kiarishi kingekuwa `Tá athas orm`. Tafsiri ya *moja kwa moja* ingekuwa `Happy is upon me`.

Mzungumzaji wa Kiarishi akitafsiri kwenda Kiingereza atasema `I feel happy`, si `Happy is upon me`, kwa sababu anaelewa maana ya sentensi, hata kama maneno na muundo wa sentensi ni tofauti.

Mpangilio rasmi wa sentensi katika Kiarishi ni:

- **kitenzi** (Tá au is)
- **kivumishi** (athas, au happy)
- **somu** (orm, au upon me)

## Tafsiri

Programu ya tafsiri ya kijinga inaweza kutafsiri maneno pekee, ikipuuza muundo wa sentensi.

✅ Ikiwa umejifunza lugha ya pili (au ya tatu au zaidi) kama mtu mzima, huenda ulianza kwa kufikiria katika lugha yako ya asili, ukitafsiri dhana neno kwa neno kichwani mwako kwenda lugha ya pili, kisha ukazungumza tafsiri yako. Hii ni sawa na kile programu za kompyuta za tafsiri ya kijinga zinavyofanya. Ni muhimu kupita hatua hii ili kufikia ufasaha!

Tafsiri ya kijinga husababisha tafsiri mbaya (na wakati mwingine za kuchekesha): `I feel happy` inatafsiriwa moja kwa moja kuwa `Mise bhraitheann athas` katika Kiarishi. Hii inamaanisha (moja kwa moja) `me feel happy` na si sentensi sahihi ya Kiarishi. Ingawa Kiingereza na Kiarishi ni lugha zinazozungumzwa kwenye visiwa viwili vilivyo karibu sana, ni lugha tofauti kabisa zenye miundo tofauti ya sarufi.

> Unaweza kutazama baadhi ya video kuhusu mila za lugha ya Kiarishi kama [hii](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Njia za kujifunza kwa mashine

Hadi sasa, umejifunza kuhusu njia ya sheria rasmi kwa usindikaji wa lugha asilia. Njia nyingine ni kupuuza maana ya maneno, na _badala yake kutumia kujifunza kwa mashine kutambua mifumo_. Hii inaweza kufanya kazi katika tafsiri ikiwa una maandishi mengi (*corpus*) au maandishi (*corpora*) katika lugha ya asili na lengwa.

Kwa mfano, fikiria kesi ya *Pride and Prejudice*, riwaya maarufu ya Kiingereza iliyoandikwa na Jane Austen mwaka wa 1813. Ikiwa utachunguza kitabu hicho kwa Kiingereza na tafsiri ya binadamu ya kitabu hicho kwa *Kifaransa*, unaweza kutambua misemo katika moja ambayo imetafsiriwa _kiidiomatikali_ katika nyingine. Utafanya hivyo kwa muda mfupi.

Kwa mfano, wakati kifungu cha Kiingereza kama `I have no money` kinatafsiriwa moja kwa moja kuwa Kifaransa, kinaweza kuwa `Je n'ai pas de monnaie`. "Monnaie" ni neno la Kifaransa lenye maana ya 'false cognate', kwani 'money' na 'monnaie' si sawa. Tafsiri bora ambayo binadamu anaweza kufanya ingekuwa `Je n'ai pas d'argent`, kwa sababu inawasilisha vyema maana kwamba huna pesa (badala ya 'chenji' ambayo ndiyo maana ya 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Picha na [Jen Looper](https://twitter.com/jenlooper)

Ikiwa modeli ya ML ina tafsiri za binadamu za kutosha kujenga modeli, inaweza kuboresha usahihi wa tafsiri kwa kutambua mifumo ya kawaida katika maandishi ambayo yamewahi kutafsiriwa na wataalamu wa binadamu wa lugha zote mbili.

### Zoezi - tafsiri

Unaweza kutumia `TextBlob` kutafsiri sentensi. Jaribu mstari maarufu wa kwanza wa **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` inafanya kazi nzuri sana katika tafsiri: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Inaweza kusemwa kwamba tafsiri ya TextBlob ni sahihi zaidi, kwa kweli, kuliko tafsiri ya Kifaransa ya 1932 ya kitabu hicho na V. Leconte na Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Katika kesi hii, tafsiri iliyoongozwa na ML inafanya kazi bora zaidi kuliko mtafsiri wa binadamu ambaye anaweka maneno yasiyo ya lazima katika mdomo wa mwandishi wa asili kwa 'ufafanuzi'.

> Nini kinaendelea hapa? Na kwa nini TextBlob ni nzuri sana katika tafsiri? Naam, nyuma ya pazia, inatumia Google translate, AI ya kisasa inayoweza kuchanganua mamilioni ya misemo kutabiri mistari bora kwa kazi husika. Hakuna kitu cha mwongozo kinachoendelea hapa na unahitaji muunganisho wa mtandao kutumia `blob.translate`.

✅ Jaribu sentensi zaidi. Ipi ni bora, ML au tafsiri ya binadamu? Katika hali zipi?

## Uchambuzi wa hisia

Eneo lingine ambapo kujifunza kwa mashine kunaweza kufanya kazi vizuri sana ni uchambuzi wa hisia. Njia isiyo ya ML ya hisia ni kutambua maneno na misemo ambayo ni 'chanya' na 'hasi'. Kisha, ukizingatia kipande kipya cha maandishi, hesabu thamani ya jumla ya maneno chanya, hasi na ya kawaida ili kutambua hisia za jumla.

Njia hii inaweza kudanganywa kwa urahisi kama ulivyoona katika kazi ya Marvin - sentensi `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ni sentensi ya hisia hasi ya kejeli, lakini algoriti rahisi hugundua 'great', 'wonderful', 'glad' kama chanya na 'waste', 'lost' na 'dark' kama hasi. Hisia za jumla zinavurugwa na maneno haya yanayokinzana.

✅ Simama kidogo na fikiria jinsi tunavyowasilisha kejeli kama wazungumzaji wa binadamu. Miondoko ya sauti ina jukumu kubwa. Jaribu kusema kifungu "Well, that film was awesome" kwa njia tofauti ili kugundua jinsi sauti yako inavyowasilisha maana.

### Njia za ML

Njia ya ML ingekuwa kukusanya kwa mikono maandishi hasi na chanya - tweets, au hakiki za filamu, au chochote ambacho binadamu ametoa alama *na* maoni yaliyoandikwa. Kisha mbinu za NLP zinaweza kutumika kwa maoni na alama, ili mifumo ijitokeze (mfano, hakiki chanya za filamu huwa na kifungu 'Oscar worthy' zaidi kuliko hakiki hasi za filamu, au hakiki chanya za migahawa husema 'gourmet' zaidi kuliko 'disgusting').

> ⚖️ **Mfano**: Ikiwa unafanya kazi katika ofisi ya mwanasiasa na kuna sheria mpya inayojadiliwa, wapiga kura wanaweza kuandika barua pepe kwa ofisi hiyo wakiiunga mkono au kupinga sheria hiyo mpya. Tuseme umepewa jukumu la kusoma barua pepe na kuzipanga katika mafungu 2, *kwa* na *dhidi*. Ikiwa kuna barua pepe nyingi, unaweza kuzidiwa ukijaribu kuzisoma zote. Je, si ingekuwa vizuri ikiwa bot ingeweza kuzisoma zote kwa niaba yako, kuzielewa na kukuambia barua pepe ipi inapaswa kuwa katika fungu gani? 
> 
> Njia moja ya kufanikisha hilo ni kutumia Kujifunza kwa Mashine. Ungefundisha modeli kwa sehemu ya barua pepe za *dhidi* na sehemu ya barua pepe za *kwa*. Modeli ingekuwa na mwelekeo wa kuhusisha misemo na maneno na upande wa dhidi na upande wa kwa, *lakini haingeweza kuelewa maudhui yoyote*, isipokuwa kwamba maneno na mifumo fulani ina uwezekano mkubwa wa kuonekana katika barua pepe za *dhidi* au *kwa*. Ungeijaribu na barua pepe ambazo hukuzitumia kufundisha modeli, na kuona ikiwa inafikia hitimisho sawa na ulilofikia. Kisha, mara tu unapokuwa na furaha na usahihi wa modeli, ungeweza kushughulikia barua pepe za baadaye bila kulazimika kusoma kila moja.

✅ Je, mchakato huu unafanana na michakato uliyotumia katika masomo ya awali?

## Zoezi - sentensi za hisia

Hisia hupimwa kwa *polarity* ya -1 hadi 1, ikimaanisha -1 ni hisia hasi zaidi, na 1 ni hisia chanya zaidi. Hisia pia hupimwa kwa alama ya 0 - 1 kwa usawa (0) na maoni binafsi (1).

Angalia tena *Pride and Prejudice* ya Jane Austen. Maandishi yanapatikana hapa [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Sampuli hapa chini inaonyesha programu fupi inayochambua hisia za sentensi za kwanza na za mwisho kutoka kitabu hicho na kuonyesha polarity ya hisia zake na alama ya usawa/maoni binafsi.

Unapaswa kutumia maktaba ya `TextBlob` (iliyotajwa hapo juu) kuamua `sentiment` (huna haja ya kuandika kikokotoo chako cha hisia) katika kazi ifuatayo.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Unaona matokeo yafuatayo:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Changamoto - angalia polarity ya hisia

Kazi yako ni kuamua, kwa kutumia polarity ya hisia, ikiwa *Pride and Prejudice* ina sentensi chanya kabisa zaidi kuliko hasi kabisa. Kwa kazi hii, unaweza kudhani kwamba alama ya polarity ya 1 au -1 ni chanya kabisa au hasi kabisa mtawalia.

**Hatua:**

1. Pakua [nakala ya Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) kutoka Project Gutenberg kama faili ya .txt. Ondoa metadata mwanzoni na mwishoni mwa faili, ukiacha maandishi ya asili pekee
2. Fungua faili hiyo katika Python na uchukue yaliyomo kama kamba
3. Unda TextBlob kwa kutumia kamba ya kitabu
4. Changanua kila sentensi katika kitabu kwa mzunguko
   1. Ikiwa polarity ni 1 au -1 hifadhi sentensi hiyo katika array au orodha ya ujumbe chanya au hasi
5. Mwishoni, chapisha sentensi zote chanya na hasi (kando) na idadi ya kila moja.

Hapa kuna [suluhisho la sampuli](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Maswali ya Maarifa

1. Hisia zinategemea maneno yanayotumika katika sentensi, lakini je, msimbo *unaelewa* maneno hayo?
2. Je, unadhani polarity ya hisia ni sahihi, au kwa maneno mengine, je, *unakubaliana* na alama hizo?
   1. Hasa, je, unakubaliana au hukubaliani na polarity chanya kabisa ya sentensi zifuatazo?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Sentensi 3 zifuatazo zilipimwa na polarity chanya kabisa, lakini ukisoma kwa makini, si sentensi chanya. Kwa nini uchambuzi wa hisia ulifikiri ni sentensi chanya?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Je, unakubaliana au hukubaliani na polarity hasi kabisa ya sentensi zifuatazo?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Mpenzi yeyote wa Jane Austen ataelewa kwamba mara nyingi hutumia vitabu vyake kukosoa vipengele vya kipuuzi vya jamii ya Kiingereza ya Regency. Elizabeth Bennett, mhusika mkuu katika *Pride and Prejudice*, ni mwangalizi wa kijamii mwenye makini (kama mwandishi) na lugha yake mara nyingi ina maana nzito. Hata Mr. Darcy (mpenzi wa hadithi) anabainisha matumizi ya Elizabeth ya lugha ya kucheza na ya kejeli: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Changamoto

Je, unaweza kumfanya Marvin kuwa bora zaidi kwa kuchambua vipengele vingine kutoka kwa maoni ya mtumiaji?

## [Maswali ya baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea
Kuna njia nyingi za kuchambua hisia kutoka kwa maandishi. Fikiria matumizi ya kibiashara ambayo yanaweza kutumia mbinu hii. Fikiria jinsi inaweza kwenda kombo. Soma zaidi kuhusu mifumo ya kisasa inayofaa kwa biashara ambayo huchambua hisia kama [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Jaribu baadhi ya sentensi kutoka "Pride and Prejudice" hapo juu na uone kama inaweza kugundua nyongeza ya maana.

## Kazi

[Leseni ya kishairi](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.