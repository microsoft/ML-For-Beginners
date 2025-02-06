# Tafsiri na uchambuzi wa hisia kwa ML

Katika masomo yaliyopita ulijifunza jinsi ya kujenga bot ya msingi kwa kutumia `TextBlob`, maktaba inayojumuisha ML kwa siri kufanya kazi za msingi za NLP kama vile uchimbaji wa misemo ya nomino. Changamoto nyingine muhimu katika taaluma ya lugha ya kompyuta ni tafsiri sahihi ya sentensi kutoka lugha moja inayozungumzwa au kuandikwa kwenda nyingine.

## [Maswali kabla ya somo](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

Tafsiri ni tatizo gumu sana kutokana na ukweli kwamba kuna maelfu ya lugha na kila moja inaweza kuwa na sheria tofauti za sarufi. Njia moja ni kubadilisha sheria rasmi za sarufi za lugha moja, kama Kiingereza, kuwa muundo usio tegemezi wa lugha, kisha kuitafsiri kwa kubadilisha tena kuwa lugha nyingine. Njia hii inamaanisha kwamba utachukua hatua zifuatazo:

1. **Utambulisho**. Tambua au tagi maneno katika lugha ya ingizo kuwa nomino, vitenzi n.k.
2. **Unda tafsiri**. Toa tafsiri ya moja kwa moja ya kila neno katika muundo wa lugha lengwa.

### Mfano wa sentensi, Kiingereza hadi Kiayalandi

Katika 'Kiingereza', sentensi _I feel happy_ ni maneno matatu kwa mpangilio:

- **subjekti** (I)
- **kitenzi** (feel)
- **kivumishi** (happy)

Hata hivyo, katika lugha ya 'Kiayalandi', sentensi hiyo hiyo ina muundo tofauti kabisa wa kisarufi - hisia kama "*happy*" au "*sad*" zinaonyeshwa kama kuwa *juu yako*.

Msemo wa Kiingereza `I feel happy` katika Kiayalandi ungekuwa `Tá athas orm`. Tafsiri ya *moja kwa moja* ingekuwa `Happy is upon me`.

Mzungumzaji wa Kiayalandi akitafsiri kwenda Kiingereza angesema `I feel happy`, sio `Happy is upon me`, kwa sababu wanaelewa maana ya sentensi, hata kama maneno na muundo wa sentensi ni tofauti.

Mpangilio rasmi wa sentensi katika Kiayalandi ni:

- **kitenzi** (Tá au is)
- **kivumishi** (athas, au happy)
- **subjekti** (orm, au upon me)

## Tafsiri

Programu ya tafsiri ya kijinga inaweza kutafsiri maneno pekee, ikipuuza muundo wa sentensi.

✅ Ikiwa umejifunza lugha ya pili (au ya tatu au zaidi) kama mtu mzima, unaweza kuwa ulianza kwa kufikiria katika lugha yako ya asili, ukitafsiri dhana neno kwa neno kichwani mwako kwenda lugha ya pili, kisha kusema tafsiri yako. Hii ni sawa na kile programu za tafsiri za kijinga za kompyuta zinavyofanya. Ni muhimu kupita hatua hii ili kufikia ufasaha!

Tafsiri ya kijinga inasababisha tafsiri mbaya (na wakati mwingine za kuchekesha): `I feel happy` inatafsiriwa moja kwa moja kuwa `Mise bhraitheann athas` katika Kiayalandi. Hiyo inamaanisha (moja kwa moja) `me feel happy` na sio sentensi halali ya Kiayalandi. Hata ingawa Kiingereza na Kiayalandi ni lugha zinazozungumzwa kwenye visiwa viwili vilivyo karibu sana, ni lugha tofauti sana zenye miundo tofauti ya sarufi.

> Unaweza kutazama baadhi ya video kuhusu mila za lugha ya Kiayalandi kama [hii](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Mbinu za kujifunza kwa mashine

Hadi sasa, umejifunza kuhusu mbinu za sheria rasmi za usindikaji wa lugha asilia. Njia nyingine ni kupuuza maana ya maneno, na _badala yake kutumia kujifunza kwa mashine kugundua mifumo_. Hii inaweza kufanya kazi katika tafsiri ikiwa una maandishi mengi (a *corpus*) au maandishi (*corpora*) katika lugha ya asili na lugha lengwa.

Kwa mfano, fikiria kesi ya *Pride and Prejudice*, riwaya maarufu ya Kiingereza iliyoandikwa na Jane Austen mwaka 1813. Ikiwa utasoma kitabu hicho kwa Kiingereza na tafsiri ya binadamu ya kitabu hicho kwa *Kifaransa*, unaweza kugundua misemo katika moja ambayo imetafsiriwa _kiidiomati_ kwenda nyingine. Utafanya hivyo kwa dakika moja.

Kwa mfano, wakati msemo wa Kiingereza kama `I have no money` unapotafsiriwa moja kwa moja kwenda Kifaransa, inaweza kuwa `Je n'ai pas de monnaie`. "Monnaie" ni neno la Kifaransa lenye maana potofu, kwani 'money' na 'monnaie' sio sawa. Tafsiri bora ambayo binadamu anaweza kufanya itakuwa `Je n'ai pas d'argent`, kwa sababu inafikisha vyema zaidi maana kwamba huna pesa (badala ya 'loose change' ambayo ni maana ya 'monnaie').

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.sw.png)

> Picha na [Jen Looper](https://twitter.com/jenlooper)

Ikiwa mfano wa ML una tafsiri za binadamu za kutosha kujenga mfano, inaweza kuboresha usahihi wa tafsiri kwa kutambua mifumo ya kawaida katika maandishi ambayo yametafsiriwa awali na wazungumzaji wa lugha zote mbili wenye ujuzi.

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

Katika kesi hii, tafsiri iliyoongozwa na ML inafanya kazi bora kuliko mtafsiri wa binadamu ambaye anaweka maneno yasiyo ya lazima katika mdomo wa mwandishi wa asili kwa 'uwazi'.

> Nini kinaendelea hapa? na kwa nini TextBlob ni nzuri sana katika tafsiri? Kweli, kwa nyuma, inatumia Google translate, AI yenye nguvu inayoweza kuchanganua mamilioni ya misemo ili kutabiri mistari bora kwa kazi inayofanyika. Hakuna kitu cha mwongozo kinachofanyika hapa na unahitaji muunganisho wa intaneti kutumia `blob.translate`.

✅ Try some more sentences. Which is better, ML or human translation? In which cases?

## Sentiment analysis

Another area where machine learning can work very well is sentiment analysis. A non-ML approach to sentiment is to identify words and phrases which are 'positive' and 'negative'. Then, given a new piece of text, calculate the total value of the positive, negative and neutral words to identify the overall sentiment. 

This approach is easily tricked as you may have seen in the Marvin task - the sentence `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` ni sentensi ya kejeli, yenye hisia hasi, lakini algorithimu rahisi inagundua 'great', 'wonderful', 'glad' kama chanya na 'waste', 'lost' na 'dark' kama hasi. Hisia za jumla zinashawishiwa na maneno haya yanayopingana.

✅ Simama kwa sekunde na fikiria jinsi tunavyowasilisha kejeli kama wazungumzaji wa binadamu. Mzigo wa sauti unachukua jukumu kubwa. Jaribu kusema msemo "Well, that film was awesome" kwa njia tofauti ili kugundua jinsi sauti yako inavyoonyesha maana.

### Mbinu za ML

Mbinu ya ML itakuwa kukusanya kwa mikono maandishi hasi na chanya - tweets, au mapitio ya filamu, au chochote ambapo binadamu ametoa alama *na* maoni yaliyoandikwa. Kisha mbinu za NLP zinaweza kutumika kwa maoni na alama, ili mifumo itokeze (mfano, mapitio chanya ya filamu yana mwelekeo wa kuwa na msemo 'Oscar worthy' zaidi kuliko mapitio hasi ya filamu, au mapitio chanya ya migahawa yanasema 'gourmet' zaidi kuliko 'disgusting').

> ⚖️ **Mfano**: Ikiwa ulifanya kazi katika ofisi ya mwanasiasa na kulikuwa na sheria mpya inayojadiliwa, wapiga kura wanaweza kuandika kwa ofisi hiyo na barua pepe za kuunga mkono au kupinga sheria hiyo mpya. Tuseme umepewa kazi ya kusoma barua pepe na kuzipanga katika vikundi 2, *kwa* na *dhidi*. Ikiwa kulikuwa na barua pepe nyingi, unaweza kuzidiwa kujaribu kuzisoma zote. Si ingekuwa vizuri ikiwa bot ingeweza kuzisoma zote kwa ajili yako, kuzielewa na kukuambia ni barua pepe gani inapaswa kuwa katika kundi gani? 
> 
> Njia moja ya kufanikisha hilo ni kutumia Kujifunza kwa Mashine. Ungefundisha mfano kwa sehemu ya barua pepe za *dhidi* na sehemu ya barua pepe za *kwa*. Mfano ungeweza kuhusisha misemo na maneno na upande wa dhidi na upande wa kwa, *lakini usingeelewa maudhui yoyote*, tu kwamba maneno na mifumo fulani ina uwezekano mkubwa wa kuonekana katika barua pepe za *dhidi* au za *kwa*. Ungeweza kuijaribu na barua pepe fulani ambazo hukuzitumia kufundisha mfano, na kuona kama inafikia hitimisho sawa na wewe. Kisha, mara tu unapokuwa na furaha na usahihi wa mfano, unaweza kushughulikia barua pepe za baadaye bila kulazimika kusoma kila moja.

✅ Je, mchakato huu unafanana na michakato uliotumia katika masomo yaliyopita?

## Zoezi - sentensi za hisia

Hisia hupimwa kwa *polarity* ya -1 hadi 1, ikimaanisha -1 ni hisia hasi zaidi, na 1 ni hisia chanya zaidi. Hisia pia hupimwa kwa alama ya 0 - 1 kwa objectivity (0) na subjectivity (1).

Angalia tena *Pride and Prejudice* ya Jane Austen. Maandishi yanapatikana hapa katika [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Sampuli hapa chini inaonyesha programu fupi inayochambua hisia za sentensi za kwanza na za mwisho kutoka kitabu hicho na kuonyesha polarity ya hisia zake na alama ya subjectivity/objectivity.

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

Kazi yako ni kuamua, kwa kutumia polarity ya hisia, ikiwa *Pride and Prejudice* ina sentensi zaidi chanya kabisa kuliko hasi kabisa. Kwa kazi hii, unaweza kudhani kwamba alama ya polarity ya 1 au -1 ni chanya kabisa au hasi kabisa kwa mtiririko huo.

**Hatua:**

1. Pakua [nakala ya Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) kutoka Project Gutenberg kama faili ya .txt. Ondoa metadata mwanzoni na mwishoni mwa faili, ukiacha tu maandishi ya asili
2. Fungua faili hiyo kwa Python na toa yaliyomo kama kamba
3. Unda TextBlob kwa kutumia kamba ya kitabu
4. Changanua kila sentensi katika kitabu kwa mzunguko
   1. Ikiwa polarity ni 1 au -1 hifadhi sentensi katika safu au orodha ya ujumbe chanya au hasi
5. Mwishoni, chapisha sentensi zote chanya na hasi (kando) na idadi ya kila moja.

Hapa kuna [suluhisho](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Ukaguzi wa Maarifa

1. Hisia zinategemea maneno yaliyotumika katika sentensi, lakini je, msimbo *unaelewa* maneno hayo?
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
   2. Sentensi 3 zifuatazo zilipata alama ya hisia chanya kabisa, lakini kwa kusoma kwa makini, sio sentensi chanya. Kwa nini uchambuzi wa hisia ulidhani ni sentensi chanya?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Je, unakubaliana au hukubaliani na polarity hasi kabisa ya sentensi zifuatazo?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Mpenzi yeyote wa Jane Austen ataelewa kwamba mara nyingi hutumia vitabu vyake kukosoa vipengele vya kijinga vya jamii ya Kiingereza ya Regency. Elizabeth Bennett, mhusika mkuu katika *Pride and Prejudice*, ni mwangalizi mzuri wa kijamii (kama mwandishi) na lugha yake mara nyingi ina madoido mengi. Hata Bw. Darcy (mpenzi katika hadithi) anabaini matumizi ya kucheza na kejeli ya Elizabeth: "Nimepata furaha ya kukujua kwa muda mrefu wa kutosha kujua kwamba unafurahia sana wakati mwingine kutoa maoni ambayo kwa kweli sio yako."

---

## 🚀Changamoto

Je, unaweza kumfanya Marvin kuwa bora zaidi kwa kutoa vipengele vingine kutoka kwa pembejeo ya mtumiaji?

## [Maswali baada ya somo](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## Mapitio na Kujisomea

Kuna njia nyingi za kutoa hisia kutoka kwa maandishi. Fikiria matumizi ya kibiashara ambayo yanaweza kutumia mbinu hii. Fikiria jinsi inavyoweza kwenda kombo. Soma zaidi kuhusu mifumo ya kisasa ya kibiashara inayochambua hisia kama [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Jaribu baadhi ya sentensi za Pride and Prejudice hapo juu na uone kama inaweza kugundua madoido.

## Kazi 

[Leseni ya kishairi](assignment.md)

**Kanusho**:
Hati hii imetafsiriwa kwa kutumia huduma za tafsiri za AI zinazotegemea mashine. Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au upungufu. Hati ya asili katika lugha yake ya asili inapaswa kuzingatiwa kama chanzo rasmi. Kwa taarifa muhimu, tafsiri ya kitaalamu ya kibinadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri potofu zinazotokana na matumizi ya tafsiri hii.