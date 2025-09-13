<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T15:58:17+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "sw"
}
-->
# Kujenga Suluhisho za Kujifunza kwa Mashine kwa AI Inayowajibika

![Muhtasari wa AI inayowajibika katika Kujifunza kwa Mashine kwenye sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote na [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Utangulizi

Katika mtaala huu, utaanza kugundua jinsi kujifunza kwa mashine kunavyoathiri maisha yetu ya kila siku. Hata sasa, mifumo na modeli zinahusika katika kazi za maamuzi ya kila siku, kama vile utambuzi wa afya, idhini ya mikopo, au kugundua udanganyifu. Kwa hivyo, ni muhimu kwamba modeli hizi zifanye kazi vizuri ili kutoa matokeo yanayoweza kuaminika. Kama programu yoyote, mifumo ya AI inaweza kukosa matarajio au kuwa na matokeo yasiyofaa. Ndiyo maana ni muhimu kuelewa na kuelezea tabia ya modeli ya AI.

Fikiria kinachoweza kutokea wakati data unayotumia kujenga modeli hizi inakosa baadhi ya demografia, kama vile rangi, jinsia, mtazamo wa kisiasa, dini, au inawakilisha demografia hizo kwa uwiano usio sawa. Je, kuhusu pale ambapo matokeo ya modeli yanatafsiriwa kupendelea demografia fulani? Matokeo yake ni nini kwa programu? Zaidi ya hayo, nini kinatokea pale ambapo modeli ina matokeo mabaya na inadhuru watu? Nani anawajibika kwa tabia ya mifumo ya AI? Haya ni baadhi ya maswali tutakayochunguza katika mtaala huu.

Katika somo hili, utaweza:

- Kuongeza ufahamu wako kuhusu umuhimu wa haki katika kujifunza kwa mashine na madhara yanayohusiana na haki.
- Kujifunza kuhusu mazoezi ya kuchunguza hali zisizo za kawaida ili kuhakikisha uaminifu na usalama.
- Kupata uelewa wa hitaji la kuwawezesha watu wote kwa kubuni mifumo jumuishi.
- Kuchunguza umuhimu wa kulinda faragha na usalama wa data na watu.
- Kuelewa umuhimu wa mbinu ya "sanduku la kioo" kuelezea tabia ya modeli za AI.
- Kuwa makini kuhusu jinsi uwajibikaji ni muhimu katika kujenga uaminifu kwa mifumo ya AI.

## Mahitaji ya awali

Kama mahitaji ya awali, tafadhali chukua Njia ya Kujifunza ya "Kanuni za AI Inayowajibika" na tazama video hapa chini kuhusu mada hii:

Jifunze zaidi kuhusu AI Inayowajibika kwa kufuata [Njia ya Kujifunza](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Mbinu ya Microsoft kwa AI Inayowajibika](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Mbinu ya Microsoft kwa AI Inayowajibika")

> 🎥 Bofya picha hapo juu kwa video: Mbinu ya Microsoft kwa AI Inayowajibika

## Haki

Mifumo ya AI inapaswa kuwachukulia watu wote kwa usawa na kuepuka kuathiri makundi yanayofanana kwa njia tofauti. Kwa mfano, mifumo ya AI inapotoa mwongozo kuhusu matibabu ya afya, maombi ya mikopo, au ajira, inapaswa kutoa mapendekezo sawa kwa kila mtu mwenye dalili, hali ya kifedha, au sifa za kitaaluma zinazofanana. Kila mmoja wetu kama binadamu hubeba upendeleo wa kurithi ambao huathiri maamuzi na vitendo vyetu. Upendeleo huu unaweza kuonekana katika data tunayotumia kufundisha mifumo ya AI. Mara nyingine, mabadiliko haya hutokea bila kukusudia. Mara nyingi ni vigumu kujua kwa makusudi wakati unaleta upendeleo katika data.

**“Kutokuwa na haki”** kunajumuisha athari mbaya, au “madhara”, kwa kundi la watu, kama vile wale wanaofafanuliwa kwa misingi ya rangi, jinsia, umri, au hali ya ulemavu. Madhara makuu yanayohusiana na haki yanaweza kuainishwa kama:

- **Ugawaji**, ikiwa jinsia au kabila fulani linapendelewa zaidi ya jingine.
- **Ubora wa huduma**. Ikiwa unafundisha data kwa hali moja maalum lakini hali halisi ni ngumu zaidi, husababisha huduma isiyofanya kazi vizuri. Kwa mfano, kifaa cha kutambua sabuni ya mkono ambacho hakikuweza kutambua watu wenye ngozi nyeusi. [Rejea](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Kudhalilisha**. Kukosoa na kuweka lebo isiyo ya haki kwa kitu au mtu. Kwa mfano, teknolojia ya kuweka lebo ya picha iliwahi kuweka lebo isiyo sahihi kwa picha za watu wenye ngozi nyeusi kama sokwe.
- **Uwiano wa juu au wa chini**. Wazo ni kwamba kundi fulani halionekani katika taaluma fulani, na huduma yoyote inayozidi kukuza hali hiyo inachangia madhara.
- **Kuweka mazoea ya kijinsia**. Kuunganisha kundi fulani na sifa zilizowekwa awali. Kwa mfano, mfumo wa kutafsiri lugha kati ya Kiingereza na Kituruki unaweza kuwa na makosa kutokana na maneno yenye uhusiano wa kijinsia.

![Tafsiri kwa Kituruki](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> Tafsiri kwa Kituruki

![Tafsiri kurudi kwa Kiingereza](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> Tafsiri kurudi kwa Kiingereza

Wakati wa kubuni na kujaribu mifumo ya AI, tunahitaji kuhakikisha kwamba AI ni ya haki na haijapangiliwa kufanya maamuzi yenye upendeleo au ubaguzi, ambayo binadamu pia wamezuiwa kufanya. Kuhakikisha haki katika AI na kujifunza kwa mashine bado ni changamoto ngumu ya kijamii na kiteknolojia.

### Uaminifu na Usalama

Ili kujenga uaminifu, mifumo ya AI inahitaji kuwa ya kuaminika, salama, na thabiti katika hali za kawaida na zisizotarajiwa. Ni muhimu kujua jinsi mifumo ya AI itakavyotenda katika hali mbalimbali, hasa pale ambapo kuna hali zisizo za kawaida. Wakati wa kujenga suluhisho za AI, kunapaswa kuwa na umakini mkubwa juu ya jinsi ya kushughulikia hali mbalimbali ambazo suluhisho za AI zitakutana nazo. Kwa mfano, gari linalojiendesha lenyewe linahitaji kuweka usalama wa watu kama kipaumbele cha juu. Kwa hivyo, AI inayotumia gari inahitaji kuzingatia hali zote zinazowezekana ambazo gari linaweza kukutana nazo kama usiku, dhoruba za radi au theluji, watoto wakikimbia barabarani, wanyama wa kipenzi, ujenzi wa barabara, n.k. Jinsi mfumo wa AI unavyoweza kushughulikia hali mbalimbali kwa uaminifu na usalama inaonyesha kiwango cha matarajio ambacho mwanasayansi wa data au msanidi wa AI alizingatia wakati wa kubuni au kujaribu mfumo.

> [🎥 Bofya hapa kwa video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Ujumuishi

Mifumo ya AI inapaswa kubuniwa ili kushirikisha na kuwawezesha kila mtu. Wakati wa kubuni na kutekeleza mifumo ya AI, wanasayansi wa data na wasanidi wa AI hutambua na kushughulikia vizuizi vinavyoweza kuwapo katika mfumo ambavyo vinaweza kuwazuia watu bila kukusudia. Kwa mfano, kuna watu bilioni 1 wenye ulemavu duniani kote. Kwa maendeleo ya AI, wanaweza kufikia aina mbalimbali za taarifa na fursa kwa urahisi zaidi katika maisha yao ya kila siku. Kwa kushughulikia vizuizi, inatoa fursa za kuleta ubunifu na kuendeleza bidhaa za AI zenye uzoefu bora ambao unawanufaisha kila mtu.

> [🎥 Bofya hapa kwa video: ujumuishi katika AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Usalama na Faragha

Mifumo ya AI inapaswa kuwa salama na kuheshimu faragha ya watu. Watu wana imani ndogo katika mifumo inayoweka faragha yao, taarifa zao, au maisha yao hatarini. Wakati wa kufundisha modeli za kujifunza kwa mashine, tunategemea data ili kutoa matokeo bora. Katika kufanya hivyo, asili ya data na uadilifu wake lazima uzingatiwe. Kwa mfano, je, data ilitolewa na mtumiaji au ilipatikana hadharani? Kisha, wakati wa kufanya kazi na data, ni muhimu kuendeleza mifumo ya AI inayoweza kulinda taarifa za siri na kupinga mashambulizi. Kadri AI inavyozidi kuenea, kulinda faragha na kuhakikisha usalama wa taarifa muhimu za kibinafsi na za kibiashara kunazidi kuwa muhimu na changamoto. Masuala ya faragha na usalama wa data yanahitaji umakini wa karibu hasa kwa AI kwa sababu upatikanaji wa data ni muhimu kwa mifumo ya AI kufanya utabiri sahihi na maamuzi yanayojulishwa kuhusu watu.

> [🎥 Bofya hapa kwa video: usalama katika AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kama sekta, tumepiga hatua kubwa katika Faragha na Usalama, ikichochewa sana na kanuni kama GDPR (Kanuni ya Jumla ya Ulinzi wa Data).
- Hata hivyo, na mifumo ya AI tunapaswa kutambua mvutano kati ya hitaji la data zaidi ya kibinafsi ili kufanya mifumo kuwa ya kibinafsi na yenye ufanisi – na faragha.
- Kama ilivyokuwa na kuzaliwa kwa kompyuta zilizounganishwa na mtandao, tunaona pia ongezeko kubwa la idadi ya masuala ya usalama yanayohusiana na AI.
- Wakati huo huo, tumeona AI ikitumika kuboresha usalama. Kwa mfano, skana nyingi za kisasa za antivirus zinatumia AI heuristics leo.
- Tunahitaji kuhakikisha kwamba michakato yetu ya Sayansi ya Data inachanganyika kwa usawa na mazoea ya faragha na usalama ya hivi karibuni.

### Uwazi

Mifumo ya AI inapaswa kueleweka. Sehemu muhimu ya uwazi ni kuelezea tabia ya mifumo ya AI na vipengele vyake. Kuboresha uelewa wa mifumo ya AI kunahitaji wadau kuelewa jinsi na kwa nini mifumo hiyo inavyofanya kazi ili waweze kutambua masuala ya utendaji, wasiwasi wa usalama na faragha, upendeleo, mazoea ya kutengwa, au matokeo yasiyotarajiwa. Tunaamini pia kwamba wale wanaotumia mifumo ya AI wanapaswa kuwa waaminifu na wazi kuhusu wakati, kwa nini, na jinsi wanavyochagua kuitumia. Pamoja na mapungufu ya mifumo wanayotumia. Kwa mfano, ikiwa benki inatumia mfumo wa AI kusaidia maamuzi ya mikopo ya watumiaji, ni muhimu kuchunguza matokeo na kuelewa ni data gani inayoshawishi mapendekezo ya mfumo. Serikali zinaanza kudhibiti AI katika sekta mbalimbali, kwa hivyo wanasayansi wa data na mashirika lazima waeleze ikiwa mfumo wa AI unakidhi mahitaji ya udhibiti, hasa pale ambapo kuna matokeo yasiyofaa.

> [🎥 Bofya hapa kwa video: uwazi katika AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kwa sababu mifumo ya AI ni ngumu sana, ni vigumu kuelewa jinsi inavyofanya kazi na kutafsiri matokeo.
- Ukosefu huu wa uelewa huathiri jinsi mifumo hii inavyosimamiwa, kuendeshwa, na kuandikwa.
- Ukosefu huu wa uelewa zaidi huathiri maamuzi yanayofanywa kwa kutumia matokeo ambayo mifumo hii inazalisha.

### Uwajibikaji

Watu wanaobuni na kupeleka mifumo ya AI lazima wawajibike kwa jinsi mifumo yao inavyofanya kazi. Hitaji la uwajibikaji ni muhimu hasa kwa teknolojia nyeti kama utambuzi wa uso. Hivi karibuni, kumekuwa na mahitaji yanayoongezeka ya teknolojia ya utambuzi wa uso, hasa kutoka kwa mashirika ya utekelezaji wa sheria yanayoona uwezo wa teknolojia hiyo katika matumizi kama vile kutafuta watoto waliopotea. Hata hivyo, teknolojia hizi zinaweza kutumiwa na serikali kuweka uhuru wa msingi wa raia wao hatarini kwa, kwa mfano, kuwezesha ufuatiliaji wa mara kwa mara wa watu fulani. Kwa hivyo, wanasayansi wa data na mashirika yanahitaji kuwajibika kwa jinsi mfumo wao wa AI unavyoathiri watu binafsi au jamii.

[![Mtafiti Mkuu wa AI Anaonya kuhusu Ufuatiliaji wa Misa Kupitia Utambuzi wa Uso](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Mbinu ya Microsoft kwa AI Inayowajibika")

> 🎥 Bofya picha hapo juu kwa video: Onyo kuhusu Ufuatiliaji wa Misa Kupitia Utambuzi wa Uso

Hatimaye, moja ya maswali makubwa kwa kizazi chetu, kama kizazi cha kwanza kinacholeta AI kwa jamii, ni jinsi ya kuhakikisha kwamba kompyuta zitabaki kuwajibika kwa watu na jinsi ya kuhakikisha kwamba watu wanaobuni kompyuta wanabaki kuwajibika kwa kila mtu mwingine.

## Tathmini ya Athari

Kabla ya kufundisha modeli ya kujifunza kwa mashine, ni muhimu kufanya tathmini ya athari ili kuelewa madhumuni ya mfumo wa AI; matumizi yaliyokusudiwa; mahali itakapotumika; na nani atakayeingiliana na mfumo. Hizi ni muhimu kwa mthibitishaji au mjaribu wa mfumo kujua ni mambo gani ya kuzingatia wakati wa kutambua hatari zinazowezekana na matokeo yanayotarajiwa.

Sehemu zifuatazo ni za kuzingatia wakati wa kufanya tathmini ya athari:

* **Athari mbaya kwa watu binafsi**. Kuwa na ufahamu wa vizuizi au mahitaji yoyote, matumizi yasiyoungwa mkono au mapungufu yoyote yanayojulikana yanayozuia utendaji wa mfumo ni muhimu ili kuhakikisha kwamba mfumo hautumiki kwa njia inayoweza kudhuru watu binafsi.
* **Mahitaji ya data**. Kupata uelewa wa jinsi na wapi mfumo utatumia data husaidia wathibitishaji kuchunguza mahitaji yoyote ya data unayohitaji kuzingatia (mfano, kanuni za GDPR au HIPPA). Zaidi ya hayo, chunguza ikiwa chanzo au wingi wa data ni muhimu kwa mafunzo.
* **Muhtasari wa athari**. Kusanya orodha ya madhara yanayoweza kutokea kutokana na matumizi ya mfumo. Katika mzunguko wa maisha wa ML, hakiki ikiwa masuala yaliyotambuliwa yamepunguzwa au kushughulikiwa.
* **Malengo yanayofaa** kwa kila moja ya kanuni sita za msingi. Tathmini ikiwa malengo kutoka kwa kila kanuni yametimizwa na ikiwa kuna mapungufu yoyote.

## Kutatua Hitilafu kwa AI Inayowajibika

Kama vile kutatua hitilafu katika programu, kutatua hitilafu katika mfumo wa AI ni mchakato muhimu wa kutambua na kutatua masuala katika mfumo. Kuna mambo mengi yanayoweza kuathiri modeli kutofanya kazi kama inavyotarajiwa au kwa uwajibikaji. Vipimo vingi vya utendaji wa modeli vya jadi ni jumla za kiasi za utendaji wa modeli, ambazo hazitoshi kuchambua jinsi modeli inavyokiuka kanuni za AI inayowajibika. Zaidi ya hayo, modeli ya kujifunza kwa mashine ni "sanduku jeusi" ambalo hufanya iwe vigumu kuelewa kinachosababisha matokeo yake au kutoa maelezo pale inapokosea. Baadaye katika kozi hii, tutajifunza jinsi ya kutumia dashibodi ya AI Inayowajibika kusaidia kutatua hitilafu katika mifumo ya AI. Dashibodi hutoa zana ya jumla kwa wanasayansi wa data na wasanidi wa AI kufanya:

* **Uchambuzi wa makosa**. Kutambua usambazaji wa makosa ya modeli ambayo yanaweza kuathiri haki au uaminifu wa mfumo.
* **Muhtasari wa modeli**. Kugundua mahali kuna tofauti katika utendaji wa modeli katika vikundi vya data.
* **Uchambuzi wa data**. Kuelewa usambazaji wa data na kutambua upendeleo wowote unaoweza kuwapo katika data ambao unaweza kusababisha masuala ya haki, ujumuishi, na uaminifu.
* **Ufafanuzi wa modeli**. Kuelewa kinachoshawishi au kuathiri utabiri wa modeli. Hii husaidia kuelezea tabia ya modeli, ambayo ni muhimu kwa uwazi na uwajibikaji.

## 🚀 Changamoto

Ili kuzuia madhara yasiletwe mwanzoni, tunapaswa:

- kuwa na utofauti wa asili na mitazamo miongoni mwa watu wanaofanya kazi kwenye mifumo
- kuwekeza katika seti za data zinazowakilisha utofauti wa jamii yetu
- kuendeleza mbinu bora katika mzunguko wa maisha wa kujifunza kwa mashine kwa kugundua na kurekebisha AI inayowajibika pale inapojitokeza

Fikiria hali halisi ambapo kutokuwa na uaminifu wa modeli kunadhihirika katika ujenzi na matumizi ya modeli. Nini kingine tunapaswa kuzingatia?

## [Maswali ya baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapit
Tazama warsha hii ili kuchunguza kwa undani mada hizi:

- Katika harakati za AI yenye uwajibikaji: Kuweka kanuni katika vitendo na Besmira Nushi, Mehrnoosh Sameki, na Amit Sharma

[![Responsible AI Toolbox: Mfumo wa chanzo huria kwa ajili ya kujenga AI yenye uwajibikaji](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Mfumo wa chanzo huria kwa ajili ya kujenga AI yenye uwajibikaji")

> 🎥 Bonyeza picha hapo juu kwa video: RAI Toolbox: Mfumo wa chanzo huria kwa ajili ya kujenga AI yenye uwajibikaji na Besmira Nushi, Mehrnoosh Sameki, na Amit Sharma

Pia, soma:

- Kituo cha rasilimali cha RAI cha Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Kikundi cha utafiti cha FATE cha Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Hifadhi ya GitHub ya Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Soma kuhusu zana za Azure Machine Learning kuhakikisha usawa:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Kazi

[Chunguza RAI Toolbox](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.