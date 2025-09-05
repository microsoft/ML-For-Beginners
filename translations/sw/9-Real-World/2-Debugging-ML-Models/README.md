<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T15:53:25+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "sw"
}
-->
# Postscript: Uchanganuzi wa Modeli ya Kujifunza kwa Mashine kwa kutumia Vipengele vya Dashibodi ya AI Inayowajibika

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Utangulizi

Kujifunza kwa mashine kunaathiri maisha yetu ya kila siku. AI inajipenyeza katika baadhi ya mifumo muhimu zaidi inayotugusa sisi kama watu binafsi na jamii yetu, kama vile afya, fedha, elimu, na ajira. Kwa mfano, mifumo na modeli zinahusika katika kazi za maamuzi ya kila siku, kama vile utambuzi wa magonjwa au kugundua udanganyifu. Kwa sababu ya maendeleo ya AI na kasi ya matumizi yake, matarajio ya kijamii yanabadilika na kanuni zinakua. Tunaendelea kuona maeneo ambapo mifumo ya AI inakosa matarajio; inafichua changamoto mpya; na serikali zinaanza kudhibiti suluhisho za AI. Kwa hivyo, ni muhimu kwamba modeli hizi zichunguzwe ili kutoa matokeo ya haki, ya kuaminika, ya kujumuisha, ya uwazi, na yenye uwajibikaji kwa kila mtu.

Katika mtaala huu, tutachunguza zana za vitendo zinazoweza kutumika kutathmini ikiwa modeli ina masuala ya AI inayowajibika. Mbinu za jadi za uchanganuzi wa modeli za kujifunza kwa mashine mara nyingi zinategemea hesabu za kiasi kama vile usahihi wa jumla au hasara ya makosa ya wastani. Fikiria kinachoweza kutokea wakati data unayotumia kujenga modeli hizi inakosa baadhi ya demografia, kama vile rangi, jinsia, mtazamo wa kisiasa, dini, au inawakilisha demografia hizo kwa uwiano usio sawa. Je, kuhusu pale ambapo matokeo ya modeli yanatafsiriwa kupendelea demografia fulani? Hii inaweza kuanzisha uwakilishi wa kupita kiasi au wa chini wa makundi haya nyeti, na kusababisha masuala ya haki, ujumuishaji, au uaminifu kutoka kwa modeli. Sababu nyingine ni kwamba modeli za kujifunza kwa mashine zinachukuliwa kuwa "masanduku meusi," jambo linalofanya iwe vigumu kuelewa na kueleza kinachosababisha utabiri wa modeli. Haya yote ni changamoto ambazo wanasayansi wa data na watengenezaji wa AI wanakabiliana nazo wanapokosa zana za kutosha za kuchunguza na kutathmini haki au uaminifu wa modeli.

Katika somo hili, utajifunza kuhusu kuchunguza modeli zako kwa kutumia:

- **Uchanganuzi wa Makosa**: Tambua maeneo katika usambazaji wa data yako ambapo modeli ina viwango vya juu vya makosa.
- **Muhtasari wa Modeli**: Fanya uchanganuzi wa kulinganisha kati ya makundi tofauti ya data ili kugundua tofauti katika vipimo vya utendaji wa modeli yako.
- **Uchanganuzi wa Data**: Chunguza maeneo ambapo kuna uwakilishi wa kupita kiasi au wa chini wa data yako, jambo linaloweza kupotosha modeli yako kupendelea demografia moja dhidi ya nyingine.
- **Umuhimu wa Vipengele**: Elewa ni vipengele gani vinavyoendesha utabiri wa modeli yako kwa kiwango cha jumla au cha ndani.

## Mahitaji ya awali

Kama mahitaji ya awali, tafadhali pitia [Zana za AI Inayowajibika kwa watengenezaji](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif kuhusu Zana za AI Inayowajibika](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Uchanganuzi wa Makosa

Vipimo vya jadi vya utendaji wa modeli vinavyotumika kupima usahihi mara nyingi ni hesabu zinazotegemea utabiri sahihi dhidi ya usio sahihi. Kwa mfano, kuamua kwamba modeli ni sahihi kwa 89% ya muda na ina hasara ya makosa ya 0.001 inaweza kuchukuliwa kuwa utendaji mzuri. Makosa mara nyingi hayajasambazwa kwa usawa katika seti yako ya data. Unaweza kupata alama ya usahihi ya 89% ya modeli lakini kugundua kwamba kuna maeneo tofauti ya data yako ambapo modeli inashindwa kwa 42% ya muda. Matokeo ya mifumo hii ya kushindwa na makundi fulani ya data yanaweza kusababisha masuala ya haki au uaminifu. Ni muhimu kuelewa maeneo ambapo modeli inafanya vizuri au vibaya. Maeneo ya data ambapo kuna idadi kubwa ya makosa katika modeli yako yanaweza kuwa demografia muhimu ya data.

![Changanua na chunguza makosa ya modeli](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Sehemu ya Uchanganuzi wa Makosa kwenye dashibodi ya RAI inaonyesha jinsi kushindwa kwa modeli kunavyosambazwa katika makundi mbalimbali kwa kutumia taswira ya mti. Hii ni muhimu katika kutambua vipengele au maeneo ambapo kuna kiwango cha juu cha makosa katika seti yako ya data. Kwa kuona mahali ambapo makosa mengi ya modeli yanatoka, unaweza kuanza kuchunguza chanzo chake. Unaweza pia kuunda makundi ya data ili kufanya uchanganuzi. Makundi haya ya data husaidia katika mchakato wa uchanganuzi ili kuamua kwa nini utendaji wa modeli ni mzuri katika kundi moja lakini una makosa katika jingine.

![Uchanganuzi wa Makosa](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Viashiria vya taswira kwenye ramani ya mti husaidia katika kutambua maeneo yenye matatizo haraka. Kwa mfano, kivuli cha rangi nyekundu kilicho giza zaidi kwenye nodi ya mti kinaonyesha kiwango cha juu cha makosa.

Ramani ya joto ni kipengele kingine cha taswira ambacho watumiaji wanaweza kutumia kuchunguza kiwango cha makosa kwa kutumia kipengele kimoja au viwili ili kupata mchango wa makosa ya modeli katika seti nzima ya data au makundi.

![Ramani ya joto ya Uchanganuzi wa Makosa](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Tumia uchanganuzi wa makosa unapohitaji:

* Kupata uelewa wa kina wa jinsi kushindwa kwa modeli kunavyosambazwa katika seti ya data na vipengele kadhaa vya ingizo.
* Kuvunja vipimo vya utendaji wa jumla ili kugundua makundi yenye makosa kwa njia ya kiotomatiki ili kuarifu hatua zako za kupunguza.

## Muhtasari wa Modeli

Kutathmini utendaji wa modeli ya kujifunza kwa mashine kunahitaji kupata uelewa wa jumla wa tabia yake. Hii inaweza kufanyika kwa kupitia zaidi ya kipimo kimoja kama vile kiwango cha makosa, usahihi, kumbukumbu, usahihi, au MAE (Makosa ya Wastani ya Kawaida) ili kupata tofauti kati ya vipimo vya utendaji. Kipimo kimoja cha utendaji kinaweza kuonekana kizuri, lakini makosa yanaweza kufichuliwa katika kipimo kingine. Aidha, kulinganisha vipimo kwa tofauti katika seti nzima ya data au makundi husaidia kufichua maeneo ambapo modeli inafanya vizuri au vibaya. Hii ni muhimu hasa katika kuona utendaji wa modeli kati ya vipengele nyeti dhidi ya visivyo nyeti (mfano, rangi ya mgonjwa, jinsia, au umri) ili kufichua uwezekano wa kutokuwa na haki ambao modeli inaweza kuwa nao. Kwa mfano, kugundua kwamba modeli ina makosa zaidi katika kundi lenye vipengele nyeti kunaweza kufichua uwezekano wa kutokuwa na haki ambao modeli inaweza kuwa nao.

Sehemu ya Muhtasari wa Modeli kwenye dashibodi ya RAI husaidia si tu katika kuchanganua vipimo vya utendaji wa uwakilishi wa data katika kundi, lakini pia inawapa watumiaji uwezo wa kulinganisha tabia ya modeli katika makundi tofauti.

![Makundi ya seti ya data - muhtasari wa modeli kwenye dashibodi ya RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Kipengele cha uchanganuzi wa msingi wa vipengele cha sehemu hii kinawaruhusu watumiaji kupunguza makundi ya data ndani ya kipengele fulani ili kutambua kasoro kwa kiwango cha kina. Kwa mfano, dashibodi ina akili ya kujengwa ndani ya kiotomatiki ya kuunda makundi kwa kipengele kilichochaguliwa na mtumiaji (mfano, *"time_in_hospital < 3"* au *"time_in_hospital >= 7"*). Hii inamwezesha mtumiaji kutenga kipengele fulani kutoka kwa kundi kubwa la data ili kuona ikiwa ni mshawishi muhimu wa matokeo yenye makosa ya modeli.

![Makundi ya vipengele - muhtasari wa modeli kwenye dashibodi ya RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Sehemu ya Muhtasari wa Modeli inaunga mkono aina mbili za vipimo vya tofauti:

**Tofauti katika utendaji wa modeli**: Vipimo hivi vinahesabu tofauti (tofauti) katika thamani za kipimo cha utendaji kilichochaguliwa katika makundi ya data. Hapa kuna mifano michache:

* Tofauti katika kiwango cha usahihi
* Tofauti katika kiwango cha makosa
* Tofauti katika usahihi
* Tofauti katika kumbukumbu
* Tofauti katika makosa ya wastani ya kawaida (MAE)

**Tofauti katika kiwango cha uteuzi**: Kipimo hiki kina tofauti katika kiwango cha uteuzi (utabiri mzuri) kati ya makundi. Mfano wa hili ni tofauti katika viwango vya idhini ya mkopo. Kiwango cha uteuzi kinamaanisha sehemu ya alama za data katika kila darasa zilizoorodheshwa kama 1 (katika uainishaji wa binary) au usambazaji wa thamani za utabiri (katika regression).

## Uchanganuzi wa Data

> "Ukitesa data vya kutosha, itakiri chochote" - Ronald Coase

Kauli hii inaonekana kali, lakini ni kweli kwamba data inaweza kudanganywa ili kuunga mkono hitimisho lolote. Udanganyifu kama huo wakati mwingine unaweza kutokea bila kukusudia. Kama binadamu, sote tuna upendeleo, na mara nyingi ni vigumu kujua kwa makusudi wakati unaleta upendeleo katika data. Kuhakikisha haki katika AI na kujifunza kwa mashine bado ni changamoto ngumu.

Data ni eneo kubwa la upofu kwa vipimo vya jadi vya utendaji wa modeli. Unaweza kuwa na alama za usahihi wa juu, lakini hii haionyeshi kila mara upendeleo wa msingi wa data ambao unaweza kuwa katika seti yako ya data. Kwa mfano, ikiwa seti ya data ya wafanyakazi ina 27% ya wanawake katika nafasi za utendaji katika kampuni na 73% ya wanaume katika kiwango sawa, modeli ya AI ya kutangaza kazi iliyofunzwa kwenye data hii inaweza kulenga zaidi hadhira ya kiume kwa nafasi za kazi za ngazi ya juu. Kuwa na usawa huu katika data kulipotosha utabiri wa modeli kupendelea jinsia moja. Hii inaonyesha suala la haki ambapo kuna upendeleo wa kijinsia katika modeli ya AI.

Sehemu ya Uchanganuzi wa Data kwenye dashibodi ya RAI husaidia kutambua maeneo ambapo kuna uwakilishi wa kupita kiasi au wa chini katika seti ya data. Inasaidia watumiaji kugundua chanzo cha makosa na masuala ya haki yanayoletwa na usawa wa data au ukosefu wa uwakilishi wa kundi fulani la data. Hii inawapa watumiaji uwezo wa kutazama seti za data kulingana na matokeo yaliyotabiriwa na halisi, makundi ya makosa, na vipengele maalum. Wakati mwingine kugundua kundi la data lisilowakilishwa vizuri kunaweza pia kufichua kwamba modeli haijifunzi vizuri, hivyo makosa ya juu. Kuwa na modeli yenye upendeleo wa data si tu suala la haki bali inaonyesha kwamba modeli si ya kujumuisha au ya kuaminika.

![Sehemu ya Uchanganuzi wa Data kwenye Dashibodi ya RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Tumia uchanganuzi wa data unapohitaji:

* Kuchunguza takwimu za seti yako ya data kwa kuchagua vichujio tofauti ili kugawanya data yako katika vipimo tofauti (pia vinajulikana kama makundi).
* Kuelewa usambazaji wa seti yako ya data katika makundi tofauti na vikundi vya vipengele.
* Kuamua ikiwa matokeo yako yanayohusiana na haki, uchanganuzi wa makosa, na usababishi (yanayotokana na vipengele vingine vya dashibodi) ni matokeo ya usambazaji wa seti yako ya data.
* Kuamua maeneo ya kukusanya data zaidi ili kupunguza makosa yanayotokana na masuala ya uwakilishi, kelele za lebo, kelele za vipengele, upendeleo wa lebo, na mambo yanayofanana.

## Ufafanuzi wa Modeli

Modeli za kujifunza kwa mashine mara nyingi huchukuliwa kuwa masanduku meusi. Kuelewa ni vipengele gani muhimu vya data vinavyoendesha utabiri wa modeli inaweza kuwa changamoto. Ni muhimu kutoa uwazi kuhusu kwa nini modeli inatoa utabiri fulani. Kwa mfano, ikiwa mfumo wa AI unatabiri kwamba mgonjwa mwenye kisukari yuko katika hatari ya kurudi hospitalini ndani ya siku 30, unapaswa kuwa na uwezo wa kutoa data inayounga mkono utabiri wake. Kuwa na viashiria vya data vinavyounga mkono huleta uwazi ili kusaidia madaktari au hospitali kufanya maamuzi yaliyojengwa vizuri. Aidha, kuwa na uwezo wa kueleza kwa nini modeli ilitoa utabiri kwa mgonjwa mmoja mmoja huwezesha uwajibikaji kwa kanuni za afya. Unapotumia modeli za kujifunza kwa mashine kwa njia zinazogusa maisha ya watu, ni muhimu kuelewa na kueleza kinachosababisha tabia ya modeli. Ufafanuzi na uelewa wa modeli husaidia kujibu maswali katika hali kama:

* Uchanganuzi wa modeli: Kwa nini modeli yangu ilifanya kosa hili? Ninawezaje kuboresha modeli yangu?
* Ushirikiano wa binadamu na AI: Ninawezaje kuelewa na kuamini maamuzi ya modeli?
* Uzingatiaji wa kanuni: Je, modeli yangu inakidhi mahitaji ya kisheria?

Sehemu ya Umuhimu wa Vipengele kwenye dashibodi ya RAI inakusaidia kuchunguza na kupata uelewa wa kina wa jinsi modeli inavyotoa utabiri. Pia ni zana muhimu kwa wataalamu wa kujifunza kwa mashine na watoa maamuzi kueleza na kuonyesha ushahidi wa vipengele vinavyoathiri tabia ya modeli kwa kufuata kanuni. Watumiaji wanaweza kuchunguza maelezo ya jumla na ya ndani ili kuthibitisha ni vipengele gani vinavyoendesha utabiri wa modeli. Maelezo ya jumla yanaorodhesha vipengele vya juu vilivyoathiri utabiri wa jumla wa modeli. Maelezo ya ndani yanaonyesha ni vipengele gani vilisababisha utabiri wa modeli kwa kesi moja. Uwezo wa kutathmini maelezo ya ndani pia ni muhimu katika kuchunguza au kukagua kesi maalum ili kuelewa na kufafanua kwa nini modeli ilitoa utabiri sahihi au usio sahihi.

![Sehemu ya Umuhimu wa Vipengele kwenye Dashibodi ya RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Maelezo ya jumla: Kwa mfano, ni vipengele gani vinavyoathiri tabia ya jumla ya modeli ya kurudi hospitalini kwa wagonjwa wenye kisukari?
* Maelezo ya ndani: Kwa mfano, kwa nini mgonjwa mwenye kisukari mwenye umri wa zaidi ya miaka 60 na aliye na historia ya kulazwa hospitalini alitabiriwa kurudi au kutorudi hospitalini ndani ya siku 30?

Katika mchakato wa kuchunguza utendaji wa modeli katika makundi tofauti, Umuhimu wa Vipengele unaonyesha kiwango cha athari kipengele kinacholeta katika makundi hayo. Husaidia kufichua kasoro wakati wa kulinganisha kiwango cha ushawishi kipengele kinacholeta katika kuendesha utabiri wenye makosa wa modeli. Sehemu ya Umuhimu wa Vipengele inaweza kuonyesha ni thamani gani katika kipengele zilizoathiri kwa njia chanya au hasi matokeo ya modeli. Kwa mfano, ikiwa modeli ilitoa utabiri usio sahihi, sehemu hii inakupa uwezo wa kuchunguza kwa kina na kubaini ni vipengele au thamani za vipengele vilivyoendesha utabiri huo. Kiwango hiki cha maelezo husaidia si tu katika uchanganuzi bali pia hutoa uwazi na uwajibikaji katika hali za ukaguzi. Hatimaye, sehemu hii inaweza kukusaidia kutambua masuala ya haki. Kwa mfano, ikiwa kipengele nyeti kama vile kabila au jinsia kina ushawishi mkubwa katika kuendesha utabiri wa modeli, hii inaweza kuwa ishara ya upendeleo wa rangi au jinsia katika modeli.

![Umuhimu wa vipengele](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Tumia uelewa wa modeli unapohitaji:

* Kuamua jinsi utabiri wa mfumo wako wa AI unavyoweza kuaminika kwa kuelewa ni vipengele gani vilivyo muhimu zaidi kwa utabiri.
* Kukaribia uchanganuzi wa modeli yako kwa kuielewa kwanza na kutambua ikiwa modeli inatumia vipengele vyenye afya au tu uhusiano wa uwongo.
* Kufichua vyanzo vya uwezekano wa kutokuwa na haki kwa kuelewa ikiwa modeli inategemea vipengele nyeti au vipengele vinavyohusiana sana navyo.
* Kujenga uaminifu wa mtumiaji katika maamuzi ya modeli yako kwa kutoa maelezo ya ndani ili kuonyesha matokeo yake.
* Kukamilisha ukaguzi wa kanuni wa mfumo wa AI ili kuthibitisha modeli na kufuatilia athari za maamuzi ya modeli kwa binadamu.

## Hitimisho

Vipengele vyote vya dashibodi ya RAI ni zana za vitendo zinazokusaidia kujenga modeli za kujifunza kwa mashine ambazo ni salama zaidi na zinazoaminika
- **Uwiano wa kupita kiasi au wa chini**. Wazo ni kwamba kundi fulani halionekani katika taaluma fulani, na huduma au kazi yoyote inayozidi kuendeleza hali hiyo inachangia madhara.

### Dashibodi ya Azure RAI

[Dashibodi ya Azure RAI](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) imejengwa kwa kutumia zana za chanzo huria zilizotengenezwa na taasisi za kitaaluma na mashirika yanayoongoza, ikiwemo Microsoft. Zana hizi ni muhimu kwa wanasayansi wa data na watengenezaji wa AI ili kuelewa vyema tabia ya modeli, kugundua na kupunguza masuala yasiyofaa kutoka kwa modeli za AI.

- Jifunze jinsi ya kutumia vipengele tofauti kwa kuangalia [hati za dashibodi ya RAI.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Angalia baadhi ya [notibuku za mfano za dashibodi ya RAI](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) kwa ajili ya kutatua changamoto za AI inayowajibika zaidi katika Azure Machine Learning.

---
## 🚀 Changamoto

Ili kuzuia upendeleo wa takwimu au data kuingizwa tangu mwanzo, tunapaswa:

- kuwa na utofauti wa asili na mitazamo miongoni mwa watu wanaofanya kazi kwenye mifumo
- kuwekeza katika seti za data zinazowakilisha utofauti wa jamii yetu
- kuendeleza mbinu bora za kugundua na kurekebisha upendeleo unapojitokeza

Fikiria hali halisi ambapo kutokuwa sawa kunaonekana katika ujenzi na matumizi ya modeli. Ni mambo gani mengine tunapaswa kuzingatia?

## [Jaribio baada ya somo](https://ff-quizzes.netlify.app/en/ml/)
## Mapitio na Kujisomea

Katika somo hili, umejifunza baadhi ya zana za vitendo za kuingiza AI inayowajibika katika ujifunzaji wa mashine.

Tazama warsha hii ili kuchunguza zaidi mada hizi:

- Dashibodi ya AI Inayowajibika: Duka moja kwa kuendesha RAI kwa vitendo na Besmira Nushi na Mehrnoosh Sameki

[![Dashibodi ya AI Inayowajibika: Duka moja kwa kuendesha RAI kwa vitendo](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Dashibodi ya AI Inayowajibika: Duka moja kwa kuendesha RAI kwa vitendo na Besmira Nushi na Mehrnoosh Sameki")

> 🎥 Bofya picha hapo juu kwa video: Dashibodi ya AI Inayowajibika: Duka moja kwa kuendesha RAI kwa vitendo na Besmira Nushi na Mehrnoosh Sameki

Rejelea nyenzo zifuatazo ili kujifunza zaidi kuhusu AI inayowajibika na jinsi ya kujenga modeli zinazoweza kuaminika zaidi:

- Zana za dashibodi ya RAI za Microsoft kwa kutatua changamoto za modeli za ML: [Rasilimali za zana za AI inayowajibika](https://aka.ms/rai-dashboard)

- Chunguza zana za AI inayowajibika: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Kituo cha rasilimali cha RAI cha Microsoft: [Rasilimali za AI Inayowajibika – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Kikundi cha utafiti cha FATE cha Microsoft: [FATE: Uadilifu, Uwajibikaji, Uwazi, na Maadili katika AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Kazi

[Chunguza dashibodi ya RAI](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.