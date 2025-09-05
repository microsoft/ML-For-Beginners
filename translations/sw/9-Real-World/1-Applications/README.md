<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T15:49:29+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "sw"
}
-->
# Postscript: Kujifunza mashine katika ulimwengu halisi

![Muhtasari wa kujifunza mashine katika ulimwengu halisi katika sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote na [Tomomi Imura](https://www.twitter.com/girlie_mac)

Katika mtaala huu, umejifunza njia nyingi za kuandaa data kwa mafunzo na kuunda mifano ya kujifunza mashine. Umejenga mfululizo wa mifano ya regression ya kawaida, clustering, classification, usindikaji wa lugha asilia, na mfululizo wa muda. Hongera! Sasa, unaweza kuwa unajiuliza yote haya ni kwa ajili ya nini... ni matumizi gani ya ulimwengu halisi kwa mifano hii?

Ingawa AI imevutia sana sekta mbalimbali, ambayo mara nyingi hutumia kujifunza kwa kina, bado kuna matumizi muhimu kwa mifano ya kujifunza mashine ya kawaida. Huenda hata unatumia baadhi ya matumizi haya leo! Katika somo hili, utachunguza jinsi sekta nane tofauti na nyanja za maarifa zinavyotumia aina hizi za mifano kuboresha utendaji, uaminifu, akili, na thamani kwa watumiaji.

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Fedha

Sekta ya fedha inatoa fursa nyingi kwa kujifunza mashine. Matatizo mengi katika eneo hili yanaweza kuigwa na kutatuliwa kwa kutumia ML.

### Kugundua udanganyifu wa kadi ya mkopo

Tulijifunza kuhusu [k-means clustering](../../5-Clustering/2-K-Means/README.md) mapema katika kozi, lakini inaweza kutumika vipi kutatua matatizo yanayohusiana na udanganyifu wa kadi ya mkopo?

K-means clustering ni muhimu katika mbinu ya kugundua udanganyifu wa kadi ya mkopo inayoitwa **outlier detection**. Outliers, au mabadiliko katika uchunguzi wa seti ya data, yanaweza kutuonyesha ikiwa kadi ya mkopo inatumiwa kwa kawaida au kuna kitu kisicho cha kawaida kinachoendelea. Kama inavyoonyeshwa katika karatasi iliyounganishwa hapa chini, unaweza kupanga data ya kadi ya mkopo kwa kutumia algorithm ya k-means clustering na kuainisha kila muamala katika kundi kulingana na jinsi inavyoonekana kuwa outlier. Kisha, unaweza kutathmini makundi yenye hatari zaidi kwa muamala wa udanganyifu dhidi ya halali.
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Usimamizi wa mali

Katika usimamizi wa mali, mtu binafsi au kampuni hushughulikia uwekezaji kwa niaba ya wateja wao. Kazi yao ni kudumisha na kukuza mali kwa muda mrefu, kwa hivyo ni muhimu kuchagua uwekezaji unaofanya vizuri.

Njia moja ya kutathmini jinsi uwekezaji fulani unavyofanya ni kupitia regression ya takwimu. [Linear regression](../../2-Regression/1-Tools/README.md) ni zana muhimu kwa kuelewa jinsi mfuko unavyofanya kazi ikilinganishwa na kiwango fulani. Tunaweza pia kuamua ikiwa matokeo ya regression ni muhimu kwa takwimu, au jinsi yanavyoweza kuathiri uwekezaji wa mteja. Unaweza hata kupanua uchambuzi wako zaidi kwa kutumia regression nyingi, ambapo sababu za ziada za hatari zinaweza kuzingatiwa. Kwa mfano wa jinsi hii ingeweza kufanya kazi kwa mfuko maalum, angalia karatasi hapa chini kuhusu kutathmini utendaji wa mfuko kwa kutumia regression.
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Elimu

Sekta ya elimu pia ni eneo la kuvutia ambapo ML inaweza kutumika. Kuna matatizo ya kuvutia ya kushughulikia kama vile kugundua udanganyifu kwenye mitihani au insha au kudhibiti upendeleo, wa makusudi au la, katika mchakato wa kusahihisha.

### Kutabiri tabia ya wanafunzi

[Coursera](https://coursera.com), mtoa kozi za mtandaoni, ana blogu nzuri ya teknolojia ambapo wanajadili maamuzi mengi ya uhandisi. Katika utafiti huu wa kesi, walichora mstari wa regression kujaribu kuchunguza uhusiano wowote kati ya alama ya chini ya NPS (Net Promoter Score) na uhifadhi wa kozi au kuacha.
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Kupunguza upendeleo

[Grammarly](https://grammarly.com), msaidizi wa uandishi unaochunguza makosa ya tahajia na sarufi, hutumia mifumo ya kisasa ya [usindikaji wa lugha asilia](../../6-NLP/README.md) katika bidhaa zake. Walichapisha utafiti wa kesi wa kuvutia katika blogu yao ya teknolojia kuhusu jinsi walivyoshughulikia upendeleo wa kijinsia katika kujifunza mashine, ambayo ulijifunza katika [somo letu la haki la utangulizi](../../1-Introduction/3-fairness/README.md).
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Biashara ya rejareja

Sekta ya rejareja inaweza kufaidika sana na matumizi ya ML, kuanzia kuboresha safari ya mteja hadi kusimamia hesabu kwa njia bora.

### Kubinafsisha safari ya mteja

Katika Wayfair, kampuni inayouza bidhaa za nyumbani kama samani, kuwasaidia wateja kupata bidhaa sahihi kwa ladha na mahitaji yao ni jambo la msingi. Katika makala hii, wahandisi kutoka kampuni hiyo wanaelezea jinsi wanavyotumia ML na NLP "kuonyesha matokeo sahihi kwa wateja". Hasa, Injini yao ya Query Intent imejengwa kutumia uchimbaji wa entiti, mafunzo ya classifier, uchimbaji wa mali na maoni, na uwekaji wa hisia kwenye maoni ya wateja. Hii ni mfano wa kawaida wa jinsi NLP inavyofanya kazi katika rejareja mtandaoni.
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Usimamizi wa hesabu

Kampuni za ubunifu na zenye wepesi kama [StitchFix](https://stitchfix.com), huduma ya sanduku inayosafirisha mavazi kwa watumiaji, hutegemea sana ML kwa mapendekezo na usimamizi wa hesabu. Timu zao za mitindo hufanya kazi pamoja na timu zao za biashara, kwa kweli: "mmoja wa wanasayansi wetu wa data alijaribu algorithm ya maumbile na kuitekeleza kwa mavazi kutabiri ni kipande gani cha mavazi kitakuwa na mafanikio ambacho hakipo leo. Tulileta hilo kwa timu ya biashara na sasa wanaweza kutumia hilo kama zana."
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Huduma ya Afya

Sekta ya huduma ya afya inaweza kutumia ML kuboresha kazi za utafiti na pia matatizo ya kimuundo kama kurudisha wagonjwa hospitalini au kuzuia magonjwa kuenea.

### Usimamizi wa majaribio ya kliniki

Sumu katika majaribio ya kliniki ni wasiwasi mkubwa kwa watengenezaji wa dawa. Kiasi gani cha sumu kinachokubalika? Katika utafiti huu, kuchambua mbinu mbalimbali za majaribio ya kliniki kulisababisha maendeleo ya mbinu mpya ya kutabiri uwezekano wa matokeo ya majaribio ya kliniki. Hasa, waliweza kutumia random forest kuzalisha [classifier](../../4-Classification/README.md) inayoweza kutofautisha kati ya vikundi vya dawa.
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Usimamizi wa kurudishwa hospitalini

Huduma ya hospitali ni ghali, hasa wakati wagonjwa wanapaswa kurudishwa. Karatasi hii inajadili kampuni inayotumia ML kutabiri uwezekano wa kurudishwa kwa kutumia [clustering](../../5-Clustering/README.md) algorithms. Makundi haya husaidia wachambuzi "kugundua vikundi vya kurudishwa ambavyo vinaweza kushiriki sababu ya kawaida".
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Usimamizi wa magonjwa

Janga la hivi karibuni limeonyesha wazi jinsi kujifunza mashine kunavyoweza kusaidia kuzuia kuenea kwa magonjwa. Katika makala hii, utatambua matumizi ya ARIMA, logistic curves, linear regression, na SARIMA. "Kazi hii ni jaribio la kuhesabu kiwango cha kuenea kwa virusi hivi na hivyo kutabiri vifo, kupona, na kesi zilizothibitishwa, ili iweze kutusaidia kujiandaa vizuri na kuishi."
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ikolojia na Teknolojia ya Kijani

Asili na ikolojia inajumuisha mifumo mingi nyeti ambapo mwingiliano kati ya wanyama na asili unazingatiwa. Ni muhimu kuweza kupima mifumo hii kwa usahihi na kuchukua hatua ipasavyo ikiwa kitu kinatokea, kama moto wa msitu au kupungua kwa idadi ya wanyama.

### Usimamizi wa misitu

Umejifunza kuhusu [Reinforcement Learning](../../8-Reinforcement/README.md) katika masomo ya awali. Inaweza kuwa muhimu sana wakati wa kujaribu kutabiri mifumo katika asili. Hasa, inaweza kutumika kufuatilia matatizo ya ikolojia kama moto wa msitu na kuenea kwa spishi vamizi. Nchini Kanada, kikundi cha watafiti kilitumia Reinforcement Learning kujenga mifano ya mienendo ya moto wa msitu kutoka picha za satelaiti. Kwa kutumia "spatially spreading process (SSP)" ya ubunifu, waliona moto wa msitu kama "wakala katika seli yoyote kwenye mandhari." "Seti ya hatua ambazo moto unaweza kuchukua kutoka eneo lolote kwa wakati wowote ni pamoja na kuenea kaskazini, kusini, mashariki, au magharibi au kutokuenea.

Mbinu hii inageuza mpangilio wa kawaida wa RL kwani mienendo ya MDP inayolingana ni kazi inayojulikana kwa kuenea kwa moto mara moja." Soma zaidi kuhusu algorithms za kawaida zinazotumiwa na kikundi hiki kwenye kiungo hapa chini.
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Ufuatiliaji wa harakati za wanyama

Ingawa kujifunza kwa kina kumeleta mapinduzi katika kufuatilia harakati za wanyama kwa kuona (unaweza kujenga [kifuatiliaji cha dubu wa polar](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) hapa), ML ya kawaida bado ina nafasi katika kazi hii.

Vihisi vya kufuatilia harakati za wanyama wa shambani na IoT hutumia aina hii ya usindikaji wa kuona, lakini mbinu za msingi za ML ni muhimu kwa kuandaa data. Kwa mfano, katika karatasi hii, mikao ya kondoo zilifuatiliwa na kuchambuliwa kwa kutumia algorithms mbalimbali za classifier. Unaweza kutambua ROC curve kwenye ukurasa wa 335.
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Usimamizi wa Nishati

Katika masomo yetu ya [utabiri wa mfululizo wa muda](../../7-TimeSeries/README.md), tulitaja dhana ya mita za maegesho za kisasa ili kuzalisha mapato kwa mji kwa msingi wa kuelewa usambazaji na mahitaji. Makala hii inajadili kwa undani jinsi clustering, regression na utabiri wa mfululizo wa muda vilivyotumika pamoja kusaidia kutabiri matumizi ya nishati ya baadaye nchini Ireland, kwa msingi wa mita za kisasa.
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Bima

Sekta ya bima ni sekta nyingine inayotumia ML kujenga na kuboresha mifano ya kifedha na actuarial.

### Usimamizi wa Mabadiliko

MetLife, mtoa huduma wa bima ya maisha, ni wazi kuhusu jinsi wanavyotathmini na kupunguza mabadiliko katika mifano yao ya kifedha. Katika makala hii utaona taswira za binary na ordinal classification. Pia utagundua taswira za utabiri.
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Sanaa, Utamaduni, na Fasihi

Katika sanaa, kwa mfano katika uandishi wa habari, kuna matatizo mengi ya kuvutia. Kugundua habari za uongo ni tatizo kubwa kwani imethibitishwa kuathiri maoni ya watu na hata kupindua demokrasia. Makumbusho pia yanaweza kufaidika kwa kutumia ML katika kila kitu kuanzia kutafuta viungo kati ya vitu hadi upangaji wa rasilimali.

### Kugundua habari za uongo

Kugundua habari za uongo kumegeuka kuwa mchezo wa paka na panya katika vyombo vya habari vya leo. Katika makala hii, watafiti wanapendekeza kwamba mfumo unaochanganya mbinu kadhaa za ML tulizojifunza unaweza kujaribiwa na mfano bora kutekelezwa: "Mfumo huu unategemea usindikaji wa lugha asilia kutoa vipengele kutoka kwa data na kisha vipengele hivi vinatumika kwa mafunzo ya classifiers za kujifunza mashine kama Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), na Logistic Regression (LR)."
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Makala hii inaonyesha jinsi kuchanganya nyanja tofauti za ML kunaweza kutoa matokeo ya kuvutia ambayo yanaweza kusaidia kuzuia habari za uongo kuenea na kuleta madhara halisi; katika kesi hii, msukumo ulikuwa kuenea kwa uvumi kuhusu matibabu ya COVID ambayo yalisababisha vurugu za umati.

### ML ya Makumbusho

Makumbusho yako katika ukingo wa mapinduzi ya AI ambapo kuorodhesha na kudijiti mkusanyiko na kutafuta viungo kati ya vitu vinakuwa rahisi kadri teknolojia inavyosonga mbele. Miradi kama [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) inasaidia kufungua siri za makusanyo yasiyofikika kama vile Hifadhi ya Vatican. Lakini, kipengele cha biashara cha makumbusho kinanufaika na mifano ya ML pia.

Kwa mfano, Taasisi ya Sanaa ya Chicago ilijenga mifano ya kutabiri kile ambacho watazamaji wanapendezwa nacho na wakati watatembelea maonyesho. Lengo ni kuunda uzoefu wa wageni uliobinafsishwa na ulioboreshwa kila wakati mtumiaji anapotembelea makumbusho. "Wakati wa mwaka wa fedha wa 2017, mfano ulitabiri mahudhurio na mapato kwa usahihi wa asilimia 1, anasema Andrew Simnick, makamu wa rais mwandamizi katika Taasisi ya Sanaa."
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Masoko

### Ugawaji wa wateja

Mikakati bora zaidi ya masoko inalenga wateja kwa njia tofauti kulingana na makundi mbalimbali. Katika makala hii, matumizi ya algorithms za Clustering yanajadiliwa kusaidia masoko tofauti. Masoko tofauti husaidia kampuni kuboresha utambuzi wa chapa, kufikia wateja zaidi, na kupata pesa zaidi.
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Changamoto

Tambua sekta nyingine inayofaidika na baadhi ya mbinu ulizojifunza katika mtaala huu, na ugundue jinsi inavyotumia ML.
## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Timu ya sayansi ya data ya Wayfair ina video kadhaa za kuvutia kuhusu jinsi wanavyotumia ML katika kampuni yao. Inafaa [kuangalia](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Kazi

[Utafutaji wa ML](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.