<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20f18ff565638be615df4174858e4a7f",
  "translation_date": "2025-08-29T13:31:08+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "tl"
}
-->
# Postscript: Machine learning sa totoong mundo

![Buod ng machine learning sa totoong mundo sa isang sketchnote](../../../../translated_images/ml-realworld.26ee2746716155771f8076598b6145e6533fe4a9e2e465ea745f46648cbf1b84.tl.png)  
> Sketchnote ni [Tomomi Imura](https://www.twitter.com/girlie_mac)

Sa kurikulum na ito, natutunan mo ang maraming paraan upang ihanda ang data para sa training at lumikha ng mga machine learning models. Gumawa ka ng serye ng mga klasikong regression, clustering, classification, natural language processing, at time series models. Binabati kita! Ngayon, maaaring iniisip mo kung para saan ang lahat ng ito... ano ang mga aplikasyon ng mga modelong ito sa totoong mundo?

Habang maraming interes sa industriya ang nakatuon sa AI, na karaniwang gumagamit ng deep learning, may mga mahalagang aplikasyon pa rin para sa mga klasikong machine learning models. Maaaring ginagamit mo na ang ilan sa mga aplikasyon na ito ngayon! Sa araling ito, tatalakayin natin kung paano ginagamit ng walong iba't ibang industriya at mga larangan ng kaalaman ang mga ganitong uri ng modelo upang gawing mas mahusay, maaasahan, matalino, at mahalaga ang kanilang mga aplikasyon para sa mga gumagamit.

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/)

## 💰 Pananalapi

Ang sektor ng pananalapi ay nag-aalok ng maraming oportunidad para sa machine learning. Maraming problema sa larangang ito ang maaaring i-modelo at lutasin gamit ang ML.

### Pagtuklas ng pandaraya sa credit card

Natuto tayo tungkol sa [k-means clustering](../../5-Clustering/2-K-Means/README.md) sa naunang bahagi ng kurso, ngunit paano ito magagamit upang lutasin ang mga problema kaugnay ng pandaraya sa credit card?

Ang k-means clustering ay kapaki-pakinabang sa isang teknik sa pagtuklas ng pandaraya sa credit card na tinatawag na **outlier detection**. Ang mga outlier, o mga paglihis sa obserbasyon ng isang set ng data, ay maaaring magpahiwatig kung ang isang credit card ay ginagamit sa normal na paraan o kung may kakaibang nangyayari. Gaya ng ipinakita sa papel na naka-link sa ibaba, maaari mong ayusin ang data ng credit card gamit ang isang k-means clustering algorithm at i-assign ang bawat transaksyon sa isang cluster batay sa kung gaano ito ka-outlier. Pagkatapos, maaari mong suriin ang mga pinaka-mapanganib na cluster para sa mga mapanlinlang kumpara sa mga lehitimong transaksyon.  
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Pamamahala ng yaman

Sa pamamahala ng yaman, ang isang indibidwal o kumpanya ay humahawak ng mga pamumuhunan para sa kanilang mga kliyente. Ang kanilang trabaho ay panatilihin at palaguin ang yaman sa pangmatagalan, kaya mahalaga ang pumili ng mga pamumuhunan na maganda ang performance.

Isang paraan upang suriin kung paano nagpe-perform ang isang partikular na pamumuhunan ay sa pamamagitan ng statistical regression. Ang [linear regression](../../2-Regression/1-Tools/README.md) ay isang mahalagang tool para maunawaan kung paano nagpe-perform ang isang pondo kumpara sa isang benchmark. Maaari rin nating matukoy kung ang mga resulta ng regression ay statistically significant, o kung gaano kalaki ang epekto nito sa mga pamumuhunan ng kliyente. Maaari mo pang palawakin ang iyong pagsusuri gamit ang multiple regression, kung saan maaaring isaalang-alang ang karagdagang mga risk factor. Para sa isang halimbawa kung paano ito gagana para sa isang partikular na pondo, tingnan ang papel sa ibaba tungkol sa pagsusuri ng performance ng pondo gamit ang regression.  
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Edukasyon

Ang sektor ng edukasyon ay isa ring napaka-interesanteng larangan kung saan maaaring gamitin ang ML. May mga kawili-wiling problema na maaaring lutasin tulad ng pagtuklas ng pandaraya sa mga pagsusulit o sanaysay o pamamahala ng bias, sinasadya man o hindi, sa proseso ng pagwawasto.

### Pagtataya ng pag-uugali ng mag-aaral

Ang [Coursera](https://coursera.com), isang online open course provider, ay may mahusay na tech blog kung saan nila tinatalakay ang maraming desisyon sa engineering. Sa case study na ito, nag-plot sila ng regression line upang subukang tuklasin ang anumang kaugnayan sa pagitan ng mababang NPS (Net Promoter Score) rating at pag-retain o pag-drop-off ng kurso.  
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Pagbawas ng bias

Ang [Grammarly](https://grammarly.com), isang writing assistant na nagche-check ng spelling at grammar errors, ay gumagamit ng mga sopistikadong [natural language processing systems](../../6-NLP/README.md) sa kanilang mga produkto. Nag-publish sila ng isang kawili-wiling case study sa kanilang tech blog tungkol sa kung paano nila hinarap ang gender bias sa machine learning, na natutunan mo sa aming [introductory fairness lesson](../../1-Introduction/3-fairness/README.md).  
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retail

Ang sektor ng retail ay tiyak na makikinabang mula sa paggamit ng ML, mula sa paglikha ng mas mahusay na customer journey hanggang sa optimal na pag-stock ng imbentaryo.

### Pag-personalize ng customer journey

Sa Wayfair, isang kumpanya na nagbebenta ng mga gamit sa bahay tulad ng muwebles, mahalaga ang pagtulong sa mga customer na makahanap ng tamang produkto para sa kanilang panlasa at pangangailangan. Sa artikulong ito, inilarawan ng mga engineer mula sa kumpanya kung paano nila ginagamit ang ML at NLP upang "ipalabas ang tamang resulta para sa mga customer". Partikular, ang kanilang Query Intent Engine ay binuo upang gumamit ng entity extraction, classifier training, asset at opinion extraction, at sentiment tagging sa mga review ng customer. Ito ay isang klasikong halimbawa kung paano gumagana ang NLP sa online retail.  
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Pamamahala ng imbentaryo

Ang mga makabago at mabilis na kumpanya tulad ng [StitchFix](https://stitchfix.com), isang box service na nagpapadala ng damit sa mga consumer, ay lubos na umaasa sa ML para sa mga rekomendasyon at pamamahala ng imbentaryo. Ang kanilang mga styling team ay nagtutulungan kasama ang kanilang mga merchandising team. Sa katunayan, "ang isa sa aming mga data scientist ay nag-eksperimento sa isang genetic algorithm at inilapat ito sa damit upang mahulaan kung ano ang magiging matagumpay na piraso ng damit na hindi pa umiiral ngayon. Dinala namin ito sa merchandise team at ngayon maaari nila itong gamitin bilang isang tool."  
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Pangangalagang Pangkalusugan

Ang sektor ng pangangalagang pangkalusugan ay maaaring gumamit ng ML upang i-optimize ang mga gawain sa pananaliksik at pati na rin ang mga problemang logistic tulad ng muling pag-admit ng mga pasyente o pagpigil sa pagkalat ng mga sakit.

### Pamamahala ng clinical trials

Ang toxicity sa clinical trials ay isang malaking alalahanin para sa mga gumagawa ng gamot. Gaano karaming toxicity ang katanggap-tanggap? Sa pag-aaral na ito, ang pagsusuri sa iba't ibang mga pamamaraan ng clinical trial ay humantong sa pagbuo ng isang bagong paraan para mahulaan ang posibilidad ng mga resulta ng clinical trial. Partikular, nagamit nila ang random forest upang makabuo ng isang [classifier](../../4-Classification/README.md) na kayang makilala ang mga grupo ng gamot.  
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Pamamahala ng muling pag-admit sa ospital

Ang pangangalaga sa ospital ay magastos, lalo na kapag kailangang muling i-admit ang mga pasyente. Tinalakay sa papel na ito ang isang kumpanya na gumagamit ng ML upang mahulaan ang potensyal na muling pag-admit gamit ang [clustering](../../5-Clustering/README.md) algorithms. Ang mga cluster na ito ay tumutulong sa mga analyst na "matuklasan ang mga grupo ng muling pag-admit na maaaring may parehong sanhi".  
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Pamamahala ng sakit

Ang kamakailang pandemya ay nagbigay-liwanag sa mga paraan kung paano makakatulong ang machine learning sa pagpigil sa pagkalat ng sakit. Sa artikulong ito, makikilala mo ang paggamit ng ARIMA, logistic curves, linear regression, at SARIMA. "Ang gawaing ito ay isang pagtatangka upang kalkulahin ang bilis ng pagkalat ng virus na ito at sa gayon ay mahulaan ang mga pagkamatay, paggaling, at kumpirmadong kaso, upang makatulong ito sa mas mahusay na paghahanda at kaligtasan."  
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekolohiya at Green Tech

Ang kalikasan at ekolohiya ay binubuo ng maraming sensitibong sistema kung saan ang ugnayan sa pagitan ng mga hayop at kalikasan ay napapansin. Mahalagang masukat nang tama ang mga sistemang ito at kumilos nang naaayon kung may mangyari, tulad ng sunog sa kagubatan o pagbaba ng populasyon ng hayop.

### Pamamahala ng kagubatan

Natuto ka tungkol sa [Reinforcement Learning](../../8-Reinforcement/README.md) sa mga nakaraang aralin. Maaari itong maging napaka-kapaki-pakinabang kapag sinusubukang hulaan ang mga pattern sa kalikasan. Partikular, maaari itong magamit upang subaybayan ang mga problemang ekolohikal tulad ng sunog sa kagubatan at pagkalat ng invasive species. Sa Canada, isang grupo ng mga mananaliksik ang gumamit ng Reinforcement Learning upang bumuo ng mga modelo ng dynamics ng sunog sa kagubatan mula sa mga satellite images. Gamit ang isang makabagong "spatially spreading process (SSP)", inilarawan nila ang isang sunog sa kagubatan bilang "ang ahente sa anumang cell sa landscape." "Ang hanay ng mga aksyon na maaaring gawin ng sunog mula sa isang lokasyon sa anumang oras ay kinabibilangan ng pagkalat sa hilaga, timog, silangan, o kanluran o hindi pagkalat."  
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Pagsubaybay sa galaw ng mga hayop

Habang ang deep learning ay lumikha ng rebolusyon sa visual na pagsubaybay sa galaw ng mga hayop (maaari kang gumawa ng sarili mong [polar bear tracker](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) dito), may lugar pa rin ang klasikong ML sa gawaing ito.

Ang mga sensor upang subaybayan ang galaw ng mga hayop sa bukid at IoT ay gumagamit ng ganitong uri ng visual processing, ngunit ang mas basic na ML techniques ay kapaki-pakinabang upang i-preprocess ang data. Halimbawa, sa papel na ito, ang mga postura ng tupa ay na-monitor at na-analyze gamit ang iba't ibang classifier algorithms. Maaaring makilala mo ang ROC curve sa pahina 335.  
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Pamamahala ng Enerhiya

Sa ating mga aralin sa [time series forecasting](../../7-TimeSeries/README.md), binanggit natin ang konsepto ng smart parking meters upang makabuo ng kita para sa isang bayan batay sa pag-unawa sa supply at demand. Tinalakay nang detalyado sa artikulong ito kung paano pinagsama ang clustering, regression, at time series forecasting upang matulungan ang paghulaan ang hinaharap na paggamit ng enerhiya sa Ireland, batay sa smart metering.  
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Seguro

Ang sektor ng seguro ay isa pang sektor na gumagamit ng ML upang bumuo at i-optimize ang mga viable financial at actuarial models.

### Pamamahala ng Volatility

Ang MetLife, isang life insurance provider, ay bukas sa paraan kung paano nila sinusuri at binabawasan ang volatility sa kanilang mga financial models. Sa artikulong ito, mapapansin mo ang mga binary at ordinal classification visualizations. Makikita mo rin ang mga forecasting visualizations.  
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Sining, Kultura, at Panitikan

Sa sining, halimbawa sa pamamahayag, maraming kawili-wiling problema. Ang pagtuklas ng pekeng balita ay isang malaking problema dahil napatunayang nakakaimpluwensya ito sa opinyon ng mga tao at kahit sa pagbagsak ng mga demokrasya. Ang mga museo ay maaari ring makinabang mula sa paggamit ng ML sa lahat ng bagay mula sa paghahanap ng mga ugnayan sa pagitan ng mga artifact hanggang sa pagpaplano ng mga mapagkukunan.

### Pagtuklas ng pekeng balita

Ang pagtuklas ng pekeng balita ay naging isang laro ng habulan sa media ngayon. Sa artikulong ito, iminungkahi ng mga mananaliksik na ang isang sistema na pinagsasama ang ilang mga ML techniques na ating pinag-aralan ay maaaring subukan at ang pinakamahusay na modelo ay i-deploy: "Ang sistemang ito ay batay sa natural language processing upang kunin ang mga feature mula sa data at pagkatapos ang mga feature na ito ay ginagamit para sa training ng machine learning classifiers tulad ng Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), at Logistic Regression (LR)."  
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Ipinapakita ng artikulong ito kung paano ang pagsasama-sama ng iba't ibang ML domains ay maaaring makabuo ng mga kawili-wiling resulta na makakatulong sa pagpigil sa pagkalat ng pekeng balita at paglikha ng tunay na pinsala; sa kasong ito, ang impetus ay ang pagkalat ng mga tsismis tungkol sa mga paggamot sa COVID na nag-udyok ng karahasan ng mob.

### Museum ML

Ang mga museo ay nasa hangganan ng isang AI rebolusyon kung saan ang pag-catalog at pag-digitize ng mga koleksyon at paghahanap ng mga ugnayan sa pagitan ng mga artifact ay nagiging mas madali habang umuunlad ang teknolohiya. Ang mga proyekto tulad ng [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) ay tumutulong na mabuksan ang mga misteryo ng mga hindi ma-access na koleksyon tulad ng Vatican Archives. Ngunit, ang aspeto ng negosyo ng mga museo ay nakikinabang din mula sa mga ML models.

Halimbawa, ang Art Institute of Chicago ay bumuo ng mga modelo upang mahulaan kung ano ang interes ng mga audience at kung kailan sila dadalo sa mga eksibisyon. Ang layunin ay lumikha ng mga individualized at optimized na karanasan ng bisita sa bawat pagbisita ng user sa museo. "Sa fiscal 2017, ang modelo ay hinulaan ang attendance at admissions na may 1 porsyento na katumpakan, sabi ni Andrew Simnick, senior vice president sa Art Institute."  
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Pag-segment ng customer

Ang pinaka-epektibong mga estratehiya sa marketing ay nagta-target ng mga customer sa iba't ibang paraan batay sa iba't ibang mga grupo. Sa artikulong ito, tinalakay ang mga gamit ng Clustering algorithms upang suportahan ang differentiated marketing. Ang differentiated marketing ay tumutulong sa mga kumpanya na mapabuti ang brand recognition, maabot ang mas maraming customer, at kumita ng mas maraming pera.  
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Hamon
Tukuyin ang isa pang sektor na nakikinabang mula sa ilang mga teknik na natutunan mo sa kurikulum na ito, at alamin kung paano nito ginagamit ang ML.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/)

## Review & Pag-aaral sa Sarili

Ang Wayfair data science team ay may ilang mga kawili-wiling video tungkol sa kung paano nila ginagamit ang ML sa kanilang kumpanya. Sulit itong [panoorin](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Takdang Aralin

[Isang ML scavenger hunt](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.