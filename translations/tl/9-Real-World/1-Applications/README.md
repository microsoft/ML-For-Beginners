<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T18:15:48+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "tl"
}
-->
# Postscript: Machine learning sa tunay na mundo

![Buod ng machine learning sa tunay na mundo sa isang sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote ni [Tomomi Imura](https://www.twitter.com/girlie_mac)

Sa kurikulum na ito, natutunan mo ang maraming paraan para ihanda ang data para sa training at lumikha ng mga machine learning model. Nagtayo ka ng serye ng mga klasikong regression, clustering, classification, natural language processing, at time series models. Binabati kita! Ngayon, maaaring iniisip mo kung para saan ang lahat ng ito... ano ang mga aplikasyon ng mga modelong ito sa tunay na mundo?

Bagamat maraming interes sa industriya ang nakatuon sa AI, na karaniwang gumagamit ng deep learning, may mga mahalagang aplikasyon pa rin para sa mga klasikong machine learning model. Maaaring ginagamit mo na ang ilan sa mga aplikasyon na ito ngayon! Sa araling ito, susuriin mo kung paano ginagamit ng walong iba't ibang industriya at mga domain ng kaalaman ang mga ganitong uri ng modelo upang gawing mas mahusay, maaasahan, matalino, at mahalaga ang kanilang mga aplikasyon para sa mga gumagamit.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Pananalapi

Ang sektor ng pananalapi ay nag-aalok ng maraming oportunidad para sa machine learning. Maraming problema sa larangang ito ang maaaring i-modelo at lutasin gamit ang ML.

### Pagtuklas ng pandaraya sa credit card

Natutunan natin ang tungkol sa [k-means clustering](../../5-Clustering/2-K-Means/README.md) sa mas maagang bahagi ng kurso, pero paano ito magagamit para lutasin ang mga problema kaugnay ng pandaraya sa credit card?

Ang k-means clustering ay kapaki-pakinabang sa isang teknik sa pagtuklas ng pandaraya sa credit card na tinatawag na **outlier detection**. Ang mga outlier, o mga paglihis sa mga obserbasyon tungkol sa isang set ng data, ay maaaring magpahiwatig kung ang isang credit card ay ginagamit sa normal na paraan o kung may kakaibang nangyayari. Tulad ng ipinakita sa papel na naka-link sa ibaba, maaari mong ayusin ang data ng credit card gamit ang isang k-means clustering algorithm at i-assign ang bawat transaksyon sa isang cluster batay sa kung gaano ito ka-outlier. Pagkatapos, maaari mong suriin ang mga pinaka-mapanganib na cluster para sa mga pandaraya kumpara sa mga lehitimong transaksyon.
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Pamamahala ng yaman

Sa pamamahala ng yaman, ang isang indibidwal o kumpanya ay humahawak ng mga pamumuhunan sa ngalan ng kanilang mga kliyente. Ang kanilang trabaho ay panatilihin at palaguin ang yaman sa pangmatagalan, kaya mahalaga ang pagpili ng mga pamumuhunan na maganda ang performance.

Isang paraan para suriin kung paano nagpe-perform ang isang partikular na pamumuhunan ay sa pamamagitan ng statistical regression. Ang [linear regression](../../2-Regression/1-Tools/README.md) ay isang mahalagang tool para maunawaan kung paano nagpe-perform ang isang fund kumpara sa isang benchmark. Maaari rin nating matukoy kung ang mga resulta ng regression ay statistically significant, o kung gaano kalaki ang epekto nito sa mga pamumuhunan ng kliyente. Maaari mo pang palawakin ang iyong pagsusuri gamit ang multiple regression, kung saan maaaring isama ang karagdagang mga risk factor. Para sa isang halimbawa kung paano ito gagana para sa isang partikular na fund, tingnan ang papel sa ibaba tungkol sa pagsusuri ng performance ng fund gamit ang regression.
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Edukasyon

Ang sektor ng edukasyon ay isa ring napaka-interesanteng larangan kung saan maaaring gamitin ang ML. May mga kawili-wiling problema na maaaring lutasin tulad ng pagtuklas ng pandaraya sa mga pagsusulit o sanaysay, o pamamahala ng bias, sinasadya man o hindi, sa proseso ng pagwawasto.

### Pagtataya ng pag-uugali ng mag-aaral

[Coursera](https://coursera.com), isang online open course provider, ay may mahusay na tech blog kung saan nila tinatalakay ang maraming desisyon sa engineering. Sa case study na ito, nag-plot sila ng regression line upang subukang tuklasin ang anumang ugnayan sa pagitan ng mababang NPS (Net Promoter Score) rating at retention o pag-drop-off sa kurso.
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Pagbawas ng bias

[Grammarly](https://grammarly.com), isang writing assistant na nagche-check ng spelling at grammar errors, ay gumagamit ng mga sopistikadong [natural language processing systems](../../6-NLP/README.md) sa kanilang mga produkto. Nag-publish sila ng isang kawili-wiling case study sa kanilang tech blog tungkol sa kung paano nila hinarap ang gender bias sa machine learning, na natutunan mo sa aming [introductory fairness lesson](../../1-Introduction/3-fairness/README.md).
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retail

Ang sektor ng retail ay tiyak na makikinabang sa paggamit ng ML, mula sa paglikha ng mas mahusay na customer journey hanggang sa optimal na pamamahala ng imbentaryo.

### Pag-personalize ng customer journey

Sa Wayfair, isang kumpanya na nagbebenta ng mga gamit sa bahay tulad ng kasangkapan, mahalaga ang pagtulong sa mga customer na makahanap ng tamang produkto para sa kanilang panlasa at pangangailangan. Sa artikulong ito, inilalarawan ng mga engineer mula sa kumpanya kung paano nila ginagamit ang ML at NLP upang "ipalabas ang tamang resulta para sa mga customer". Kapansin-pansin, ang kanilang Query Intent Engine ay binuo upang gumamit ng entity extraction, classifier training, asset at opinion extraction, at sentiment tagging sa mga review ng customer. Ito ay isang klasikong halimbawa kung paano gumagana ang NLP sa online retail.
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Pamamahala ng imbentaryo

Ang mga makabago at mabilis na kumpanya tulad ng [StitchFix](https://stitchfix.com), isang box service na nagpapadala ng damit sa mga consumer, ay lubos na umaasa sa ML para sa mga rekomendasyon at pamamahala ng imbentaryo. Ang kanilang mga styling team ay nagtutulungan sa kanilang mga merchandising team, sa katunayan: "isa sa aming mga data scientist ay nag-eksperimento sa isang genetic algorithm at inilapat ito sa damit upang mahulaan kung ano ang magiging matagumpay na piraso ng damit na hindi pa umiiral ngayon. Ibinahagi namin ito sa merchandise team at ngayon maaari nilang gamitin ito bilang isang tool."
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Pangangalaga sa Kalusugan

Ang sektor ng pangangalaga sa kalusugan ay maaaring gumamit ng ML upang i-optimize ang mga gawain sa pananaliksik at pati na rin ang mga problema sa logistics tulad ng readmitting ng mga pasyente o pagpigil sa pagkalat ng mga sakit.

### Pamamahala ng clinical trials

Ang toxicity sa clinical trials ay isang malaking alalahanin para sa mga gumagawa ng gamot. Gaano karaming toxicity ang katanggap-tanggap? Sa pag-aaral na ito, ang pagsusuri sa iba't ibang mga clinical trial method ay humantong sa pagbuo ng isang bagong diskarte para sa pagtataya ng mga posibilidad ng mga resulta ng clinical trial. Partikular, nagamit nila ang random forest upang makabuo ng isang [classifier](../../4-Classification/README.md) na kayang mag-diskrimina sa pagitan ng mga grupo ng gamot.
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Pamamahala ng readmission sa ospital

Ang pangangalaga sa ospital ay magastos, lalo na kapag kailangang ma-readmit ang mga pasyente. Ang papel na ito ay tinatalakay ang isang kumpanya na gumagamit ng ML upang mahulaan ang potensyal na readmission gamit ang [clustering](../../5-Clustering/README.md) algorithms. Ang mga cluster na ito ay tumutulong sa mga analyst upang "matuklasan ang mga grupo ng readmissions na maaaring may karaniwang sanhi".
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Pamamahala ng sakit

Ang kamakailang pandemya ay nagbigay-diin sa mga paraan kung paano makakatulong ang machine learning sa pagpigil sa pagkalat ng sakit. Sa artikulong ito, makikilala mo ang paggamit ng ARIMA, logistic curves, linear regression, at SARIMA. "Ang gawaing ito ay isang pagsubok upang kalkulahin ang rate ng pagkalat ng virus na ito at sa gayon ay mahulaan ang mga pagkamatay, paggaling, at kumpirmadong kaso, upang makatulong sa mas mahusay na paghahanda at kaligtasan."
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekolohiya at Green Tech

Ang kalikasan at ekolohiya ay binubuo ng maraming sensitibong sistema kung saan ang interaksyon sa pagitan ng mga hayop at kalikasan ay napapansin. Mahalagang masukat ang mga sistemang ito nang tama at kumilos nang naaayon kung may mangyari, tulad ng forest fire o pagbaba ng populasyon ng hayop.

### Pamamahala ng kagubatan

Natutunan mo ang tungkol sa [Reinforcement Learning](../../8-Reinforcement/README.md) sa mga nakaraang aralin. Maaari itong maging napaka-kapaki-pakinabang kapag sinusubukang hulaan ang mga pattern sa kalikasan. Partikular, maaari itong magamit upang subaybayan ang mga problema sa ekolohiya tulad ng forest fires at pagkalat ng invasive species. Sa Canada, isang grupo ng mga mananaliksik ang gumamit ng Reinforcement Learning upang bumuo ng mga modelo ng forest wildfire dynamics mula sa mga satellite images. Gamit ang isang makabago na "spatially spreading process (SSP)", inilarawan nila ang isang forest fire bilang "ang agent sa anumang cell sa landscape." "Ang set ng mga aksyon na maaaring gawin ng apoy mula sa isang lokasyon sa anumang punto ng oras ay kinabibilangan ng pagkalat sa hilaga, timog, silangan, o kanluran o hindi pagkalat.

Ang diskarte na ito ay nagbabaliktad sa karaniwang RL setup dahil ang dynamics ng kaukulang Markov Decision Process (MDP) ay isang kilalang function para sa agarang pagkalat ng wildfire." Basahin ang higit pa tungkol sa mga klasikong algorithm na ginamit ng grupong ito sa link sa ibaba.
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Motion sensing ng mga hayop

Bagamat ang deep learning ay lumikha ng rebolusyon sa visual tracking ng mga galaw ng hayop (maaari kang gumawa ng sarili mong [polar bear tracker](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) dito), may lugar pa rin ang klasikong ML sa gawaing ito.

Ang mga sensor upang subaybayan ang mga galaw ng mga hayop sa sakahan at IoT ay gumagamit ng ganitong uri ng visual processing, ngunit ang mas basic na ML techniques ay kapaki-pakinabang para sa pag-preprocess ng data. Halimbawa, sa papel na ito, ang mga postura ng tupa ay na-monitor at na-analyze gamit ang iba't ibang classifier algorithms. Maaaring makilala mo ang ROC curve sa pahina 335.
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Pamamahala ng Enerhiya

Sa ating mga aralin sa [time series forecasting](../../7-TimeSeries/README.md), binanggit natin ang konsepto ng smart parking meters upang makabuo ng kita para sa isang bayan batay sa pag-unawa sa supply at demand. Ang artikulong ito ay nagdedetalye kung paano pinagsama ang clustering, regression, at time series forecasting upang makatulong sa pagtataya ng hinaharap na paggamit ng enerhiya sa Ireland, batay sa smart metering.
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Insurance

Ang sektor ng insurance ay isa pang sektor na gumagamit ng ML upang bumuo at i-optimize ang mga viable financial at actuarial models.

### Pamamahala ng Volatility

Ang MetLife, isang life insurance provider, ay bukas sa paraan kung paano nila sinusuri at binabawasan ang volatility sa kanilang mga financial models. Sa artikulong ito, makikita mo ang binary at ordinal classification visualizations. Makikita mo rin ang forecasting visualizations.
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Sining, Kultura, at Panitikan

Sa sining, halimbawa sa journalism, maraming kawili-wiling problema. Ang pagtuklas ng pekeng balita ay isang malaking problema dahil napatunayan na ito ay nakakaimpluwensya sa opinyon ng mga tao at maging sa pagbagsak ng mga demokrasya. Ang mga museo ay maaari ring makinabang mula sa paggamit ng ML sa lahat ng bagay mula sa paghahanap ng mga koneksyon sa pagitan ng mga artifact hanggang sa pagpaplano ng mga resources.

### Pagtuklas ng pekeng balita

Ang pagtuklas ng pekeng balita ay naging isang laro ng habulan sa media ngayon. Sa artikulong ito, iminumungkahi ng mga mananaliksik na ang isang sistema na pinagsasama ang ilang mga ML techniques na ating pinag-aralan ay maaaring subukan at ang pinakamahusay na modelo ay i-deploy: "Ang sistemang ito ay batay sa natural language processing upang kunin ang mga feature mula sa data at pagkatapos ang mga feature na ito ay ginagamit para sa training ng machine learning classifiers tulad ng Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), at Logistic Regression(LR)."
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Ipinapakita ng artikulong ito kung paano ang pagsasama-sama ng iba't ibang ML domains ay maaaring makabuo ng mga kawili-wiling resulta na makakatulong sa pagpigil sa pagkalat ng pekeng balita at paglikha ng tunay na pinsala; sa kasong ito, ang impetus ay ang pagkalat ng mga tsismis tungkol sa mga paggamot sa COVID na nag-udyok ng karahasan ng mob.

### Museum ML

Ang mga museo ay nasa hangganan ng isang AI rebolusyon kung saan ang pag-catalog at pag-digitize ng mga koleksyon at paghahanap ng mga koneksyon sa pagitan ng mga artifact ay nagiging mas madali habang umuunlad ang teknolohiya. Ang mga proyekto tulad ng [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) ay tumutulong sa pagbubukas ng mga misteryo ng mga hindi ma-access na koleksyon tulad ng Vatican Archives. Ngunit, ang business aspect ng mga museo ay nakikinabang din sa mga ML models.

Halimbawa, ang Art Institute of Chicago ay nagtayo ng mga modelo upang mahulaan kung ano ang interes ng mga audience at kung kailan sila dadalo sa mga eksibisyon. Ang layunin ay lumikha ng individualized at optimized na karanasan ng bisita sa bawat oras na bumisita ang user sa museo. "Noong fiscal 2017, ang modelo ay hinulaan ang attendance at admissions sa loob ng 1 porsyento ng katumpakan, sabi ni Andrew Simnick, senior vice president sa Art Institute."
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Customer segmentation

Ang pinaka-epektibong mga estratehiya sa marketing ay nagta-target sa mga customer sa iba't ibang paraan batay sa iba't ibang mga grupo. Sa artikulong ito, tinalakay ang mga gamit ng Clustering algorithms upang suportahan ang differentiated marketing. Ang differentiated marketing ay tumutulong sa mga kumpanya na mapabuti ang brand recognition, maabot ang mas maraming customer, at kumita ng mas maraming pera.
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Hamon

Tukuyin ang isa pang sektor na nakikinabang mula sa ilan sa mga teknik na natutunan mo sa kurikulum na ito, at alamin kung paano ito gumagamit ng ML.
## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Pagrepaso at Sariling Pag-aaral

Ang Wayfair data science team ay may ilang mga kawili-wiling video tungkol sa kung paano nila ginagamit ang ML sa kanilang kumpanya. Sulit itong [panoorin](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Takdang-Aralin

[Isang ML scavenger hunt](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.