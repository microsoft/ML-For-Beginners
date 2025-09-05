<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T18:18:45+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "tl"
}
-->
# Panimula sa machine learning

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML para sa mga baguhan - Panimula sa Machine Learning para sa mga Baguhan](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML para sa mga baguhan - Panimula sa Machine Learning para sa mga Baguhan")

> 🎥 I-click ang imahe sa itaas para sa isang maikling video na tumatalakay sa araling ito.

Maligayang pagdating sa kursong ito tungkol sa klasikong machine learning para sa mga baguhan! Kung ikaw ay ganap na bago sa paksang ito, o isang bihasang practitioner ng ML na nais mag-refresh ng kaalaman, masaya kaming makasama ka! Layunin naming lumikha ng isang magiliw na panimulang lugar para sa iyong pag-aaral ng ML at ikalulugod naming suriin, tumugon, at isama ang iyong [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Panimula sa ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Panimula sa ML")

> 🎥 I-click ang imahe sa itaas para sa isang video: Ipinakikilala ni John Guttag ng MIT ang machine learning

---
## Pagsisimula sa machine learning

Bago simulan ang kurikulum na ito, kailangan mong ihanda ang iyong computer upang magpatakbo ng mga notebook nang lokal.

- **I-configure ang iyong makina gamit ang mga video na ito**. Gamitin ang mga sumusunod na link upang matutunan [kung paano mag-install ng Python](https://youtu.be/CXZYvNRIAKM) sa iyong sistema at [mag-setup ng text editor](https://youtu.be/EU8eayHWoZg) para sa development.
- **Matutong Python**. Inirerekomenda rin na magkaroon ng pangunahing kaalaman sa [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), isang programming language na kapaki-pakinabang para sa mga data scientist na ginagamit natin sa kursong ito.
- **Matutong Node.js at JavaScript**. Gagamit din tayo ng JavaScript sa ilang bahagi ng kursong ito kapag gumagawa ng mga web app, kaya kailangan mong magkaroon ng [node](https://nodejs.org) at [npm](https://www.npmjs.com/) na naka-install, pati na rin ang [Visual Studio Code](https://code.visualstudio.com/) para sa parehong Python at JavaScript development.
- **Gumawa ng GitHub account**. Dahil natagpuan mo kami dito sa [GitHub](https://github.com), maaaring mayroon ka nang account, ngunit kung wala pa, gumawa ng isa at i-fork ang kurikulum na ito upang magamit sa iyong sarili. (Huwag kalimutang magbigay ng star 😊)
- **Galugarin ang Scikit-learn**. Magkaroon ng kaalaman sa [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), isang set ng ML libraries na binabanggit natin sa mga araling ito.

---
## Ano ang machine learning?

Ang terminong 'machine learning' ay isa sa mga pinakasikat at madalas gamitin na termino sa kasalukuyan. Malaki ang posibilidad na narinig mo na ang terminong ito kahit isang beses kung may kaalaman ka sa teknolohiya, anuman ang larangan mo. Gayunpaman, ang mekanika ng machine learning ay misteryo para sa karamihan. Para sa isang baguhan sa machine learning, maaaring nakakatakot ang paksa. Kaya mahalagang maunawaan kung ano talaga ang machine learning, at matutunan ito nang paunti-unti, sa pamamagitan ng mga praktikal na halimbawa.

---
## Ang hype curve

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Ipinapakita ng Google Trends ang kamakailang 'hype curve' ng terminong 'machine learning'

---
## Isang misteryosong uniberso

Namumuhay tayo sa isang uniberso na puno ng mga kamangha-manghang misteryo. Ang mga dakilang siyentipiko tulad nina Stephen Hawking, Albert Einstein, at marami pang iba ay naglaan ng kanilang buhay sa paghahanap ng makabuluhang impormasyon upang matuklasan ang mga misteryo ng mundo sa paligid natin. Ito ang kalagayan ng tao sa pag-aaral: ang isang bata ay natututo ng mga bagong bagay at natutuklasan ang istruktura ng kanilang mundo taon-taon habang sila ay lumalaki.

---
## Ang utak ng bata

Ang utak at pandama ng isang bata ay nakikita ang mga katotohanan sa kanilang paligid at unti-unting natututo ng mga nakatagong pattern ng buhay na tumutulong sa bata na bumuo ng mga lohikal na tuntunin upang makilala ang mga natutunang pattern. Ang proseso ng pag-aaral ng utak ng tao ang dahilan kung bakit ang tao ang pinaka-sopistikadong nilalang sa mundo. Ang patuloy na pag-aaral sa pamamagitan ng pagtuklas ng mga nakatagong pattern at pagkatapos ay paglikha ng mga inobasyon mula sa mga pattern na ito ay nagbibigay-daan sa atin na patuloy na pagbutihin ang ating sarili sa buong buhay natin. Ang kakayahan sa pag-aaral at pag-evolve na ito ay may kaugnayan sa konsepto na tinatawag na [brain plasticity](https://www.simplypsychology.org/brain-plasticity.html). Sa pangkalahatan, maaari nating iguhit ang ilang motivational na pagkakatulad sa pagitan ng proseso ng pag-aaral ng utak ng tao at mga konsepto ng machine learning.

---
## Ang utak ng tao

Ang [utak ng tao](https://www.livescience.com/29365-human-brain.html) ay nakikita ang mga bagay mula sa totoong mundo, pinoproseso ang nakitang impormasyon, gumagawa ng makatuwirang desisyon, at gumaganap ng mga tiyak na aksyon batay sa mga sitwasyon. Ito ang tinatawag nating matalinong pag-uugali. Kapag pinrograma natin ang isang imitasyon ng matalinong proseso ng pag-uugali sa isang makina, ito ay tinatawag na artificial intelligence (AI).

---
## Ilang terminolohiya

Bagama't maaaring malito ang mga termino, ang machine learning (ML) ay isang mahalagang subset ng artificial intelligence. **Ang ML ay nakatuon sa paggamit ng mga espesyal na algorithm upang matuklasan ang makabuluhang impormasyon at hanapin ang mga nakatagong pattern mula sa nakitang data upang suportahan ang proseso ng makatuwirang paggawa ng desisyon**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Isang diagram na nagpapakita ng mga relasyon sa pagitan ng AI, ML, deep learning, at data science. Infographic ni [Jen Looper](https://twitter.com/jenlooper) na inspirasyon mula sa [graphic na ito](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Mga konseptong tatalakayin

Sa kurikulum na ito, tatalakayin natin ang mga pangunahing konsepto ng machine learning na dapat malaman ng isang baguhan. Tatalakayin natin ang tinatawag na 'klasikong machine learning' gamit ang Scikit-learn, isang mahusay na library na ginagamit ng maraming estudyante upang matutunan ang mga pangunahing kaalaman. Upang maunawaan ang mas malawak na konsepto ng artificial intelligence o deep learning, mahalaga ang matibay na pundasyon sa machine learning, kaya nais naming ibigay ito dito.

---
## Sa kursong ito matututunan mo:

- mga pangunahing konsepto ng machine learning
- ang kasaysayan ng ML
- ML at pagiging patas
- mga teknik sa regression ML
- mga teknik sa classification ML
- mga teknik sa clustering ML
- mga teknik sa natural language processing ML
- mga teknik sa time series forecasting ML
- reinforcement learning
- mga aplikasyon ng ML sa totoong mundo

---
## Ano ang hindi natin tatalakayin

- deep learning
- neural networks
- AI

Upang magkaroon ng mas mahusay na karanasan sa pag-aaral, iiwasan natin ang mga komplikasyon ng neural networks, 'deep learning' - ang paggawa ng mga modelong may maraming layer gamit ang neural networks - at AI, na tatalakayin natin sa ibang kurikulum. Mag-aalok din kami ng paparating na kurikulum sa data science upang mag-focus sa aspeto ng mas malawak na larangan na ito.

---
## Bakit mag-aral ng machine learning?

Ang machine learning, mula sa perspektibo ng sistema, ay tinutukoy bilang paglikha ng mga automated na sistema na maaaring matuto ng mga nakatagong pattern mula sa data upang makatulong sa paggawa ng matalinong desisyon.

Ang motibasyong ito ay maluwag na inspirasyon ng kung paano natututo ang utak ng tao ng ilang bagay batay sa data na nakikita nito mula sa labas ng mundo.

✅ Mag-isip ng isang minuto kung bakit maaaring nais ng isang negosyo na gumamit ng mga estratehiya sa machine learning kumpara sa paggawa ng isang hard-coded na rules-based engine.

---
## Mga aplikasyon ng machine learning

Ang mga aplikasyon ng machine learning ay halos nasa lahat ng lugar ngayon, at kasing laganap ng data na dumadaloy sa ating mga lipunan, na nabuo ng ating mga smartphone, mga konektadong device, at iba pang sistema. Isinasaalang-alang ang napakalaking potensyal ng mga makabagong machine learning algorithm, sinisiyasat ng mga mananaliksik ang kanilang kakayahan upang lutasin ang mga multi-dimensional at multi-disciplinary na totoong problema na may magagandang positibong resulta.

---
## Mga halimbawa ng applied ML

**Maraming paraan upang magamit ang machine learning**:

- Upang mahulaan ang posibilidad ng sakit mula sa kasaysayan ng medikal ng isang pasyente o mga ulat.
- Upang gamitin ang data ng panahon upang mahulaan ang mga kaganapan sa panahon.
- Upang maunawaan ang damdamin ng isang teksto.
- Upang matukoy ang pekeng balita upang mapigilan ang pagkalat ng propaganda.

Ang finance, economics, earth science, space exploration, biomedical engineering, cognitive science, at maging ang mga larangan sa humanities ay nag-aangkop ng machine learning upang lutasin ang mga mahihirap na problema sa pagproseso ng data sa kanilang larangan.

---
## Konklusyon

Ang machine learning ay nag-aautomat ng proseso ng pagtuklas ng pattern sa pamamagitan ng paghahanap ng makabuluhang mga insight mula sa totoong mundo o generated na data. Napatunayan nitong napakahalaga sa negosyo, kalusugan, at mga aplikasyon sa pananalapi, bukod sa iba pa.

Sa malapit na hinaharap, ang pag-unawa sa mga pangunahing kaalaman ng machine learning ay magiging mahalaga para sa mga tao mula sa anumang larangan dahil sa malawakang paggamit nito.

---
# 🚀 Hamon

Gumuhit, sa papel o gamit ang isang online na app tulad ng [Excalidraw](https://excalidraw.com/), ng iyong pag-unawa sa mga pagkakaiba sa pagitan ng AI, ML, deep learning, at data science. Magdagdag ng ilang ideya ng mga problemang mahusay na malulutas ng bawat isa sa mga teknik na ito.

# [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---
# Review & Self Study

Upang matutunan pa kung paano ka maaaring magtrabaho gamit ang mga ML algorithm sa cloud, sundan ang [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Kumuha ng [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) tungkol sa mga pangunahing kaalaman ng ML.

---
# Takdang-Aralin

[Simulan ang pag-aaral](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na pinagmulan. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.