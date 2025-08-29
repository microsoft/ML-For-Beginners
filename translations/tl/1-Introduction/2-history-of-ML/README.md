<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b2d11df10030cacc41427a1fbc8bc8f1",
  "translation_date": "2025-08-29T13:46:16+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "tl"
}
-->
# Kasaysayan ng Machine Learning

![Buod ng Kasaysayan ng Machine Learning sa isang sketchnote](../../../../translated_images/ml-history.a1bdfd4ce1f464d9a0502f38d355ffda384c95cd5278297a46c9a391b5053bc4.tl.png)
> Sketchnote ni [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/3/)

---

[![ML para sa mga baguhan - Kasaysayan ng Machine Learning](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML para sa mga baguhan - Kasaysayan ng Machine Learning")

> 🎥 I-click ang larawan sa itaas para sa isang maikling video tungkol sa araling ito.

Sa araling ito, tatalakayin natin ang mga pangunahing milestone sa kasaysayan ng machine learning at artificial intelligence.

Ang kasaysayan ng artificial intelligence (AI) bilang isang larangan ay konektado sa kasaysayan ng machine learning, dahil ang mga algorithm at mga pag-unlad sa komputasyon na bumubuo sa ML ay nag-ambag sa pag-unlad ng AI. Mahalagang tandaan na, bagama't nagsimulang mabuo ang mga larangang ito bilang magkakaibang paksa noong 1950s, may mga mahahalagang [algorithmic, statistical, mathematical, computational, at teknikal na mga tuklas](https://wikipedia.org/wiki/Timeline_of_machine_learning) na nauna at sumabay sa panahong ito. Sa katunayan, ang mga tao ay nag-iisip tungkol sa mga tanong na ito sa loob ng [daan-daang taon](https://wikipedia.org/wiki/History_of_artificial_intelligence): tinatalakay ng artikulong ito ang mga makasaysayang pundasyon ng ideya ng isang 'nag-iisip na makina.'

---
## Mahahalagang Tuklas

- 1763, 1812 [Bayes Theorem](https://wikipedia.org/wiki/Bayes%27_theorem) at ang mga nauna nito. Ang theorem na ito at ang mga aplikasyon nito ay nagiging batayan ng inference, na naglalarawan ng posibilidad ng isang pangyayari batay sa naunang kaalaman.
- 1805 [Least Square Theory](https://wikipedia.org/wiki/Least_squares) ni Adrien-Marie Legendre, isang Pranses na matematikal. Ang teoryang ito, na matututunan mo sa aming Regression unit, ay tumutulong sa data fitting.
- 1913 [Markov Chains](https://wikipedia.org/wiki/Markov_chain), ipinangalan sa Rusong matematikal na si Andrey Markov, ay ginagamit upang ilarawan ang isang sunod-sunod na posibleng mga pangyayari batay sa isang naunang estado.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) ay isang uri ng linear classifier na imbento ng Amerikanong psychologist na si Frank Rosenblatt na naging pundasyon ng mga pag-unlad sa deep learning.

---

- 1967 [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor) ay isang algorithm na orihinal na dinisenyo upang mag-mapa ng mga ruta. Sa konteksto ng ML, ginagamit ito upang makakita ng mga pattern.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) ay ginagamit upang sanayin ang [feedforward neural networks](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network) ay mga artificial neural networks na nagmula sa feedforward neural networks na lumilikha ng temporal graphs.

✅ Mag-research ng kaunti. Anong iba pang mga petsa ang mahalaga sa kasaysayan ng ML at AI?

---
## 1950: Mga Makina na Nag-iisip

Si Alan Turing, isang tunay na kahanga-hangang tao na nahirang [ng publiko noong 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) bilang pinakadakilang siyentipiko ng ika-20 siglo, ay kinikilala bilang isa sa mga naglatag ng pundasyon para sa konsepto ng isang 'makina na maaaring mag-isip.' Hinarap niya ang mga kritiko at ang kanyang sariling pangangailangan para sa empirikal na ebidensya ng konseptong ito sa pamamagitan ng paglikha ng [Turing Test](https://www.bbc.com/news/technology-18475646), na iyong pag-aaralan sa aming mga aralin sa NLP.

---
## 1956: Dartmouth Summer Research Project

"Ang Dartmouth Summer Research Project sa artificial intelligence ay isang mahalagang kaganapan para sa larangan ng artificial intelligence," at dito unang ginamit ang terminong 'artificial intelligence' ([source](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Ang bawat aspeto ng pagkatuto o anumang iba pang katangian ng katalinuhan ay maaaring mailarawan nang eksakto upang ang isang makina ay magawang gayahin ito.

---

Ang pangunahing mananaliksik, ang propesor ng matematika na si John McCarthy, ay umaasang "magpatuloy batay sa hinuha na ang bawat aspeto ng pagkatuto o anumang iba pang katangian ng katalinuhan ay maaaring mailarawan nang eksakto upang ang isang makina ay magawang gayahin ito." Ang mga kalahok ay kinabibilangan ng isa pang tanyag sa larangan, si Marvin Minsky.

Ang workshop na ito ay kinikilala bilang nagpasimula at nag-udyok ng ilang mga talakayan kabilang ang "ang pag-usbong ng mga simbolikong pamamaraan, mga sistema na nakatuon sa limitadong mga domain (mga maagang expert systems), at mga deductive systems laban sa mga inductive systems." ([source](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Ang Ginintuang Panahon"

Mula 1950s hanggang kalagitnaan ng '70s, mataas ang optimismo na kayang lutasin ng AI ang maraming problema. Noong 1967, buong kumpiyansa na sinabi ni Marvin Minsky na "Sa loob ng isang henerasyon ... ang problema ng paglikha ng 'artificial intelligence' ay halos malulutas na." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Ang pananaliksik sa natural language processing ay umunlad, ang paghahanap ay pinino at ginawang mas makapangyarihan, at ang konsepto ng 'micro-worlds' ay nilikha, kung saan ang mga simpleng gawain ay natatapos gamit ang mga simpleng tagubilin sa wika.

---

Ang pananaliksik ay mahusay na pinondohan ng mga ahensya ng gobyerno, nagkaroon ng mga pag-unlad sa komputasyon at mga algorithm, at ang mga prototype ng mga intelligent na makina ay nabuo. Ilan sa mga makinang ito ay kinabibilangan ng:

* [Shakey the robot](https://wikipedia.org/wiki/Shakey_the_robot), na kayang magmaniobra at magdesisyon kung paano isasagawa ang mga gawain nang 'matalino'.

    ![Shakey, isang intelligent na robot](../../../../translated_images/shakey.4dc17819c447c05bf4b52f76da0bdd28817d056fdb906252ec20124dd4cfa55e.tl.jpg)
    > Shakey noong 1972

---

* Eliza, isang maagang 'chatterbot', ay kayang makipag-usap sa mga tao at kumilos bilang isang primitibong 'therapist'. Malalaman mo pa ang tungkol kay Eliza sa mga aralin sa NLP.

    ![Eliza, isang bot](../../../../translated_images/eliza.84397454cda9559bb5ec296b5b8fff067571c0cccc5405f9c1ab1c3f105c075c.tl.png)
    > Isang bersyon ni Eliza, isang chatbot

---

* "Blocks world" ay isang halimbawa ng micro-world kung saan ang mga bloke ay maaaring itumpok at ayusin, at ang mga eksperimento sa pagtuturo sa mga makina na gumawa ng mga desisyon ay maaaring subukan. Ang mga pag-unlad na binuo gamit ang mga library tulad ng [SHRDLU](https://wikipedia.org/wiki/SHRDLU) ay tumulong sa pagpapalago ng language processing.

    [![blocks world gamit ang SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world gamit ang SHRDLU")

    > 🎥 I-click ang larawan sa itaas para sa isang video: Blocks world gamit ang SHRDLU

---
## 1974 - 1980: "AI Winter"

Sa kalagitnaan ng 1970s, naging malinaw na ang pagiging kumplikado ng paggawa ng 'mga intelligent na makina' ay hindi sapat na naipaliwanag at ang mga pangako nito, batay sa kakayahan ng kompyuter noong panahong iyon, ay labis na napalaki. Nawalan ng pondo at bumagal ang kumpiyansa sa larangan. Ilan sa mga isyung nakaapekto sa kumpiyansa ay kinabibilangan ng:
---
- **Mga Limitasyon**. Ang compute power ay masyadong limitado.
- **Combinatorial explosion**. Ang dami ng mga parameter na kailangang sanayin ay lumago nang eksponensyal habang mas maraming hinihingi sa mga kompyuter, nang walang kasabay na pag-unlad sa compute power at kakayahan.
- **Kakulangan ng data**. May kakulangan ng data na humadlang sa proseso ng pagsubok, pagbuo, at pagpapabuti ng mga algorithm.
- **Tama ba ang mga tanong na tinatanong natin?**. Ang mismong mga tanong na tinatanong ay nagsimulang kuwestyunin. Ang mga mananaliksik ay nagsimulang makatanggap ng kritisismo tungkol sa kanilang mga pamamaraan:
  - Ang mga Turing test ay kinuwestyon sa pamamagitan ng, bukod sa iba pang mga ideya, ang 'chinese room theory' na nagsasabing, "ang pag-program ng isang digital na kompyuter ay maaaring magmukhang nauunawaan nito ang wika ngunit hindi makakalikha ng tunay na pag-unawa." ([source](https://plato.stanford.edu/entries/chinese-room/))
  - Ang etika ng pagpapakilala ng mga artificial intelligences tulad ng "therapist" ELIZA sa lipunan ay hinamon.

---

Kasabay nito, iba't ibang mga paaralan ng pag-iisip sa AI ang nagsimulang mabuo. Nabuo ang isang dichotomy sa pagitan ng ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) na mga kasanayan. Ang mga _scruffy_ na laboratoryo ay nag-aayos ng mga programa nang paulit-ulit hanggang makuha ang nais na resulta. Ang mga _neat_ na laboratoryo ay "nakatuon sa lohika at pormal na paglutas ng problema". Ang ELIZA at SHRDLU ay kilalang mga _scruffy_ na sistema. Noong 1980s, habang lumitaw ang pangangailangan na gawing reproducible ang mga ML system, ang _neat_ na pamamaraan ay unti-unting nangibabaw dahil ang mga resulta nito ay mas madaling maipaliwanag.

---
## 1980s Expert Systems

Habang lumalago ang larangan, naging mas malinaw ang benepisyo nito sa negosyo, at noong 1980s, dumami ang 'expert systems'. "Ang mga expert systems ay kabilang sa mga unang tunay na matagumpay na anyo ng artificial intelligence (AI) software." ([source](https://wikipedia.org/wiki/Expert_system)).

Ang ganitong uri ng sistema ay talagang _hybrid_, na binubuo ng isang rules engine na nagtatakda ng mga kinakailangan sa negosyo, at isang inference engine na gumagamit ng rules system upang makabuo ng mga bagong impormasyon.

Ang panahong ito ay nagbigay din ng mas maraming pansin sa neural networks.

---
## 1987 - 1993: AI 'Chill'

Ang pagdami ng mga specialized expert systems hardware ay nagkaroon ng hindi magandang epekto ng pagiging masyadong specialized. Ang pag-usbong ng mga personal na kompyuter ay nakipagkumpitensya rin sa mga malalaking, specialized, centralized systems. Nagsimula na ang democratization ng computing, at kalaunan ay nagbigay-daan ito sa modernong pagsabog ng big data.

---
## 1993 - 2011

Ang panahong ito ay nagdala ng bagong era para sa ML at AI upang malutas ang ilan sa mga problemang dulot ng kakulangan ng data at compute power noong nakaraan. Ang dami ng data ay mabilis na dumami at naging mas madaling ma-access, para sa mabuti at masama, lalo na sa pag-usbong ng smartphone noong 2007. Ang compute power ay lumago nang eksponensyal, at ang mga algorithm ay sumabay sa pag-unlad. Ang larangan ay nagsimulang maging mas mature habang ang mga malayang araw ng nakaraan ay unti-unting naging isang tunay na disiplina.

---
## Ngayon

Ngayon, ang machine learning at AI ay halos naaabot na ang bawat bahagi ng ating buhay. Ang panahong ito ay nangangailangan ng maingat na pag-unawa sa mga panganib at posibleng epekto ng mga algorithm na ito sa buhay ng tao. Tulad ng sinabi ni Brad Smith ng Microsoft, "Ang teknolohiya ng impormasyon ay nagdadala ng mga isyu na tumatama sa puso ng mga pangunahing proteksyon ng karapatang pantao tulad ng privacy at kalayaan sa pagpapahayag. Ang mga isyung ito ay nagpapataas ng responsibilidad para sa mga kumpanya ng teknolohiya na lumikha ng mga produktong ito. Sa aming pananaw, nangangailangan din ito ng maingat na regulasyon ng gobyerno at ng pagbuo ng mga pamantayan sa mga katanggap-tanggap na paggamit" ([source](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Hindi pa tiyak kung ano ang hinaharap, ngunit mahalagang maunawaan ang mga sistemang ito ng kompyuter at ang software at mga algorithm na kanilang pinapatakbo. Inaasahan namin na ang kurikulum na ito ay makakatulong sa iyo na magkaroon ng mas mahusay na pag-unawa upang ikaw mismo ang makapagdesisyon.

[![Ang kasaysayan ng deep learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Ang kasaysayan ng deep learning")
> 🎥 I-click ang larawan sa itaas para sa isang video: Tinalakay ni Yann LeCun ang kasaysayan ng deep learning sa leksyong ito

---
## 🚀Hamunin

Suriin ang isa sa mga makasaysayang sandaling ito at alamin ang higit pa tungkol sa mga taong nasa likod nito. May mga kahanga-hangang personalidad, at walang siyentipikong tuklas na nalikha sa isang cultural vacuum. Ano ang iyong natuklasan?

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/4/)

---
## Review at Pag-aaral sa Sarili

Narito ang mga bagay na maaaring panoorin at pakinggan:

[Ang podcast na ito kung saan tinalakay ni Amy Boyd ang ebolusyon ng AI](http://runasradio.com/Shows/Show/739)

[![Ang kasaysayan ng AI ni Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Ang kasaysayan ng AI ni Amy Boyd")

---

## Takdang-Aralin

[Gumawa ng timeline](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.