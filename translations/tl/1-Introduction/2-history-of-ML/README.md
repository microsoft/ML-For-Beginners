<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T18:19:18+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "tl"
}
-->
# Kasaysayan ng Machine Learning

![Buod ng Kasaysayan ng Machine Learning sa isang sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote ni [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML para sa mga nagsisimula - Kasaysayan ng Machine Learning](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML para sa mga nagsisimula - Kasaysayan ng Machine Learning")

> 🎥 I-click ang imahe sa itaas para sa isang maikling video na tumatalakay sa araling ito.

Sa araling ito, tatalakayin natin ang mga pangunahing milestone sa kasaysayan ng machine learning at artificial intelligence.

Ang kasaysayan ng artificial intelligence (AI) bilang isang larangan ay konektado sa kasaysayan ng machine learning, dahil ang mga algorithm at computational advances na bumubuo sa ML ay nag-ambag sa pag-unlad ng AI. Mahalagang tandaan na, bagama't nagsimulang mabuo ang mga larangang ito bilang magkakaibang paksa noong 1950s, may mga mahahalagang [algorithmic, statistical, mathematical, computational, at technical discoveries](https://wikipedia.org/wiki/Timeline_of_machine_learning) na nauna at sumabay sa panahong ito. Sa katunayan, matagal nang iniisip ng mga tao ang mga tanong na ito sa loob ng [daan-daang taon](https://wikipedia.org/wiki/History_of_artificial_intelligence): ang artikulong ito ay tumatalakay sa intelektwal na pundasyon ng ideya ng isang 'thinking machine.'

---
## Mga Mahahalagang Diskubre

- 1763, 1812 [Bayes Theorem](https://wikipedia.org/wiki/Bayes%27_theorem) at ang mga nauna nito. Ang theorem na ito at ang mga aplikasyon nito ay naglalarawan ng probabilidad ng isang pangyayari batay sa naunang kaalaman.
- 1805 [Least Square Theory](https://wikipedia.org/wiki/Least_squares) ni Adrien-Marie Legendre, isang French mathematician. Ang teoryang ito, na matututunan mo sa aming Regression unit, ay tumutulong sa data fitting.
- 1913 [Markov Chains](https://wikipedia.org/wiki/Markov_chain), na ipinangalan kay Andrey Markov, isang Russian mathematician, ay ginagamit upang ilarawan ang isang sunod-sunod na posibleng pangyayari batay sa nakaraang estado.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron), isang uri ng linear classifier na imbento ni Frank Rosenblatt, isang American psychologist, na naging pundasyon ng mga pag-unlad sa deep learning.

---

- 1967 [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor), isang algorithm na orihinal na dinisenyo para sa pagmamapa ng ruta. Sa konteksto ng ML, ginagamit ito upang tukuyin ang mga pattern.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation), ginagamit upang sanayin ang [feedforward neural networks](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network), mga artificial neural networks na nagmula sa feedforward neural networks na lumilikha ng temporal graphs.

✅ Mag-research ng kaunti. Anong iba pang mga petsa ang tumatak bilang mahalaga sa kasaysayan ng ML at AI?

---
## 1950: Mga makinang nag-iisip

Si Alan Turing, isang tunay na kahanga-hangang tao na itinanghal [ng publiko noong 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) bilang pinakadakilang siyentipiko ng ika-20 siglo, ay kinikilala sa pagtulong na mailatag ang pundasyon para sa konsepto ng isang 'makinang nag-iisip.' Hinarap niya ang mga kritiko at ang kanyang sariling pangangailangan para sa empirikal na ebidensya ng konseptong ito sa pamamagitan ng paglikha ng [Turing Test](https://www.bbc.com/news/technology-18475646), na iyong matututunan sa aming NLP lessons.

---
## 1956: Dartmouth Summer Research Project

"Ang Dartmouth Summer Research Project sa artificial intelligence ay isang mahalagang kaganapan para sa larangan ng artificial intelligence," at dito unang ginamit ang terminong 'artificial intelligence' ([source](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Ang bawat aspeto ng pag-aaral o anumang iba pang tampok ng katalinuhan ay maaaring sa prinsipyo ay mailarawan nang eksakto upang ang isang makina ay magawang gayahin ito.

---

Ang pangunahing mananaliksik, si John McCarthy, isang propesor ng matematika, ay umaasang "magpatuloy batay sa hinuha na ang bawat aspeto ng pag-aaral o anumang iba pang tampok ng katalinuhan ay maaaring sa prinsipyo ay mailarawan nang eksakto upang ang isang makina ay magawang gayahin ito." Ang mga kalahok ay kinabibilangan ng isa pang kilalang tao sa larangan, si Marvin Minsky.

Ang workshop ay kinikilala sa pagsisimula at paghikayat ng ilang talakayan kabilang ang "ang pag-usbong ng symbolic methods, mga sistema na nakatuon sa limitadong domain (mga unang expert systems), at deductive systems laban sa inductive systems." ([source](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Ang ginintuang taon"

Mula 1950s hanggang kalagitnaan ng '70s, mataas ang optimismo na ang AI ay maaaring makapagbigay solusyon sa maraming problema. Noong 1967, tiwala si Marvin Minsky na "Sa loob ng isang henerasyon ... ang problema ng paglikha ng 'artificial intelligence' ay malulutas nang malaki." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Ang pananaliksik sa natural language processing ay umunlad, ang paghahanap ay pinino at ginawang mas makapangyarihan, at ang konsepto ng 'micro-worlds' ay nilikha, kung saan ang mga simpleng gawain ay natatapos gamit ang mga simpleng utos sa wika.

---

Ang pananaliksik ay mahusay na pinondohan ng mga ahensya ng gobyerno, nagkaroon ng mga pag-unlad sa computation at algorithms, at ang mga prototype ng intelligent machines ay nabuo. Ilan sa mga makinang ito ay:

* [Shakey the robot](https://wikipedia.org/wiki/Shakey_the_robot), na kayang magmaneho at magdesisyon kung paano isasagawa ang mga gawain nang 'matalino'.

    ![Shakey, isang intelligent robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey noong 1972

---

* Eliza, isang maagang 'chatterbot', ay kayang makipag-usap sa mga tao at kumilos bilang isang primitive 'therapist'. Malalaman mo pa ang tungkol kay Eliza sa NLP lessons.

    ![Eliza, isang bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > Isang bersyon ni Eliza, isang chatbot

---

* "Blocks world" ay isang halimbawa ng micro-world kung saan ang mga blocks ay maaaring itumpok at ayusin, at ang mga eksperimento sa pagtuturo sa mga makina na gumawa ng desisyon ay maaaring subukan. Ang mga pag-unlad na ginawa gamit ang mga library tulad ng [SHRDLU](https://wikipedia.org/wiki/SHRDLU) ay tumulong sa pagpapalakas ng language processing.

    [![blocks world gamit ang SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world gamit ang SHRDLU")

    > 🎥 I-click ang imahe sa itaas para sa isang video: Blocks world gamit ang SHRDLU

---
## 1974 - 1980: "AI Winter"

Sa kalagitnaan ng 1970s, naging malinaw na ang pagiging kumplikado ng paggawa ng 'intelligent machines' ay hindi sapat na naipaliwanag at ang pangako nito, batay sa available na compute power, ay labis na napalaki. Ang pondo ay naubos at ang tiwala sa larangan ay bumagal. Ilan sa mga isyung nakaapekto sa tiwala ay:
---
- **Limitasyon**. Ang compute power ay masyadong limitado.
- **Combinatorial explosion**. Ang dami ng mga parameter na kailangang sanayin ay lumago nang eksponensyal habang mas maraming hinihingi sa mga computer, nang walang kasabay na pag-unlad ng compute power at kakayahan.
- **Kakulangan ng data**. May kakulangan ng data na humadlang sa proseso ng pagsubok, pagbuo, at pagpapabuti ng mga algorithm.
- **Tama ba ang mga tanong na tinatanong natin?**. Ang mismong mga tanong na tinatanong ay nagsimulang kuwestyunin. Ang mga mananaliksik ay nagsimulang tumanggap ng kritisismo tungkol sa kanilang mga pamamaraan:
  - Ang Turing tests ay kinuwestyon sa pamamagitan ng, bukod sa iba pang ideya, ng 'chinese room theory' na nagsasabing, "ang pag-program ng isang digital computer ay maaaring magmukhang nauunawaan ang wika ngunit hindi makakalikha ng tunay na pag-unawa." ([source](https://plato.stanford.edu/entries/chinese-room/))
  - Ang etika ng pagpapakilala ng artificial intelligences tulad ng "therapist" ELIZA sa lipunan ay hinamon.

---

Kasabay nito, iba't ibang mga paaralan ng pag-iisip sa AI ang nagsimulang mabuo. Isang dichotomy ang naitatag sa pagitan ng ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) na mga kasanayan. Ang mga _scruffy_ na laboratoryo ay nag-aayos ng mga programa nang paulit-ulit hanggang sa makuha ang nais na resulta. Ang mga _neat_ na laboratoryo ay "nakatuon sa lohika at pormal na paglutas ng problema". Ang ELIZA at SHRDLU ay kilalang mga _scruffy_ na sistema. Noong 1980s, habang lumitaw ang pangangailangan na gawing reproducible ang mga ML system, ang _neat_ na pamamaraan ay unti-unting naging pangunahing dahil ang mga resulta nito ay mas madaling ipaliwanag.

---
## 1980s Expert systems

Habang lumalaki ang larangan, naging mas malinaw ang benepisyo nito sa negosyo, at noong 1980s gayundin ang paglaganap ng 'expert systems'. "Ang mga expert systems ay kabilang sa mga unang tunay na matagumpay na anyo ng artificial intelligence (AI) software." ([source](https://wikipedia.org/wiki/Expert_system)).

Ang ganitong uri ng sistema ay talagang _hybrid_, na binubuo ng isang rules engine na nagtatakda ng mga kinakailangan sa negosyo, at isang inference engine na gumagamit ng rules system upang makabuo ng mga bagong katotohanan.

Ang panahong ito ay nagbigay din ng mas maraming pansin sa neural networks.

---
## 1987 - 1993: AI 'Chill'

Ang paglaganap ng mga specialized expert systems hardware ay nagkaroon ng hindi magandang epekto ng pagiging masyadong specialized. Ang pag-usbong ng personal computers ay nakipagkumpitensya rin sa mga malalaking, specialized, centralized systems. Nagsimula na ang democratization ng computing, na kalaunan ay nagbigay-daan sa modernong pagsabog ng big data.

---
## 1993 - 2011

Ang panahong ito ay nagbigay-daan sa bagong era para sa ML at AI upang malutas ang ilan sa mga problemang dulot ng kakulangan ng data at compute power noong una. Ang dami ng data ay mabilis na dumami at naging mas malawak na magagamit, para sa mabuti at masama, lalo na sa pag-usbong ng smartphone noong 2007. Ang compute power ay lumago nang eksponensyal, at ang mga algorithm ay umunlad kasabay nito. Ang larangan ay nagsimulang mag-mature habang ang mga malayang araw ng nakaraan ay nagsimulang mabuo bilang isang tunay na disiplina.

---
## Ngayon

Ngayon, ang machine learning at AI ay nakakaapekto sa halos bawat bahagi ng ating buhay. Ang panahong ito ay nangangailangan ng maingat na pag-unawa sa mga panganib at potensyal na epekto ng mga algorithm na ito sa buhay ng tao. Tulad ng sinabi ni Brad Smith ng Microsoft, "Ang teknolohiya ng impormasyon ay nagdudulot ng mga isyu na tumutukoy sa mga pangunahing proteksyon ng karapatang pantao tulad ng privacy at kalayaan sa pagpapahayag. Ang mga isyung ito ay nagpapataas ng responsibilidad para sa mga kumpanya ng teknolohiya na lumilikha ng mga produktong ito. Sa aming pananaw, nangangailangan din ito ng maingat na regulasyon ng gobyerno at ng pagbuo ng mga pamantayan sa mga katanggap-tanggap na paggamit" ([source](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Hindi pa natin alam kung ano ang hinaharap, ngunit mahalagang maunawaan ang mga sistemang ito ng computer at ang software at mga algorithm na kanilang pinapatakbo. Inaasahan namin na ang kurikulum na ito ay makakatulong sa iyo na magkaroon ng mas mahusay na pag-unawa upang ikaw mismo ang makapagdesisyon.

[![Ang kasaysayan ng deep learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Ang kasaysayan ng deep learning")
> 🎥 I-click ang imahe sa itaas para sa isang video: Tinalakay ni Yann LeCun ang kasaysayan ng deep learning sa lecture na ito

---
## 🚀Hamunin

Pag-aralan ang isa sa mga makasaysayang sandali na ito at alamin ang higit pa tungkol sa mga tao sa likod nito. May mga kahanga-hangang karakter, at walang scientific discovery na kailanman nilikha sa isang cultural vacuum. Ano ang iyong natuklasan?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

---
## Review & Self Study

Narito ang mga item na pwedeng panoorin at pakinggan:

[Ang podcast na ito kung saan tinalakay ni Amy Boyd ang ebolusyon ng AI](http://runasradio.com/Shows/Show/739)

[![Ang kasaysayan ng AI ni Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Ang kasaysayan ng AI ni Amy Boyd")

---

## Takdang Aralin

[Gumawa ng timeline](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na pinagmulan. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.