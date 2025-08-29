<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-08-29T14:08:13+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "tl"
}
-->
# Panimula sa reinforcement learning

Ang reinforcement learning, RL, ay itinuturing bilang isa sa mga pangunahing paradigma ng machine learning, kasabay ng supervised learning at unsupervised learning. Ang RL ay tungkol sa paggawa ng mga desisyon: paghahatid ng tamang desisyon o kahit papaano ay pagkatuto mula rito.

Isipin mo na mayroon kang isang simulated na kapaligiran tulad ng stock market. Ano ang mangyayari kung magpataw ka ng isang partikular na regulasyon? Magkakaroon ba ito ng positibo o negatibong epekto? Kung may negatibong mangyari, kailangan mong tanggapin ang _negative reinforcement_, matuto mula rito, at baguhin ang direksyon. Kung positibo ang resulta, kailangan mong palakasin ang _positive reinforcement_.

![peter and the wolf](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.tl.png)

> Si Peter at ang kanyang mga kaibigan ay kailangang tumakas mula sa gutom na lobo! Larawan ni [Jen Looper](https://twitter.com/jenlooper)

## Pang-rehiyonal na paksa: Peter and the Wolf (Russia)

[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) ay isang musikal na kwentong pambata na isinulat ng Russian composer na si [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Ito ay kwento tungkol sa batang si Peter, na matapang na lumabas ng kanyang bahay papunta sa clearing ng kagubatan upang habulin ang lobo. Sa seksyong ito, magtuturo tayo ng mga algorithm ng machine learning na makakatulong kay Peter:

- **Mag-explore** sa paligid at bumuo ng optimal na navigation map
- **Matuto** kung paano gumamit ng skateboard at magbalanse dito, upang makagalaw nang mas mabilis.

[![Peter and the Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 I-click ang larawan sa itaas upang pakinggan ang Peter and the Wolf ni Prokofiev

## Reinforcement learning

Sa mga nakaraang seksyon, nakita mo ang dalawang halimbawa ng mga problema sa machine learning:

- **Supervised**, kung saan mayroon tayong mga dataset na nagmumungkahi ng mga sample na solusyon sa problemang nais nating lutasin. Ang [Classification](../4-Classification/README.md) at [Regression](../2-Regression/README.md) ay mga supervised learning tasks.
- **Unsupervised**, kung saan wala tayong labeled na training data. Ang pangunahing halimbawa ng unsupervised learning ay [Clustering](../5-Clustering/README.md).

Sa seksyong ito, ipakikilala namin sa iyo ang isang bagong uri ng problema sa pag-aaral na hindi nangangailangan ng labeled na training data. Mayroong ilang uri ng ganitong mga problema:

- **[Semi-supervised learning](https://wikipedia.org/wiki/Semi-supervised_learning)**, kung saan mayroon tayong maraming unlabeled na data na maaaring gamitin upang i-pre-train ang modelo.
- **[Reinforcement learning](https://wikipedia.org/wiki/Reinforcement_learning)**, kung saan natututo ang isang agent kung paano kumilos sa pamamagitan ng pagsasagawa ng mga eksperimento sa isang simulated na kapaligiran.

### Halimbawa - laro sa computer

Halimbawa, nais mong turuan ang isang computer na maglaro ng isang laro, tulad ng chess, o [Super Mario](https://wikipedia.org/wiki/Super_Mario). Para maglaro ang computer, kailangan nitong hulaan kung anong galaw ang gagawin sa bawat estado ng laro. Bagama't maaaring mukhang isang problema sa classification ito, hindi ito ganoon - dahil wala tayong dataset na may mga estado at kaukulang aksyon. Bagama't maaaring mayroon tayong data tulad ng mga umiiral na chess matches o recording ng mga manlalaro ng Super Mario, malamang na hindi sapat ang data na iyon upang masakop ang malaking bilang ng mga posibleng estado.

Sa halip na maghanap ng umiiral na data ng laro, ang **Reinforcement Learning** (RL) ay batay sa ideya ng *pagpapalaro sa computer* nang maraming beses at pagmamasid sa resulta. Kaya, upang magamit ang Reinforcement Learning, kailangan natin ng dalawang bagay:

- **Isang kapaligiran** at **isang simulator** na magpapahintulot sa atin na maglaro nang maraming beses. Ang simulator na ito ang magtatakda ng lahat ng mga patakaran ng laro pati na rin ang mga posibleng estado at aksyon.

- **Isang reward function**, na magsasabi sa atin kung gaano kahusay ang ginawa natin sa bawat galaw o laro.

Ang pangunahing pagkakaiba ng RL sa iba pang uri ng machine learning ay sa RL, kadalasan hindi natin alam kung panalo o talo tayo hanggang matapos ang laro. Kaya, hindi natin masasabing mabuti o masama ang isang partikular na galaw lamang - makakatanggap lang tayo ng reward sa dulo ng laro. At ang layunin natin ay magdisenyo ng mga algorithm na magpapahintulot sa atin na mag-train ng modelo sa ilalim ng hindi tiyak na mga kondisyon. Matututo tayo tungkol sa isang RL algorithm na tinatawag na **Q-learning**.

## Mga Aralin

1. [Panimula sa reinforcement learning at Q-Learning](1-QLearning/README.md)
2. [Paggamit ng gym simulation environment](2-Gym/README.md)

## Mga Kredito

"Introduction to Reinforcement Learning" ay isinulat nang may ♥️ ni [Dmitry Soshnikov](http://soshnikov.com)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.