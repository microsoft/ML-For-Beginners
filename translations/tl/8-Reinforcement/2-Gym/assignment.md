<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-08-29T14:18:19+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "tl"
}
-->
# Sanayin ang Mountain Car

Ang [OpenAI Gym](http://gym.openai.com) ay dinisenyo sa paraang lahat ng mga environment ay may parehong API - ibig sabihin, pareho ang mga method na `reset`, `step`, at `render`, at pareho rin ang mga abstraction ng **action space** at **observation space**. Dahil dito, posible na i-adapt ang parehong reinforcement learning algorithms sa iba't ibang environment na may kaunting pagbabago sa code.

## Ang Mountain Car Environment

Ang [Mountain Car environment](https://gym.openai.com/envs/MountainCar-v0/) ay may kasamang kotse na naipit sa isang lambak:

Ang layunin ay makalabas sa lambak at makuha ang bandila, sa pamamagitan ng paggawa ng isa sa mga sumusunod na aksyon sa bawat hakbang:

| Halaga | Kahulugan |
|---|---|
| 0 | Magpabilis pakaliwa |
| 1 | Huwag magpabilis |
| 2 | Magpabilis pakanan |

Ang pangunahing hamon sa problemang ito ay ang makina ng kotse ay hindi sapat na malakas upang makaakyat sa bundok sa isang pasada lamang. Kaya, ang tanging paraan upang magtagumpay ay ang magmaneho pabalik-balik upang makabuo ng momentum.

Ang observation space ay binubuo lamang ng dalawang halaga:

| Bilang | Obserbasyon  | Min | Max |
|-------|--------------|-----|-----|
|  0    | Posisyon ng Kotse | -1.2 | 0.6 |
|  1    | Bilis ng Kotse     | -0.07 | 0.07 |

Ang reward system para sa mountain car ay medyo mahirap:

 * Ang reward na 0 ay ibinibigay kung ang agent ay nakarating sa bandila (posisyon = 0.5) sa tuktok ng bundok.
 * Ang reward na -1 ay ibinibigay kung ang posisyon ng agent ay mas mababa sa 0.5.

Ang episode ay nagtatapos kung ang posisyon ng kotse ay higit sa 0.5, o kung ang haba ng episode ay lumampas sa 200.

## Mga Instruksyon

I-adapt ang ating reinforcement learning algorithm upang malutas ang problema ng mountain car. Magsimula sa umiiral na [notebook.ipynb](notebook.ipynb) na code, palitan ang environment, baguhin ang mga state discretization function, at subukang sanayin ang umiiral na algorithm na may kaunting pagbabago sa code. I-optimize ang resulta sa pamamagitan ng pag-aayos ng mga hyperparameter.

> **Tandaan**: Malamang na kailanganin ang pag-aayos ng mga hyperparameter upang magtagumpay ang algorithm.

## Rubric

| Pamantayan | Napakahusay | Katamtaman | Kailangan ng Pagpapabuti |
| ---------- | ----------- | ---------- | ------------------------ |
|            | Ang Q-Learning algorithm ay matagumpay na na-adapt mula sa CartPole na halimbawa, na may kaunting pagbabago sa code, at nagawang lutasin ang problema ng pagkuha ng bandila sa ilalim ng 200 hakbang. | Ang bagong Q-Learning algorithm ay inangkop mula sa Internet, ngunit mahusay na naidokumento; o ang umiiral na algorithm ay inangkop, ngunit hindi nakamit ang nais na resulta. | Hindi nagawang matagumpay na i-adapt ang anumang algorithm, ngunit may malalaking hakbang na ginawa patungo sa solusyon (hal. ipinatupad ang state discretization, Q-Table data structure, atbp.) |

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.