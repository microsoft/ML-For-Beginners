<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-08-29T14:14:33+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "tl"
}
-->
# Isang Mas Realistikong Mundo

Sa ating sitwasyon, halos hindi napapagod o nagugutom si Peter habang gumagalaw. Sa isang mas realistikong mundo, kailangan niyang umupo at magpahinga paminsan-minsan, at kailangan din niyang kumain. Gawin natin ang ating mundo na mas makatotohanan sa pamamagitan ng pagpapatupad ng mga sumusunod na patakaran:

1. Sa bawat paglipat mula sa isang lugar patungo sa iba, nawawalan si Peter ng **enerhiya** at nagkakaroon ng **pagkapagod**.
2. Makakakuha si Peter ng mas maraming enerhiya sa pamamagitan ng pagkain ng mansanas.
3. Mawawala ang pagkapagod ni Peter sa pamamagitan ng pagpapahinga sa ilalim ng puno o sa damuhan (halimbawa, paglalakad sa isang lokasyon ng board na may puno o damo - berdeng lugar).
4. Kailangan hanapin at patayin ni Peter ang lobo.
5. Upang mapatay ang lobo, kailangang may tiyak na antas ng enerhiya at pagkapagod si Peter, kung hindi ay matatalo siya sa laban.

## Mga Instruksyon

Gamitin ang orihinal na [notebook.ipynb](notebook.ipynb) bilang panimulang punto para sa iyong solusyon.

Baguhin ang reward function ayon sa mga patakaran ng laro, patakbuhin ang reinforcement learning algorithm upang matutunan ang pinakamahusay na estratehiya para manalo sa laro, at ikumpara ang mga resulta ng random walk sa iyong algorithm batay sa bilang ng mga larong napanalunan at natalo.

> **Note**: Sa iyong bagong mundo, mas kumplikado ang estado, at bukod sa posisyon ng tao, kasama rin dito ang antas ng pagkapagod at enerhiya. Maaari mong piliing i-representa ang estado bilang isang tuple (Board, energy, fatigue), o magdeklara ng isang klase para sa estado (maaari mo ring i-derive ito mula sa `Board`), o baguhin ang orihinal na `Board` class sa [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Sa iyong solusyon, panatilihin ang code na responsable para sa random walk strategy, at ikumpara ang mga resulta ng iyong algorithm sa random walk sa dulo.

> **Note**: Maaaring kailanganin mong ayusin ang mga hyperparameter upang gumana ito, lalo na ang bilang ng mga epochs. Dahil ang tagumpay sa laro (pakikipaglaban sa lobo) ay isang bihirang pangyayari, maaari mong asahan ang mas mahabang oras ng pagsasanay.

## Rubric

| Pamantayan | Natatangi                                                                                                                                                                                             | Katanggap-tanggap                                                                                                                                                                      | Kailangan ng Pagpapabuti                                                                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
|            | Ang notebook ay nagpapakita ng depinisyon ng mga bagong patakaran ng mundo, Q-Learning algorithm, at ilang tekstuwal na paliwanag. Ang Q-Learning ay kayang makabuluhang mapabuti ang resulta kumpara sa random walk. | Ang notebook ay ipinakita, naipatupad ang Q-Learning at napabuti ang resulta kumpara sa random walk, ngunit hindi gaanong makabuluhan; o ang notebook ay kulang sa dokumentasyon at hindi maayos ang pagkaka-istruktura ng code. | May ilang pagtatangka na i-redefine ang mga patakaran ng mundo, ngunit hindi gumagana ang Q-Learning algorithm, o hindi ganap na naipaliwanag ang reward function. |

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.