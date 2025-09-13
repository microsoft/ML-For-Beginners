<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T16:47:52+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "sw"
}
-->
# Kufundisha Gari la Mlima

[OpenAI Gym](http://gym.openai.com) imeundwa kwa namna ambayo mazingira yote yanatoa API sawa - yaani, mbinu sawa `reset`, `step` na `render`, na dhana sawa za **action space** na **observation space**. Hivyo basi, inapaswa kuwa rahisi kubadilisha algoriti za kujifunza kwa kuimarisha ili zifanye kazi katika mazingira tofauti kwa mabadiliko madogo ya msimbo.

## Mazingira ya Gari la Mlima

[Mazingira ya Gari la Mlima](https://gym.openai.com/envs/MountainCar-v0/) lina gari lililokwama kwenye bonde:

Lengo ni kutoka kwenye bonde na kufikia bendera, kwa kufanya mojawapo ya vitendo vifuatavyo katika kila hatua:

| Thamani | Maana |
|---|---|
| 0 | Kuongeza kasi kwenda kushoto |
| 1 | Kutokuongeza kasi |
| 2 | Kuongeza kasi kwenda kulia |

Hata hivyo, changamoto kuu ya tatizo hili ni kwamba injini ya gari haina nguvu ya kutosha kupanda mlima kwa mara moja. Kwa hivyo, njia pekee ya kufanikiwa ni kuendesha gari mbele na nyuma ili kujenga kasi.

Eneo la uchunguzi lina thamani mbili tu:

| Namba | Uchunguzi  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Nafasi ya Gari | -1.2| 0.6 |
|  1  | Kasi ya Gari | -0.07 | 0.07 |

Mfumo wa zawadi kwa gari la mlima ni mgumu kidogo:

 * Zawadi ya 0 inatolewa ikiwa wakala amefikia bendera (nafasi = 0.5) juu ya mlima.
 * Zawadi ya -1 inatolewa ikiwa nafasi ya wakala ni chini ya 0.5.

Kipindi kinamalizika ikiwa nafasi ya gari ni zaidi ya 0.5, au urefu wa kipindi ni zaidi ya 200.

## Maelekezo

Badilisha algoriti yetu ya kujifunza kwa kuimarisha ili kutatua tatizo la gari la mlima. Anza na msimbo uliopo katika [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), badilisha mazingira mapya, badilisha kazi za kugawa hali, na jaribu kufanya algoriti iliyopo ifanye mafunzo kwa mabadiliko madogo ya msimbo. Boresha matokeo kwa kurekebisha hyperparameters.

> **Note**: Marekebisho ya hyperparameters yanahitajika ili kufanya algoriti kufikia matokeo.

## Rubric

| Vigezo | Bora Zaidi | Inafaa | Inahitaji Kuboresha |
| -------- | --------- | -------- | ----------------- |
|          | Algoriti ya Q-Learning imebadilishwa kwa mafanikio kutoka mfano wa CartPole, kwa mabadiliko madogo ya msimbo, na inaweza kutatua tatizo la kufikia bendera chini ya hatua 200. | Algoriti mpya ya Q-Learning imechukuliwa kutoka mtandao, lakini imeelezwa vizuri; au algoriti iliyopo imebadilishwa, lakini haifiki matokeo yanayotarajiwa | Mwanafunzi hakuweza kubadilisha algoriti yoyote kwa mafanikio, lakini amefanya hatua kubwa kuelekea suluhisho (ameunda kugawa hali, muundo wa data wa Q-Table, nk.) |

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.