# Funza Gari la Mlima

[OpenAI Gym](http://gym.openai.com) imeundwa kwa namna ambayo mazingira yote yanatoa API sawa - yaani njia sawa za `reset`, `step` na `render`, na dhana sawa za **nafasi ya hatua** na **nafasi ya uchunguzi**. Kwa hivyo inapaswa kuwa rahisi kubadilisha algoriti za kujifunza kwa kuimarisha kwa mazingira tofauti kwa mabadiliko madogo ya msimbo.

## Mazingira ya Gari la Mlima

[Mazingira ya Gari la Mlima](https://gym.openai.com/envs/MountainCar-v0/) lina gari lililokwama kwenye bonde:
Lengo ni kutoka kwenye bonde na kushika bendera, kwa kufanya moja ya hatua zifuatazo katika kila hatua:

| Thamani | Maana |
|---|---|
| 0 | Kuongeza kasi kwenda kushoto |
| 1 | Kutokuongeza kasi |
| 2 | Kuongeza kasi kwenda kulia |

Ujanja mkuu wa tatizo hili ni kwamba injini ya gari haina nguvu ya kutosha kupanda mlima kwa mzunguko mmoja. Kwa hivyo, njia pekee ya kufanikiwa ni kuendesha mbele na nyuma ili kujenga mwendo.

Nafasi ya uchunguzi ina thamani mbili tu:

| Nambari | Uchunguzi  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Nafasi ya Gari | -1.2| 0.6 |
|  1  | Kasi ya Gari | -0.07 | 0.07 |

Mfumo wa zawadi kwa gari la mlima ni mgumu kidogo:

 * Zawadi ya 0 inatolewa ikiwa wakala atafikia bendera (nafasi = 0.5) juu ya mlima.
 * Zawadi ya -1 inatolewa ikiwa nafasi ya wakala ni chini ya 0.5.

Kipindi kinamalizika ikiwa nafasi ya gari ni zaidi ya 0.5, au urefu wa kipindi ni zaidi ya 200.
## Maelekezo

Badilisha algoriti yetu ya kujifunza kwa kuimarisha ili kutatua tatizo la gari la mlima. Anza na msimbo uliopo katika [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), badilisha mazingira mapya, badilisha kazi za kugawanya hali, na jaribu kufanya algoriti iliyopo kufunza kwa mabadiliko madogo ya msimbo. Boresha matokeo kwa kurekebisha vigezo vya hyper.

> **Note**: Marekebisho ya vigezo vya hyper yanaweza kuhitajika ili kufanya algoriti kufikia lengo.
## Rubric

| Kigezo | Bora | Kutosha | Inahitaji Kuboresha |
| -------- | --------- | -------- | ----------------- |
|          | Algoriti ya Q-Learning imebadilishwa kwa mafanikio kutoka mfano wa CartPole, kwa mabadiliko madogo ya msimbo, ambayo ina uwezo wa kutatua tatizo la kushika bendera chini ya hatua 200. | Algoriti mpya ya Q-Learning imechukuliwa kutoka mtandaoni, lakini imeandikwa vizuri; au algoriti iliyopo imebadilishwa, lakini haifiki matokeo yanayotarajiwa | Mwanafunzi hakuweza kubadilisha algoriti yoyote kwa mafanikio, lakini amechukua hatua kubwa kuelekea suluhisho (ameunda kazi za kugawanya hali, muundo wa data wa Q-Table, n.k.) |

**Onyo**: 
Hati hii imetafsiriwa kwa kutumia huduma za tafsiri za AI zinazotumia mashine. Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kwamba tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwepo kwa usahihi. Hati ya asili katika lugha yake ya asili inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa habari muhimu, tafsiri ya kibinadamu ya kitaalamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri potofu zinazotokana na matumizi ya tafsiri hii.