<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T16:42:48+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "sw"
}
-->
# Dunia Halisi Zaidi

Katika hali yetu, Peter aliweza kusafiri karibu bila kuchoka au kuhisi njaa. Katika dunia halisi zaidi, anapaswa kukaa chini na kupumzika mara kwa mara, na pia kujilisha. Hebu tufanye dunia yetu iwe halisi zaidi kwa kutekeleza sheria zifuatazo:

1. Kwa kusafiri kutoka sehemu moja hadi nyingine, Peter hupoteza **nguvu** na kupata **uchovu**.
2. Peter anaweza kupata nguvu zaidi kwa kula matufaha.
3. Peter anaweza kuondoa uchovu kwa kupumzika chini ya mti au kwenye nyasi (yaani, kutembea hadi eneo la ubao lenye mti au nyasi - uwanja wa kijani).
4. Peter anahitaji kutafuta na kumuua mbwa mwitu.
5. Ili kumuua mbwa mwitu, Peter anahitaji kuwa na viwango fulani vya nguvu na uchovu, vinginevyo atashindwa katika vita.

## Maelekezo

Tumia [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) ya awali kama sehemu ya kuanzia kwa suluhisho lako.

Badilisha kazi ya malipo hapo juu kulingana na sheria za mchezo, endesha algoriti ya kujifunza kwa kuimarisha ili kujifunza mkakati bora wa kushinda mchezo, na linganisha matokeo ya kutembea bila mpangilio na algoriti yako kwa kuzingatia idadi ya michezo iliyoshinda na iliyopotezwa.

> **Note**: Katika dunia yako mpya, hali ni ngumu zaidi, na pamoja na nafasi ya binadamu pia inajumuisha viwango vya uchovu na nguvu. Unaweza kuchagua kuwakilisha hali kama tuple (Board,energy,fatigue), au kufafanua darasa kwa hali (unaweza pia kutaka kulirithi kutoka `Board`), au hata kubadilisha darasa la awali la `Board` ndani ya [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Katika suluhisho lako, tafadhali hifadhi msimbo unaohusika na mkakati wa kutembea bila mpangilio, na linganisha matokeo ya algoriti yako na kutembea bila mpangilio mwishoni.

> **Note**: Unaweza kuhitaji kurekebisha hyperparameters ili kufanya kazi, hasa idadi ya epochs. Kwa sababu mafanikio ya mchezo (kupigana na mbwa mwitu) ni tukio nadra, unaweza kutarajia muda mrefu zaidi wa mafunzo.

## Rubric

| Vigezo   | Bora                                                                                                                                                                                                  | Inayotosheleza                                                                                                                                                                         | Inahitaji Kuboresha                                                                                                                        |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook imewasilishwa na ufafanuzi wa sheria mpya za dunia, algoriti ya Q-Learning na maelezo ya maandishi. Q-Learning ina uwezo wa kuboresha matokeo kwa kiasi kikubwa ikilinganishwa na kutembea bila mpangilio. | Notebook imewasilishwa, Q-Learning imefanywa na inaboresha matokeo ikilinganishwa na kutembea bila mpangilio, lakini si kwa kiasi kikubwa; au notebook haijafafanuliwa vizuri na msimbo haujapangwa vizuri | Jaribio fulani la kufafanua sheria za dunia limefanywa, lakini algoriti ya Q-Learning haifanyi kazi, au kazi ya malipo haijafafanuliwa kikamilifu |

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.