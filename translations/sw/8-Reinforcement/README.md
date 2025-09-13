<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T16:35:13+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "sw"
}
-->
# Utangulizi wa kujifunza kwa kuimarisha

Kujifunza kwa kuimarisha, RL, huchukuliwa kama mojawapo ya mifumo ya msingi ya kujifunza kwa mashine, sambamba na kujifunza kwa kusimamiwa na kujifunza bila kusimamiwa. RL inahusu maamuzi: kufanya maamuzi sahihi au angalau kujifunza kutoka kwa maamuzi hayo.

Fikiria una mazingira yaliyosimuliwa kama soko la hisa. Nini hutokea ikiwa utaweka kanuni fulani? Je, ina athari chanya au hasi? Ikiwa kitu hasi kinatokea, unahitaji kuchukua _kuimarisha hasi_, kujifunza kutoka kwayo, na kubadilisha mwelekeo. Ikiwa ni matokeo chanya, unahitaji kujenga juu ya _kuimarisha chanya_.

![peter na mbwa mwitu](../../../8-Reinforcement/images/peter.png)

> Peter na marafiki zake wanahitaji kutoroka mbwa mwitu mwenye njaa! Picha na [Jen Looper](https://twitter.com/jenlooper)

## Mada ya Kieneo: Peter na Mbwa Mwitu (Urusi)

[Peter na Mbwa Mwitu](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) ni hadithi ya muziki iliyoandikwa na mtunzi wa Kirusi [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Ni hadithi kuhusu kijana shupavu Peter, ambaye kwa ujasiri anatoka nyumbani kwake kwenda uwanda wa msitu kumfukuza mbwa mwitu. Katika sehemu hii, tutafundisha algoriti za kujifunza kwa mashine zitakazomsaidia Peter:

- **Kuchunguza** eneo linalomzunguka na kujenga ramani bora ya urambazaji.
- **Kujifunza** jinsi ya kutumia skateboard na kudumisha usawa wake, ili kuzunguka kwa kasi zaidi.

[![Peter na Mbwa Mwitu](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Bofya picha hapo juu kusikiliza Peter na Mbwa Mwitu na Prokofiev

## Kujifunza kwa kuimarisha

Katika sehemu zilizopita, umeona mifano miwili ya matatizo ya kujifunza kwa mashine:

- **Kusimamiwa**, ambapo tuna seti za data zinazopendekeza suluhisho za mfano kwa tatizo tunalotaka kutatua. [Uainishaji](../4-Classification/README.md) na [urekebishaji](../2-Regression/README.md) ni kazi za kujifunza kwa kusimamiwa.
- **Bila kusimamiwa**, ambapo hatuna data ya mafunzo yenye lebo. Mfano mkuu wa kujifunza bila kusimamiwa ni [Kugawanya makundi](../5-Clustering/README.md).

Katika sehemu hii, tutakutambulisha aina mpya ya tatizo la kujifunza ambalo halihitaji data ya mafunzo yenye lebo. Kuna aina kadhaa za matatizo kama haya:

- **[Kujifunza kwa nusu kusimamiwa](https://wikipedia.org/wiki/Semi-supervised_learning)**, ambapo tuna data nyingi isiyo na lebo inayoweza kutumika kufundisha awali mfano.
- **[Kujifunza kwa kuimarisha](https://wikipedia.org/wiki/Reinforcement_learning)**, ambapo wakala hujifunza jinsi ya kuendesha mambo kwa kufanya majaribio katika mazingira yaliyosimuliwa.

### Mfano - mchezo wa kompyuta

Fikiria unataka kufundisha kompyuta kucheza mchezo, kama vile chess, au [Super Mario](https://wikipedia.org/wiki/Super_Mario). Ili kompyuta icheze mchezo, tunahitaji kuitabiria hatua gani ichukue katika kila hali ya mchezo. Ingawa hili linaweza kuonekana kama tatizo la uainishaji, si hivyo - kwa sababu hatuna seti ya data yenye hali na hatua zinazolingana. Ingawa tunaweza kuwa na data kama vile mechi zilizopo za chess au rekodi za wachezaji wakicheza Super Mario, kuna uwezekano kwamba data hiyo haitatosheleza kufunika idadi kubwa ya hali zinazowezekana.

Badala ya kutafuta data ya mchezo iliyopo, **Kujifunza kwa Kuimarisha** (RL) kunategemea wazo la *kuifanya kompyuta icheze* mara nyingi na kuchunguza matokeo. Hivyo basi, ili kutumia Kujifunza kwa Kuimarisha, tunahitaji vitu viwili:

- **Mazingira** na **kisimulizi** kinachoturuhusu kucheza mchezo mara nyingi. Kisimulizi hiki kingeeleza sheria zote za mchezo pamoja na hali na hatua zinazowezekana.

- **Kazi ya malipo**, ambayo ingetueleza jinsi tulivyofanya vizuri katika kila hatua au mchezo.

Tofauti kuu kati ya aina nyingine za kujifunza kwa mashine na RL ni kwamba katika RL kwa kawaida hatujui kama tumeshinda au tumeshindwa hadi tumalize mchezo. Hivyo basi, hatuwezi kusema kama hatua fulani pekee ni nzuri au la - tunapokea tu malipo mwishoni mwa mchezo. Na lengo letu ni kubuni algoriti zitakazotuwezesha kufundisha mfano chini ya hali zisizo na uhakika. Tutajifunza kuhusu algoriti moja ya RL inayoitwa **Q-learning**.

## Masomo

1. [Utangulizi wa kujifunza kwa kuimarisha na Q-Learning](1-QLearning/README.md)
2. [Kutumia mazingira ya simulizi ya gym](2-Gym/README.md)

## Shukrani

"Utangulizi wa Kujifunza kwa Kuimarisha" uliandikwa kwa ♥️ na [Dmitry Soshnikov](http://soshnikov.com)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.