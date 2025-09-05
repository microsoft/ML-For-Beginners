<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T16:48:06+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "hu"
}
-->
# Hegyi Autó Tanítása

[OpenAI Gym](http://gym.openai.com) úgy lett kialakítva, hogy minden környezet ugyanazt az API-t biztosítsa - azaz ugyanazokat a `reset`, `step` és `render` metódusokat, valamint az **akciótér** és **megfigyelési tér** azonos absztrakcióit. Ezért lehetségesnek kell lennie, hogy ugyanazokat a megerősítéses tanulási algoritmusokat minimális kódmódosítással különböző környezetekhez igazítsuk.

## A Hegyi Autó Környezet

A [Hegyi Autó környezet](https://gym.openai.com/envs/MountainCar-v0/) egy völgyben ragadt autót tartalmaz:

A cél az, hogy kijussunk a völgyből és megszerezzük a zászlót, az alábbi akciók egyikének végrehajtásával minden lépésben:

| Érték | Jelentés |
|---|---|
| 0 | Balra gyorsítás |
| 1 | Nem gyorsít |
| 2 | Jobbra gyorsítás |

A probléma fő trükkje azonban az, hogy az autó motorja nem elég erős ahhoz, hogy egyetlen menetben felmásszon a hegyre. Ezért az egyetlen módja a sikernek az, hogy oda-vissza vezetünk, hogy lendületet gyűjtsünk.

A megfigyelési tér mindössze két értékből áll:

| Szám | Megfigyelés  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Autó pozíció | -1.2| 0.6 |
|  1  | Autó sebesség | -0.07 | 0.07 |

A jutalmazási rendszer a hegyi autó esetében meglehetősen trükkös:

 * 0 jutalom jár, ha az ügynök elérte a zászlót (pozíció = 0.5) a hegy tetején.
 * -1 jutalom jár, ha az ügynök pozíciója kevesebb, mint 0.5.

Az epizód véget ér, ha az autó pozíciója több mint 0.5, vagy ha az epizód hossza meghaladja a 200-at.

## Útmutató

Igazítsd a megerősítéses tanulási algoritmusunkat a hegyi autó probléma megoldásához. Kezdd a meglévő [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) kóddal, cseréld ki az új környezetet, módosítsd az állapot diszkretizációs függvényeket, és próbáld meg a meglévő algoritmust minimális kódmódosítással betanítani. Optimalizáld az eredményt a hiperparaméterek beállításával.

> **Megjegyzés**: Valószínűleg szükség lesz a hiperparaméterek beállítására, hogy az algoritmus konvergáljon.

## Értékelési Kritériumok

| Kritérium | Kiváló | Megfelelő | Fejlesztésre szorul |
| --------- | ------- | --------- | ------------------- |
|          | A Q-Learning algoritmus sikeresen adaptálva lett a CartPole példából, minimális kódmódosítással, és képes megoldani a zászló megszerzésének problémáját 200 lépés alatt. | Egy új Q-Learning algoritmus lett átvéve az internetről, de jól dokumentált; vagy a meglévő algoritmus lett adaptálva, de nem érte el a kívánt eredményeket. | A hallgató nem tudott sikeresen adaptálni semmilyen algoritmust, de jelentős lépéseket tett a megoldás felé (megvalósította az állapot diszkretizációt, Q-Table adatstruktúrát stb.). |

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.