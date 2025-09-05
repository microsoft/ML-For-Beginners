<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T16:43:11+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "hu"
}
-->
# Egy Reálisabb Világ

A mi helyzetünkben Péter szinte fáradtság vagy éhség nélkül tudott mozogni. Egy reálisabb világban időnként le kell ülnie pihenni, és ennie is kell. Tegyük világunkat reálisabbá az alábbi szabályok bevezetésével:

1. Amikor Péter egyik helyről a másikra mozog, **energiát** veszít és **fáradtságot** szerez.
2. Péter több energiát nyerhet, ha almát eszik.
3. Péter megszabadulhat a fáradtságtól, ha a fa alatt vagy a füvön pihen (azaz olyan mezőre lép, ahol fa vagy fű van - zöld mező).
4. Péternek meg kell találnia és le kell győznie a farkast.
5. Ahhoz, hogy legyőzze a farkast, Péternek bizonyos energiaszinttel és fáradtsági szinttel kell rendelkeznie, különben elveszíti a csatát.

## Útmutató

Használd az eredeti [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) jegyzetfüzetet kiindulópontként a megoldáshoz.

Módosítsd a jutalomfüggvényt a játék szabályai szerint, futtasd a megerősítéses tanulási algoritmust, hogy megtaláld a legjobb stratégiát a játék megnyeréséhez, és hasonlítsd össze az eredményeket a véletlenszerű lépésekkel az alapján, hogy hány játékot nyertél vagy vesztettél.

> **Note**: Az új világban az állapot összetettebb, és az ember pozíciója mellett magában foglalja a fáradtság és az energiaszinteket is. Az állapotot ábrázolhatod egy tuple formájában (Board,energy,fatigue), definiálhatsz egy osztályt az állapothoz (akár származtathatod a `Board` osztályból), vagy módosíthatod az eredeti `Board` osztályt a [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py) fájlban.

A megoldásodban tartsd meg a véletlenszerű lépés stratégiájáért felelős kódot, és hasonlítsd össze az algoritmusod eredményeit a véletlenszerű lépésekkel a végén.

> **Note**: Lehet, hogy módosítanod kell a hiperparamétereket, hogy működjön, különösen az epochok számát. Mivel a játék sikere (a farkassal való harc) ritka esemény, sokkal hosszabb tanulási időre számíthatsz.

## Értékelési Szempontok

| Kritérium | Kiváló                                                                                                                                                                                                 | Megfelelő                                                                                                                                                                              | Fejlesztésre Szorul                                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Egy jegyzetfüzet bemutatja az új világ szabályait, a Q-Learning algoritmust és néhány szöveges magyarázatot. A Q-Learning jelentősen javítja az eredményeket a véletlenszerű lépésekhez képest.         | Jegyzetfüzet bemutatva, Q-Learning implementálva és javítja az eredményeket a véletlenszerű lépésekhez képest, de nem jelentősen; vagy a jegyzetfüzet rosszul dokumentált, a kód nem jól strukturált | Kísérlet történt a világ szabályainak újradefiniálására, de a Q-Learning algoritmus nem működik, vagy a jutalomfüggvény nincs teljesen definiálva |

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.